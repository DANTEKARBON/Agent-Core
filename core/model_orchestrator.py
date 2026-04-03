import yaml
import time
from typing import Tuple, Dict, Any, Optional
from logger_config import logger
from core.model_registry import ModelRegistry
from model_manager import ModelManager
from adapters.factory import LLMAdapterFactory
from adapters.base_llm_adapter import BaseLLMAdapter
from core.cache import TTLCache
from core.resilience import with_timeout, CircuitBreaker

class ModelOrchestrator:
    def __init__(self, model_manager: ModelManager, model_registry: ModelRegistry):
        self.model_manager = model_manager
        self.registry = model_registry
        self._adapters: Dict[str, BaseLLMAdapter] = {}
        self._config = self._load_config()
        self._router_prompt_template = self._get_router_prompt()
        self._role_configs = self._config.get("roles", {})
        self._role_model_mapping = self._config.get("role_model_mapping", {})
        self._fallback_chain = self._config.get("fallback_chain", ["assistant", "coder"])
        self._cache = TTLCache(ttl_seconds=3600, enabled=True)
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        logger.info("ModelOrchestrator инициализирован", extra={
            "roles": list(self._role_configs.keys()),
            "mapping": self._role_model_mapping,
            "fallback": self._fallback_chain
        })

    def _load_config(self) -> dict:
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _get_router_prompt(self) -> str:
        try:
            return self._config["prompts"]["router"]["template"]
        except KeyError:
            logger.error("Промпт для роутера не найден в config.yaml")
            return "Ты классификатор. Отвечай только coder, reasoner или assistant. Запрос: {query}\nОтвет:"

    def _get_role_config(self, role: str) -> dict:
        return self._role_configs.get(role, {})

    def _get_model_name_for_role(self, role: str) -> Optional[str]:
        return self._role_model_mapping.get(role)

    def _get_circuit_breaker(self, model_name: str) -> CircuitBreaker:
        if model_name not in self._circuit_breakers:
            resilience_cfg = self._config.get("resilience", {})
            threshold = resilience_cfg.get("circuit_breaker_failure_threshold", 3)
            recovery = resilience_cfg.get("circuit_breaker_recovery_timeout", 60)
            self._circuit_breakers[model_name] = CircuitBreaker(failure_threshold=threshold, recovery_timeout=recovery)
        return self._circuit_breakers[model_name]

    def _classify(self, query: str) -> Tuple[str, str]:
        try:
            phi3_model_name = self._get_model_name_for_role("assistant") or "phi3"
            adapter = self._get_adapter(phi3_model_name)
            prompt = self._router_prompt_template.format(query=query)
            response = adapter.generate_contract(prompt, max_tokens=10, temperature=0.0)
            raw_answer = response.text.strip().lower() if response.text else ""
            
            if raw_answer:
                for marker in ["<|end|>", "<|user|>", "<|assistant|>", "<|system|>"]:
                    if marker in raw_answer:
                        raw_answer = raw_answer.split(marker)[0].strip()
            
            if not raw_answer:
                raw_answer = "empty_response"
                logger.warning("Классификатор вернул пустой ответ, используем empty_response")
        except Exception as e:
            logger.error(f"Ошибка в классификаторе: {e}", exc_info=True)
            return "assistant", f"error:{str(e)}"

        logger.info("Сырой ответ классификатора", extra={"raw": raw_answer})
        first_word = raw_answer.split()[0] if raw_answer else ""
        logger.info("Первое слово", extra={"word": first_word})

        valid_roles = {"coder", "reasoner", "assistant"}
        if first_word in valid_roles:
            target = first_word
        else:
            logger.warning(f"Неверная классификация: {first_word}, используем assistant")
            target = "assistant"

        if not target:
            target = "assistant"
        if not raw_answer:
            raw_answer = "unknown"
        return target, raw_answer

    def _get_adapter(self, model_name: str) -> BaseLLMAdapter:
        if model_name not in self._adapters:
            info = self.registry.get_model_info(model_name)
            adapter_config = {
                "type": "mlx_openai",
                "base_url": f"http://127.0.0.1:{info['port']}",
                "model_name": model_name,
                "max_tokens": 800,
                "temperature": 0.7
            }
            adapter = LLMAdapterFactory.create(adapter_config, model_manager=self.model_manager)
            self._adapters[model_name] = adapter
            logger.info(f"Создан адаптер для {model_name}", extra={"port": info['port']})
        return self._adapters[model_name]

    @with_timeout(60)
    def _generate_with_timeout(self, adapter, prompt: str, max_tokens: int, temperature: float):
        return adapter.generate_contract(prompt, max_tokens=max_tokens, temperature=temperature)

    def process(self, query: str) -> Tuple[str, str, Dict[str, Any], str]:
        target_role, raw_classifier = self._classify(query)
        if not target_role:
            target_role = "assistant"
        if not raw_classifier:
            raw_classifier = "empty"

        logger.info("classified", extra={"target_role": target_role, "raw_classifier": raw_classifier})

        roles_to_try = [target_role] + [r for r in self._fallback_chain if r != target_role]
        last_error = None

        for role in roles_to_try:
            model_name = self._get_model_name_for_role(role)
            if not model_name:
                logger.warning(f"Нет модели для роли {role}, пропускаем")
                continue

            role_cfg = self._get_role_config(role)
            template = role_cfg.get("prompt_template", "Пользователь: {query}\nОтвет:")
            final_prompt = template.format(query=query)
            max_tokens = role_cfg.get("max_tokens", 800)
            temperature = role_cfg.get("temperature", 0.7)

            adapter = self._get_adapter(model_name)
            cb = self._get_circuit_breaker(model_name)

            try:
                start_gen = time.time()
                def call_adapter():
                    return self._generate_with_timeout(adapter, final_prompt, max_tokens, temperature)
                response = cb.call(call_adapter)
                total_time_ms = (time.time() - start_gen) * 1000

                if response.status == "error":
                    raise Exception(response.error or "Unknown adapter error")

                metrics = {
                    "in": response.prompt_tokens,
                    "out": response.completion_tokens,
                    "ttft_ms": response.ttft_ms if response.ttft_ms else 0.0,
                    "total_ms": total_time_ms,
                }
                if metrics["out"] > 0 and total_time_ms > 0:
                    metrics["t/s"] = metrics["out"] / (total_time_ms / 1000.0)
                else:
                    metrics["t/s"] = 0.0

                logger.info(f"Генерация {role} через модель {model_name}", extra=metrics)
                return response.text, role, metrics, raw_classifier

            except Exception as e:
                logger.error(f"Ошибка при генерации через роль {role} (модель {model_name}): {type(e).__name__}: {e}")
                last_error = e

        error_msg = f"Не удалось обработать запрос. Последняя ошибка: {last_error}"
        logger.error(error_msg)
        return f"[Ошибка] {error_msg}", "error", {}, raw_classifier

    def clear_cache(self):
        self._adapters.clear()
        self._cache.clear()
        logger.info("Кэш адаптеров и TTLCache очищены")

    def get_cache_stats(self) -> dict:
        return {"cached_adapters": len(self._adapters), "cache": self._cache.stats()}

    def save_cache(self, filepath: str):
        self._cache.save(filepath)

    def load_cache(self, filepath: str):
        self._cache.load(filepath)

    def shutdown(self):
        logger.info("Shutting down ModelOrchestrator")
        for adapter in self._adapters.values():
            if hasattr(adapter, 'close'):
                adapter.close()
        self.model_manager.stop()
