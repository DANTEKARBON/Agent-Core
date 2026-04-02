import logging
from typing import Optional
from core.contracts import LLMResponse
from core.cache import TTLCache
from adapters.factory import LLMAdapterFactory
from model_manager import ModelManager
from core.resilience import with_timeout, with_retry

logger = logging.getLogger(__name__)

class GenerationService:
    def __init__(self, primary_adapter, fallback_adapter, prompts, config, model_manager: ModelManager, cache: TTLCache):
        self.messages = []
        self.primary_adapter = primary_adapter
        self.fallback_adapter = fallback_adapter
        self.prompts = prompts
        self.config = config
        self.model_manager = model_manager
        self.cache = cache

        resilience_cfg = config.get("resilience", {})
        self.timeout_seconds = resilience_cfg.get("timeout_seconds", 30)
        self.retry_attempts = resilience_cfg.get("retry_attempts", 2)
        self.retry_delay = resilience_cfg.get("retry_delay_seconds", 1.0)

    def _ensure_model_loaded(self, model_name: str) -> bool:
        model_info = self.config["models"].get(model_name)
        if not model_info:
            logger.error(f"Model {model_name} not found in config")
            return False
        return self.model_manager.load_model(model_name, model_info["port"], model_info["path"])

    def generate(self, target_role: str, prompt_text: str) -> str:
        cache_key_params = {
            "max_tokens": self.config.get("llm_backends", {}).get("primary", {}).get("max_tokens", 500),
            "temperature": self.config.get("llm_backends", {}).get("primary", {}).get("temperature", 0.7)
        }
        cached = self.cache.get(prompt_text, target_role, cache_key_params)
        if cached is not None:
            logger.info(f"Generation cache hit for {target_role}")
            return cached

        adapter = self.primary_adapter
        model_name = self.config.get("llm_backends", {}).get("primary", {}).get("model_name")
        if target_role != "phi3":
            model_info = self.config["models"].get(target_role)
            if model_info:
                self._ensure_model_loaded(target_role)
                adapter = LLMAdapterFactory.create({
                    "type": "mlx_openai",
                    "base_url": f"http://127.0.0.1:{model_info['port']}",
                    "model_name": target_role,
                })
            else:
                logger.warning(f"No model config for {target_role}, using primary")
        else:
            self._ensure_model_loaded(model_name)

        prompt_template = self.prompts.get(target_role, {}).get("template", "")
        if not prompt_template:
            prompt_template = "{query}"
        final_prompt = prompt_template.format(query=prompt_text)

        @with_retry(max_attempts=self.retry_attempts, delay=self.retry_delay)
        @with_timeout(self.timeout_seconds)
        def call_with_retry():
            return adapter.generate_contract(final_prompt)

        try:
            result = call_with_retry()
            used_model_name = self.config.get("llm_backends", {}).get("primary", {}).get("model_name") if adapter == self.primary_adapter else target_role
            self.model_manager.update_last_used(used_model_name)
            logger.info(f"Generation metrics for {target_role}: tokens_in={result.tokens_input}, tokens_out={result.tokens_output}, latency={result.latency_ms:.2f}ms")
            if result.status == "error":
                raise Exception(result.error)
            self.cache.set(final_prompt, target_role, result.text, cache_key_params)
            return result.text
        except Exception as e:
            logger.error(f"Primary failed after retries: {e}")
            fallback_chain = self.config.get("fallback_chain", [])
            if fallback_chain:
                for fallback_model in fallback_chain:
                    try:
                        model_info = self.config["models"].get(fallback_model)
                        if not model_info:
                            continue
                        self.model_manager.load_model(fallback_model, model_info["port"], model_info["path"])
                        fallback_adapter = LLMAdapterFactory.create({
                            "type": "mlx_openai",
                            "base_url": f"http://127.0.0.1:{model_info['port']}",
                            "model_name": fallback_model,
                            "max_tokens": cache_key_params["max_tokens"],
                            "temperature": cache_key_params["temperature"]
                        })
                        @with_retry(max_attempts=self.retry_attempts, delay=self.retry_delay)
                        @with_timeout(self.timeout_seconds)
                        def call_fallback():
                            return fallback_adapter.generate_contract(final_prompt)

                        fallback_result = call_fallback()
                        if fallback_result.status == "ok":
                            logger.info(f"Fallback succeeded with {fallback_model}")
                            self.model_manager.update_last_used(fallback_model)
                            return fallback_result.text
                    except Exception as fb_e:
                        logger.error(f"Fallback {fallback_model} failed: {fb_e}")
                return f"[Error: {e}, all fallbacks failed]"
            else:
                if self.fallback_adapter:
                    try:
                        @with_retry(max_attempts=self.retry_attempts, delay=self.retry_delay)
                        @with_timeout(self.timeout_seconds)
                        def call_legacy():
                            return self.fallback_adapter.generate_contract(final_prompt)
                        return call_legacy().text
                    except:
                        return f"[Error: {e}, legacy fallback failed]"
                return f"[Error: {e}]"

    def reset_context(self):
        self.messages = []
        logger.debug("Generation service context reset")

    def clear_cache(self):
        if self.cache:
            self.cache.clear()
            logger.info("Generation cache cleared")