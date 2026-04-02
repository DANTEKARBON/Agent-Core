import logging
from typing import Optional
from core.contracts import LLMResponse
from core.cache import TTLCache
from adapters.factory import LLMAdapterFactory
from model_manager import ModelManager
from core.resilience import with_timeout, with_retry

logger = logging.getLogger(__name__)

class RouterService:
    def __init__(self, primary_adapter, prompts, config, model_manager: ModelManager, cache: TTLCache):
        self.primary_adapter = primary_adapter
        self.prompts = prompts
        self.config = config
        self.model_manager = model_manager
        self.cache = cache

        # Resilience параметры из config.yaml
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

    def classify(self, query: str) -> str:
        if not self.primary_adapter:
            logger.error("Primary adapter not available")
            return "phi3"

        router_model_name = self.config.get("llm_backends", {}).get("primary", {}).get("model_name")
        if router_model_name:
            self._ensure_model_loaded(router_model_name)
        else:
            router_model_name = "phi3"
            self._ensure_model_loaded(router_model_name)

        router_prompt_template = self.prompts.get("router", {}).get("template", "")
        if not router_prompt_template:
            logger.error("Router prompt template not found")
            return "phi3"

        prompt = router_prompt_template.format(query=query)

        # Cache lookup
        cache_key_params = {"max_tokens": 5, "temperature": 0.0}
        cached = self.cache.get(prompt, "router", cache_key_params)
        if cached is not None:
            logger.info(f"Classification cache hit, returning {cached}")
            return cached

        @with_retry(max_attempts=self.retry_attempts, delay=self.retry_delay)
        @with_timeout(self.timeout_seconds)
        def call_with_retry():
            return self.primary_adapter.generate_contract(prompt, max_tokens=5, temperature=0.0)

        try:
            result: LLMResponse = call_with_retry()
            if result.status == "error":
                logger.error(f"Classification error: {result.error}")
                return "phi3"

            raw_response = result.text.strip().lower()
            for marker in ["<|end|>", "<|user|>", "<|assistant|>", "<|system|>"]:
                if marker in raw_response:
                    raw_response = raw_response.split(marker)[0].strip()

            logger.info(f"Raw classification response: {raw_response}")
            first_word = raw_response.split()[0] if raw_response else ""
            logger.info(f"First word: {first_word}")

            classification = "phi3"
            if first_word in ("coder", "reasoner"):
                classification = first_word

            self.cache.set(prompt, "router", classification, cache_key_params)
            return classification
        except Exception as e:
            logger.error(f"Classification error after retries: {e}")
            return "phi3"

    def clear_cache(self):
        if hasattr(self, 'cache') and self.cache:
            self.cache.clear()
            logger.info("Router cache cleared")