import logging
import yaml
from model_manager import ModelManager
from adapters.factory import LLMAdapterFactory
from core.cache import TTLCache
from core.router_service import RouterService
from core.generation_service import GenerationService
from core.tracing import get_request_id
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    def __init__(self, model_manager: Optional[ModelManager] = None, cleanup_interval=60, idle_timeout=300, max_concurrent_models=2):
        if model_manager is not None:
            self.model_manager = model_manager
            logger.info("Using provided model manager instance")
        else:
            self.model_manager = ModelManager(
                cleanup_interval=cleanup_interval,
                idle_timeout=idle_timeout,
                max_concurrent_models=max_concurrent_models
            )
            logger.info("Created new model manager instance")

        self.config = self._load_config()
        self.llm_backends = self.config.get("llm_backends", {})
        self.prompts = self.config.get("prompts", {})
        self.primary_adapter = self._create_adapter_from_config(self.llm_backends.get("primary", {}))
        self.fallback_adapter = self._create_adapter_from_config(self.llm_backends.get("fallback", {}))
        self.cache = TTLCache(ttl_seconds=3600, enabled=True)

        self.router_service = RouterService(
            primary_adapter=self.primary_adapter,
            prompts=self.prompts,
            config=self.config,
            model_manager=self.model_manager,
            cache=self.cache
        )
        self.generation_service = GenerationService(
            primary_adapter=self.primary_adapter,
            fallback_adapter=self.fallback_adapter,
            prompts=self.prompts,
            config=self.config,
            model_manager=self.model_manager,
            cache=self.cache
        )

    def _load_config(self):
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)

    def _create_adapter_from_config(self, backend_config: Dict[str, Any]) -> Optional[Any]:
        if not backend_config:
            return None
        try:
            return LLMAdapterFactory.create(backend_config)
        except Exception as e:
            logger.error(f"Failed to create adapter: {e}")
            return None

    def process(self, prompt: str) -> str:
        request_id = get_request_id()
        logger.info(f"[{request_id}] received: {prompt[:100]}")
        target_role = self.router_service.classify(prompt)
        logger.info(f"[{request_id}] classified as: {target_role}")
        response = self.generation_service.generate(target_role, prompt)
        logger.info(f"[{request_id}] completed")
        return response

    def shutdown(self):
        logger.info("Shutting down orchestrator")
        self.model_manager.stop()

    def reset_context(self):
        self.clear_cache()
        if hasattr(self.generation_service, 'reset_context'):
            self.generation_service.reset_context()
            logger.debug("Orchestrator context reset")

    def clear_cache(self):
        if hasattr(self.router_service, 'clear_cache'):
            self.router_service.clear_cache()
        if hasattr(self.generation_service, 'clear_cache'):
            self.generation_service.clear_cache()
        logger.info("All caches cleared")