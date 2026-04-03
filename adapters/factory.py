from typing import Dict, Any, Optional
from adapters.base_llm_adapter import BaseLLMAdapter

class MockAdapter(BaseLLMAdapter):
    def __init__(self, response: str = "Mock response from fallback"):
        self.response = response

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        return {
            "response": self.response,
            "metadata": {
                "model": "mock",
                "input_tokens": 0,
                "output_tokens": 0,
                "generation_time_ms": 0.0,
                "ttft_ms": 0.0,
                "timestamp": 0.0
            }
        }

    def load_model(self) -> bool:
        return True

    def unload_model(self) -> bool:
        return True

class LLMAdapterFactory:
    @staticmethod
    def create(backend_config: Dict[str, Any], model_manager=None) -> BaseLLMAdapter:
        backend_type = backend_config.get("type", "").lower()
        if backend_type == "mlx_openai":
            from adapters.mlx_openai_adapter import MLXOpenAIAdapter
            return MLXOpenAIAdapter(
                base_url=backend_config.get("base_url"),
                model_name=backend_config.get("model_name", "default"),
                model_manager=model_manager,
                max_tokens=backend_config.get("max_tokens", 500),
                temperature=backend_config.get("temperature", 0.7)
            )
        elif backend_type == "mock":
            return MockAdapter(response=backend_config.get("response", "Mock response"))
        elif backend_type == "ollama":
            from adapters.ollama_adapter import OllamaAdapter
            return OllamaAdapter(
                base_url=backend_config.get("base_url", "http://localhost:11434"),
                model=backend_config.get("model_name", "llama2")
            )
        elif backend_type == "llamacpp":
            from adapters.llamacpp_adapter import LlamaCppAdapter
            return LlamaCppAdapter(
                base_url=backend_config.get("base_url", "http://localhost:8080"),
                model=backend_config.get("model_name", "default")
            )
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")
