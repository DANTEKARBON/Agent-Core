from .base import LLMAdapter
from .mlx_openai_adapter import MLXOpenAIAdapter

def create_llm_adapter(model_config):
    """Создаёт адаптер для модели на основе её конфигурации."""
    adapter_type = model_config.get("type")
    params = model_config.get("params", {})
    if adapter_type == "mlx_openai":
        return MLXOpenAIAdapter(params)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
