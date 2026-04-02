from adapters.base_llm_adapter import BaseLLMAdapter
from typing import Dict, Any

class OllamaAdapter(BaseLLMAdapter):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        self.base_url = base_url
        self.model = model

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        # Заглушка
        return {"response": f"[Ollama stub] {prompt[:50]}...", "metadata": {}}
    def load_model(self) -> bool: return True
    def unload_model(self) -> bool: return True
