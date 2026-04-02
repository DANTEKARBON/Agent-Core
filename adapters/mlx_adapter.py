import requests
from typing import Dict, Any
from .base import LLMAdapter

class MLXAdapter(LLMAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://127.0.0.1:10240")
        self.model_path = config.get("model_path")
        self.max_tokens = config.get("max_tokens", 100)
        self.temperature = config.get("temperature", 0.7)

    def generate(self, prompt: str, **kwargs) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model_path,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature)
        }
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[Ошибка MLX адаптера: {e}]"
