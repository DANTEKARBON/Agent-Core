import requests
import json
import time
from typing import Dict, Any
from adapters.base_llm_adapter import BaseLLMAdapter

class MLXOpenAIAdapter(BaseLLMAdapter):
    def __init__(self, base_url: str, model_path: str = "default", max_tokens: int = 500, temperature: float = 0.7):
        self.base_url = base_url.rstrip('/')
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model_path,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": False
        }
        start_time = time.time()
        ttft_ms = None
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            generation_time_ms = (time.time() - start_time) * 1000
            # TTFT для не-стриминга примерно равно общему времени
            ttft_ms = generation_time_ms
            # Пытаемся извлечь токены, если есть в ответе
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            text = data["choices"][0]["message"]["content"]
            return {
                "response": text,
                "metadata": {
                    "model": self.model_path,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "generation_time_ms": generation_time_ms,
                    "ttft_ms": ttft_ms,
                    "timestamp": start_time
                }
            }
        except Exception as e:
            generation_time_ms = (time.time() - start_time) * 1000
            # В случае ошибки возвращаем пустой ответ с метаданными об ошибке
            return {
                "response": f"[Ошибка MLX адаптера: {e}]",
                "metadata": {
                    "model": self.model_path,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "generation_time_ms": generation_time_ms,
                    "ttft_ms": None,
                    "timestamp": start_time,
                    "error": str(e)
                }
            }

    def load_model(self) -> bool:
        # Для mlx-openai-server модель уже запущена отдельно, просто проверяем доступность
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except:
            return False

    def unload_model(self) -> bool:
        # Не управляем процессом, просто возвращаем True
        return True
