import requests
import json
import time
import logging
from typing import Dict, Any, Optional
from adapters.base_llm_adapter import BaseLLMAdapter
from core.contracts import LLMResponse

logger = logging.getLogger(__name__)

class MLXOpenAIAdapter(BaseLLMAdapter):
    """
    Адаптер для MLX-сервера с полной обработкой ошибок и таймаутами.
    Поддерживает автоматическую загрузку модели через ModelManager.
    """

    def __init__(self, base_url: str, model_name: str, model_manager=None, max_tokens: int = 500, temperature: float = 0.7):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.model_manager = model_manager
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._session = requests.Session()
        self._timeout = (5, 120)

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def _ensure_model_loaded(self):
        """Вызывает ModelManager для загрузки модели, если менеджер передан."""
        if self.model_manager:
            self.model_manager.ensure_model_loaded(self.model_name)
            self.model_manager.update_last_used(self.model_name)

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        # Гарантируем, что модель загружена
        self._ensure_model_loaded()

        url = f"{self.base_url}/v1/chat/completions"
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stop": ["\n\n", "User:", "Пользователь:"],
        }
        start_time = time.time()
        ttft_ms = None
        full_text = ""
        input_tokens = 0
        output_tokens = 0

        try:
            response = self._session.post(url, json=payload, timeout=self._timeout, stream=True)
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data_str)
                        if ttft_ms is None:
                            ttft_ms = (time.time() - start_time) * 1000
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                full_text += content
                        if 'usage' in chunk:
                            usage = chunk['usage']
                            input_tokens = usage.get('prompt_tokens', 0)
                            output_tokens = usage.get('completion_tokens', 0)
                    except json.JSONDecodeError:
                        continue

            generation_time_ms = (time.time() - start_time) * 1000
            if ttft_ms is None:
                ttft_ms = generation_time_ms

            if input_tokens == 0:
                input_tokens = self._estimate_tokens(prompt)
            if output_tokens == 0:
                output_tokens = self._estimate_tokens(full_text)

            return {
                "response": full_text,
                "metadata": {
                    "model": self.model_name,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "generation_time_ms": generation_time_ms,
                    "ttft_ms": ttft_ms,
                    "timestamp": start_time,
                    "status": "success"
                }
            }
        except requests.exceptions.Timeout as e:
            generation_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Timeout: {e}"
            logger.error(f"MLX timeout: {error_msg}")
            return {
                "response": "",
                "metadata": {
                    "model": self.model_name,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "generation_time_ms": generation_time_ms,
                    "ttft_ms": None,
                    "timestamp": start_time,
                    "status": "error",
                    "error": error_msg
                }
            }
        except requests.exceptions.ConnectionError as e:
            generation_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Connection error: {e}"
            logger.error(f"MLX connection error: {error_msg}")
            return {
                "response": "",
                "metadata": {
                    "model": self.model_name,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "generation_time_ms": generation_time_ms,
                    "ttft_ms": None,
                    "timestamp": start_time,
                    "status": "error",
                    "error": error_msg
                }
            }
        except Exception as e:
            generation_time_ms = (time.time() - start_time) * 1000
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"MLX request failed: {error_msg}")
            return {
                "response": "",
                "metadata": {
                    "model": self.model_name,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "generation_time_ms": generation_time_ms,
                    "ttft_ms": None,
                    "timestamp": start_time,
                    "status": "error",
                    "error": error_msg
                }
            }

    def generate_contract(self, prompt: str, max_tokens: int = 5, temperature: float = 0.0) -> LLMResponse:
        self._ensure_model_loaded()
        result = self.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        text = result.get("response", "")
        meta = result.get("metadata", {})
        status = meta.get("status", "success")
        error = meta.get("error")
        return LLMResponse(
            text=text.strip(),
            prompt_tokens=meta.get("input_tokens", 0),
            completion_tokens=meta.get("output_tokens", 0),
            total_tokens=meta.get("input_tokens", 0) + meta.get("output_tokens", 0),
            status=status,
            error=error,
            ttft_ms=meta.get("ttft_ms")
        )

    def load_model(self) -> bool:
        try:
            r = self._session.get(f"{self.base_url}/v1/models", timeout=5)
            return r.status_code == 200
        except:
            return False

    def unload_model(self) -> bool:
        return True

    def close(self):
        self._session.close()
