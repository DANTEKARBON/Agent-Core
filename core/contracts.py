from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMResponse:
    """Стандартный контракт ответа LLM."""
    text: str
    tokens_input: int
    tokens_output: int
    latency_ms: float
    model: str
    status: str  # "ok", "error", "fallback"
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "LLMResponse":
        """Преобразует старый словарный формат (от адаптера) в LLMResponse."""
        metadata = data.get("metadata", {})
        return cls(
            text=data.get("response", ""),
            tokens_input=metadata.get("input_tokens", 0),
            tokens_output=metadata.get("output_tokens", 0),
            latency_ms=metadata.get("generation_time_ms", 0.0),
            model=metadata.get("model", "unknown"),
            status="error" if metadata.get("error") else "ok",
            error=metadata.get("error"),
        )
