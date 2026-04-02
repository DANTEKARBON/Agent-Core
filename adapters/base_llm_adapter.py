from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from core.contracts import LLMResponse

class BaseLLMAdapter(ABC):
    """
    Базовый интерфейс для всех LLM-адаптеров.
    """
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        [DEPRECATED] Возвращает словарь для обратной совместимости.
        Новый код должен использовать generate_contract.
        """
        pass
    
    def generate_contract(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Возвращает стандартизированный LLMResponse.
        По умолчанию преобразует результат generate().
        """
        raw = self.generate(prompt, **kwargs)
        return LLMResponse.from_dict(raw)
    
    @abstractmethod
    def load_model(self) -> bool:
        pass
    
    @abstractmethod
    def unload_model(self) -> bool:
        pass
