from abc import ABC, abstractmethod
from core.contracts import LLMResponse

class BaseLLMAdapter(ABC):
    """
    Базовый интерфейс для всех LLM-адаптеров.
    """
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Отправляет запрос модели и возвращает структурированный ответ LLMResponse.
        """
        pass
    
    @abstractmethod
    def load_model(self) -> bool:
        """
        Загружает модель (если адаптер управляет загрузкой самостоятельно).
        Возвращает True при успехе.
        """
        pass
    
    @abstractmethod
    def unload_model(self) -> bool:
        """
        Выгружает модель.
        """
        pass
