class LLMError(Exception):
    """Базовое исключение для LLM слоя."""
    pass

class ModelLoadError(LLMError):
    """Ошибка загрузки модели."""
    pass

class ModelTimeoutError(LLMError):
    """Таймаут при обращении к модели."""
    pass

class ModelResponseError(LLMError):
    """Некорректный ответ модели."""
    pass

class ClassificationError(LLMError):
    """Ошибка классификации запроса."""
    pass
