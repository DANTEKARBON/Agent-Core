"""
Кастомные исключения для agent-core.
"""

class AgentError(Exception):
    """Базовое исключение для всех ошибок агента."""
    pass

class ModelError(AgentError):
    """Ошибка, связанная с LLM-моделью."""
    pass

class FallbackExhaustedError(AgentError):
    """Все модели из fallback-цепочки не смогли обработать запрос."""
    pass

class CircuitBreakerOpenError(AgentError):
    """Circuit breaker разомкнут."""
    pass

class TimeoutError(AgentError):
    """Превышено время ожидания."""
    pass

class CacheError(AgentError):
    """Ошибка при работе с кэшем."""
    pass
