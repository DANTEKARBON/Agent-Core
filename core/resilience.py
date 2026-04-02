import time
import threading
import functools
from typing import Callable, Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """
    Circuit breaker pattern. Состояния:
    - CLOSED: нормальная работа, пропускаем запросы.
    - OPEN: запросы блокируются, возвращается ошибка.
    - HALF_OPEN: пробуем один запрос, если успех -> закрываем, иначе снова открываем.
    """
    def __init__(self, name: str, failure_threshold: int = 3, recovery_timeout: float = 60.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = "CLOSED"
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        with self._lock:
            if self._state == "OPEN":
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = "HALF_OPEN"
                    logger.info(f"CircuitBreaker {self.name} перешёл в HALF_OPEN")
                else:
                    raise Exception(f"CircuitBreaker {self.name} OPEN, запрос отклонён")
            # Для HALF_OPEN или CLOSED пропускаем вызов
        try:
            result = func(*args, **kwargs)
            # Успех: сбрасываем счётчик
            with self._lock:
                if self._state == "HALF_OPEN":
                    self._state = "CLOSED"
                    self._failure_count = 0
                    logger.info(f"CircuitBreaker {self.name} закрыт после успеха")
                elif self._state == "CLOSED":
                    self._failure_count = 0
            return result
        except Exception as e:
            with self._lock:
                self._failure_count += 1
                self._last_failure_time = time.time()
                if self._state == "CLOSED" and self._failure_count >= self.failure_threshold:
                    self._state = "OPEN"
                    logger.warning(f"CircuitBreaker {self.name} перешёл в OPEN после {self._failure_count} ошибок")
                elif self._state == "HALF_OPEN":
                    self._state = "OPEN"
                    logger.warning(f"CircuitBreaker {self.name} вернулся в OPEN из HALF_OPEN после ошибки")
            raise

def with_retry(max_attempts: int = 2, delay: float = 1.0, backoff: float = 1.0):
    """
    Декоратор для повторных попыток при исключениях.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Retry {attempt}/{max_attempts} failed: {e}")
                    if attempt < max_attempts:
                        time.sleep(delay * (backoff ** (attempt - 1)))
            raise last_exception
        return wrapper
    return decorator

def with_timeout(seconds: float):
    """
    Декоратор для таймаута. Использует threading.Timer для прерывания.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            if thread.is_alive():
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            if exception[0]:
                raise exception[0]
            return result[0]
        return wrapper
    return decorator
