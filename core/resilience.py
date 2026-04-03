import time
import functools
import threading
from typing import Callable, Any, Tuple
from core.errors import TimeoutError
from logger_config import logger

def with_retry(attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, 
               exceptions: Tuple[Exception] = (Exception,), max_attempts: int = None):
    actual_attempts = max_attempts if max_attempts is not None else attempts
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            current_delay = delay
            for attempt in range(actual_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt == actual_attempts - 1:
                        raise
                    time.sleep(current_delay)
                    current_delay *= backoff
            raise last_exc
        return wrapper
    return decorator

def with_timeout(seconds: float):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            error = [None]
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    error[0] = e
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            if thread.is_alive():
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds}s")
            if error[0] is not None:
                raise error[0]
            return result[0]
        return wrapper
    return decorator

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.state = "closed"  # closed, open, half-open
        self.last_failure_time = 0
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs):
        with self._lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    logger.info("Circuit breaker переходит в half-open")
                    self.state = "half-open"
                else:
                    raise Exception("Circuit breaker is open")
        try:
            result = func(*args, **kwargs)
            with self._lock:
                if self.state == "half-open":
                    logger.info("Circuit breaker закрыт (успешный вызов)")
                    self.state = "closed"
                    self.failures = 0
            return result
        except Exception as e:
            with self._lock:
                self.failures += 1
                self.last_failure_time = time.time()
                if self.failures >= self.failure_threshold:
                    logger.warning(f"Circuit breaker открыт после {self.failures} ошибок")
                    self.state = "open"
            raise e
