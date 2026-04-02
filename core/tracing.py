"""
Модуль трассировки запросов.
Предоставляет контекстную переменную для request_id и утилиты для работы с ней.
"""

import uuid
import contextvars
from typing import Optional

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('request_id', default='')

def generate_request_id() -> str:
    """Генерирует короткий уникальный идентификатор запроса."""
    return uuid.uuid4().hex[:8]

def set_request_id(request_id: str) -> None:
    """Устанавливает request_id в текущем контексте."""
    _request_id_var.set(request_id)

def get_request_id() -> str:
    """Возвращает request_id из текущего контекста."""
    return _request_id_var.get()

class trace_context:
    """
    Контекстный менеджер для установки request_id на время выполнения блока.
    Если request_id не передан, генерируется новый.
    """
    def __init__(self, request_id: Optional[str] = None):
        self.request_id = request_id or generate_request_id()
        self.token = None

    def __enter__(self):
        self.token = _request_id_var.set(self.request_id)
        return self

    def __exit__(self, *args):
        _request_id_var.reset(self.token)
