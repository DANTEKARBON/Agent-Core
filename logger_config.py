"""
Настройка структурированного логирования.
Логи пишутся в файл в формате JSON и в консоль в текстовом формате.
"""

import logging
import sys
from pathlib import Path

import structlog
from structlog.processors import JSONRenderer
from structlog.dev import ConsoleRenderer

# Импортируем функцию получения request_id
from core.tracing import get_request_id

def add_request_id(logger, method_name, event_dict):
    """
    Добавляет request_id в словарь события, если он установлен в контексте.
    """
    request_id = get_request_id()
    if request_id:
        event_dict['request_id'] = request_id
    return event_dict

# Общие процессоры для всех логгеров
shared_processors = [
    structlog.stdlib.add_log_level,
    structlog.stdlib.PositionalArgumentsFormatter(),
    add_request_id,                     # наш кастомный процессор
    structlog.processors.TimeStamper(fmt="iso"),
]

# Конфигурация structlog для стандартного использования
structlog.configure(
    processors=shared_processors + [
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        JSONRenderer()                  # по умолчанию JSON, но для консоли переопределим
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Настройка стандартного модуля logging
logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
    handlers=[]
)

# Создаём папку для логов, если её нет
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Обработчик для файла (JSON)
file_handler = logging.FileHandler("logs/agent.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(message)s"))

# Обработчик для консоли (текст)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Переопределяем процессоры для консольного обработчика, чтобы он выводил текст
def console_renderer():
    return structlog.dev.ConsoleRenderer()

# Для консоли используем отдельную конфигурацию процессоров
console_processors = shared_processors + [
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
    structlog.processors.UnicodeDecoder(),
    console_renderer(),
]

# Создаём логгер для консоли
console_logger = structlog.wrap_logger(
    logging.getLogger("console"),
    processors=console_processors,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Добавляем обработчики
logging.getLogger().addHandler(file_handler)
logging.getLogger().addHandler(console_handler)

# Для удобства экспортируем логгер
logger = structlog.get_logger()
