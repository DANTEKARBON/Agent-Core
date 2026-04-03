import logging
import json
from pathlib import Path
from datetime import datetime
from core.tracing import get_request_id

# Убедимся, что директория logs существует
Path("logs").mkdir(exist_ok=True)

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
        }
        request_id = getattr(record, 'request_id', None)
        if not request_id:
            request_id = get_request_id()
        if request_id:
            log_entry["request_id"] = request_id

        if hasattr(record, 'extra') and isinstance(record.extra, dict):
            log_entry.update(record.extra)

        if record.name:
            log_entry["logger"] = record.name

        return json.dumps(log_entry, ensure_ascii=False)

# Настройка корневого логгера — ТОЛЬКО ФАЙЛ
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Удаляем все существующие обработчики
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Добавляем только файловый обработчик
file_handler = logging.FileHandler("logs/agent.log", encoding="utf-8")
file_handler.setFormatter(JSONFormatter())
root_logger.addHandler(file_handler)

# Создаём логгер для использования в коде
logger = logging.getLogger("agent-core")

# Обёртка для поддержки extra
original_log = logger._log

def _log(self, level, msg, args, exc_info=None, extra=None, **kwargs):
    if extra is None:
        extra = {}
    if 'request_id' not in extra:
        req_id = get_request_id()
        if req_id:
            extra['request_id'] = req_id
    original_log(level, msg, args, exc_info=exc_info, extra={'extra': extra} if extra else None, **kwargs)

logger._log = _log.__get__(logger, type(logger))

# Добавим удобные функции
def info(msg, extra=None, **kwargs):
    logger.info(msg, extra={'extra': extra} if extra else None, **kwargs)

def error(msg, extra=None, **kwargs):
    logger.error(msg, extra={'extra': extra} if extra else None, **kwargs)

def warning(msg, extra=None, **kwargs):
    logger.warning(msg, extra={'extra': extra} if extra else None, **kwargs)

def debug(msg, extra=None, **kwargs):
    logger.debug(msg, extra={'extra': extra} if extra else None, **kwargs)

def exception(msg, extra=None, **kwargs):
    logger.exception(msg, extra={'extra': extra} if extra else None, **kwargs)
