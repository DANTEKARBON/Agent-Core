import yaml
from pathlib import Path
from typing import Dict, Optional, List
from logger_config import logger

class ModelRegistry:
    """
    Единый реестр моделей. Загружает config.yaml и предоставляет
    информацию о моделях (порт, путь, размер).
    """
    _instance = None

    def __new__(cls, config_path: str = "config.yaml"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: str = "config.yaml"):
        if self._initialized:
            return
        self.config_path = config_path
        self._load_config()
        self._initialized = True
        logger.info("ModelRegistry инициализирован", extra={"models": list(self.models.keys())})

    def _load_config(self):
        """Загружает конфигурацию из YAML-файла."""
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Конфиг не найден: {self.config_path}")
        with open(config_file, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)
        self.models = full_config.get("models", {})
        if not self.models:
            raise ValueError("В config.yaml нет секции 'models'")

    def get_model_info(self, name: str) -> Dict[str, any]:
        """
        Возвращает информацию о модели.
        :param name: имя модели (phi3, coder, reasoner)
        :return: словарь с ключами path, port (и, возможно, size_gb)
        :raises KeyError: если модели нет в реестре
        """
        if name not in self.models:
            raise KeyError(f"Модель '{name}' не найдена. Доступны: {self.list_models()}")
        return self.models[name].copy()

    def list_models(self) -> List[str]:
        """Возвращает список имён всех моделей."""
        return list(self.models.keys())

    def validate_model(self, name: str) -> bool:
        """
        Проверяет, существует ли модель в конфиге и указаны ли обязательные поля.
        """
        try:
            info = self.get_model_info(name)
            required = {"path", "port"}
            return required.issubset(info.keys())
        except KeyError:
            return False

    def get_port(self, name: str) -> int:
        return self.get_model_info(name)["port"]

    def get_path(self, name: str) -> str:
        return self.get_model_info(name)["path"]

# Для удобного импорта
default_registry = ModelRegistry()
