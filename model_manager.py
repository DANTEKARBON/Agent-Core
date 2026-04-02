import subprocess
import time
import logging
import threading
import socket
import os
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

# Конфигурация моделей (укажите реальные пути)
MODEL_CONFIG = {
    "reasoner": {"port": 10242, "model_path": "/path/to/reasoner"},
    "classifier": {"port": 10243, "model_path": "/path/to/classifier"},
    "coder": {"port": 10244, "model_path": "/path/to/coder"},
    "phi3": {"port": 10245, "model_path": "/path/to/phi3"},
}

class ModelProcess:
    def __init__(self, name: str, port: int, model_path: str, process: subprocess.Popen):
        self.name = name
        self.port = port
        self.model_path = model_path
        self.process = process
        self.last_used = time.time()

    def is_alive(self) -> bool:
        return self.process.poll() is None

    def update_used(self):
        self.last_used = time.time()


class ModelManager:
    def __init__(self,
                 cleanup_interval: int = 60,
                 idle_timeout: int = 300,
                 max_concurrent_models: int = 2,
                 port_check_timeout: int = 10):
        self.models: Dict[str, ModelProcess] = {}
        self._lock = threading.Lock()
        self.cleanup_interval = cleanup_interval
        self.idle_timeout = idle_timeout
        self.max_concurrent_models = max_concurrent_models
        self.port_check_timeout = port_check_timeout
        self._stop_cleanup = threading.Event()

        os.makedirs("logs", exist_ok=True)

        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        logger.info(f"ModelManager initialized: cleanup_interval={cleanup_interval}s, "
                    f"idle_timeout={idle_timeout}s, max_models={max_concurrent_models}")
        logger.info("Cleanup worker thread started")

    def _launch(self, name: str, port: int, model_path: str) -> Optional[subprocess.Popen]:
        cmd = [
            "mlx-openai-server", "launch",
            "--model-path", model_path,
            "--model-type", "lm",
            "--port", str(port),
            "--host", "127.0.0.1",
            "--max-concurrency", "1",
            "--queue-timeout", "300",
            "--queue-size", "100"
        ]
        try:
            log_file = open(f"logs/{name}_server.log", "w")
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
            return proc
        except Exception as e:
            logger.error(f"Ошибка запуска {name}: {e}")
            return None

    def _wait_for_port(self, port: int, timeout: int = None) -> bool:
        if timeout is None:
            timeout = self.port_check_timeout
        start = time.time()
        while time.time() - start < timeout:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    return True
            except (socket.error, ConnectionRefusedError):
                time.sleep(0.5)
        return False

    def _is_memory_error(self, error_msg: str) -> bool:
        error_lower = error_msg.lower()
        return any(kw in error_lower for kw in [
            "out of memory", "memory allocation",
            "insufficient memory", "metal out of memory",
            "cannot allocate memory"
        ])

    def _emergency_memory_cleanup(self):
        logger.warning("Emergency memory cleanup: unloading idle models")
        cutoff = time.time() - 30
        to_unload = []
        with self._lock:
            for name, mp in self.models.items():
                if mp.last_used < cutoff:
                    to_unload.append(name)
        for name in to_unload:
            self.unload_model(name)

    def _evict_lru(self):
        if not self.models:
            return
        lru_name = min(self.models.items(), key=lambda x: x[1].last_used)[0]
        logger.info(f"LRU eviction: unloading {lru_name}")
        self.unload_model(lru_name)

    def _load_model(self, name: str, port: int, model_path: str, retries: int = 2) -> bool:
        with self._lock:
            if name in self.models:
                mp = self.models[name]
                if mp.is_alive():
                    mp.update_used()
                    logger.info(f"Модель {name} уже загружена, обновлено время использования")
                    return True
                else:
                    logger.warning(f"Модель {name} была загружена, но процесс мёртв. Удаляем.")
                    del self.models[name]

            if len(self.models) >= self.max_concurrent_models:
                self._evict_lru()

        for attempt in range(retries):
            try:
                logger.info(f"Запуск модели {name} (попытка {attempt+1})...")
                proc = self._launch(name, port, model_path)
                if proc is None:
                    continue

                if not self._wait_for_port(port):
                    logger.error(f"Модель {name} не ответила на порту {port} после запуска")
                    proc.terminate()
                    continue

                if proc.poll() is None:
                    with self._lock:
                        self.models[name] = ModelProcess(name, port, model_path, proc)
                    logger.info(f"Модель {name} загружена на порту {port}")
                    return True
                else:
                    logger.error(f"Модель {name} завершилась сразу после запуска (код {proc.returncode})")
                    try:
                        with open(f"logs/{name}_server.log", "r") as f:
                            stderr_content = f.read()
                            logger.error(f"Содержимое лога {name}_server.log: {stderr_content[:500]}")
                    except:
                        pass
            except Exception as e:
                logger.error(f"Ошибка при загрузке {name} (попытка {attempt+1}): {e}")
                if self._is_memory_error(str(e)):
                    self._emergency_memory_cleanup()
        return False

    def load_model(self, name: str, port: int = None, model_path: str = None) -> bool:
        """
        Загружает модель.
        Если порт и путь не указаны, берёт из MODEL_CONFIG.
        Поддерживает вызов как с одним аргументом (только имя), так и с тремя.
        """
        if port is None or model_path is None:
            if name not in MODEL_CONFIG:
                logger.error(f"Model {name} not configured")
                return False
            config = MODEL_CONFIG[name]
            port = port or config["port"]
            model_path = model_path or config["model_path"]
        return self._load_model(name, port, model_path)

    def unload_model(self, name: str) -> bool:
        with self._lock:
            if name not in self.models:
                return False
            mp = self.models[name]

        try:
            mp.process.terminate()
            mp.process.wait(timeout=10)
            logger.info(f"Модель {name} выгружена")
        except subprocess.TimeoutExpired:
            mp.process.kill()
            mp.process.wait()
            logger.warning(f"Модель {name} пришлось убить")
        except Exception as e:
            logger.error(f"Ошибка при выгрузке {name}: {e}")
            return False
        finally:
            with self._lock:
                if name in self.models:
                    del self.models[name]
        return True

    def is_loaded(self, name: str) -> bool:
        with self._lock:
            if name not in self.models:
                return False
            mp = self.models[name]
            if not mp.is_alive():
                del self.models[name]
                return False
            return True

    def update_last_used(self, name: str):
        with self._lock:
            if name in self.models:
                self.models[name].update_used()
                logger.debug(f"Обновлено время использования модели {name}")

    def get_status(self) -> Dict[str, Dict[str, str]]:
        """Возвращает статус всех моделей."""
        status = {}
        with self._lock:
            for name, mp in self.models.items():
                status[name] = {
                    "status": "running" if mp.is_alive() else "stopped",
                    "port": mp.port
                }
        return status

    def _cleanup_worker(self):
        logger.info("Cleanup worker started running")
        while not self._stop_cleanup.is_set():
            try:
                time.sleep(self.cleanup_interval)
                now = time.time()
                to_unload = []
                dead_processes = []

                with self._lock:
                    for name, mp in self.models.items():
                        if not mp.is_alive():
                            dead_processes.append(name)
                            continue
                        idle = now - mp.last_used
                        logger.info(f"Cleanup check: модель {name} idle={idle:.0f}s (timeout={self.idle_timeout}s)")
                        if idle > self.idle_timeout:
                            to_unload.append(name)

                for name in dead_processes:
                    logger.warning(f"Найден мёртвый процесс модели {name}, удаляем запись")
                    with self._lock:
                        if name in self.models:
                            del self.models[name]

                for name in to_unload:
                    self.unload_model(name)

            except Exception as e:
                logger.error(f"Ошибка в cleanup worker: {e}", exc_info=True)

    def stop(self):
        """Останавливает cleanup worker и выгружает все модели."""
        self._stop_cleanup.set()
        self._cleanup_thread.join(timeout=5)
        for name in list(self.models.keys()):
            self.unload_model(name)
        logger.info("ModelManager остановлен")

    def shutdown(self):
        """Алиас для stop(), используется для graceful shutdown."""
        logger.info("ModelManager shutting down...")
        self.stop()
        logger.info("ModelManager shutdown complete")