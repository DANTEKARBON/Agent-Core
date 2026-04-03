import subprocess
import time
import threading
import socket
import os
import requests
from typing import Dict, Optional, List
import psutil
from logger_config import logger

class ModelProcess:
    def __init__(self, name: str, port: int, model_path: str, process: subprocess.Popen, size_gb: float = 0.0):
        self.name = name
        self.port = port
        self.model_path = model_path
        self.process = process
        self.last_used = time.time()
        self.size_gb = size_gb

    def is_alive(self) -> bool:
        return self.process.poll() is None

    def update_used(self):
        self.last_used = time.time()


class ModelManager:
    def __init__(self,
                 cleanup_interval: int = 60,
                 idle_timeout: int = 600,
                 max_concurrent_models: int = 2,
                 port_check_timeout: int = 60,
                 memory_reserve_gb: float = 1.5,
                 health_check_interval: int = 30,
                 registry=None):
        self.models: Dict[str, ModelProcess] = {}
        self._lock = threading.Lock()
        self.cleanup_interval = cleanup_interval
        self.idle_timeout = idle_timeout
        self.max_concurrent_models = max_concurrent_models
        self.port_check_timeout = port_check_timeout
        self.memory_reserve_gb = memory_reserve_gb
        self._stop_cleanup = threading.Event()
        self.health_check_interval = health_check_interval
        self.registry = registry  # ModelRegistry instance

        os.makedirs("logs", exist_ok=True)

        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=False)
        self._cleanup_thread.start()
        
        self._health_thread = threading.Thread(target=self._health_check_worker, daemon=False)
        self._health_thread.start()
        
        logger.info(f"ModelManager инициализирован", extra={"cleanup_interval": cleanup_interval, "idle_timeout": idle_timeout, "max_models": max_concurrent_models, "port_check_timeout": port_check_timeout})

    def _estimate_model_size(self, model_path: str) -> float:
        try:
            if os.path.isfile(model_path):
                return os.path.getsize(model_path) / (1024**3)
            elif os.path.isdir(model_path):
                total = 0
                for root, dirs, files in os.walk(model_path):
                    for f in files:
                        total += os.path.getsize(os.path.join(root, f))
                return total / (1024**3)
        except Exception as e:
            logger.warning(f"Не удалось оценить размер модели {model_path}: {e}")
        return 2.0

    def _select_model_to_evict(self, required_free_gb: float) -> List[str]:
        with self._lock:
            if not self.models:
                return []
            sorted_by_age = sorted(self.models.items(), key=lambda x: x[1].last_used)
            to_unload = []
            freed = 0.0
            for name, mp in sorted_by_age:
                if freed >= required_free_gb:
                    break
                to_unload.append(name)
                freed += mp.size_gb
            if freed < required_free_gb:
                remaining = [(n, m) for n, m in self.models.items() if n not in to_unload]
                sorted_by_size = sorted(remaining, key=lambda x: x[1].size_gb, reverse=True)
                for name, mp in sorted_by_size:
                    if freed >= required_free_gb:
                        break
                    if name not in to_unload:
                        to_unload.append(name)
                        freed += mp.size_gb
            return to_unload

    def _launch(self, name: str, port: int, model_path: str) -> Optional[subprocess.Popen]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) == 0:
                logger.error(f"Порт {port} уже занят, невозможно запустить {name}")
                return None

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
        logger.warning("Аварийная очистка памяти: выгрузка неактивных моделей")
        cutoff = time.time() - 30
        to_unload = []
        with self._lock:
            for name, mp in self.models.items():
                if mp.last_used < cutoff:
                    to_unload.append(name)
        for name in to_unload:
            self.unload_model(name)

    def load_model(self, name: str, port: int, model_path: str, retries: int = 2) -> bool:
        size_gb = self._estimate_model_size(model_path)
        required_free = size_gb + self.memory_reserve_gb
        free_gb = psutil.virtual_memory().available / (1024**3)
        
        if free_gb < required_free:
            logger.warning(f"Недостаточно памяти: свободно {free_gb:.1f}GB, требуется {required_free:.1f}GB")
            to_unload = self._select_model_to_evict(required_free - free_gb)
            for m in to_unload:
                logger.info(f"Выгружаем {m} для освобождения памяти под {name}")
                self.unload_model(m)
            free_gb = psutil.virtual_memory().available / (1024**3)
            if free_gb < required_free:
                logger.error(f"После выгрузки всё ещё мало памяти: {free_gb:.1f}GB, отказ от загрузки {name}")
                return False

        with self._lock:
            if name in self.models:
                mp = self.models[name]
                if mp.is_alive():
                    mp.update_used()
                    logger.info(f"Модель {name} уже загружена, обновлено время использования")
                    return True
                else:
                    logger.warning(f"Модель {name} была загружена, но процесс мёртв. Удаляем запись.")
                    del self.models[name]

            if len(self.models) >= self.max_concurrent_models:
                self._evict_lru()

        for attempt in range(retries):
            try:
                logger.info(f"Запуск {name} (попытка {attempt+1})...")
                proc = self._launch(name, port, model_path)
                if proc is None:
                    continue

                if not self._wait_for_port(port):
                    logger.error(f"{name} не ответила на порту {port}")
                    proc.terminate()
                    continue

                if proc.poll() is None:
                    with self._lock:
                        self.models[name] = ModelProcess(name, port, model_path, proc, size_gb)
                    try:
                        child = psutil.Process(proc.pid)
                        mem_used = child.memory_info().rss / (1024**3)
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        logger.warning(f"Не удалось получить память для {name}: {e}")
                        mem_used = 0.0
                    logger.info(f"✅ {name} загружена", extra={"port": port, "mem": f"{mem_used:.1f}GB", "estimated_size": f"{size_gb:.1f}GB"})
                    return True
                else:
                    logger.error(f"{name} завершилась сразу после запуска (код {proc.returncode})")
                    try:
                        with open(f"logs/{name}_server.log", "r") as f:
                            stderr_content = f.read()
                            logger.error(f"Лог {name}_server.log: {stderr_content[:500]}")
                    except:
                        pass
            except Exception as e:
                logger.error(f"Ошибка при загрузке {name} (попытка {attempt+1}): {e}")
                if self._is_memory_error(str(e)):
                    self._emergency_memory_cleanup()
        return False

    def ensure_model_loaded(self, name: str) -> bool:
        """Загружает модель по имени, используя registry для получения порта и пути."""
        if self.is_loaded(name):
            self.update_last_used(name)
            return True
        if not self.registry:
            logger.error(f"ModelManager не имеет registry, невозможно загрузить {name}")
            return False
        try:
            info = self.registry.get_model_info(name)
            port = info["port"]
            path = info["path"]
            return self.load_model(name, port, path)
        except KeyError:
            logger.error(f"Модель {name} не найдена в реестре")
            return False

    def _evict_lru(self):
        if not self.models:
            return
        lru_name = min(self.models.items(), key=lambda x: x[1].last_used)[0]
        logger.info(f"LRU вытеснение: выгружаем {lru_name}")
        self.unload_model(lru_name)

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

    def get_status(self) -> Dict[str, Dict[str, str]]:
        status = {}
        with self._lock:
            for name, mp in self.models.items():
                status[name] = {
                    "status": "running" if mp.is_alive() else "stopped",
                    "port": mp.port,
                    "size_gb": round(mp.size_gb, 2)
                }
        return status

    def _check_model_health(self, mp: ModelProcess) -> bool:
        try:
            url = f"http://127.0.0.1:{mp.port}/v1/models"
            r = requests.get(url, timeout=5)
            return r.status_code == 200
        except Exception as e:
            logger.debug(f"Health check failed for {mp.name}: {e}")
            return False

    def _health_check_worker(self):
        while not self._stop_cleanup.is_set():
            try:
                time.sleep(self.health_check_interval)
                with self._lock:
                    models_copy = list(self.models.items())
                for name, mp in models_copy:
                    if not mp.is_alive():
                        continue
                    if not self._check_model_health(mp):
                        logger.warning(f"Health check: модель {name} не отвечает, принудительная выгрузка")
                        self.unload_model(name)
            except Exception as e:
                logger.error(f"Ошибка в health_check_worker: {e}", exc_info=True)

    def _cleanup_worker(self):
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
                        if idle > self.idle_timeout:
                            to_unload.append(name)

                for name in dead_processes:
                    logger.warning(f"Найден мёртвый процесс {name}, удаляем запись")
                    with self._lock:
                        if name in self.models:
                            del self.models[name]

                for name in to_unload:
                    self.unload_model(name)

            except Exception as e:
                logger.error(f"Ошибка в cleanup worker: {e}", exc_info=True)

    def stop(self):
        self._stop_cleanup.set()
        self._cleanup_thread.join(timeout=5)
        self._health_thread.join(timeout=5)
        for name in list(self.models.keys()):
            self.unload_model(name)
        logger.info("ModelManager остановлен")

    def shutdown(self):
        logger.info("ModelManager завершает работу...")
        self.stop()
        logger.info("ModelManager остановлен полностью")
