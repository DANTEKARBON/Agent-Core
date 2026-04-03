#!/usr/bin/env python3
import sys
import json
import signal
import re
import yaml
import time
import psutil
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style
from prompt_toolkit.patch_stdout import patch_stdout

from core.model_orchestrator import ModelOrchestrator
from core.model_registry import ModelRegistry
from core.tracing import trace_context, get_request_id
from logger_config import logger
import model_manager

console = Console(color_system=None)

orchestrator: Optional[ModelOrchestrator] = None
mm: Optional[model_manager.ModelManager] = None
registry: Optional[ModelRegistry] = None

CACHE_FILE = "cache_dump.json"

def get_model_names():
    if registry:
        return registry.list_models()
    return []

command_completer = WordCompleter(
    ['/status', '/load', '/unload', '/trace', '/clear_cache', '/reset', '/stats', '/reload_config', '/exit', '/help'] + get_model_names(),
    ignore_case=True,
    sentence=True
)

prompt_style = Style.from_dict({'prompt': 'ansicyan bold'})
history = FileHistory('.cli_history')

def signal_handler(sig, frame):
    logger.info(f"Получен сигнал {sig}, завершение...")
    if orchestrator:
        if hasattr(orchestrator, 'save_cache'):
            orchestrator.save_cache(CACHE_FILE)
        orchestrator.shutdown()
    sys.exit(0)

def clean_model_output(text: str) -> str:
    text = re.sub(r'<\|[a-zA-Z_]+\|>', '', text)
    text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
    return text

def show_trace(request_id: str):
    log_file = Path("logs/agent.log")
    if not log_file.exists():
        console.print(f"Лог-файл {log_file} не найден.")
        return

    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    matching = []
    for line in lines:
        if f'"{request_id}"' in line or f'request_id": "{request_id}"' in line:
            try:
                data = json.loads(line)
                matching.append(data)
            except:
                matching.append({"raw": line.strip()})

    if not matching:
        console.print(f"Нет записей для request_id {request_id}.")
        return

    for entry in matching:
        if "raw" in entry:
            console.print(entry["raw"])
        else:
            ts = entry.get('timestamp', '')[:19]
            level = entry.get('level', 'info').upper()
            event = entry.get('event', '')
            extra = {k:v for k,v in entry.items() if k not in ('timestamp','level','event','request_id')}
            console.print(f"[{ts}] {level}: {event}")
            if extra:
                console.print(f"  {extra}")

def reload_config():
    global orchestrator, registry, mm
    try:
        # Перезагружаем реестр
        registry = ModelRegistry()
        # Обновляем ModelManager (передаём новый registry)
        if mm:
            mm.registry = registry
        # Обновляем оркестратор (создаём новый, старый корректно останавливаем)
        if orchestrator:
            orchestrator.shutdown()
        orchestrator = ModelOrchestrator(model_manager=mm, model_registry=registry)
        if hasattr(orchestrator, 'load_cache'):
            orchestrator.load_cache(CACHE_FILE)
        console.print("[green]✔ Конфигурация перезагружена[/green]")
        logger.info("Конфигурация перезагружена по команде /reload_config")
    except Exception as e:
        logger.error(f"Ошибка перезагрузки конфигурации: {e}")
        console.print(f"[red]Ошибка перезагрузки: {e}[/red]")

def handle_command(cmd: str):
    parts = cmd.strip().split()
    if not parts:
        return
    command = parts[0].lower()

    if command == '/exit':
        logger.info("Завершение работы...")
        if orchestrator:
            if hasattr(orchestrator, 'save_cache'):
                orchestrator.save_cache(CACHE_FILE)
            orchestrator.shutdown()
        sys.exit(0)

    elif command == '/status':
        if mm:
            status = mm.get_status()
            mem_info = {}
            total_mem_gb = psutil.virtual_memory().total / (1024**3)
            available_mem_gb = psutil.virtual_memory().available / (1024**3)
            used_by_models_gb = 0.0
            for model, info in status.items():
                port = info.get('port')
                proc_mem = 0.0
                for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                    try:
                        for conn in proc.net_connections(kind='inet'):
                            if conn.laddr.port == port and conn.status == 'LISTEN':
                                proc_mem = proc.memory_info().rss / (1024**3)
                                break
                        if proc_mem > 0:
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                mem_info[model] = proc_mem
                used_by_models_gb += proc_mem
            
            table = Table(title=f"Состояние моделей (система: {used_by_models_gb:.1f}GB / {total_mem_gb:.1f}GB, свободно: {available_mem_gb:.1f}GB)")
            table.add_column("Модель")
            table.add_column("Статус")
            table.add_column("Порт")
            table.add_column("Оцен.размер(GB)", justify="right")
            table.add_column("Реал.память(GB)", justify="right")
            for model, info in status.items():
                mem = mem_info.get(model, 0.0)
                est = info.get('size_gb', 0.0)
                table.add_row(model, info['status'], str(info.get('port', 'N/A')), f"{est:.1f}", f"{mem:.1f}")
            console.print(table)
        else:
            logger.error("Model manager не инициализирован.")

    elif command == '/load':
        if len(parts) < 2:
            console.print("[yellow]Использование: /load <имя_модели>[/yellow]")
            console.print(f"Доступные модели: {', '.join(get_model_names())}")
            return
        model_name = parts[1]
        if mm:
            if mm.is_loaded(model_name):
                console.print(f"[green]✔ Модель {model_name} уже загружена[/green]")
                return
            try:
                if not registry:
                    logger.error("ModelRegistry не инициализирован")
                    return
                model_info = registry.get_model_info(model_name)
                port = model_info["port"]
                path = model_info["path"]
                with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
                    progress.add_task(description="Запуск модели...", total=None)
                    success = mm.load_model(model_name, port, path)
                if success:
                    console.print(f"[green]✔ Модель {model_name} успешно загружена[/green]")
                else:
                    logger.error(f"Не удалось загрузить {model_name}.")
            except KeyError:
                logger.error(f"Модель {model_name} не найдена в реестре")
                console.print(f"[red]Модель {model_name} не найдена в config.yaml[/red]")
            except Exception as e:
                logger.error(f"Ошибка: {e}")
        else:
            logger.error("Model manager не инициализирован.")

    elif command == '/unload':
        if len(parts) < 2:
            console.print("[yellow]Использование: /unload <имя_модели>[/yellow]")
            return
        model_name = parts[1]
        if mm:
            if mm.unload_model(model_name):
                logger.info(f"Модель {model_name} выгружена.")
                console.print(f"[green]✔ Модель {model_name} выгружена[/green]")
            else:
                logger.warning(f"Модель {model_name} не загружена или не удалось выгрузить.")
                console.print(f"[yellow]Модель {model_name} не была загружена[/yellow]")
        else:
            logger.error("Model manager не инициализирован.")

    elif command == '/trace':
        if len(parts) < 2:
            console.print("[yellow]Использование: /trace <request_id>[/yellow]")
            return
        show_trace(parts[1])

    elif command == '/clear_cache':
        if orchestrator:
            orchestrator.clear_cache()
            console.print("[green]✔ Кэш адаптеров очищен[/green]")
        else:
            logger.error("Оркестратор недоступен.")

    elif command == '/reset':
        if orchestrator:
            orchestrator.clear_cache()
            console.print("[green]✔ Кэш адаптеров очищен (сброс контекста)[/green]")
        else:
            logger.error("Оркестратор недоступен.")

    elif command == '/stats':
        if orchestrator and mm:
            cache_stats = orchestrator.get_cache_stats()
            model_status = mm.get_status()
            loaded_models = [m for m, info in model_status.items() if info['status'] == 'running']
            console.print(f"📊 Кэш адаптеров: {cache_stats.get('cached_adapters', 0)}")
            console.print(f"🔌 Загруженные модели: {', '.join(loaded_models) if loaded_models else 'нет'}")
        else:
            logger.error("Оркестратор или ModelManager недоступны.")

    elif command == '/reload_config':
        reload_config()

    elif command == '/help':
        help_text = """
Доступные команды:
  /status               - показать загруженные модели и память
  /load <модель>        - загрузить модель (из config.yaml)
  /unload <модель>      - выгрузить модель
  /trace <request_id>   - показать логи по запросу
  /clear_cache          - очистить кэш адаптеров
  /reset                - очистить кэш адаптеров (сброс контекста)
  /stats                - показать статистику кэша и загруженные модели
  /reload_config        - перезагрузить конфигурацию без перезапуска
  /exit                 - выход
  /help                 - эта справка
        """
        console.print(Panel(help_text, title="Помощь"))

    else:
        console.print(f"Неизвестная команда: {command}")

def main():
    global orchestrator, mm, registry

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        runtime_config = config.get("runtime", {})
        cleanup_interval = runtime_config.get("cleanup_interval", 60)
        idle_timeout = runtime_config.get("idle_timeout", 600)
        max_concurrent_models = runtime_config.get("max_concurrent_models", 2)
        logger.info(f"Конфигурация runtime загружена", extra={"cleanup": cleanup_interval, "idle": idle_timeout, "max_models": max_concurrent_models})
    except Exception as e:
        logger.error(f"Не удалось загрузить config.yaml, используются значения по умолчанию: {e}")
        cleanup_interval = 60
        idle_timeout = 600
        max_concurrent_models = 2

    # Сначала создаём реестр
    try:
        registry = ModelRegistry()
        logger.info("ModelRegistry инициализирован", extra={"models": registry.list_models()})
    except Exception as e:
        logger.error(f"Не удалось инициализировать ModelRegistry: {e}")
        sys.exit(1)

    # Затем ModelManager с передачей registry
    try:
        mm = model_manager.ModelManager(
            cleanup_interval=cleanup_interval,
            idle_timeout=idle_timeout,
            max_concurrent_models=max_concurrent_models,
            registry=registry
        )
        logger.info("Model manager инициализирован")
    except Exception as e:
        logger.error(f"Не удалось инициализировать model manager: {e}")
        mm = None

    # Оркестратор
    try:
        orchestrator = ModelOrchestrator(model_manager=mm, model_registry=registry)
        if hasattr(orchestrator, 'load_cache'):
            orchestrator.load_cache(CACHE_FILE)
    except Exception as e:
        logger.exception("Не удалось инициализировать оркестратор")
        sys.exit(1)

    console.print("[bold magenta]Dante Agent Core готов к работе (новая архитектура).[/bold magenta]")
    console.print("💡 Введите [yellow]/help[/yellow] для списка команд или просто задайте вопрос.")

    session = PromptSession(history=history, auto_suggest=AutoSuggestFromHistory(), completer=command_completer, style=prompt_style)

    with patch_stdout():
        while True:
            try:
                user_input = session.prompt([('class:prompt', '> ')], completer=command_completer)
                user_input = user_input.strip()
                if not user_input:
                    continue

                if user_input.startswith('/'):
                    handle_command(user_input)
                    continue

                start_time = time.time()
                with trace_context() as ctx:
                    response, target_role, metrics, raw_classifier = orchestrator.process(user_input)
                    request_id = ctx.request_id
                elapsed_ms = (time.time() - start_time) * 1000

                clean_response = clean_model_output(response)

                if target_role is None or target_role == "":
                    target_role = "unknown"
                if raw_classifier is None or raw_classifier == "":
                    raw_classifier = "empty"

                display_raw = raw_classifier
                if "ошибка" in raw_classifier.lower() or "exception" in raw_classifier.lower() or "traceback" in raw_classifier.lower():
                    display_raw = "error"
                elif len(raw_classifier) > 30:
                    display_raw = raw_classifier[:27] + "..."

                title_parts = [f"[{target_role}]"]
                title_parts.append(f"[raw:{display_raw}]")
                title_parts.append(f"id:{request_id}")
                title_parts.append(f"{elapsed_ms:.0f}мс")
                if metrics:
                    if 'in' in metrics and 'out' in metrics:
                        title_parts.append(f"tok {metrics['in']}→{metrics['out']}")
                    if 'ttft_ms' in metrics and metrics['ttft_ms'] is not None:
                        title_parts.append(f"TTFT {metrics['ttft_ms']:.0f}мс")
                    if 't/s' in metrics and metrics['t/s'] > 0:
                        title_parts.append(f"{metrics['t/s']:.1f} tok/s")
                title = " | ".join(title_parts)

                panel = Panel(clean_response, title=title)
                console.print(panel)
                console.print("─" * console.width)

            except KeyboardInterrupt:
                console.print("\nПрерывание. Используйте /exit для выхода.")
                continue
            except EOFError:
                console.print("\nВыход...")
                break
            except Exception as e:
                logger.exception("Необработанная ошибка в главном цикле")
                console.print(f"Ошибка: {e}")

    if orchestrator:
        orchestrator.shutdown()

if __name__ == "__main__":
    main()
