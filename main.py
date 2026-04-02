#!/usr/bin/env python3
"""
Основная точка входа. Обеспечивает интерактивный CLI с поддержкой команд.
Использует rich для цветного вывода (если доступен), иначе обычный print.
"""

import sys
import json
import signal
import re
import yaml
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    def rprint(*args, **kwargs):
        print(*args)

from core.orchestrator import AgentOrchestrator
from core.tracing import trace_context
from logger_config import logger
import model_manager

orchestrator: Optional[AgentOrchestrator] = None
mm: Optional[model_manager.ModelManager] = None

def signal_handler(sig, frame):
    logger.info(f"Received signal {sig}, shutting down...")
    if orchestrator:
        orchestrator.shutdown()
    if mm is not None and hasattr(mm, "shutdown"):
        mm.shutdown()
    sys.exit(0)

def clean_model_output(text: str) -> str:
    text = re.sub(r'<\|[a-zA-Z_]+\|>', '', text)
    text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
    return text

def show_trace(request_id: str):
    log_file = Path("logs/agent.log")
    if not log_file.exists():
        print(f"Лог-файл {log_file} не найден.")
        return

    entries = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get('request_id') == request_id:
                entries.append(record)

    if not entries:
        print(f"Записи для request_id {request_id} не найдены.")
        return

    entries.sort(key=lambda x: x.get('timestamp', ''))
    for entry in entries:
        timestamp = entry.get('timestamp', '')
        level = entry.get('level', 'info').upper()
        event = entry.get('event', '')
        if RICH_AVAILABLE:
            console.print(f"[{timestamp}] [bold]{level}[/bold]: {event}")
        else:
            print(f"[{timestamp}] {level}: {event}")
        for k, v in entry.items():
            if k not in ('timestamp', 'level', 'event', 'request_id'):
                if RICH_AVAILABLE:
                    console.print(f"  [dim]{k}: {v}[/dim]")
                else:
                    print(f"  {k}: {v}")

def handle_command(cmd: str):
    parts = cmd.split()
    if not parts:
        return
    command = parts[0].lower()

    if command == '/exit':
        print("Shutting down...")
        if orchestrator:
            orchestrator.shutdown()
        if mm is not None and hasattr(mm, "shutdown"):
            mm.shutdown()
        sys.exit(0)

    elif command == '/status':
        if mm:
            status = mm.get_status()
            if RICH_AVAILABLE:
                table = Table(title="Model Status")
                table.add_column("Model", style="cyan")
                table.add_column("Status", style="green")
                table.add_column("Port", style="yellow")
                for model, info in status.items():
                    table.add_row(model, info['status'], str(info.get('port', 'N/A')))
                console.print(table)
            else:
                print("Model status:")
                for model, info in status.items():
                    print(f"  {model}: {info['status']} (port {info.get('port', 'N/A')})")
        else:
            print("Model manager not initialized.")

    elif command == '/load' and len(parts) > 1:
        model_name = parts[1]
        if mm:
            print(f"Loading {model_name}...")
            try:
                mm.load_model(model_name)
                print(f"Model {model_name} loaded successfully.")
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
        else:
            print("Model manager not initialized.")

    elif command == '/unload' and len(parts) > 1:
        model_name = parts[1]
        if mm:
            print(f"Unloading {model_name}...")
            try:
                mm.unload_model(model_name)
                print(f"Model {model_name} unloaded.")
            except Exception as e:
                print(f"Failed to unload {model_name}: {e}")
        else:
            print("Model manager not initialized.")

    elif command == '/trace' and len(parts) > 1:
        request_id = parts[1]
        show_trace(request_id)

    elif command == '/clear_cache':
        if orchestrator and hasattr(orchestrator, 'clear_cache'):
            orchestrator.clear_cache()
            print("Cache cleared.")
        else:
            print("Clear cache method not available.")

    elif command == '/reset':
        if orchestrator and hasattr(orchestrator, 'reset_context'):
            orchestrator.reset_context()
            print("Context reset.")
        else:
            print("Reset context method not available.")

    elif command == '/help':
        print("Available commands:")
        print("  /status           - Show model status")
        print("  /load <model>     - Load a model (if not already loaded)")
        print("  /unload <model>   - Unload a model")
        print("  /trace <id>       - Show trace for a request ID")
        print("  /clear_cache      - Clear classification and generation caches")
        print("  /reset            - Reset conversation context")
        print("  /exit             - Exit the program")
        print("  /help             - Show this help")
    else:
        print(f"Unknown command: {command}")

def main():
    global orchestrator, mm

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Загружаем конфиг, чтобы получить runtime-параметры
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        runtime_config = config.get("runtime", {})
        cleanup_interval = runtime_config.get("cleanup_interval", 60)
        idle_timeout = runtime_config.get("idle_timeout", 300)
        max_concurrent_models = runtime_config.get("max_concurrent_models", 2)
        logger.info(f"Runtime config loaded: cleanup={cleanup_interval}s, idle_timeout={idle_timeout}s, max_models={max_concurrent_models}")
    except Exception as e:
        logger.error(f"Failed to load config.yaml, using defaults: {e}")
        cleanup_interval = 60
        idle_timeout = 300
        max_concurrent_models = 2

    # Создаём единый model_manager с параметрами из конфига
    try:
        mm = model_manager.ModelManager(
            cleanup_interval=cleanup_interval,
            idle_timeout=idle_timeout,
            max_concurrent_models=max_concurrent_models
        )
        logger.info("Model manager initialized with runtime config")
    except Exception as e:
        logger.error(f"Failed to initialize model manager: {e}")
        mm = None

    # Инициализируем оркестратор, передавая ему тот же model_manager
    try:
        orchestrator = AgentOrchestrator(model_manager=mm)
    except Exception as e:
        logger.exception("Failed to initialize orchestrator")
        sys.exit(1)

    if RICH_AVAILABLE:
        console.print("[bold green]Dante Agent Core started.[/bold green]")
        console.print("Type [bold]/help[/bold] for commands, or enter a query.")
    else:
        print("Dante Agent Core started.")
        print("Type /help for commands, or enter a query.")

    while True:
        try:
            user_input = input("\n> ").strip()
            if user_input.startswith(">"):
                user_input = user_input[1:].strip()
            if not user_input:
                continue

            if user_input.startswith('/'):
                handle_command(user_input)
                continue

            with trace_context() as ctx:
                # Удалён вызов orchestrator.reset_context() — теперь контекст не сбрасывается на каждый запрос
                response = orchestrator.process(user_input)

                clean_response = clean_model_output(response)

                if RICH_AVAILABLE:
                    console.print(Panel(clean_response, title="Response", border_style="blue"))
                else:
                    print("Response:")
                    print(clean_response)

        except KeyboardInterrupt:
            print("\nInterrupted. Use /exit to quit.")
            continue
        except Exception as e:
            logger.exception("Unhandled error in main loop")
            print(f"Error: {e}")

if __name__ == "__main__":
    main()