#!/bin/bash

# Настройки
PROJECT_DIR="/Users/dante14594/Library/Mobile Documents/iCloud~md~obsidian/Documents/agent-core"
BACKUP_DIR="$PROJECT_DIR/backups"
GIT_BRANCH="main"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$BACKUP_DIR/backup.log"

mkdir -p "$BACKUP_DIR"
cd "$PROJECT_DIR"

echo "[$TIMESTAMP] Запуск бэкапа..." >> "$LOG_FILE"

# 1. Git backup (если есть изменения)
if [ -d ".git" ]; then
    git add .
    if ! git diff --cached --quiet; then
        git commit -m "Auto backup $TIMESTAMP"
        git push origin $GIT_BRANCH
        echo "[$TIMESTAMP] Git коммит и пуш выполнены" >> "$LOG_FILE"
    else
        echo "[$TIMESTAMP] Нет изменений для git" >> "$LOG_FILE"
    fi
else
    echo "[$TIMESTAMP] Git репозиторий не найден" >> "$LOG_FILE"
fi

# 2. Локальный архив (исключая venv, .git, logs, backups)
ARCHIVE_NAME="agent-core-backup-$TIMESTAMP.tar.gz"
tar -czf "$BACKUP_DIR/$ARCHIVE_NAME" \
    --exclude=venv \
    --exclude=.git \
    --exclude=logs \
    --exclude=backups \
    --exclude=__pycache__ \
    --exclude=.pytest_cache \
    --exclude=cache_dump.json \
    --exclude=.cli_history \
    .

echo "[$TIMESTAMP] Создан локальный архив: $BACKUP_DIR/$ARCHIVE_NAME" >> "$LOG_FILE"

# 3. Удаляем архивы старше 30 дней
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete
echo "[$TIMESTAMP] Старые архивы (>30 дней) удалены" >> "$LOG_FILE"

echo "[$TIMESTAMP] Бэкап завершён" >> "$LOG_FILE"
