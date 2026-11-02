#!/usr/bin/env bash
# Ежедневная копия базы (pr-100); её запускает vitiguard-backup.timer.
# Сжатый дамп pg_dump в /opt/vitiguard/backups, хранятся последние 14.
# Восстановление: gunzip -c <файл> | docker compose ... exec -T db \
#   psql -U vitiguard -d vitiguard
set -euo pipefail

ROOT=/opt/vitiguard
KEEP=14
mkdir -p "$ROOT/backups"
cd "$ROOT/VitiGuard-MD_back"
file="$ROOT/backups/vitiguard-$(date -u +%Y-%m-%d).sql.gz"
docker compose -f compose.yaml -f compose.prod.yaml exec -T db \
    pg_dump -U vitiguard -d vitiguard --no-owner | gzip > "$file.part"
mv "$file.part" "$file"
ls -1t "$ROOT"/backups/vitiguard-*.sql.gz | tail -n +$((KEEP + 1)) | xargs -r rm --
