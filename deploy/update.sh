#!/usr/bin/env bash
# Обновление сервера из main (pr-100); его запускает vitiguard-update.timer.
# Если в обоих репозиториях нет нового — ничего не пересобирается
set -euo pipefail

ROOT=/opt/vitiguard
changed=0
for repo in VitiGuard-MD_back VitiGuard-MD_front; do
    cd "$ROOT/$repo"
    git fetch --quiet origin main
    if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
        git reset --quiet --hard origin/main
        changed=1
    fi
done
[ "$changed" = 1 ] || exit 0

cd "$ROOT/VitiGuard-MD_back"
docker compose -f compose.yaml -f compose.prod.yaml up -d --build --remove-orphans
# Меню бота и вебхук — на случай, если поменялись команды
if grep -q '^VITIGUARD_TELEGRAM_BOT_TOKEN=.\+' .env; then
    docker compose -f compose.yaml -f compose.prod.yaml run --rm api \
        python -m app.cli.telegram_webhook
fi
docker image prune -f >/dev/null
