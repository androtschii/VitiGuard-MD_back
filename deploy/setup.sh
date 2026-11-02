#!/usr/bin/env bash
# Первая установка VitiGuard MD на чистую Ubuntu 24.04 (pr-100). Запуск на
# сервере от пользователя с sudo:
#
#   SITE_ADDRESS=1-2-3-4.sslip.io BOT_TOKEN=<токен> BOT_USERNAME=VitiGuard_bot \
#     bash setup.sh
#
# Модели (models/*.onnx и *.gate.json) в git не хранятся: их копируют в
# /opt/vitiguard/VitiGuard-MD_back/models до запуска (см. docs/deploy.md).
set -euo pipefail

ROOT=/opt/vitiguard
: "${SITE_ADDRESS:?нужен адрес сайта, например 1-2-3-4.sslip.io}"

# Подкачка на 4 ГБ: на сервере с 1–2 ГБ памяти сборка сайта и образов
# без неё падает от нехватки памяти
if [ ! -f /swapfile ] && [ "$(free -m | awk '/^Mem:/ {print $2}')" -lt 7000 ]; then
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile >/dev/null
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

# Docker из официального репозитория (вместе с compose)
if ! command -v docker >/dev/null; then
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER"
fi

# Порты сайта открыты явно и в начале цепочки: у некоторых провайдеров
# (например, Oracle) iptables по умолчанию отклоняет всё, кроме SSH.
# Правила сохраняются после перезагрузки
for port in 80 443; do
    sudo iptables -C INPUT -p tcp --dport "$port" -j ACCEPT 2>/dev/null \
        || sudo iptables -I INPUT -p tcp --dport "$port" -j ACCEPT
done
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y iptables-persistent >/dev/null
sudo netfilter-persistent save

sudo mkdir -p "$ROOT" && sudo chown "$USER" "$ROOT"
for repo in VitiGuard-MD_back VitiGuard-MD_front; do
    [ -d "$ROOT/$repo" ] || git clone "https://github.com/androtschii/$repo.git" "$ROOT/$repo"
done

cd "$ROOT/VitiGuard-MD_back"
# Секреты создаются один раз и живут только на сервере
if [ ! -f .env ]; then
    secret() { openssl rand -hex 32; }
    cat > .env <<ENV
SITE_ADDRESS=$SITE_ADDRESS
VITIGUARD_SECRET_KEY=$(secret)
POSTGRES_PASSWORD=$(secret)
S3_ACCESS_KEY=vitiguard
S3_SECRET_KEY=$(secret)
VITIGUARD_METRICS_TOKEN=$(secret)
VITIGUARD_TELEGRAM_BOT_TOKEN=${BOT_TOKEN:-}
VITIGUARD_TELEGRAM_BOT_USERNAME=${BOT_USERNAME:-}
VITIGUARD_TELEGRAM_WEBHOOK_SECRET=$(secret)
VITIGUARD_TELEGRAM_WEBAPP_URL=https://$SITE_ADDRESS
ENV
    chmod 600 .env
fi

ls models/diagnosis.onnx >/dev/null || { echo "Нет models/diagnosis.onnx — скопируйте модели"; exit 1; }

sudo docker compose -f compose.yaml -f compose.prod.yaml up -d --build
if grep -q '^VITIGUARD_TELEGRAM_BOT_TOKEN=.\+' .env; then
    sudo docker compose -f compose.yaml -f compose.prod.yaml run --rm api \
        python -m app.cli.telegram_webhook
fi

# Автообновление: раз в 10 минут, если в main что-то новое
# От того же пользователя, что склонировал код: от root git отказался бы
# работать с чужой папкой («dubious ownership»); пользователь в группе docker
sudo tee /etc/systemd/system/vitiguard-update.service >/dev/null <<UNIT
[Unit]
Description=VitiGuard MD: обновление из main
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=$USER
ExecStart=/bin/bash $ROOT/VitiGuard-MD_back/deploy/update.sh
UNIT
sudo cp deploy/vitiguard-update.timer /etc/systemd/system/

# Копия базы раз в сутки: база живёт на этой же машине
sudo tee /etc/systemd/system/vitiguard-backup.service >/dev/null <<UNIT
[Unit]
Description=VitiGuard MD: копия базы
After=docker.service

[Service]
Type=oneshot
User=$USER
ExecStart=/bin/bash $ROOT/VitiGuard-MD_back/deploy/backup.sh
UNIT
sudo cp deploy/vitiguard-backup.timer /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable --now vitiguard-update.timer vitiguard-backup.timer

echo "Готово: https://$SITE_ADDRESS"
