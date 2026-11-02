# Развёртывание на сервере

VitiGuard MD развёртывается на одной виртуальной машине через Docker Compose: подойдёт любая машина с Ubuntu 24.04, от 2 ГБ памяти и 25 ГБ диска (выбран DigitalOcean Droplet; образы собираются и для amd64, и для ARM).

## Что запускается

| Сервис | Назначение | Снаружи |
|---|---|---|
| `web` | Caddy: собранный сайт, прокси `/api` на API, сертификат Let's Encrypt | порты 80 и 443 |
| `api` | FastAPI, миграции при запуске | нет |
| `worker`, `beat` | фоновые задачи Celery и расписание | нет |
| `db` | PostgreSQL 18 + PostGIS (образ `docker/db/Dockerfile` для ARM и amd64) | нет |
| `redis`, `storage`, `mailpit` | очередь и кэш, хранилище фото, почта | нет |

Сайт и API работают на одном домене: клиент обращается к `/api/v1` без CORS. Telegram-бот получает обновления вебхуком на `https://<домен>/api/v1/telegram/webhook`.

## Первая установка

1. Создать машину: Ubuntu 24.04, публичный IPv4; если у провайдера включён сетевой фильтр (Firewall, Security List) — разрешить входящие TCP 22, 80 и 443.
2. Адрес сайта без покупки домена: `<IP через дефисы>.sslip.io`, например `130-61-12-34.sslip.io`. Такое имя указывает на IP машины, и Let's Encrypt выдаёт на него сертификат.
3. Скопировать на сервер скрипт и модели (весов нет в git):

   ```bash
   ssh root@<IP> "sudo mkdir -p /opt/vitiguard && sudo chown root /opt/vitiguard"
   ssh root@<IP> "git clone https://github.com/androtschii/VitiGuard-MD_back.git /opt/vitiguard/VitiGuard-MD_back"
   scp models/*.onnx models/*.gate.json root@<IP>:/opt/vitiguard/VitiGuard-MD_back/models/
   ```

4. Запустить установку на сервере:

   ```bash
   cd /opt/vitiguard/VitiGuard-MD_back
   SITE_ADDRESS=130-61-12-34.sslip.io BOT_TOKEN=<токен> BOT_USERNAME=VitiGuard_bot bash deploy/setup.sh
   ```

   Скрипт добавляет подкачку на небольших машинах, ставит Docker, открывает 80 и 443 в iptables, клонирует сайт, создаёт `.env` со случайными секретами, собирает и запускает сервисы, регистрирует вебхук бота и включает автообновление.

## Обновление

`vitiguard-update.timer` раз в 10 минут запускает `deploy/update.sh`: если в `main` любого из репозиториев есть новое, код обновляется, образы пересобираются и сервисы перезапускаются; вебхук и меню бота настраиваются заново. Ключи доступа к серверу нигде, кроме самого сервера, не хранятся.

## База данных и копии

PostgreSQL с PostGIS работает на той же машине в контейнере `db`; данные — в томе Docker `pgdata` и переживают перезапуски и обновления. `vitiguard-backup.timer` каждую ночь в 03:30 UTC делает сжатый дамп в `/opt/vitiguard/backups` и хранит последние 14. Восстановление:

```bash
gunzip -c /opt/vitiguard/backups/vitiguard-<дата>.sql.gz | docker compose -f compose.yaml -f compose.prod.yaml exec -T db psql -U vitiguard -d vitiguard
```

Копии лежат на том же диске: от ошибки или неудачной миграции они спасают, от потери машины — нет. Для этого их стоит иногда скачивать к себе (`scp root@<IP>:/opt/vitiguard/backups/*.gz .`).

## Секреты

Только в `/opt/vitiguard/VitiGuard-MD_back/.env` на сервере (права 600): ключ подписи токенов, пароли базы и хранилища, токен метрик, токен и секрет вебхука бота. В git их нет.

## Проверка

```bash
curl https://<домен>/api/v1/info
docker compose -f compose.yaml -f compose.prod.yaml ps
docker compose -f compose.yaml -f compose.prod.yaml exec api python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/readyz').read())"
```
