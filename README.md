# VitiGuard MD — backend

[![CI](https://github.com/androtschii/VitiGuard-MD_back/actions/workflows/ci.yml/badge.svg)](https://github.com/androtschii/VitiGuard-MD_back/actions/workflows/ci.yml)

Бэкенд системы мониторинга здоровья виноградников Республики Молдова: диагностика болезней по фото листьев, спутниковый мониторинг Sentinel-2 (NDVI/NDRE) и прогноз риска грибковых заболеваний по метеоданным.

Фронтенд: [VitiGuard-MD_front](https://github.com/androtschii/VitiGuard-MD_front)

## Стек

Python 3.12, FastAPI, Pydantic v2, uv. Дальше по плану: PostgreSQL + PostGIS, SQLAlchemy, Alembic, Celery + Redis, Docker, PyTorch, ONNX Runtime.

## Запуск

Нужен [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
cp .env.example .env   # при необходимости
uv run uvicorn app.main:app --reload
```

API: http://127.0.0.1:8000, документация: http://127.0.0.1:8000/docs

## Запуск в Docker

```bash
docker compose up --build
```

Поднимаются API, воркер Celery, PostgreSQL 18 с PostGIS 3.6 и Redis 8. Миграции применяются автоматически при старте API. Порты и пароли задаются в `.env` (пример — `.env.example`), без него используются значения по умолчанию.

## Фоновые задачи (Celery)

Тяжёлая работа (нейросеть, растры, прогнозы) выполняется воркером, а не в запросе API. Redis хранит и очередь, и результаты. Воркер запускается вместе с остальными сервисами; проверить всю цепочку «API → Redis → воркер → API»:

```bash
docker compose exec api python -c "from app.worker.celery_app import celery_app; print(celery_app.send_task('system.ping').get(timeout=15))"
```

Воркер без Docker: `uv run celery -A app.worker.celery_app worker --loglevel=info`. В Windows добавьте `--pool=solo`: многопроцессный пул Celery там не работает. Новые задачи кладутся в `app/tasks/`, а их модули — в `TASK_MODULES` (`app/worker/celery_app.py`).

## Миграции БД

```bash
docker compose up -d db
uv run alembic upgrade head
uv run alembic revision --autogenerate -m "описание изменений"
```

## Тесты

Интеграционные тесты (`-m integration`) поднимают PostgreSQL с PostGIS и Redis через Testcontainers, поэтому нужен запущенный Docker. Без него они пропускаются.

```bash
uv run pytest
uv run pytest --cov   # с отчётом о покрытии, минимум 85 %
```

## Проверка кода

```bash
uv run ruff check .
uv run black --check .
uv run mypy
```

Автоисправление: `uv run ruff check --fix .` и `uv run black .`

## Структура

```
app/
  main.py          фабрика приложения
  core/config.py   настройки (переменные окружения VITIGUARD_*)
  api/             роутеры и зависимости
  schemas/         Pydantic-схемы
  worker/          приложение Celery
  tasks/           фоновые задачи
tests/
docs/
  roadmap.md       план работ
  decisions.md     принятые технические решения
```
