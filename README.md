# VitiGuard MD — backend

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

## Тесты

```bash
uv run pytest
```

## Структура

```
app/
  main.py          фабрика приложения
  core/config.py   настройки (переменные окружения VITIGUARD_*)
  api/             роутеры и зависимости
  schemas/         Pydantic-схемы
tests/
docs/
  roadmap.md       план работ
  decisions.md     принятые технические решения
```
