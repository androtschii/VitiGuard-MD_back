# Сборка в два этапа (pr-098): uv и его кэш нужны только для установки
# зависимостей, в итоговый образ попадают окружение и код

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Зависимости — отдельным слоем: он пересобирается только при изменении uv.lock
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Колесо rasterio несёт GDAL с собой, но берёт из системы libexpat (разбор XML
# метаданных GeoTIFF), которой нет в slim-образе
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 vitiguard

WORKDIR /app

# Окружение собрано на том же базовом образе: ссылки на интерпретатор в нём
# ведут в тот же /usr/local/bin/python3.12
COPY --from=builder /app/.venv /app/.venv
COPY alembic.ini ./
COPY migrations ./migrations
COPY app ./app
COPY models ./models

USER vitiguard

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
