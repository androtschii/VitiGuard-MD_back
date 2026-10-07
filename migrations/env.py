import asyncio
from logging.config import fileConfig

from alembic import context
from geoalchemy2 import alembic_helpers
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

import app.models  # noqa: F401  # регистрирует модели в Base.metadata
from app.core.config import get_settings
from app.db.base import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def get_url() -> str:
    # URL, заданный явно (например, в тестах), важнее настроек приложения
    return config.get_main_option("sqlalchemy.url") or str(get_settings().database_url)


def run_migrations_offline() -> None:
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=alembic_helpers.include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    # Помощники GeoAlchemy2 пропускают служебные таблицы PostGIS (spatial_ref_sys),
    # а writer заменяет create_table на create_geospatial_table, чтобы GIST-индекс
    # создавался в миграции один раз, а не ещё и событием GeoAlchemy2
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_object=alembic_helpers.include_object,
        render_item=alembic_helpers.render_item,
        process_revision_directives=alembic_helpers.writer,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    # Образ postgis/postgis добавляет в search_path схемы topology и tiger,
    # и без этого ограничения автогенерация видит их таблицы
    engine = create_async_engine(
        get_url(),
        poolclass=pool.NullPool,
        connect_args={"server_settings": {"search_path": "public"}},
    )
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
