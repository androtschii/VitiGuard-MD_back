from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings


def create_engine(settings: Settings) -> AsyncEngine:
    """Движок создаётся лениво: соединение с базой открывается при первом запросе."""
    return create_async_engine(
        str(settings.database_url),
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_timeout=settings.database_pool_timeout,
        # Перед выдачей из пула соединение проверяется: после перезапуска базы или
        # обрыва сети запрос не падает на «мёртвом» соединении, а получает новое
        pool_pre_ping=True,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    # После commit атрибуты объектов не сбрасываются: в async-коде их повторное
    # чтение потребовало бы нового запроса и упало бы с MissingGreenlet
    return async_sessionmaker(engine, expire_on_commit=False)
