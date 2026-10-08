from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.db.session import create_engine, create_session_factory


def create_app(settings: Settings | None = None) -> FastAPI:
    """Собирает приложение; в тестах можно передать свои настройки."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        engine = create_engine(settings)
        application.state.engine = engine
        application.state.session_factory = create_session_factory(engine)
        yield
        # Закрывает все соединения пула, чтобы база не держала «зависшие» сессии
        await engine.dispose()

    application = FastAPI(
        title=settings.app_name,
        version=__version__,
        debug=settings.debug,
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.include_router(api_router, prefix=settings.api_v1_prefix)
    return application


app = create_app()
