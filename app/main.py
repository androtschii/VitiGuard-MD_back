from fastapi import FastAPI

from app import __version__
from app.api.router import api_router
from app.core.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Собирает приложение; в тестах можно передать свои настройки."""
    settings = settings or get_settings()

    application = FastAPI(
        title=settings.app_name,
        version=__version__,
        debug=settings.debug,
    )
    application.state.settings = settings
    application.include_router(api_router, prefix=settings.api_v1_prefix)
    return application


app = create_app()
