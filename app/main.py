from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx2 as httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.router import api_router
from app.api.routes import health
from app.core.config import Settings, get_settings
from app.core.errors import register_error_handlers
from app.core.security_headers import SecurityHeadersMiddleware
from app.db.session import create_engine, create_session_factory
from app.integrations.cache import RedisCache
from app.integrations.notifier import RedisNotifier


def create_app(settings: Settings | None = None) -> FastAPI:
    """Собирает приложение; в тестах можно передать свои настройки."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        engine = create_engine(settings)
        application.state.engine = engine
        application.state.session_factory = create_session_factory(engine)
        # Один HTTP-клиент на приложение: соединения с внешними сервисами
        # переиспользуются, а не открываются заново на каждый запрос
        application.state.http_client = httpx.AsyncClient(
            timeout=settings.external_http_timeout,
            transport=httpx.AsyncHTTPTransport(retries=2),
            headers={"User-Agent": f"vitiguard-md/{__version__}"},
        )
        cache = RedisCache.from_settings(settings)
        application.state.cache = cache
        notifier = RedisNotifier.from_settings(settings)
        application.state.notifier = notifier
        yield
        await notifier.close()
        await cache.close()
        await application.state.http_client.aclose()
        # Закрывает все соединения пула, чтобы база не держала «зависшие» сессии
        await engine.dispose()

    application = FastAPI(
        title=settings.app_name,
        version=__version__,
        debug=settings.debug,
        lifespan=lifespan,
    )
    application.state.settings = settings
    register_error_handlers(application)
    if settings.cors_origins:
        # Только перечисленные адреса и никогда не «*»: с credentials браузер
        # «*» не примет, а без credentials не дошла бы refresh-cookie
        application.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
            # Из него клиент берёт имя скачанного PDF или XLSX
            expose_headers=["Content-Disposition"],
        )
    # Добавлен последним — значит внешний: заголовки получают все ответы,
    # и preflight-ответы CORS тоже
    application.add_middleware(
        SecurityHeadersMiddleware, hsts=settings.environment == "production"
    )
    # Проверки состояния — в корне, вне версии API: их адрес ждут Docker и балансировщик
    application.include_router(health.router)
    application.include_router(api_router, prefix=settings.api_v1_prefix)
    return application


app = create_app()
