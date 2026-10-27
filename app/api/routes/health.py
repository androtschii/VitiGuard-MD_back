"""Проверки состояния для Docker и балансировщика (pr-092).

/healthz — процесс жив и отвечает: ничего внешнего не трогает, чтобы сбой базы
не привёл к бесконечным перезапускам контейнера. /readyz — можно ли слать
запросы: доступны ли PostgreSQL и Redis. Ответы публичные, поэтому без текста
ошибок и адресов."""

import asyncio
from collections.abc import Awaitable

from fastapi import APIRouter, Request, Response, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.integrations.cache import RedisCache
from app.schemas.health import CheckStatus, Liveness, Readiness

# Проверка готовности не должна висеть дольше, чем ждёт балансировщик
CHECK_TIMEOUT_SECONDS = 2.0

router = APIRouter(tags=["service"])


@router.get("/healthz", summary="Процесс жив")
async def healthz() -> Liveness:
    return Liveness()


async def _check(probe: Awaitable[object]) -> CheckStatus:
    try:
        await asyncio.wait_for(probe, CHECK_TIMEOUT_SECONDS)
    except Exception:
        # Любая ошибка — отказ, тайм-аут, неверный пароль — значит «не готов»
        return "error"
    return "ok"


async def _ping_database(engine: AsyncEngine) -> None:
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))


async def _ping_redis(cache: RedisCache) -> None:
    await cache.client.ping()


@router.get(
    "/readyz",
    summary="Готов принимать запросы: база и Redis доступны",
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": Readiness}},
)
async def readyz(request: Request, response: Response) -> Readiness:
    database, redis = await asyncio.gather(
        _check(_ping_database(request.app.state.engine)),
        _check(_ping_redis(request.app.state.cache)),
    )
    checks: dict[str, CheckStatus] = {"database": database, "redis": redis}
    ready = all(value == "ok" for value in checks.values())
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return Readiness(status="ready" if ready else "not_ready", checks=checks)
