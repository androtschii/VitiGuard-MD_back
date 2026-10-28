"""Ограничение частоты запросов (pr-090).

Фиксированное окно в Redis: INCR счётчика окна и срок жизни, равный окну, —
одним MULTI. Свой код, а не SlowAPI: тот хранит счётчики через синхронный
клиент Redis, а он остановил бы весь асинхронный сервер на время запроса.

Счётчики — защита от перебора паролей и от исчерпания модели, а не учёт: если
Redis недоступен, запросы пропускаются (как и с кэшем), сервис не падает."""

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import structlog
from fastapi import Request
from redis.exceptions import RedisError

from app.api.deps import CurrentUserDep, SettingsDep
from app.core.errors import TooManyRequestsError

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Limit:
    name: str
    times: int
    seconds: int


# Перебор паролей: 10 попыток в минуту с адреса достаточно человеку,
# опечатавшемуся несколько раз, и мало для подбора
LOGIN = Limit("login", 10, 60)
TELEGRAM_LOGIN = Limit("telegram-login", 10, 60)
# Массовая регистрация и рассылка писем сброса на чужие адреса
REGISTER = Limit("register", 5, 600)
PASSWORD_RESET = Limit("password-reset", 5, 3600)
# Модель — самая дорогая операция; 30 фото в минуту — с запасом для работы в поле
DIAGNOSE_LIMIT = Limit("diagnose", 30, 60)


async def hit(request: Request, limit: Limit, subject: str) -> None:
    """Засчитать запрос; при превышении — 429 с Retry-After."""
    now = int(time.time())
    window = now // limit.seconds
    key = f"rl:{limit.name}:{subject}:{window}"
    try:
        async with request.app.state.cache.client.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, limit.seconds)
            count, _ = await pipe.execute()
    except (RedisError, OSError):
        logger.warning("rate_limit_unavailable", limit=limit.name)
        return
    if count > limit.times:
        retry_after = limit.seconds - now % limit.seconds
        raise TooManyRequestsError(
            "Слишком много запросов. Повторите позже", retry_after
        )


def client_address(request: Request) -> str:
    # За прокси адрес клиента подставляет uvicorn --proxy-headers
    return request.client.host if request.client else "unknown"


def per_address(limit: Limit) -> Callable[[Request, SettingsDep], Awaitable[None]]:
    async def dependency(request: Request, settings: SettingsDep) -> None:
        if settings.rate_limits_on:
            await hit(request, limit, client_address(request))

    return dependency


def per_user(
    limit: Limit,
) -> Callable[[Request, SettingsDep, CurrentUserDep], Awaitable[None]]:
    # Для вошедших — по пользователю: в поле многие сидят за одним адресом
    # мобильного оператора
    async def dependency(
        request: Request, settings: SettingsDep, user: CurrentUserDep
    ) -> None:
        if settings.rate_limits_on:
            await hit(request, limit, str(user.id))

    return dependency
