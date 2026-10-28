"""Журнал в JSON через structlog (pr-089).

Одна строка JSON на событие: её разбирают Loki, ELK или `jq` без регулярных
выражений. Через тот же конвейер идут и записи стандартного logging (uvicorn,
Celery, httpx), поэтому формат один на весь сервис. Секреты вычищаются до
записи: журнал читают больше людей, чем имеют доступ к базе."""

import logging
import re
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from app.core.config import Settings

MASK = "***"
# Ключи, значение которых никогда не пишется: пароли, токены, cookie, данные
# запуска мини-приложения Telegram (по ним можно войти)
SECRET_KEYS = frozenset(
    {
        "password",
        "new_password",
        "token",
        "access_token",
        "refresh_token",
        "authorization",
        "cookie",
        "set-cookie",
        "secret",
        "secret_key",
        "init_data",
        "code",
    }
)
SECRET_PATTERNS = [
    # Токен бота стоит прямо в адресе запросов к Bot API: httpx пишет адрес
    (re.compile(r"bot\d+:[A-Za-z0-9_-]{20,}"), f"bot{MASK}"),
    # Секреты в параметрах адреса: сброс пароля, привязка Telegram, вход в WebSocket
    (
        re.compile(r"([?&](?:token|code|init_data|tgWebAppData)=)[^&\s\"']+"),
        rf"\1{MASK}",
    ),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._-]+", re.IGNORECASE), rf"\1{MASK}"),
]


def mask_text(text: str) -> str:
    for pattern, replacement in SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _mask(value: Any) -> Any:
    if isinstance(value, str):
        return mask_text(value)
    if isinstance(value, MutableMapping):
        return {
            key: MASK if str(key).lower() in SECRET_KEYS else _mask(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_mask(item) for item in value]
    return value


def redact_secrets(_: Any, __: str, event_dict: EventDict) -> EventDict:
    """Процессор structlog: секреты по ключу и по виду значения."""
    return dict(_mask(event_dict))


def configure_logging(settings: Settings) -> None:
    """Вызывается один раз при старте API и воркера."""
    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.ExtraAdder(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.format_exc_info,
        redact_secrets,
    ]
    # На своей машине — читаемые цветные строки, в остальных окружениях — JSON
    renderer: Processor = (
        structlog.dev.ConsoleRenderer()
        if settings.environment == "local"
        else structlog.processors.JSONRenderer(ensure_ascii=False)
    )
    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
        )
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(settings.log_level)
    # Свои обработчики uvicorn писали бы мимо JSON
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers.clear()
        logging.getLogger(name).propagate = True
    # Журнал запросов пишет наша обёртка (с id запроса и длительностью)
    logging.getLogger("uvicorn.access").disabled = True
    # httpx пишет каждый внешний запрос на INFO — это шум
    logging.getLogger("httpx").setLevel(logging.WARNING)
