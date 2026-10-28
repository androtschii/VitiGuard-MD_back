"""Запись о каждом HTTP-запросе и его id (pr-089).

Id берётся из заголовка X-Request-ID (его ставит балансировщик) или создаётся
заново, попадает во все записи журнала этого запроса и возвращается клиентом
в ответе: по нему ошибку из жалобы пользователя находят в журнале."""

import re
import time
import uuid

import structlog
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = "X-Request-ID"
# Чужой id принимается, только если он похож на id: иначе в журнал можно было
# бы записать что угодно
VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
# Docker опрашивает их каждые несколько секунд — в журнале это шум
QUIET_PATHS = frozenset({"/healthz", "/readyz"})

logger = structlog.get_logger("app.request")


class RequestLogMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope["headers"]).get(REQUEST_ID_HEADER.lower().encode(), b"")
        request_id = incoming.decode("latin-1")
        if not VALID_REQUEST_ID.match(request_id):
            request_id = uuid.uuid4().hex
        status_code = 500
        started = time.perf_counter()

        async def send_with_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            await self.app(scope, receive, send_with_id)
        finally:
            path = scope["path"]
            # Адрес без параметров: в них бывают токены и личные данные
            log = logger.debug if path in QUIET_PATHS else logger.info
            log(
                "request",
                method=scope["method"],
                path=path,
                status=status_code,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            structlog.contextvars.unbind_contextvars("request_id")
