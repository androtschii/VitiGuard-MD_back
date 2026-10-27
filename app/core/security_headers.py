"""Заголовки безопасности на всех ответах API (pr-095).

Чистая ASGI-обёртка, а не BaseHTTPMiddleware: та буферизует ответ и ломает
потоковые ответы и WebSocket. Заголовок, который ручка поставила сама, не
перезаписывается."""

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

BASE_HEADERS = {
    # Браузер не угадывает тип по содержимому: JSON не исполнится как скрипт
    "X-Content-Type-Options": "nosniff",
    # API не встраивается во фреймы чужих сайтов (кликджекинг)
    "X-Frame-Options": "DENY",
    # Полный адрес с параметрами не уходит на другие сайты
    "Referrer-Policy": "strict-origin-when-cross-origin",
    # Камера и геолокация нужны клиенту для снимков листа, микрофон — нет
    "Permissions-Policy": "camera=(self), geolocation=(self), microphone=()",
}
# Только в production: на машине разработчика HTTPS нет, а HSTS браузер
# запоминает на год и перестал бы открывать http://127.0.0.1
HSTS = ("Strict-Transport-Security", "max-age=31536000; includeSubDomains")


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp, *, hsts: bool) -> None:
        self.app = app
        self.headers = dict(BASE_HEADERS)
        if hsts:
            self.headers[HSTS[0]] = HSTS[1]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in self.headers.items():
                    headers.setdefault(name, value)
            await send(message)

        await self.app(scope, receive, send_with_headers)
