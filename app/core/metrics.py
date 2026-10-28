"""Метрики Prometheus (pr-088).

Счётчик запросов и гистограмма длительности по методу, шаблону маршрута и
статусу. Шаблон (`/vineyards/{vineyard_id}`), а не адрес: иначе каждый id
участка давал бы свой ряд, и Prometheus захлебнулся бы числом рядов. Запросы
мимо маршрутов (404 на произвольный адрес) сводятся в один ряд по той же
причине."""

import time

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    PlatformCollector,
    ProcessCollector,
)
from starlette.types import ASGIApp, Message, Receive, Scope, Send

UNMATCHED = "unmatched"
# Сам опрос метрик и проверки состояния — служебные, их в метриках API не считаем
SKIPPED_PATHS = frozenset({"/metrics", "/healthz", "/readyz"})


class Metrics:
    """Свой реестр на приложение: в тестах каждое приложение считает отдельно,
    без ошибки «метрика уже зарегистрирована»."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        # Память, процессор и открытые файлы процесса (в Linux-контейнере)
        ProcessCollector(registry=self.registry)
        PlatformCollector(registry=self.registry)
        self.requests = Counter(
            "http_requests_total",
            "HTTP-запросы к API",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.duration = Histogram(
            "http_request_duration_seconds",
            "Длительность HTTP-запроса",
            ["method", "route"],
            # От быстрых ответов из кэша до диагностики по фото и PDF-отчёта
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
            registry=self.registry,
        )


class MetricsMiddleware:
    def __init__(self, app: ASGIApp, *, metrics: Metrics, api_prefix: str) -> None:
        self.app = app
        self.metrics = metrics
        self.api_prefix = api_prefix

    def route_template(self, scope: Scope) -> str:
        route = getattr(scope.get("route"), "path", None)
        if route is None:
            return UNMATCHED
        # У маршрута вложенного роутера путь без префикса подключения (/info):
        # возвращаем полный шаблон, как его видит клиент
        if scope["path"].startswith(self.api_prefix) and not route.startswith(
            self.api_prefix
        ):
            return f"{self.api_prefix}{route}"
        return str(route)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in SKIPPED_PATHS:
            await self.app(scope, receive, send)
            return
        status_code = 500
        started = time.perf_counter()

        async def send_with_status(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_with_status)
        finally:
            route = self.route_template(scope)
            method = scope["method"]
            self.metrics.requests.labels(method, route, str(status_code)).inc()
            self.metrics.duration.labels(method, route).observe(
                time.perf_counter() - started
            )
