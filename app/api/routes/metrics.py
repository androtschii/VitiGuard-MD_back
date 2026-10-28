"""Ручка /metrics для Prometheus (pr-088). Вне /api/v1 и вне документации API:
её опрашивает Prometheus, а не клиент."""

import hmac
from typing import Annotated

from fastapi import APIRouter, Header, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.deps import SettingsDep
from app.core.errors import UnauthorizedError

router = APIRouter(include_in_schema=False)


@router.get("/metrics")
async def metrics(
    request: Request,
    settings: SettingsDep,
    authorization: Annotated[str | None, Header()] = None,
) -> Response:
    # Метрики раскрывают маршруты и нагрузку: на сервере их закрывают токеном,
    # который знает только Prometheus. Без токена — открыты (своя машина)
    if settings.metrics_token is not None:
        expected = f"Bearer {settings.metrics_token.get_secret_value()}"
        # Сравнение за постоянное время: по скорости ответа токен не подобрать
        if not hmac.compare_digest((authorization or "").encode(), expected.encode()):
            raise UnauthorizedError("Нужен токен метрик")
    body = generate_latest(request.app.state.metrics.registry)
    return Response(body, media_type=CONTENT_TYPE_LATEST)
