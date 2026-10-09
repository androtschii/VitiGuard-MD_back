from types import SimpleNamespace

import httpx2 as httpx
import pytest

from app.api import deps
from app.core.config import Settings
from app.integrations.open_meteo import OpenMeteoClient
from app.worker.celery_app import celery_app


def test_reset_mailer_sends_task_through_project_celery_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[tuple[str, tuple[str, str]]] = []
    monkeypatch.setattr(
        celery_app, "send_task", lambda name, args: sent.append((name, args))
    )

    deps.get_reset_mailer()("grower@example.md", "https://app/reset?token=t")

    assert sent == [
        ("mail.password_reset", ("grower@example.md", "https://app/reset?token=t"))
    ]


def test_project_celery_app_uses_configured_broker() -> None:
    # Регрессия: письмо уходило в Celery по умолчанию (amqp на localhost)
    assert celery_app.conf.broker_url.startswith("redis://")


async def test_open_meteo_uses_the_shared_http_client() -> None:
    async with httpx.AsyncClient() as http:
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(http_client=http))
        )

        client = deps.get_open_meteo(request, Settings(_env_file=None))  # type: ignore[arg-type]

        assert isinstance(client, OpenMeteoClient)
        assert client._http is http
