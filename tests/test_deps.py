from types import SimpleNamespace

import httpx2 as httpx
import pytest

from app.api import deps
from app.core.config import Settings
from app.integrations.open_meteo import OpenMeteoClient
from app.tasks import mail


def test_reset_mailer_queues_celery_task(monkeypatch: pytest.MonkeyPatch) -> None:
    queued: list[tuple[str, str]] = []
    monkeypatch.setattr(
        mail.send_password_reset_email,
        "delay",
        lambda to, url: queued.append((to, url)),
    )

    deps.get_reset_mailer()("grower@example.md", "https://app/reset?token=t")

    assert queued == [("grower@example.md", "https://app/reset?token=t")]


async def test_open_meteo_uses_the_shared_http_client() -> None:
    async with httpx.AsyncClient() as http:
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(http_client=http))
        )

        client = deps.get_open_meteo(request, Settings(_env_file=None))  # type: ignore[arg-type]

        assert isinstance(client, OpenMeteoClient)
        assert client._http is http
