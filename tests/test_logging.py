import json
import logging

import pytest
import structlog
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.logging import MASK, configure_logging, mask_text, redact_secrets
from app.main import create_app
from app.worker.celery_app import _configure_worker_logging

BOT_TOKEN = "8934350711:AAE0xrjJgBiEDXH0EKSc9QUDhhKUHL0Hxyz"


def json_lines(text: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in text.splitlines() if line.startswith("{")]


@pytest.fixture
def test_settings() -> Settings:
    return Settings(_env_file=None, environment="test")


def test_mask_text_hides_tokens_in_urls_and_headers() -> None:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    assert mask_text(url) == f"https://api.telegram.org/bot{MASK}/sendMessage"
    assert mask_text("/reset?token=abc123&lang=ru") == f"/reset?token={MASK}&lang=ru"
    assert mask_text("/ws/alerts?token=eyJ.a.b") == f"/ws/alerts?token={MASK}"
    assert mask_text("Bearer eyJhbGciOi.x.y") == f"Bearer {MASK}"
    assert mask_text("обычный текст") == "обычный текст"


def test_redact_secrets_by_key_at_any_depth() -> None:
    event = redact_secrets(
        None,
        "info",
        {
            "event": "login",
            "password": "hunter2",
            "user": {"email": "a@b.md", "Refresh_Token": "r"},
            "items": [{"init_data": "x"}, "ok"],
            "count": 3,
        },
    )

    assert event == {
        "event": "login",
        "password": MASK,
        "user": {"email": "a@b.md", "Refresh_Token": MASK},
        "items": [{"init_data": MASK}, "ok"],
        "count": 3,
    }


def test_json_lines_for_structlog_and_stdlib(
    test_settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging(test_settings)

    structlog.get_logger("app.test").info("signed_in", password="p", user_id=7)
    logging.getLogger("httpx").warning(
        "HTTP Request: POST https://api.telegram.org/bot%s/getUpdates", BOT_TOKEN
    )

    records = json_lines(capsys.readouterr().out)
    signed_in = next(r for r in records if r["event"] == "signed_in")
    assert signed_in["password"] == MASK
    assert signed_in["user_id"] == 7
    assert signed_in["level"] == "info"
    assert signed_in["logger"] == "app.test"
    assert "timestamp" in signed_in
    httpx_record = next(r for r in records if r["logger"] == "httpx")
    assert BOT_TOKEN not in str(httpx_record["event"])
    assert f"bot{MASK}" in str(httpx_record["event"])


def test_httpx_info_is_silenced(test_settings: Settings) -> None:
    configure_logging(test_settings)

    assert not logging.getLogger("httpx").isEnabledFor(logging.INFO)


def test_local_environment_writes_readable_lines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(Settings(_env_file=None, environment="local"))

    structlog.get_logger("app.test").info("started")

    out = capsys.readouterr().out
    assert "started" in out
    assert json_lines(out) == []


def test_worker_uses_same_logging(capsys: pytest.CaptureFixture[str]) -> None:
    _configure_worker_logging()

    assert logging.getLogger().handlers
    assert logging.getLogger("uvicorn.access").disabled


def test_request_is_logged_with_request_id(
    test_settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    with TestClient(create_app(test_settings)) as client:
        capsys.readouterr()
        response = client.get("/api/v1/info?token=secret-value")

    request_id = response.headers["X-Request-ID"]
    assert len(request_id) == 32
    record = next(
        r for r in json_lines(capsys.readouterr().out) if r["event"] == "request"
    )
    assert record["request_id"] == request_id
    assert record["method"] == "GET"
    # Адрес — без параметров: в них бывают токены
    assert record["path"] == "/api/v1/info"
    assert record["status"] == 200
    assert isinstance(record["duration_ms"], float)


def test_incoming_request_id_is_kept_only_if_valid(test_settings: Settings) -> None:
    with TestClient(create_app(test_settings)) as client:
        kept = client.get("/api/v1/info", headers={"X-Request-ID": "lb-1234abcd"})
        replaced = client.get("/api/v1/info", headers={"X-Request-ID": "bad id\n"})

    assert kept.headers["X-Request-ID"] == "lb-1234abcd"
    assert replaced.headers["X-Request-ID"] != "bad id\n"
    assert len(replaced.headers["X-Request-ID"]) == 32


def test_health_checks_are_not_logged_at_info(
    test_settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    with TestClient(create_app(test_settings)) as client:
        capsys.readouterr()
        client.get("/healthz")

    assert not any(r["event"] == "request" for r in json_lines(capsys.readouterr().out))
