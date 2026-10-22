import hashlib
import hmac
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlencode

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.errors import UnauthorizedError
from app.services.telegram_auth import telegram_user_id

TOKEN = "123456:secret-bot-token"
NOW = datetime(2026, 10, 22, 12, tzinfo=UTC)
RunSql = Callable[[str], list[tuple[object, ...]]]


def init_data(
    user_id: int = 42,
    signed_at: datetime = NOW,
    token: str = TOKEN,
    **extra: str,
) -> str:
    """Данные запуска, подписанные так же, как их подписывает Telegram."""
    fields = {
        "query_id": "AAHdF6IQAAAAAN0XohDhrOrc",
        "user": json.dumps({"id": user_id, "first_name": "Ион"}, ensure_ascii=False),
        "auth_date": str(int(signed_at.timestamp())),
        **extra,
    }
    check = "\n".join(f"{key}={fields[key]}" for key in sorted(fields))
    secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    fields["hash"] = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    return urlencode(fields)


def test_signed_data_gives_telegram_user() -> None:
    assert telegram_user_id(init_data(), TOKEN, NOW) == 42
    # Новые поля Telegram (например, подпись для третьих сторон) входят в проверку
    assert telegram_user_id(init_data(signature="abc"), TOKEN, NOW) == 42


@pytest.mark.parametrize(
    "data",
    [
        init_data(token="other:token"),
        init_data().replace("42", "43"),
        init_data() + "&extra=1",
        "user=%7B%7D",
        "",
    ],
    ids=["other-bot", "tampered-user", "added-field", "no-hash", "empty"],
)
def test_forged_data_is_rejected(data: str) -> None:
    with pytest.raises(UnauthorizedError, match="не прошли проверку"):
        telegram_user_id(data, TOKEN, NOW)


def test_signed_data_without_user_is_rejected() -> None:
    # Подпись верна, но поля user нет: войти некем
    check = f"auth_date={int(NOW.timestamp())}"
    secret = hmac.new(b"WebAppData", TOKEN.encode(), hashlib.sha256).digest()
    signed = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()

    with pytest.raises(UnauthorizedError, match="не прошли проверку"):
        telegram_user_id(f"{check}&hash={signed}", TOKEN, NOW)


def test_stale_data_is_rejected() -> None:
    with pytest.raises(UnauthorizedError, match="устарели"):
        telegram_user_id(init_data(signed_at=NOW - timedelta(days=2)), TOKEN, NOW)


def enable_bot(client: TestClient) -> None:
    app: Any = client.app
    app.state.settings = app.state.settings.model_copy(
        update={"telegram_bot_token": SecretStr(TOKEN)}
    )


def register(client: TestClient, email: str = "grower@example.md") -> None:
    client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "secret123", "full_name": "Ион"},
    )


@pytest.mark.integration
def test_linked_user_signs_in_without_password(
    api_client: TestClient, run_sql: RunSql
) -> None:
    enable_bot(api_client)
    register(api_client)
    run_sql("UPDATE users SET telegram_chat_id = 42")
    data = init_data(signed_at=datetime.now(UTC))

    response = api_client.post("/api/v1/auth/telegram", json={"init_data": data})

    assert response.status_code == 200
    assert "refresh_token" in response.cookies
    token = response.json()["access_token"]
    me = api_client.get(
        "/api/v1/users/me", headers={"Authorization": f"Bearer {token}"}
    ).json()
    assert me["email"] == "grower@example.md"

    run_sql("UPDATE users SET is_active = false")
    blocked = api_client.post("/api/v1/auth/telegram", json={"init_data": data})
    assert blocked.status_code == 401


@pytest.mark.integration
def test_unlinked_or_forged_telegram_cannot_sign_in(api_client: TestClient) -> None:
    enable_bot(api_client)
    register(api_client)
    url = "/api/v1/auth/telegram"

    unlinked = api_client.post(
        url, json={"init_data": init_data(signed_at=datetime.now(UTC))}
    )
    forged = api_client.post(
        url, json={"init_data": init_data(signed_at=datetime.now(UTC), token="x:y")}
    )

    assert unlinked.status_code == 401
    assert unlinked.json() == {
        "detail": "Этот Telegram не привязан к учётной записи VitiGuard MD"
    }
    assert forged.status_code == 401
    assert api_client.post(url, json={"init_data": ""}).status_code == 422


@pytest.mark.integration
def test_without_bot_telegram_sign_in_is_off(api_client: TestClient) -> None:
    response = api_client.post("/api/v1/auth/telegram", json={"init_data": init_data()})

    assert response.status_code == 503
