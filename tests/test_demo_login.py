from collections.abc import Callable
from typing import Any

import pytest
from conftest import set_role
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import Settings

pytestmark = pytest.mark.integration

RunSql = Callable[[str], list[tuple[object, ...]]]
URL = "/api/v1/auth/demo"


def enable(client: TestClient, enabled: bool = True) -> None:
    app: Any = client.app
    app.state.settings = app.state.settings.model_copy(
        update={"demo_login_enabled": enabled}
    )


def demo_account(client: TestClient, run_sql: RunSql, email: str, role: str) -> None:
    client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "secret123", "full_name": "Демо"},
    )
    set_role(run_sql, email, role)


def test_demo_login_is_off_by_default(api_client: TestClient) -> None:
    assert api_client.get("/api/v1/info").json()["demo_login"] is False
    assert api_client.post(URL, json={"role": "user"}).status_code == 404


def test_demo_login_opens_session_of_role(
    api_client: TestClient, run_sql: RunSql
) -> None:
    enable(api_client)
    demo_account(api_client, run_sql, "agronom@vitiguard.md", "agronomist")

    assert api_client.get("/api/v1/info").json()["demo_login"] is True
    response = api_client.post(URL, json={"role": "agronomist"})

    assert response.status_code == 200
    # Сессия как при обычном входе: refresh-токен в HttpOnly-cookie
    assert "refresh_token=" in response.headers["set-cookie"]
    assert "httponly" in response.headers["set-cookie"].lower()
    headers = {"Authorization": f"Bearer {response.json()['access_token']}"}
    me = api_client.get("/api/v1/users/me", headers=headers).json()
    assert (me["email"], me["role"]) == ("agronom@vitiguard.md", "agronomist")


def test_demo_login_needs_account_with_that_role(
    api_client: TestClient, run_sql: RunSql
) -> None:
    enable(api_client)
    # Аккаунта нет
    assert api_client.post(URL, json={"role": "admin"}).status_code == 404
    # Аккаунт есть, но роль другая — администратором он не войдёт
    demo_account(api_client, run_sql, "admin@vitiguard.md", "user")
    assert api_client.post(URL, json={"role": "admin"}).status_code == 404
    # Неизвестная роль
    assert api_client.post(URL, json={"role": "root"}).status_code == 422


def test_demo_login_cannot_be_enabled_in_production() -> None:
    with pytest.raises(ValidationError, match="демо-вход"):
        Settings(
            _env_file=None,
            environment="production",
            secret_key="k" * 48,  # type: ignore[arg-type]
            database_url="postgresql+asyncpg://vitiguard:own@db:5432/vitiguard",  # type: ignore[arg-type]
            s3_secret_key="s" * 40,  # type: ignore[arg-type]
            demo_login_enabled=True,
        )
