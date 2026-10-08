import uuid
from collections.abc import Callable

import httpx2 as httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.tokens import TokenType, create_refresh_token, decode_token, hash_token

pytestmark = pytest.mark.integration

REFRESH_URL = "/api/v1/auth/refresh"
LOGOUT_URL = "/api/v1/auth/logout"
RunSql = Callable[[str], list[tuple[object, ...]]]
CREDENTIALS = {"email": "grower@example.md", "password": "secret123"}
SETTINGS = Settings(_env_file=None)


@pytest.fixture
def refresh_token(api_client: TestClient) -> str:
    api_client.post(
        "/api/v1/auth/register", json={**CREDENTIALS, "full_name": "Андрей Бахов"}
    )
    response = api_client.post("/api/v1/auth/login", json=CREDENTIALS)
    token = response.cookies["refresh_token"]
    # Cookie с флагом Secure клиент по http:// сам не отправит; в тестах её
    # передаём явно, чтобы управлять тем, какой токен уходит
    api_client.cookies.clear()
    return token


def post(client: TestClient, url: str, token: str | None) -> httpx.Response:
    headers = {"Cookie": f"refresh_token={token}"} if token else {}
    return client.post(url, headers=headers)


def test_refresh_rotates_tokens(
    api_client: TestClient, run_sql: RunSql, refresh_token: str
) -> None:
    response = post(api_client, REFRESH_URL, refresh_token)

    assert response.status_code == 200
    new_token = response.cookies["refresh_token"]
    assert new_token != refresh_token
    claims = decode_token(response.json()["access_token"], TokenType.ACCESS, SETTINGS)
    assert claims.token_type is TokenType.ACCESS
    rows = run_sql("SELECT token_hash, revoked_at IS NOT NULL FROM refresh_tokens")
    assert set(rows) == {
        (hash_token(refresh_token), True),
        (hash_token(new_token), False),
    }


def test_new_cookie_has_the_same_protection(
    api_client: TestClient, refresh_token: str
) -> None:
    cookie = post(api_client, REFRESH_URL, refresh_token).headers["set-cookie"]

    for flag in ("HttpOnly", "Secure", "SameSite=strict", "Path=/api/v1/auth"):
        assert flag in cookie


def test_reused_token_revokes_every_session(
    api_client: TestClient, run_sql: RunSql, refresh_token: str
) -> None:
    new_token = post(api_client, REFRESH_URL, refresh_token).cookies["refresh_token"]

    reuse = post(api_client, REFRESH_URL, refresh_token)
    after_reuse = post(api_client, REFRESH_URL, new_token)

    assert reuse.status_code == 401
    assert after_reuse.status_code == 401
    assert run_sql("SELECT count(*) FROM refresh_tokens WHERE revoked_at IS NULL") == [
        (0,)
    ]


@pytest.mark.parametrize("token", [None, "not-a-token"])
def test_missing_or_garbage_token_gives_401(
    api_client: TestClient, token: str | None
) -> None:
    response = post(api_client, REFRESH_URL, token)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.usefixtures("refresh_token")
def test_signed_token_unknown_to_database_gives_401(
    api_client: TestClient, run_sql: RunSql
) -> None:
    # Подпись верная, но такой токен сервер не выдавал (или запись удалена)
    ((user_id,),) = run_sql("SELECT id FROM users")
    assert isinstance(user_id, uuid.UUID)
    forged = create_refresh_token(user_id, SETTINGS).token

    assert post(api_client, REFRESH_URL, forged).status_code == 401


def test_access_token_is_not_accepted_as_refresh(api_client: TestClient) -> None:
    api_client.post(
        "/api/v1/auth/register", json={**CREDENTIALS, "full_name": "Андрей Бахов"}
    )
    access = api_client.post("/api/v1/auth/login", json=CREDENTIALS).json()[
        "access_token"
    ]
    api_client.cookies.clear()

    assert post(api_client, REFRESH_URL, access).status_code == 401


def test_disabled_user_cannot_refresh(
    api_client: TestClient, run_sql: RunSql, refresh_token: str
) -> None:
    run_sql("UPDATE users SET is_active = false")

    assert post(api_client, REFRESH_URL, refresh_token).status_code == 401


def test_logout_revokes_token_and_deletes_cookie(
    api_client: TestClient, run_sql: RunSql, refresh_token: str
) -> None:
    response = post(api_client, LOGOUT_URL, refresh_token)

    assert response.status_code == 204
    cookie = response.headers["set-cookie"]
    assert cookie.startswith('refresh_token=""')
    assert "Max-Age=0" in cookie
    assert "Path=/api/v1/auth" in cookie
    assert run_sql("SELECT count(*) FROM refresh_tokens WHERE revoked_at IS NULL") == [
        (0,)
    ]
    assert post(api_client, REFRESH_URL, refresh_token).status_code == 401


@pytest.mark.parametrize("token", [None, "not-a-token"])
def test_logout_without_valid_session_still_succeeds(
    api_client: TestClient, token: str | None
) -> None:
    assert post(api_client, LOGOUT_URL, token).status_code == 204


def test_repeated_logout_is_harmless(
    api_client: TestClient, refresh_token: str
) -> None:
    assert post(api_client, LOGOUT_URL, refresh_token).status_code == 204
    assert post(api_client, LOGOUT_URL, refresh_token).status_code == 204
