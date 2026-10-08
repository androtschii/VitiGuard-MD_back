from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from pwdlib.hashers.argon2 import Argon2Hasher

from app.core.config import Settings
from app.core.tokens import TokenType, decode_token, hash_token
from app.models.user import UserRole

pytestmark = pytest.mark.integration

URL = "/api/v1/auth/login"
RunSql = Callable[[str], list[tuple[object, ...]]]
EMAIL = "grower@example.md"
PASSWORD = "secret123"


@pytest.fixture
def registered(api_client: TestClient) -> None:
    response = api_client.post(
        "/api/v1/auth/register",
        json={
            "email": EMAIL,
            "password": PASSWORD,
            "full_name": "Андрей Бахов",
            "role": "agronomist",
        },
    )
    assert response.status_code == 201


@pytest.mark.usefixtures("registered")
def test_login_returns_access_token(api_client: TestClient, run_sql: RunSql) -> None:
    response = api_client.post(URL, json={"email": EMAIL, "password": PASSWORD})

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    claims = decode_token(
        body["access_token"], TokenType.ACCESS, Settings(_env_file=None)
    )
    ((user_id,),) = run_sql("SELECT id FROM users")
    assert claims.subject == user_id
    assert claims.role is UserRole.AGRONOMIST


@pytest.mark.usefixtures("registered")
def test_refresh_token_comes_only_in_protected_cookie(api_client: TestClient) -> None:
    response = api_client.post(URL, json={"email": EMAIL, "password": PASSWORD})

    assert "refresh_token" not in response.json()
    cookie = response.headers["set-cookie"]
    assert cookie.startswith("refresh_token=")
    assert "HttpOnly" in cookie
    assert "Secure" in cookie
    assert "SameSite=strict" in cookie
    assert "Path=/api/v1/auth" in cookie
    assert "Max-Age=2592000" in cookie


@pytest.mark.usefixtures("registered")
def test_only_refresh_token_hash_is_stored(
    api_client: TestClient, run_sql: RunSql
) -> None:
    response = api_client.post(URL, json={"email": EMAIL, "password": PASSWORD})
    refresh = response.cookies["refresh_token"]

    rows = run_sql("SELECT token_hash, revoked_at FROM refresh_tokens")

    assert rows == [(hash_token(refresh), None)]


@pytest.mark.usefixtures("registered")
def test_email_is_case_insensitive(api_client: TestClient) -> None:
    response = api_client.post(
        URL, json={"email": "Grower@Example.MD", "password": PASSWORD}
    )

    assert response.status_code == 200


@pytest.mark.usefixtures("registered")
def test_wrong_password_and_unknown_email_look_the_same(
    api_client: TestClient, run_sql: RunSql
) -> None:
    wrong_password = api_client.post(URL, json={"email": EMAIL, "password": "nope1"})
    unknown_email = api_client.post(
        URL, json={"email": "nobody@example.md", "password": PASSWORD}
    )

    for response in (wrong_password, unknown_email):
        assert response.status_code == 401
        assert response.json() == {"detail": "Неверный email или пароль"}
        assert response.headers["www-authenticate"] == "Bearer"
        assert "set-cookie" not in response.headers
    assert run_sql("SELECT count(*) FROM refresh_tokens") == [(0,)]


@pytest.mark.usefixtures("registered")
def test_disabled_account_is_reported_only_with_correct_password(
    api_client: TestClient, run_sql: RunSql
) -> None:
    run_sql("UPDATE users SET is_active = false")

    correct = api_client.post(URL, json={"email": EMAIL, "password": PASSWORD})
    wrong = api_client.post(URL, json={"email": EMAIL, "password": "nope1"})

    assert correct.status_code == 403
    assert correct.json() == {"detail": "Учётная запись отключена"}
    assert wrong.status_code == 401


@pytest.mark.usefixtures("registered")
def test_outdated_password_hash_is_upgraded(
    api_client: TestClient, run_sql: RunSql
) -> None:
    weak_hash = Argon2Hasher(time_cost=1).hash(PASSWORD)
    run_sql(f"UPDATE users SET hashed_password = '{weak_hash}'")

    response = api_client.post(URL, json={"email": EMAIL, "password": PASSWORD})

    assert response.status_code == 200
    ((stored,),) = run_sql("SELECT hashed_password FROM users")
    assert isinstance(stored, str)
    assert stored != weak_hash
    assert ",t=1," not in stored


def test_empty_password_gives_422(api_client: TestClient) -> None:
    response = api_client.post(URL, json={"email": EMAIL, "password": ""})

    assert response.status_code == 422


def test_login_is_documented_in_openapi(api_client: TestClient) -> None:
    operation = api_client.get("/openapi.json").json()["paths"][URL]["post"]

    assert {"200", "401", "403", "422"} <= set(operation["responses"])
