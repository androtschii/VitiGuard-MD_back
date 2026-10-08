from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.security import verify_password

pytestmark = pytest.mark.integration

URL = "/api/v1/auth/register"
RunSql = Callable[[str], list[tuple[object, ...]]]


def payload(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "email": "grower@example.md",
        "password": "secret123",
        "full_name": "Андрей Бахов",
        "role": "user",
    }
    data.update(overrides)
    return data


def test_registration_creates_user(api_client: TestClient, run_sql: RunSql) -> None:
    response = api_client.post(URL, json=payload(role="agronomist"))

    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "grower@example.md"
    assert body["full_name"] == "Андрей Бахов"
    assert body["role"] == "agronomist"
    assert body["is_active"] is True
    assert {"id", "created_at", "updated_at"} <= set(body)
    assert run_sql("SELECT email, role FROM users") == [
        ("grower@example.md", "agronomist")
    ]


def test_response_never_contains_password_or_hash(api_client: TestClient) -> None:
    response = api_client.post(URL, json=payload())

    assert "password" not in response.text
    assert "secret123" not in response.text
    assert "argon2" not in response.text


def test_password_is_stored_only_as_argon2_hash(
    api_client: TestClient, run_sql: RunSql
) -> None:
    api_client.post(URL, json=payload())

    (stored,) = run_sql("SELECT hashed_password FROM users")[0]

    assert isinstance(stored, str)
    assert stored.startswith("$argon2id$")
    assert "secret123" not in stored
    assert verify_password("secret123", stored) is True
    assert verify_password("secret124", stored) is False


def test_role_defaults_to_user(api_client: TestClient) -> None:
    body = {k: v for k, v in payload().items() if k != "role"}

    assert api_client.post(URL, json=body).json()["role"] == "user"


def test_email_is_stored_in_lower_case(api_client: TestClient, run_sql: RunSql) -> None:
    response = api_client.post(URL, json=payload(email="Grower@Example.MD"))

    assert response.json()["email"] == "grower@example.md"
    assert run_sql("SELECT email FROM users") == [("grower@example.md",)]


def test_taken_email_gives_conflict(api_client: TestClient, run_sql: RunSql) -> None:
    api_client.post(URL, json=payload())

    response = api_client.post(URL, json=payload(full_name="Другой человек"))

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Пользователь с таким email уже зарегистрирован"
    }
    assert run_sql("SELECT count(*) FROM users") == [(1,)]


def test_email_uniqueness_ignores_case(api_client: TestClient, run_sql: RunSql) -> None:
    api_client.post(URL, json=payload())

    response = api_client.post(URL, json=payload(email="GROWER@EXAMPLE.MD"))

    assert response.status_code == 409
    assert run_sql("SELECT count(*) FROM users") == [(1,)]


def test_service_keeps_working_after_conflict(api_client: TestClient) -> None:
    api_client.post(URL, json=payload())
    api_client.post(URL, json=payload())

    response = api_client.post(URL, json=payload(email="second@example.md"))

    assert response.status_code == 201


@pytest.mark.parametrize(
    "overrides",
    [
        {"role": "admin"},
        {"role": "superuser"},
        {"email": "grower"},
        {"password": "abc1"},
        {"password": "12345678"},
        {"password": "abcdefgh"},
        {"password": "a1" * 65},
        {"full_name": "А"},
        {"is_admin": True},
    ],
)
def test_invalid_data_gives_422_and_creates_nothing(
    api_client: TestClient, run_sql: RunSql, overrides: dict[str, Any]
) -> None:
    response = api_client.post(URL, json=payload(**overrides))

    assert response.status_code == 422
    assert run_sql("SELECT count(*) FROM users") == [(0,)]


@pytest.mark.parametrize("missing", ["email", "password", "full_name"])
def test_missing_field_gives_422(api_client: TestClient, missing: str) -> None:
    body = {k: v for k, v in payload().items() if k != missing}

    assert api_client.post(URL, json=body).status_code == 422


def test_validation_error_does_not_echo_the_password(api_client: TestClient) -> None:
    response = api_client.post(URL, json=payload(password="abc1"))

    assert response.status_code == 422
    assert "abc1" not in response.text
    assert "input" not in response.text
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "password"]
    assert set(error) == {"loc", "msg", "type"}


def test_registration_is_documented_in_openapi(api_client: TestClient) -> None:
    operation = api_client.get("/openapi.json").json()["paths"][
        "/api/v1/auth/register"
    ]["post"]

    assert set(operation["responses"]) >= {"201", "409", "422"}
