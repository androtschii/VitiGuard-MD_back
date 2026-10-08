from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

URL = "/api/v1/users/me"
RunSql = Callable[[str], list[tuple[object, ...]]]
CREDENTIALS = {"email": "grower@example.md", "password": "secret123"}


@pytest.fixture
def headers(api_client: TestClient) -> dict[str, str]:
    api_client.post(
        "/api/v1/auth/register",
        json={**CREDENTIALS, "full_name": "Андрей Бахов", "role": "agronomist"},
    )
    token = api_client.post("/api/v1/auth/login", json=CREDENTIALS).json()[
        "access_token"
    ]
    return {"Authorization": f"Bearer {token}"}


def test_profile_of_current_user(
    api_client: TestClient, headers: dict[str, str]
) -> None:
    response = api_client.get(URL, headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["email"] == "grower@example.md"
    assert body["full_name"] == "Андрей Бахов"
    assert body["role"] == "agronomist"
    assert "hashed_password" not in body


def test_name_can_be_changed(
    api_client: TestClient, run_sql: RunSql, headers: dict[str, str]
) -> None:
    before = api_client.get(URL, headers=headers).json()

    response = api_client.patch(
        URL, headers=headers, json={"full_name": "  Ион Попеску  "}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["full_name"] == "Ион Попеску"
    assert body["updated_at"] > before["updated_at"]
    assert run_sql("SELECT full_name FROM users") == [("Ион Попеску",)]


@pytest.mark.parametrize(
    "payload",
    [
        {"role": "admin"},
        {"full_name": "Ион", "role": "admin"},
        {"full_name": "Ион", "email": "other@example.md"},
        {"full_name": "Ион", "is_active": False},
        {"full_name": " "},
        {},
    ],
    ids=["role", "role-with-name", "email", "is-active", "blank-name", "empty"],
)
def test_protected_fields_cannot_be_changed(
    api_client: TestClient,
    run_sql: RunSql,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> None:
    response = api_client.patch(URL, headers=headers, json=payload)

    assert response.status_code == 422
    assert run_sql("SELECT email, role, is_active FROM users") == [
        ("grower@example.md", "agronomist", True)
    ]


@pytest.mark.parametrize("method", ["get", "patch"])
def test_profile_requires_login(api_client: TestClient, method: str) -> None:
    response = api_client.request(method, URL, json={"full_name": "Ион"})

    assert response.status_code == 401


def test_profile_is_documented_as_protected(api_client: TestClient) -> None:
    path = api_client.get("/openapi.json").json()["paths"][URL]

    for method in ("get", "patch"):
        assert path[method]["security"] == [{"HTTPBearer": []}]
        assert "401" in path[method]["responses"]
