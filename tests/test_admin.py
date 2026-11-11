import uuid
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

RunSql = Callable[[str], list[tuple[object, ...]]]
URL = "/api/v1/admin/users"


def register(client: TestClient, email: str, name: str = "Ион") -> None:
    client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "secret123", "full_name": name},
    )


def login(client: TestClient, email: str) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login", json={"email": email, "password": "secret123"}
    )
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def admin(api_client: TestClient, run_sql: RunSql) -> dict[str, str]:
    register(api_client, "admin@vitiguard.md", "Админ")
    register(api_client, "grower@example.md", "Ион Попеску")
    register(api_client, "agro@example.md", "Мария")
    run_sql("UPDATE users SET role = 'admin' WHERE email = 'admin@vitiguard.md'")
    return login(api_client, "admin@vitiguard.md")


def user_id(client: TestClient, headers: dict[str, str], email: str) -> str:
    items = client.get(URL, headers=headers, params={"q": email}).json()["items"]
    found: str = items[0]["id"]
    return found


def test_admin_lists_and_searches_users(
    api_client: TestClient, admin: dict[str, str]
) -> None:
    page = api_client.get(URL, headers=admin, params={"size": 2}).json()

    assert (page["total"], page["pages"], len(page["items"])) == (3, 2, 2)
    assert set(page["items"][0]) >= {"email", "role", "is_active", "telegram_linked"}
    found = api_client.get(URL, headers=admin, params={"q": "попеску"}).json()
    assert [item["email"] for item in found["items"]] == ["grower@example.md"]
    blank = api_client.get(URL, headers=admin, params={"q": "  "}).json()
    assert blank["total"] == 3


def test_admin_changes_role_and_blocks(
    api_client: TestClient, admin: dict[str, str], run_sql: RunSql
) -> None:
    grower = login(api_client, "grower@example.md")
    target = user_id(api_client, admin, "grower@example.md")

    promoted = api_client.patch(
        f"{URL}/{target}", headers=admin, json={"role": "agronomist"}
    )
    assert promoted.json()["role"] == "agronomist"

    blocked = api_client.patch(
        f"{URL}/{target}", headers=admin, json={"is_active": False}
    )
    assert blocked.json()["is_active"] is False
    # Доступ закрывается сразу: и по выданному токену, и по refresh-cookie
    assert api_client.get("/api/v1/users/me", headers=grower).status_code == 401
    revoked = run_sql(
        "SELECT count(*) FROM refresh_tokens t JOIN users u ON u.id = t.user_id "
        "WHERE u.email = 'grower@example.md' AND t.revoked_at IS NULL"
    )
    assert revoked == [(0,)]

    api_client.patch(f"{URL}/{target}", headers=admin, json={"is_active": True})
    assert (
        api_client.get(
            "/api/v1/users/me", headers=login(api_client, "grower@example.md")
        ).status_code
        == 200
    )


def test_admin_cannot_lock_himself_out(
    api_client: TestClient, admin: dict[str, str]
) -> None:
    me = api_client.get("/api/v1/users/me", headers=admin).json()["id"]

    demote = api_client.patch(f"{URL}/{me}", headers=admin, json={"role": "user"})
    block = api_client.patch(f"{URL}/{me}", headers=admin, json={"is_active": False})
    keep = api_client.patch(
        f"{URL}/{me}", headers=admin, json={"role": "admin", "is_active": True}
    )

    assert demote.status_code == block.status_code == 400
    assert demote.json() == {
        "detail": "Нельзя снять права администратора с самого себя"
    }
    assert keep.status_code == 200


def test_only_admin_gets_in(api_client: TestClient, admin: dict[str, str]) -> None:
    grower = login(api_client, "grower@example.md")

    assert api_client.get(URL, headers=grower).status_code == 403
    assert api_client.get(URL).status_code == 401
    missing = api_client.patch(f"{URL}/{uuid.uuid4()}", headers=admin, json={})
    assert missing.status_code == 404
    bad_role = api_client.patch(
        f"{URL}/{uuid.uuid4()}", headers=admin, json={"role": "owner"}
    )
    assert bad_role.status_code == 422


def test_admin_sends_password_reset_link(
    api_client: TestClient, admin: dict[str, str]
) -> None:
    from app.api.deps import get_reset_mailer

    sent: list[tuple[str, str]] = []
    app = api_client.app
    app.dependency_overrides[get_reset_mailer] = lambda: (  # type: ignore[attr-defined]
        lambda to, url: sent.append((to, url))
    )
    try:
        target = user_id(api_client, admin, "grower@example.md")
        url = f"{URL}/{target}/password-reset"

        assert api_client.post(url, headers=admin).status_code == 202
        assert [to for to, _ in sent] == ["grower@example.md"]
        assert "/reset-password?token=" in sent[0][1]

        # Пароль администратору не показывается ни в одном ответе
        listed = api_client.get(URL, headers=admin).json()["items"]
        assert all("password" not in key for item in listed for key in item)

        # Простой пользователь ссылку для чужого аккаунта не отправит
        grower = login(api_client, "grower@example.md")
        assert api_client.post(url, headers=grower).status_code == 403

        missing = f"{URL}/{uuid.uuid4()}/password-reset"
        assert api_client.post(missing, headers=admin).status_code == 404

        api_client.patch(f"{URL}/{target}", headers=admin, json={"is_active": False})
        assert api_client.post(url, headers=admin).status_code == 400
        assert len(sent) == 1
    finally:
        app.dependency_overrides.pop(get_reset_mailer, None)  # type: ignore[attr-defined]
