from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_reset_mailer
from app.core.config import Settings
from app.core.tokens import hash_token
from app.main import create_app

pytestmark = pytest.mark.integration

REQUEST_URL = "/api/v1/auth/password-reset"
CONFIRM_URL = "/api/v1/auth/password-reset/confirm"
RunSql = Callable[[str], list[tuple[object, ...]]]
EMAIL = "grower@example.md"
OLD_PASSWORD = "secret123"
NEW_PASSWORD = "newpass456"


class SentMail(list[tuple[str, str]]):
    """Письма, которые приложение «отправило» (вместо очереди Celery)."""

    def tokens(self) -> list[str]:
        return [parse_qs(urlparse(url).query)["token"][0] for _, url in self]


@pytest.fixture
def mail() -> SentMail:
    return SentMail()


@pytest.fixture
def client(
    migrated_database_url: str, run_sql: RunSql, mail: SentMail
) -> Iterator[TestClient]:
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url=migrated_database_url,  # type: ignore[arg-type]
        frontend_url="https://app.vitiguard.md",
    )
    app = create_app(settings)
    app.dependency_overrides[get_reset_mailer] = lambda: (
        lambda to, url: mail.append((to, url))
    )
    with TestClient(app) as test_client:
        test_client.post(
            "/api/v1/auth/register",
            json={"email": EMAIL, "password": OLD_PASSWORD, "full_name": "Андрей"},
        )
        yield test_client
    run_sql("TRUNCATE users CASCADE")


def login(client: TestClient, password: str) -> int:
    response = client.post(
        "/api/v1/auth/login", json={"email": EMAIL, "password": password}
    )
    return response.status_code


def confirm(client: TestClient, token: str, password: str = NEW_PASSWORD) -> Any:
    return client.post(CONFIRM_URL, json={"token": token, "new_password": password})


def test_request_sends_link_to_frontend(
    client: TestClient, run_sql: RunSql, mail: SentMail
) -> None:
    response = client.post(REQUEST_URL, json={"email": "Grower@Example.MD"})

    assert response.status_code == 202
    ((to, url),) = mail
    assert to == EMAIL
    assert url.startswith("https://app.vitiguard.md/reset-password?token=")
    # В базе только хэш ключа
    assert run_sql("SELECT token_hash FROM password_reset_tokens") == [
        (hash_token(mail.tokens()[0]),)
    ]


@pytest.mark.parametrize("email", ["nobody@example.md", EMAIL])
def test_response_does_not_reveal_whether_account_exists(
    client: TestClient, run_sql: RunSql, mail: SentMail, email: str
) -> None:
    run_sql("UPDATE users SET is_active = false")  # для существующего — отключён

    response = client.post(REQUEST_URL, json={"email": email})

    assert response.status_code == 202
    assert response.content == b""
    assert mail == []


def test_new_password_works_and_old_does_not(
    client: TestClient, mail: SentMail
) -> None:
    client.post(REQUEST_URL, json={"email": EMAIL})

    response = confirm(client, mail.tokens()[0])

    assert response.status_code == 204
    assert login(client, NEW_PASSWORD) == 200
    assert login(client, OLD_PASSWORD) == 401


def test_link_works_only_once(client: TestClient, mail: SentMail) -> None:
    client.post(REQUEST_URL, json={"email": EMAIL})
    token = mail.tokens()[0]

    assert confirm(client, token).status_code == 204
    second = confirm(client, token, "another789")

    assert second.status_code == 400
    assert second.json() == {
        "detail": "Ссылка недействительна или устарела, запросите новую"
    }
    assert login(client, NEW_PASSWORD) == 200


def test_only_the_latest_link_works(client: TestClient, mail: SentMail) -> None:
    client.post(REQUEST_URL, json={"email": EMAIL})
    client.post(REQUEST_URL, json={"email": EMAIL})
    first, latest = mail.tokens()

    assert confirm(client, first).status_code == 400
    assert confirm(client, latest).status_code == 204


def test_expired_link_is_rejected(
    client: TestClient, run_sql: RunSql, mail: SentMail
) -> None:
    client.post(REQUEST_URL, json={"email": EMAIL})
    run_sql("UPDATE password_reset_tokens SET expires_at = now() - interval '1 minute'")

    assert confirm(client, mail.tokens()[0]).status_code == 400
    assert login(client, OLD_PASSWORD) == 200


def test_unknown_token_is_rejected(client: TestClient) -> None:
    assert confirm(client, "made-up-token").status_code == 400


def test_password_change_ends_all_sessions(
    client: TestClient, run_sql: RunSql, mail: SentMail
) -> None:
    assert login(client, OLD_PASSWORD) == 200
    client.post(REQUEST_URL, json={"email": EMAIL})

    confirm(client, mail.tokens()[0])

    assert run_sql("SELECT count(*) FROM refresh_tokens WHERE revoked_at IS NULL") == [
        (0,)
    ]


def test_weak_new_password_gives_422_and_keeps_link(
    client: TestClient, mail: SentMail
) -> None:
    client.post(REQUEST_URL, json={"email": EMAIL})
    token = mail.tokens()[0]

    assert confirm(client, token, "short").status_code == 422
    # Ключ не потрачен на неудачную попытку
    assert confirm(client, token).status_code == 204


def test_account_disabled_after_request_cannot_reset(
    client: TestClient, run_sql: RunSql, mail: SentMail
) -> None:
    client.post(REQUEST_URL, json={"email": EMAIL})
    run_sql("UPDATE users SET is_active = false")

    assert confirm(client, mail.tokens()[0]).status_code == 400
