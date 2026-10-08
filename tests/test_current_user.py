import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.api.deps import CurrentUserDep
from app.core.config import Settings
from app.core.tokens import create_access_token
from app.main import create_app
from app.models.user import UserRole

pytestmark = pytest.mark.integration

RunSql = Callable[[str], list[tuple[object, ...]]]
CREDENTIALS = {"email": "grower@example.md", "password": "secret123"}
URL = "/api/v1/whoami"


@pytest.fixture
def settings_with_db(migrated_database_url: str) -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        database_url=migrated_database_url,  # type: ignore[arg-type]
    )


@pytest.fixture
def client(settings_with_db: Settings, run_sql: RunSql) -> Iterator[TestClient]:
    """Приложение с защищённым эндпоинтом только для теста."""
    app = create_app(settings_with_db)

    @app.get(URL)
    async def whoami(user: CurrentUserDep) -> dict[str, str]:
        return {"email": user.email, "role": user.role}

    with TestClient(app) as test_client:
        yield test_client
    run_sql("TRUNCATE users CASCADE")


@pytest.fixture
def access_token(client: TestClient) -> str:
    client.post(
        "/api/v1/auth/register", json={**CREDENTIALS, "full_name": "Андрей Бахов"}
    )
    token: str = client.post("/api/v1/auth/login", json=CREDENTIALS).json()[
        "access_token"
    ]
    return token


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_valid_token_gives_the_user(client: TestClient, access_token: str) -> None:
    response = client.get(URL, headers=bearer(access_token))

    assert response.status_code == 200
    assert response.json() == {"email": "grower@example.md", "role": "user"}


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer"},
        {"Authorization": "Basic Z3Jvd2VyOnNlY3JldA=="},
        {"Authorization": "Bearer not-a-token"},
    ],
    ids=["no-header", "empty", "other-scheme", "garbage"],
)
def test_missing_or_invalid_credentials_give_401(
    client: TestClient, headers: dict[str, str]
) -> None:
    response = client.get(URL, headers=headers)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_refresh_token_is_not_accepted(client: TestClient, access_token: str) -> None:
    refresh = client.cookies["refresh_token"]

    assert client.get(URL, headers=bearer(refresh)).status_code == 401


def test_expired_token_is_reported(
    client: TestClient, settings_with_db: Settings, run_sql: RunSql, access_token: str
) -> None:
    ((user_id,),) = run_sql("SELECT id FROM users")
    assert isinstance(user_id, uuid.UUID)
    issued_long_ago = datetime.now(UTC) - timedelta(hours=1)
    expired = create_access_token(
        user_id, UserRole.USER, settings_with_db, now=issued_long_ago
    ).token

    response = client.get(URL, headers=bearer(expired))

    assert response.status_code == 401
    assert response.json() == {"detail": "Срок действия токена истёк"}


def test_disabled_user_loses_access_immediately(
    client: TestClient, run_sql: RunSql, access_token: str
) -> None:
    run_sql("UPDATE users SET is_active = false")

    assert client.get(URL, headers=bearer(access_token)).status_code == 401


def test_deleted_user_loses_access_immediately(
    client: TestClient, run_sql: RunSql, access_token: str
) -> None:
    run_sql("DELETE FROM users")

    assert client.get(URL, headers=bearer(access_token)).status_code == 401


def test_role_comes_from_database(
    client: TestClient, run_sql: RunSql, access_token: str
) -> None:
    # В токене роль user, но сервер верит базе: смена роли действует сразу
    run_sql("UPDATE users SET role = 'agronomist'")

    response = client.get(URL, headers=bearer(access_token))

    assert response.json()["role"] == "agronomist"


def test_bearer_scheme_is_documented(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert schema["components"]["securitySchemes"]["HTTPBearer"]["scheme"] == "bearer"
    assert schema["paths"][URL]["get"]["security"] == [{"HTTPBearer": []}]
