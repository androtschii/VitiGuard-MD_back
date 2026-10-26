import asyncio
import uuid
from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient
from redis.asyncio import Redis
from starlette.websockets import WebSocketDisconnect

from app.core.config import Settings
from app.integrations.notifier import RedisNotifier
from app.main import create_app

pytestmark = pytest.mark.integration

RunSql = Callable[[str], list[tuple[object, ...]]]
CREDENTIALS = {"email": "grower@example.md", "password": "secret123"}
SQUARE = {
    "type": "Polygon",
    "coordinates": [
        [[28.86, 47.14], [28.862, 47.14], [28.862, 47.141], [28.86, 47.14]]
    ],
}


@pytest.fixture
def settings(migrated_database_url: str, redis_url: str) -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        database_url=migrated_database_url,  # type: ignore[arg-type]
        cache_url=f"{redis_url}/2",  # type: ignore[arg-type]
    )


@pytest.fixture
def client(settings: Settings, run_sql: RunSql) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client
    run_sql("TRUNCATE users CASCADE")


def login(client: TestClient) -> tuple[str, str]:
    client.post("/api/v1/auth/register", json={**CREDENTIALS, "full_name": "Ион"})
    token: str = client.post("/api/v1/auth/login", json=CREDENTIALS).json()[
        "access_token"
    ]
    user_id: str = client.get(
        "/api/v1/users/me", headers={"Authorization": f"Bearer {token}"}
    ).json()["id"]
    return token, user_id


def publish(settings: Settings, user_id: str) -> None:
    async def run() -> None:
        notifier = RedisNotifier.from_settings(settings)
        await notifier.alerts_changed(uuid.UUID(user_id))
        await notifier.close()

    asyncio.run(run())


def test_unread_count_in_real_time(
    client: TestClient, settings: Settings, run_sql: RunSql
) -> None:
    token, user_id = login(client)
    headers = {"Authorization": f"Bearer {token}"}
    vineyard = client.post(
        "/api/v1/vineyards",
        headers=headers,
        json={"name": "Криково", "geometry": SQUARE},
    ).json()

    with client.websocket_connect(f"/api/v1/ws/alerts?token={token}") as socket:
        # Сразу после подключения — текущее число
        assert socket.receive_json() == {"unread": 0}

        # Пересчёт риска создал предупреждение и подал сигнал
        run_sql(
            "INSERT INTO alerts (user_id, vineyard_id, disease, day, level)"
            f" VALUES ('{user_id}', '{vineyard['id']}', 'downy_mildew',"
            " '2026-06-10', 'high')"
        )
        publish(settings, user_id)
        assert socket.receive_json() == {"unread": 1}

        # Прочитано в другой вкладке — значок обновляется и здесь
        alert_id = run_sql("SELECT id FROM alerts")[0][0]
        client.post(f"/api/v1/alerts/{alert_id}/read", headers=headers)
        assert socket.receive_json() == {"unread": 0}


@pytest.mark.parametrize("token", ["", "not-a-token"])
def test_socket_requires_access_token(client: TestClient, token: str) -> None:
    with (
        pytest.raises(WebSocketDisconnect) as closed,
        client.websocket_connect(f"/api/v1/ws/alerts?token={token}") as socket,
    ):
        socket.receive_json()

    assert closed.value.code == 1008


def test_socket_rejects_deleted_user(client: TestClient, run_sql: RunSql) -> None:
    token, _ = login(client)
    run_sql("UPDATE users SET is_active = false")

    with (
        pytest.raises(WebSocketDisconnect) as closed,
        client.websocket_connect(f"/api/v1/ws/alerts?token={token}") as socket,
    ):
        socket.receive_json()

    assert closed.value.code == 1008


async def test_notifier_survives_unavailable_redis() -> None:
    # Порт, где Redis нет: сигнал теряется, но ошибки нет
    notifier = RedisNotifier(
        Redis.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=0.2)
    )

    await notifier.alerts_changed(uuid.uuid4())
    await notifier.close()
