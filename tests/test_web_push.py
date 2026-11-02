import asyncio
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from pywebpush import WebPushException

from app.core.config import Settings
from app.db.session import create_engine, create_session_factory
from app.models.push_subscription import PushSubscription
from app.services import web_push
from app.services.risk import local_today
from app.services.web_push import PushRejected, deliver, send_push_alerts

CREDENTIALS = {"email": "grower@example.md", "password": "secret123"}
POLYGON = {
    "type": "MultiPolygon",
    "coordinates": [
        [[[28.8, 47.0], [28.81, 47.0], [28.81, 47.01], [28.8, 47.01], [28.8, 47.0]]]
    ],
}
SUBSCRIPTION = {
    "endpoint": "https://push.example.org/send/abc",
    "keys": {"p256dh": "key", "auth": "secret"},
}
RunSql = Callable[[str], list[tuple[object, ...]]]
VAPID = {
    "vapid_public_key": "public",
    "vapid_private_key": SecretStr("private"),
    "vapid_subject": "mailto:admin@vitiguard.md",
}


def sign_in(client: TestClient, email: str = CREDENTIALS["email"]) -> dict[str, str]:
    body = {**CREDENTIALS, "email": email}
    client.post("/api/v1/auth/register", json={**body, "full_name": "Ион"})
    token = client.post("/api/v1/auth/login", json=body).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def configure(client: TestClient, **values: Any) -> Settings:
    app = client.app
    settings: Settings = app.state.settings  # type: ignore[attr-defined]
    updated = settings.model_copy(update=values)
    app.state.settings = updated  # type: ignore[attr-defined]
    return updated


@pytest.mark.integration
def test_key_only_when_configured(api_client: TestClient) -> None:
    headers = sign_in(api_client)

    assert api_client.get("/api/v1/push/key", headers=headers).status_code == 503
    configure(api_client, **VAPID)
    response = api_client.get("/api/v1/push/key", headers=headers)
    assert response.json() == {"public_key": "public"}


@pytest.mark.integration
def test_subscription_moves_to_new_owner_and_unsubscribes(
    api_client: TestClient, run_sql: RunSql
) -> None:
    first = sign_in(api_client)
    url = "/api/v1/push/subscriptions"
    assert api_client.post(url, headers=first, json=SUBSCRIPTION).status_code == 204
    second = sign_in(api_client, "other@example.md")
    again = {**SUBSCRIPTION, "keys": {"p256dh": "new", "auth": "new"}}
    assert api_client.post(url, headers=second, json=again).status_code == 204

    rows = run_sql(
        "SELECT u.email, s.p256dh FROM push_subscriptions s JOIN users u ON u.id = s.user_id"
    )
    assert rows == [("other@example.md", "new")]
    endpoint = {"endpoint": SUBSCRIPTION["endpoint"]}
    # Чужую подписку не удалить
    api_client.delete(url, headers=first, params=endpoint)
    assert run_sql("SELECT count(*) FROM push_subscriptions") == [(1,)]
    assert api_client.delete(url, headers=second, params=endpoint).status_code == 204
    assert run_sql("SELECT count(*) FROM push_subscriptions") == [(0,)]


@pytest.mark.integration
def test_rejects_bad_subscription(api_client: TestClient) -> None:
    headers = sign_in(api_client)
    bad = {"endpoint": "not a url", "keys": SUBSCRIPTION["keys"]}

    response = api_client.post("/api/v1/push/subscriptions", headers=headers, json=bad)

    assert response.status_code == 422


def make_alert(client: TestClient, run_sql: RunSql, headers: dict[str, str]) -> str:
    vineyard = client.post(
        "/api/v1/vineyards",
        headers=headers,
        json={"name": "Криково", "geometry": POLYGON},
    ).json()["id"]
    run_sql(
        "INSERT INTO alerts (user_id, vineyard_id, disease, day, level) "
        f"SELECT owner_id, id, 'downy_mildew', '{local_today()}', 'high' "
        f"FROM vineyards WHERE id = '{vineyard}'"
    )
    return str(vineyard)


async def push(settings: Settings, send: Any) -> int:
    engine = create_engine(settings)
    try:
        async with create_session_factory(engine)() as session:
            return await send_push_alerts(session, settings, local_today(), send)
    finally:
        await engine.dispose()


@pytest.mark.integration
def test_sends_once_and_drops_gone_subscriptions(
    api_client: TestClient, run_sql: RunSql
) -> None:
    headers = sign_in(api_client)
    url = "/api/v1/push/subscriptions"
    api_client.post(url, headers=headers, json=SUBSCRIPTION)
    gone = {**SUBSCRIPTION, "endpoint": "https://push.example.org/send/gone"}
    api_client.post(url, headers=headers, json=gone)
    broken = {**SUBSCRIPTION, "endpoint": "https://push.example.org/send/broken"}
    api_client.post(url, headers=headers, json=broken)
    make_alert(api_client, run_sql, headers)
    settings = configure(api_client, **VAPID)
    sent: list[tuple[str, dict[str, str]]] = []

    def send(subscription: PushSubscription, payload: str, _: Settings) -> None:
        if subscription.endpoint.endswith("gone"):
            raise PushRejected(gone=True)
        if subscription.endpoint.endswith("broken"):
            raise PushRejected(gone=False)
        sent.append((subscription.endpoint, json.loads(payload)))

    assert asyncio.run(push(settings, send)) == 1
    assert sent[0][0] == SUBSCRIPTION["endpoint"]
    assert "Криково" in sent[0][1]["body"]
    assert sent[0][1]["url"] == "/alerts"
    endpoints = {row[0] for row in run_sql("SELECT endpoint FROM push_subscriptions")}
    assert endpoints == {SUBSCRIPTION["endpoint"], broken["endpoint"]}
    # Повторно то же предупреждение не уходит
    assert asyncio.run(push(settings, send)) == 0


@pytest.mark.integration
def test_without_delivery_alert_waits_for_next_time(
    api_client: TestClient, run_sql: RunSql
) -> None:
    headers = sign_in(api_client)
    api_client.post("/api/v1/push/subscriptions", headers=headers, json=SUBSCRIPTION)
    make_alert(api_client, run_sql, headers)
    settings = configure(api_client, **VAPID)

    def fail(*_: Any) -> None:
        raise PushRejected(gone=False)

    assert asyncio.run(push(settings, fail)) == 0
    assert run_sql("SELECT pushed_at FROM alerts") == [(None,)]


@pytest.mark.integration
def test_nothing_to_send(api_client: TestClient) -> None:
    settings = configure(api_client, **VAPID)

    assert asyncio.run(push(settings, lambda *_: None)) == 0


def test_deliver_maps_push_service_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None, **VAPID)  # type: ignore[arg-type]
    subscription = PushSubscription(
        endpoint="https://p.example/x", p256dh="k", auth="a"
    )
    calls: list[dict[str, Any]] = []

    def fake(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(web_push, "webpush", fake)
    deliver(subscription, "{}", settings)
    assert calls[0]["vapid_private_key"] == "private"
    assert calls[0]["vapid_claims"] == {"sub": "mailto:admin@vitiguard.md"}

    for status, gone in ((410, True), (500, False), (None, False)):

        def failing(status: int | None = status, **_: Any) -> None:
            response = SimpleNamespace(status_code=status) if status else None
            raise WebPushException("нет", response=response)  # type: ignore[arg-type]

        monkeypatch.setattr(web_push, "webpush", failing)
        with pytest.raises(PushRejected) as error:
            deliver(subscription, "{}", settings)
        assert error.value.gone is gone
