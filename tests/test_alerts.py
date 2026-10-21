from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from geoalchemy2 import WKTElement
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Alert, Disease, DiseaseRisk, RiskLevel, User, Vineyard
from app.services import risk as risk_service
from app.services.alerts import raise_alerts

pytestmark = pytest.mark.integration

RunSql = Callable[[str], list[tuple[object, ...]]]
PLOT = "MULTIPOLYGON(((28.86 47.14, 28.87 47.14, 28.87 47.15, 28.86 47.14)))"
TODAY = date(2026, 6, 10)
CREDENTIALS = {"email": "grower@example.md", "password": "secret123"}
SQUARE: dict[str, Any] = {
    "type": "Polygon",
    "coordinates": [
        [[28.86, 47.14], [28.862, 47.14], [28.862, 47.141], [28.86, 47.14]]
    ],
}


def risk(
    vineyard: Vineyard, day: date, disease: Disease, method: str, level: RiskLevel
) -> DiseaseRisk:
    return DiseaseRisk(
        vineyard=vineyard,
        disease=disease,
        target_date=day,
        method=method,
        score=0.5,
        level=level,
    )


async def alerts_of(session: AsyncSession) -> list[tuple[Disease, date, RiskLevel]]:
    rows = await session.scalars(select(Alert).order_by(Alert.day, Alert.disease))
    return [(alert.disease, alert.day, alert.level) for alert in rows]


async def test_alerts_for_high_risk_in_the_next_days(session: AsyncSession) -> None:
    owner = User(email="grower@example.md", hashed_password="hash")
    vineyard = Vineyard(owner=owner, name="Криково", geom=WKTElement(PLOT, srid=4326))
    tomorrow = TODAY + timedelta(days=1)
    session.add_all(
        [
            # Милдью сегодня: худшее из двух моделей — высокий
            risk(vineyard, TODAY, Disease.DOWNY_MILDEW, "three_tens", RiskLevel.HIGH),
            risk(vineyard, TODAY, Disease.DOWNY_MILDEW, "goidanich", RiskLevel.LOW),
            risk(
                vineyard,
                tomorrow,
                Disease.POWDERY_MILDEW,
                "gubler_thomas",
                RiskLevel.MEDIUM,
            ),
            # Вчера и через три дня — вне окна предупреждений
            risk(
                vineyard,
                TODAY - timedelta(days=1),
                Disease.DOWNY_MILDEW,
                "three_tens",
                RiskLevel.HIGH,
            ),
            risk(
                vineyard,
                TODAY + timedelta(days=3),
                Disease.DOWNY_MILDEW,
                "three_tens",
                RiskLevel.HIGH,
            ),
        ]
    )
    await session.flush()

    assert await raise_alerts(session, vineyard, TODAY) == 1
    assert await alerts_of(session) == [(Disease.DOWNY_MILDEW, TODAY, RiskLevel.HIGH)]
    # Порог «средний» добавляет оидиум завтра; повтор не плодит дублей
    assert await raise_alerts(session, vineyard, TODAY, RiskLevel.MEDIUM) == 2
    assert await raise_alerts(session, vineyard, TODAY, RiskLevel.MEDIUM) == 2
    assert await alerts_of(session) == [
        (Disease.DOWNY_MILDEW, TODAY, RiskLevel.HIGH),
        (Disease.POWDERY_MILDEW, tomorrow, RiskLevel.MEDIUM),
    ]


async def test_rising_level_makes_alert_unread_again(session: AsyncSession) -> None:
    owner = User(email="grower@example.md", hashed_password="hash")
    vineyard = Vineyard(owner=owner, name="Криково", geom=WKTElement(PLOT, srid=4326))
    medium = risk(
        vineyard, TODAY, Disease.POWDERY_MILDEW, "gubler_thomas", RiskLevel.MEDIUM
    )
    session.add(medium)
    await session.flush()
    await raise_alerts(session, vineyard, TODAY, RiskLevel.MEDIUM)
    alert = await session.scalar(select(Alert))
    assert alert is not None
    alert.read_at = datetime(2026, 6, 10, 8, tzinfo=UTC)
    await session.flush()

    # Риск снизился — предупреждение не трогаем
    medium.level = RiskLevel.LOW
    await session.flush()
    assert await raise_alerts(session, vineyard, TODAY, RiskLevel.LOW) == 1
    await session.refresh(alert)
    assert (alert.level, alert.read_at is None) == (RiskLevel.MEDIUM, False)

    # Риск вырос — уровень обновлён, предупреждение снова непрочитанное
    medium.level = RiskLevel.HIGH
    await session.flush()
    await raise_alerts(session, vineyard, TODAY)
    await session.refresh(alert)
    assert (alert.level, alert.read_at) == (RiskLevel.HIGH, None)


async def test_no_alerts_without_risk(session: AsyncSession) -> None:
    owner = User(email="grower@example.md", hashed_password="hash")
    vineyard = Vineyard(owner=owner, name="Криково", geom=WKTElement(PLOT, srid=4326))
    session.add(risk(vineyard, TODAY, Disease.DOWNY_MILDEW, "goidanich", RiskLevel.LOW))
    await session.flush()

    assert await raise_alerts(session, vineyard, TODAY) == 0
    assert await alerts_of(session) == []


def login(client: TestClient, email: str = CREDENTIALS["email"]) -> dict[str, str]:
    credentials = {**CREDENTIALS, "email": email}
    client.post("/api/v1/auth/register", json={**credentials, "full_name": "Ион"})
    token = client.post("/api/v1/auth/login", json=credentials).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def add_alert(
    run_sql: RunSql, client: TestClient, headers: dict[str, str], day: str
) -> str:
    vineyard = client.post(
        "/api/v1/vineyards",
        headers=headers,
        json={"name": "Криково", "geometry": SQUARE},
    ).json()
    run_sql(
        "INSERT INTO alerts (user_id, vineyard_id, disease, day, level)"
        f" SELECT owner_id, id, 'downy_mildew', '{day}', 'high' FROM vineyards"
        f" WHERE id = '{vineyard['id']}'"
    )
    rows = run_sql(f"SELECT id FROM alerts WHERE vineyard_id = '{vineyard['id']}'")
    return str(rows[0][0])


def test_alerts_api(
    api_client: TestClient, run_sql: RunSql, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(risk_service, "local_today", lambda: TODAY)
    from app.api.routes import alerts as alerts_route

    monkeypatch.setattr(alerts_route, "local_today", lambda: TODAY)
    headers = login(api_client)
    alert_id = add_alert(run_sql, api_client, headers, "2026-06-11")
    url = "/api/v1/alerts"

    listed = api_client.get(url, headers=headers).json()
    assert listed["total"] == 1
    item = listed["items"][0]
    assert item["vineyard_name"] == "Криково"
    assert item["disease"] == "downy_mildew"
    assert item["level"] == "high"
    assert item["read_at"] is None
    assert item["forecast"] is True
    assert api_client.get(f"{url}/unread-count", headers=headers).json() == {"count": 1}

    read = api_client.post(f"{url}/{alert_id}/read", headers=headers).json()
    assert read["read_at"] is not None
    # Повторная отметка не меняет время прочтения
    again = api_client.post(f"{url}/{alert_id}/read", headers=headers).json()
    assert again["read_at"] == read["read_at"]
    assert (
        api_client.get(url, headers=headers, params={"unread": True}).json()["total"]
        == 0
    )

    run_sql(f"UPDATE alerts SET read_at = NULL WHERE id = '{alert_id}'")
    assert api_client.post(f"{url}/read-all", headers=headers).json() == {"updated": 1}
    assert api_client.get(f"{url}/unread-count", headers=headers).json() == {"count": 0}


def test_alerts_are_private(api_client: TestClient, run_sql: RunSql) -> None:
    headers = login(api_client)
    alert_id = add_alert(run_sql, api_client, headers, "2026-06-10")
    stranger = login(api_client, "neighbour@example.md")

    assert api_client.get("/api/v1/alerts", headers=stranger).json()["total"] == 0
    response = api_client.post(f"/api/v1/alerts/{alert_id}/read", headers=stranger)
    assert response.status_code == 404
    assert api_client.get("/api/v1/alerts").status_code == 401
