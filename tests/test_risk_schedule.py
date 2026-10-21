from collections.abc import Callable
from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.errors import ExternalServiceError
from app.models import Vineyard
from app.tasks import risk as risk_tasks
from app.worker.celery_app import celery_app

RunSql = Callable[[str], list[tuple[object, ...]]]
CREDENTIALS = {"email": "grower@example.md", "password": "secret123"}
SQUARE: dict[str, Any] = {
    "type": "Polygon",
    "coordinates": [
        [[28.86, 47.14], [28.862, 47.14], [28.862, 47.141], [28.86, 47.14]]
    ],
}


def test_beat_refreshes_risks_every_six_hours() -> None:
    entry = celery_app.conf.beat_schedule["risk-refresh"]

    assert entry["task"] == risk_tasks.REFRESH_ALL
    assert entry["schedule"] == 6 * 3600
    assert risk_tasks.refresh_all.soft_time_limit == 30 * 60


def login(client: TestClient, email: str = CREDENTIALS["email"]) -> dict[str, str]:
    credentials = {**CREDENTIALS, "email": email}
    client.post("/api/v1/auth/register", json={**credentials, "full_name": "Ион"})
    token = client.post("/api/v1/auth/login", json=credentials).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def create_vineyard(client: TestClient, headers: dict[str, str], name: str) -> str:
    shifted = {
        **SQUARE,
        "coordinates": [
            [[x + 0.01 * len(name), y] for x, y in SQUARE["coordinates"][0]]
        ],
    }
    response = client.post(
        "/api/v1/vineyards", headers=headers, json={"name": name, "geometry": shifted}
    )
    vineyard_id: str = response.json()["id"]
    return vineyard_id


@pytest.mark.integration
def test_refresh_all_survives_one_failure(
    api_client: TestClient,
    migrated_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = login(api_client)
    create_vineyard(api_client, headers, "Криково")
    create_vineyard(api_client, headers, "Сбой погоды")
    seen: list[str] = []

    async def fake_refresh(_session: Any, _client: Any, vineyard: Vineyard) -> int:
        seen.append(vineyard.name)
        if vineyard.name == "Сбой погоды":
            raise ExternalServiceError("Сервис погоды недоступен")
        return 1

    monkeypatch.setattr(risk_tasks, "refresh_risks", fake_refresh)
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url=migrated_database_url,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(risk_tasks, "get_settings", lambda: settings)

    assert risk_tasks.refresh_all() == {"refreshed": 1, "failed": 1, "notified": 0}
    assert sorted(seen) == ["Криково", "Сбой погоды"]


def insert_risk(
    run_sql: RunSql, vineyard_id: str, day: str, disease: str, method: str, level: str
) -> None:
    run_sql(
        "INSERT INTO disease_risks (vineyard_id, disease, target_date, method, score,"
        f" level) VALUES ('{vineyard_id}', '{disease}', '{day}', '{method}', 0.5,"
        f" '{level}')"
    )


@pytest.mark.integration
def test_today_levels_of_my_vineyards(
    api_client: TestClient, run_sql: RunSql, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services import risk as risk_service

    monkeypatch.setattr(risk_service, "local_today", lambda: date(2026, 6, 10))
    headers = login(api_client)
    mine = create_vineyard(api_client, headers, "Криково")
    quiet = create_vineyard(api_client, headers, "Без оценок")
    foreign = create_vineyard(
        api_client, login(api_client, "neighbour@example.md"), "Соседний"
    )
    today, yesterday = "2026-06-10", "2026-06-09"
    insert_risk(run_sql, mine, today, "downy_mildew", "three_tens", "high")
    insert_risk(run_sql, mine, today, "downy_mildew", "goidanich", "low")
    insert_risk(run_sql, mine, today, "powdery_mildew", "gubler_thomas", "medium")
    insert_risk(run_sql, mine, yesterday, "powdery_mildew", "gubler_thomas", "high")
    insert_risk(run_sql, foreign, today, "downy_mildew", "three_tens", "high")

    response = api_client.get("/api/v1/risks/today", headers=headers)

    assert response.status_code == 200
    assert response.json() == [
        {
            "vineyard_id": mine,
            "day": today,
            "downy": "high",
            "powdery": "medium",
        }
    ]
    assert quiet not in {item["vineyard_id"] for item in response.json()}
    assert api_client.get("/api/v1/risks/today").status_code == 401
