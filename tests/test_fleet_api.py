"""Тесты API маршрутов координации флота комбайнов (sprint-27)."""

import uuid
from unittest.mock import MagicMock

from starlette.testclient import TestClient

from app.api.deps import get_current_user
from app.main import create_app
from app.models.user import User


def _mock_agronomist() -> User:
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.email = "agronomist@vitiguard.md"
    user.role = "agronomist"
    user.is_active = True
    return user


def test_fleet_dispatch_endpoint() -> None:
    """Тест POST /api/v1/robot/fleet/dispatch."""
    app = create_app()
    user = _mock_agronomist()
    app.dependency_overrides[get_current_user] = lambda: user

    client = TestClient(app)

    payload = {
        "robots": [
            {
                "robot_id": "harvester-alpha",
                "battery_pct": 88.0,
                "current_speed_mps": 0.4,
                "bunker_capacity_kg": 350.0,
                "current_bunker_kg": 20.0,
                "status": "idle",
            }
        ],
        "rows": [
            {
                "row_id": "r10",
                "row_number": 10,
                "length_m": 250.0,
                "start_point": [47.14, 28.86],
                "end_point": [47.16, 28.86],
                "estimated_yield_kg": 95.0,
                "slope_pct": 1.5,
                "status": "pending",
            }
        ],
    }

    response = client.post("/api/v1/robot/fleet/dispatch", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["assigned_rows_count"] == 1
    assert data["unassigned_rows_count"] == 0
    assert len(data["assignments"]) == 1
    assignment = data["assignments"][0]
    assert assignment["robot_id"] == "harvester-alpha"
    assert assignment["row_id"] == "r10"
    assert assignment["expected_harvest_kg"] == 95.0
    assert assignment["estimated_time_min"] > 0
    assert "fleet_progress" in data


def test_fleet_unloading_and_headland_endpoints() -> None:
    """Тест эндпоинтов разгрузки и блокировки разворотной полосы."""
    app = create_app()
    user = _mock_agronomist()
    app.dependency_overrides[get_current_user] = lambda: user

    client = TestClient(app)

    # 1. Запрос на выгрузку бункера
    unload_res = client.post(
        "/api/v1/robot/fleet/request-unloading",
        json={
            "robot_id": "harvester-beta",
            "current_bunker_kg": 310.0,
            "location": [47.142, 28.864],
        },
    )
    assert unload_res.status_code == 200
    unload_data = unload_res.json()
    assert unload_data["status"] == "scheduled"
    assert unload_data["robot_id"] == "harvester-beta"

    # 2. Блокировка разворотной зоны
    lock_res = client.post(
        "/api/v1/robot/fleet/headland/lock?zone_id=zone_east_1&robot_id=harvester-beta"
    )
    assert lock_res.status_code == 200
    assert lock_res.json()["granted"] is True

    # 3. Попытка другого робота занять ту же зону -> отклонено
    lock_clash_res = client.post(
        "/api/v1/robot/fleet/headland/lock?zone_id=zone_east_1&robot_id=harvester-gamma"
    )
    assert lock_clash_res.status_code == 200
    assert lock_clash_res.json()["granted"] is False

    # 4. Освобождение зоны
    release_res = client.post(
        "/api/v1/robot/fleet/headland/release?zone_id=zone_east_1&robot_id=harvester-beta"
    )
    assert release_res.status_code == 200
    assert release_res.json()["status"] == "released"
