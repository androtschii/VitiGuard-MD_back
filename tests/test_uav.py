"""Тесты API маршрутов анализа мультиспектральных данных БПЛА (sprint-30)."""

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


def test_uav_analyze_transects_endpoint() -> None:
    """Тест POST /api/v1/uav/analyze-transects."""
    app = create_app()
    user = _mock_agronomist()
    app.dependency_overrides[get_current_user] = lambda: user

    client = TestClient(app)

    payload = {
        "field_id": "plot_purcari_merlot",
        "variety_name": "Merlot",
        "nominal_vine_spacing_m": 1.25,
        "canopy_ndvi_threshold": 0.4,
        "transects": [
            {
                "row_id": "row_01",
                "row_length_m": 20.0,
                "start_point": [47.14, 28.86],
                "end_point": [47.142, 28.86],
                "pixels": [
                    {
                        "offset_m": 0.0,
                        "green": 0.1,
                        "red": 0.08,
                        "red_edge": 0.2,
                        "nir": 0.55,
                    },
                    {
                        "offset_m": 2.0,
                        "green": 0.1,
                        "red": 0.08,
                        "red_edge": 0.2,
                        "nir": 0.55,
                    },
                    # Пропуск лозы 4 м
                    {
                        "offset_m": 4.0,
                        "green": 0.15,
                        "red": 0.25,
                        "red_edge": 0.28,
                        "nir": 0.3,
                    },
                    {
                        "offset_m": 6.0,
                        "green": 0.15,
                        "red": 0.25,
                        "red_edge": 0.28,
                        "nir": 0.3,
                    },
                    # Возврат в норму
                    {
                        "offset_m": 8.0,
                        "green": 0.1,
                        "red": 0.08,
                        "red_edge": 0.2,
                        "nir": 0.55,
                    },
                    {
                        "offset_m": 12.0,
                        "green": 0.1,
                        "red": 0.08,
                        "red_edge": 0.2,
                        "nir": 0.55,
                    },
                ],
            }
        ],
    }

    res = client.post("/api/v1/uav/analyze-transects", json=payload)
    assert res.status_code == 200, res.text
    data = res.json()

    assert data["field_id"] == "plot_purcari_merlot"
    assert data["gaps_count"] == 1
    assert data["total_missing_vines"] >= 1
    gap = data["missing_gaps"][0]
    assert gap["row_id"] == "row_01"
    assert "geojson_gaps" in data


def test_uav_replanting_order_endpoint() -> None:
    """Тест POST /api/v1/uav/replanting-order."""
    app = create_app()
    user = _mock_agronomist()
    app.dependency_overrides[get_current_user] = lambda: user

    client = TestClient(app)

    payload = {
        "field_id": "plot_purcari_merlot",
        "variety_name": "Merlot",
        "total_nominal_vines": 500,
        "gaps": [
            {
                "gap_id": "row_01_gap_1",
                "row_id": "row_01",
                "start_offset_m": 4.0,
                "end_offset_m": 8.0,
                "gap_length_m": 4.0,
                "estimated_missing_vines": 3,
                "gps_coordinate": [47.1405, 28.86],
            }
        ],
    }

    res = client.post("/api/v1/uav/replanting-order", json=payload)
    assert res.status_code == 200, res.text
    data = res.json()

    assert data["field_id"] == "plot_purcari_merlot"
    assert data["total_missing_vines"] == 3
    assert data["replanting_order_vines_count"] == 4  # 3 * 1.05 = 3.15 -> 4
    assert data["mortality_rate_pct"] == 0.6
    assert data["geojson_missing_points"]["type"] == "FeatureCollection"


def test_uav_indices_preview_endpoint() -> None:
    """Тест POST /api/v1/uav/indices-preview."""
    app = create_app()
    user = _mock_agronomist()
    app.dependency_overrides[get_current_user] = lambda: user

    client = TestClient(app)

    res = client.post(
        "/api/v1/uav/indices-preview",
        json={
            "offset_m": 1.0,
            "green": 0.12,
            "red": 0.08,
            "red_edge": 0.20,
            "nir": 0.55,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert 0.70 <= data["ndvi"] <= 0.80
    assert 0.40 <= data["ndre"] <= 0.50
    assert 0.60 <= data["gndvi"] <= 0.70
