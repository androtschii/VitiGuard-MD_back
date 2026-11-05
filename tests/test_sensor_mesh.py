"""Тесты сервиса сенсорной сети микроклимата и моделей предупреждения заморозков/града (sprint-29)."""

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from app.api.deps import get_current_user
from app.main import create_app
from app.models.user import User
from app.schemas.sensor_mesh import SensorTelemetryDTO
from app.services.sensor_mesh import (
    calculate_dew_point,
    calculate_wet_bulb_temp,
    evaluate_frost_risk,
    evaluate_hail_risk,
    interpolate_spatial_point,
)


def _mock_agronomist() -> User:
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.email = "agronomist@vitiguard.md"
    user.role = "agronomist"
    user.is_active = True
    return user


def test_dew_point_and_wet_bulb_temp() -> None:
    """Проверка расчёта точки росы и температуры влажного термометра."""
    # 20 °C при влажности 50% -> точка росы ~9.3 °C
    dp = calculate_dew_point(temp_c=20.0, humidity_pct=50.0)
    assert dp == pytest.approx(9.27, rel=1e-1)

    # 5 °C при 40% -> смоченный термометр ~0.0..1.5 °C (интенсивное испарение охлаждает лист)
    tw = calculate_wet_bulb_temp(temp_c=5.0, humidity_pct=40.0)
    assert 0.0 <= tw <= 2.0

    # 0 °C при 95% -> tw близка к 0 °C
    tw_saturated = calculate_wet_bulb_temp(temp_c=0.0, humidity_pct=95.0)
    assert tw_saturated == pytest.approx(-0.3, abs=0.5)


def test_interpolate_spatial_point_and_frost_pocket() -> None:
    """Проверка интерполяции микроклимата и обнаружения морозных карманов."""
    now = datetime.now(UTC)
    sensors = [
        SensorTelemetryDTO(
            sensor_id="node-hilltop",
            vineyard_id="v-1",
            timestamp=now,
            lat=47.14,
            lon=28.86,
            elevation_m=160.0,
            canopy_temperature_c=3.5,
            canopy_humidity_pct=70.0,
        ),
        SensorTelemetryDTO(
            sensor_id="node-mid-slope",
            vineyard_id="v-1",
            timestamp=now,
            lat=47.142,
            lon=28.862,
            elevation_m=130.0,
            canopy_temperature_c=1.8,
            canopy_humidity_pct=80.0,
        ),
    ]

    # Точка на склоне
    t_interp, rh_interp, is_pocket = interpolate_spatial_point(
        target_lat=47.141,
        target_lon=28.861,
        target_elevation_m=145.0,
        sensors=sensors,
    )
    assert 1.5 <= t_interp <= 4.0
    assert 65.0 <= rh_interp <= 85.0
    assert is_pocket is False

    # Точка в глубокой низине со стоячим влажным воздухом (морозный карман)
    _, _, is_deep_pocket = interpolate_spatial_point(
        target_lat=47.143,
        target_lon=28.863,
        target_elevation_m=100.0,  # существенно ниже среднего уровня 145м
        sensors=[
            SensorTelemetryDTO(
                sensor_id="s1",
                vineyard_id="v-1",
                timestamp=now,
                lat=47.14,
                lon=28.86,
                elevation_m=160.0,
                canopy_temperature_c=1.0,
                canopy_humidity_pct=92.0,
            )
        ],
    )
    assert is_deep_pocket is True


def test_evaluate_frost_risk_with_thermal_inversion() -> None:
    """Проверка детекции температурной инверсии и рекомендации ветровых машин."""
    now = datetime.now(UTC)
    sensors = [
        SensorTelemetryDTO(
            sensor_id="s-top",
            vineyard_id="purcari-plot",
            timestamp=now,
            lat=47.14,
            lon=28.86,
            elevation_m=175.0,
            canopy_temperature_c=3.8,  # тёплый верхний слой воздуха
            canopy_humidity_pct=60.0,
        ),
        SensorTelemetryDTO(
            sensor_id="s-bottom",
            vineyard_id="purcari-plot",
            timestamp=now,
            lat=47.144,
            lon=28.864,
            elevation_m=110.0,
            canopy_temperature_c=-1.5,  # стекание холодного воздуха в низину
            canopy_humidity_pct=88.0,
        ),
    ]

    report = evaluate_frost_risk("purcari-plot", sensors, phenology_stage="BBCH 09")

    assert report.has_thermal_inversion is True
    assert report.inversion_delta_c == pytest.approx(5.3, rel=1e-1)
    assert report.risk_level in ("moderate", "critical")
    assert report.wet_bulb_temp_c < 0.0
    assert "s-bottom" in report.frost_pocket_sensors

    # При сильной инверсии рекомендуется ветровая машина
    mitigation_text = " ".join(report.recommended_mitigation)
    assert "ветровые машины" in mitigation_text.lower()


def test_evaluate_hail_risk() -> None:
    """Проверка оценки градового риска при резком падении барометрического давления."""
    now = datetime.now(UTC)
    telemetry = SensorTelemetryDTO(
        sensor_id="node-hail-1",
        vineyard_id="v-hail",
        timestamp=now,
        lat=47.14,
        lon=28.86,
        elevation_m=130.0,
        canopy_temperature_c=28.5,
        canopy_humidity_pct=78.0,
        barometric_pressure_hpa=1004.0,
    )

    # Резкий спад 3.5 гПа/час при +28.5 °C и 78% влажности -> экстремальный риск града
    hail_report = evaluate_hail_risk("v-hail", telemetry, pressure_trend_1h_hpa=-3.5)
    assert hail_report.hail_risk_level == "severe"
    assert hail_report.netting_action_required is True
    assert "штормовое предупреждение" in hail_report.warning_message.lower()

    # Спокойная погода (спад 0.3 гПа/ч)
    calm_report = evaluate_hail_risk("v-hail", telemetry, pressure_trend_1h_hpa=-0.3)
    assert calm_report.hail_risk_level == "low"
    assert calm_report.netting_action_required is False


def test_sensors_api_endpoints() -> None:
    """Тест эндпоинтов API /api/v1/sensors."""
    app = create_app()
    user = _mock_agronomist()
    app.dependency_overrides[get_current_user] = lambda: user

    client = TestClient(app)

    telemetry_payload = {
        "sensor_id": "lora-node-01",
        "vineyard_id": "v-cricova",
        "timestamp": "2026-05-04T04:30:00Z",
        "lat": 47.14,
        "lon": 28.86,
        "elevation_m": 125.0,
        "canopy_temperature_c": 1.2,
        "canopy_humidity_pct": 85.0,
        "leaf_wetness_pct": 95.0,
        "soil": {
            "depth_10cm_pct": 28.5,
            "depth_30cm_pct": 32.0,
            "depth_60cm_pct": 38.0,
            "soil_temp_10cm_c": 11.5,
        },
    }

    # 1. Приём телеметрии
    res_tel = client.post("/api/v1/sensors/telemetry", json=telemetry_payload)
    assert res_tel.status_code == 200
    assert res_tel.json()["status"] == "recorded"

    # 2. Сводка сенсорной сети
    res_mesh = client.post(
        "/api/v1/sensors/mesh-status",
        json={"vineyard_id": "v-cricova", "sensors": [telemetry_payload]},
    )
    assert res_mesh.status_code == 200
    mesh_data = res_mesh.json()
    assert mesh_data["active_sensors_count"] == 1
    assert len(mesh_data["spatial_grid"]) == 9  # сетка 3х3

    # 3. Оценка риска заморозков
    res_frost = client.post(
        "/api/v1/sensors/frost-risk",
        json={
            "vineyard_id": "v-cricova",
            "sensors": [telemetry_payload],
            "phenology_stage": "BBCH 09",
        },
    )
    assert res_frost.status_code == 200
    assert "wet_bulb_temp_c" in res_frost.json()

    # 4. Оценка риска града
    hail_telemetry = {
        **telemetry_payload,
        "canopy_temperature_c": 27.5,
        "canopy_humidity_pct": 75.0,
    }
    res_hail = client.post(
        "/api/v1/sensors/hail-risk",
        json={
            "vineyard_id": "v-cricova",
            "telemetry": hail_telemetry,
            "pressure_drop_1h_hpa": -2.5,
        },
    )
    assert res_hail.status_code == 200
    assert res_hail.json()["hail_risk_level"] in ("moderate", "high", "severe")
