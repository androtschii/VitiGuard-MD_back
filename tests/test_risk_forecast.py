import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from risk_model import build_risk_model

from app.core.config import Settings
from app.ml.risk_model import RiskModel
from app.services.risk import local_today, season_year
from app.services.risk_features import FEATURES

CREDENTIALS = {"email": "grower@example.md", "password": "secret123"}
POLYGON = {
    "type": "MultiPolygon",
    "coordinates": [
        [[[28.8, 47.0], [28.81, 47.0], [28.81, 47.01], [28.8, 47.01], [28.8, 47.0]]]
    ],
}
RunSql = Callable[[str], list[tuple[object, ...]]]


def test_model_reads_probability_and_threshold(tmp_path: Path) -> None:
    model = RiskModel(build_risk_model(tmp_path / "risk.onnx", threshold=0.4))
    rain = FEATURES.index("rain_2d")

    wet = [0.0] * len(FEATURES)
    wet[rain] = 30.0
    dry = [0.0] * len(FEATURES)

    assert model.name == "lightgbm"
    assert model.threshold == 0.4
    assert model.probability(wet) > 0.99
    assert model.probability(dry) < 0.01


def test_model_rejects_other_features(tmp_path: Path) -> None:
    path = build_risk_model(tmp_path / "risk.onnx")
    meta = json.loads(path.with_suffix(".json").read_text())
    path.with_suffix(".json").write_text(
        json.dumps({**meta, "features": list(reversed(FEATURES))})
    )

    with pytest.raises(ValueError, match="не совпадают"):
        RiskModel(path)


def use_model(client: TestClient, path: Path | None) -> None:
    app = client.app
    settings: Settings = app.state.settings  # type: ignore[attr-defined]
    app.state.settings = settings.model_copy(  # type: ignore[attr-defined]
        update={"risk_model_path": path or Path("no-such-model.onnx")}
    )
    app.state.risk_model = None  # type: ignore[attr-defined]


def sign_in_with_plot(client: TestClient) -> tuple[dict[str, str], str]:
    client.post("/api/v1/auth/register", json={**CREDENTIALS, "full_name": "Ион"})
    token = client.post("/api/v1/auth/login", json=CREDENTIALS).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    vineyard = client.post(
        "/api/v1/vineyards",
        headers=headers,
        json={"name": "Криково", "geometry": POLYGON},
    ).json()["id"]
    return headers, str(vineyard)


def fill_weather(
    run_sql: RunSql, vineyard: str, days: int, rain: float, month: int = 5
) -> None:
    # Полные сутки по часам этого сезона до 25-го числа; последние двое суток —
    # с дождём. Месяц по умолчанию — май, внутри окна «трёх десяток»
    end = datetime(season_year(local_today()), month, 25, tzinfo=UTC)
    start = end - timedelta(days=days)
    run_sql(
        "INSERT INTO weather_data (vineyard_id, observed_at, temperature_c, "
        "relative_humidity, precipitation_mm) "
        f"SELECT '{vineyard}', moment, 18, 80, "
        f"CASE WHEN moment >= '{(end - timedelta(days=2)).isoformat()}' "
        f"THEN {rain} ELSE 0 END "
        f"FROM generate_series('{start.isoformat()}'::timestamptz, "
        f"'{(end - timedelta(hours=1)).isoformat()}'::timestamptz, interval '1 hour') AS moment"
    )


@pytest.mark.integration
def test_without_model_service_unavailable(api_client: TestClient) -> None:
    use_model(api_client, None)
    headers, vineyard = sign_in_with_plot(api_client)

    response = api_client.get(
        f"/api/v1/vineyards/{vineyard}/risks/next-day", headers=headers
    )

    assert response.status_code == 503


@pytest.mark.integration
def test_short_history_gives_no_forecast(
    api_client: TestClient, run_sql: RunSql, tmp_path: Path
) -> None:
    use_model(api_client, build_risk_model(tmp_path / "risk.onnx"))
    headers, vineyard = sign_in_with_plot(api_client)
    fill_weather(run_sql, vineyard, days=5, rain=1)

    response = api_client.get(
        f"/api/v1/vineyards/{vineyard}/risks/next-day", headers=headers
    )

    assert response.status_code == 200
    assert response.json() is None


@pytest.mark.integration
def test_rain_gives_high_next_day_risk(
    api_client: TestClient, run_sql: RunSql, tmp_path: Path
) -> None:
    use_model(api_client, build_risk_model(tmp_path / "risk.onnx", threshold=0.5))
    headers, vineyard = sign_in_with_plot(api_client)
    fill_weather(run_sql, vineyard, days=20, rain=1)

    body = api_client.get(
        f"/api/v1/vineyards/{vineyard}/risks/next-day", headers=headers
    ).json()

    assert body["day"] == f"{season_year(local_today())}-05-25"
    assert body["high"] is True
    assert body["probability"] > 0.99
    assert body["threshold"] == 0.5
    assert body["model"] == "lightgbm"
    # Модель загружается один раз и дальше берётся из памяти
    again = api_client.get(
        f"/api/v1/vineyards/{vineyard}/risks/next-day", headers=headers
    )
    assert again.json() == body


@pytest.mark.integration
def test_outside_three_tens_window_gives_no_forecast(
    api_client: TestClient, run_sql: RunSql, tmp_path: Path
) -> None:
    use_model(api_client, build_risk_model(tmp_path / "risk.onnx"))
    headers, vineyard = sign_in_with_plot(api_client)
    fill_weather(run_sql, vineyard, days=20, rain=30, month=8)

    response = api_client.get(
        f"/api/v1/vineyards/{vineyard}/risks/next-day", headers=headers
    )

    assert response.json() is None
