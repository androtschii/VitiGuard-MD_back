from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.services.anomaly import Observation, detect_anomalies

RunSql = Callable[[str], list[tuple[object, ...]]]
START = datetime(2026, 6, 1, 9, 17, tzinfo=UTC)
# Шум индекса между чистыми снимками — около сотой
NOISE = (0.0, 0.01, -0.01, 0.005, -0.005, 0.0)


def season(
    drops: dict[int, float] | None = None, days: int = 5, count: int = 18
) -> list[Observation]:
    """Плато NDVI 0,7 со снимками каждые days дней и просадками по номерам снимков."""
    drops = drops or {}
    return [
        Observation(
            START + timedelta(days=days * index),
            0.7 + NOISE[index % len(NOISE)] - drops.get(index, 0.0),
        )
        for index in range(count)
    ]


def peers_of(series: list[Observation], shift: float = 0.0) -> list[Observation]:
    # Три соседних участка снимаются в те же дни
    return [
        Observation(item.acquired_at, 0.68 + offset - shift)
        for item in series
        for offset in (0.0, 0.01, 0.02)
    ]


def test_quiet_season_has_no_alarms() -> None:
    series = season()

    assert detect_anomalies(series, peers_of(series)) == []


def test_single_drop_is_suspected() -> None:
    series = season({8: 0.2})

    [anomaly] = detect_anomalies(series, peers_of(series))

    assert anomaly.acquired_at == series[8].acquired_at
    assert anomaly.level == "suspected"
    assert anomaly.below_norm == pytest.approx(0.2, abs=0.015)
    assert anomaly.below_peers is not None
    assert anomaly.below_peers == pytest.approx(0.19, abs=0.015)
    assert anomaly.z >= 2


def test_two_drops_in_a_row_are_confirmed() -> None:
    series = season({8: 0.2, 9: 0.18})

    levels = [a.level for a in detect_anomalies(series, peers_of(series))]

    assert levels == ["suspected", "confirmed"]


def test_small_drop_is_not_an_alarm() -> None:
    series = season({8: 0.06})

    assert detect_anomalies(series, peers_of(series)) == []


def test_drop_shared_by_neighbours_is_not_an_alarm() -> None:
    # Засуха или град во всём районе: соседи просели так же
    series = season({8: 0.2})
    peers = [
        Observation(item.acquired_at, item.ndvi - 0.02)
        for item in series
        for _ in range(3)
    ]

    assert detect_anomalies(series, peers) == []


def test_without_neighbours_own_norm_is_enough() -> None:
    series = season({8: 0.2})

    [anomaly] = detect_anomalies(series, [])

    assert anomaly.below_peers is None


def test_short_series_gives_nothing() -> None:
    assert detect_anomalies(season({1: 0.3}, count=3), []) == []
    assert detect_anomalies([], []) == []


def test_lonely_scan_has_no_norm() -> None:
    # Снимки в 30 днях друг от друга: у каждого нет соседей по окну ±20 дней
    series = season({4: 0.3}, days=30, count=6)

    assert detect_anomalies(series, []) == []


def login(api_client: TestClient, email: str) -> dict[str, str]:
    credentials = {"email": email, "password": "secret123"}
    api_client.post("/api/v1/auth/register", json={**credentials, "full_name": "Ион"})
    token = api_client.post("/api/v1/auth/login", json=credentials).json()[
        "access_token"
    ]
    return {"Authorization": f"Bearer {token}"}


def create_vineyard(
    api_client: TestClient, headers: dict[str, str], west: float
) -> str:
    ring = [
        [west, 47.14],
        [west + 0.002, 47.14],
        [west + 0.002, 47.141],
        [west, 47.14],
    ]
    response = api_client.post(
        "/api/v1/vineyards",
        headers=headers,
        json={
            "name": f"Участок {west}",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        },
    )
    vineyard_id: str = response.json()["id"]
    return vineyard_id


def add_scans(
    run_sql: RunSql, vineyard_id: str, series: list[Observation], cloud: float = 0
) -> None:
    for index, item in enumerate(series):
        run_sql(
            "WITH scan AS (INSERT INTO satellite_scans (vineyard_id, product_id, "
            "acquired_at, cloud_cover, parcel_cloud_cover, status) VALUES "
            f"('{vineyard_id}', 'S2_{vineyard_id[:8]}_{index}_{cloud}', "
            f"'{item.acquired_at.isoformat()}', 0, {cloud}, 'completed') RETURNING id) "
            "INSERT INTO vegetation_indices (scan_id, index_type, mean_value, "
            f"min_value, max_value, std_value) SELECT id, 'ndvi', {item.ndvi}, "
            f"{item.ndvi}, {item.ndvi}, 0 FROM scan"
        )


@pytest.mark.integration
def test_anomalies_endpoint(api_client: TestClient, run_sql: RunSql) -> None:
    headers = login(api_client, "grower@example.md")
    vineyard = create_vineyard(api_client, headers, 28.860)
    neighbour = create_vineyard(api_client, headers, 28.870)
    damaged = season({8: 0.2, 9: 0.2})
    add_scans(run_sql, vineyard, damaged)
    add_scans(run_sql, neighbour, season())
    # Снимок под облаками над участком не учитывается, как бы низко ни был NDVI
    add_scans(
        run_sql,
        vineyard,
        [Observation(damaged[4].acquired_at + timedelta(hours=1), 0.2)],
        cloud=45,
    )
    url = f"/api/v1/vineyards/{vineyard}/vegetation/anomalies"

    anomalies = api_client.get(url, headers=headers).json()

    assert [a["level"] for a in anomalies] == ["suspected", "confirmed"]
    assert anomalies[0]["acquired_at"].startswith(
        damaged[8].acquired_at.date().isoformat()
    )
    # У соседа меньше трёх участков-соседей в те же дни: сравнение только с нормой
    assert anomalies[0]["below_peers"] is None
    assert anomalies[0]["below_norm"] == pytest.approx(0.2, abs=0.015)
    assert (
        api_client.get(
            f"/api/v1/vineyards/{neighbour}/vegetation/anomalies", headers=headers
        ).json()
        == []
    )
    stranger = login(api_client, "stranger@example.md")
    assert api_client.get(url, headers=stranger).status_code == 404
