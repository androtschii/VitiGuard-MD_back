import asyncio
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.integrations.overpass import OsmVineyard, OverpassClient, OverpassResult
from app.tasks import osm as osm_tasks
from app.worker.celery_app import celery_app

pytestmark = pytest.mark.integration

RunSql = Callable[[str], list[tuple[object, ...]]]
SQUARE = [
    [28.86, 47.14],
    [28.862, 47.14],
    [28.862, 47.141],
    [28.86, 47.141],
    [28.86, 47.14],
]
# «Восьмёрка»: PostGIS превращает её в два треугольника
BOWTIE = [
    [28.9, 47.14],
    [28.902, 47.141],
    [28.902, 47.14],
    [28.9, 47.141],
    [28.9, 47.14],
]


def record(
    osm_id: str, ring: list[list[float]], name: str | None = None
) -> OsmVineyard:
    return OsmVineyard(osm_id, name, {"type": "MultiPolygon", "coordinates": [[ring]]})


@pytest.fixture
def run_import(
    migrated_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    run_sql: RunSql,
) -> Iterator[Callable[[list[OsmVineyard], int], dict[str, int]]]:
    """Импорт через ту же функцию, что и задача Celery, но без сети: Overpass
    подменяется готовым результатом."""

    def run(records: list[OsmVineyard], skipped: int = 0) -> dict[str, int]:
        async def fake(_: OverpassClient) -> OverpassResult:
            return OverpassResult(records, skipped)

        monkeypatch.setattr(OverpassClient, "vineyards", fake)
        settings = Settings(
            _env_file=None,
            environment="test",
            database_url=migrated_database_url,  # type: ignore[arg-type]
        )
        return asyncio.run(osm_tasks.run_import(settings))

    yield run
    run_sql("TRUNCATE osm_vineyards")


def login(api_client: TestClient, email: str) -> dict[str, str]:
    credentials = {"email": email, "password": "secret123"}
    api_client.post("/api/v1/auth/register", json={**credentials, "full_name": "Тест"})
    token = api_client.post("/api/v1/auth/login", json=credentials).json()[
        "access_token"
    ]
    return {"Authorization": f"Bearer {token}"}


def test_import_syncs_with_fresh_export(
    run_import: Callable[..., dict[str, int]], run_sql: RunSql
) -> None:
    first = run_import([record("way/1", SQUARE, "Cricova"), record("way/2", BOWTIE)], 4)

    assert first == {"imported": 2, "deleted": 0, "skipped": 4}
    # Самопересекающийся контур исправлен, а не отброшен
    assert run_sql(
        "SELECT osm_id, name, ST_IsValid(geom), GeometryType(geom)"
        " FROM osm_vineyards ORDER BY osm_id"
    ) == [
        ("way/1", "Cricova", True, "MULTIPOLYGON"),
        ("way/2", None, True, "MULTIPOLYGON"),
    ]

    moved = [[x + 0.001, y] for x, y in SQUARE]
    second = run_import([record("way/1", moved, "Cricova Nord")])

    assert second == {"imported": 1, "deleted": 1, "skipped": 0}
    assert run_sql(
        "SELECT osm_id, name, round(ST_XMin(geom)::numeric, 4)::float FROM osm_vineyards"
    ) == [("way/1", "Cricova Nord", 28.861)]


def test_layer_for_map_window(
    api_client: TestClient, run_import: Callable[..., dict[str, int]]
) -> None:
    run_import([record("way/1", SQUARE, "Cricova"), record("way/2", BOWTIE)])
    headers = login(api_client, "grower@example.md")

    body = api_client.get(
        "/api/v1/osm/vineyards",
        headers=headers,
        params={"bbox": "28.85,47.13,28.87,47.15"},
    ).json()

    [feature] = body["features"]
    assert feature["properties"]["osm_id"] == "way/1"
    assert feature["properties"]["area_ha"] == pytest.approx(1.69, abs=0.01)
    assert feature["geometry"]["type"] == "MultiPolygon"


@pytest.mark.parametrize(
    ("bbox", "message"),
    [
        ("28.85,47.13,28.87", "четыре числа"),
        ("a,b,c,d", "четыре числа"),
        ("28.87,47.13,28.85,47.15", "заданы неверно"),
        ("27,46,29,48", "больше 1°"),
    ],
)
def test_layer_validates_bbox(api_client: TestClient, bbox: str, message: str) -> None:
    headers = login(api_client, "grower@example.md")

    response = api_client.get(
        "/api/v1/osm/vineyards", headers=headers, params={"bbox": bbox}
    )

    assert response.status_code == 400
    assert message in response.json()["detail"]


def test_layer_requires_login(api_client: TestClient) -> None:
    response = api_client.get(
        "/api/v1/osm/vineyards", params={"bbox": "28,47,28.1,47.1"}
    )

    assert response.status_code == 401


def test_only_admin_starts_import(
    api_client: TestClient, run_sql: RunSql, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[str] = []

    class FakeTask:
        id = "task-1"

    def send_task(name: str) -> FakeTask:
        sent.append(name)
        return FakeTask()

    monkeypatch.setattr(celery_app, "send_task", send_task)
    grower = login(api_client, "grower@example.md")
    admin = login(api_client, "admin@example.md")
    run_sql("UPDATE users SET role = 'admin' WHERE email = 'admin@example.md'")

    assert (
        api_client.post("/api/v1/osm/vineyards/import", headers=grower).status_code
        == 403
    )
    started = api_client.post("/api/v1/osm/vineyards/import", headers=admin)

    assert started.status_code == 202
    assert started.json() == {"task_id": "task-1"}
    assert sent == ["osm.import_vineyards"]


def create_vineyard(api_client: TestClient, headers: dict[str, str], ring: Any) -> str:
    response = api_client.post(
        "/api/v1/vineyards",
        headers=headers,
        json={
            "name": "Участок",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        },
    )
    vineyard_id: str = response.json()["id"]
    return vineyard_id


def test_vineyard_context(
    api_client: TestClient, run_import: Callable[..., dict[str, int]]
) -> None:
    headers = login(api_client, "grower@example.md")
    inside = create_vineyard(api_client, headers, SQUARE)
    far = create_vineyard(api_client, headers, [[x + 0.5, y] for x, y in SQUARE])
    # OSM-виноградник совпадает с первым участком, второй — в 1,5 км от первого
    run_import(
        [record("way/1", SQUARE), record("way/2", [[x + 0.02, y] for x, y in SQUARE])]
    )

    near = api_client.get(f"/api/v1/vineyards/{inside}/context", headers=headers).json()
    alone = api_client.get(f"/api/v1/vineyards/{far}/context", headers=headers).json()

    assert near["osm_overlap_ratio"] == pytest.approx(1.0, abs=0.001)
    assert near["osm_vineyards_within_1km"] == 1
    assert near["nearest_osm_distance_m"] == 0
    assert alone == {
        "osm_overlap_ratio": 0,
        "osm_vineyards_within_1km": 0,
        "nearest_osm_distance_m": pytest.approx(36_000, rel=0.05),
        "district": alone["district"],
    }


def test_context_of_foreign_vineyard_looks_missing(api_client: TestClient) -> None:
    owner = login(api_client, "owner@example.md")
    vineyard = create_vineyard(api_client, owner, SQUARE)
    stranger = login(api_client, "stranger@example.md")

    response = api_client.get(f"/api/v1/vineyards/{vineyard}/context", headers=stranger)

    assert response.status_code == 404
