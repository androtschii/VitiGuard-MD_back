from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

RunSql = Callable[[str], list[tuple[object, ...]]]
# Около 1,7 га в Кишинёве (у Крикова)
SQUARE = [
    [28.86, 47.14],
    [28.862, 47.14],
    [28.862, 47.141],
    [28.86, 47.141],
    [28.86, 47.14],
]
# Тот же квадрат в 80 км западнее
WEST = [[x - 1.0, y] for x, y in SQUARE]


def login(api_client: TestClient, email: str) -> dict[str, str]:
    credentials = {"email": email, "password": "secret123"}
    api_client.post("/api/v1/auth/register", json={**credentials, "full_name": "Тест"})
    token = api_client.post("/api/v1/auth/login", json=credentials).json()[
        "access_token"
    ]
    return {"Authorization": f"Bearer {token}"}


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


@pytest.fixture
def osm_vineyard(run_sql: RunSql) -> Iterator[Callable[[str, Any], None]]:
    def add(osm_id: str, ring: Any) -> None:
        coordinates = ",".join(f"{x} {y}" for x, y in ring)
        run_sql(
            "INSERT INTO osm_vineyards (osm_id, geom) VALUES "
            f"('{osm_id}', ST_GeomFromText('MULTIPOLYGON((({coordinates})))', 4326))"
        )

    yield add
    run_sql("TRUNCATE osm_vineyards")


def by_iso(stats: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {district["iso"]: district for district in stats}


def test_districts_loaded_by_migration(run_sql: RunSql) -> None:
    assert run_sql("SELECT count(*) FROM districts") == [(37,)]
    assert run_sql("SELECT name FROM districts WHERE iso = 'MD-CU'") == [("Chișinău",)]


def test_stats_require_login(api_client: TestClient) -> None:
    assert api_client.get("/api/v1/regions/stats").status_code == 401


def test_stats_without_vineyards(api_client: TestClient) -> None:
    headers = login(api_client, "grower@example.md")

    stats = api_client.get("/api/v1/regions/stats", headers=headers).json()

    assert len(stats) == 37
    # По алфавиту с учётом диакритики (правила сортировки базы, а не Python)
    assert [d["name"] for d in stats[:3]] == ["Anenii Noi", "Bălți", "Basarabeasca"]
    chisinau = by_iso(stats)["MD-CU"]
    # Муниципий Кишинёв — около 563 км²
    assert chisinau["area_ha"] == pytest.approx(56_300, rel=0.05)
    assert {
        key: chisinau[key]
        for key in (
            "osm_vineyards",
            "osm_vineyard_area_ha",
            "osm_vineyard_share",
            "my_vineyards",
            "my_vineyard_area_ha",
        )
    } == dict.fromkeys(
        (
            "osm_vineyards",
            "osm_vineyard_area_ha",
            "osm_vineyard_share",
            "my_vineyards",
            "my_vineyard_area_ha",
        ),
        0,
    )


def test_stats_count_osm_and_own_vineyards(
    api_client: TestClient, osm_vineyard: Callable[[str, Any], None]
) -> None:
    grower = login(api_client, "grower@example.md")
    neighbour = login(api_client, "neighbour@example.md")
    create_vineyard(api_client, grower, SQUARE)
    create_vineyard(api_client, grower, [[x + 0.01, y] for x, y in SQUARE])
    # Участки других пользователей в статистику не попадают
    create_vineyard(api_client, neighbour, SQUARE)
    osm_vineyard("way/1", SQUARE)
    osm_vineyard("way/2", WEST)

    stats = by_iso(api_client.get("/api/v1/regions/stats", headers=grower).json())

    chisinau = stats["MD-CU"]
    assert chisinau["osm_vineyards"] == 1
    assert chisinau["osm_vineyard_area_ha"] == pytest.approx(1.7, abs=0.1)
    assert chisinau["osm_vineyard_share"] == pytest.approx(1.7 / 56_300, abs=1e-4)
    assert chisinau["my_vineyards"] == 2
    assert chisinau["my_vineyard_area_ha"] == pytest.approx(3.4, abs=0.1)
    others = [d for iso, d in stats.items() if iso != "MD-CU"]
    assert sum(d["osm_vineyards"] for d in others) == 1
    assert sum(d["my_vineyards"] for d in others) == 0


def test_context_names_district(api_client: TestClient) -> None:
    headers = login(api_client, "grower@example.md")
    inside = create_vineyard(api_client, headers, SQUARE)
    # Чёрное море: ни в одном районе
    abroad = create_vineyard(
        api_client, headers, [[x + 2.0, y - 2.5] for x, y in SQUARE]
    )

    def district(vineyard: str) -> Any:
        url = f"/api/v1/vineyards/{vineyard}/context"
        return api_client.get(url, headers=headers).json()["district"]

    assert district(inside) == {"iso": "MD-CU", "name": "Chișinău"}
    assert district(abroad) is None
