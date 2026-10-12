from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

URL = "/api/v1/vineyards"
RunSql = Callable[[str], list[tuple[object, ...]]]
CREDENTIALS = {"email": "grower@example.md", "password": "secret123"}
RING = [
    [28.86, 47.14],
    [28.862, 47.14],
    [28.862, 47.141],
    [28.86, 47.141],
    [28.86, 47.14],
]


@pytest.fixture
def headers(api_client: TestClient) -> dict[str, str]:
    api_client.post("/api/v1/auth/register", json={**CREDENTIALS, "full_name": "Ион"})
    token = api_client.post("/api/v1/auth/login", json=CREDENTIALS).json()[
        "access_token"
    ]
    return {"Authorization": f"Bearer {token}"}


def test_polygon_is_saved_as_multipolygon_of_current_user(
    api_client: TestClient, run_sql: RunSql, headers: dict[str, str]
) -> None:
    response = api_client.post(
        URL,
        headers=headers,
        json={
            "name": "  Крикова, участок 1  ",
            "geometry": {"type": "Polygon", "coordinates": [RING]},
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Крикова, участок 1"
    assert body["geometry"] == {"type": "MultiPolygon", "coordinates": [[RING]]}
    assert body["created_at"]
    rows = run_sql(
        "SELECT u.email, GeometryType(v.geom), ST_SRID(v.geom)"
        f" FROM vineyards v JOIN users u ON u.id = v.owner_id WHERE v.id = '{body['id']}'"
    )
    assert rows == [("grower@example.md", "MULTIPOLYGON", 4326)]


def test_multipolygon_keeps_all_parts(
    api_client: TestClient, run_sql: RunSql, headers: dict[str, str]
) -> None:
    second = [[x + 0.01, y] for x, y in RING]

    response = api_client.post(
        URL,
        headers=headers,
        json={
            "name": "Два контура",
            "geometry": {"type": "MultiPolygon", "coordinates": [[RING], [second]]},
        },
    )

    assert response.status_code == 201
    assert len(response.json()["geometry"]["coordinates"]) == 2
    assert run_sql(
        f"SELECT ST_NumGeometries(geom) FROM vineyards WHERE id = '{response.json()['id']}'"
    ) == [(2,)]


def test_requires_login(api_client: TestClient) -> None:
    response = api_client.post(
        URL,
        json={
            "name": "Участок",
            "geometry": {"type": "Polygon", "coordinates": [RING]},
        },
    )

    assert response.status_code == 401


def test_unclosed_ring_is_rejected(
    api_client: TestClient, run_sql: RunSql, headers: dict[str, str]
) -> None:
    response = api_client.post(
        URL,
        headers=headers,
        json={
            "name": "Участок",
            "geometry": {"type": "Polygon", "coordinates": [RING[:-1]]},
        },
    )

    assert response.status_code == 422
    assert run_sql("SELECT count(*) FROM vineyards") == [(0,)]
