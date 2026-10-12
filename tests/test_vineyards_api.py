from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from httpx2 import Response

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


def login_as(api_client: TestClient, email: str) -> dict[str, str]:
    credentials = {"email": email, "password": "secret123"}
    api_client.post("/api/v1/auth/register", json={**credentials, "full_name": "Сосед"})
    token = api_client.post("/api/v1/auth/login", json=credentials).json()[
        "access_token"
    ]
    return {"Authorization": f"Bearer {token}"}


def create(
    api_client: TestClient, headers: dict[str, str], name: str, slot: int = 0
) -> str:
    # Участки одного владельца не накладываются: каждый — на своём месте
    ring = [[x + 0.003 * slot, y] for x, y in RING]
    response = api_client.post(
        URL,
        headers=headers,
        json={"name": name, "geometry": {"type": "Polygon", "coordinates": [ring]}},
    )
    vineyard_id: str = response.json()["id"]
    return vineyard_id


def test_list_contains_only_own_vineyards_sorted_by_name(
    api_client: TestClient, headers: dict[str, str]
) -> None:
    for slot, name in enumerate(["Криково", "Пуркарь", "Милештий Мици"]):
        create(api_client, headers, name, slot)
    create(api_client, login_as(api_client, "neighbour@example.md"), "Чужой участок")

    response = api_client.get(URL, headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert [item["name"] for item in body["items"]] == [
        "Криково",
        "Милештий Мици",
        "Пуркарь",
    ]
    assert body["total"] == 3
    assert body["items"][0]["geometry"] == {
        "type": "MultiPolygon",
        "coordinates": [[RING]],
    }


def test_list_is_paginated(api_client: TestClient, headers: dict[str, str]) -> None:
    for number in range(5):
        create(api_client, headers, f"Участок {number}", number)

    body = api_client.get(URL, headers=headers, params={"page": 2, "size": 2}).json()

    assert [item["name"] for item in body["items"]] == ["Участок 2", "Участок 3"]
    assert (body["total"], body["page"], body["size"], body["pages"]) == (5, 2, 2, 3)


def test_empty_list(api_client: TestClient, headers: dict[str, str]) -> None:
    body = api_client.get(URL, headers=headers).json()

    assert (body["items"], body["total"], body["pages"]) == ([], 0, 0)


def test_vineyard_details(api_client: TestClient, headers: dict[str, str]) -> None:
    vineyard_id = create(api_client, headers, "Криково")

    response = api_client.get(f"{URL}/{vineyard_id}", headers=headers)

    assert response.status_code == 200
    assert response.json()["id"] == vineyard_id
    assert response.json()["name"] == "Криково"


def test_foreign_vineyard_looks_missing(
    api_client: TestClient, headers: dict[str, str]
) -> None:
    foreign_id = create(
        api_client, login_as(api_client, "neighbour@example.md"), "Чужой участок"
    )

    foreign = api_client.get(f"{URL}/{foreign_id}", headers=headers)
    missing = api_client.get(
        f"{URL}/01900000-0000-7000-8000-000000000000", headers=headers
    )

    assert foreign.status_code == missing.status_code == 404
    assert foreign.json() == missing.json()


def test_list_and_details_require_login(api_client: TestClient) -> None:
    assert api_client.get(URL).status_code == 401
    assert (
        api_client.get(f"{URL}/01900000-0000-7000-8000-000000000000").status_code == 401
    )


def post_ring(
    api_client: TestClient, headers: dict[str, str], name: str, ring: list[list[float]]
) -> Response:
    return api_client.post(
        URL,
        headers=headers,
        json={"name": name, "geometry": {"type": "Polygon", "coordinates": [ring]}},
    )


def test_self_intersecting_outline_is_rejected(
    api_client: TestClient, run_sql: RunSql, headers: dict[str, str]
) -> None:
    # «Восьмёрка»: стороны пересекаются в середине
    bowtie = [
        [28.86, 47.14],
        [28.862, 47.141],
        [28.862, 47.14],
        [28.86, 47.141],
        [28.86, 47.14],
    ]

    response = post_ring(api_client, headers, "Восьмёрка", bowtie)

    assert response.status_code == 400
    assert response.json()["detail"] == "Контур пересекает сам себя"
    assert run_sql("SELECT count(*) FROM vineyards") == [(0,)]


def test_overlap_with_own_vineyard_is_rejected(
    api_client: TestClient, headers: dict[str, str]
) -> None:
    create(api_client, headers, "Криково")
    shifted = [[x + 0.001, y] for x, y in RING]

    response = post_ring(api_client, headers, "Внахлёст", shifted)

    assert response.status_code == 409
    assert response.json()["detail"] == "Контур накладывается на участок «Криково»"


def test_neighbour_sharing_border_is_accepted(
    api_client: TestClient, headers: dict[str, str]
) -> None:
    create(api_client, headers, "Криково")
    # Сдвиг ровно на ширину участка: общая сторона, но не общая земля
    neighbour = [[x + 0.002, y] for x, y in RING]

    assert post_ring(api_client, headers, "Сосед по меже", neighbour).status_code == 201


def test_other_owners_vineyards_do_not_block(
    api_client: TestClient, headers: dict[str, str]
) -> None:
    create(api_client, login_as(api_client, "neighbour@example.md"), "Арендованный")

    assert post_ring(api_client, headers, "Тот же контур", RING).status_code == 201
