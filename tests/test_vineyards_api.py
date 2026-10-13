import io
import json
import zipfile
from collections.abc import Callable

import pytest
import shapefile
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


def test_area_is_calculated_on_ellipsoid(
    api_client: TestClient, run_sql: RunSql, headers: dict[str, str]
) -> None:
    # 0,002° × 0,001° у Крикова: около 152 × 111 м
    body = post_ring(api_client, headers, "Криково", RING).json()

    assert body["area_ha"] == pytest.approx(1.69, abs=0.01)
    [(expected,)] = run_sql(
        "SELECT ST_Area(geom::geography) / 10000 FROM vineyards"
        f" WHERE id = '{body['id']}'"
    )
    assert body["area_ha"] == pytest.approx(expected)


def test_area_counts_all_parts(api_client: TestClient, headers: dict[str, str]) -> None:
    second = [[x + 0.01, y] for x, y in RING]

    one = post_ring(api_client, headers, "Одна часть", RING).json()["area_ha"]
    vineyard_id = api_client.post(
        URL,
        headers=headers,
        json={
            "name": "Две части",
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [[[[x, y + 0.01] for x, y in RING]], [second]],
            },
        },
    ).json()["id"]

    listed = api_client.get(f"{URL}/{vineyard_id}", headers=headers).json()
    assert listed["area_ha"] == pytest.approx(2 * one, rel=1e-3)


NEARBY = f"{URL}/nearby"
# Точка внутри участка в позиции 0 (центр RING)
INSIDE = {"lat": 47.1405, "lon": 28.861}


def test_nearby_starts_with_vineyard_containing_point(
    api_client: TestClient, headers: dict[str, str]
) -> None:
    for slot, name in enumerate(["Здесь", "Рядом", "Дальше"]):
        create(api_client, headers, name, slot)

    response = api_client.get(NEARBY, headers=headers, params=INSIDE)

    assert response.status_code == 200
    found = response.json()
    assert [item["name"] for item in found] == ["Здесь", "Рядом", "Дальше"]
    assert found[0]["distance_m"] == 0
    # От точки до края соседнего участка — 0,002° долготы: около 152 м на этой широте
    assert found[1]["distance_m"] == pytest.approx(152, abs=2)
    assert found[1]["area_ha"] > 0


def test_nearby_respects_limit_and_distance(
    api_client: TestClient, headers: dict[str, str]
) -> None:
    for slot in range(3):
        create(api_client, headers, f"Участок {slot}", slot)

    limited = api_client.get(NEARBY, headers=headers, params={**INSIDE, "limit": 1})
    near_only = api_client.get(
        NEARBY, headers=headers, params={**INSIDE, "max_distance_km": 0.2}
    )

    assert [item["name"] for item in limited.json()] == ["Участок 0"]
    assert [item["name"] for item in near_only.json()] == ["Участок 0", "Участок 1"]


def test_nearby_ignores_other_owners(
    api_client: TestClient, headers: dict[str, str]
) -> None:
    create(api_client, login_as(api_client, "neighbour@example.md"), "Чужой")

    assert api_client.get(NEARBY, headers=headers, params=INSIDE).json() == []


@pytest.mark.parametrize(
    "params",
    [
        {"lat": 95, "lon": 28.86},
        {"lat": 47.14, "lon": 200},
        {"lat": 47.14},
        {**INSIDE, "limit": 0},
        {**INSIDE, "max_distance_km": 1000},
    ],
)
def test_nearby_validates_query(
    api_client: TestClient, headers: dict[str, str], params: dict[str, float]
) -> None:
    assert api_client.get(NEARBY, headers=headers, params=params).status_code == 422


def test_nearby_requires_login(api_client: TestClient) -> None:
    assert api_client.get(NEARBY, params=INSIDE).status_code == 401


IMPORT = f"{URL}/import"


def feature(
    name: str, ring: list[list[float]], kind: str = "Polygon"
) -> dict[str, object]:
    return {
        "type": "Feature",
        "properties": {"name": name},
        "geometry": {"type": kind, "coordinates": [ring]},
    }


def upload(
    api_client: TestClient, headers: dict[str, str], content: bytes, name: str
) -> Response:
    return api_client.post(IMPORT, headers=headers, files={"file": (name, content)})


def test_import_creates_valid_and_reports_skipped(
    api_client: TestClient, run_sql: RunSql, headers: dict[str, str]
) -> None:
    bowtie = [
        [28.9, 47.14],
        [28.902, 47.141],
        [28.902, 47.14],
        [28.9, 47.141],
        [28.9, 47.14],
    ]
    collection = {
        "type": "FeatureCollection",
        "features": [
            feature("Криково", RING),
            feature("Пуркарь", [[x + 0.003, y] for x, y in RING]),
            feature("Внахлёст", [[x + 0.001, y] for x, y in RING]),
            feature("Восьмёрка", bowtie),
            {
                "type": "Feature",
                "properties": {"name": "Колодец"},
                "geometry": {"type": "Point", "coordinates": [28.8, 47]},
            },
            feature("Незамкнутый", RING[:-1]),
        ],
    }

    response = upload(
        api_client, headers, json.dumps(collection).encode(), "farm.geojson"
    )

    assert response.status_code == 200
    body = response.json()
    assert [item["name"] for item in body["created"]] == ["Криково", "Пуркарь"]
    assert body["created"][0]["area_ha"] > 0
    assert body["skipped"] == [
        {
            "index": 3,
            "name": "Внахлёст",
            "reason": "Контур накладывается на участок «Криково»",
        },
        {"index": 4, "name": "Восьмёрка", "reason": "Контур пересекает сам себя"},
        {"index": 5, "name": "Колодец", "reason": "Не полигон (Point)"},
        {"index": 6, "name": "Незамкнутый", "reason": "Контур не замкнут или неполон"},
    ]
    assert run_sql("SELECT count(*) FROM vineyards") == [(2,)]


def test_import_kml_names_unnamed_objects(
    api_client: TestClient, headers: dict[str, str]
) -> None:
    coordinates = " ".join(f"{x},{y},0" for x, y in RING)
    content = (
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark>'
        "<Polygon><outerBoundaryIs><LinearRing><coordinates>"
        f"{coordinates}</coordinates></LinearRing></outerBoundaryIs></Polygon>"
        "</Placemark></Document></kml>"
    ).encode()

    body = upload(api_client, headers, content, "farm.kml").json()

    assert [item["name"] for item in body["created"]] == ["Участок 1"]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b'{"type": "FeatureCollection", "features": []}', "В файле нет объектов"),
        (b"name;lat;lon", "Нужен файл GeoJSON или KML"),
    ],
)
def test_import_rejects_useless_files(
    api_client: TestClient, headers: dict[str, str], content: bytes, message: str
) -> None:
    response = upload(api_client, headers, content, "farm.txt")

    assert response.status_code == 400
    assert response.json()["detail"] == message


def test_import_limits(
    api_client: TestClient, headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.api.routes import vineyards

    monkeypatch.setattr(vineyards, "MAX_IMPORT_FEATURES", 1)
    two = {
        "type": "FeatureCollection",
        "features": [feature("a", RING), feature("b", RING)],
    }
    too_many = upload(api_client, headers, json.dumps(two).encode(), "farm.geojson")

    monkeypatch.setattr(vineyards, "MAX_IMPORT_BYTES", 10)
    too_big = upload(api_client, headers, b"{" + b" " * 20 + b"}", "farm.geojson")

    assert too_many.status_code == too_big.status_code == 400
    assert "разделите его" in too_many.json()["detail"]
    assert too_big.json()["detail"] == "Файл больше 5 МБ"


def test_import_requires_login(api_client: TestClient) -> None:
    response = api_client.post(IMPORT, files={"file": ("farm.geojson", b"{}")})

    assert response.status_code == 401


EXPORT = f"{URL}/export"


def test_export_geojson(api_client: TestClient, headers: dict[str, str]) -> None:
    create(api_client, headers, "Криково")

    response = api_client.get(EXPORT, headers=headers)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/geo+json"
    assert 'filename="vineyards.geojson"' in response.headers["content-disposition"]
    [exported] = response.json()["features"]
    assert exported["properties"]["name"] == "Криково"
    assert exported["properties"]["area_ha"] == pytest.approx(1.69, abs=0.01)
    assert exported["geometry"] == {"type": "MultiPolygon", "coordinates": [[RING]]}


def test_export_shapefile(api_client: TestClient, headers: dict[str, str]) -> None:
    create(api_client, headers, "Криково")
    create(api_client, headers, "Пуркарь", 1)

    response = api_client.get(EXPORT, headers=headers, params={"format": "shapefile"})

    assert response.headers["content-type"] == "application/zip"
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    assert sorted(archive.namelist()) == [
        f"vineyards.{ext}" for ext in ["cpg", "dbf", "prj", "shp", "shx"]
    ]
    assert archive.read("vineyards.cpg") == b"UTF-8"
    assert b"WGS_1984" in archive.read("vineyards.prj")
    reader = shapefile.Reader(
        shp=io.BytesIO(archive.read("vineyards.shp")),
        shx=io.BytesIO(archive.read("vineyards.shx")),
        dbf=io.BytesIO(archive.read("vineyards.dbf")),
        encoding="utf-8",
    )
    assert [record["name"] for record in reader.records()] == ["Криково", "Пуркарь"]
    # Внешний контур в Shapefile — по часовой стрелке: обход RING развёрнут
    assert [list(point) for point in reader.shape(0).points] == RING[::-1]


def test_export_one_vineyard_and_roundtrip(
    api_client: TestClient, headers: dict[str, str]
) -> None:
    vineyard_id = create(api_client, headers, "Криково")

    exported = api_client.get(f"{URL}/{vineyard_id}/export", headers=headers)
    assert f'filename="vineyard-{vineyard_id}.geojson"' in (
        exported.headers["content-disposition"]
    )

    # Выгруженный файл снова импортируется — у другого пользователя
    other = login_as(api_client, "neighbour@example.md")
    body = upload(api_client, other, exported.content, "export.geojson").json()
    assert [item["name"] for item in body["created"]] == ["Криково"]
    assert body["created"][0]["geometry"]["coordinates"] == [[RING]]


def test_export_of_foreign_vineyard_looks_missing(
    api_client: TestClient, headers: dict[str, str]
) -> None:
    foreign = create(api_client, login_as(api_client, "neighbour@example.md"), "Чужой")

    response = api_client.get(f"{URL}/{foreign}/export", headers=headers)

    assert response.status_code == 404


def test_export_validates_format_and_login(
    api_client: TestClient, headers: dict[str, str]
) -> None:
    unknown = api_client.get(EXPORT, headers=headers, params={"format": "kmz"})

    assert unknown.status_code == 422
    assert api_client.get(EXPORT).status_code == 401


@pytest.mark.parametrize("export_format", ["geojson", "shapefile"])
def test_export_without_vineyards(
    api_client: TestClient, headers: dict[str, str], export_format: str
) -> None:
    response = api_client.get(EXPORT, headers=headers, params={"format": export_format})

    assert response.status_code == 200
    assert response.content


def merlot_id(api_client: TestClient, headers: dict[str, str]) -> str:
    varieties = api_client.get("/api/v1/varieties", headers=headers).json()
    variety_id: str = next(v["id"] for v in varieties if v["name"] == "Merlot")
    return variety_id


def test_variety_and_planting_year(
    api_client: TestClient, headers: dict[str, str]
) -> None:
    merlot = merlot_id(api_client, headers)

    created = api_client.post(
        URL,
        headers=headers,
        json={
            "name": "Криково",
            "geometry": {"type": "Polygon", "coordinates": [RING]},
            "variety_id": merlot,
            "planting_year": 2015,
        },
    )

    assert created.status_code == 201
    assert created.json()["variety"] == {
        "id": merlot,
        "name": "Merlot",
        "berry_color": "red",
    }
    assert created.json()["planting_year"] == 2015
    listed = api_client.get(URL, headers=headers).json()["items"][0]
    assert listed["variety"]["name"] == "Merlot"


def test_variety_and_year_are_optional(
    api_client: TestClient, headers: dict[str, str]
) -> None:
    vineyard_id = create(api_client, headers, "Без сорта")

    body = api_client.get(f"{URL}/{vineyard_id}", headers=headers).json()

    assert (body["variety"], body["planting_year"]) == (None, None)


@pytest.mark.parametrize(
    ("extra", "status"),
    [
        ({"planting_year": 1899}, 422),
        ({"planting_year": 2999}, 422),
        ({"variety_id": "01900000-0000-7000-8000-000000000000"}, 400),
    ],
)
def test_variety_and_year_are_validated(
    api_client: TestClient,
    headers: dict[str, str],
    extra: dict[str, object],
    status: int,
) -> None:
    response = api_client.post(
        URL,
        headers=headers,
        json={
            "name": "Криково",
            "geometry": {"type": "Polygon", "coordinates": [RING]},
            **extra,
        },
    )

    assert response.status_code == status


def test_update_vineyard(api_client: TestClient, headers: dict[str, str]) -> None:
    vineyard_id = create(api_client, headers, "Криково")
    merlot = merlot_id(api_client, headers)
    moved = [[round(x + 0.0005, 6), y] for x, y in RING]

    response = api_client.patch(
        f"{URL}/{vineyard_id}",
        headers=headers,
        json={
            "name": "Криково, северный склон",
            "geometry": {"type": "Polygon", "coordinates": [moved]},
            "variety_id": merlot,
            "planting_year": 2018,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Криково, северный склон"
    # Сдвинутый контур пересекается со старым — сам с собой участок не конфликтует
    assert body["geometry"]["coordinates"] == [[moved]]
    assert (body["variety"]["name"], body["planting_year"]) == ("Merlot", 2018)

    cleared = api_client.patch(
        f"{URL}/{vineyard_id}",
        headers=headers,
        json={"variety_id": None, "planting_year": None},
    ).json()
    assert (cleared["variety"], cleared["planting_year"]) == (None, None)
    assert cleared["name"] == "Криково, северный склон"


def test_update_checks_overlap_and_validation(
    api_client: TestClient, headers: dict[str, str]
) -> None:
    create(api_client, headers, "Криково")
    neighbour = create(api_client, headers, "Сосед", 1)
    onto_first = [[x + 0.001, y] for x, y in RING]

    overlap = api_client.patch(
        f"{URL}/{neighbour}",
        headers=headers,
        json={"geometry": {"type": "Polygon", "coordinates": [onto_first]}},
    )
    erased = api_client.patch(
        f"{URL}/{neighbour}", headers=headers, json={"name": None}
    )
    unknown = api_client.patch(
        f"{URL}/{neighbour}",
        headers=headers,
        json={"variety_id": "01900000-0000-7000-8000-000000000000"},
    )

    assert overlap.status_code == 409
    assert erased.status_code == 422
    assert unknown.status_code == 400


def test_delete_vineyard(
    api_client: TestClient, run_sql: RunSql, headers: dict[str, str]
) -> None:
    vineyard_id = create(api_client, headers, "Криково")

    assert api_client.delete(f"{URL}/{vineyard_id}", headers=headers).status_code == 204
    assert api_client.get(f"{URL}/{vineyard_id}", headers=headers).status_code == 404
    assert run_sql("SELECT count(*) FROM vineyards") == [(0,)]


def test_foreign_vineyard_cannot_be_changed(
    api_client: TestClient, headers: dict[str, str]
) -> None:
    foreign = create(api_client, login_as(api_client, "neighbour@example.md"), "Чужой")

    assert (
        api_client.patch(
            f"{URL}/{foreign}", headers=headers, json={"name": "Мой"}
        ).status_code
        == 404
    )
    assert api_client.delete(f"{URL}/{foreign}", headers=headers).status_code == 404


def test_deleting_variety_keeps_vineyards(
    api_client: TestClient, headers: dict[str, str]
) -> None:
    api_client.post(
        "/api/v1/auth/register",
        json={
            "email": "agro2@example.md",
            "password": "secret123",
            "full_name": "Агроном",
            "role": "agronomist",
        },
    )
    agro = api_client.post(
        "/api/v1/auth/login",
        json={"email": "agro2@example.md", "password": "secret123"},
    ).json()["access_token"]
    agro_headers = {"Authorization": f"Bearer {agro}"}
    variety = api_client.post(
        "/api/v1/varieties",
        headers=agro_headers,
        json={"name": "Viorica", "berry_color": "white"},
    ).json()["id"]
    created = api_client.post(
        URL,
        headers=headers,
        json={
            "name": "Криково",
            "geometry": {"type": "Polygon", "coordinates": [RING]},
            "variety_id": variety,
        },
    ).json()["id"]

    api_client.delete(f"/api/v1/varieties/{variety}", headers=agro_headers)

    body = api_client.get(f"{URL}/{created}", headers=headers).json()
    assert body["variety"] is None
