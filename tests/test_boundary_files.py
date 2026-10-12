import json

import pytest

from app.core.errors import BadRequestError
from app.services.boundary_files import BoundaryCandidate, parse_boundaries

RING = [[28.86, 47.14], [28.862, 47.14], [28.862, 47.141], [28.86, 47.14]]
POLYGON = {"type": "Polygon", "coordinates": [RING]}


def kml(placemarks: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        f"{placemarks}</Document></kml>"
    ).encode()


def ring_text(ring: list[list[float]], altitude: bool = False) -> str:
    return " ".join(f"{x},{y}{',120' if altitude else ''}" for x, y in ring)


def kml_polygon(ring: list[list[float]], hole: list[list[float]] | None = None) -> str:
    inner = (
        f"<innerBoundaryIs><LinearRing><coordinates>{ring_text(hole)}"
        "</coordinates></LinearRing></innerBoundaryIs>"
        if hole
        else ""
    )
    return (
        "<Polygon><outerBoundaryIs><LinearRing><coordinates>"
        f"{ring_text(ring, altitude=True)}</coordinates></LinearRing>"
        f"</outerBoundaryIs>{inner}</Polygon>"
    )


def test_feature_collection_with_names_from_properties() -> None:
    content = json.dumps(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": "Криково"},
                    "geometry": POLYGON,
                },
                {"type": "Feature", "properties": {"NAME": 7}, "geometry": POLYGON},
                {"type": "Feature", "properties": None, "geometry": POLYGON},
            ],
        }
    ).encode()

    assert [item.name for item in parse_boundaries(content)] == ["Криково", "7", None]


def test_single_feature_and_bare_geometry() -> None:
    feature = {
        "type": "Feature",
        "properties": {"title": "Пуркарь"},
        "geometry": POLYGON,
    }

    assert parse_boundaries(json.dumps(feature).encode()) == [
        BoundaryCandidate("Пуркарь", POLYGON)
    ]
    assert parse_boundaries(json.dumps(POLYGON).encode()) == [
        BoundaryCandidate(None, POLYGON)
    ]


def test_bom_is_ignored() -> None:
    content = "﻿".encode() + json.dumps(POLYGON).encode()

    assert parse_boundaries(content)[0].geometry == POLYGON


def test_non_polygons_are_reported() -> None:
    content = json.dumps(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {"type": "Point", "coordinates": [28.8, 47]},
                },
                {"type": "Feature", "properties": {}, "geometry": None},
                "мусор",
            ],
        }
    ).encode()

    assert [item.problem for item in parse_boundaries(content)] == [
        "Не полигон (Point)",
        "Нет геометрии",
        "Нет геометрии",
    ]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"{not json", "не читается как GeoJSON"),
        (b'{"type": "Topology"}', "нет объектов GeoJSON"),
        (b"lon,lat\n28.8,47", "Нужен файл GeoJSON или KML"),
        (b"<kml><Placemark>", "не читается как KML"),
    ],
)
def test_unreadable_files_are_rejected(content: bytes, message: str) -> None:
    with pytest.raises(BadRequestError, match=message):
        parse_boundaries(content)


def test_kml_polygon_drops_altitude() -> None:
    content = kml(f"<Placemark><name> Криково </name>{kml_polygon(RING)}</Placemark>")

    assert parse_boundaries(content) == [BoundaryCandidate("Криково", POLYGON)]


def test_kml_hole_and_multigeometry() -> None:
    hole = [
        [28.8605, 47.1402],
        [28.861, 47.1402],
        [28.861, 47.1404],
        [28.8605, 47.1402],
    ]
    other = [[x + 0.01, y] for x, y in RING]
    content = kml(
        "<Placemark><name>С прудом</name>"
        f"{kml_polygon(RING, hole)}</Placemark>"
        "<Placemark><name>Две части</name><MultiGeometry>"
        f"{kml_polygon(RING)}{kml_polygon(other)}</MultiGeometry></Placemark>"
    )

    with_hole, two_parts = parse_boundaries(content)

    assert with_hole.geometry == {"type": "Polygon", "coordinates": [RING, hole]}
    assert two_parts.geometry == {
        "type": "MultiPolygon",
        "coordinates": [[RING], [other]],
    }


def test_kml_points_and_broken_coordinates_are_reported() -> None:
    content = kml(
        "<Placemark><Point><coordinates>28.8,47</coordinates></Point></Placemark>"
        "<Placemark><name>Сломан</name><Polygon><outerBoundaryIs><LinearRing>"
        "<coordinates>abc,def</coordinates></LinearRing></outerBoundaryIs>"
        "</Polygon></Placemark>"
    )

    assert [(item.name, item.problem) for item in parse_boundaries(content)] == [
        (None, "Не полигон"),
        ("Сломан", "Координаты не читаются"),
    ]


def test_xml_entities_are_refused() -> None:
    # «Миллиард смешков»: сущности, раскрывающиеся в гигабайты текста
    bomb = (
        b'<?xml version="1.0"?><!DOCTYPE kml [<!ENTITY a "aaaaaaaaaa">'
        b'<!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">]>'
        b'<kml xmlns="http://www.opengis.net/kml/2.2"><name>&b;</name></kml>'
    )

    with pytest.raises(BadRequestError, match="недопустимые конструкции"):
        parse_boundaries(bomb)
