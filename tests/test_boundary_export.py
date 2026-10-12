from app.services.boundary_export import _shapefile_parts

# Внешний контур против часовой стрелки, вырез — по часовой (как в RFC 7946)
OUTER = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]]
HOLE = [[1.0, 1.0], [1.0, 2.0], [2.0, 2.0], [2.0, 1.0], [1.0, 1.0]]


def test_rings_are_rewound_for_shapefile() -> None:
    outer, hole = _shapefile_parts(
        {"type": "MultiPolygon", "coordinates": [[OUTER, HOLE]]}
    )

    assert outer == OUTER[::-1]
    assert hole == HOLE[::-1]


def test_correct_rings_are_kept() -> None:
    clockwise = OUTER[::-1]

    assert _shapefile_parts({"type": "MultiPolygon", "coordinates": [[clockwise]]}) == [
        clockwise
    ]
