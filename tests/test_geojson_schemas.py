from typing import Any

import pytest
from pydantic import ValidationError

from app.schemas.vineyard import VineyardCreate

RING = [[28.86, 47.14], [28.862, 47.14], [28.862, 47.141], [28.86, 47.14]]


def payload(geometry: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"name": "Крикова, участок 1", "geometry": geometry, **extra}


def test_polygon_is_accepted() -> None:
    data = VineyardCreate.model_validate(
        payload({"type": "Polygon", "coordinates": [RING]})
    )

    assert data.geometry.type == "Polygon"
    assert data.name == "Крикова, участок 1"


def test_multipolygon_is_accepted() -> None:
    data = VineyardCreate.model_validate(
        payload({"type": "MultiPolygon", "coordinates": [[RING], [RING]]})
    )

    assert data.geometry.type == "MultiPolygon"


@pytest.mark.parametrize(
    ("geometry", "message"),
    [
        (
            {"type": "Polygon", "coordinates": [[*RING[:-1], [28.9, 47.2]]]},
            "не замкнут",
        ),
        ({"type": "Polygon", "coordinates": [RING[:2] + RING[:1]]}, "at least 4"),
        ({"type": "Polygon", "coordinates": []}, "at least 1"),
        (
            {
                "type": "Polygon",
                "coordinates": [[[200, 47.14], *RING[1:-1], [200, 47.14]]],
            },
            "less than or equal to 180",
        ),
        (
            {
                "type": "Polygon",
                "coordinates": [[[28.86, 95], *RING[1:-1], [28.86, 95]]],
            },
            "less than or equal to 90",
        ),
        ({"type": "Point", "coordinates": [28.86, 47.14]}, "does not match any"),
    ],
)
def test_invalid_geometry_is_rejected(geometry: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        VineyardCreate.model_validate(payload(geometry))


def test_third_coordinate_is_rejected() -> None:
    ring = [[*point, 120] for point in RING]

    with pytest.raises(ValidationError):
        VineyardCreate.model_validate(
            payload({"type": "Polygon", "coordinates": [ring]})
        )


@pytest.mark.parametrize("name", ["", "   ", "x" * 256])
def test_name_must_be_meaningful(name: str) -> None:
    with pytest.raises(ValidationError):
        VineyardCreate.model_validate(
            {"name": name, "geometry": {"type": "Polygon", "coordinates": [RING]}}
        )


def test_owner_cannot_be_passed() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        VineyardCreate.model_validate(
            payload({"type": "Polygon", "coordinates": [RING]}, owner_id="x")
        )
