import pytest
from diagnosis_model import gps_photo, photo

from app.services.diagnosis import photo_location


def test_location_from_exif() -> None:
    # 47°08'24.36" N, 28°51'36" E — Криково
    location = photo_location(gps_photo((47, 8, 24.36), (28, 51, 36)))

    assert location is not None
    assert location.lat == pytest.approx(47.140100, abs=1e-6)
    assert location.lon == pytest.approx(28.86, abs=1e-6)


def test_southern_and_western_hemispheres_are_negative() -> None:
    location = photo_location(gps_photo((33, 52, 4), (151, 12, 36), ("S", "W")))

    assert location is not None
    assert location.lat < 0
    assert location.lon < 0


@pytest.mark.parametrize(
    "data",
    [
        photo((255, 0, 0)),
        photo((255, 0, 0), image_format="PNG"),
        gps_photo((0, 0, 0), (0, 0, 0)),
        gps_photo((95, 0, 0), (28, 51, 36)),
        b"not an image",
    ],
    ids=["no-exif", "png", "zeros", "out-of-range", "garbage"],
)
def test_no_location(data: bytes) -> None:
    assert photo_location(data) is None
