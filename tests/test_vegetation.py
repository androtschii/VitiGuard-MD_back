from pathlib import Path
from typing import Any

import numpy as np
import pytest
from raster_scene import CLEAR, CLOUD, PARCEL, make_scene

from app.services.vegetation import (
    ParcelBands,
    compute_indices,
    normalized_difference,
    read_parcel,
)


def test_indices_of_clear_parcel(tmp_path: Path) -> None:
    bands = read_parcel(make_scene(tmp_path), PARCEL)
    result = compute_indices(bands)

    # Около 100 × 100 м — примерно сотня пикселей 10 м
    assert 80 <= result.parcel_pixels <= 130
    assert result.clear_pixels == result.parcel_pixels
    assert result.cloud_share == 0
    assert result.ndvi is not None
    assert result.ndre is not None
    assert result.ndvi.mean == pytest.approx((0.3 - 0.05) / 0.35, abs=1e-3)
    assert result.ndvi.std == pytest.approx(0, abs=1e-3)
    assert result.ndre.mean == pytest.approx(0.5, abs=1e-3)
    # Каналы 20 м приведены к сетке 10 м
    assert bands.rededge.shape == bands.red.shape == bands.scl.shape


def test_reflectance_offset_is_applied(tmp_path: Path) -> None:
    # Смещение −0,1 (снимок без вычтенного смещения): красный 0,15 − 0,1 = 0,05
    bands = read_parcel(make_scene(tmp_path, offset=-0.1, red=1500), PARCEL)

    assert float(bands.red[bands.inside].mean()) == pytest.approx(0.05, abs=1e-6)


def test_cloudy_parcel_is_skipped(tmp_path: Path) -> None:
    result = compute_indices(read_parcel(make_scene(tmp_path, scl=CLOUD), PARCEL))

    assert result.ndvi is None
    assert result.ndre is None
    assert result.cloud_share == 1


def test_partly_cloudy_parcel_uses_clear_pixels(tmp_path: Path) -> None:
    scl = np.full((20, 20), CLEAR, dtype=np.uint16)
    # Облако над северной половиной растра задевает часть участка
    scl[:9] = CLOUD
    result = compute_indices(read_parcel(make_scene(tmp_path, scl=scl), PARCEL))

    assert 0 < result.cloud_share < 0.5
    assert result.ndvi is not None
    assert result.ndvi.mean == pytest.approx(0.714, abs=1e-3)


def test_parcel_smaller_than_pixel_takes_its_pixel(tmp_path: Path) -> None:
    tiny: dict[str, Any] = {
        "type": "Polygon",
        "coordinates": [
            [
                [28.86050, 47.14040],
                [28.86052, 47.14040],
                [28.86052, 47.14041],
                [28.86050, 47.14040],
            ]
        ],
    }
    result = compute_indices(read_parcel(make_scene(tmp_path), tiny))

    assert result.parcel_pixels >= 1
    assert result.ndvi is not None


def test_normalized_difference_without_signal_is_nan() -> None:
    values = normalized_difference(
        np.array([0.3, 0.0, 0.9]), np.array([0.1, 0.0, -0.5])
    )

    assert values[0] == pytest.approx(0.5)
    assert np.isnan(values[1])
    # Шум отрицательного отражения не выводит индекс за [−1, 1]
    assert values[2] == 1


def test_no_finite_values_gives_no_index() -> None:
    zeros = np.zeros((2, 2), dtype=np.float32)
    bands = ParcelBands(
        red=zeros,
        rededge=zeros,
        nir=zeros,
        nir_narrow=zeros,
        scl=np.full((2, 2), CLEAR),
        inside=np.ones((2, 2), dtype=bool),
    )

    result = compute_indices(bands)

    assert result.clear_pixels == 4
    assert result.ndvi is None
    assert result.ndre is None


def test_empty_parcel_counts_as_cloudy() -> None:
    empty = np.zeros((0, 0), dtype=np.float32)
    bands = ParcelBands(empty, empty, empty, empty, empty, empty.astype(bool))

    assert compute_indices(bands).cloud_share == 1
