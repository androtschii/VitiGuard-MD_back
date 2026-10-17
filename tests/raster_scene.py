"""Синтетический снимок Sentinel-2 из локальных GeoTIFF вокруг участка у
Крикова: каналы с известными значениями, чтобы проверять расчёт индексов без сети."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_geom

from app.integrations.earth_search import Band, EarthScene

CRS = "EPSG:32635"
# Квадрат около 100 × 100 м у Крикова
PARCEL = {
    "type": "Polygon",
    "coordinates": [
        [
            [28.8600, 47.1400],
            [28.8613, 47.1400],
            [28.8613, 47.1409],
            [28.8600, 47.1409],
            [28.8600, 47.1400],
        ]
    ],
}
CLEAR, CLOUD = 4, 9


def write_band(
    path: Path, value: int | np.ndarray, resolution: int, origin: tuple[float, float]
) -> str:
    size = 400 // resolution
    data = (
        np.full((size, size), value, dtype=np.uint16) if np.isscalar(value) else value
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=1,
        dtype="uint16",
        crs=CRS,
        transform=from_origin(*origin, resolution, resolution),
    ) as dataset:
        dataset.write(np.asarray(data, dtype=np.uint16), 1)
    return str(path)


def make_scene(
    tmp_path: Path,
    scl: int | np.ndarray = CLEAR,
    offset: float = 0.0,
    red: int = 500,
    product_id: str = "S2A_35TPN_20260829_1_L2A",
    acquired_at: datetime = datetime(2026, 8, 29, 9, 17, tzinfo=UTC),
) -> EarthScene:
    """Снимок из локальных GeoTIFF вокруг участка: красный 0,05, ближний ИК 0,3,
    red edge 0,1, узкий ближний ИК 0,3 (при scale 0,0001 и нулевом смещении)."""
    local = transform_geom("EPSG:4326", CRS, PARCEL)
    xs = [x for x, _ in local["coordinates"][0]]
    ys = [y for _, y in local["coordinates"][0]]
    # Растр 400 × 400 м с запасом вокруг участка, углы — по сетке 20 м
    origin = (
        float(np.floor(min(xs) / 20) * 20 - 140),
        float(np.ceil(max(ys) / 20) * 20 + 140),
    )
    values = {"B04": (red, 10), "B08": (3000, 10), "B05": (1000, 20), "B8A": (3000, 20)}
    folder = tmp_path / product_id
    folder.mkdir(exist_ok=True)
    bands = {
        name: Band(
            href=write_band(folder / f"{name}.tif", value, resolution, origin),
            scale=0.0001,
            offset=offset,
        )
        for name, (value, resolution) in values.items()
    }
    bands["SCL"] = Band(href=write_band(folder / "SCL.tif", scl, 20, origin))
    return EarthScene(
        product_id=product_id,
        acquired_at=acquired_at,
        cloud_cover=0.0,
        platform="sentinel-2a",
        tile="35TPN",
        bands=bands,
    )
