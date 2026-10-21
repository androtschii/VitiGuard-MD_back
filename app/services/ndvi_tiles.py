"""Тайлы XYZ картограмм NDVI для карты: один растровый слой на все участки
пользователя вместо картинки на каждый участок.

Картограммы лежат в базе в сетке UTM снимка с углами в WGS 84 (pr-063).
Тайл веб-меркатора собирается из них выборкой ближайшего пикселя: для центра
каждого пикселя тайла находится пиксель картограммы, в который он попадает."""

import io
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

import numpy as np
from geoalchemy2 import functions as geo
from PIL import Image
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import ProcessingStatus
from app.models.satellite import SatelliteScan
from app.models.vineyard import Vineyard

TILE = 256
# Мельче участок в несколько гектаров занимает меньше пикселя тайла
MIN_ZOOM = 10
MAX_ZOOM = 19


@dataclass(frozen=True)
class NdviMapImage:
    png: bytes
    # [долгота, широта]: верхний левый, верхний правый, нижний правый, нижний левый
    corners: list[list[float]]


def tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Запад, юг, восток, север тайла в градусах."""
    n = 2**z

    def lat(row: float) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * row / n))))

    return x / n * 360 - 180, lat(y + 1), (x + 1) / n * 360 - 180, lat(y)


def _pixel_centers(z: int, x: int, y: int) -> tuple[np.ndarray, np.ndarray]:
    n = 2**z
    offsets = (np.arange(TILE) + 0.5) / TILE
    lon = (x + offsets) / n * 360 - 180
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + offsets) / n))))
    return np.meshgrid(lon, lat)


def render_tile(z: int, x: int, y: int, maps: list[NdviMapImage]) -> bytes | None:
    """PNG тайла или None, если ни одна картограмма в него не попала."""
    lon, lat = _pixel_centers(z, x, y)
    tile = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    for item in maps:
        image = np.asarray(Image.open(io.BytesIO(item.png)).convert("RGBA"))
        height, width = image.shape[:2]
        top_left, top_right, _, bottom_left = (np.array(c) for c in item.corners)
        # Окно снимка — параллелограмм: доля по стороне «вправо» и «вниз»
        # находится из двух векторов сторон
        axes = np.column_stack([top_right - top_left, bottom_left - top_left])
        u, v = np.linalg.solve(
            axes, np.stack([lon - top_left[0], lat - top_left[1]]).reshape(2, -1)
        )
        col = np.floor(u * width).astype(int).reshape(TILE, TILE)
        row = np.floor(v * height).astype(int).reshape(TILE, TILE)
        inside = (col >= 0) & (col < width) & (row >= 0) & (row < height)
        pixels = np.zeros_like(tile)
        pixels[inside] = image[row[inside], col[inside]]
        visible = pixels[..., 3] > 0
        tile[visible] = pixels[visible]
    if not tile[..., 3].any():
        return None
    buffer = io.BytesIO()
    Image.fromarray(tile, mode="RGBA").save(buffer, "PNG", optimize=True)
    return buffer.getvalue()


async def latest_maps(
    session: AsyncSession,
    owner_id: uuid.UUID,
    bounds: tuple[float, float, float, float],
    day: date | None,
) -> list[NdviMapImage]:
    """Последняя картограмма каждого участка владельца в границах тайла,
    снятая не позже дня day (None — без ограничения)."""
    conditions = [
        Vineyard.owner_id == owner_id,
        geo.ST_Intersects(Vineyard.geom, geo.ST_MakeEnvelope(*bounds, 4326)),
        SatelliteScan.status == ProcessingStatus.COMPLETED,
        SatelliteScan.ndvi_corners.is_not(None),
    ]
    if day is not None:
        end = datetime.combine(day + timedelta(days=1), time.min, UTC)
        conditions.append(SatelliteScan.acquired_at < end)
    rows = await session.execute(
        select(SatelliteScan.ndvi_image, SatelliteScan.ndvi_corners)
        .join(Vineyard, Vineyard.id == SatelliteScan.vineyard_id)
        .where(*conditions)
        .order_by(SatelliteScan.vineyard_id, SatelliteScan.acquired_at.desc())
        .ext(distinct_on(SatelliteScan.vineyard_id))
    )
    return [
        NdviMapImage(png, corners)
        for png, corners in rows
        if png is not None and corners is not None
    ]
