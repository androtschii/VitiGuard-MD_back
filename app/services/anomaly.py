"""Аномалии NDVI на участке: просадка относительно своей нормы и соседей.

Метод выбран в ml/pr-019: на реальных рядах простая оценка просадки точнее
Isolation Forest и LOF и объяснима агроному — «NDVI на 0,15 ниже обычного для
этих дней и ниже соседних участков»."""

import statistics
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import ProcessingStatus
from app.models.satellite import IndexType, SatelliteScan, VegetationIndex
from app.models.vineyard import Vineyard

# Норма участка — медиана своих снимков в этом окне, соседей — в узком окне:
# общая для района засуха или листопад в те же дни не тревога
SELF_WINDOW = timedelta(days=20)
PEER_WINDOW = timedelta(days=3)
MIN_SELF = 2
MIN_PEERS = 3
# Разброс по трём-четырём снимкам ненадёжен: в начале сезона тревог нет
MIN_SCANS = 5
# Остаточные облака и тени дают «просадки» того же порядка, что болезнь
MAX_PARCEL_CLOUD = 20
# Меньшие просадки по одному снимку надёжно не отличить от шума снимков
MIN_DROP = 0.1
# Разброс отклонений от нормы не меньше этого: шум индекса по чистым снимкам
MIN_SCALE = 0.03
MIN_Z = 2.0

Level = Literal["suspected", "confirmed"]


@dataclass(frozen=True)
class Observation:
    acquired_at: datetime
    ndvi: float


@dataclass(frozen=True)
class Anomaly:
    acquired_at: datetime
    ndvi: float
    # Насколько NDVI ниже своей нормы и ниже соседей (None — соседей мало)
    below_norm: float
    below_peers: float | None
    z: float
    # suspected — один снимок, confirmed — и предыдущий чистый снимок тоже
    level: Level


def _median_near(
    values: Sequence[Observation], moment: datetime, window: timedelta, minimum: int
) -> float | None:
    near = [item.ndvi for item in values if abs(item.acquired_at - moment) <= window]
    return statistics.median(near) if len(near) >= minimum else None


def detect_anomalies(
    series: Sequence[Observation], peers: Sequence[Observation]
) -> list[Anomaly]:
    """Снимки участка с просадкой NDVI. series и peers — чистые снимки по дате."""
    deviations: list[float | None] = []
    for index, item in enumerate(series):
        # Без самого снимка: одна просадка не сдвигает собственную норму
        others = [other for i, other in enumerate(series) if i != index]
        norm = _median_near(others, item.acquired_at, SELF_WINDOW, MIN_SELF)
        deviations.append(None if norm is None else item.ndvi - norm)

    known = [value for value in deviations if value is not None]
    if len(known) < MIN_SCANS:
        return []
    center = statistics.median(known)
    mad = statistics.median(abs(value - center) for value in known)
    # Устойчивая оценка разброса: сами просадки его не раздувают
    scale = max(1.4826 * mad, MIN_SCALE)

    anomalies: list[Anomaly] = []
    previous_dropped = False
    for item, deviation in zip(series, deviations, strict=True):
        if deviation is None:
            previous_dropped = False
            continue
        peer = _median_near(peers, item.acquired_at, PEER_WINDOW, MIN_PEERS)
        below_peers = None if peer is None else peer - item.ndvi
        z = -deviation / scale
        dropped = (
            -deviation >= MIN_DROP
            and z >= MIN_Z
            and (below_peers is None or below_peers >= MIN_DROP / 2)
        )
        if dropped:
            anomalies.append(
                Anomaly(
                    acquired_at=item.acquired_at,
                    ndvi=item.ndvi,
                    below_norm=round(-deviation, 4),
                    below_peers=None if below_peers is None else round(below_peers, 4),
                    z=round(z, 2),
                    level="confirmed" if previous_dropped else "suspected",
                )
            )
        previous_dropped = dropped
    return anomalies


async def _clear_ndvi(
    session: AsyncSession, vineyard_ids: Sequence[uuid.UUID]
) -> dict[uuid.UUID, list[Observation]]:
    rows = await session.execute(
        select(
            SatelliteScan.vineyard_id,
            SatelliteScan.acquired_at,
            VegetationIndex.mean_value,
        )
        .join(VegetationIndex, VegetationIndex.scan_id == SatelliteScan.id)
        .where(
            SatelliteScan.vineyard_id.in_(vineyard_ids),
            SatelliteScan.status == ProcessingStatus.COMPLETED,
            SatelliteScan.parcel_cloud_cover <= MAX_PARCEL_CLOUD,
            VegetationIndex.index_type == IndexType.NDVI,
        )
        .order_by(SatelliteScan.acquired_at)
    )
    series: dict[uuid.UUID, list[Observation]] = {}
    for vineyard_id, acquired_at, ndvi in rows:
        series.setdefault(vineyard_id, []).append(Observation(acquired_at, ndvi))
    return series


async def vineyard_anomalies(
    session: AsyncSession, vineyard: Vineyard
) -> list[Anomaly]:
    """Соседи — другие участки того же владельца: участки других пользователей
    приватны, и даже их медиана по двум-трём участкам выдала бы чужие значения."""
    ids = (
        await session.scalars(
            select(Vineyard.id).where(Vineyard.owner_id == vineyard.owner_id)
        )
    ).all()
    series = await _clear_ndvi(session, ids)
    peers = [
        item
        for vineyard_id, items in series.items()
        if vineyard_id != vineyard.id
        for item in items
    ]
    return detect_anomalies(series.get(vineyard.id, []), peers)
