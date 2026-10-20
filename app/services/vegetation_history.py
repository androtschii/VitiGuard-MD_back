"""История вегетационных индексов участка (pr-060, pr-064, pr-066, pr-069).

Обработка снимков — фоновая задача: для каждого снимка нового периода из
каналов читается окно участка и считаются NDVI и NDRE, результат сохраняется.
Уже обработанные снимки пропускаются, поэтому задачу можно запускать повторно —
например, раз в несколько дней, чтобы подтянуть новые снимки сезона."""

import asyncio
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.errors import NotFoundError
from app.integrations.earth_search import EarthScene, EarthSearchClient, cloud_query
from app.integrations.earth_search import deduplicate as deduplicate_scenes
from app.models.enums import ProcessingStatus
from app.models.satellite import IndexType, SatelliteScan, VegetationIndex
from app.models.vineyard import Vineyard
from app.services.satellite import vineyard_geojson
from app.services.vegetation import (
    IndexStats,
    NdviMap,
    ParcelIndices,
    compute_indices,
    ndvi_map,
    read_parcel,
)


@dataclass(frozen=True)
class ProcessingStats:
    # Снимков в каталоге за период (без повторов)
    found: int
    # Обработано сейчас; из них закрыто облаками над участком
    processed: int
    cloudy: int
    # Обработаны раньше — пропущены
    skipped: int
    # Не удалось прочитать каналы (сбой хранилища)
    failed: int


def _read_scene(
    scene: EarthScene, geometry: dict[str, Any]
) -> tuple[ParcelIndices, NdviMap]:
    bands = read_parcel(scene, geometry)
    return compute_indices(bands), ndvi_map(bands)


def _index(index_type: IndexType, stats: IndexStats) -> VegetationIndex:
    return VegetationIndex(
        index_type=index_type,
        mean_value=stats.mean,
        min_value=stats.min,
        max_value=stats.max,
        std_value=stats.std,
    )


async def process_vineyard(
    session: AsyncSession,
    client: EarthSearchClient,
    vineyard_id: uuid.UUID,
    start: datetime,
    end: datetime,
    max_scene_cloud_cover: float,
) -> ProcessingStats:
    vineyard = await session.get(Vineyard, vineyard_id)
    if vineyard is None:
        raise NotFoundError("Участок не найден")
    geometry = await vineyard_geojson(session, vineyard)
    # Облачность сцены — лишь грубый отсев: облака над участком проверяются
    # по маске SCL, а сцена с 40 % облаков часто чиста над самим участком
    scenes = deduplicate_scenes(
        await client.search(
            geometry, start, end, extra_body=cloud_query(max_scene_cloud_cover)
        )
    )
    # Только идентификаторы и статусы: объекты сессии устаревают после каждого
    # сохранения, а их поля в асинхронной сессии лениво не подгружаются
    has_ndvi = (
        select(VegetationIndex.id)
        .where(
            VegetationIndex.scan_id == SatelliteScan.id,
            VegetationIndex.index_type == IndexType.NDVI,
        )
        .exists()
    )
    known = {
        product_id: (scan_id, status, has_map, with_ndvi)
        for scan_id, product_id, status, has_map, with_ndvi in await session.execute(
            select(
                SatelliteScan.id,
                SatelliteScan.product_id,
                SatelliteScan.status,
                SatelliteScan.ndvi_corners.is_not(None),
                has_ndvi,
            ).where(SatelliteScan.vineyard_id == vineyard_id)
        )
    }
    processed = cloudy = skipped = failed = 0
    for scene in scenes:
        entry = known.get(scene.product_id)
        completed = entry is not None and entry[1] is ProcessingStatus.COMPLETED
        # Готовый снимок читается заново, только если к его индексам нет картограммы
        if entry is not None and completed and (entry[2] or not entry[3]):
            skipped += 1
            continue
        previous = (
            await session.get(
                SatelliteScan,
                entry[0],
                options=[selectinload(SatelliteScan.indices)],
                populate_existing=True,
            )
            if entry is not None
            else None
        )
        if previous is not None:
            scan = previous
        else:
            scan = SatelliteScan(
                vineyard_id=vineyard_id,
                product_id=scene.product_id,
                acquired_at=scene.acquired_at,
                cloud_cover=round(scene.cloud_cover, 2),
                platform=scene.platform,
            )
        has_failed = False
        try:
            # Чтение растров блокирует поток — в отдельном потоке
            result, parcel_map = await asyncio.to_thread(_read_scene, scene, geometry)
        except OSError as error:
            # Ошибки rasterio при чтении по HTTP — подклассы OSError. У готового
            # снимка индексы остаются, картограмма будет в следующий раз
            has_failed = True
            if not completed:
                scan.status = ProcessingStatus.FAILED
                scan.error = str(error)[:1000]
        else:
            if not completed:
                scan.status = ProcessingStatus.COMPLETED
                scan.error = None
                scan.parcel_cloud_cover = round(result.cloud_share * 100, 1)
                if result.ndvi is not None:
                    scan.indices.append(_index(IndexType.NDVI, result.ndvi))
                if result.ndre is not None:
                    scan.indices.append(_index(IndexType.NDRE, result.ndre))
            if result.ndvi is not None:
                scan.ndvi_image = parcel_map.png
                scan.ndvi_corners = [list(corner) for corner in parcel_map.corners]
        has_indices = bool(scan.indices)
        session.add(scan)
        # Каждый снимок сохраняется сразу: обработка сезона идёт минуты, и при
        # сбое посередине готовая часть не теряется
        try:
            await session.commit()
        except IntegrityError:
            # Тот же снимок только что сохранила параллельная задача
            await session.rollback()
            skipped += 1
            continue
        if has_failed:
            failed += 1
        else:
            processed += 1
            cloudy += not has_indices
    return ProcessingStats(len(scenes), processed, cloudy, skipped, failed)


async def vegetation_history(
    session: AsyncSession, vineyard_id: uuid.UUID
) -> Sequence[SatelliteScan]:
    """Обработанные снимки участка по дате, вместе с индексами. Снимки, закрытые
    облаками над участком, тоже возвращаются — без индексов: клиент показывает,
    почему в ряду пропуск."""
    scans = await session.scalars(
        select(SatelliteScan)
        .where(
            SatelliteScan.vineyard_id == vineyard_id,
            SatelliteScan.status == ProcessingStatus.COMPLETED,
        )
        .options(selectinload(SatelliteScan.indices))
        .order_by(SatelliteScan.acquired_at)
    )
    return scans.all()


async def ndvi_image(
    session: AsyncSession, vineyard_id: uuid.UUID, scan_id: uuid.UUID
) -> bytes:
    image = await session.scalar(
        select(SatelliteScan.ndvi_image).where(
            SatelliteScan.id == scan_id, SatelliteScan.vineyard_id == vineyard_id
        )
    )
    if image is None:
        raise NotFoundError("Картограмма NDVI не найдена")
    return image
