"""Проверка контура участка перед сохранением.

Pydantic проверяет структуру GeoJSON (app/schemas/geojson.py), а здесь PostGIS
проверяет топологию: контур без самопересечений и без наложения на другие
участки того же владельца."""

import json
import uuid
from dataclasses import dataclass
from typing import Any

from geoalchemy2 import functions as geo
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import and_, not_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import AppError, BadRequestError, ConflictError
from app.models.vineyard import Vineyard
from app.repositories.vineyard import VineyardRepository
from app.schemas.geojson import AreaGeometry
from app.services.boundary_files import BoundaryCandidate

AREA_GEOMETRY: TypeAdapter[AreaGeometry] = TypeAdapter(AreaGeometry)
MAX_NAME_LENGTH = 255

# Причины из ST_IsValidReason, которые пользователь может исправить сам
REASONS = {
    "Self-intersection": "Контур пересекает сам себя",
    "Ring Self-intersection": "Контур пересекает сам себя",
    "Too few points": "В контуре слишком мало точек",
    "Hole lies outside shell": "Вырез лежит за пределами участка",
    "Nested shells": "Одна часть участка лежит внутри другой",
    "Interior is disconnected": "Вырезы делят участок на несвязанные части",
}


def _reason_message(reason: str) -> str:
    # Причина приходит с координатами: "Self-intersection[28.86 47.14]"
    kind = reason.split("[", 1)[0].strip()
    return REASONS.get(kind, f"Контур некорректен: {reason}")


async def validate_geometry(
    session: AsyncSession,
    owner_id: uuid.UUID,
    geometry: dict[str, Any],
) -> None:
    geom = geo.ST_GeomFromGeoJSON(json.dumps(geometry))

    valid, reason = (
        await session.execute(select(geo.ST_IsValid(geom), geo.ST_IsValidReason(geom)))
    ).one()
    if not valid:
        raise BadRequestError(_reason_message(reason))

    # Касание по границе допустимо: соседние участки делят межу. Нельзя, чтобы
    # одна и та же земля попала в два участка — тогда дважды посчитаются и
    # площадь, и снимки, и риски
    overlap = and_(
        Vineyard.owner_id == owner_id,
        geo.ST_Intersects(Vineyard.geom, geom),
        not_(geo.ST_Touches(Vineyard.geom, geom)),
    )
    name = await session.scalar(
        select(Vineyard.name).where(overlap).order_by(Vineyard.name).limit(1)
    )
    if name is not None:
        raise ConflictError(f"Контур накладывается на участок «{name}»")


@dataclass(frozen=True)
class SkippedBoundary:
    index: int
    name: str | None
    reason: str


async def import_boundaries(
    session: AsyncSession, owner_id: uuid.UUID, candidates: list[BoundaryCandidate]
) -> tuple[list[Vineyard], list[SkippedBoundary]]:
    """Создаёт участки из контуров файла. Контур, который нельзя сохранить,
    пропускается с причиной, остальные создаются: один кривой объект в файле
    из сотни не должен отменять весь импорт."""
    repository = VineyardRepository(session)
    created: list[Vineyard] = []
    skipped: list[SkippedBoundary] = []
    for index, candidate in enumerate(candidates, start=1):
        if candidate.geometry is None:
            skipped.append(
                SkippedBoundary(index, candidate.name, candidate.problem or "")
            )
            continue
        try:
            geometry = AREA_GEOMETRY.validate_python(candidate.geometry).model_dump()
        except ValidationError:
            skipped.append(
                SkippedBoundary(index, candidate.name, "Контур не замкнут или неполон")
            )
            continue
        try:
            await validate_geometry(session, owner_id, geometry)
        except AppError as error:
            skipped.append(SkippedBoundary(index, candidate.name, error.detail))
            continue
        # Без названия в файле участок получает номер объекта
        name = (candidate.name or f"Участок {index}")[:MAX_NAME_LENGTH]
        created.append(await repository.create_from_geojson(owner_id, name, geometry))
    return created, skipped
