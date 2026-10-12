import json
import uuid
from typing import Any

from geoalchemy2 import functions as geo
from sqlalchemy import func, select

from app.core.errors import NotFoundError
from app.models.vineyard import Vineyard
from app.repositories.base import Repository
from app.schemas.pagination import PageParams

# Участок вместе с контуром в GeoJSON — так его отдаёт API
type VineyardWithGeometry = tuple[Vineyard, dict[str, Any]]


class VineyardRepository(Repository[Vineyard]):
    model = Vineyard
    not_found_message = "Участок не найден"

    async def create_from_geojson(
        self, owner_id: uuid.UUID, name: str, geometry: dict[str, Any]
    ) -> Vineyard:
        # GeoJSON разбирает сам PostGIS (SRID 4326 по RFC 7946), а ST_Multi
        # приводит Polygon к MultiPolygon — тип столбца
        geom = geo.ST_Multi(geo.ST_GeomFromGeoJSON(json.dumps(geometry)))
        return await self.create(owner_id=owner_id, name=name, geom=geom)

    async def geojson(self, vineyard: Vineyard) -> dict[str, Any]:
        text = await self.session.scalar(
            select(geo.ST_AsGeoJSON(Vineyard.geom)).where(Vineyard.id == vineyard.id)
        )
        parsed: dict[str, Any] = json.loads(text or "null")
        return parsed

    async def list_for_owner(
        self, owner_id: uuid.UUID, params: PageParams
    ) -> tuple[list[VineyardWithGeometry], int]:
        """Страница участков владельца вместе с контурами — одним запросом,
        а не отдельным запросом за контуром каждого участка."""
        rows = await self.session.execute(
            select(Vineyard, geo.ST_AsGeoJSON(Vineyard.geom))
            .where(Vineyard.owner_id == owner_id)
            # По названию — так пользователь ищет участок глазами; id — для
            # стабильного порядка участков с одинаковыми названиями
            .order_by(Vineyard.name, Vineyard.id)
            .limit(params.size)
            .offset(params.offset)
        )
        total = await self.session.scalar(
            select(func.count())
            .select_from(Vineyard)
            .where(Vineyard.owner_id == owner_id)
        )
        items = [(vineyard, json.loads(text)) for vineyard, text in rows]
        return items, total or 0

    async def get_for_owner(
        self, owner_id: uuid.UUID, vineyard_id: uuid.UUID
    ) -> VineyardWithGeometry:
        """Участок владельца. Чужой участок неотличим от несуществующего: ответ
        «нет доступа» подтвердил бы, что участок с таким id есть."""
        row = (
            await self.session.execute(
                select(Vineyard, geo.ST_AsGeoJSON(Vineyard.geom)).where(
                    Vineyard.id == vineyard_id, Vineyard.owner_id == owner_id
                )
            )
        ).first()
        if row is None:
            raise NotFoundError(self.not_found_message)
        vineyard, text = row
        return vineyard, json.loads(text)
