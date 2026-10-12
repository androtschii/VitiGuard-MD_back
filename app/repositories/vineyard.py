import json
import uuid
from typing import Any, NamedTuple

from geoalchemy2 import Geography
from geoalchemy2 import functions as geo
from sqlalchemy import Row, cast, func, select

from app.core.errors import NotFoundError
from app.models.vineyard import Vineyard
from app.repositories.base import Repository
from app.schemas.pagination import PageParams

SQUARE_METERS_IN_HECTARE = 10_000

# Площадь по эллипсоиду WGS84: geography считает в квадратных метрах на
# поверхности Земли, а не в «квадратных градусах», как geometry в SRID 4326
AREA_HA = geo.ST_Area(cast(Vineyard.geom, Geography)) / SQUARE_METERS_IN_HECTARE


class VineyardView(NamedTuple):
    """Участок в том виде, в каком его отдаёт API: с контуром в GeoJSON и площадью."""

    vineyard: Vineyard
    geometry: dict[str, Any]
    area_ha: float


def _view(row: Row[Vineyard, str, float]) -> VineyardView:
    vineyard, geojson, area_ha = row
    return VineyardView(vineyard, json.loads(geojson), float(area_ha))


# Контур и площадь берутся в том же запросе, что и участок
VIEW_COLUMNS = (Vineyard, geo.ST_AsGeoJSON(Vineyard.geom), AREA_HA)


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

    async def view(self, vineyard: Vineyard) -> VineyardView:
        row = (
            await self.session.execute(
                select(*VIEW_COLUMNS).where(Vineyard.id == vineyard.id)
            )
        ).one()
        return _view(row)

    async def list_for_owner(
        self, owner_id: uuid.UUID, params: PageParams
    ) -> tuple[list[VineyardView], int]:
        """Страница участков владельца вместе с контурами и площадями — одним
        запросом, а не отдельным запросом для каждого участка."""
        rows = await self.session.execute(
            select(*VIEW_COLUMNS)
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
        return [_view(row) for row in rows], total or 0

    async def get_for_owner(
        self, owner_id: uuid.UUID, vineyard_id: uuid.UUID
    ) -> VineyardView:
        """Участок владельца. Чужой участок неотличим от несуществующего: ответ
        «нет доступа» подтвердил бы, что участок с таким id есть."""
        row = (
            await self.session.execute(
                select(*VIEW_COLUMNS).where(
                    Vineyard.id == vineyard_id, Vineyard.owner_id == owner_id
                )
            )
        ).first()
        if row is None:
            raise NotFoundError(self.not_found_message)
        return _view(row)
