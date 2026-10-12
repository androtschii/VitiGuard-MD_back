import json
import uuid
from typing import Any

from geoalchemy2 import functions as geo
from sqlalchemy import select

from app.models.vineyard import Vineyard
from app.repositories.base import Repository


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
