from pydantic import Field

from app.schemas.base import EntityResponse, RequestSchema
from app.schemas.geojson import AreaGeometry, MultiPolygonRead


class VineyardCreate(RequestSchema):
    name: str = Field(min_length=1, max_length=255)
    geometry: AreaGeometry


class VineyardRead(EntityResponse):
    name: str
    geometry: MultiPolygonRead
    # Площадь считает PostGIS по эллипсоиду WGS84 (ST_Area по geography)
    area_ha: float = Field(ge=0)
