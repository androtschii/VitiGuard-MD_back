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


class NearbyQuery(RequestSchema):
    """Точка, от которой ищутся участки: например, геолокация из Telegram-бота."""

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    limit: int = Field(default=5, ge=1, le=20)
    # Дальше 50 км «ближайший участок» уже не про то, где стоит пользователь
    max_distance_km: float = Field(default=50, gt=0, le=500)


class VineyardNearby(VineyardRead):
    # 0 — точка внутри участка
    distance_m: float = Field(ge=0)
