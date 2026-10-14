import uuid
from datetime import date
from typing import Annotated

from pydantic import AfterValidator, Field, field_validator

from app.models.variety import BerryColor
from app.schemas.base import EntityResponse, RequestSchema, ResponseSchema
from app.schemas.geojson import AreaGeometry, MultiPolygonRead


def _not_in_future(year: int) -> int:
    # Будущий год — опечатка; посадку, которая только планируется, участком не заводят
    if year > date.today().year:
        raise ValueError("Год посадки не может быть в будущем")
    return year


PlantingYear = Annotated[int, Field(ge=1900), AfterValidator(_not_in_future)]


class VineyardCreate(RequestSchema):
    name: str = Field(min_length=1, max_length=255)
    geometry: AreaGeometry
    variety_id: uuid.UUID | None = None
    planting_year: PlantingYear | None = None


class VineyardUpdate(RequestSchema):
    """Частичное изменение: меняются только переданные поля. null в сорте или
    годе посадки — «не указано», это тоже изменение."""

    name: str | None = Field(default=None, min_length=1, max_length=255)
    geometry: AreaGeometry | None = None
    variety_id: uuid.UUID | None = None
    planting_year: PlantingYear | None = None

    @field_validator("name", "geometry")
    @classmethod
    def required_when_passed[T](cls, value: T | None) -> T:
        # Название и контур у участка есть всегда: их можно заменить, но не стереть
        if value is None:
            raise ValueError("Поле нельзя очистить")
        return value


class VarietyBrief(ResponseSchema):
    """Сорт в карточке участка — без агрономических подробностей справочника."""

    id: uuid.UUID
    name: str
    berry_color: BerryColor


class VineyardRead(EntityResponse):
    name: str
    geometry: MultiPolygonRead
    # Площадь считает PostGIS по эллипсоиду WGS84 (ST_Area по geography)
    area_ha: float = Field(ge=0)
    variety: VarietyBrief | None
    planting_year: int | None


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


class SkippedBoundaryRead(ResponseSchema):
    # Номер объекта в файле, с единицы — как его видит человек в QGIS
    index: int
    name: str | None
    reason: str


class BoundaryImportResult(ResponseSchema):
    created: list[VineyardRead]
    skipped: list[SkippedBoundaryRead]


class VineyardContextRead(ResponseSchema):
    """Местность вокруг участка по данным OpenStreetMap."""

    osm_overlap_ratio: float = Field(ge=0, le=1)
    osm_vineyards_within_1km: int = Field(ge=0)
    nearest_osm_distance_m: float | None
