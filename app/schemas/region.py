from pydantic import Field

from app.schemas.base import ResponseSchema


class DistrictRef(ResponseSchema):
    """Район Молдовы: код ISO 3166-2 и название."""

    iso: str = Field(examples=["MD-CU"])
    name: str = Field(examples=["Chișinău"])


class DistrictStats(DistrictRef):
    """Виноградарство района: по OpenStreetMap и по участкам пользователя."""

    area_ha: float = Field(ge=0)
    osm_vineyards: int = Field(ge=0)
    osm_vineyard_area_ha: float = Field(ge=0)
    # Доля площади района под виноградниками по OSM, 0..1
    osm_vineyard_share: float = Field(ge=0, le=1)
    my_vineyards: int = Field(ge=0)
    my_vineyard_area_ha: float = Field(ge=0)
