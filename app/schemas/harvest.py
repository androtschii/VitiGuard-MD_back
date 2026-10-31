import uuid

from pydantic import Field

from app.schemas.base import EntityResponse, RequestSchema


class HarvestSave(RequestSchema):
    """Прогноз и факт урожая участка за сезон."""

    season: int = Field(ge=1950, le=2100, examples=[2026])
    estimated_kg: float | None = Field(default=None, ge=0, le=10_000_000)
    actual_kg: float | None = Field(default=None, ge=0, le=10_000_000)
    sugar_brix: float | None = Field(
        default=None, ge=0, le=45, description="Сахар в ягоде при сборе, °Brix"
    )


class HarvestRead(EntityResponse):
    """Урожай участка за сезон."""

    vineyard_id: uuid.UUID
    season: int
    estimated_kg: float | None
    actual_kg: float | None
    sugar_brix: float | None
