from datetime import date, datetime
from typing import Any

from pydantic import Field

from app.models.enums import Disease, RiskLevel
from app.schemas.base import ResponseSchema


class RiskRead(ResponseSchema):
    """Риск болезни на участке на дату по одной агромодели."""

    target_date: date
    disease: Disease
    method: str = Field(examples=["three_tens", "goidanich", "gubler_thomas"])
    score: float = Field(ge=0, le=1)
    level: RiskLevel
    # Входные значения модели: из чего получилась оценка
    factors: dict[str, Any] | None
    updated_at: datetime
