import uuid
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


class TodayRiskRead(ResponseSchema):
    """Уровни риска участка на сегодня — для раскраски участков на карте."""

    vineyard_id: uuid.UUID
    day: date
    downy: RiskLevel
    powdery: RiskLevel


class OutlookDayRead(ResponseSchema):
    """Вероятность вспышки в день; null — день вне сезонного окна моделей."""

    day: date
    downy: float | None = Field(ge=0, le=1)
    powdery: float | None = Field(ge=0, le=1)


class RiskOutlookRead(ResponseSchema):
    """Вероятность вспышки милдью и оидиума на 7 дней по ансамблевому прогнозу."""

    members: int = Field(ge=0, description="Членов ансамбля в расчёте")
    days: list[OutlookDayRead]
    downy_week: float | None = Field(ge=0, le=1)
    powdery_week: float | None = Field(ge=0, le=1)
