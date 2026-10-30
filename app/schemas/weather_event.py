import uuid
from datetime import date
from typing import Literal

from pydantic import Field

from app.schemas.base import EntityResponse, RequestSchema

Kind = Literal["frost", "hail", "wind", "drought", "heat"]


class WeatherEventCreate(RequestSchema):
    """Погодное событие на участке."""

    occurred_on: date = Field(description="День события, не позже сегодняшнего")
    kind: Kind = Field(description="Что случилось", examples=["frost"])
    severity: int = Field(ge=1, le=3, description="1 — слабо, 2 — заметно, 3 — сильно")
    note: str | None = Field(default=None, max_length=500)


class WeatherEventRead(EntityResponse):
    """Погодное событие из журнала участка."""

    vineyard_id: uuid.UUID
    occurred_on: date
    kind: str
    severity: int
    note: str | None
