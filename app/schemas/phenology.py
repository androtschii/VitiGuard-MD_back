import uuid
from datetime import date
from typing import Literal

from pydantic import Field

from app.schemas.base import EntityResponse, RequestSchema

# Основные фазы шкалы Эйхгорна–Ленца: от зимней почки до начала листопада
Stage = Literal[1, 4, 12, 19, 23, 27, 31, 35, 38, 43]


class PhenologyCreate(RequestSchema):
    """Отметка фазы развития лозы."""

    observed_on: date = Field(description="День наблюдения, не позже сегодняшнего")
    stage: Stage = Field(description="Фаза по шкале Эйхгорна–Ленца", examples=[23])


class PhenologyRead(EntityResponse):
    """Отмеченная фаза развития лозы."""

    vineyard_id: uuid.UUID
    observed_on: date
    stage: int
