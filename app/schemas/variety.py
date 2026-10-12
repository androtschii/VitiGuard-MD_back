from pydantic import Field, field_validator

from app.models.variety import BerryColor, Susceptibility
from app.schemas.base import EntityResponse, RequestSchema


class VarietyCreate(RequestSchema):
    name: str = Field(min_length=1, max_length=100)
    berry_color: BerryColor
    downy_mildew_susceptibility: Susceptibility | None = None
    powdery_mildew_susceptibility: Susceptibility | None = None
    description: str | None = Field(default=None, max_length=2000)


class VarietyUpdate(RequestSchema):
    """Частичное изменение: меняются только переданные поля. null в
    восприимчивости — «неизвестно», это тоже изменение."""

    name: str | None = Field(default=None, min_length=1, max_length=100)
    berry_color: BerryColor | None = None
    downy_mildew_susceptibility: Susceptibility | None = None
    powdery_mildew_susceptibility: Susceptibility | None = None
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("name", "berry_color")
    @classmethod
    def required_when_passed[T](cls, value: T | None) -> T:
        # Название и цвет ягоды у сорта есть всегда: их можно заменить, но не стереть
        if value is None:
            raise ValueError("Поле нельзя очистить")
        return value


class VarietyRead(EntityResponse):
    name: str
    berry_color: BerryColor
    downy_mildew_susceptibility: Susceptibility | None
    powdery_mildew_susceptibility: Susceptibility | None
    description: str | None
