from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.base import ResponseSchema


class IndexValueRead(ResponseSchema):
    """Индекс по пикселям участка без облаков: среднее, разброс и пределы."""

    mean: float = Field(ge=-1, le=1)
    min: float = Field(ge=-1, le=1)
    max: float = Field(ge=-1, le=1)
    std: float = Field(ge=0)


class VegetationObservationRead(ResponseSchema):
    """Один снимок Sentinel-2 по участку."""

    acquired_at: datetime
    platform: str | None = Field(examples=["sentinel-2a"])
    # Облачность всей сцены и над самим участком, %
    scene_cloud_cover: float = Field(ge=0, le=100)
    parcel_cloud_cover: float | None = Field(ge=0, le=100)
    # None — участок был закрыт облаками
    ndvi: IndexValueRead | None
    ndre: IndexValueRead | None


class VegetationRefreshStarted(BaseModel):
    task_id: str
