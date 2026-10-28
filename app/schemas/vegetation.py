import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.base import ResponseSchema


class IndexValueRead(ResponseSchema):
    """Индекс по пикселям участка без облаков: среднее, разброс и пределы."""

    mean: float = Field(ge=-1, le=1)
    min: float = Field(ge=-1, le=1)
    max: float = Field(ge=-1, le=1)
    std: float = Field(ge=0)


class NdviMapRead(ResponseSchema):
    """Углы картограммы NDVI на карте; сам PNG — GET .../vegetation/{id}/ndvi.png."""

    # [долгота, широта]: верхний левый, верхний правый, нижний правый, нижний левый
    corners: list[list[float]] = Field(min_length=4, max_length=4)


class VegetationObservationRead(ResponseSchema):
    """Один снимок Sentinel-2 по участку."""

    id: uuid.UUID
    acquired_at: datetime
    platform: str | None = Field(examples=["sentinel-2a"])
    # Облачность всей сцены и над самим участком, %
    scene_cloud_cover: float = Field(ge=0, le=100)
    parcel_cloud_cover: float | None = Field(ge=0, le=100)
    # None — участок был закрыт облаками
    ndvi: IndexValueRead | None
    ndre: IndexValueRead | None
    # None — участок был закрыт облаками или снимок обработан до картограмм
    ndvi_map: NdviMapRead | None


class NdviAnomalyRead(ResponseSchema):
    """Снимок с просадкой NDVI: ниже своей нормы для этих дней и ниже соседей."""

    acquired_at: datetime
    ndvi: float = Field(ge=-1, le=1)
    # Насколько NDVI ниже медианы своих снимков ±20 дней
    below_norm: float
    # Насколько ниже медианы других участков ±3 дня; None — их снимков мало
    below_peers: float | None
    # Просадка в разбросах отклонений участка от своей нормы
    z: float
    # suspected — один снимок; confirmed — предыдущий чистый снимок тоже с просадкой
    level: Literal["suspected", "confirmed"]


class VegetationRefreshStarted(BaseModel):
    """Пересчёт индексов вегетации по новым снимкам поставлен в очередь."""

    task_id: str
