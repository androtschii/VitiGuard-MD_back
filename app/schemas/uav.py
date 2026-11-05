"""Pydantic-схемы для обработки данных БПЛА и инвентаризации выпадов лозы (sprint-30)."""

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.base import RequestSchema


class MultispectralPixelDTO(BaseModel):
    """Срез мультиспектральных каналов в точке шпалеры."""

    offset_m: float = Field(ge=0.0, description="Метраж от начала ряда (м)")
    green: float = Field(ge=0.0, le=1.0, description="Отражение в зелёном спектре")
    red: float = Field(ge=0.0, le=1.0, description="Отражение в красном спектре")
    red_edge: float = Field(
        ge=0.0, le=1.0, description="Отражение в канале RedEdge (~730 нм)"
    )
    nir: float = Field(ge=0.0, le=1.0, description="Отражение в ближнем ИК (~840 нм)")


class RowTransectDTO(BaseModel):
    """Мультиспектральный трансект отдельного ряда."""

    row_id: str = Field(description="Идентификатор ряда", examples=["row_north_07"])
    row_length_m: float = Field(gt=0.0, description="Длина ряда (м)")
    start_point: tuple[float, float] = Field(description="(lat, lon) начала ряда")
    end_point: tuple[float, float] = Field(description="(lat, lon) конца ряда")
    pixels: list[MultispectralPixelDTO]


class MissingVineGapDTO(BaseModel):
    """Обнаруженный разрыв в шпалере с выпавшими кустами."""

    gap_id: str
    row_id: str
    start_offset_m: float
    end_offset_m: float
    gap_length_m: float
    estimated_missing_vines: int
    gps_coordinate: tuple[float, float]


class CanopyStressZoneDTO(BaseModel):
    """Очаг хлороза / стресса хлорофилла."""

    zone_id: str
    row_id: str
    start_offset_m: float
    end_offset_m: float
    mean_ndre: float
    mean_ndvi: float
    stress_severity: str


class UAVAnalysisRequest(RequestSchema):
    """Запрос на детальный анализ мультиспектральных трансектов БПЛА."""

    field_id: str
    variety_name: str = Field(default="Cabernet Sauvignon")
    nominal_vine_spacing_m: float = Field(default=1.25, gt=0.5, le=3.0)
    canopy_ndvi_threshold: float = Field(default=0.40, ge=0.1, le=0.9)
    transects: list[RowTransectDTO]


class UAVAnalysisResponse(BaseModel):
    """Результаты ультравысокоточного БПЛА-анализа ряда."""

    field_id: str
    missing_gaps: list[MissingVineGapDTO]
    stress_zones: list[CanopyStressZoneDTO]
    total_missing_vines: int
    gaps_count: int
    geojson_gaps: dict[str, Any]


class ReplantingOrderRequest(RequestSchema):
    """Запрос на составление ведомости подсадки саженцев."""

    field_id: str
    variety_name: str
    total_nominal_vines: int = Field(
        gt=0, description="Плановое проектное число кустов на участке"
    )
    gaps: list[MissingVineGapDTO]


class ReplantingOrderResponse(BaseModel):
    """Итоговая спецификация заказа саженцев для питомника."""

    field_id: str
    variety_name: str
    total_nominal_vines: int
    total_missing_vines: int
    mortality_rate_pct: float
    replanting_order_vines_count: int
    gaps_count: int
    geojson_missing_points: dict[str, Any]
