"""Pydantic-схемы для беспроводной сенсорной сети (LoRaWAN) и мониторинга микроклимата (sprint-29)."""

from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.base import RequestSchema


class SoilMoistureProbeDTO(BaseModel):
    """Показания многоуровневого почвенного зонда."""

    depth_10cm_pct: float = Field(
        ge=0.0, le=100.0, description="Влажность почвы в верхнем слое (10 см, % VWC)"
    )
    depth_30cm_pct: float = Field(
        ge=0.0, le=100.0, description="Влажность в корневой зоне (30 см, % VWC)"
    )
    depth_60cm_pct: float = Field(
        ge=0.0, le=100.0, description="Глубинный запас влаги (60 см, % VWC)"
    )
    soil_temp_10cm_c: float = Field(
        default=15.0, description="Температура почвы на глубине 10 см (°C)"
    )


class SensorTelemetryDTO(RequestSchema):
    """Телеметрический пакет от полевого датчика LoRaWAN."""

    sensor_id: str = Field(
        description="DevEUI или идентификатор датчика",
        examples=["lora-node-purcari-04"],
    )
    vineyard_id: str = Field(description="Идентификатор виноградника")
    timestamp: datetime = Field(description="Время фиксации замера")
    lat: float = Field(ge=-90.0, le=90.0, description="Широта")
    lon: float = Field(ge=-180.0, le=180.0, description="Долгота")
    elevation_m: float = Field(default=120.0, description="Высота над уровнем моря (м)")
    canopy_temperature_c: float = Field(description="Температура в кроне куста (°C)")
    canopy_humidity_pct: float = Field(
        ge=0.0, le=100.0, description="Относительная влажность в кроне (%)"
    )
    leaf_wetness_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Сенсор увлажнения листа (0 - абсолютно сухой, 100 - капельная влага)",
    )
    solar_radiation_w_m2: float = Field(
        default=0.0, ge=0.0, description="Солнечная радиация (Вт/м²)"
    )
    barometric_pressure_hpa: float = Field(
        default=1013.25, description="Атмосферное давление (гПа)"
    )
    battery_v: float = Field(
        default=3.6, ge=2.0, le=4.5, description="Напряжение аккумулятора узла (В)"
    )
    soil: SoilMoistureProbeDTO | None = None


class SpatialPointDTO(BaseModel):
    """Точка пространственной сетки микроклимата."""

    lat: float
    lon: float
    elevation_m: float
    interpolated_temp_c: float
    interpolated_humidity_pct: float
    frost_pocket_risk: bool


class FrostRiskReportDTO(BaseModel):
    """Оценка риска радиационных и адвективных заморозков."""

    vineyard_id: str
    risk_level: str = Field(description="none | low | moderate | critical")
    min_canopy_temp_c: float
    wet_bulb_temp_c: float
    has_thermal_inversion: bool
    inversion_delta_c: float = Field(
        description="Разность температур между вершиной склона и низиной"
    )
    critical_phenology_stage: str = Field(
        description="Фаза лозы (напр. BBCH 09 - распускание почек)"
    )
    frost_pocket_sensors: list[str]
    recommended_mitigation: list[str]
    estimated_window_hours: float


class HailStormRiskReportDTO(BaseModel):
    """Оценка риска конвективного градового шторма."""

    vineyard_id: str
    hail_risk_level: str = Field(description="low | moderate | high | severe")
    pressure_drop_rate_hpa_h: float
    convective_energy_index: float
    warning_message: str
    netting_action_required: bool


class VineyardMeshStatusResponse(BaseModel):
    """Сводный статус сенсорной сети виноградника."""

    vineyard_id: str
    active_sensors_count: int
    sensors: list[SensorTelemetryDTO]
    mean_canopy_temp_c: float
    mean_canopy_humidity_pct: float
    mean_leaf_wetness_pct: float
    spatial_grid: list[SpatialPointDTO]
