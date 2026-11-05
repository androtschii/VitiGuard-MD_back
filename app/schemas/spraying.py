"""Pydantic-схемы для дифференцированного опрыскивания (VRA) и планирования обработок (sprint-28)."""

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.base import RequestSchema


class SprayerEquipmentDTO(BaseModel):
    """Параметры и технические характеристики садового/виноградникового опрыскивателя."""

    sprayer_id: str = Field(
        description="Идентификатор опрыскивателя", examples=["berthoud-winair-01"]
    )
    nominal_application_rate_l_ha: float = Field(
        default=350.0,
        ge=50.0,
        le=1200.0,
        description="Базовая эталонная норма расхода рабочей жидкости (л/га)",
    )
    min_application_rate_l_ha: float = Field(
        default=120.0,
        ge=30.0,
        description="Минимально допустимая норма расхода с учётом каплеобразования (л/га)",
    )
    max_application_rate_l_ha: float = Field(
        default=550.0,
        le=1500.0,
        description="Максимальная норма расхода при сильном поражении (л/га)",
    )
    boom_width_m: float = Field(
        default=2.5, gt=0.5, description="Ширина захвата (междурядье, м)"
    )
    travel_speed_kmh: float = Field(
        default=6.0, gt=1.0, le=15.0, description="Рабочая скорость агрегата (км/ч)"
    )
    chemical_concentration_g_l: float = Field(
        default=8.0,
        gt=0.0,
        description="Концентрация действующего вещества в рабочем растворе (г/л)",
    )
    chemical_price_per_kg_mdl: float = Field(
        default=450.0,
        ge=0.0,
        description="Стоимость 1 кг действующего вещества (MDL)",
    )


class SpraySectorDTO(BaseModel):
    """Сектор ряда под обработку."""

    sector_id: str
    row_number: int
    length_m: float
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    canopy_height_m: float = Field(
        default=1.4, ge=0.4, le=2.5, description="Высота листовой стены (м)"
    )
    canopy_width_m: float = Field(
        default=0.45, ge=0.1, le=1.2, description="Толщина листового полога (м)"
    )
    ndvi_index: float = Field(
        default=0.65, ge=0.0, le=1.0, description="Индекс вегетации (Sentinel-2 / БПЛА)"
    )
    disease_pressure_score: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Оценка инфекционного фона (0..1: оидиум, милдью, серая гниль)",
    )


class PrescriptionZoneDTO(BaseModel):
    """Расчётная зона дифференцированной нормы внесения."""

    sector_id: str
    target_rate_l_ha: float
    dosage_multiplier: float
    target_flow_rate_l_min: float
    tree_row_volume_m3_ha: float
    active_ingredient_rate_kg_ha: float
    status: str = Field(default="ok", description="ok | capped_min | capped_max")
    geometry_geojson: dict[str, Any]


class VRAOptimizationReportDTO(BaseModel):
    """Сравнительная экономика и экологический эффект VRA по сравнению со сплошным внесением."""

    treated_area_ha: float
    uniform_liquid_total_liters: float
    vra_liquid_total_liters: float
    liquid_saved_liters: float
    liquid_saved_pct: float
    uniform_chemical_total_kg: float
    vra_chemical_total_kg: float
    chemical_saved_kg: float
    chemical_saved_pct: float
    cost_savings_mdl: float


class VRAPrescriptionRequest(RequestSchema):
    """Запрос на расчёт карты дифференцированного внесения СЗР."""

    equipment: SprayerEquipmentDTO
    sectors: list[SpraySectorDTO]


class VRAPrescriptionResponse(BaseModel):
    """Карта дифференцированного внесения СЗР."""

    prescription_zones: list[PrescriptionZoneDTO]
    summary: VRAOptimizationReportDTO
    geojson_feature_collection: dict[str, Any]


class IsoXmlExportRequest(RequestSchema):
    """Запрос на экспорт карты задач в стандарт ISO 11783-10 (TaskData XML)."""

    task_name: str = Field(
        default="VitiGuard_VRA_Task", description="Название технологической операции"
    )
    field_name: str = Field(
        default="Vineyard_Sector_A", description="Название поля/участка"
    )
    equipment: SprayerEquipmentDTO
    sectors: list[SpraySectorDTO]


class IsoXmlExportResponse(BaseModel):
    """Результат экспорта в ISO-XML."""

    xml_content: str
    filename: str
    zones_count: int
