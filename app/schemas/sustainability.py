"""Pydantic-схемы для расчёта углеродного баланса (OIV GHG) и экологического аудита (sprint-31)."""

from pydantic import BaseModel, Field

from app.schemas.base import RequestSchema


class VineyardPracticesDTO(BaseModel):
    """Агротехнические практики и параметры виноградника за сезон."""

    area_ha: float = Field(gt=0.0, description="Площадь участка (га)")
    vines_per_ha: int = Field(
        default=3333, gt=500, le=10000, description="Плотность посадки (кустов/га)"
    )
    vine_age_years: int = Field(
        default=12, ge=1, le=100, description="Возраст виноградника (лет)"
    )
    inter_row_cover_crop: str = Field(
        default="permanent_grass",
        description="Залужение междурядий: bare_soil (пар), spontaneous (сорняки), permanent_grass (многолетние злаки), legumes (бобовые смеси)",
    )
    pruning_residue_management: str = Field(
        default="mulched_in_situ",
        description="Обращение с лозой после обрезки: burned (сжигание), removed (вывоз), mulched_in_situ (измельчение в междурядье), biochar (биоуголь)",
    )
    diesel_tractor_liters_per_ha: float = Field(
        default=95.0,
        ge=0.0,
        le=500.0,
        description="Расход дизельного топлива на обработку (л/га/год)",
    )
    electric_robot_operations: bool = Field(
        default=False,
        description="Применение автономных электрических роботов-комбайнов и опрыскивателей",
    )
    nitrogen_fertilizer_kg_n_ha: float = Field(
        default=30.0,
        ge=0.0,
        le=200.0,
        description="Внесение азотных удобрений (кг действующего N/га/год)",
    )
    copper_applied_kg_ha: float = Field(
        default=2.4,
        ge=0.0,
        le=20.0,
        description="Суммарно внесено меди за сезон (кг Cu/га/год)",
    )
    synthetic_herbicides_used: bool = Field(
        default=False,
        description="Использование химических гербицидов (глифосат и др.) в приствольной полосе",
    )


class CarbonBreakdownDTO(BaseModel):
    """Детализация структуры углеродного баланса (т CO2-экв / га / год)."""

    vine_woody_biomass_sequestration: float
    soil_cover_crop_sequestration: float
    pruning_mulch_retention: float
    total_sequestration_tco2e_ha: float

    diesel_tractor_emissions_tco2e_ha: float
    electric_equipment_emissions_tco2e_ha: float
    nitrogen_n2o_emissions_tco2e_ha: float
    total_emissions_tco2e_ha: float

    net_carbon_balance_tco2e_ha: float


class CarbonBalanceResponse(BaseModel):
    """Сводный отчёт углеродного баланса участка по протоколу OIV."""

    vineyard_id: str
    area_ha: float
    breakdown_per_ha: CarbonBreakdownDTO
    total_farm_net_tco2e_year: float
    is_carbon_sink: bool
    electric_robot_diesel_reduction_pct: float
    potential_carbon_credits_eur: float
    oiv_compliance_standard: str = "OIV-CST 503-2016"


class CarbonBalanceRequest(RequestSchema):
    """Запрос на расчёт углеродного баланса виноградника."""

    vineyard_id: str
    practices: VineyardPracticesDTO
    carbon_credit_price_eur_ton: float = Field(
        default=35.0, ge=0.0, description="Рыночная цена углеродной квоты (€/т)"
    )


class OrganicAuditResponse(BaseModel):
    """Результат аудита соответствия регламентам органического виноделия (EU 2018/848)."""

    vineyard_id: str
    compliance_score_pct: float
    is_organic_certified_ready: bool
    copper_limit_status: str  # ok | warning | exceeded
    copper_applied_kg_ha: float
    herbicide_free: bool
    biodiversity_cover_crop_status: str
    deficiencies: list[str]
    recommendations: list[str]


class OrganicAuditRequest(RequestSchema):
    """Запрос на экологический аудит органической сертификации."""

    vineyard_id: str
    practices: VineyardPracticesDTO


class RegionalCarbonBenchmarkDTO(BaseModel):
    """Региональные эталонные бенчмарки виноградарства Молдовы и Европы."""

    region_name: str
    conventional_net_tco2e_ha: float
    regenerative_net_tco2e_ha: float
    mean_diesel_l_ha: float
    recommended_cover_crop: str
