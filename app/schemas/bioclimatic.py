"""Pydantic-схемы для биоклиматических индексов, фенологии BBCH и окна сбора урожая."""

from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.base import RequestSchema


class PhenologyEstimateResponse(BaseModel):
    """Оценка текущей фенологической фазы развития виноградной лозы."""

    current_bbch: int = Field(
        description="Код текущей фазы по шкале BBCH (0-99)", examples=[75]
    )
    current_stage_name: str = Field(
        description="Название текущей фазы",
        examples=["Смыкание ягод (размер «горошина»)"],
    )
    accumulated_gdd: float = Field(
        description="Накопленная сумма активных температур (GDD, база 10 °C)",
        examples=[820.5],
    )
    next_bbch: int | None = Field(
        default=None, description="Код следующей фазы BBCH", examples=[81]
    )
    next_stage_name: str | None = Field(
        default=None,
        description="Название следующей фазы",
        examples=["Начало созревания (веaison)"],
    )
    gdd_to_next: float = Field(
        description="Требуется градусов-дней до следующей фазы",
        examples=[160.0],
    )
    progress_percentage: float = Field(
        description="Процент прохождения текущей межфазной стадии (0-100 %)",
        examples=[45.2],
    )
    is_season_complete: bool = Field(
        description="Завершён ли вегетационный цикл лозы", examples=[False]
    )
    variety_name: str = Field(description="Определённый сорт винограда")


class WinklerIndexResponse(BaseModel):
    """Индекс тепла Винклера (GDD за 01.04 - 31.10)."""

    gdd_sum: float = Field(description="Сумма активных температур", examples=[1720.5])
    region_class: str = Field(description="Класс региона", examples=["Region III"])
    region_title: str = Field(
        description="Характеристика климатической зоны",
        examples=["Тёплый климат (Warm)"],
    )
    suitable_varieties: list[str] = Field(description="Рекомендуемые сорта винограда")


class HuglinIndexResponse(BaseModel):
    """Гелиотермический индекс Гуглина (HI)."""

    hi_value: float = Field(description="Значение индекса HI", examples=[2140.0])
    thermal_class: str = Field(
        description="Класс теплообеспеченности", examples=["HI-4 (Тёплый)"]
    )
    thermal_description: str = Field(
        description="Агрономическая интерпретация условий созревания"
    )


class CoolNightIndexResponse(BaseModel):
    """Индекс прохладных ночей (CI, средняя T_min сентября)."""

    ci_temperature: float = Field(
        description="Средний ночной минимум сентября (°C)", examples=[13.2]
    )
    night_class: str = Field(
        description="Класс ночных температур", examples=["CI+1 (Прохладные ночи)"]
    )
    aroma_color_potential: str = Field(
        description="Потенциал синтеза антоцианов и сохранения свежей кислотности"
    )


class SelyaninovHtcResponse(BaseModel):
    """Гидротермический коэффициент Селянинова (ГТК)."""

    htc_value: float = Field(description="Значение коэффициента ГТК", examples=[0.85])
    moisture_class: str = Field(
        description="Зона увлажнения", examples=["Слабозасушливая зона"]
    )
    irrigation_need: str = Field(
        description="Рекомендации по водному режиму и орошению"
    )


class TerroirBioclimaticProfileResponse(BaseModel):
    """Комплексный биоклиматический паспорт терруара."""

    winkler: WinklerIndexResponse
    huglin: HuglinIndexResponse
    cool_night: CoolNightIndexResponse
    selyaninov: SelyaninovHtcResponse


class HarvestWindowForecastRequest(RequestSchema):
    """Запрос на расчёт оптимального окна сбора урожая."""

    target_sugar_brix: float | None = Field(
        default=None,
        description="Желаемая сахаристость ягод (°Brix). Если не указана, берётся эталон для сорта.",
        examples=[22.5],
    )
    current_sugar_brix: float | None = Field(
        default=None,
        description="Фактическая сахаристость ягод по полевому рефрактометру (°Brix).",
        examples=[18.4],
    )


class HarvestWindowForecastResponse(BaseModel):
    """Прогноз окна сбора урожая и сопутствующих агрорисков."""

    variety_name: str
    target_sugar_brix: float
    predicted_brix_trajectory: list[dict[str, Any]]
    optimal_window_start: date | None
    optimal_window_end: date | None
    harvest_urgency: str
    risks_detected: list[str]
    agronomic_recommendation: str


class VarietyCatalogItemResponse(BaseModel):
    """Карточка сорта в справочнике термических потребностей."""

    name: str
    code: str
    berry_color: str
    target_brix_default: float
    description: str
    gdd_thresholds: dict[int, float]
