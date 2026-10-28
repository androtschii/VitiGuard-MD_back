import uuid
from datetime import date

from pydantic import Field

from app.models.enums import Disease, Fungicide
from app.schemas.base import EntityResponse, RequestSchema, ResponseSchema
from app.services.treatments import Action


class TreatmentCreate(RequestSchema):
    """Запись о проведённой обработке участка."""

    applied_on: date = Field(description="Дата обработки, не позже сегодняшней")
    fungicide: Fungicide = Field(
        description="Тип препарата по действию: от него зависит срок защиты"
    )
    product: str | None = Field(
        default=None,
        max_length=200,
        examples=["Ридомил Голд"],
        description="Название препарата для журнала, в расчёт не входит",
    )


class TreatmentRead(EntityResponse):
    """Обработка из журнала участка."""

    vineyard_id: uuid.UUID
    applied_on: date
    fungicide: Fungicide
    product: str | None


class RecommendationRead(ResponseSchema):
    """Рекомендация плана: что сделать, против какой болезни и до какого дня."""

    action: Action
    disease: Disease
    # null — осмотр, препарат не нужен
    fungicide: Fungicide | None
    due_by: date = Field(description="Последний день, когда действие ещё имеет смысл")
    risk_day: date = Field(description="День риска, к которому относится рекомендация")
    reason: str = Field(
        examples=[
            "downy_infection_forecast",
            "downy_infection_past",
            "downy_symptoms",
            "powdery_high",
            "powdery_high_hot",
        ]
    )


class TreatmentPlanRead(ResponseSchema):
    """Что делать на неделе и до какого дня участок защищён."""

    today: date
    protected_until: dict[Disease, date | None]
    recommendations: list[RecommendationRead]
