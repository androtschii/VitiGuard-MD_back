import uuid
from datetime import date

from pydantic import Field

from app.models.enums import Disease, Fungicide
from app.schemas.base import EntityResponse, RequestSchema, ResponseSchema
from app.services.treatments import Action


class TreatmentCreate(RequestSchema):
    applied_on: date
    fungicide: Fungicide
    product: str | None = Field(default=None, max_length=200, examples=["Ридомил Голд"])


class TreatmentRead(EntityResponse):
    vineyard_id: uuid.UUID
    applied_on: date
    fungicide: Fungicide
    product: str | None


class RecommendationRead(ResponseSchema):
    action: Action
    disease: Disease
    # null — осмотр, препарат не нужен
    fungicide: Fungicide | None
    due_by: date
    risk_day: date
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
