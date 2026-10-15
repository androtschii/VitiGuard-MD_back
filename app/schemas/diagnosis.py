from pydantic import Field

from app.schemas.base import ResponseSchema


class DiseaseProbability(ResponseSchema):
    disease: str = Field(examples=["powdery_mildew"])
    probability: float = Field(ge=0, le=1)


class DiagnosisRead(ResponseSchema):
    """Результат классификации фото листа."""

    model: str = Field(examples=["efficientnet_b0"])
    disease: str = Field(examples=["powdery_mildew"])
    confidence: float = Field(ge=0, le=1)
    # Модель не уверена: лист стоит переснять крупным планом при дневном свете
    low_confidence: bool
    top: list[DiseaseProbability]
