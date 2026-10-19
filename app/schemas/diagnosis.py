from pydantic import Field

from app.schemas.base import ResponseSchema


class DiseaseProbability(ResponseSchema):
    disease: str = Field(examples=["powdery_mildew"])
    probability: float = Field(ge=0, le=1)


class LesionRead(ResponseSchema):
    """Поражённые участки листа по сегментации U-Net."""

    # Прозрачный PNG (data URL) с пропорциями исходного фото: красным — поражение
    mask: str
    # Доля поражённой площади листа, 0..1
    severity: float = Field(ge=0, le=1)


class ClassifierModelRead(ResponseSchema):
    """Классификатор для диагностики в браузере без сети (frontend/pr-071)."""

    name: str = Field(examples=["efficientnet_b0"])
    # Порядок классов — порядок выходов модели
    classes: list[str]
    image_size: int = Field(examples=[224])
    # Нормализация входа по каналам RGB, как при обучении
    mean: list[float]
    std: list[float]
    version: str = Field(examples=["3f1c9a0b7d2e4f60"])
    size_bytes: int


class DiagnosisRead(ResponseSchema):
    """Результат классификации фото листа."""

    model: str = Field(examples=["efficientnet_b0"])
    disease: str = Field(examples=["powdery_mildew"])
    confidence: float = Field(ge=0, le=1)
    # Модель не уверена: лист стоит переснять крупным планом при дневном свете
    low_confidence: bool
    top: list[DiseaseProbability]
    # Тепловая карта: какие области фото повлияли на ответ (прозрачный PNG,
    # data URL); None — модель экспортирована без карты активации
    heatmap: str | None
    # None — модель сегментации не загружена на сервер
    lesion: LesionRead | None
