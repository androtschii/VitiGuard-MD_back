import uuid
from datetime import date
from typing import Literal

from pydantic import AwareDatetime, Field

from app.schemas.base import EntityResponse, RequestSchema, ResponseSchema
from app.schemas.pagination import PageParams

Overlay = Literal["heatmap", "lesion"]


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
    # Запись в истории проверок (save=true); None — фото не сохранялось
    id: uuid.UUID | None = None


class DiagnosisJobStarted(ResponseSchema):
    task_id: str


class DiagnosisJobRead(ResponseSchema):
    """Диагностика в фоне: pending — в очереди, running — идёт, done — готово,
    failed — не удалось."""

    status: Literal["pending", "running", "done", "failed"]
    result: DiagnosisRead | None = None


class ReviewRead(ResponseSchema):
    """Заключение агронома по проверке."""

    disease: str
    comment: str | None
    reviewed_at: AwareDatetime
    # Имя эксперта; None — его учётная запись удалена
    reviewer: str | None


class ReviewUpdate(RequestSchema):
    # Класс — как у модели: строчные латинские буквы и подчёркивание
    disease: str = Field(pattern=r"^[a-z_]{1,50}$", examples=["downy_mildew"])
    comment: str | None = Field(default=None, max_length=1000)


class DiagnosisRecordRead(EntityResponse):
    """Проверка из истории (pr-053): фото, диагноз и участок, если указан."""

    vineyard_id: uuid.UUID | None
    vineyard_name: str | None
    model: str
    model_version: str
    disease: str
    confidence: float = Field(ge=0, le=1)
    # Вероятности классов по убыванию
    top: list[DiseaseProbability]
    severity: float | None = Field(ge=0, le=1)
    # Сохранённые наложения: GET /diagnoses/{id}/overlays/{heatmap|lesion}
    overlays: list[Overlay]
    # None — агроном ещё не смотрел
    review: ReviewRead | None


class DiagnosisQuery(PageParams):
    """Страница истории и фильтры: ?disease=esca&date_from=2026-06-01&page=2."""

    disease: str | None = Field(default=None, max_length=50)
    vineyard_id: uuid.UUID | None = None
    # Границы включительно, по календарным дням UTC
    date_from: date | None = None
    date_to: date | None = None
