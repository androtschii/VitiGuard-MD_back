import uuid
from datetime import date
from typing import Literal

from pydantic import AwareDatetime, Field

from app.schemas.base import EntityResponse, RequestSchema, ResponseSchema
from app.schemas.pagination import PageParams

Overlay = Literal["heatmap", "lesion"]


class DiseaseProbability(ResponseSchema):
    """Вероятность класса по ответу модели диагностики."""

    disease: str = Field(examples=["powdery_mildew"])
    probability: float = Field(ge=0, le=1, description="Вероятность класса, 0..1")


class LesionRead(ResponseSchema):
    """Поражённые участки листа по сегментации U-Net."""

    # Прозрачный PNG (data URL) с пропорциями исходного фото: красным — поражение
    mask: str
    # Доля поражённой площади листа, 0..1
    severity: float = Field(ge=0, le=1, description="Доля площади листа под поражением")


class LocationRead(ResponseSchema):
    """Где снято фото: координаты из EXIF, WGS 84."""

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


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


class RobotModelRead(ResponseSchema):
    """Модель детектора лозы (ствол + гроздь) для бортового компьютера робота."""

    name: str = Field(default="vine_yolo")
    classes: list[str] = Field(default=["trunk", "bunch"])
    image_size: int = Field(default=640)
    version: str = Field(examples=["v001", "3f1c9a0b7d2e4f60"])
    size_bytes: int
    format: str = Field(default="onnx")


class VineDetectionItem(ResponseSchema):
    """Рамка ствола или грозди на снимке."""

    label: Literal["trunk", "bunch"] = Field(description="Ствол лозы или гроздь")
    confidence: float = Field(ge=0, le=1)
    box: tuple[float, float, float, float] = Field(
        description="Рамка x1, y1, x2, y2 в долях ширины и высоты снимка",
        examples=[[0.31, 0.12, 0.45, 0.98]],
    )


class VineDetectionRead(ResponseSchema):
    """Стволы и грозди на фото ряда виноградника."""

    model: str = Field(examples=["vine_yolo11n"])
    version: str = Field(examples=["3f1c9a0b7d2e4f60"])
    image_width: int
    image_height: int
    trunks: int
    bunches: int
    detections: list[VineDetectionItem]


class DiagnosisRead(ResponseSchema):
    """Результат классификации фото листа."""

    model: str = Field(examples=["efficientnet_b0"])
    disease: str = Field(examples=["powdery_mildew"])
    confidence: float = Field(ge=0, le=1, description="Вероятность диагноза, 0..1")
    # Модель не уверена: лист стоит переснять крупным планом при дневном свете
    low_confidence: bool
    grape_leaf: bool = Field(
        default=True,
        description="Похоже ли фото на лист винограда. false — диагноз не "
        "показывается и проверка не сохраняется: модель знает только болезни "
        "винограда и на любом снимке выбрала бы одну из них",
    )
    leaf_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Оценка отсева «лист винограда», 0..1; null — отсев не обучен",
    )
    top: list[DiseaseProbability]
    # Тепловая карта: какие области фото повлияли на ответ (прозрачный PNG,
    # data URL); None — модель экспортирована без карты активации
    heatmap: str | None
    # None — модель сегментации не загружена на сервер
    lesion: LesionRead | None
    # Запись в истории проверок (save=true); None — фото не сохранялось
    id: uuid.UUID | None = None
    # Координаты съёмки из EXIF; None — в файле их нет
    location: LocationRead | None = None


class DiagnosisJobStarted(ResponseSchema):
    """Фоновая диагностика поставлена в очередь; результат — по task_id."""

    task_id: str = Field(description="Id фоновой задачи для опроса результата")


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
    """Заключение агронома по проверке листа."""

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
    location: LocationRead | None
    # None — агроном ещё не смотрел
    review: ReviewRead | None
    # Кто снял фото: заполняется в очереди агронома
    owner_name: str | None = None


class DiagnosisQuery(PageParams):
    """Страница истории и фильтры: ?disease=esca&date_from=2026-06-01&page=2."""

    disease: str | None = Field(default=None, max_length=50)
    vineyard_id: uuid.UUID | None = None
    # Границы включительно, по календарным дням UTC
    date_from: date | None = None
    date_to: date | None = None


class ReviewQueueQuery(DiagnosisQuery):
    """Очередь агронома: проверки всех виноградарей, по умолчанию без заключения."""

    reviewed: bool | None = Field(
        default=False, description="false — ждут заключения, true — с заключением"
    )
