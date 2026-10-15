"""Диагностика по фото листа: проверка снимка, классификация, тепловая карта и
маска поражения."""

import io

from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.errors import BadRequestError, UnsupportedMediaError
from app.ml.classifier import LeafClassifier
from app.ml.overlays import HEATMAP_SIDE, heatmap, lesion_mask, overlay_size
from app.ml.segmentation import LesionSegmenter
from app.schemas.diagnosis import DiagnosisRead, DiseaseProbability, LesionRead

# Ниже этой вероятности ответ помечается как неуверенный. На полевых снимках
# модели ошибаются заметно чаще, чем на лабораторных (ml/pr-010), и агроному
# важно видеть, когда ответу нельзя доверять без проверки
LOW_CONFIDENCE = 0.6

FORMATS = {"JPEG", "PNG", "WEBP"}
# Меньше этого лист не разглядеть: модель всё равно видит снимок как 224×224
MIN_SIDE = 64
# Защита от «бомбы распаковки»: маленький файл с огромным размером в пикселях
MAX_PIXELS = 40_000_000
WRONG_FORMAT = "Нужна фотография в формате JPEG, PNG или WebP"


def read_leaf_image(data: bytes) -> Image.Image:
    """Формат определяется по содержимому файла, а не по расширению или
    заголовку Content-Type: их задаёт клиент. Размер проверяется по заголовку
    файла до распаковки пикселей."""
    try:
        image = Image.open(io.BytesIO(data))
    except UnidentifiedImageError as error:
        raise UnsupportedMediaError(WRONG_FORMAT) from error
    if image.format not in FORMATS:
        raise UnsupportedMediaError(WRONG_FORMAT)
    width, height = image.size
    if width * height > MAX_PIXELS:
        raise BadRequestError("Слишком большое разрешение снимка: больше 40 Мпикс")
    if min(width, height) < MIN_SIDE:
        raise BadRequestError(
            f"Снимок слишком мал: нужно не меньше {MIN_SIDE} пикселей по каждой стороне"
        )
    try:
        image.load()
    except OSError as error:
        raise BadRequestError("Файл изображения повреждён") from error
    # Фото с телефона хранят поворот в EXIF: без его учёта лист оказался бы на боку
    return ImageOps.exif_transpose(image)


def analyze_leaf(
    leaf: Image.Image,
    classifier: LeafClassifier,
    segmenter: LesionSegmenter | None,
    top_k: int,
) -> DiagnosisRead:
    result = classifier.classify(leaf)
    best = result.predictions[0]
    size = overlay_size(*leaf.size)
    lesion = None
    if segmenter is not None:
        mask = segmenter.segment(leaf)
        lesion = LesionRead(
            mask=lesion_mask(mask.lesion, size), severity=round(mask.severity, 4)
        )
    return DiagnosisRead(
        model=classifier.name,
        disease=best.disease,
        confidence=round(best.probability, 4),
        low_confidence=best.probability < LOW_CONFIDENCE,
        top=[
            DiseaseProbability(disease=p.disease, probability=round(p.probability, 4))
            for p in result.predictions[:top_k]
        ],
        heatmap=(
            None
            if result.activation is None
            else heatmap(result.activation, overlay_size(*leaf.size, HEATMAP_SIDE))
        ),
        lesion=lesion,
    )
