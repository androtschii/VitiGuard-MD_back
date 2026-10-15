"""Проверка фото листа перед классификацией."""

import io

from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.errors import BadRequestError, UnsupportedMediaError

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
