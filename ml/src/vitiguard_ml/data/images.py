from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int
    dhash: str


def difference_hash(image: Image.Image) -> str:
    """Перцептивный хэш dHash (64 бита): одинаков у копий одного снимка, которые
    пересжаты или уменьшены, — такие копии часто встречаются в разных датасетах,
    а SHA-256 у них разный."""
    small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    # Бит = 1, если пиксель ярче соседа справа: 8 строк × 8 сравнений
    bits = 0
    for bit in (pixels[:, :-1] > pixels[:, 1:]).flatten():
        bits = (bits << 1) | int(bit)
    return f"{bits:016x}"


def read_image_info(path: Path) -> ImageInfo | None:
    """Размер и dHash изображения или None, если файл не читается как картинка."""
    try:
        with Image.open(path) as image:
            image.load()
            return ImageInfo(image.width, image.height, difference_hash(image))
    except (UnidentifiedImageError, OSError):
        return None
