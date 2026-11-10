"""Преобразования изображений для обучения и оценки (Albumentations).

Albumentations входит в группу зависимостей train и импортируется только здесь."""

from typing import Any

import albumentations as A
import numpy as np
from PIL import Image

# Средние и разброс ImageNet: модели предобучены с такой нормализацией
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Светлый ровный фон лабораторного снимка: почти серый и яркий пиксель
BACKGROUND_SPREAD = 25
BACKGROUND_BRIGHTNESS = 170
# Тона листвы, земли и тени для фона «как в поле»
FOLIAGE = np.array(
    [[46, 82, 30], [78, 120, 44], [30, 50, 22], [96, 84, 52], [20, 28, 18]],
    dtype=np.float32,
)


def foliage_background(
    image: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Белый фон лабораторного снимка заменяется пятнами листвы и земли.

    В поле лист снимают на фоне других листьев, а не на белой бумаге: без
    этого модель может связать болезнь с фоном снимка, а не с листом."""
    rng = rng or np.random.default_rng()
    pixels = image.astype(np.float32)
    spread = pixels.max(axis=2) - pixels.min(axis=2)
    background = (spread < BACKGROUND_SPREAD) & (
        pixels.mean(axis=2) > BACKGROUND_BRIGHTNESS
    )
    if not background.any():
        return image
    height, width = background.shape
    cells = rng.integers(0, len(FOLIAGE), size=(6, 6))
    coarse = FOLIAGE[cells].astype(np.uint8)
    texture = np.asarray(
        Image.fromarray(coarse).resize((width, height), Image.Resampling.BICUBIC),
        dtype=np.float32,
    )
    texture += rng.normal(0, 8, size=texture.shape)
    result = np.where(background[..., None], texture, pixels)
    return np.clip(result, 0, 255).astype(np.uint8)


def _foliage(image: np.ndarray, **_: Any) -> np.ndarray:
    return foliage_background(image)


def field_steps() -> list[Any]:
    """Отличия полевого снимка: фон из листвы, тень, малое разрешение (фото
    из мессенджера около 300 точек)."""
    return [
        A.Lambda(image=_foliage, name="foliage_background", p=0.5),
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1), shadow_intensity_range=(0.3, 0.6), p=0.3
        ),
        A.Downscale(scale_range=(0.25, 0.6), p=0.3),
    ]


def build_train_transform(image_size: int, field: bool = False) -> Any:
    """Аугментации обучения. Имитируют то, чем полевой снимок отличается от
    лабораторного: другой ракурс и масштаб, освещение, размытие в руке,
    частично закрытый лист. С field=True — ещё фон, тень и малое разрешение."""
    return A.Compose(
        [
            *(field_steps() if field else []),
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.6, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=15, p=0.5
            ),
            A.OneOf(
                [A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=(3, 7))],
                p=0.3,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.05, 0.15),
                hole_width_range=(0.05, 0.15),
                p=0.3,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int) -> Any:
    """Проверка и тест — без случайности: результат одинаков при каждом прогоне."""
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
