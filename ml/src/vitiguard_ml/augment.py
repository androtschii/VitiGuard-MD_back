"""Преобразования изображений для обучения и оценки (Albumentations).

Albumentations входит в группу зависимостей train и импортируется только здесь."""

from typing import Any

import albumentations as A

# Средние и разброс ImageNet: модели предобучены с такой нормализацией
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: int) -> Any:
    """Аугментации обучения. Имитируют то, чем полевой снимок отличается от
    лабораторного: другой ракурс и масштаб, освещение, размытие в руке,
    частично закрытый лист."""
    return A.Compose(
        [
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
