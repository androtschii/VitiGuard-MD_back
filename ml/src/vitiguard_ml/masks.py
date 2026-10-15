"""Автоматические маски поражения листа средствами OpenCV.

Готовой попиксельной разметки поражений винограда нет, поэтому маски для
обучения U-Net строятся правилами: сначала отделяется лист от фона, затем на
листе — участки, цвет которых не похож на здоровую зелень. Это «черновая»
разметка: её качество проверяется выборочно глазами, а U-Net учится обобщать её.

OpenCV входит в группу зависимостей train и импортируется только здесь."""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

# Оттенок в OpenCV — 0..179. Здоровая зелень листа — примерно 35..85
HEALTHY_HUE = (35, 85)
# Фон PlantVillage — серый, NGLD — белый: у фона мало насыщенности
MIN_LEAF_SATURATION = 40
# Мелкие пятна — шум сжатия, а не симптомы
MIN_SPOT_PIXELS = 12


@dataclass(frozen=True)
class LesionMasks:
    leaf: NDArray[np.bool_]
    lesion: NDArray[np.bool_]

    @property
    def severity(self) -> float:
        """Доля поражённой площади листа, 0..1."""
        leaf_pixels = int(self.leaf.sum())
        return float(self.lesion.sum() / leaf_pixels) if leaf_pixels else 0.0


def _clean(mask: NDArray[np.uint8], kernel_size: int) -> NDArray[np.uint8]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return np.asarray(closed, dtype=np.uint8)


def leaf_mask(rgb: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Лист — самая крупная насыщенная область снимка с заполненными дырами."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    raw = (hsv[..., 1] >= MIN_LEAF_SATURATION).astype(np.uint8) * 255
    raw = _clean(raw, 5)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw)
    if count <= 1:
        return np.zeros(rgb.shape[:2], dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    leaf = (labels == largest).astype(np.uint8) * 255
    # Пятна поражения внутри листа бывают ненасыщенными (белый налёт оидиума):
    # заливка внешнего контура возвращает их в лист
    contours, _ = cv2.findContours(leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(leaf)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    filled_mask: NDArray[np.bool_] = filled > 0
    return filled_mask


def lesion_masks(rgb: NDArray[np.uint8]) -> LesionMasks:
    """Поражение — пиксели листа с оттенком вне здоровой зелени (бурые, жёлтые
    пятна) или светлые малонасыщенные (налёт)."""
    leaf = leaf_mask(rgb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    off_green = (hue < HEALTHY_HUE[0]) | (hue > HEALTHY_HUE[1])
    coating = (saturation < MIN_LEAF_SATURATION) & (value > 170)
    raw = ((off_green | coating) & leaf).astype(np.uint8) * 255
    raw = _clean(raw, 3)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= MIN_SPOT_PIXELS
    return LesionMasks(leaf=leaf, lesion=keep[labels] & leaf)
