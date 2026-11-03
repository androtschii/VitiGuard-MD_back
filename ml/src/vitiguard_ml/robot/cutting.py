"""Определение точки среза плодоножки грозди для робота-сборщика (ml/pr-024).

По маске грозди и карте глубины вычисляет 3D-координаты точки среза (ножки над
гроздью) в системе координат камеры, вектор подхода секатора и безопасный
отступ от верхних ягод. Перевод в координаты руки требует калибровки
камера-рука на роботе. Пороги и направление ножки — исходные допущения: их
нужно проверить на кадрах камеры робота (ml/pr-028).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Внутренние параметры камеры (пинхол-модель)."""

    fx: float
    fy: float
    cx: float
    cy: float

    def deproject(self, u: float, v: float, depth: float) -> tuple[float, float, float]:
        """Пиксель (u, v) и глубина Z в метрах -> (X, Y, Z) в системе координат камеры."""
        if depth <= 0:
            return 0.0, 0.0, 0.0
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        return float(x), float(y), float(depth)


@dataclass(frozen=True)
class CutTarget:
    """Целевая точка и параметры среза для захвата робота."""

    pixel: tuple[int, int]
    point_3d: tuple[float, float, float]
    approach_vector: tuple[float, float, float]
    stem_width_px: float
    # Направление ножки найдено по снимку, а не взято вертикальным по умолчанию
    stem_from_image: bool
    safe_to_cut: bool


def find_peduncle_base(mask: np.ndarray) -> tuple[int, int] | None:
    """Находит точку прикрепления плодоножки к верхней части грозди.

    Вход: бинарная маска грозди (uint8 или bool, 2D).
    Выход: координаты (x, y) вершины грозди.
    """
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0:
        return None

    top_y = int(np.min(y_indices))
    # Берём верхний срез грозди (первые 5% высоты или не менее 3 строк)
    max_y = int(np.max(y_indices))
    height = max_y - top_y + 1
    band_height = max(3, int(height * 0.05))
    band_mask = (y_indices >= top_y) & (y_indices < top_y + band_height)

    if not np.any(band_mask):
        return int(x_indices[0]), top_y

    center_x = int(np.median(x_indices[band_mask]))
    return center_x, top_y


def estimate_stem_direction(
    rgb: np.ndarray,
    base_x: int,
    base_y: int,
    search_height: int = 50,
    search_half_width: int = 25,
) -> tuple[float, float, float]:
    """Оценивает направление ножки вверх от точки прикрепления.

    Возвращает (dx, dy, stem_width_px), где dx, dy нормализованы.
    """
    _h, w = rgb.shape[:2]
    y_min = max(0, base_y - search_height)
    y_max = base_y
    x_min = max(0, base_x - search_half_width)
    x_max = min(w, base_x + search_half_width + 1)

    roi = rgb[y_min:y_max, x_min:x_max]
    if roi.size == 0 or roi.shape[0] < 5:
        return 0.0, -1.0, 6.0

    # В оттенках серого ножка и побег обычно темнее неба или имеют чёткие вертикальные границы
    gray = (0.299 * roi[..., 0] + 0.587 * roi[..., 1] + 0.114 * roi[..., 2]).astype(
        np.float32
    )

    # Поиск тёмного вертикального тяжа в строках над основанием
    col_profiles = np.mean(gray, axis=0)
    best_local_x = int(np.argmin(col_profiles))
    stem_x = x_min + best_local_x
    dx = float(stem_x - base_x)
    dy = float(y_min - base_y)

    length = float(np.hypot(dx, dy))
    if length > 0:
        dx /= length
        dy /= length
    else:
        dx, dy = 0.0, -1.0

    # Ширина ножки — число столбцов темнее среднего по области
    threshold = float(np.mean(col_profiles))
    dark_pixels = int(np.sum(col_profiles < threshold))
    stem_width = max(3.0, float(dark_pixels))

    return dx, dy, stem_width


def compute_cut_target(
    mask: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    rgb: np.ndarray | None = None,
    offset_mm: float = 20.0,
    min_distance_m: float = 0.20,
    max_distance_m: float = 1.20,
) -> CutTarget | None:
    """Вычисляет целевую точку среза над гроздью.

    offset_mm: безопасный отступ вверх по ножке (обычно 15-25 мм над началом ягод).
    depth_map: карта глубины в метрах (float32).
    """
    base = find_peduncle_base(mask)
    if base is None:
        return None

    base_x, base_y = base

    # Считываем среднюю глубину в области вершины грозди
    h, w = depth_map.shape[:2]
    y_rad, x_rad = 5, 5
    y1, y2 = max(0, base_y - y_rad), min(h, base_y + y_rad + 1)
    x1, x2 = max(0, base_x - x_rad), min(w, base_x + x_rad + 1)
    depth_roi = depth_map[y1:y2, x1:x2]
    valid_depths = depth_roi[depth_roi > 0]

    base_depth = float(np.median(valid_depths)) if len(valid_depths) > 0 else 0.0

    if rgb is not None:
        dx, dy, stem_w = estimate_stem_direction(rgb, base_x, base_y)
    else:
        dx, dy, stem_w = 0.0, -1.0, 6.0

    # Пиксельное смещение, соответствующее offset_mm на данной глубине
    if base_depth > 0:
        offset_m = offset_mm / 1000.0
        offset_px = (offset_m * intrinsics.fy) / base_depth
    else:
        offset_px = 30.0

    cut_x = round(base_x + dx * offset_px)
    cut_y = round(base_y + dy * offset_px)

    cut_x = max(0, min(w - 1, cut_x))
    cut_y = max(0, min(h - 1, cut_y))

    # Глубина в точке среза
    cut_depth = float(depth_map[cut_y, cut_x])
    if cut_depth <= 0:
        cut_depth = base_depth

    point_3d = intrinsics.deproject(cut_x, cut_y, cut_depth)

    # Вектор подхода секатора (перпендикулярен направлению ножки в 3D)
    norm = float(np.hypot(dx, dy))
    if norm > 0:
        approach_x = -dy / norm
        approach_y = dx / norm
    else:
        approach_x, approach_y = 1.0, 0.0
    approach_vector = (float(approach_x), float(approach_y), 0.0)

    # Проверка на досягаемость и безопасность
    safe = (
        cut_depth >= min_distance_m
        and cut_depth <= max_distance_m
        and cut_y > 5
        and cut_y < h - 5
        and cut_x > 5
        and cut_x < w - 5
    )

    return CutTarget(
        pixel=(cut_x, cut_y),
        point_3d=point_3d,
        approach_vector=approach_vector,
        stem_width_px=stem_w,
        stem_from_image=rgb is not None,
        safe_to_cut=safe,
    )
