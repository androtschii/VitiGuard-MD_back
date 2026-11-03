"""Оценка зрелости винограда по цвету ягод против содержания сахара в °Brix (ml/pr-027).

В точном виноделии и роботизированном сборе зрелость грозди перед срезом
определяется спектрально-колориметрическим анализом:
1. Для красных сортов (Cabernet Sauvignon, Merlot, Feteasca Neagra, Syrah):
   - Индекс CIRG (Color Index for Red Grapes, Carreño et al.):
     CIRG = (180 - H*) / (L* + C*) в цветовом пространстве CIE L*a*b*;
   - Веразон (veraison): динамика перехода хлорофилла в антоцианы
     (зелёный -> розовый -> красный -> тёмно-фиолетовый);
2. Для белых сортов (Chardonnay, Sauvignon Blanc, Feteasca Alba, Feteasca Regala):
   - Индекс пожелтения/прозрачности (Yellowness / Translucency Index):
     распад хлорофилла и накопление каротиноидов (b* / L*);
3. Калибровочная модель пересчёта цвета в содержание растворимых сухих веществ (°Brix):
   - Связывает колориметрический профиль ягод со шкалой сахаристости (°Brix)
     для планирования кондиций сусла (сухое вино 20-22.5 °Brix, десертное 23+ °Brix).

Запуск из каталога ml:
  uv run python -m vitiguard_ml.robot.maturity --image data/bunch.jpg --variety cabernet_sauvignon
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

RED_VARIETIES = {
    "cabernet_sauvignon",
    "feteasca_neagra",
    "merlot",
    "rara_neagra",
    "syrah",
    "pinot_noir",
}

WHITE_VARIETIES = {
    "chardonnay",
    "sauvignon_blanc",
    "feteasca_alba",
    "feteasca_regala",
    "aligote",
}


@dataclass(frozen=True)
class MaturityAssessment:
    variety: str
    is_red: bool
    color_index: float  # CIRG для красных или YI для белых сортов
    sugar_brix: float  # Оценка сахара в ягоде (°Brix)
    stage: str  # "unripe" | "veraison" | "harvest_ready" | "overripe"
    harvest_suitability: str  # "too_early" | "ready_table_wine" | "late_harvest"
    mean_lab: tuple[float, float, float]
    metrics: dict[str, float]


def srgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """Перевод линейного/sRGB в CIE XYZ (D65 illuminant)."""
    # Гамма-коррекция
    mask = rgb > 0.04045
    linear = np.empty_like(rgb)
    linear[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    linear[~mask] = rgb[~mask] / 12.92

    # Матрица sRGB -> XYZ (D65)
    r = linear[:, :, 0]
    g = linear[:, :, 1]
    b = linear[:, :, 2]

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    return np.stack([x, y, z], axis=-1)


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    """Перевод CIE XYZ в CIE L*a*b* относительно стандартного белого D65."""
    # Референсный белый D65
    xn, yn, zn = 0.95047, 1.00000, 1.08883

    x = xyz[:, :, 0] / xn
    y = xyz[:, :, 1] / yn
    z = xyz[:, :, 2] / zn

    delta = 6.0 / 29.0

    def f(t: np.ndarray) -> np.ndarray:
        m = t > delta**3
        res = np.empty_like(t)
        res[m] = np.cbrt(t[m])
        res[~m] = t[~m] / (3.0 * delta**2) + (4.0 / 29.0)
        return res

    fx = f(x)
    fy = f(y)
    fz = f(z)

    l_star = 116.0 * fy - 16.0
    a_star = 500.0 * (fx - fy)
    b_star = 200.0 * (fy - fz)

    return np.stack([l_star, a_star, b_star], axis=-1)


def compute_cirg(l_star: float, a_star: float, b_star: float) -> float:
    """Color Index for Red Grapes (CIRG):

    CIRG = (180 - H*) / (L* + C*)
    где C* = sqrt(a*^2 + b*^2), H* = arctan2(b*, a*) в градусах.
    """
    chroma = math.sqrt(a_star**2 + b_star**2)
    # Угол цветового тона (0..360)
    hue_rad = math.atan2(b_star, a_star)
    hue_deg = math.degrees(hue_rad)
    if hue_deg < 0:
        hue_deg += 360.0

    denom = max(1.0, l_star + chroma)
    # Коррекция квадранта для фиолетово-чёрных тонов (300°..360°)
    h_eff = 360.0 - hue_deg if hue_deg > 180.0 else hue_deg

    cirg = (180.0 - h_eff) / denom
    return round(max(0.0, cirg), 3)


def compute_white_yellowness(l_star: float, a_star: float, b_star: float) -> float:
    """Индекс пожелтения и прозрачности для белых сортов:

    Отражает снижение хлорофилла (a* растет от отрицательного к околонулевому)
    и накопление солнечного янтаря (b* растет).
    """
    denom = max(10.0, l_star)
    yi = (b_star * 1.5 - a_star * 0.8) / denom * 10.0
    return round(max(0.0, yi), 3)


def estimate_sugar_brix(color_index: float, is_red: bool) -> float:
    """Калибровочная регрессия между спектральным индексом и сахаристостью ягод (°Brix).

    Откалибрована по полевым диапазонам сбора винограда:
    - Зелёная незрелая ягода: 10 - 14 °Brix;
    - Веразон: 14 - 18 °Brix;
    - Техническая зрелость: 19.5 - 22.5 °Brix;
    - Поздний сбор / перезревание: 23+ °Brix.
    """
    if is_red:
        # Для CIRG:
        # < 2.0 -> 11-13 Brix
        # 3.0 -> ~16 Brix (середина веразона)
        # 5.0 -> ~20.5 Brix
        # 6.0+ -> 22.5-24.5 Brix
        brix = 10.5 + 13.5 / (1.0 + math.exp(-1.1 * (color_index - 3.8)))
    else:
        # Для индекса белых сортов YI:
        # 1.0 -> ~13 Brix
        # 2.5 -> ~18 Brix
        # 4.0+ -> ~21.5-23 Brix
        brix = 11.0 + 12.5 / (1.0 + math.exp(-0.9 * (color_index - 2.6)))

    return round(brix, 1)


def evaluate_maturity(
    bunch_image: Image.Image,
    variety: str = "cabernet_sauvignon",
) -> MaturityAssessment:
    """Оценивает зрелость грозди и сахаристость в °Brix по снимку."""
    v_norm = variety.lower().strip()
    is_red = v_norm not in WHITE_VARIETIES

    # Нормализация изображения и отсечение фона
    rgb = np.asarray(bunch_image.convert("RGB"), dtype=np.float32) / 255.0
    intensity = (rgb[:, :, 0] + rgb[:, :, 1] + rgb[:, :, 2]) / 3.0
    berry_mask = (intensity > 0.08) & (intensity < 0.90)

    if np.count_nonzero(berry_mask) < 50:
        # Запасной случай для пустого фрагмента
        berry_pixels = rgb.reshape(-1, 3)
    else:
        berry_pixels = rgb[berry_mask]

    # Средний цвет в sRGB
    mean_rgb = np.mean(berry_pixels, axis=0)
    dummy_img = mean_rgb.reshape(1, 1, 3)
    xyz = srgb_to_xyz(dummy_img)
    lab = xyz_to_lab(xyz)[0, 0]

    l_val = float(lab[0])
    a_val = float(lab[1])
    b_val = float(lab[2])

    if is_red:
        c_idx = compute_cirg(l_val, a_val, b_val)
    else:
        c_idx = compute_white_yellowness(l_val, a_val, b_val)

    sugar = estimate_sugar_brix(c_idx, is_red=is_red)

    # Определение стадии зрелости
    if sugar < 15.0:
        stage = "unripe"
        suitability = "too_early"
    elif sugar < 19.5:
        stage = "veraison"
        suitability = "too_early"
    elif sugar <= 23.0:
        stage = "harvest_ready"
        suitability = "ready_table_wine"
    else:
        stage = "overripe"
        suitability = "late_harvest"

    return MaturityAssessment(
        variety=variety,
        is_red=is_red,
        color_index=c_idx,
        sugar_brix=sugar,
        stage=stage,
        harvest_suitability=suitability,
        mean_lab=(round(l_val, 2), round(a_val, 2), round(b_val, 2)),
        metrics={
            "L_star": round(l_val, 2),
            "a_star": round(a_val, 2),
            "b_star": round(b_val, 2),
            "chroma": round(math.sqrt(a_val**2 + b_val**2), 2),
        },
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", type=Path, required=True, help="Путь к снимку грозди"
    )
    parser.add_argument(
        "--variety",
        type=str,
        default="cabernet_sauvignon",
        help="Сорт винограда (например, cabernet_sauvignon, chardonnay)",
    )
    args = parser.parse_args(argv)

    if not args.image.exists():
        raise FileNotFoundError(f"Файл снимка не найден: {args.image}")

    with Image.open(args.image) as img:
        img.load()
        assessment = evaluate_maturity(img, variety=args.variety)

    print(
        json.dumps(
            {
                "variety": assessment.variety,
                "is_red": assessment.is_red,
                "color_index": assessment.color_index,
                "sugar_brix": assessment.sugar_brix,
                "stage": assessment.stage,
                "harvest_suitability": assessment.harvest_suitability,
                "mean_lab": assessment.mean_lab,
                "metrics": assessment.metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
