import numpy as np
from PIL import Image

from vitiguard_ml.robot.maturity import (
    compute_cirg,
    estimate_sugar_brix,
    evaluate_maturity,
    srgb_to_xyz,
    xyz_to_lab,
)


def test_color_space_conversions() -> None:
    # Проверка нейтрального серого
    rgb = np.full((1, 1, 3), 0.5, dtype=np.float32)
    xyz = srgb_to_xyz(rgb)
    lab = xyz_to_lab(xyz)

    l_val = lab[0, 0, 0]
    a_val = lab[0, 0, 1]
    b_val = lab[0, 0, 2]

    # Для sRGB 0.5: L* около 53.4, a* и b* близки к нулю
    assert 50.0 < l_val < 60.0
    assert abs(a_val) < 1.0
    assert abs(b_val) < 1.0


def test_cirg_calculation() -> None:
    # Тёмно-фиолетовая спелая ягода: низкая яркость L*=25, лёгкий фиолетовый оттенок
    cirg_mature = compute_cirg(l_star=25.0, a_star=8.0, b_star=-5.0)
    # Зелёная незрелая ягода: L*=60, a*=-15 (зелёный), b*=25 (жёлтый)
    cirg_unripe = compute_cirg(l_star=60.0, a_star=-15.0, b_star=25.0)

    assert cirg_mature > cirg_unripe
    assert cirg_mature >= 4.0


def test_sugar_brix_estimation() -> None:
    # Низкий индекс -> незрелая ягода (11-14 Brix)
    brix_low = estimate_sugar_brix(color_index=1.5, is_red=True)
    # Высокий индекс -> спелая ягода (21-24 Brix)
    brix_high = estimate_sugar_brix(color_index=6.0, is_red=True)

    assert brix_low < 15.0
    assert brix_high > 20.0
    assert brix_low < brix_high


def test_evaluate_maturity_red_variety() -> None:
    # Тёмно-рубиновое/фиолетовое изображение спелого Каберне
    arr = np.zeros((80, 80, 3), dtype=np.uint8)
    arr[:, :] = [30, 20, 45]
    img = Image.fromarray(arr)

    res = evaluate_maturity(img, variety="cabernet_sauvignon")

    assert res.is_red is True
    assert res.variety == "cabernet_sauvignon"
    assert res.sugar_brix >= 16.0
    assert res.stage in ("harvest_ready", "veraison", "overripe")


def test_evaluate_maturity_white_variety() -> None:
    # Золотисто-янтарное изображение спелого Шардоне
    arr = np.zeros((80, 80, 3), dtype=np.uint8)
    arr[:, :] = [180, 170, 70]
    img = Image.fromarray(arr)

    res = evaluate_maturity(img, variety="chardonnay")

    assert res.is_red is False
    assert res.variety == "chardonnay"
    assert res.sugar_brix > 18.0
