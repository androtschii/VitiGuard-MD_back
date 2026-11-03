import numpy as np
from PIL import Image

from vitiguard_ml.robot.variety import (
    classify_leaf_variety,
    compute_morphometrics,
    extract_leaf_mask,
)


def test_extract_leaf_mask() -> None:
    # 100x100 зелёный лист на белом фоне
    arr = np.full((100, 100, 3), 245, dtype=np.uint8)  # Белый фон
    # В центре зелёный квадрат 60x60
    arr[20:80, 20:80] = [60, 140, 40]
    img = Image.fromarray(arr)

    mask = extract_leaf_mask(img)
    assert np.count_nonzero(mask) == 60 * 60
    assert mask[50, 50] == 1
    assert mask[5, 5] == 0


def test_compute_morphometrics() -> None:
    # Квадратная маска 50x50
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 1

    metrics = compute_morphometrics(mask)
    assert 0.95 <= metrics["aspect_ratio"] <= 1.05
    assert metrics["circularity"] > 0.60
    assert metrics["solidity"] > 0.80


def test_classify_leaf_variety() -> None:
    # Имитация круглого листа с вырезками (Каберне/Фетяска)
    arr = np.full((120, 120, 3), 240, dtype=np.uint8)
    arr[20:100, 20:100] = [50, 130, 45]
    img = Image.fromarray(arr)

    res = classify_leaf_variety(img)
    assert res.predicted_variety in (
        "cabernet_sauvignon",
        "chardonnay",
        "feteasca_alba",
        "feteasca_neagra",
        "feteasca_regala",
        "merlot",
        "rara_neagra",
        "sauvignon_blanc",
    )
    assert res.confidence > 0.0
    assert len(res.top_candidates) > 0
    assert "aspect_ratio" in res.morphometrics
