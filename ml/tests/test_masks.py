import numpy as np
import pytest

pytest.importorskip("cv2")

from vitiguard_ml.masks import lesion_masks

GREEN = (60, 140, 50)
BROWN = (120, 70, 30)
GREY = (128, 128, 128)


def leaf_image(spot: tuple[int, int, int] | None = None) -> np.ndarray:
    """Серый фон, как в PlantVillage, круглый зелёный «лист», при желании —
    квадратное пятно внутри."""
    image = np.full((128, 128, 3), GREY, dtype=np.uint8)
    yy, xx = np.mgrid[:128, :128]
    image[(yy - 64) ** 2 + (xx - 64) ** 2 < 50**2] = GREEN
    if spot is not None:
        image[50:70, 50:70] = spot
    return image


def test_healthy_leaf_has_no_lesion() -> None:
    masks = lesion_masks(leaf_image())

    assert masks.leaf.sum() == pytest.approx(np.pi * 50**2, rel=0.05)
    assert masks.severity == pytest.approx(0, abs=0.01)


def test_brown_spot_is_lesion() -> None:
    masks = lesion_masks(leaf_image(BROWN))

    assert masks.lesion[60, 60]
    assert not masks.lesion[64, 30]
    assert masks.severity == pytest.approx(400 / (np.pi * 50**2), rel=0.2)


def test_white_coating_stays_inside_leaf() -> None:
    # Налёт оидиума ненасыщенный, но он — часть листа и поражение
    masks = lesion_masks(leaf_image((235, 235, 235)))

    assert masks.leaf[60, 60]
    assert masks.lesion[60, 60]


def test_blank_image_has_no_leaf() -> None:
    masks = lesion_masks(np.full((64, 64, 3), GREY, dtype=np.uint8))

    assert masks.leaf.sum() == 0
    assert masks.severity == 0.0
