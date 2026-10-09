import numpy as np
import numpy.typing as npt
import pytest

pytest.importorskip("albumentations")

from vitiguard_ml.augment import (
    build_eval_transform,
    build_train_transform,
)

Image = npt.NDArray[np.uint8]


@pytest.fixture
def image() -> Image:
    generator = np.random.default_rng(0)
    return generator.integers(0, 256, size=(256, 300, 3), dtype=np.uint8)


def test_train_transform_gives_normalized_square(image: Image) -> None:
    result = build_train_transform(224)(image=image)["image"]

    assert result.shape == (224, 224, 3)
    assert result.dtype == np.float32
    assert -3 < result.mean() < 3


def test_train_transform_is_random(image: Image) -> None:
    transform = build_train_transform(224)

    first = transform(image=image)["image"]

    assert not np.array_equal(first, transform(image=image)["image"])


def test_eval_transform_is_deterministic(image: Image) -> None:
    transform = build_eval_transform(224)

    first = transform(image=image)["image"]

    assert first.shape == (224, 224, 3)
    assert np.array_equal(first, transform(image=image)["image"])
