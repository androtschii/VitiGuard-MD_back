import random

import numpy as np

from vitiguard_ml.seed import set_seed


def draw() -> tuple[float, float]:
    return random.random(), float(np.random.rand())


def test_same_seed_gives_same_numbers() -> None:
    set_seed(7)
    first = draw()
    set_seed(7)

    assert draw() == first


def test_different_seeds_give_different_numbers() -> None:
    set_seed(1)
    first = draw()
    set_seed(2)

    assert draw() != first
