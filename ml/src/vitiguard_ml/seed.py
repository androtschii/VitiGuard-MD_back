import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Фиксирует генераторы случайных чисел, чтобы эксперимент повторялся.

    PyTorch настраивается, только если установлен: подготовка данных работает и
    без него."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Детерминированные алгоритмы cuDNN медленнее, но дают одинаковый результат
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
