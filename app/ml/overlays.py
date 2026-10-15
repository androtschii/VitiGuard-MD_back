"""Наложения для клиента: прозрачные PNG поверх фото листа.

Модели видят снимок сжатым в квадрат, поэтому карта растягивается обратно до
пропорций исходного фото: наложение того же размера совпадает с фото на экране."""

import base64
import io

import numpy as np
from PIL import Image

# Длинная сторона наложения: экран показывает фото не больше этого. Тепловая
# карта размыта по природе (сетка 7×7), поэтому отдаётся вдвое меньше — браузер
# растягивает её плавно, а файл в несколько раз легче
MAX_SIDE = 448
HEATMAP_SIDE = 224
# Палитра тепловой карты (как inferno): холодные области тёмные и прозрачные,
# горячие — жёлтые и плотные
STOPS = np.array([0, 0.25, 0.5, 0.75, 1])
COLORS = np.array(
    [[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]],
    dtype=np.float32,
)
LESION_RGBA = (220, 38, 38, 150)


def overlay_size(width: int, height: int, side: int = MAX_SIDE) -> tuple[int, int]:
    scale = side / max(width, height)
    return max(round(width * scale), 1), max(round(height * scale), 1)


def to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def heatmap(activation: np.ndarray, size: tuple[int, int]) -> str:
    """Карта активации 0..1 → цветной PNG с прозрачностью по силе активации."""
    smooth = Image.fromarray(activation.astype(np.float32), mode="F").resize(
        size, Image.Resampling.BICUBIC
    )
    values = np.clip(np.asarray(smooth), 0, 1)
    rgb = np.stack(
        [np.interp(values, STOPS, COLORS[:, channel]) for channel in range(3)],
        axis=-1,
    )
    alpha = np.clip(values * 1.2, 0, 1) * 200
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1).astype(np.uint8)
    return to_data_url(Image.fromarray(rgba, mode="RGBA"))


def lesion_mask(lesion: np.ndarray, size: tuple[int, int]) -> str:
    """Маска поражения → полупрозрачная красная заливка поражённых участков."""
    mask = Image.fromarray(lesion.astype(np.uint8) * 255, mode="L").resize(
        size, Image.Resampling.NEAREST
    )
    layer = Image.new("RGBA", size, LESION_RGBA)
    transparent = Image.new("RGBA", size, (0, 0, 0, 0))
    return to_data_url(Image.composite(layer, transparent, mask))
