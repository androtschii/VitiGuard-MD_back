"""Вегетационные индексы участка по снимку Sentinel-2 (pr-059 — pr-062).

Из каждого канала читается только окно над участком: файлы каналов — Cloud-
Optimized GeoTIFF, и GDAL скачивает по HTTP лишь нужные блоки. Каналы 20 м
(B05, B8A, SCL) приводятся к сетке 10 м (B04, B08), пиксели вне контура и под
облаками отбрасываются, по остальным считаются NDVI и NDRE.

Функции синхронные: чтение растров блокирует поток, поэтому они выполняются в
задаче Celery, а не в обработчике запроса."""

import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.features import bounds as geometry_bounds
from rasterio.features import geometry_mask
from rasterio.warp import transform as transform_points
from rasterio.warp import transform_geom
from rasterio.windows import Window, from_bounds
from rasterio.windows import bounds as window_bounds

from app.integrations.earth_search import Band, EarthScene

# Классы маски SCL, по которым индекс достоверен: растительность и открытая
# почва (междурядья). Облака, их тени, снег, вода и брак отбрасываются
VALID_SCL = (4, 5)
# Если над участком меньше половины пикселей без облаков, снимок по нему
# не считается: среднее по оставшимся краям участка вводило бы в заблуждение
MIN_CLEAR_SHARE = 0.5
# Настройки GDAL для чтения COG по HTTP: не запрашивать список файлов каталога
# и не повторять ошибки — только то, что нужно для окна
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "3",
}
# Палитра картограммы NDVI (RdYlGn): голая почва красная, густая листва зелёная
NDVI_STOPS = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
NDVI_COLORS = np.array(
    [
        [165, 0, 38],
        [244, 109, 67],
        [254, 224, 139],
        [166, 217, 106],
        [26, 152, 80],
        [0, 104, 55],
    ],
    dtype=np.float32,
)

Corners = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class ParcelBands:
    """Каналы в окне участка на общей сетке 10 м: отражение 0..1 и маска SCL."""

    red: np.ndarray
    rededge: np.ndarray
    nir: np.ndarray
    nir_narrow: np.ndarray
    scl: np.ndarray
    # True — пиксель внутри контура участка
    inside: np.ndarray
    # Углы окна в WGS 84, [долгота, широта]: верхний левый, верхний правый,
    # нижний правый, нижний левый — порядок слоя-картинки на карте
    corners: Corners = ()


@dataclass(frozen=True)
class NdviMap:
    png: bytes
    corners: Corners


@dataclass(frozen=True)
class IndexStats:
    mean: float
    min: float
    max: float
    std: float


@dataclass(frozen=True)
class ParcelIndices:
    # Пиксели 10 м внутри контура и из них — без облаков и теней
    parcel_pixels: int
    clear_pixels: int
    # None — над участком слишком облачно, индекс не считается
    ndvi: IndexStats | None
    ndre: IndexStats | None

    @property
    def cloud_share(self) -> float:
        """Доля участка под облаками и тенями по маске SCL, 0..1."""
        if self.parcel_pixels == 0:
            return 1.0
        return 1 - self.clear_pixels / self.parcel_pixels


def _reflectance(values: np.ndarray, band: Band) -> np.ndarray:
    reflectance: np.ndarray = values.astype(np.float32) * band.scale + band.offset
    return reflectance


def read_parcel(scene: EarthScene, geometry: dict[str, Any]) -> ParcelBands:
    """Окно участка из каналов снимка. geometry — GeoJSON в WGS 84."""
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(scene.bands["B04"].href) as red:
            local = transform_geom("EPSG:4326", red.crs, geometry)
            window: Window = (
                from_bounds(*geometry_bounds(local), red.transform)
                .round_offsets()
                .round_lengths()
            )
            # Участок меньше пикселя всё равно занимает хотя бы один
            window = Window(
                window.col_off,
                window.row_off,
                max(int(window.width), 1),
                max(int(window.height), 1),
            )
            transform = red.window_transform(window)
            shape = (int(window.height), int(window.width))
            box = window_bounds(window, red.transform)
            red_values = red.read(1, window=window, boundless=True, fill_value=0)

        def read(band: str, resampling: Resampling) -> np.ndarray:
            # Настройки GDAL действуют в своём потоке — задаются и здесь
            with (
                rasterio.Env(**GDAL_ENV),
                rasterio.open(scene.bands[band].href) as dataset,
            ):
                values: np.ndarray = dataset.read(
                    1,
                    window=from_bounds(*box, dataset.transform),
                    out_shape=shape,
                    resampling=resampling,
                    boundless=True,
                    fill_value=0,
                )
                return values

        # Каналы — отдельные файлы: читаются параллельно, по запросу на файл.
        # Классы маски нельзя усреднять — для неё ближайший пиксель
        with ThreadPoolExecutor(max_workers=4) as pool:
            scl, nir, rededge, nir_narrow = pool.map(
                read,
                ("SCL", "B08", "B05", "B8A"),
                (
                    Resampling.nearest,
                    Resampling.bilinear,
                    Resampling.bilinear,
                    Resampling.bilinear,
                ),
            )

    inside = ~geometry_mask([local], out_shape=shape, transform=transform)
    if not inside.any():
        # Контур уже пикселя: берётся пиксель, в который попадает участок
        inside = ~geometry_mask(
            [local], out_shape=shape, transform=transform, all_touched=True
        )
    left, bottom, right, top = box
    lons, lats = transform_points(
        red.crs, "EPSG:4326", [left, right, right, left], [top, top, bottom, bottom]
    )
    return ParcelBands(
        red=_reflectance(red_values, scene.bands["B04"]),
        rededge=_reflectance(rededge, scene.bands["B05"]),
        nir=_reflectance(nir, scene.bands["B08"]),
        nir_narrow=_reflectance(nir_narrow, scene.bands["B8A"]),
        scl=scl,
        inside=inside,
        corners=tuple(
            (round(lon, 7), round(lat, 7)) for lon, lat in zip(lons, lats, strict=True)
        ),
    )


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a − b) / (a + b), ограниченное [−1, 1]; при нулевой сумме — NaN."""
    total = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(total > 0, (a - b) / total, np.nan)
    clipped: np.ndarray = np.clip(result, -1, 1)
    return clipped


def _stats(values: np.ndarray) -> IndexStats | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return IndexStats(
        mean=round(float(finite.mean()), 4),
        min=round(float(finite.min()), 4),
        max=round(float(finite.max()), 4),
        std=round(float(finite.std()), 4),
    )


def compute_indices(bands: ParcelBands) -> ParcelIndices:
    """NDVI = (B08 − B04) / (B08 + B04) — количество зелёной биомассы;
    NDRE = (B8A − B05) / (B8A + B05) — содержание хлорофилла, раньше отражает
    стресс. Считаются по пикселям участка без облаков и теней."""
    clear = bands.inside & np.isin(bands.scl, VALID_SCL)
    parcel_pixels = int(bands.inside.sum())
    clear_pixels = int(clear.sum())
    if parcel_pixels == 0 or clear_pixels / parcel_pixels < MIN_CLEAR_SHARE:
        return ParcelIndices(parcel_pixels, clear_pixels, None, None)
    ndvi = normalized_difference(bands.nir, bands.red)[clear]
    ndre = normalized_difference(bands.nir_narrow, bands.rededge)[clear]
    return ParcelIndices(parcel_pixels, clear_pixels, _stats(ndvi), _stats(ndre))


def ndvi_map(bands: ParcelBands) -> NdviMap:
    """Картограмма NDVI: пиксель 10 м — пиксель PNG. Вне контура, под облаками
    и без сигнала — прозрачно, отрицательный NDVI окрашен как ноль."""
    ndvi = normalized_difference(bands.nir, bands.red)
    visible = bands.inside & np.isin(bands.scl, VALID_SCL) & np.isfinite(ndvi)
    values = np.clip(np.nan_to_num(ndvi), 0, 1)
    rgb = np.stack(
        [
            np.interp(values, NDVI_STOPS, NDVI_COLORS[:, channel])
            for channel in range(3)
        ],
        axis=-1,
    )
    alpha = np.where(visible, 255, 0)
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, "PNG", optimize=True)
    return NdviMap(buffer.getvalue(), bands.corners)
