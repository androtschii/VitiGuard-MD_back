"""Снимки Sentinel-2 L2A в открытом хранилище AWS (каталог Earth Search).

Те же данные Copernicus, что в Copernicus Data Space, но каналы лежат в
открытом хранилище в формате Cloud-Optimized GeoTIFF: файл читается по HTTP без
учётной записи, и можно скачать только окно над участком, а не весь снимок
110 × 110 км."""

from datetime import UTC, datetime

from pydantic import BaseModel

from app.integrations.copernicus import StacClient, _Feature

# Канал Sentinel-2 → имя ресурса в Earth Search
BANDS = {
    "B04": "red",
    "B05": "rededge1",
    "B08": "nir",
    "B8A": "nir08",
    "SCL": "scl",
}


class Band(BaseModel):
    href: str
    # Отражение = значение × scale + offset
    scale: float = 1.0
    offset: float = 0.0


class EarthScene(BaseModel):
    product_id: str
    acquired_at: datetime
    cloud_cover: float
    platform: str
    tile: str
    bands: dict[str, Band]


def _earth_scene(feature: _Feature) -> EarthScene:
    properties = feature.properties
    acquired_at = datetime.fromisoformat(properties["datetime"])
    # С версии обработки 04.00 ESA прибавляет к значениям 1000, и метаданные
    # канала велят вычесть 0,1 отражения. Earth Search уже вычел это смещение
    # при переводе в COG и отмечает это флагом — повторное вычитание дало бы
    # отрицательное отражение красного канала и NDVI больше единицы
    offset_applied = bool(properties.get("earthsearch:boa_offset_applied"))
    bands = {}
    for band, name in BANDS.items():
        asset = feature.assets[name]
        raster = asset.raster_bands[0] if asset.raster_bands else {}
        bands[band] = Band(
            href=asset.href,
            scale=raster.get("scale", 1.0),
            offset=0.0 if offset_applied else raster.get("offset", 0.0),
        )
    return EarthScene(
        product_id=feature.id,
        acquired_at=(
            acquired_at if acquired_at.tzinfo else acquired_at.replace(tzinfo=UTC)
        ),
        cloud_cover=properties.get("eo:cloud_cover", 100),
        platform=properties["platform"],
        tile=str(properties.get("grid:code", "")).removeprefix("MGRS-"),
        bands=bands,
    )


def cloud_query(max_cloud_cover: float) -> dict[str, object]:
    """Earth Search фильтрует облачность расширением query, а не CQL2 —
    фильтр CQL2 он молча игнорирует."""
    return {"query": {"eo:cloud_cover": {"lte": max_cloud_cover}}}


class EarthSearchClient(StacClient[EarthScene]):
    def _search_url(self) -> str:
        return f"{self._settings.earth_search_url}/search"

    def _parse(self, feature: _Feature) -> EarthScene:
        return _earth_scene(feature)


def _version(scene: EarthScene) -> int:
    # S2A_35TPN_20260829_1_L2A: номер перед _L2A растёт при переобработке
    parts = scene.product_id.split("_")
    return int(parts[-2]) if len(parts) >= 2 and parts[-2].isdigit() else 0


def deduplicate(scenes: list[EarthScene]) -> list[EarthScene]:
    """Одна съёмка — один снимок. Участок на стыке тайлов MGRS виден в двух
    тайлах одного пролёта, а снимки бывают переобработаны новой версией; спутник
    проходит над точкой не чаще раза в сутки, поэтому повтором считается тот же
    спутник в тот же день. Остаётся новейшая обработка с меньшей облачностью."""
    best: dict[tuple[str, object], EarthScene] = {}
    for scene in scenes:
        key = (scene.platform, scene.acquired_at.date())
        current = best.get(key)
        rank = (_version(scene), -scene.cloud_cover, scene.tile)
        if current is None or rank > (
            _version(current),
            -current.cloud_cover,
            current.tile,
        ):
            best[key] = scene
    return sorted(best.values(), key=lambda scene: scene.acquired_at)
