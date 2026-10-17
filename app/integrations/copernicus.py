"""Клиент каталога снимков Copernicus Data Space Ecosystem (STAC API).

Поиск снимков Sentinel-2 уровня L2A (отражение у поверхности, с маской
классификации сцены SCL) работает без учётной записи. Сами файлы каналов
лежат в хранилище eodata и скачиваются с учётной записью (pr-059)."""

from datetime import UTC, datetime
from typing import Any, ClassVar

import httpx2 as httpx
from pydantic import BaseModel, Field, ValidationError

from app.core.config import Settings
from app.core.errors import ExternalServiceError

COLLECTION = "sentinel-2-l2a"
# Каналы, нужные системе: красный и ближний ИК — для NDVI, red edge и узкий ближний
# ИК — для NDRE, SCL — маска облаков и теней
BAND_ASSETS = {
    "B04": "B04_10m",
    "B05": "B05_20m",
    "B08": "B08_10m",
    "B8A": "B8A_20m",
    "SCL": "SCL_20m",
}
PAGE_SIZE = 100
# Предел на один поиск: сезон над одним участком — это десятки снимков, а
# тысячи означали бы ошибку в запросе (например, геометрию на весь континент)
MAX_SCENES = 1000
UNAVAILABLE = "Каталог спутниковых снимков недоступен, попробуйте позже"


class Scene(BaseModel):
    """Снимок Sentinel-2 из каталога."""

    product_id: str
    acquired_at: datetime
    cloud_cover: float
    platform: str
    tile: str
    # Канал → адрес файла в хранилище eodata
    assets: dict[str, str]


class _Asset(BaseModel):
    href: str
    # Тип данных, масштаб и смещение значений канала (расширение STAC raster)
    raster_bands: list[dict[str, Any]] = Field(default=[], alias="raster:bands")


class _Properties(BaseModel):
    datetime: datetime
    platform: str
    cloud_cover: float | None = None
    grid_code: str | None = None


class _Feature(BaseModel):
    id: str
    properties: dict[str, Any]
    assets: dict[str, _Asset]


class _Link(BaseModel):
    rel: str
    href: str
    method: str = "GET"
    body: dict[str, Any] | None = None


class _Page(BaseModel):
    features: list[_Feature]
    links: list[_Link] = []


def _scene(feature: _Feature) -> Scene:
    properties = _Properties.model_validate(
        {
            **feature.properties,
            "cloud_cover": feature.properties.get("eo:cloud_cover"),
            "grid_code": feature.properties.get("grid:code"),
        }
    )
    acquired_at = properties.datetime
    return Scene(
        product_id=feature.id,
        acquired_at=(
            acquired_at if acquired_at.tzinfo else acquired_at.replace(tzinfo=UTC)
        ),
        cloud_cover=(
            properties.cloud_cover if properties.cloud_cover is not None else 100
        ),
        platform=properties.platform,
        tile=(properties.grid_code or "").removeprefix("MGRS-"),
        assets={
            band: feature.assets[name].href
            for band, name in BAND_ASSETS.items()
            if name in feature.assets
        },
    )


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class StacClient[S]:
    """Постраничный поиск в каталоге STAC. Каталоги различаются адресом,
    коллекцией и тем, как из записи каталога получается снимок."""

    collection: ClassVar[str] = COLLECTION

    def __init__(self, http: httpx.AsyncClient, settings: Settings) -> None:
        self._http = http
        self._settings = settings

    def _search_url(self) -> str:
        raise NotImplementedError

    def _parse(self, feature: _Feature) -> S:
        raise NotImplementedError

    async def search(
        self,
        geometry: dict[str, Any],
        start: datetime,
        end: datetime,
        extra_filter: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> list[S]:
        """Снимки, пересекающие geometry (GeoJSON, WGS 84), за период, по
        возрастанию даты. Страницы каталога загружаются по ссылкам next."""
        body: dict[str, Any] = {
            "collections": [self.collection],
            "intersects": geometry,
            "datetime": f"{_iso(start)}/{_iso(end)}",
            "limit": PAGE_SIZE,
            "sortby": [{"field": "properties.datetime", "direction": "asc"}],
        }
        if extra_filter is not None:
            body["filter-lang"] = "cql2-json"
            body["filter"] = extra_filter
        # Каталоги поддерживают разные расширения фильтрации: CQL2 или query
        body.update(extra_body or {})

        scenes: list[S] = []
        request: tuple[str, str, dict[str, Any] | None] = (
            "POST",
            self._search_url(),
            body,
        )
        while True:
            page = await self._fetch(*request)
            try:
                scenes.extend(self._parse(feature) for feature in page.features)
            except (ValidationError, KeyError) as error:
                raise ExternalServiceError(UNAVAILABLE) from error
            if len(scenes) > MAX_SCENES:
                raise ExternalServiceError(
                    f"Найдено больше {MAX_SCENES} снимков: сузьте область или период"
                )
            following = next((link for link in page.links if link.rel == "next"), None)
            if following is None or not page.features:
                return scenes
            request = (following.method.upper(), following.href, following.body)

    async def _fetch(self, method: str, url: str, body: dict[str, Any] | None) -> _Page:
        try:
            if method == "POST":
                response = await self._http.post(url, json=body)
            else:
                response = await self._http.get(url)
            response.raise_for_status()
            return _Page.model_validate(response.json())
        except (httpx.HTTPError, ValueError) as error:
            # ValidationError — подкласс ValueError
            raise ExternalServiceError(UNAVAILABLE) from error


class CopernicusClient(StacClient[Scene]):
    """Каталог Copernicus Data Space Ecosystem: поиск без учётной записи, а файлы
    каналов — в хранилище eodata, доступном только с ней."""

    def _search_url(self) -> str:
        return f"{self._settings.copernicus_stac_url}/search"

    def _parse(self, feature: _Feature) -> Scene:
        return _scene(feature)
