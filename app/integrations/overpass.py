"""Клиент Overpass API: границы виноградников Молдовы из OpenStreetMap.

Полигоны landuse=vineyard — открытые данные OSM (лицензия ODbL). Пути (way)
приходят замкнутыми контурами, мультиполигоны (relation) — набором путей с
ролями outer и inner."""

from dataclasses import dataclass, field
from typing import Any

import httpx2 as httpx

from app.core.config import Settings
from app.core.errors import ExternalServiceError

UNAVAILABLE = "Overpass API недоступен, попробуйте позже"

# Граница страны по ISO-коду — та же, что использует сам OSM; out geom
# возвращает координаты вершин сразу, без отдельных запросов за узлами
QUERY = """
[out:json][timeout:{timeout}];
area["ISO3166-1"="MD"][admin_level=2]->.md;
(
  way["landuse"="vineyard"](area.md);
  relation["landuse"="vineyard"](area.md);
);
out geom;
"""

Ring = list[list[float]]


@dataclass(frozen=True)
class OsmVineyard:
    osm_id: str
    name: str | None
    # MultiPolygon в GeoJSON: участок-relation может состоять из нескольких частей
    geometry: dict[str, Any]


@dataclass
class OverpassResult:
    vineyards: list[OsmVineyard] = field(default_factory=list)
    # Объекты, из которых не собрать полигон: незамкнутые пути, мультиполигоны,
    # контур которых разбит на несколько путей
    skipped: int = 0


def _ring(points: list[dict[str, float]]) -> Ring | None:
    ring = [[point["lon"], point["lat"]] for point in points]
    # Полигону нужны минимум три вершины и совпадающие первая и последняя точки
    if len(ring) < 4 or ring[0] != ring[-1]:
        return None
    return ring


def _parse(element: dict[str, Any]) -> list[list[Ring]] | None:
    if element["type"] == "way":
        ring = _ring(element.get("geometry", []))
        return [[ring]] if ring else None

    outers: list[Ring] = []
    inners: list[Ring] = []
    for member in element.get("members", []):
        if member.get("type") != "way":
            continue
        ring = _ring(member.get("geometry", []))
        if ring is None:
            # Контур, разрезанный на несколько путей, нужно сшивать; таких
            # мультиполигонов мало, и они пропускаются целиком
            return None
        (inners if member.get("role") == "inner" else outers).append(ring)
    if not outers:
        return None
    # Вырезы относятся к первой внешней части: в виноградниках вырезы редки, а
    # точную принадлежность вырезов PostGIS всё равно проверяет при импорте
    return [[outers[0], *inners], *[[outer] for outer in outers[1:]]]


class OverpassClient:
    def __init__(self, http: httpx.AsyncClient, settings: Settings) -> None:
        self._http = http
        self._settings = settings

    async def vineyards(self) -> OverpassResult:
        timeout = self._settings.overpass_timeout
        try:
            response = await self._http.post(
                self._settings.overpass_url,
                data={"data": QUERY.format(timeout=timeout)},
                # Сервер Overpass сам ограничен timeout секундами, клиент ждёт чуть дольше
                timeout=timeout + 30,
            )
            response.raise_for_status()
            elements = response.json()["elements"]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
            raise ExternalServiceError(UNAVAILABLE) from error

        result = OverpassResult()
        for element in elements:
            polygons = _parse(element)
            if polygons is None:
                result.skipped += 1
                continue
            result.vineyards.append(
                OsmVineyard(
                    osm_id=f"{element['type']}/{element['id']}",
                    name=element.get("tags", {}).get("name"),
                    geometry={"type": "MultiPolygon", "coordinates": polygons},
                )
            )
        return result
