"""Чтение контуров участков из файлов GeoJSON и KML.

Здесь только разбор: на выходе — контуры с названиями, которые дальше
проверяются так же, как контур, нарисованный на карте."""

import json
from dataclasses import dataclass
from typing import Any
from xml.etree.ElementTree import Element

from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException

from app.core.errors import BadRequestError

KML = "{http://www.opengis.net/kml/2.2}"

# Ключи названия в свойствах GeoJSON: так их пишут QGIS, geojson.io, OSM
NAME_KEYS = ("name", "Name", "NAME", "title", "label")


@dataclass(frozen=True)
class BoundaryCandidate:
    """Контур из файла: название, геометрия GeoJSON или причина, по которой
    объект не подходит (например, это точка, а не полигон)."""

    name: str | None
    geometry: dict[str, Any] | None
    problem: str | None = None


def parse_boundaries(content: bytes) -> list[BoundaryCandidate]:
    text = content.decode("utf-8-sig", errors="replace").lstrip()
    if text.startswith("{"):
        return _parse_geojson(text)
    if text.startswith("<"):
        return _parse_kml(content)
    raise BadRequestError("Нужен файл GeoJSON или KML")


def _parse_geojson(text: str) -> list[BoundaryCandidate]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise BadRequestError(f"Файл не читается как GeoJSON: {error.msg}") from error

    match data:
        case {"type": "FeatureCollection", "features": list(features)}:
            pass
        case {"type": "Feature"}:
            features = [data]
        case {"type": str(), "coordinates": _}:
            features = [{"type": "Feature", "properties": {}, "geometry": data}]
        case _:
            raise BadRequestError("В файле нет объектов GeoJSON")

    candidates = []
    for feature in features:
        if not isinstance(feature, dict):
            candidates.append(BoundaryCandidate(None, None, "Нет геометрии"))
            continue
        properties = feature.get("properties")
        if not isinstance(properties, dict):
            properties = {}
        name = next(
            (str(properties[key]) for key in NAME_KEYS if properties.get(key)), None
        )
        candidates.append(_candidate(name, feature.get("geometry")))
    return candidates


def _candidate(name: str | None, geometry: Any) -> BoundaryCandidate:
    if isinstance(geometry, dict) and geometry.get("type") in (
        "Polygon",
        "MultiPolygon",
    ):
        return BoundaryCandidate(name, geometry)
    kind = geometry.get("type") if isinstance(geometry, dict) else None
    return BoundaryCandidate(
        name, None, f"Не полигон ({kind})" if kind else "Нет геометрии"
    )


def _parse_kml(content: bytes) -> list[BoundaryCandidate]:
    try:
        root = ElementTree.fromstring(content)
    except DefusedXmlException as error:
        # Сущности и DTD в KML не нужны: такие файлы — попытка атаки на парсер
        raise BadRequestError(
            "Файл KML содержит недопустимые конструкции XML"
        ) from error
    except ElementTree.ParseError as error:
        raise BadRequestError(f"Файл не читается как KML: {error}") from error

    candidates = []
    for placemark in root.iter(f"{KML}Placemark"):
        name = (placemark.findtext(f"{KML}name") or "").strip() or None
        try:
            polygons = [
                _kml_polygon(polygon) for polygon in placemark.iter(f"{KML}Polygon")
            ]
        except ValueError:
            candidates.append(BoundaryCandidate(name, None, "Координаты не читаются"))
            continue
        if not polygons:
            candidates.append(BoundaryCandidate(name, None, "Не полигон"))
        elif len(polygons) == 1:
            candidates.append(
                BoundaryCandidate(name, {"type": "Polygon", "coordinates": polygons[0]})
            )
        else:
            # Несколько полигонов в MultiGeometry — один участок из нескольких частей
            candidates.append(
                BoundaryCandidate(
                    name, {"type": "MultiPolygon", "coordinates": polygons}
                )
            )
    return candidates


def _kml_polygon(polygon: Element) -> list[list[list[float]]]:
    outer = polygon.find(f"{KML}outerBoundaryIs/{KML}LinearRing/{KML}coordinates")
    inner = polygon.findall(f"{KML}innerBoundaryIs/{KML}LinearRing/{KML}coordinates")
    return [_kml_ring(ring) for ring in [outer, *inner] if ring is not None]


def _kml_ring(coordinates: Element) -> list[list[float]]:
    # «lon,lat[,alt] lon,lat[,alt] …»: высоту отбрасываем, участку она не нужна
    points = []
    for chunk in (coordinates.text or "").split():
        lon, lat, *_ = (float(value) for value in chunk.split(","))
        points.append([lon, lat])
    return points
