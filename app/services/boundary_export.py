"""Выгрузка контуров участков в GeoJSON и Shapefile — для QGIS, ArcGIS и
других ГИС, с которыми работают агрономы и кадастровые инженеры."""

import io
import json
import zipfile
from enum import StrEnum
from itertools import pairwise
from typing import Any

import shapefile

from app.repositories.vineyard import VineyardView


class ExportFormat(StrEnum):
    """Формат выгрузки контуров участков: GeoJSON, KML или Shapefile."""

    geojson = "geojson"
    shapefile = "shapefile"


# WGS84 в формате ESRI WKT: без .prj ГИС не знает, в каких координатах файл
WGS84_PRJ = (
    'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,'
    '298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
)


def _properties(view: VineyardView) -> dict[str, Any]:
    return {
        "id": str(view.vineyard.id),
        "name": view.vineyard.name,
        "area_ha": round(view.area_ha, 4),
        "created": view.vineyard.created_at.date().isoformat(),
    }


def to_geojson(views: list[VineyardView]) -> bytes:
    collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": _properties(view),
                "geometry": view.geometry,
            }
            for view in views
        ],
    }
    return json.dumps(collection, ensure_ascii=False).encode()


def _signed_area(ring: list[list[float]]) -> float:
    return sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in pairwise(ring)) / 2


def _oriented(ring: list[list[float]], clockwise: bool) -> list[list[float]]:
    is_clockwise = _signed_area(ring) < 0
    return ring if is_clockwise == clockwise else ring[::-1]


def _shapefile_parts(geometry: dict[str, Any]) -> list[list[list[float]]]:
    """Кольца MultiPolygon в порядке Shapefile: внешний контур — по часовой
    стрелке, вырезы — против. В GeoJSON (RFC 7946) направления обратные, и без
    разворота ГИС приняли бы участок за вырез."""
    parts = []
    for polygon in geometry["coordinates"]:
        outer, *holes = polygon
        parts.append(_oriented(outer, clockwise=True))
        parts.extend(_oriented(hole, clockwise=False) for hole in holes)
    return parts


def to_shapefile_zip(views: list[VineyardView], basename: str) -> bytes:
    shp, shx, dbf = io.BytesIO(), io.BytesIO(), io.BytesIO()
    with shapefile.Writer(
        shp=shp, shx=shx, dbf=dbf, shapeType=shapefile.POLYGON, encoding="utf-8"
    ) as writer:
        # Имена полей dBase — не длиннее 10 символов
        writer.field("id", "C", size=36)
        writer.field("name", "C", size=254)
        writer.field("area_ha", "N", size=14, decimal=4)
        writer.field("created", "C", size=10)
        for view in views:
            writer.poly(_shapefile_parts(view.geometry))
            properties = _properties(view)
            writer.record(
                properties["id"],
                properties["name"],
                properties["area_ha"],
                properties["created"],
            )

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr(f"{basename}.shp", shp.getvalue())
        zipped.writestr(f"{basename}.shx", shx.getvalue())
        zipped.writestr(f"{basename}.dbf", dbf.getvalue())
        zipped.writestr(f"{basename}.prj", WGS84_PRJ)
        # Кодировка атрибутов: без .cpg QGIS читает кириллицу как Latin-1
        zipped.writestr(f"{basename}.cpg", "UTF-8")
    return archive.getvalue()
