"""Геометрии GeoJSON (RFC 7946) для участков: Polygon и MultiPolygon в WGS84.

Pydantic проверяет только структуру: долготу и широту, замкнутость колец,
минимальное число точек. Топологию (самопересечения) проверяет PostGIS."""

from typing import Annotated, Literal

from pydantic import AfterValidator, Field

from app.schemas.base import RequestSchema, ResponseSchema

Longitude = Annotated[float, Field(ge=-180, le=180)]
Latitude = Annotated[float, Field(ge=-90, le=90)]
# Высота (третья координата) участку не нужна, поэтому точка — ровно два числа
Position = tuple[Longitude, Latitude]


def _closed(ring: list[Position]) -> list[Position]:
    if ring[0] != ring[-1]:
        raise ValueError("Контур не замкнут: первая и последняя точки должны совпадать")
    return ring


# Треугольник — минимальная фигура: три вершины и повтор первой
LinearRing = Annotated[list[Position], Field(min_length=4), AfterValidator(_closed)]
# Первое кольцо — внешний контур, остальные — дыры (пруд, постройка)
PolygonCoordinates = Annotated[list[LinearRing], Field(min_length=1)]


class PolygonGeometry(RequestSchema):
    """Геометрия GeoJSON Polygon в WGS 84 (долгота, широта)."""

    type: Literal["Polygon"]
    coordinates: PolygonCoordinates


class MultiPolygonGeometry(RequestSchema):
    """Геометрия GeoJSON MultiPolygon в WGS 84 (долгота, широта)."""

    type: Literal["MultiPolygon"]
    coordinates: Annotated[list[PolygonCoordinates], Field(min_length=1)]


# Тип геометрии выбирается по полю type, и ошибка указывает на нужный вариант
AreaGeometry = Annotated[
    PolygonGeometry | MultiPolygonGeometry, Field(discriminator="type")
]


class MultiPolygonRead(ResponseSchema):
    """Контур участка в ответах. В базе участок всегда MultiPolygon, поэтому
    клиенту не нужно разбирать два варианта."""

    type: Literal["MultiPolygon"]
    coordinates: list[list[list[tuple[float, float]]]]
