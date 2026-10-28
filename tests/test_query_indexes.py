"""Запросы приложения используют индексы (pr-091).

Тест перехватывает SQL, который на самом деле выполняют сервисы, и смотрит
его план с запретом перебора таблицы: если выражение запроса разойдётся с
выражением индекса (например, другое приведение к geography), план покажет
Seq Scan, и тест упадёт — раньше, чем замедлится сервер."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession

from app.integrations.overpass import OsmVineyard as OsmRecord
from app.models import User
from app.models.vineyard import Vineyard
from app.repositories.vineyard import VineyardRepository
from app.services.ndvi_tiles import latest_maps
from app.services.osm import import_osm_vineyards, vineyard_context

pytestmark = pytest.mark.integration


def square(west: float, south: float = 47.14, side: float = 0.002) -> dict[str, Any]:
    ring = [
        [west, south],
        [west + side, south],
        [west + side, south + side / 2],
        [west, south + side / 2],
        [west, south],
    ]
    return {"type": "Polygon", "coordinates": [ring]}


async def add_owner(session: AsyncSession) -> User:
    owner = User(email="grower@example.md", hashed_password="hash")
    session.add(owner)
    await session.flush()
    return owner


async def add_vineyard(
    session: AsyncSession, owner: User, name: str, geometry: dict[str, Any]
) -> Vineyard:
    return await VineyardRepository(session).create_from_geojson(
        owner.id, name, geometry
    )


@contextmanager
def captured(session: AsyncSession) -> Iterator[list[tuple[str, Any]]]:
    statements: list[tuple[str, Any]] = []

    def remember(_: Any, __: Any, statement: str, parameters: Any, *___: Any) -> None:
        statements.append((statement, parameters))

    engine = session.get_bind()
    event.listen(engine, "before_cursor_execute", remember)
    try:
        yield statements
    finally:
        event.remove(engine, "before_cursor_execute", remember)


async def plans(
    session: AsyncSession, statements: list[tuple[str, Any]], marker: str
) -> list[str]:
    connection = await session.connection()
    # Индекс должен быть пригоден: на тестовых десятках строк планировщик и
    # так выбрал бы перебор, поэтому перебор запрещается
    await connection.exec_driver_sql("SET LOCAL enable_seqscan = off")
    found = []
    for statement, parameters in statements:
        if marker in statement:
            result = await connection.exec_driver_sql(
                f"EXPLAIN {statement}", parameters
            )
            found.append("\n".join(row[0] for row in result))
    return found


async def test_context_queries_use_geography_index(session: AsyncSession) -> None:
    owner = await add_owner(session)
    vineyard = await add_vineyard(session, owner, "Участок", square(28.860))
    await import_osm_vineyards(
        session,
        [OsmRecord(osm_id="way/1", name=None, geometry=square(28.866))],
    )

    with captured(session) as statements:
        await vineyard_context(session, vineyard)

    # Соседи в 1 км и ближайший — по geography: оба через новый индекс
    geography = [
        (statement, parameters)
        for statement, parameters in statements
        if "osm_vineyards" in statement
        and ("ST_DWithin" in statement or "<->" in statement)
    ]
    assert len(geography) == 2
    for plan in await plans(session, geography, "osm_vineyards"):
        assert "ix_osm_vineyards_geom_geography" in plan, plan


async def test_nearby_vineyards_use_geography_index(session: AsyncSession) -> None:
    owner = await add_owner(session)
    await add_vineyard(session, owner, "Участок", square(28.860))

    with captured(session) as statements:
        await VineyardRepository(session).nearest(
            owner.id, 28.861, 47.1405, limit=5, max_distance_m=2_000
        )

    # Ищется среди участков одного владельца: их единицы, и индекс по
    # владельцу отбирает их быстрее пространственного — отдельный индекс
    # по geography для участков не нужен
    [plan] = await plans(session, statements, "ST_DWithin")
    assert "ix_vineyards_owner_id" in plan, plan


async def test_nearest_osm_beyond_50_km_is_not_reported(session: AsyncSession) -> None:
    owner = await add_owner(session)
    vineyard = await add_vineyard(session, owner, "Участок", square(28.860))
    # 0,8° долготы на широте Крикова — около 60 км
    await import_osm_vineyards(
        session,
        [OsmRecord(osm_id="way/9", name=None, geometry=square(29.660))],
    )

    context = await vineyard_context(session, vineyard)

    assert context.osm_vineyards_within_1km == 0
    assert context.nearest_osm_distance_m is None


async def test_latest_ndvi_maps_use_scan_time_index(session: AsyncSession) -> None:
    owner = await add_owner(session)
    await add_vineyard(session, owner, "Участок", square(28.860))

    with captured(session) as statements:
        await latest_maps(session, owner.id, (28.85, 47.13, 28.87, 47.15), None)

    # Последний снимок каждого участка — по индексу (участок, время съёмки),
    # без сортировки всех снимков: их у участка десятки за каждый сезон
    [plan] = await plans(session, statements, "satellite_scans")
    assert "ix_satellite_scans_vineyard_id_acquired_at" in plan, plan
