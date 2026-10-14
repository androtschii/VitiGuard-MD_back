import asyncio

import httpx2 as httpx
from celery import shared_task

from app import __version__
from app.core.config import Settings, get_settings
from app.db.session import create_engine, create_session_factory
from app.integrations.overpass import OverpassClient
from app.services.osm import import_osm_vineyards


async def run_import(settings: Settings) -> dict[str, int]:
    async with httpx.AsyncClient(
        headers={"User-Agent": f"vitiguard-md/{__version__}"}
    ) as http:
        result = await OverpassClient(http, settings).vineyards()
    engine = create_engine(settings)
    try:
        async with create_session_factory(engine)() as session:
            stats = await import_osm_vineyards(session, result.vineyards)
            await session.commit()
    finally:
        await engine.dispose()
    return {
        "imported": stats.imported,
        "deleted": stats.deleted,
        "skipped": result.skipped,
    }


# Выгрузка всей страны идёт минуту и больше — только в фоне. Повтор при сбое не
# нужен: импорт запускают вручную, а Overpass при перегрузке отвечает отказом
@shared_task(name="osm.import_vineyards")
def import_vineyards() -> dict[str, int]:
    return asyncio.run(run_import(get_settings()))
