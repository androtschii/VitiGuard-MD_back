import asyncio
import uuid
from datetime import datetime

import httpx2 as httpx
from celery import shared_task

from app.core.config import Settings, get_settings
from app.db.session import create_engine, create_session_factory
from app.integrations.earth_search import EarthSearchClient
from app.services.vegetation_history import process_vineyard

SOFT_TIME_LIMIT = 30 * 60


async def run_vegetation(
    settings: Settings, vineyard_id: str, start: str, end: str
) -> dict[str, int]:
    engine = create_engine(settings)
    try:
        async with (
            httpx.AsyncClient(timeout=settings.external_http_timeout) as http,
            create_session_factory(engine)() as session,
        ):
            stats = await process_vineyard(
                session,
                EarthSearchClient(http, settings),
                uuid.UUID(vineyard_id),
                datetime.fromisoformat(start),
                datetime.fromisoformat(end),
                settings.vegetation_max_scene_cloud_cover,
            )
    finally:
        await engine.dispose()
    return vars(stats)


# Сезон — десятки снимков по несколько секунд: только в фоне. Аргументы —
# строки, чтобы задача сериализовалась в JSON. Общего лимита в 5 минут на
# сезон не хватает (70 снимков — около 10 минут), поэтому у задачи свой. Если
# задача всё же оборвётся, повторный запуск продолжит с необработанных снимков
@shared_task(
    name="satellite.process_vineyard",
    soft_time_limit=SOFT_TIME_LIMIT,
    time_limit=SOFT_TIME_LIMIT + 60,
)
def process_vineyard_task(vineyard_id: str, start: str, end: str) -> dict[str, int]:
    return asyncio.run(run_vegetation(get_settings(), vineyard_id, start, end))
