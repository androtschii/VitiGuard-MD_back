import asyncio

import httpx2 as httpx
from celery import shared_task
from sqlalchemy import select

from app.core.config import Settings, get_settings
from app.core.errors import ExternalServiceError
from app.db.session import create_engine, create_session_factory
from app.integrations.open_meteo import OpenMeteoClient
from app.models.vineyard import Vineyard
from app.services.risk import refresh_risks

REFRESH_ALL = "risk.refresh_all"


async def run_refresh_all(settings: Settings) -> dict[str, int]:
    """Пересчёт риска всех участков. Сбой погоды по одному участку не
    останавливает остальные — участок пересчитается в следующий раз."""
    engine = create_engine(settings)
    refreshed = failed = 0
    try:
        async with (
            httpx.AsyncClient(timeout=settings.external_http_timeout) as http,
            create_session_factory(engine)() as session,
        ):
            client = OpenMeteoClient(http, settings)
            vineyards = (await session.scalars(select(Vineyard))).all()
            for vineyard in vineyards:
                try:
                    await refresh_risks(session, client, vineyard)
                except ExternalServiceError:
                    await session.rollback()
                    failed += 1
                else:
                    refreshed += 1
    finally:
        await engine.dispose()
    return {"refreshed": refreshed, "failed": failed}


# Запускается Celery Beat по расписанию (pr-079). Участков может быть много, а
# на каждый — два запроса к Open-Meteo, поэтому у задачи свой лимит времени
@shared_task(name=REFRESH_ALL, soft_time_limit=30 * 60, time_limit=31 * 60)
def refresh_all() -> dict[str, int]:
    return asyncio.run(run_refresh_all(get_settings()))
