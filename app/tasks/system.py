import asyncio
from dataclasses import asdict
from datetime import UTC, datetime

from celery import shared_task

from app.core.config import Settings, get_settings
from app.db.session import create_engine, create_session_factory
from app.integrations.object_storage import ObjectStorage, Storage
from app.services.cleanup import cleanup


@shared_task(name="system.ping")
def ping() -> str:
    """Проверка всей цепочки: API → Redis → воркер → Redis → API."""
    return "pong"


CLEANUP = "system.cleanup"


async def run_cleanup(settings: Settings, storage: Storage) -> dict[str, int]:
    engine = create_engine(settings)
    try:
        async with create_session_factory(engine)() as session:
            stats = await cleanup(session, storage, datetime.now(UTC))
    finally:
        await engine.dispose()
    return asdict(stats)


# Раз в сутки по расписанию Celery Beat (pr-097)
@shared_task(name=CLEANUP)
def cleanup_task() -> dict[str, int]:
    settings = get_settings()
    return asyncio.run(run_cleanup(settings, ObjectStorage.from_settings(settings)))
