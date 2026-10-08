from celery import shared_task


@shared_task(name="system.ping")
def ping() -> str:
    """Проверка всей цепочки: API → Redis → воркер → Redis → API."""
    return "pong"
