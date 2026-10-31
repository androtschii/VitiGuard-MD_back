from typing import Any

from celery import Celery
from celery.schedules import crontab
from celery.signals import setup_logging

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging

# Модули с задачами; новые задачи (диагностика, спутниковые индексы, прогнозы)
# добавляются сюда
TASK_MODULES = [
    "app.tasks.diagnosis",
    "app.tasks.mail",
    "app.tasks.osm",
    "app.tasks.risk",
    "app.tasks.satellite",
    "app.tasks.telegram",
    "app.tasks.system",
]


def create_celery(settings: Settings) -> Celery:
    application = Celery(
        "vitiguard",
        broker=str(settings.celery_broker_url),
        backend=str(settings.celery_result_backend),
        include=TASK_MODULES,
    )
    application.conf.update(
        # JSON вместо pickle: из очереди нельзя выполнить произвольный код
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        # Задача подтверждается после выполнения, а не при получении: если воркер
        # упал в середине, задача вернётся в очередь. Поэтому задачи должны быть
        # идемпотентными — повторный запуск даёт тот же результат
        task_acks_late=True,
        # Воркер берёт по одной задаче: тяжёлые задачи (нейросеть, растры) не
        # должны ждать в буфере занятого воркера, пока соседний простаивает
        worker_prefetch_multiplier=1,
        task_track_started=True,
        task_soft_time_limit=settings.celery_task_soft_time_limit,
        task_time_limit=settings.celery_task_time_limit,
        # Результаты нужны вызывающему коду недолго, потом их удаляет Redis
        result_expires=3600,
        # Расписание Celery Beat: риск всех участков — по свежей погоде и
        # прогнозу каждые несколько часов (Open-Meteo обновляет прогноз раз в час)
        beat_schedule={
            "risk-refresh": {
                "task": "risk.refresh_all",
                "schedule": settings.risk_refresh_hours * 3600,
            },
            # Очистка временных файлов и устаревших записей (pr-097): в 1:30 UTC,
            # ночью в Молдове, когда участками никто не занимается
            "cleanup": {
                "task": "system.cleanup",
                "schedule": crontab(hour=1, minute=30),
            },
        },
        # Redis возвращает неподтверждённую задачу в очередь по истечении этого
        # срока: он должен быть больше самой долгой задачи
        broker_transport_options={"visibility_timeout": 3600},
        broker_connection_retry_on_startup=True,
    )
    return application


celery_app = create_celery(get_settings())


# Без этого обработчика Celery настроил бы свой журнал в обычном тексте; с ним
# воркер пишет тот же JSON, что и API
@setup_logging.connect
def _configure_worker_logging(**_: Any) -> None:
    configure_logging(get_settings())
