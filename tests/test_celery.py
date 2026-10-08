from collections.abc import Iterator

import pytest
from celery import Celery
from celery.contrib.testing.worker import start_worker

from app.core.config import Settings
from app.tasks.system import ping
from app.worker.celery_app import TASK_MODULES, celery_app, create_celery


def test_celery_is_configured_from_settings() -> None:
    settings = Settings(
        _env_file=None,
        celery_task_soft_time_limit=100,
        celery_task_time_limit=120,
    )

    app = create_celery(settings)

    assert app.conf.broker_url == "redis://127.0.0.1:6379/0"
    assert app.conf.result_backend == "redis://127.0.0.1:6379/1"
    assert app.conf.task_soft_time_limit == 100
    assert app.conf.task_time_limit == 120
    assert list(app.conf.include) == TASK_MODULES


def test_celery_accepts_only_json() -> None:
    conf = create_celery(Settings(_env_file=None)).conf

    assert conf.task_serializer == "json"
    assert conf.result_serializer == "json"
    assert list(conf.accept_content) == ["json"]


def test_celery_is_safe_for_long_tasks() -> None:
    conf = create_celery(Settings(_env_file=None)).conf

    assert conf.task_acks_late is True
    assert conf.worker_prefetch_multiplier == 1


def test_module_level_app_is_available_for_the_worker() -> None:
    assert celery_app.main == "vitiguard"


def test_ping_task_returns_pong() -> None:
    assert ping() == "pong"


@pytest.fixture
def celery_with_worker(redis_url: str) -> Iterator[Celery]:
    app = create_celery(
        Settings(
            _env_file=None,
            celery_broker_url=f"{redis_url}/0",  # type: ignore[arg-type]
            celery_result_backend=f"{redis_url}/1",  # type: ignore[arg-type]
        )
    )
    # solo — воркер в одном процессе: работает и в Windows, и внутри теста
    with start_worker(app, pool="solo", perform_ping_check=False, loglevel="WARNING"):
        yield app


@pytest.mark.integration
def test_worker_runs_task_through_redis(celery_with_worker: Celery) -> None:
    result = celery_with_worker.send_task("system.ping")

    assert result.get(timeout=15) == "pong"
