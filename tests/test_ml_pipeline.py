"""Сквозной путь фото листа через API с тестовыми моделями вместо настоящих
весов: классификатор с картой активации, U-Net, наложения, история, заключение
агронома и диагностика в фоне через настоящий воркер Celery и Redis."""

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from celery import Celery
from celery.contrib.testing.worker import start_worker
from diagnosis_model import build_model, build_segmenter, gps_photo
from fastapi.testclient import TestClient
from memory_storage import MemoryStorage

from app.api.routes import ai as ai_routes
from app.core.config import Settings
from app.integrations import object_storage
from app.tasks import diagnosis as diagnosis_task
from app.worker.celery_app import create_celery

pytestmark = pytest.mark.integration

CREDENTIALS = {"email": "agro@example.md", "password": "secret123"}
# Красный лист: тестовый классификатор отвечает «healthy», U-Net видит
# поражение там, где красный канал преобладает, — на всём листе
LEAF = gps_photo((47, 8, 24), (28, 51, 36), color=(220, 0, 0))


@pytest.fixture
def storage() -> MemoryStorage:
    return MemoryStorage()


@pytest.fixture
def models(tmp_path: Path) -> dict[str, Path]:
    return {
        "diagnosis_model_path": build_model(tmp_path / "cam.onnx", with_cam=True),
        "lesion_model_path": build_segmenter(tmp_path / "unet.onnx"),
    }


@pytest.fixture
def client(
    api_client: TestClient, storage: MemoryStorage, models: dict[str, Path]
) -> TestClient:
    app: Any = api_client.app
    app.state.settings = app.state.settings.model_copy(update=models)
    app.state.classifier = None
    app.state.segmenter = None
    app.state.storage = storage
    return api_client


def login(client: TestClient) -> dict[str, str]:
    client.post(
        "/api/v1/auth/register",
        json={**CREDENTIALS, "full_name": "Мария", "role": "agronomist"},
    )
    token = client.post("/api/v1/auth/login", json=CREDENTIALS).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def upload(client: TestClient, url: str, headers: dict[str, str]) -> Any:
    return client.post(
        url,
        headers=headers,
        params={"save": "true", "top_k": "2"},
        files={"image": ("leaf.jpg", LEAF, "image/jpeg")},
    )


def test_photo_goes_through_whole_pipeline(
    client: TestClient, storage: MemoryStorage
) -> None:
    headers = login(client)

    diagnosis = upload(client, "/api/v1/ai/diagnose", headers).json()

    assert diagnosis["disease"] == "healthy"
    assert [item["disease"] for item in diagnosis["top"]][:1] == ["healthy"]
    assert len(diagnosis["top"]) == 2
    assert diagnosis["heatmap"].startswith("data:image/png;base64,")
    assert diagnosis["lesion"]["severity"] == pytest.approx(1)
    assert diagnosis["location"] == {"lat": pytest.approx(47.14), "lon": 28.86}
    # Фото, тепловая карта и маска
    assert len(storage.files) == 3

    url = f"/api/v1/diagnoses/{diagnosis['id']}"
    [record] = client.get("/api/v1/diagnoses", headers=headers).json()["items"]
    assert record["id"] == diagnosis["id"]
    assert record["overlays"] == ["heatmap", "lesion"]
    assert record["severity"] == pytest.approx(1)
    assert record["location"] == diagnosis["location"]
    for overlay in ("heatmap", "lesion"):
        image = client.get(f"{url}/overlays/{overlay}", headers=headers)
        assert image.content.startswith(b"\x89PNG")

    review = client.put(
        f"{url}/review", headers=headers, json={"disease": "downy_mildew"}
    ).json()["review"]
    assert review["disease"] == "downy_mildew"

    assert client.delete(url, headers=headers).status_code == 204
    assert storage.files == {}
    assert client.get("/api/v1/diagnoses", headers=headers).json()["total"] == 0


@pytest.fixture
def worker(
    redis_url: str,
    migrated_database_url: str,
    storage: MemoryStorage,
    models: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Celery]:
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url=migrated_database_url,  # type: ignore[arg-type]
        celery_broker_url=f"{redis_url}/0",  # type: ignore[arg-type]
        celery_result_backend=f"{redis_url}/1",  # type: ignore[arg-type]
        diagnosis_model_path=models["diagnosis_model_path"],
        lesion_model_path=models["lesion_model_path"],
    )
    app = create_celery(settings)
    # API и воркер — в одном процессе: общая очередь, настройки и хранилище
    monkeypatch.setattr(ai_routes, "celery_app", app)
    monkeypatch.setattr(diagnosis_task, "get_settings", lambda: settings)
    monkeypatch.setattr(
        object_storage.ObjectStorage, "from_settings", lambda _: storage
    )
    with start_worker(app, pool="solo", perform_ping_check=False, loglevel="WARNING"):
        yield app


def test_background_diagnosis_through_worker(
    client: TestClient, storage: MemoryStorage, worker: Celery
) -> None:
    headers = login(client)

    started = upload(client, "/api/v1/ai/diagnose/jobs", headers)
    assert started.status_code == 202
    url = f"/api/v1/ai/diagnose/jobs/{started.json()['task_id']}"

    deadline = time.monotonic() + 30
    job = client.get(url, headers=headers).json()
    while job["status"] in ("pending", "running") and time.monotonic() < deadline:
        time.sleep(0.2)
        job = client.get(url, headers=headers).json()

    assert job["status"] == "done"
    result = job["result"]
    assert result["disease"] == "healthy"
    assert result["location"]["lon"] == 28.86
    # Временное фото удалено, в хранилище — фото проверки и два наложения
    assert not any(key.startswith("jobs/") for key in storage.files)
    assert len(storage.files) == 3
    [record] = client.get("/api/v1/diagnoses", headers=headers).json()["items"]
    assert record["id"] == result["id"]
    assert record["overlays"] == ["heatmap", "lesion"]
