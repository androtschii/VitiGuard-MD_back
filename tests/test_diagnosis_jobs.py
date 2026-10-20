import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from diagnosis_model import build_model, build_segmenter, photo
from fastapi.testclient import TestClient
from memory_storage import MemoryStorage

from app.api.routes import ai as ai_routes
from app.core.config import Settings
from app.core.errors import UnsupportedMediaError
from app.tasks import diagnosis as diagnosis_task
from app.tasks.diagnosis import DIAGNOSE, run_diagnosis
from app.worker.celery_app import TASK_MODULES, celery_app

CREDENTIALS = {"email": "grower@example.md", "password": "secret123"}
URL = "/api/v1/ai/diagnose/jobs"


def model_settings(tmp_path: Path, **values: Any) -> Settings:
    return Settings(
        _env_file=None,
        diagnosis_model_path=build_model(tmp_path / "tiny.onnx"),
        lesion_model_path=build_segmenter(tmp_path / "unet.onnx"),
        **values,
    )


async def test_worker_diagnoses_photo_and_removes_it(tmp_path: Path) -> None:
    storage = MemoryStorage()
    storage.put("jobs/u/t.jpg", photo((220, 0, 0)), "image/jpeg")

    result = await run_diagnosis(
        model_settings(tmp_path), storage, "u", "jobs/u/t.jpg", 2, False, None
    )

    assert result["owner_id"] == "u"
    assert result["diagnosis"]["disease"] == "healthy"
    assert len(result["diagnosis"]["top"]) == 2
    assert result["diagnosis"]["lesion"]["severity"] == pytest.approx(1)
    assert result["diagnosis"]["id"] is None
    assert storage.files == {}


async def test_broken_photo_is_removed_too(tmp_path: Path) -> None:
    storage = MemoryStorage()
    storage.put("jobs/u/t.jpg", b"not an image", "image/jpeg")

    with pytest.raises(UnsupportedMediaError):
        await run_diagnosis(
            model_settings(tmp_path), storage, "u", "jobs/u/t.jpg", 3, False, None
        )

    assert storage.files == {}


def test_task_is_registered_for_worker() -> None:
    assert "app.tasks.diagnosis" in TASK_MODULES
    assert diagnosis_task.diagnose_task.name == DIAGNOSE


def login(client: TestClient, email: str = CREDENTIALS["email"]) -> dict[str, str]:
    credentials = {**CREDENTIALS, "email": email}
    client.post("/api/v1/auth/register", json={**credentials, "full_name": "Ион"})
    token = client.post("/api/v1/auth/login", json=credentials).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def storage() -> MemoryStorage:
    return MemoryStorage()


@pytest.fixture
def client(
    api_client: TestClient, tmp_path: Path, storage: MemoryStorage
) -> TestClient:
    app: Any = api_client.app
    settings: Settings = app.state.settings
    app.state.settings = settings.model_copy(
        update={"diagnosis_model_path": build_model(tmp_path / "tiny.onnx")}
    )
    app.state.storage = storage
    return api_client


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def send_task(name: str, args: list[Any], task_id: str) -> SimpleNamespace:
        calls.append({"name": name, "args": args, "task_id": task_id})
        return SimpleNamespace(id=task_id)

    monkeypatch.setattr(celery_app, "send_task", send_task)
    return calls


def start(
    client: TestClient, headers: dict[str, str], data: bytes, **params: str
) -> Any:
    return client.post(
        URL,
        headers=headers,
        params=params,
        files={"image": ("leaf.jpg", data, "image/jpeg")},
    )


@pytest.mark.integration
def test_start_queues_task_with_photo(
    client: TestClient, storage: MemoryStorage, sent: list[dict[str, Any]]
) -> None:
    headers = login(client)

    response = start(client, headers, photo((220, 0, 0)), top_k="2")

    assert response.status_code == 202
    task_id = response.json()["task_id"]
    [call] = sent
    assert call["name"] == DIAGNOSE
    assert call["task_id"] == task_id
    owner, key, top_k, save, vineyard = call["args"]
    assert key == f"jobs/{owner}/{task_id}.jpg"
    assert (top_k, save, vineyard) == (2, False, None)
    assert list(storage.files) == [key]


@pytest.mark.integration
def test_bad_photo_is_rejected_before_queue(
    client: TestClient, storage: MemoryStorage, sent: list[dict[str, Any]]
) -> None:
    headers = login(client)

    response = start(client, headers, b"%PDF-1.7")

    assert response.status_code == 415
    assert sent == []
    assert storage.files == {}


@pytest.mark.integration
def test_without_model_nothing_is_queued(
    client: TestClient, sent: list[dict[str, Any]], tmp_path: Path
) -> None:
    app: Any = client.app
    app.state.settings = app.state.settings.model_copy(
        update={"diagnosis_model_path": tmp_path / "missing.onnx"}
    )

    response = start(client, login(client), photo((220, 0, 0)))

    assert response.status_code == 503
    assert sent == []


@pytest.mark.integration
@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("PENDING", "pending"),
        ("STARTED", "running"),
        ("RETRY", "running"),
        ("FAILURE", "failed"),
    ],
)
def test_job_status(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, state: str, expected: str
) -> None:
    monkeypatch.setattr(ai_routes, "job_state", lambda _: (state, None))

    response = client.get(
        f"{URL}/01a13ea2-b18e-77b7-aa02-2481b87c6b80", headers=login(client)
    )

    assert response.json() == {"status": expected, "result": None}


@pytest.mark.integration
def test_finished_job_is_shown_only_to_owner(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    headers = login(client)
    owner = client.get("/api/v1/users/me", headers=headers).json()["id"]
    storage = MemoryStorage()
    storage.put("key", photo((220, 0, 0)), "image/jpeg")
    result = asyncio.run(
        run_diagnosis(model_settings(tmp_path), storage, owner, "key", 3, False, None)
    )
    monkeypatch.setattr(ai_routes, "job_state", lambda _: ("SUCCESS", result))
    url = f"{URL}/01a13ea2-b18e-77b7-aa02-2481b87c6b80"

    done = client.get(url, headers=headers).json()

    assert done["status"] == "done"
    assert done["result"]["disease"] == "healthy"
    assert client.get(url, headers=login(client, "other@example.md")).status_code == (
        404
    )


@pytest.mark.integration
def test_worker_saves_to_history(
    client: TestClient, migrated_database_url: str, tmp_path: Path
) -> None:
    headers = login(client)
    owner = client.get("/api/v1/users/me", headers=headers).json()["id"]
    storage = MemoryStorage()
    storage.put("jobs/key.jpg", photo((220, 0, 0)), "image/jpeg")
    settings = model_settings(
        tmp_path,
        database_url=migrated_database_url,
    )
    result = asyncio.run(
        run_diagnosis(settings, storage, owner, "jobs/key.jpg", 3, True, None)
    )

    history = client.get("/api/v1/diagnoses", headers=headers).json()
    assert history["total"] == 1
    assert history["items"][0]["id"] == result["diagnosis"]["id"]
    assert history["items"][0]["severity"] == pytest.approx(1)
    assert "jobs/key.jpg" not in storage.files
    assert len(storage.files) == 1
