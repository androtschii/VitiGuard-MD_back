import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from diagnosis_model import photo
from fastapi.testclient import TestClient

from app.api.deps import get_current_user
from app.core.config import Settings
from app.models.user import User

FRAME = photo((40, 120, 40))


def test_robot_telemetry_endpoint(client: TestClient, tmp_path: Path) -> None:
    app = client.app
    settings: Settings = app.state.settings  # type: ignore[attr-defined]
    fake_vine = tmp_path / "vine.onnx"
    fake_vine.write_bytes(b"dummy-vine-model-bytes")
    app.state.settings = settings.model_copy(  # type: ignore[attr-defined]
        update={"vine_model_path": fake_vine}
    )

    mock_user = MagicMock(spec=User)
    mock_user.id = uuid.uuid4()
    mock_user.is_active = True

    payload = {
        "robot_id": "harvester-test-01",
        "battery_pct": 78.5,
        "speed_mps": 0.6,
        "current_row": "row_12",
        "lat": 47.02,
        "lon": 28.83,
        "camera_fps": 28.4,
        "infer_latency_ms": 22.1,
        "active_model_version": "v000_old",
        "harvested_bunches": 45,
        "discarded_bunches": 2,
    }

    # Неавторизованный запрос возвращает 401
    assert client.post("/api/v1/robot/telemetry", json=payload).status_code == 401

    app.dependency_overrides[get_current_user] = lambda: mock_user  # type: ignore[attr-defined]
    try:
        resp = client.post(
            "/api/v1/robot/telemetry",
            json=payload,
            headers={"Authorization": "Bearer mock-token"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["robot_id"] == "harvester-test-01"
        assert data["status"] == "ok"
        assert data["update_available"] is True
        assert data["latest_model_version"] is not None
    finally:
        app.dependency_overrides.clear()  # type: ignore[attr-defined]


def upload(
    client: TestClient,
    robot_id: str,
    metadata: str = '{"tta_discrepancy": 0.35}',
    content: bytes = FRAME,
    filename: str = "hard_frame.jpg",
) -> Any:
    return client.post(
        "/api/v1/robot/active-learning/upload",
        files={"image": (filename, content, "image/jpeg")},
        data={
            "robot_id": robot_id,
            "uncertainty_score": "0.78",
            "metadata_json": metadata,
        },
        headers={"Authorization": "Bearer mock-token"},
    )


@pytest.fixture
def robot_client(client: TestClient, tmp_path: Path) -> Iterator[TestClient]:
    app: Any = client.app
    app.state.settings = app.state.settings.model_copy(
        update={"robot_upload_dir": tmp_path / "uploads"}
    )
    mock_user = MagicMock(spec=User)
    mock_user.id = uuid.uuid4()
    mock_user.is_active = True
    app.dependency_overrides[get_current_user] = lambda: mock_user
    yield client
    app.dependency_overrides.clear()


def test_robot_active_learning_upload(robot_client: TestClient, tmp_path: Path) -> None:
    resp = upload(robot_client, "harvester-test-01")

    assert resp.status_code == 201
    res = resp.json()
    assert res["status"] == "queued_for_label_studio"
    assert res["bytes_received"] == len(FRAME)
    frames = list((tmp_path / "uploads" / "harvester-test-01").iterdir())
    assert [frame.read_bytes() for frame in frames] == [FRAME]
    queue = (tmp_path / "uploads" / "active_learning_queue.jsonl").read_text(
        encoding="utf-8"
    )
    assert json.loads(queue)["meta"] == {"tta_discrepancy": 0.35}


@pytest.mark.parametrize("robot_id", ["../../outside", "a/b", "робот", ""])
def test_robot_id_cannot_leave_upload_dir(
    robot_client: TestClient, tmp_path: Path, robot_id: str
) -> None:
    assert upload(robot_client, robot_id).status_code == 422
    assert not (tmp_path / "outside").exists()


def test_broken_metadata_is_rejected(robot_client: TestClient, tmp_path: Path) -> None:
    resp = upload(robot_client, "harvester-test-01", metadata="{not json")

    assert resp.status_code == 400
    assert not (tmp_path / "uploads").exists()


def test_only_images_are_saved_with_extension_from_content(
    robot_client: TestClient, tmp_path: Path
) -> None:
    # Не изображение — 415, на диск ничего не попадает
    resp = upload(robot_client, "harvester-test-01", content=b"MZ not an image")
    assert resp.status_code == 415
    assert not (tmp_path / "uploads").exists()

    # Расширение берётся из содержимого, а не из имени, которое прислал клиент
    png = photo((40, 120, 40), image_format="PNG")
    resp = upload(robot_client, "harvester-test-01", content=png, filename="x.exe")
    assert resp.status_code == 201
    frames = list((tmp_path / "uploads" / "harvester-test-01").iterdir())
    assert [frame.suffix for frame in frames] == [".png"]
