import uuid
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.api.deps import get_current_user
from app.core.config import Settings
from app.models.user import User


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


def test_robot_active_learning_upload(client: TestClient) -> None:
    app = client.app
    mock_user = MagicMock(spec=User)
    mock_user.id = uuid.uuid4()
    mock_user.is_active = True

    frame_bytes = b"fake-jpg-binary-image-data-for-active-learning"
    files = {"image": ("hard_frame.jpg", frame_bytes, "image/jpeg")}
    data = {
        "robot_id": "harvester-test-01",
        "uncertainty_score": "0.78",
        "metadata_json": '{"tta_discrepancy": 0.35}',
    }

    app.dependency_overrides[get_current_user] = lambda: mock_user  # type: ignore[attr-defined]
    try:
        resp = client.post(
            "/api/v1/robot/active-learning/upload",
            files=files,
            data=data,
            headers={"Authorization": "Bearer mock-token"},
        )
        assert resp.status_code == 201
        res = resp.json()
        assert res["status"] == "queued_for_label_studio"
        assert res["bytes_received"] == len(frame_bytes)
    finally:
        app.dependency_overrides.clear()  # type: ignore[attr-defined]
