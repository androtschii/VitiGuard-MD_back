from pathlib import Path

import pytest
from diagnosis_model import photo
from fastapi.testclient import TestClient
from vine_model import build_vine_model

from app.core.config import Settings

pytestmark = pytest.mark.integration

URL = "/api/v1/ai/vine/detect"
CREDENTIALS = {"email": "vine@example.md", "password": "secret123"}


def use_vine_model(client: TestClient, path: Path) -> None:
    app = client.app
    settings: Settings = app.state.settings  # type: ignore[attr-defined]
    app.state.settings = settings.model_copy(  # type: ignore[attr-defined]
        update={"vine_model_path": path}
    )
    app.state.vine_detector = None  # type: ignore[attr-defined]


@pytest.fixture
def headers(api_client: TestClient) -> dict[str, str]:
    api_client.post("/api/v1/auth/register", json={**CREDENTIALS, "full_name": "Ион"})
    token = api_client.post("/api/v1/auth/login", json=CREDENTIALS).json()[
        "access_token"
    ]
    return {"Authorization": f"Bearer {token}"}


def detect(
    client: TestClient, headers: dict[str, str], data: bytes, **params: float
) -> tuple[int, dict[str, object]]:
    response = client.post(
        URL,
        headers=headers,
        params=params,
        files={"image": ("row.jpg", data, "image/jpeg")},
    )
    return response.status_code, response.json()


def test_detect_trunks_and_bunches(
    api_client: TestClient, headers: dict[str, str], tmp_path: Path
) -> None:
    use_vine_model(api_client, build_vine_model(tmp_path / "vine.onnx"))

    code, body = detect(api_client, headers, photo((0, 120, 0), size=(128, 64)))

    assert code == 200
    assert body["model"] == "tiny-yolo"
    assert len(str(body["version"])) == 16
    assert (body["image_width"], body["image_height"]) == (128, 64)
    assert (body["trunks"], body["bunches"]) == (1, 1)
    detections = body["detections"]
    assert isinstance(detections, list)
    assert detections[0]["label"] == "trunk"
    assert detections[0]["box"] == pytest.approx([0.375, 0, 0.625, 1])

    # Порог выше уверенности гроздей — остаётся только ствол
    code, body = detect(
        api_client, headers, photo((0, 120, 0), size=(128, 64)), confidence=0.8
    )
    assert (body["trunks"], body["bunches"]) == (1, 0)


def test_detect_errors(
    api_client: TestClient, headers: dict[str, str], tmp_path: Path
) -> None:
    use_vine_model(api_client, tmp_path / "absent.onnx")
    assert detect(api_client, headers, photo((0, 120, 0)))[0] == 503

    use_vine_model(api_client, build_vine_model(tmp_path / "vine.onnx"))
    assert detect(api_client, headers, b"not an image")[0] == 415
    assert detect(api_client, headers, photo((0, 120, 0)), confidence=1.5)[0] == 422
    response = api_client.post(
        URL, files={"image": ("row.jpg", photo((0, 120, 0)), "image/jpeg")}
    )
    assert response.status_code == 401
