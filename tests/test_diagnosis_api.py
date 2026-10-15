from pathlib import Path

import pytest
from diagnosis_model import build_model, photo
from fastapi.testclient import TestClient

from app.core.config import Settings

pytestmark = pytest.mark.integration

URL = "/api/v1/ai/diagnose"
CREDENTIALS = {"email": "grower@example.md", "password": "secret123"}


def use_model(client: TestClient, path: Path) -> None:
    app = client.app
    settings: Settings = app.state.settings  # type: ignore[attr-defined]
    app.state.settings = settings.model_copy(  # type: ignore[attr-defined]
        update={"diagnosis_model_path": path}
    )
    app.state.classifier = None  # type: ignore[attr-defined]


@pytest.fixture
def client(api_client: TestClient, tmp_path: Path) -> TestClient:
    use_model(api_client, build_model(tmp_path / "tiny.onnx"))
    return api_client


@pytest.fixture
def headers(client: TestClient) -> dict[str, str]:
    client.post("/api/v1/auth/register", json={**CREDENTIALS, "full_name": "Ион"})
    token = client.post("/api/v1/auth/login", json=CREDENTIALS).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def diagnose(
    client: TestClient, headers: dict[str, str], data: bytes, **params: int
) -> dict[str, object]:
    response = client.post(
        URL,
        headers=headers,
        params=params,
        files={"image": ("leaf.jpg", data, "image/jpeg")},
    )
    body: dict[str, object] = {"status": response.status_code, **response.json()}
    return body


def test_diagnose_confident(client: TestClient, headers: dict[str, str]) -> None:
    result = diagnose(client, headers, photo((255, 0, 0)))

    assert result["status"] == 200
    assert result["model"] == "tiny"
    assert result["disease"] == "healthy"
    assert result["confidence"] == pytest.approx(1, abs=0.001)
    assert result["low_confidence"] is False
    top = result["top"]
    assert isinstance(top, list)
    assert top[0]["disease"] == "healthy"
    assert len(top) == 3


def test_diagnose_marks_uncertain_answer(
    client: TestClient, headers: dict[str, str]
) -> None:
    # Серый снимок: ни один канал не преобладает, и модель не уверена
    result = diagnose(client, headers, photo((128, 128, 128)), top_k=6)

    assert result["status"] == 200
    assert result["disease"] == "powdery_mildew"
    assert isinstance(result["confidence"], float)
    assert result["confidence"] < 0.6
    assert result["low_confidence"] is True
    assert isinstance(result["top"], list)
    assert len(result["top"]) == 6


def test_model_is_loaded_once(client: TestClient, headers: dict[str, str]) -> None:
    diagnose(client, headers, photo((255, 0, 0)))
    loaded = client.app.state.classifier  # type: ignore[attr-defined]
    diagnose(client, headers, photo((0, 255, 0)))

    assert client.app.state.classifier is loaded  # type: ignore[attr-defined]


def test_rejects_large_file(client: TestClient, headers: dict[str, str]) -> None:
    result = diagnose(client, headers, b"0" * (10 * 1024 * 1024 + 1))

    assert result == {"status": 413, "detail": "Файл больше 10 МБ"}


def test_rejects_not_image(client: TestClient, headers: dict[str, str]) -> None:
    assert diagnose(client, headers, b"%PDF-1.7")["status"] == 415


def test_requires_login(client: TestClient) -> None:
    response = client.post(URL, files={"image": ("leaf.jpg", photo((255, 0, 0)))})

    assert response.status_code == 401


def test_without_model_gives_503(
    client: TestClient, headers: dict[str, str], tmp_path: Path
) -> None:
    use_model(client, tmp_path / "missing.onnx")

    result = diagnose(client, headers, photo((255, 0, 0)))

    assert result == {
        "status": 503,
        "detail": "Модель диагностики не загружена на сервер",
    }
