from pathlib import Path

import pytest
from diagnosis_model import build_model, build_segmenter, photo
from fastapi.testclient import TestClient

from app.core.config import Settings

pytestmark = pytest.mark.integration

URL = "/api/v1/ai/diagnose"
CREDENTIALS = {"email": "grower@example.md", "password": "secret123"}


def use_models(client: TestClient, classifier: Path, segmenter: Path) -> None:
    """Тестовые модели вместо настоящих: те могут лежать в models/ локально."""
    app = client.app
    settings: Settings = app.state.settings  # type: ignore[attr-defined]
    app.state.settings = settings.model_copy(  # type: ignore[attr-defined]
        update={"diagnosis_model_path": classifier, "lesion_model_path": segmenter}
    )
    app.state.classifier = None  # type: ignore[attr-defined]
    app.state.segmenter = None  # type: ignore[attr-defined]


@pytest.fixture
def client(api_client: TestClient, tmp_path: Path) -> TestClient:
    use_models(api_client, build_model(tmp_path / "tiny.onnx"), tmp_path / "no.onnx")
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
    # Модель без карты активации, модели сегментации нет
    assert result["heatmap"] is None
    assert result["lesion"] is None


def test_diagnose_with_heatmap_and_lesion(
    client: TestClient, headers: dict[str, str], tmp_path: Path
) -> None:
    use_models(
        client,
        build_model(tmp_path / "cam.onnx", with_cam=True),
        build_segmenter(tmp_path / "unet.onnx"),
    )

    result = diagnose(client, headers, photo((220, 0, 0)))

    assert result["status"] == 200
    assert isinstance(result["heatmap"], str)
    assert result["heatmap"].startswith("data:image/png;base64,")
    lesion = result["lesion"]
    assert isinstance(lesion, dict)
    assert lesion["mask"].startswith("data:image/png;base64,")
    assert lesion["severity"] == pytest.approx(1)
    segmenter = client.app.state.segmenter  # type: ignore[attr-defined]
    diagnose(client, headers, photo((220, 0, 0)))
    assert client.app.state.segmenter is segmenter  # type: ignore[attr-defined]


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
    use_models(client, tmp_path / "missing.onnx", tmp_path / "no.onnx")

    result = diagnose(client, headers, photo((255, 0, 0)))

    assert result == {
        "status": 503,
        "detail": "Модель диагностики не загружена на сервер",
    }


def test_classifier_model_for_offline_use(
    client: TestClient, headers: dict[str, str]
) -> None:
    meta = client.get("/api/v1/ai/models/classifier", headers=headers)

    assert meta.status_code == 200
    body = meta.json()
    assert body["name"] == "tiny"
    assert body["image_size"] > 0
    assert body["mean"] == [0.485, 0.456, 0.406]
    assert body["std"] == [0.229, 0.224, 0.225]
    assert len(body["version"]) == 16

    file = client.get("/api/v1/ai/models/classifier/file", headers=headers)

    assert file.status_code == 200
    assert file.headers["content-type"] == "application/octet-stream"
    assert file.headers["etag"] == f'"{body["version"]}"'
    assert len(file.content) == body["size_bytes"]
    assert client.get("/api/v1/ai/models/classifier").status_code == 401


def test_classifier_model_missing(
    client: TestClient, headers: dict[str, str], tmp_path: Path
) -> None:
    use_models(client, tmp_path / "absent.onnx", tmp_path / "no.onnx")

    for url in ("/api/v1/ai/models/classifier", "/api/v1/ai/models/classifier/file"):
        assert client.get(url, headers=headers).status_code == 503
