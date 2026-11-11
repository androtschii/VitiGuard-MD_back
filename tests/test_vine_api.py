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


def test_vine_history_saved_listed_and_private(
    api_client: TestClient, headers: dict[str, str], tmp_path: Path
) -> None:
    from memory_storage import MemoryStorage

    storage = MemoryStorage()
    api_client.app.state.storage = storage  # type: ignore[attr-defined]
    use_vine_model(api_client, build_vine_model(tmp_path / "vine.onnx"))
    field = {
        "type": "Polygon",
        "coordinates": [
            [[28.86, 47.14], [28.862, 47.14], [28.862, 47.141], [28.86, 47.14]]
        ],
    }
    vineyard = api_client.post(
        "/api/v1/vineyards",
        headers=headers,
        json={"name": "Криково", "geometry": field},
    ).json()["id"]

    # Без save фото не сохраняется
    code, body = detect(api_client, headers, photo((0, 120, 0), size=(128, 64)))
    assert code == 200
    assert body["id"] is None
    assert storage.files == {}

    response = api_client.post(
        URL,
        headers=headers,
        params={"save": "true", "vineyard_id": vineyard},
        files={"image": ("row.jpg", photo((0, 120, 0), size=(128, 64)), "image/jpeg")},
    )
    saved = response.json()
    assert response.status_code == 200
    assert saved["id"] is not None
    assert len(storage.files) == 1

    history = api_client.get("/api/v1/vine-checks", headers=headers).json()
    assert history["total"] == 1
    item = history["items"][0]
    assert item["vineyard_name"] == "Криково"
    assert (item["trunks"], item["bunches"]) == (1, 1)
    assert item["detections"][0]["box"] == pytest.approx([0.375, 0, 0.625, 1])
    image = api_client.get(f"/api/v1/vine-checks/{saved['id']}/image", headers=headers)
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"

    # Чужой участок при сохранении и чужой анализ — как несуществующие
    CREDENTIALS_2 = {"email": "neighbour@example.md", "password": "secret123"}
    api_client.post(
        "/api/v1/auth/register", json={**CREDENTIALS_2, "full_name": "Сосед"}
    )
    token = api_client.post("/api/v1/auth/login", json=CREDENTIALS_2).json()[
        "access_token"
    ]
    stranger = {"Authorization": f"Bearer {token}"}
    assert api_client.get("/api/v1/vine-checks", headers=stranger).json()["total"] == 0
    url = f"/api/v1/vine-checks/{saved['id']}"
    assert api_client.get(f"{url}/image", headers=stranger).status_code == 404
    assert api_client.delete(url, headers=stranger).status_code == 404
    foreign = api_client.post(
        URL,
        headers=stranger,
        params={"save": "true", "vineyard_id": vineyard},
        files={"image": ("row.jpg", photo((0, 120, 0)), "image/jpeg")},
    )
    assert foreign.status_code == 404
    assert len(storage.files) == 1

    # Удаление убирает и запись, и фото
    assert api_client.delete(url, headers=headers).status_code == 204
    assert storage.files == {}
    assert api_client.get("/api/v1/vine-checks", headers=headers).json()["total"] == 0
    assert api_client.get("/api/v1/vine-checks").status_code == 401
