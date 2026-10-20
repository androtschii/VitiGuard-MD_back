import io
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from diagnosis_model import build_model, build_segmenter, photo
from fastapi.testclient import TestClient
from memory_storage import MemoryStorage
from PIL import Image

from app.core.config import Settings
from app.models.diagnosis import DiseaseResult
from app.schemas.diagnosis import DiagnosisRead, DiseaseProbability, LesionRead
from app.services.diagnosis_history import save_diagnosis, storage_key

RunSql = Callable[[str], list[tuple[object, ...]]]
CREDENTIALS = {"email": "grower@example.md", "password": "secret123"}
SQUARE: dict[str, Any] = {
    "type": "Polygon",
    "coordinates": [
        [[28.86, 47.14], [28.862, 47.14], [28.862, 47.141], [28.86, 47.14]]
    ],
}


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
        update={
            "diagnosis_model_path": build_model(tmp_path / "tiny.onnx"),
            "lesion_model_path": tmp_path / "no.onnx",
        }
    )
    app.state.classifier = None
    app.state.segmenter = None
    app.state.storage = storage
    return api_client


def diagnose(
    client: TestClient, headers: dict[str, str], **params: str
) -> dict[str, Any]:
    response = client.post(
        "/api/v1/ai/diagnose",
        headers=headers,
        params=params,
        files={"image": ("leaf.jpg", photo((255, 0, 0)), "image/jpeg")},
    )
    body: dict[str, Any] = {"status": response.status_code, **response.json()}
    return body


def create_vineyard(client: TestClient, headers: dict[str, str]) -> str:
    response = client.post(
        "/api/v1/vineyards",
        headers=headers,
        json={"name": "Криково", "geometry": SQUARE},
    )
    vineyard_id: str = response.json()["id"]
    return vineyard_id


@pytest.mark.integration
def test_saved_diagnosis_in_history(client: TestClient, storage: MemoryStorage) -> None:
    headers = login(client)
    vineyard = create_vineyard(client, headers)

    # Без save фото никуда не сохраняется
    assert diagnose(client, headers)["id"] is None
    saved = diagnose(client, headers, save="true", vineyard_id=vineyard)

    assert saved["status"] == 200
    assert len(storage.files) == 1
    history = client.get("/api/v1/diagnoses", headers=headers).json()
    assert history["total"] == 1
    record = history["items"][0]
    assert record["id"] == saved["id"]
    assert record["vineyard_name"] == "Криково"
    assert record["disease"] == "healthy"
    assert record["model"] == "tiny"
    assert len(record["model_version"]) == 16
    assert record["top"][0]["disease"] == "healthy"
    assert record["severity"] is None

    image = client.get(f"/api/v1/diagnoses/{saved['id']}/image", headers=headers)
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(image.content)).size == (100, 80)


@pytest.mark.integration
def test_history_keeps_severity(client: TestClient, tmp_path: Path) -> None:
    app: Any = client.app
    settings: Settings = app.state.settings
    app.state.settings = settings.model_copy(
        update={"lesion_model_path": build_segmenter(tmp_path / "unet.onnx")}
    )
    headers = login(client)

    saved = diagnose(client, headers, save="true")

    record = client.get("/api/v1/diagnoses", headers=headers).json()["items"][0]
    assert record["severity"] == pytest.approx(saved["lesion"]["severity"])
    assert record["severity"] == pytest.approx(1)


@pytest.mark.integration
def test_history_filters(client: TestClient) -> None:
    headers = login(client)
    vineyard = create_vineyard(client, headers)
    diagnose(client, headers, save="true", vineyard_id=vineyard)
    diagnose(client, headers, save="true")

    def total(**params: str) -> int:
        response = client.get("/api/v1/diagnoses", headers=headers, params=params)
        count: int = response.json()["total"]
        return count

    assert total() == 2
    assert total(vineyard_id=vineyard) == 1
    assert total(disease="healthy") == 2
    assert total(disease="esca") == 0
    assert total(date_from="2000-01-01", date_to="2999-12-31") == 2
    assert total(date_from="2999-01-01") == 0
    assert total(date_to="2000-01-01") == 0


@pytest.mark.integration
def test_history_is_private_and_deletable(
    client: TestClient, storage: MemoryStorage
) -> None:
    headers = login(client)
    saved = diagnose(client, headers, save="true")
    stranger = login(client, "neighbour@example.md")

    assert client.get("/api/v1/diagnoses", headers=stranger).json()["total"] == 0
    url = f"/api/v1/diagnoses/{saved['id']}"
    assert client.get(f"{url}/image", headers=stranger).status_code == 404
    assert client.delete(url, headers=stranger).status_code == 404
    # Чужой участок при сохранении — как несуществующий, фото не сохраняется
    foreign = create_vineyard(client, stranger)
    assert diagnose(client, headers, save="true", vineyard_id=foreign)["status"] == 404
    assert len(storage.files) == 1

    assert client.delete(url, headers=headers).status_code == 204

    assert storage.files == {}
    assert client.get("/api/v1/diagnoses", headers=headers).json()["total"] == 0
    assert client.get("/api/v1/diagnoses").status_code == 401


@pytest.mark.integration
def test_lost_photo_is_not_found(client: TestClient, storage: MemoryStorage) -> None:
    headers = login(client)
    saved = diagnose(client, headers, save="true")
    storage.files.clear()

    response = client.get(f"/api/v1/diagnoses/{saved['id']}/image", headers=headers)

    assert response.status_code == 404


def test_storage_key_follows_image_format() -> None:
    owner, image = uuid.uuid4(), uuid.uuid4()
    png = Image.new("RGB", (2, 2))
    png.format = "PNG"

    assert storage_key(owner, image, png) == f"{owner}/{image}.png"
    # Формат неизвестен (картинка создана в памяти) — JPEG
    assert storage_key(owner, image, Image.new("RGB", (2, 2))).endswith(".jpg")


class RecordingSession:
    """Сессия без базы: запоминает добавленные объекты."""

    def __init__(self) -> None:
        self.added: list[Any] = []

    def add(self, instance: Any) -> None:
        if getattr(instance, "id", None) is None:
            instance.id = uuid.uuid4()
        self.added.append(instance)

    async def flush(self) -> None:
        return None

    async def commit(self) -> None:
        return None


class FailingSession(RecordingSession):
    """Сессия, у которой не прошёл commit: фото не должно остаться в хранилище."""

    async def commit(self) -> None:
        raise RuntimeError("database is down")


def tiny_diagnosis(lesion: LesionRead | None = None) -> DiagnosisRead:
    return DiagnosisRead(
        model="tiny",
        disease="healthy",
        confidence=1,
        low_confidence=False,
        top=[DiseaseProbability(disease="healthy", probability=1)],
        heatmap=None,
        lesion=lesion,
    )


@pytest.mark.parametrize("severity", [0.27, None])
async def test_severity_saved_with_result(severity: float | None) -> None:
    session = RecordingSession()
    lesion = None if severity is None else LesionRead(mask="", severity=severity)

    await save_diagnosis(
        session,  # type: ignore[arg-type]
        MemoryStorage(),
        uuid.uuid4(),
        None,
        b"jpeg",
        Image.new("RGB", (4, 4)),
        tiny_diagnosis(lesion),
        "v1",
    )

    [result] = [item for item in session.added if isinstance(item, DiseaseResult)]
    assert result.severity == severity


async def test_failed_commit_removes_photo() -> None:
    storage = MemoryStorage()
    leaf = Image.new("RGB", (4, 4))
    diagnosis = tiny_diagnosis()

    with pytest.raises(RuntimeError):
        await save_diagnosis(
            FailingSession(),  # type: ignore[arg-type]
            storage,
            uuid.uuid4(),
            None,
            b"jpeg",
            leaf,
            diagnosis,
            "v1",
        )

    assert storage.files == {}


def login_agronomist(
    client: TestClient, email: str = "agro@example.md"
) -> dict[str, str]:
    credentials = {**CREDENTIALS, "email": email}
    client.post(
        "/api/v1/auth/register",
        json={**credentials, "full_name": "Мария", "role": "agronomist"},
    )
    token = client.post("/api/v1/auth/login", json=credentials).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.integration
def test_agronomist_reviews_diagnosis(client: TestClient, run_sql: RunSql) -> None:
    headers = login_agronomist(client)
    saved = diagnose(client, headers, save="true")
    url = f"/api/v1/diagnoses/{saved['id']}/review"

    confirmed = client.put(url, headers=headers, json={"disease": "healthy"})

    assert confirmed.status_code == 200
    review = confirmed.json()["review"]
    assert review["disease"] == "healthy"
    assert review["comment"] is None
    assert review["reviewer"] == "Мария"

    corrected = client.put(
        url,
        headers=headers,
        json={"disease": "downy_mildew", "comment": "Пятна снизу листа"},
    ).json()["review"]
    assert corrected["disease"] == "downy_mildew"
    assert corrected["comment"] == "Пятна снизу листа"
    listed = client.get("/api/v1/diagnoses", headers=headers).json()["items"][0]
    assert listed["review"]["disease"] == "downy_mildew"
    # Модель сказала своё: заключение не меняет её ответ
    assert listed["disease"] == "healthy"

    # Учётная запись эксперта удалена — заключение остаётся без имени
    run_sql(
        f"UPDATE disease_results SET reviewed_by_id = NULL WHERE id = '{saved['id']}'"
    )
    listed = client.get("/api/v1/diagnoses", headers=headers).json()["items"][0]
    assert listed["review"]["reviewer"] is None

    assert client.delete(url, headers=headers).status_code == 204
    listed = client.get("/api/v1/diagnoses", headers=headers).json()["items"][0]
    assert listed["review"] is None


@pytest.mark.integration
def test_review_rules(client: TestClient) -> None:
    grower = login(client)
    own = diagnose(client, grower, save="true")
    agronomist = login_agronomist(client)
    url = f"/api/v1/diagnoses/{own['id']}/review"

    # Простому пользователю заключение недоступно
    assert client.put(url, headers=grower, json={"disease": "esca"}).status_code == 403
    # Чужую проверку агроном не видит
    assert (
        client.put(url, headers=agronomist, json={"disease": "esca"}).status_code == 404
    )
    assert client.delete(url, headers=agronomist).status_code == 404
    mine = diagnose(client, agronomist, save="true")
    bad = client.put(
        f"/api/v1/diagnoses/{mine['id']}/review",
        headers=agronomist,
        json={"disease": "Не болезнь"},
    )
    assert bad.status_code == 422
