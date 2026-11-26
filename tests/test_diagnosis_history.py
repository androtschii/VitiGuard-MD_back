import io
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from conftest import set_role
from diagnosis_model import build_model, build_segmenter, gps_photo, photo
from fastapi.testclient import TestClient
from memory_storage import MemoryStorage
from PIL import Image

from app.core.config import Settings
from app.ml.overlays import PNG_DATA_URL
from app.models.diagnosis import DiseaseResult, LeafImage
from app.schemas.diagnosis import DiagnosisRead, DiseaseProbability, LesionRead
from app.services.diagnosis_history import overlay_key, save_diagnosis, storage_key

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
    assert record["overlays"] == []
    assert record["location"] is None

    image = client.get(f"/api/v1/diagnoses/{saved['id']}/image", headers=headers)
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(image.content)).size == (100, 80)


@pytest.mark.integration
def test_history_keeps_severity(
    client: TestClient, tmp_path: Path, storage: MemoryStorage
) -> None:
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
    # Маска сохранена, тепловой карты у тестового классификатора нет
    assert record["overlays"] == ["lesion"]
    url = f"/api/v1/diagnoses/{saved['id']}"
    lesion = client.get(f"{url}/overlays/lesion", headers=headers)
    assert lesion.status_code == 200
    assert lesion.headers["content-type"] == "image/png"
    assert lesion.content.startswith(b"\x89PNG")
    assert client.get(f"{url}/overlays/heatmap", headers=headers).status_code == 404
    assert client.get(f"{url}/overlays/other", headers=headers).status_code == 422
    stranger = login(client, "neighbour@example.md")
    assert client.get(f"{url}/overlays/lesion", headers=stranger).status_code == 404

    assert client.delete(url, headers=headers).status_code == 204
    assert storage.files == {}


@pytest.mark.integration
def test_history_keeps_photo_location(client: TestClient) -> None:
    headers = login(client)

    saved = client.post(
        "/api/v1/ai/diagnose",
        headers=headers,
        params={"save": "true"},
        files={
            "image": ("leaf.jpg", gps_photo((47, 8, 24), (28, 51, 36)), "image/jpeg")
        },
    ).json()

    assert saved["location"] == {"lat": pytest.approx(47.14), "lon": 28.86}
    record = client.get("/api/v1/diagnoses", headers=headers).json()["items"][0]
    assert record["location"] == saved["location"]
    # Участок не подставляется по координатам: его выбирает пользователь
    assert record["vineyard_id"] is None


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


def overlaid_diagnosis() -> DiagnosisRead:
    diagnosis = tiny_diagnosis(LesionRead(mask=PNG_DATA_URL + "bWFzaw==", severity=0.2))
    diagnosis.heatmap = PNG_DATA_URL + "aGVhdA=="
    return diagnosis


async def save(session: RecordingSession, storage: MemoryStorage) -> None:
    await save_diagnosis(
        session,  # type: ignore[arg-type]
        storage,
        uuid.uuid4(),
        None,
        b"jpeg",
        Image.new("RGB", (4, 4)),
        overlaid_diagnosis(),
        "v1",
    )


async def test_overlays_saved_next_to_photo() -> None:
    session, storage = RecordingSession(), MemoryStorage()

    await save(session, storage)

    [result] = [item for item in session.added if isinstance(item, DiseaseResult)]
    [image] = [item for item in session.added if isinstance(item, LeafImage)]
    photo = image.storage_key.removesuffix(".jpg")
    assert result.heatmap_key == f"{photo}.heatmap.png"
    assert result.lesion_key == f"{photo}.lesion.png"
    assert storage.files[result.heatmap_key] == (b"heat", "image/png")
    assert storage.files[result.lesion_key] == (b"mask", "image/png")
    assert len(storage.files) == 3


async def test_failed_commit_removes_overlays() -> None:
    storage = MemoryStorage()

    with pytest.raises(RuntimeError):
        await save(FailingSession(), storage)

    assert storage.files == {}


def test_overlay_key_follows_photo() -> None:
    assert overlay_key("owner/photo.webp", "lesion") == "owner/photo.lesion.png"


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
    client: TestClient, run_sql: RunSql, email: str = "agro@example.md"
) -> dict[str, str]:
    credentials = {**CREDENTIALS, "email": email}
    client.post(
        "/api/v1/auth/register",
        json={**credentials, "full_name": "Мария"},
    )
    set_role(run_sql, email, "agronomist")
    token = client.post("/api/v1/auth/login", json=credentials).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.integration
def test_agronomist_reviews_diagnosis(client: TestClient, run_sql: RunSql) -> None:
    headers = login_agronomist(client, run_sql)
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
def test_review_rules(client: TestClient, run_sql: RunSql) -> None:
    grower = login(client)
    own = diagnose(client, grower, save="true")
    agronomist = login_agronomist(client, run_sql)
    url = f"/api/v1/diagnoses/{own['id']}/review"

    # Простому пользователю заключение недоступно
    assert client.put(url, headers=grower, json={"disease": "esca"}).status_code == 403
    assert (
        client.get("/api/v1/diagnoses/review-queue", headers=grower).status_code == 403
    )
    # Несуществующая проверка
    missing = "/api/v1/diagnoses/00000000-0000-0000-0000-000000000000/review"
    assert (
        client.put(missing, headers=agronomist, json={"disease": "esca"}).status_code
        == 404
    )
    bad = client.put(
        f"/api/v1/diagnoses/{own['id']}/review",
        headers=agronomist,
        json={"disease": "Не болезнь"},
    )
    assert bad.status_code == 422


@pytest.mark.integration
def test_not_a_grape_leaf_is_not_saved(
    client: TestClient, storage: MemoryStorage, tmp_path: Path
) -> None:
    app: Any = client.app
    app.state.settings = app.state.settings.model_copy(
        update={
            "diagnosis_model_path": build_model(tmp_path / "gate.onnx", with_gate=True)
        }
    )
    app.state.classifier = None
    headers = login(client)

    def send(color: tuple[int, int, int]) -> dict[str, Any]:
        response = client.post(
            "/api/v1/ai/diagnose",
            headers=headers,
            params={"save": "true"},
            files={"image": ("leaf.jpg", photo(color), "image/jpeg")},
        )
        body: dict[str, Any] = response.json()
        return body

    # Отсев тестовой модели: «красный» снимок — не лист винограда
    other = send((255, 0, 0))
    leaf = send((0, 200, 0))

    assert other["grape_leaf"] is False
    assert other["id"] is None
    assert leaf["grape_leaf"] is True
    assert leaf["id"] is not None
    assert len(storage.files) == 1
    history = client.get("/api/v1/diagnoses", headers=headers).json()
    assert [item["id"] for item in history["items"]] == [leaf["id"]]


@pytest.mark.integration
def test_agronomist_review_queue(client: TestClient, run_sql: RunSql) -> None:
    grower = login(client)
    first = diagnose(client, grower, save="true")
    second = diagnose(client, grower, save="true")
    agronomist = login_agronomist(client, run_sql)
    queue = "/api/v1/diagnoses/review-queue"

    # Проверки виноградаря попадают в очередь агронома с именем владельца
    waiting = client.get(queue, headers=agronomist).json()
    assert waiting["total"] == 2
    assert {item["owner_name"] for item in waiting["items"]} == {"Ион"}
    # В обычной истории имени владельца нет: это его собственные проверки
    own = client.get("/api/v1/diagnoses", headers=grower).json()["items"]
    assert {item["owner_name"] for item in own} == {None}

    # Агроном видит фото и даёт заключение по чужой проверке
    url = f"/api/v1/diagnoses/{first['id']}"
    assert client.get(f"{url}/image", headers=agronomist).status_code == 200
    reviewed = client.put(
        f"{url}/review",
        headers=agronomist,
        json={"disease": "downy_mildew", "comment": "Пятна снизу листа"},
    ).json()
    assert reviewed["owner_name"] == "Ион"
    assert reviewed["review"]["reviewer"] == "Мария"

    # Проверка с заключением уходит из очереди ожидающих
    assert [i["id"] for i in client.get(queue, headers=agronomist).json()["items"]] == [
        second["id"]
    ]
    done = client.get(queue, headers=agronomist, params={"reviewed": "true"}).json()
    assert [item["id"] for item in done["items"]] == [first["id"]]
    # Виноградарь видит заключение в своей истории
    mine = client.get("/api/v1/diagnoses", headers=grower).json()["items"]
    assert {i["id"]: i["review"] for i in mine}[first["id"]]["disease"] == (
        "downy_mildew"
    )

    # Удалить чужую проверку агроном не может
    assert client.delete(url, headers=agronomist).status_code == 404
    # Заключение агроном может отозвать
    assert client.delete(f"{url}/review", headers=agronomist).status_code == 204
    assert client.get(queue, headers=agronomist).json()["total"] == 2


# Уведомление владельцу фото о заключении агронома


class FakeTelegram:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.sent: list[tuple[int, str]] = []

    async def send_message(self, chat_id: int, text: str, **_: Any) -> None:
        if self.fail:
            raise RuntimeError("Telegram недоступен")
        self.sent.append((chat_id, text))


def notify(
    client: TestClient,
    result_id: str,
    telegram: Any = None,
    send: Any = None,
    vapid: bool = False,
) -> dict[str, int]:
    import asyncio

    from pydantic import SecretStr

    from app.db.session import create_engine, create_session_factory
    from app.services.review_notify import notify_review

    settings: Settings = client.app.state.settings  # type: ignore[attr-defined]
    if vapid:
        settings = settings.model_copy(
            update={"vapid_public_key": "p", "vapid_private_key": SecretStr("k")}
        )

    async def run() -> dict[str, int]:
        engine = create_engine(settings)
        try:
            async with create_session_factory(engine)() as session:
                kwargs = {} if send is None else {"send": send}
                return await notify_review(
                    session, settings, telegram, uuid.UUID(result_id), **kwargs
                )
        finally:
            await engine.dispose()

    return asyncio.run(run())


@pytest.mark.integration
def test_review_of_someone_elses_photo_notifies_owner(
    client: TestClient, run_sql: RunSql
) -> None:
    grower = login(client)
    saved = diagnose(client, grower, save="true")
    agronomist = login_agronomist(client, run_sql)
    notices: list[str] = client.app.state.review_notices  # type: ignore[attr-defined]
    url = f"/api/v1/diagnoses/{saved['id']}/review"

    assert (
        client.put(url, headers=agronomist, json={"disease": "esca"}).status_code == 200
    )
    assert notices == [saved["id"]]
    # Отзыв заключения и проверка своего фото уведомлений не шлют
    assert client.delete(url, headers=agronomist).status_code == 204
    own = diagnose(client, agronomist, save="true")
    own_url = f"/api/v1/diagnoses/{own['id']}/review"
    client.put(own_url, headers=agronomist, json={"disease": "healthy"})
    assert notices == [saved["id"]]


@pytest.mark.integration
def test_review_notice_goes_to_telegram_and_push(
    client: TestClient, run_sql: RunSql
) -> None:
    from app.services.web_push import PushRejected

    grower = login(client)
    saved = diagnose(client, grower, save="true")
    agronomist = login_agronomist(client, run_sql)
    client.put(
        f"/api/v1/diagnoses/{saved['id']}/review",
        headers=agronomist,
        json={"disease": "downy_mildew", "comment": "Пятна снизу листа"},
    )
    run_sql(
        "UPDATE users SET telegram_chat_id = 777 "
        f"WHERE email = '{CREDENTIALS['email']}'"
    )
    for endpoint in ("ok", "gone", "broken"):
        client.post(
            "/api/v1/push/subscriptions",
            headers=grower,
            json={
                "endpoint": f"https://push.example.org/{endpoint}",
                "keys": {"p256dh": "key", "auth": "secret"},
            },
        )
    pushed: list[str] = []

    def send(subscription: Any, payload: str, _: Settings) -> None:
        if subscription.endpoint.endswith("gone"):
            raise PushRejected(gone=True)
        if subscription.endpoint.endswith("broken"):
            raise PushRejected(gone=False)
        pushed.append(payload)

    telegram = FakeTelegram()
    sent = notify(client, saved["id"], telegram, send, vapid=True)

    assert sent == {"telegram": 1, "push": 1}
    chat_id, text = telegram.sent[0]
    assert chat_id == 777
    assert "Агроном проверил ваш снимок" in text
    assert "милдью" in text
    assert "Комментарий: Пятна снизу листа" in text
    assert '"url": "/history"' in pushed[0]
    endpoints = {row[0] for row in run_sql("SELECT endpoint FROM push_subscriptions")}
    assert endpoints == {
        "https://push.example.org/ok",
        "https://push.example.org/broken",
    }

    # Telegram недоступен — push всё равно уходит; без VAPID push пропускается
    failing = FakeTelegram(fail=True)
    assert notify(client, saved["id"], failing, send, vapid=True) == {
        "telegram": 0,
        "push": 1,
    }
    assert notify(client, saved["id"], None, send) == {"telegram": 0, "push": 0}


@pytest.mark.integration
def test_review_notice_skips_missing_withdrawn_and_inactive(
    client: TestClient, run_sql: RunSql
) -> None:
    grower = login(client)
    saved = diagnose(client, grower, save="true")
    agronomist = login_agronomist(client, run_sql)
    telegram = FakeTelegram()
    run_sql(
        "UPDATE users SET telegram_chat_id = 777 "
        f"WHERE email = '{CREDENTIALS['email']}'"
    )
    nothing = {"telegram": 0, "push": 0}

    assert notify(client, str(uuid.uuid4()), telegram) == nothing
    # Заключения нет (не дано или отозвано, пока задача ждала очереди)
    assert notify(client, saved["id"], telegram) == nothing
    client.put(
        f"/api/v1/diagnoses/{saved['id']}/review",
        headers=agronomist,
        json={"disease": "healthy"},
    )
    run_sql(
        f"UPDATE users SET is_active = false WHERE email = '{CREDENTIALS['email']}'"
    )
    assert notify(client, saved["id"], telegram) == nothing
    assert telegram.sent == []


def test_review_task_runs_with_and_without_telegram(
    migrated_database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic import SecretStr

    from app.tasks import review as review_tasks

    seen: list[tuple[bool, uuid.UUID]] = []

    async def fake_notify(
        _session: Any, _settings: Any, telegram: Any, result_id: uuid.UUID
    ) -> dict[str, int]:
        seen.append((telegram is not None, result_id))
        return {"telegram": 0, "push": 0}

    monkeypatch.setattr(review_tasks, "notify_review", fake_notify)
    base = Settings(
        _env_file=None,
        environment="test",
        database_url=migrated_database_url,  # type: ignore[arg-type]
    )
    result_id = uuid.uuid4()
    for token in (SecretStr("123:token"), None):
        settings = base.model_copy(update={"telegram_bot_token": token})
        monkeypatch.setattr(review_tasks, "get_settings", lambda s=settings: s)
        assert review_tasks.notify_review_task(str(result_id)) == {
            "telegram": 0,
            "push": 0,
        }

    assert seen == [(True, result_id), (False, result_id)]


def test_review_notice_is_queued_in_celery(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.api import deps
    from app.tasks.review import NOTIFY_REVIEW
    from app.worker.celery_app import celery_app

    queued: list[tuple[str, tuple[str, ...]]] = []
    monkeypatch.setattr(
        celery_app, "send_task", lambda name, args: queued.append((name, args))
    )

    deps.get_review_notifier()("result-id")

    assert queued == [(NOTIFY_REVIEW, ("result-id",))]


def test_review_text_without_comment() -> None:
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from app.services.review_notify import review_text

    result: Any = SimpleNamespace(
        expert_class="powdery_mildew",
        expert_comment=None,
        image=SimpleNamespace(created_at=datetime(2026, 7, 3, 22, 30, tzinfo=UTC)),
    )

    # 22:30 UTC — уже 4 июля в Кишинёве
    assert review_text(result) == "Агроном проверил ваш снимок от 04.07: оидиум."
