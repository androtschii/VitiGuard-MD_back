import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx2 as httpx
import pytest
from diagnosis_model import build_model, build_segmenter, photo
from fastapi.testclient import TestClient
from memory_storage import MemoryStorage
from pydantic import SecretStr

from app.api.deps import get_task_queue, get_telegram
from app.core.config import Settings
from app.core.errors import ExternalServiceError
from app.integrations.telegram import TelegramClient
from app.services.telegram_bot import DIAGNOSE_PHOTO, diagnosis_text
from app.tasks.telegram import diagnose_photo
from app.worker.celery_app import TASK_MODULES

TOKEN = "123456:secret-bot-token"
SECRET = "webhook-secret"
BOT = Settings(
    _env_file=None,
    telegram_bot_token=SecretStr(TOKEN),
    telegram_webhook_secret=SecretStr(SECRET),
)


def telegram_with(handler: Any) -> TelegramClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TelegramClient(http, BOT)


async def test_client_sends_messages_with_bot_token() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "result": {}})

    await telegram_with(handler).send_message(42, "Привет")

    assert str(requests[0].url) == f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    assert json.loads(requests[0].content) == {"chat_id": 42, "text": "Привет"}


async def test_client_downloads_photo() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/getFile"):
            result = {"file_path": "photos/leaf.jpg", "file_size": 4}
            return httpx.Response(200, json={"ok": True, "result": result})
        assert request.url.path == f"/file/bot{TOKEN}/photos/leaf.jpg"
        return httpx.Response(200, content=b"leaf")

    assert await telegram_with(handler).download_file("file-1", 10) == b"leaf"
    # Слишком большой файл не скачивается вовсе или отбрасывается после
    assert await telegram_with(handler).download_file("file-1", 3) is None


async def test_client_drops_file_larger_than_promised() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/getFile"):
            result = {"file_path": "photos/leaf.jpg"}
            return httpx.Response(200, json={"ok": True, "result": result})
        return httpx.Response(200, content=b"x" * 20)

    assert await telegram_with(handler).download_file("file-1", 10) is None


@pytest.mark.parametrize(
    "handler",
    [
        lambda request: httpx.Response(502),
        lambda request: httpx.Response(200, json={"ok": False}),
        lambda request: (
            httpx.Response(200, json={"ok": True, "result": {"file_path": "a"}})
            if request.url.path.endswith("/getFile")
            else httpx.Response(404)
        ),
    ],
    ids=["telegram-down", "bad-answer", "file-gone"],
)
async def test_client_errors_do_not_leak_token(handler: Any) -> None:
    with pytest.raises(ExternalServiceError) as caught:
        await telegram_with(handler).download_file("file-1", 10)

    # В цепочке исключения был бы адрес запроса с токеном
    assert caught.value.__cause__ is None or TOKEN not in str(caught.value.__cause__)
    assert TOKEN not in str(caught.value)


def test_client_needs_token() -> None:
    with pytest.raises(ValueError, match="токен"):
        TelegramClient(httpx.AsyncClient(), Settings(_env_file=None))


def test_diagnosis_text() -> None:
    assert diagnosis_text("downy_mildew", 0.914, False) == (
        "Диагноз: милдью (уверенность 91 %).\n"
        "Проверка сохранена в истории VitiGuard MD."
    )
    assert "Модель не уверена" in diagnosis_text("esca", 0.4, True)


def test_task_is_registered_for_worker() -> None:
    assert "app.tasks.telegram" in TASK_MODULES


class FakeTelegram:
    def __init__(self, data: bytes | None = None, fail: bool = False) -> None:
        self.sent: list[tuple[int, str]] = []
        self.data = data
        self.fail = fail

    async def send_message(self, chat_id: int, text: str) -> None:
        if self.fail:
            raise ExternalServiceError("Telegram недоступен")
        self.sent.append((chat_id, text))

    async def download_file(self, file_id: str, max_bytes: int) -> bytes | None:
        return self.data


def model_settings(tmp_path: Path, **values: Any) -> Settings:
    return Settings(
        _env_file=None,
        diagnosis_model_path=build_model(tmp_path / "tiny.onnx"),
        lesion_model_path=build_segmenter(tmp_path / "unet.onnx"),
        **values,
    )


@pytest.mark.parametrize(
    ("data", "answer"),
    [(None, "больше 10 МБ"), (b"not an image", "нужен JPEG, PNG или WebP")],
)
async def test_task_explains_bad_photo(
    tmp_path: Path, data: bytes | None, answer: str
) -> None:
    telegram = FakeTelegram(data)

    reply = await diagnose_photo(
        model_settings(tmp_path),
        MemoryStorage(),
        telegram,  # type: ignore[arg-type]
        42,
        "u",
        "file-1",
    )

    assert answer in reply
    assert telegram.sent == [(42, reply)]


async def test_task_without_model_says_so(tmp_path: Path) -> None:
    telegram = FakeTelegram(photo((220, 0, 0)))
    settings = Settings(_env_file=None, diagnosis_model_path=tmp_path / "none.onnx")

    reply = await diagnose_photo(
        settings, MemoryStorage(), telegram, 42, "u", "f"  # type: ignore[arg-type]
    )

    assert "модель не загружена" in reply


CREDENTIALS = {"email": "grower@example.md", "password": "secret123"}


def login(client: TestClient, email: str = CREDENTIALS["email"]) -> dict[str, str]:
    credentials = {**CREDENTIALS, "email": email}
    client.post("/api/v1/auth/register", json={**credentials, "full_name": "Ион"})
    token = client.post("/api/v1/auth/login", json=credentials).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def bot(api_client: TestClient) -> Iterator[dict[str, Any]]:
    """Приложение с настроенным ботом: ответы и задачи копятся в списках."""
    app: Any = api_client.app
    app.state.settings = app.state.settings.model_copy(
        update={
            "telegram_bot_token": SecretStr(TOKEN),
            "telegram_webhook_secret": SecretStr(SECRET),
        }
    )
    telegram = FakeTelegram()
    queued: list[tuple[str, list[Any]]] = []
    app.dependency_overrides[get_telegram] = lambda: telegram
    app.dependency_overrides[get_task_queue] = lambda: (
        lambda name, args: queued.append((name, args))
    )
    yield {"client": api_client, "telegram": telegram, "queued": queued}
    app.dependency_overrides.clear()


def send(client: TestClient, message: dict[str, Any], secret: str = SECRET) -> int:
    response = client.post(
        "/api/v1/telegram/webhook",
        headers={"X-Telegram-Bot-Api-Secret-Token": secret},
        json={"update_id": 1, "message": {"chat": {"id": 42}, **message}},
    )
    return response.status_code


@pytest.mark.integration
def test_chat_is_linked_by_one_time_link(bot: dict[str, Any]) -> None:
    client, telegram = bot["client"], bot["telegram"]
    headers = login(client)

    link = client.post("/api/v1/users/me/telegram", headers=headers).json()
    code = link["url"].removeprefix("https://t.me/VitiGuardMDBot?start=")

    assert link["expires_in"] == 900
    assert send(client, {"text": f"/start {code}"}) == 200
    assert telegram.sent[-1][1].startswith("Готово: чат привязан к grower@example.md")
    assert client.get("/api/v1/users/me", headers=headers).json()["telegram_linked"]
    # Второй раз та же ссылка не срабатывает
    send(client, {"text": f"/start {code}"})
    assert "устарела" in telegram.sent[-1][1]

    assert (
        client.delete("/api/v1/users/me/telegram", headers=headers).status_code == 204
    )
    me = client.get("/api/v1/users/me", headers=headers).json()
    assert me["telegram_linked"] is False
    assert "telegram_chat_id" not in me


@pytest.mark.integration
def test_chat_moves_to_the_last_linked_account(bot: dict[str, Any]) -> None:
    client = bot["client"]
    first, second = login(client), login(client, "neighbour@example.md")
    for headers in (first, second):
        url = client.post("/api/v1/users/me/telegram", headers=headers).json()["url"]
        send(client, {"text": f"/start {url.rsplit('=', 1)[1]}"})

    assert not client.get("/api/v1/users/me", headers=first).json()["telegram_linked"]
    assert client.get("/api/v1/users/me", headers=second).json()["telegram_linked"]


@pytest.mark.integration
def test_photo_from_linked_chat_goes_to_diagnosis(bot: dict[str, Any]) -> None:
    client, telegram, queued = bot["client"], bot["telegram"], bot["queued"]
    headers = login(client)
    owner = client.get("/api/v1/users/me", headers=headers).json()["id"]
    url = client.post("/api/v1/users/me/telegram", headers=headers).json()["url"]
    send(client, {"text": f"/start {url.rsplit('=', 1)[1]}"})

    sizes = [
        {"file_id": "small", "width": 90, "height": 90},
        {"file_id": "large", "width": 1280, "height": 1280},
    ]
    assert send(client, {"photo": sizes}) == 200
    send(client, {"document": {"file_id": "raw", "mime_type": "image/jpeg"}})
    send(client, {"document": {"file_id": "pdf", "mime_type": "application/pdf"}})
    send(client, {"text": "привет"})

    assert queued == [
        (DIAGNOSE_PHOTO, [42, owner, "large"]),
        (DIAGNOSE_PHOTO, [42, owner, "raw"]),
    ]
    replies = [text for _, text in telegram.sent[1:]]
    assert replies[:2] == ["Фото получено, анализирую…"] * 2
    assert replies[2].startswith("Пришлите фото листа")
    assert replies[3].startswith("Пришлите фото листа")


@pytest.mark.integration
def test_unlinked_chat_is_told_how_to_link(bot: dict[str, Any]) -> None:
    client, telegram, queued = bot["client"], bot["telegram"], bot["queued"]

    send(client, {"photo": [{"file_id": "f", "width": 10, "height": 10}]})
    send(client, {"text": "/start"})

    assert queued == []
    assert "не привязан" in telegram.sent[0][1]
    assert telegram.sent[1][1].startswith("Пришлите фото листа")


@pytest.mark.integration
def test_webhook_is_protected(bot: dict[str, Any]) -> None:
    client = bot["client"]

    assert send(client, {"text": "/start"}, secret="guess") == 403
    # Обновление без сообщения (например, правка) просто принимается
    response = client.post(
        "/api/v1/telegram/webhook",
        headers={"X-Telegram-Bot-Api-Secret-Token": SECRET},
        json={"update_id": 2},
    )
    assert response.status_code == 200


@pytest.mark.integration
def test_telegram_failure_does_not_make_it_retry(bot: dict[str, Any]) -> None:
    bot["telegram"].fail = True

    assert send(bot["client"], {"text": "/start"}) == 200


@pytest.mark.integration
def test_without_bot_everything_is_off(api_client: TestClient) -> None:
    headers = login(api_client)

    assert (
        api_client.post("/api/v1/users/me/telegram", headers=headers).status_code == 503
    )
    assert send(api_client, {"text": "/start"}) == 404


@pytest.mark.integration
def test_task_saves_photo_to_history(
    api_client: TestClient, migrated_database_url: str, tmp_path: Path
) -> None:
    headers = login(api_client)
    owner = api_client.get("/api/v1/users/me", headers=headers).json()["id"]
    telegram = FakeTelegram(photo((220, 0, 0)))
    storage = MemoryStorage()

    reply = asyncio.run(
        diagnose_photo(
            model_settings(tmp_path, database_url=migrated_database_url),
            storage,
            telegram,  # type: ignore[arg-type]
            42,
            owner,
            "file-1",
        )
    )

    assert reply.startswith("Диагноз: здоровый лист")
    history = api_client.get("/api/v1/diagnoses", headers=headers).json()
    assert history["total"] == 1
    assert history["items"][0]["vineyard_id"] is None
    assert telegram.sent == [(42, reply)]


@pytest.mark.integration
def test_link_of_blocked_account_does_not_work(
    bot: dict[str, Any], run_sql: Any
) -> None:
    client, telegram = bot["client"], bot["telegram"]
    url = client.post("/api/v1/users/me/telegram", headers=login(client)).json()["url"]
    run_sql("UPDATE users SET is_active = false")

    send(client, {"text": f"/start {url.rsplit('=', 1)[1]}"})

    assert "устарела" in telegram.sent[-1][1]
