from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from geoalchemy2 import WKTElement
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_task_queue, get_telegram
from app.core.config import Settings
from app.core.errors import ExternalServiceError
from app.models import Alert, DiseaseRisk, User, Vineyard
from app.models.enums import Disease, RiskLevel
from app.services.alerts import raise_alerts
from app.services.telegram_bot import alerts_text, send_alerts
from app.tasks import risk as risk_tasks

pytestmark = pytest.mark.integration

RunSql = Callable[[str], list[tuple[object, ...]]]
TODAY = date(2026, 6, 10)
PLOT = "MULTIPOLYGON(((28.86 47.14, 28.87 47.14, 28.87 47.15, 28.86 47.14)))"
SECRET = "webhook-secret"


class FakeTelegram:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.sent: list[tuple[int, str, dict[str, Any] | None]] = []
        self.answers: list[str] = []
        self.edited: list[tuple[int, int, dict[str, Any]]] = []

    async def send_message(
        self, chat_id: int, text: str, reply_markup: dict[str, Any] | None = None
    ) -> None:
        if self.fail:
            raise ExternalServiceError("Telegram недоступен")
        self.sent.append((chat_id, text, reply_markup))

    async def answer_callback(self, callback_id: str, text: str) -> None:
        self.answers.append(text)

    async def edit_reply_markup(
        self, chat_id: int, message_id: int, reply_markup: dict[str, Any]
    ) -> None:
        self.edited.append((chat_id, message_id, reply_markup))


def test_alerts_text() -> None:
    text = alerts_text(
        [
            ("Криково", Disease.DOWNY_MILDEW, TODAY, RiskLevel.HIGH),
            ("Криково", Disease.POWDERY_MILDEW, date(2026, 6, 11), RiskLevel.MEDIUM),
            ("Пуркарь", Disease.DOWNY_MILDEW, date(2026, 6, 12), RiskLevel.HIGH),
        ],
        TODAY,
    )

    assert text.splitlines() == [
        "Предупреждения VitiGuard MD:",
        "• Криково: высокий риск милдью сегодня",
        "• Криково: средний риск оидиума завтра",
        "• Пуркарь: высокий риск милдью 12.06",
        "Что делать — в окне «Риск по дням» участка.",
    ]


def alert(
    vineyard: Vineyard,
    day: date,
    disease: Disease = Disease.DOWNY_MILDEW,
    **values: Any,
) -> Alert:
    return Alert(
        user=vineyard.owner,
        vineyard=vineyard,
        disease=disease,
        day=day,
        level=RiskLevel.HIGH,
        **values,
    )


async def test_new_alerts_go_to_owner_chat_once(session: AsyncSession) -> None:
    owner = User(email="grower@example.md", hashed_password="hash", telegram_chat_id=42)
    plot = Vineyard(owner=owner, name="Криково", geom=WKTElement(PLOT, srid=4326))
    muted = Vineyard(
        owner=owner,
        name="Без уведомлений",
        geom=WKTElement(PLOT, srid=4326),
        telegram_alerts=False,
    )
    stranger = User(email="other@example.md", hashed_password="hash")
    unlinked = Vineyard(owner=stranger, name="Чужой", geom=WKTElement(PLOT, srid=4326))
    read = datetime(2026, 6, 10, 8, tzinfo=UTC)
    session.add_all(
        [
            alert(plot, TODAY),
            alert(plot, TODAY + timedelta(days=1), Disease.POWDERY_MILDEW),
            # Прошлое, прочитанное, отправленное, выключенный участок, чат не привязан
            alert(plot, TODAY - timedelta(days=1)),
            alert(plot, TODAY + timedelta(days=2), read_at=read),
            alert(
                plot,
                TODAY + timedelta(days=2),
                Disease.POWDERY_MILDEW,
                notified_at=read,
            ),
            alert(muted, TODAY),
            alert(unlinked, TODAY),
        ]
    )
    await session.commit()
    telegram = FakeTelegram()

    assert await send_alerts(session, telegram, TODAY) == 1  # type: ignore[arg-type]

    [(chat_id, text, _)] = telegram.sent
    assert chat_id == 42
    assert text.splitlines()[1:3] == [
        "• Криково: высокий риск милдью сегодня",
        "• Криково: высокий риск оидиума завтра",
    ]
    assert await send_alerts(session, telegram, TODAY) == 0  # type: ignore[arg-type]
    assert len(telegram.sent) == 1


async def test_failed_send_is_retried_next_time(session: AsyncSession) -> None:
    owner = User(email="grower@example.md", hashed_password="hash", telegram_chat_id=42)
    plot = Vineyard(owner=owner, name="Криково", geom=WKTElement(PLOT, srid=4326))
    session.add(alert(plot, TODAY))
    await session.commit()

    assert await send_alerts(session, FakeTelegram(fail=True), TODAY) == 0  # type: ignore[arg-type]
    assert await send_alerts(session, FakeTelegram(), TODAY) == 1  # type: ignore[arg-type]


async def test_raised_level_is_sent_again(session: AsyncSession) -> None:
    owner = User(email="grower@example.md", hashed_password="hash")
    plot = Vineyard(owner=owner, name="Криково", geom=WKTElement(PLOT, srid=4326))
    sent = datetime(2026, 6, 10, 6, tzinfo=UTC)
    session.add_all(
        [
            Alert(
                user=owner,
                vineyard=plot,
                disease=Disease.DOWNY_MILDEW,
                day=TODAY,
                level=RiskLevel.MEDIUM,
                notified_at=sent,
            ),
            DiseaseRisk(
                vineyard=plot,
                disease=Disease.DOWNY_MILDEW,
                target_date=TODAY,
                method="three_tens",
                score=1,
                level=RiskLevel.HIGH,
            ),
        ]
    )
    await session.flush()

    await raise_alerts(session, plot, TODAY, RiskLevel.MEDIUM)

    stored = await session.scalar(select(Alert))
    assert stored is not None
    await session.refresh(stored)
    assert (stored.level, stored.notified_at) == (RiskLevel.HIGH, None)


def test_refresh_all_sends_alerts_when_bot_is_set(
    migrated_database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[date] = []

    async def fake_send(_session: Any, _telegram: Any, today: date) -> int:
        calls.append(today)
        return 3

    monkeypatch.setattr(risk_tasks, "send_alerts", fake_send)
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url=migrated_database_url,  # type: ignore[arg-type]
        telegram_bot_token=SecretStr("123:token"),
    )
    monkeypatch.setattr(risk_tasks, "get_settings", lambda: settings)

    assert risk_tasks.refresh_all() == {"refreshed": 0, "failed": 0, "notified": 3}
    assert len(calls) == 1


CREDENTIALS = {"email": "grower@example.md", "password": "secret123"}


def login(client: TestClient, email: str = CREDENTIALS["email"]) -> dict[str, str]:
    credentials = {**CREDENTIALS, "email": email}
    client.post("/api/v1/auth/register", json={**credentials, "full_name": "Ион"})
    token = client.post("/api/v1/auth/login", json=credentials).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def create_vineyard(
    client: TestClient, headers: dict[str, str], name: str, west: float = 28.86
) -> str:
    ring = [[west, 47.14], [west + 0.002, 47.14], [west + 0.002, 47.141], [west, 47.14]]
    vineyard_id: str = client.post(
        "/api/v1/vineyards",
        headers=headers,
        json={"name": name, "geometry": {"type": "Polygon", "coordinates": [ring]}},
    ).json()["id"]
    return vineyard_id


@pytest.fixture
def bot(api_client: TestClient) -> Iterator[tuple[TestClient, FakeTelegram]]:
    app: Any = api_client.app
    app.state.settings = app.state.settings.model_copy(
        update={
            "telegram_bot_token": SecretStr("123:token"),
            "telegram_webhook_secret": SecretStr(SECRET),
        }
    )
    telegram = FakeTelegram()
    app.dependency_overrides[get_telegram] = lambda: telegram
    app.dependency_overrides[get_task_queue] = lambda: lambda name, args: None
    yield api_client, telegram
    app.dependency_overrides.clear()


def link(client: TestClient, headers: dict[str, str], chat_id: int = 42) -> None:
    url = client.post("/api/v1/users/me/telegram", headers=headers).json()["url"]
    update(
        client, {"message": {"chat": {"id": chat_id}, "text": f"/start {url[-22:]}"}}
    )


def update(client: TestClient, body: dict[str, Any]) -> None:
    response = client.post(
        "/api/v1/telegram/webhook",
        headers={"X-Telegram-Bot-Api-Secret-Token": SECRET},
        json={"update_id": 1, **body},
    )
    assert response.status_code == 200


def press(client: TestClient, data: str | None, chat_id: int = 42) -> None:
    query: dict[str, Any] = {
        "id": "query-1",
        "message": {"message_id": 7, "chat": {"id": chat_id}},
    }
    if data is not None:
        query["data"] = data
    update(client, {"callback_query": query})


def test_alerts_command_toggles_vineyards(
    bot: tuple[TestClient, FakeTelegram], run_sql: RunSql
) -> None:
    client, telegram = bot
    headers = login(client)
    link(client, headers)
    update(client, {"message": {"chat": {"id": 42}, "text": "/alerts"}})
    assert telegram.sent[-1][1].startswith("Участков пока нет")

    plot = create_vineyard(client, headers, "Криково")
    other = create_vineyard(client, headers, "Пуркарь", west=28.9)
    update(client, {"message": {"chat": {"id": 42}, "text": "/alerts"}})

    assert telegram.sent[-1][2] == {
        "inline_keyboard": [
            [{"text": "🔔 Криково", "callback_data": f"alerts:{plot}"}],
            [{"text": "🔔 Пуркарь", "callback_data": f"alerts:{other}"}],
        ]
    }

    press(client, f"alerts:{plot}")

    assert telegram.answers == ["Криково: предупреждения выключены"]
    assert telegram.edited[-1][:2] == (42, 7)
    assert telegram.edited[-1][2]["inline_keyboard"][0][0]["text"] == "🔕 Криково"
    assert run_sql(f"SELECT telegram_alerts FROM vineyards WHERE id = '{plot}'") == [
        (False,)
    ]
    press(client, f"alerts:{plot}")
    assert telegram.answers[-1] == "Криково: предупреждения включены"


def test_foreign_or_broken_buttons_are_refused(
    bot: tuple[TestClient, FakeTelegram],
) -> None:
    client, telegram = bot
    owner = login(client)
    plot = create_vineyard(client, owner, "Криково")
    link(client, login(client, "neighbour@example.md"), chat_id=77)

    press(client, f"alerts:{plot}", chat_id=77)
    press(client, "alerts:not-a-uuid", chat_id=77)
    press(client, "other:1", chat_id=77)
    press(client, None, chat_id=77)
    update(client, {"callback_query": {"id": "query-2", "data": f"alerts:{plot}"}})

    assert telegram.answers == [
        "Участок не найден",
        "Участок не найден",
        "Кнопка устарела",
        "Кнопка устарела",
        "Кнопка устарела",
    ]
    assert telegram.edited == []
