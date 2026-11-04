import uuid
from datetime import UTC, date, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from geoalchemy2 import WKTElement
from sqlalchemy.ext.asyncio import AsyncSession
from test_telegram_alerts import FakeTelegram, bot, create_vineyard, link, login, update

from app.models import Disease, DiseaseRisk, RiskLevel, User, Vineyard
from app.models.diagnosis import DiseaseResult, LeafImage
from app.models.enums import Fungicide, ProcessingStatus
from app.services import farm_bot
from app.services.farm_bot import (
    CheckRow,
    disease_reply,
    farm_reply,
    farm_topic,
    history_text,
    plan_text,
    risk_text,
)
from app.services.risk import TodayLevels
from app.services.telegram_bot import NOT_LINKED
from app.services.treatments import Action, Plan, Recommendation

__all__ = ["bot"]

TODAY = date(2026, 6, 10)
PLOT = "MULTIPOLYGON(((28.86 47.14, 28.87 47.14, 28.87 47.15, 28.86 47.14)))"


def test_farm_topic_understands_own_words() -> None:
    assert farm_topic("Где сегодня опасно?") == "/risk"
    assert farm_topic("Какой риск на участках") == "/risk"
    assert farm_topic("Когда обрабатывать от милдью?") == "/plan"
    assert farm_topic("покажи план") == "/plan"
    assert farm_topic("Мои последние проверки") == "/history"
    assert farm_topic("Какое вино к рыбе?") is None


def test_disease_reply_by_name() -> None:
    answer = disease_reply("Что такое оидиум?")

    assert answer is not None
    assert answer.startswith("Оидиум")
    assert "Признаки:" in answer
    assert "Что делать:" in answer
    # «Ложная мучнистая роса» — это милдью, а не оидиум
    assert (disease_reply("ложная мучнистая роса") or "").startswith("Милдью")
    assert (disease_reply("черная гниль на ягодах") or "").startswith("Чёрная")
    assert disease_reply("Привет") is None


def test_risk_text_lists_every_vineyard() -> None:
    owner = User(email="grower@example.md", hashed_password="hash")
    plot = Vineyard(owner=owner, name="Криково", geom=WKTElement(PLOT, srid=4326))
    other = Vineyard(owner=owner, name="Пуркарь", geom=WKTElement(PLOT, srid=4326))
    plot.id, other.id = uuid.uuid4(), uuid.uuid4()

    text = risk_text(
        [plot, other],
        [TodayLevels(plot.id, TODAY, RiskLevel.HIGH, RiskLevel.MEDIUM)],
        TODAY,
    )

    assert text.splitlines() == [
        "Риск болезней на сегодня, 10.06:",
        "• Криково: милдью 🔴 высокий, оидиум 🟡 средний",
        "• Пуркарь: оценки на сегодня нет",
        "Оценки нет вне сезона или если риск участка ещё не пересчитан.",
    ]
    assert risk_text([], [], TODAY).startswith("Участков пока нет")


def test_plan_text_explains_each_step() -> None:
    plan = Plan(
        today=TODAY,
        protected_until={
            Disease.DOWNY_MILDEW: None,
            Disease.POWDERY_MILDEW: date(2026, 6, 14),
        },
        recommendations=[
            Recommendation(
                Action.PROTECT,
                Disease.DOWNY_MILDEW,
                Fungicide.CONTACT,
                date(2026, 6, 11),
                date(2026, 6, 12),
                "downy_infection_forecast",
            ),
            Recommendation(
                Action.INSPECT,
                Disease.DOWNY_MILDEW,
                None,
                date(2026, 6, 20),
                date(2026, 6, 20),
                "downy_symptoms",
            ),
        ],
    )
    quiet = Plan(TODAY, {Disease.DOWNY_MILDEW: None}, [])

    lines = plan_text([("Криково", plan), ("Пуркарь", quiet)], total=3).splitlines()

    assert lines[:7] == [
        "План обработок на неделю:",
        "Криково:",
        "  защищён от оидиума до 14.06",
        "  до 11.06: защитная обработка от милдью контактным препаратом (медь)"
        " — по прогнозу заражение милдью 12.06",
        "  до 20.06: осмотр кустов — ожидаются пятна милдью после заражения",
        "Пуркарь:",
        "  обработки на неделю не нужны",
    ]
    assert lines[7] == "Ещё участков: 1 — план есть на сайте."
    assert plan_text([], total=0).startswith("Участков пока нет")


def test_history_text() -> None:
    rows = [
        CheckRow(
            datetime(2026, 6, 10, 6, 30, tzinfo=UTC),
            "downy_mildew",
            0.913,
            "Криково",
            "powdery_mildew",
        ),
        CheckRow(datetime(2026, 6, 9, 15, 0, tzinfo=UTC), "healthy", 0.88, None, None),
    ]

    assert history_text(rows).splitlines() == [
        "Последние проверки листьев:",
        "• 10.06 09:30, Криково: милдью, 91 % (агроном: оидиум)",
        "• 09.06 18:00: здоровый лист, 88 %",
    ]
    assert history_text([]).startswith("Сохранённых проверок пока нет")


@pytest.mark.integration
async def test_farm_reply_reads_owner_data(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(farm_bot, "local_today", lambda: TODAY)
    owner = User(email="grower@example.md", hashed_password="hash")
    neighbour = User(email="neighbour@example.md", hashed_password="hash")
    plot = Vineyard(owner=owner, name="Криково", geom=WKTElement(PLOT, srid=4326))
    foreign = Vineyard(owner=neighbour, name="Чужой", geom=WKTElement(PLOT, srid=4326))
    image = LeafImage(
        owner=owner,
        vineyard=plot,
        storage_key="leaves/1.jpg",
        content_type="image/jpeg",
        size_bytes=1,
        width=1,
        height=1,
    )
    session.add_all(
        [
            foreign,
            DiseaseRisk(
                vineyard=plot,
                disease=Disease.DOWNY_MILDEW,
                target_date=TODAY,
                method="three_tens",
                score=1,
                level=RiskLevel.HIGH,
            ),
            DiseaseResult(
                image=image,
                model_name="efficientnet_b0",
                model_version="1",
                status=ProcessingStatus.COMPLETED,
                predicted_class="downy_mildew",
                confidence=0.9,
            ),
            # Незавершённый анализ в историю не попадает
            DiseaseResult(image=image, model_name="efficientnet_b0", model_version="1"),
        ]
    )
    await session.flush()

    risk = await farm_reply(session, owner.id, "/risk")
    history = await farm_reply(session, owner.id, "/history")
    plan = await farm_reply(session, owner.id, "/plan")

    assert "• Криково: милдью 🔴 высокий, оидиум 🟢 низкий" in risk
    assert "Чужой" not in risk
    assert history.count("\n•") == 1
    assert ", Криково: милдью, 90 %" in history
    assert plan.startswith("План обработок на неделю:\nКриково:")


@pytest.mark.integration
def test_bot_answers_farm_questions(bot: tuple[TestClient, FakeTelegram]) -> None:
    client, telegram = bot

    def ask(text: str) -> str:
        update(client, {"message": {"chat": {"id": 42}, "text": text}})
        return telegram.sent[-1][1]

    # Справка о болезнях — и без привязанного чата
    assert ask("/diseases").startswith("Справка о болезнях")
    assert ask("Что такое эска?").startswith("Эска")
    assert ask("/risk") == NOT_LINKED

    headers = login(client)
    link(client, headers)
    assert ask("/risk").startswith("Участков пока нет")

    create_vineyard(client, headers, "Криково")
    answers: dict[str, Any] = {
        "/risk": "Риск болезней на сегодня",
        "Где сегодня опасно?": "Риск болезней на сегодня",
        "/plan": "План обработок на неделю",
        "Когда обрабатывать от милдью?": "План обработок на неделю",
        "/history": "Сохранённых проверок пока нет",
    }
    for question, start in answers.items():
        assert ask(question).startswith(start), question
