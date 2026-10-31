from typing import Any

import pytest

from app.services.telegram_bot import Update, handle_update
from app.services.wine_guide import (
    BUY_TEXT,
    LOCATION_KEYBOARD,
    NOTHING_NEAR,
    SPRAY_TEXT,
    WINE_TEXT,
    guide_reply,
    load_wineries,
    nearest_text,
)


class FakeTelegram:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str, Any]] = []

    async def send_message(self, chat_id: int, text: str, keyboard: Any = None) -> None:
        self.sent.append((chat_id, text, keyboard))


async def reply_to(**message: Any) -> tuple[int, str, Any]:
    telegram = FakeTelegram()
    update = Update.model_validate(
        {"update_id": 1, "message": {"chat": {"id": 7}, **message}}
    )
    await handle_update(
        update,
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        telegram,  # type: ignore[arg-type]
        lambda _name, _args: None,
    )
    return telegram.sent[0]


def test_known_wineries_are_in_the_list() -> None:
    names = {winery.name for winery in load_wineries()}

    assert {"Vinăria Novak", "Chateau Purcari", "Castelul Mimi"} <= names
    novak = next(item for item in load_wineries() if item.name == "Vinăria Novak")
    assert novak.approximate is True


def test_nearest_sorted_by_distance_and_marks_approximate() -> None:
    # Рядом с Вулканештами первой идёт примерная точка Novak
    text = nearest_text(45.68, 28.40, count=1)

    assert "Vinăria Novak" in text
    assert "Точный адрес не подтверждён" in text
    assert nearest_text(0, 0) == NOTHING_NEAR


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("Где купить вино?", BUY_TEXT),
        ("Чем брызгать от клеща", SPRAY_TEXT),
        ("Какое вино к мясу", WINE_TEXT),
        ("Как дела?", None),
    ],
)
def test_guide_reply(question: str, answer: str | None) -> None:
    assert guide_reply(question) == answer


@pytest.mark.parametrize(
    ("command", "answer"), [("/spray", SPRAY_TEXT), ("/wine", WINE_TEXT)]
)
async def test_commands_work_without_linked_chat(command: str, answer: str) -> None:
    assert await reply_to(text=command) == (7, answer, None)


async def test_buy_asks_for_location() -> None:
    assert await reply_to(text="/buy") == (7, BUY_TEXT, LOCATION_KEYBOARD)


async def test_location_gets_nearest_places() -> None:
    _, text, _ = await reply_to(location={"latitude": 46.53, "longitude": 29.87})

    assert "Chateau Purcari" in text


async def test_question_in_own_words_is_answered() -> None:
    assert await reply_to(text="чем брызгать от клеща") == (7, SPRAY_TEXT, None)
