"""Telegram-бот (pr-082): виноградарь присылает фото листа прямо из поля,
бот отвечает диагнозом и сохраняет проверку в историю учётной записи.

Чат привязывается к учётной записи одноразовой ссылкой из профиля
(`t.me/<бот>?start=<код>`): бот не спрашивает пароль, а код живёт 15 минут и
срабатывает один раз. Анализ фото идёт в задаче Celery — Telegram ждёт ответа
на вебхук недолго, а модель и загрузка файла занимают секунды.

В тот же чат приходят предупреждения о риске болезней (pr-083); по каким
участкам — владелец выбирает кнопками по команде /alerts."""

import secrets
import uuid
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from typing import Any

from pydantic import BaseModel
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ExternalServiceError
from app.integrations.cache import Cache
from app.integrations.telegram import TelegramClient
from app.models.alert import Alert
from app.models.enums import Disease, RiskLevel
from app.models.user import User
from app.models.vineyard import Vineyard
from app.services.wine_guide import (
    BUY_TEXT,
    LOCATION_KEYBOARD,
    SPRAY_TEXT,
    WINE_TEXT,
    guide_reply,
    nearest_text,
)

LINK_TTL_SECONDS = 15 * 60
DIAGNOSE_PHOTO = "telegram.diagnose_photo"

DISEASE_NAMES = {
    "healthy": "здоровый лист",
    "black_rot": "чёрная гниль",
    "esca": "эска",
    "leaf_blight": "пятнистость листьев",
    "downy_mildew": "милдью",
    "powdery_mildew": "оидиум",
}

HELP = (
    "Пришлите фото листа винограда — один лист крупным планом, при дневном "
    "свете. Бот определит болезнь и сохранит проверку в историю VitiGuard MD. "
    "Предупреждения о риске по участкам — команда /alerts."
)
PHOTO_HINT = (
    "Как проверить лист:\n"
    "1. Нажмите скрепку и выберите «Камера» или фото из галереи.\n"
    "2. Снимите один лист целиком, крупно, при дневном свете, без бликов.\n"
    "3. Отправьте — через несколько секунд придёт диагноз.\n"
    "Совет: отправьте фото «файлом», тогда сохранятся координаты съёмки и "
    "проверка попадёт на свой участок."
)
WELCOME = (
    "Здравствуйте! Это бот VitiGuard MD — мониторинг здоровья виноградников.\n"
    "📷 Пришлите фото листа — бот определит болезнь.\n"
    "🔔 /alerts — предупреждения о риске по участкам.\n"
    "📱 /app — открыть приложение на телефоне."
)
NO_APP = (
    "Адрес приложения для телефона пока не настроен. Откройте VitiGuard MD "
    "в браузере компьютера."
)

# Меню команд у поля ввода: «/» или кнопка «Меню»
COMMANDS = [
    ("start", "Начало и ссылка на приложение"),
    ("photo", "Как проверить лист по фото"),
    ("alerts", "Предупреждения о риске по участкам"),
    ("spray", "Чем обрабатывать виноград"),
    ("wine", "Какое вино выбрать"),
    ("buy", "Где купить вино рядом"),
    ("app", "Открыть VitiGuard MD на телефоне"),
    ("help", "Что умеет бот"),
]
QUICK_DIAGNOSIS_PATH = "/diagnosis/quick"


def https_url(url: str | None) -> str | None:
    """Адрес для кнопок: Telegram принимает в кнопках только https."""
    return url.rstrip("/") if url and url.startswith("https://") else None


def app_keyboard(app_url: str | None) -> dict[str, Any] | None:
    """Кнопки под сообщением: мини-приложение прямо в Telegram и обычная
    ссылка — открыть то же приложение в браузере телефона."""
    url = https_url(app_url)
    if url is None:
        return None
    return {
        "inline_keyboard": [
            [
                {
                    "text": "📷 Проверить лист в приложении",
                    "web_app": {"url": f"{url}{QUICK_DIAGNOSIS_PATH}"},
                }
            ],
            [{"text": "🌐 Открыть в браузере телефона", "url": f"{url}/"}],
        ]
    }


async def configure_bot(telegram: TelegramClient, app_url: str | None) -> None:
    """Меню команд и кнопка меню бота; вызывается при запуске бота."""
    await telegram.set_commands(COMMANDS)
    url = https_url(app_url)
    await telegram.set_menu_button(
        "Приложение" if url else None,
        f"{url}{QUICK_DIAGNOSIS_PATH}" if url else None,
    )


NOT_LINKED = (
    "Этот чат не привязан к учётной записи. Откройте профиль в VitiGuard MD, "
    "нажмите «Подключить Telegram» и перейдите по ссылке."
)


class Chat(BaseModel):
    """Чат Telegram из обновления бота."""

    id: int


class PhotoSize(BaseModel):
    """Один из размеров фото, присланного боту."""

    file_id: str
    width: int
    height: int


class Document(BaseModel):
    """Файл, присланный боту документом (без сжатия, с EXIF)."""

    file_id: str
    mime_type: str | None = None


class Location(BaseModel):
    """Точка, которую пользователь отправил кнопкой."""

    latitude: float
    longitude: float


class Message(BaseModel):
    """Сообщение Telegram из обновления бота: текст, фото или документ."""

    message_id: int = 0
    chat: Chat
    text: str | None = None
    photo: list[PhotoSize] = []
    # Фото, отправленное файлом: без сжатия и с EXIF, где бывают координаты
    document: Document | None = None
    location: Location | None = None


class CallbackQuery(BaseModel):
    """Нажатие кнопки под сообщением бота."""

    id: str
    message: Message | None = None
    data: str | None = None


class Update(BaseModel):
    """Обновление Telegram Bot API (вебхук или getUpdates)."""

    update_id: int
    message: Message | None = None
    callback_query: CallbackQuery | None = None


def link_key(code: str) -> str:
    return f"telegram:link:{code}"


async def create_link_code(cache: Cache, user_id: uuid.UUID) -> str:
    # 16 байт — 22 символа: deep-link Telegram допускает до 64 из [A-Za-z0-9_-]
    code = secrets.token_urlsafe(16)
    await cache.set_json(link_key(code), str(user_id), LINK_TTL_SECONDS)
    return code


async def link_chat(
    session: AsyncSession, cache: Cache, code: str, chat_id: int
) -> User | None:
    user_id = await cache.pop_json(link_key(code))
    if user_id is None:
        return None
    user = await session.get(User, uuid.UUID(user_id))
    if user is None or not user.is_active:
        return None
    # Чат мог быть привязан к другой учётной записи: теперь он принадлежит этой
    await session.execute(
        update(User)
        .where(User.telegram_chat_id == chat_id)
        .values(telegram_chat_id=None)
    )
    user.telegram_chat_id = chat_id
    await session.commit()
    return user


def photo_file(message: Message) -> str | None:
    if message.photo:
        # Telegram присылает одно фото в нескольких размерах: берём крупнейший
        return max(message.photo, key=lambda size: size.width * size.height).file_id
    document = message.document
    if document and (document.mime_type or "").startswith("image/"):
        return document.file_id
    return None


async def handle_update(
    update_: Update,
    session: AsyncSession,
    cache: Cache,
    telegram: TelegramClient,
    enqueue: Callable[[str, list[Any]], object],
    app_url: str | None = None,
) -> None:
    if update_.callback_query is not None:
        await toggle_subscription(session, telegram, update_.callback_query)
        return
    message = update_.message
    if message is None:
        return
    chat_id = message.chat.id
    text = (message.text or "").strip()

    keyboard = app_keyboard(app_url)
    if text.startswith("/start"):
        code = text.removeprefix("/start").strip()
        if not code:
            linked = await session.scalar(
                select(User.id).where(User.telegram_chat_id == chat_id, User.is_active)
            )
            await telegram.send_message(
                chat_id,
                WELCOME if linked else f"{WELCOME}\n\n{NOT_LINKED}",
                keyboard,
            )
            return
        user = await link_chat(session, cache, code, chat_id)
        await telegram.send_message(
            chat_id,
            (
                f"Готово: чат привязан к {user.email}. {HELP}"
                if user
                else "Ссылка устарела или уже использована. Получите новую в профиле."
            ),
            keyboard if user else None,
        )
        return
    # Подсказки и ссылка на приложение — и без привязанного чата
    if text == "/app":
        await telegram.send_message(
            chat_id,
            (
                "Откройте VitiGuard MD: в Telegram — кнопкой ниже, "
                "или в браузере телефона по ссылке."
                if keyboard
                else NO_APP
            ),
            keyboard,
        )
        return
    if text == "/photo":
        await telegram.send_message(chat_id, PHOTO_HINT)
        return
    if text == "/help":
        await telegram.send_message(chat_id, f"{HELP}\n\n{WELCOME}", keyboard)
        return
    # Подсказки про обработки и вино не требуют учётной записи
    if text == "/spray":
        await telegram.send_message(chat_id, SPRAY_TEXT)
        return
    if text == "/wine":
        await telegram.send_message(chat_id, WINE_TEXT)
        return
    if text == "/buy":
        await telegram.send_message(chat_id, BUY_TEXT, LOCATION_KEYBOARD)
        return
    if message.location is not None:
        await telegram.send_message(
            chat_id,
            nearest_text(message.location.latitude, message.location.longitude),
        )
        return
    answer = guide_reply(text) if text else None
    if answer is not None:
        await telegram.send_message(
            chat_id, answer, LOCATION_KEYBOARD if answer == BUY_TEXT else None
        )
        return

    owner = await session.scalar(
        select(User).where(User.telegram_chat_id == chat_id, User.is_active)
    )
    file_id = photo_file(message)
    if owner is None:
        await telegram.send_message(chat_id, NOT_LINKED)
    elif text == "/alerts":
        vineyards = await owned_vineyards(session, owner)
        await telegram.send_message(
            chat_id,
            (
                "Предупреждения о риске болезней по участкам. "
                "Нажмите на участок, чтобы включить или выключить."
                if vineyards
                else "Участков пока нет: нарисуйте их на карте в VitiGuard MD."
            ),
            alerts_keyboard(vineyards) if vineyards else None,
        )
    elif file_id is None:
        await telegram.send_message(chat_id, HELP)
    else:
        enqueue(DIAGNOSE_PHOTO, [chat_id, str(owner.id), file_id])
        await telegram.send_message(chat_id, "Фото получено, анализирую…")


def diagnosis_text(disease: str, confidence: float, low_confidence: bool) -> str:
    name = DISEASE_NAMES.get(disease, disease)
    lines = [f"Диагноз: {name} (уверенность {round(confidence * 100)} %)."]
    if low_confidence:
        lines.append(
            "Модель не уверена: переснимите один лист крупным планом при "
            "дневном свете, без бликов и теней."
        )
    lines.append("Проверка сохранена в истории VitiGuard MD.")
    return "\n".join(lines)


ALERTS_PREFIX = "alerts:"
LEVEL_NAMES = {RiskLevel.MEDIUM: "средний", RiskLevel.HIGH: "высокий"}


async def owned_vineyards(session: AsyncSession, owner: User) -> list[Vineyard]:
    rows = await session.scalars(
        select(Vineyard).where(Vineyard.owner_id == owner.id).order_by(Vineyard.name)
    )
    return list(rows)


def alerts_keyboard(vineyards: Sequence[Vineyard]) -> dict[str, Any]:
    """Кнопка на участок: колокольчик — предупреждения приходят. В данных
    кнопки — только номер участка (до 64 байт), владелец проверяется заново."""
    return {
        "inline_keyboard": [
            [
                {
                    "text": f"{'🔔' if vineyard.telegram_alerts else '🔕'} {vineyard.name}",
                    "callback_data": f"{ALERTS_PREFIX}{vineyard.id}",
                }
            ]
            for vineyard in vineyards
        ]
    }


async def toggle_subscription(
    session: AsyncSession, telegram: TelegramClient, query: CallbackQuery
) -> None:
    message = query.message
    data = query.data or ""
    if message is None or not data.startswith(ALERTS_PREFIX):
        await telegram.answer_callback(query.id, "Кнопка устарела")
        return
    owner = await session.scalar(
        select(User).where(User.telegram_chat_id == message.chat.id, User.is_active)
    )
    try:
        vineyard_id = uuid.UUID(data.removeprefix(ALERTS_PREFIX))
    except ValueError:
        vineyard_id = None
    vineyard = (
        await session.get(Vineyard, vineyard_id) if vineyard_id is not None else None
    )
    # Кнопку мог нажать чат, уже отвязанный от владельца участка
    if owner is None or vineyard is None or vineyard.owner_id != owner.id:
        await telegram.answer_callback(query.id, "Участок не найден")
        return
    vineyard.telegram_alerts = not vineyard.telegram_alerts
    await session.commit()
    await telegram.answer_callback(
        query.id,
        f"{vineyard.name}: предупреждения "
        + ("включены" if vineyard.telegram_alerts else "выключены"),
    )
    await telegram.edit_reply_markup(
        message.chat.id,
        message.message_id,
        alerts_keyboard(await owned_vineyards(session, owner)),
    )


# «Риск чего»: родительный падеж
RISK_OF = {Disease.DOWNY_MILDEW: "милдью", Disease.POWDERY_MILDEW: "оидиума"}

AlertRow = tuple[str, Disease, date, RiskLevel]


def alerts_text(rows: Sequence[AlertRow], today: date) -> str:
    lines = ["Предупреждения VitiGuard MD:"]
    for name, disease, day, level in rows:
        when = {today: "сегодня", today + timedelta(days=1): "завтра"}.get(
            day, day.strftime("%d.%m")
        )
        lines.append(
            f"• {name}: {LEVEL_NAMES.get(level, 'низкий')} риск "
            f"{RISK_OF[disease]} {when}"
        )
    lines.append("Что делать — в окне «Риск по дням» участка.")
    return "\n".join(lines)


async def send_alerts(
    session: AsyncSession, telegram: TelegramClient, today: date
) -> int:
    """Новые и усилившиеся предупреждения — в чаты владельцев (pr-083): одно
    сообщение на чат. Прошедшие дни не отправляются. Если Telegram не принял
    сообщение, отметка не ставится, и оно уйдёт при следующем пересчёте.
    Возвращает число отправленных сообщений."""
    rows = await session.execute(
        select(
            Alert.id,
            User.telegram_chat_id,
            Vineyard.name,
            Alert.disease,
            Alert.day,
            Alert.level,
        )
        .join(User, User.id == Alert.user_id)
        .join(Vineyard, Vineyard.id == Alert.vineyard_id)
        .where(
            Alert.notified_at.is_(None),
            Alert.read_at.is_(None),
            Alert.day >= today,
            User.telegram_chat_id.is_not(None),
            User.is_active,
            Vineyard.telegram_alerts,
        )
        .order_by(User.telegram_chat_id, Vineyard.name, Alert.day, Alert.disease)
    )
    by_chat: dict[int, list[tuple[uuid.UUID, AlertRow]]] = {}
    for alert_id, chat_id, name, disease, day, level in rows:
        # Чат не NULL — это условие запроса
        assert chat_id is not None
        by_chat.setdefault(chat_id, []).append((alert_id, (name, disease, day, level)))
    sent = 0
    for chat_id, items in by_chat.items():
        try:
            await telegram.send_message(
                chat_id, alerts_text([row for _, row in items], today)
            )
        except ExternalServiceError:
            continue
        await session.execute(
            update(Alert)
            .where(Alert.id.in_([alert_id for alert_id, _ in items]))
            .values(notified_at=func.now())
        )
        await session.commit()
        sent += 1
    return sent
