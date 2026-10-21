"""Telegram-бот (pr-082): виноградарь присылает фото листа прямо из поля,
бот отвечает диагнозом и сохраняет проверку в историю учётной записи.

Чат привязывается к учётной записи одноразовой ссылкой из профиля
(`t.me/<бот>?start=<код>`): бот не спрашивает пароль, а код живёт 15 минут и
срабатывает один раз. Анализ фото идёт в задаче Celery — Telegram ждёт ответа
на вебхук недолго, а модель и загрузка файла занимают секунды."""

import secrets
import uuid
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.integrations.cache import Cache
from app.integrations.telegram import TelegramClient
from app.models.user import User

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
    "свете. Бот определит болезнь и сохранит проверку в историю VitiGuard MD."
)
NOT_LINKED = (
    "Этот чат не привязан к учётной записи. Откройте профиль в VitiGuard MD, "
    "нажмите «Подключить Telegram» и перейдите по ссылке."
)


class Chat(BaseModel):
    id: int


class PhotoSize(BaseModel):
    file_id: str
    width: int
    height: int


class Document(BaseModel):
    file_id: str
    mime_type: str | None = None


class Message(BaseModel):
    chat: Chat
    text: str | None = None
    photo: list[PhotoSize] = []
    # Фото, отправленное файлом: без сжатия и с EXIF, где бывают координаты
    document: Document | None = None


class Update(BaseModel):
    update_id: int
    message: Message | None = None


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
) -> None:
    message = update_.message
    if message is None:
        return
    chat_id = message.chat.id
    text = (message.text or "").strip()

    if text.startswith("/start"):
        code = text.removeprefix("/start").strip()
        if not code:
            await telegram.send_message(chat_id, HELP)
            return
        user = await link_chat(session, cache, code, chat_id)
        await telegram.send_message(
            chat_id,
            (
                f"Готово: чат привязан к {user.email}. {HELP}"
                if user
                else "Ссылка устарела или уже использована. Получите новую в профиле."
            ),
        )
        return

    owner = await session.scalar(
        select(User).where(User.telegram_chat_id == chat_id, User.is_active)
    )
    file_id = photo_file(message)
    if owner is None:
        await telegram.send_message(chat_id, NOT_LINKED)
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
