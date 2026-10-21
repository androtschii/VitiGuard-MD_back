import asyncio
import uuid

import httpx2 as httpx
from celery import shared_task

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.db.session import create_engine, create_session_factory
from app.integrations.object_storage import ObjectStorage, Storage
from app.integrations.telegram import TelegramClient
from app.ml.model_file import describe_model
from app.services.diagnosis import analyze_leaf, photo_location, read_leaf_image
from app.services.diagnosis_history import save_diagnosis
from app.services.telegram_bot import DIAGNOSE_PHOTO, diagnosis_text
from app.tasks.diagnosis import load_classifier, load_segmenter

MAX_IMAGE_BYTES = 10 * 1024 * 1024


async def diagnose_photo(
    settings: Settings,
    storage: Storage,
    telegram: TelegramClient,
    chat_id: int,
    owner_id: str,
    file_id: str,
) -> str:
    """Фото из чата: загрузка, диагноз, запись в историю и ответ. Ответ
    приходит в чат при любом исходе — молчание бота выглядело бы поломкой."""
    if not settings.diagnosis_model_path.exists():
        reply = "Диагностика сейчас недоступна: модель не загружена на сервер."
        await telegram.send_message(chat_id, reply)
        return reply
    data = await telegram.download_file(file_id, MAX_IMAGE_BYTES)
    if data is None:
        reply = "Файл больше 10 МБ — пришлите фото поменьше."
        await telegram.send_message(chat_id, reply)
        return reply
    try:
        leaf = read_leaf_image(data)
    except AppError:
        reply = "Не удалось прочитать фото: нужен JPEG, PNG или WebP."
        await telegram.send_message(chat_id, reply)
        return reply
    diagnosis = await asyncio.to_thread(
        analyze_leaf,
        leaf,
        load_classifier(settings.diagnosis_model_path),
        load_segmenter(settings.lesion_model_path),
        3,
    )
    diagnosis.location = photo_location(data)
    version = describe_model(settings.diagnosis_model_path).version
    engine = create_engine(settings)
    try:
        async with create_session_factory(engine)() as session:
            # Из чата фото приходит без участка: его можно указать в истории
            diagnosis.id = await save_diagnosis(
                session,
                storage,
                uuid.UUID(owner_id),
                None,
                data,
                leaf,
                diagnosis,
                version,
            )
    finally:
        await engine.dispose()
    reply = diagnosis_text(
        diagnosis.disease, diagnosis.confidence, diagnosis.low_confidence
    )
    await telegram.send_message(chat_id, reply)
    return reply


@shared_task(name=DIAGNOSE_PHOTO)
def diagnose_photo_task(chat_id: int, owner_id: str, file_id: str) -> str:
    settings = get_settings()

    async def run() -> str:
        async with httpx.AsyncClient(timeout=30) as http:
            return await diagnose_photo(
                settings,
                ObjectStorage.from_settings(settings),
                TelegramClient(http, settings),
                chat_id,
                owner_id,
                file_id,
            )

    return asyncio.run(run())
