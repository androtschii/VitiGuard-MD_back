"""Уведомление владельцу фото о заключении агронома.

Агроном работает с очередью проверок всех виноградарей; без уведомления
владелец узнал бы о заключении, только открыв историю. Сообщение уходит в
Telegram (если чат привязан) и Web Push на все подписки владельца. Отдельно в
базе уведомление не хранится: само заключение уже есть в истории проверок.
"""

import asyncio
import json
import uuid

import structlog
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import Settings
from app.integrations.telegram import TelegramClient
from app.models.diagnosis import DiseaseResult, LeafImage
from app.models.push_subscription import PushSubscription
from app.services.risk import local_today
from app.services.telegram_bot import DISEASE_NAMES
from app.services.web_push import Deliver, PushRejected, deliver

logger = structlog.get_logger(__name__)

PUSH_TITLE = "Заключение агронома"


def review_text(result: DiseaseResult) -> str:
    disease = result.expert_class or ""
    # Дата снимка по Кишинёву, как во всём интерфейсе
    day = local_today(result.image.created_at).strftime("%d.%m")
    text = (
        f"Агроном проверил ваш снимок от {day}: {DISEASE_NAMES.get(disease, disease)}."
    )
    if result.expert_comment:
        text += f"\nКомментарий: {result.expert_comment}"
    return text


async def notify_review(
    session: AsyncSession,
    settings: Settings,
    telegram: TelegramClient | None,
    result_id: uuid.UUID,
    send: Deliver = deliver,
) -> dict[str, int]:
    """Отправляет владельцу текст заключения. Ошибка одного канала не мешает
    другому. Возвращает число доставленных сообщений по каналам."""
    sent = {"telegram": 0, "push": 0}
    result = await session.scalar(
        select(DiseaseResult)
        .where(DiseaseResult.id == result_id)
        .options(selectinload(DiseaseResult.image).selectinload(LeafImage.owner))
    )
    # Заключение могли отозвать, пока задача ждала в очереди
    if result is None or result.expert_class is None:
        return sent
    owner = result.image.owner
    if not owner.is_active:
        return sent
    text = review_text(result)

    if telegram is not None and owner.telegram_chat_id is not None:
        try:
            await telegram.send_message(owner.telegram_chat_id, text)
            sent["telegram"] = 1
        except Exception:
            logger.warning("review_notice_telegram_failed")

    if settings.vapid_private_key is not None:
        payload = json.dumps(
            {"title": PUSH_TITLE, "body": text, "url": "/history"},
            ensure_ascii=False,
        )
        subscriptions = (
            await session.scalars(
                select(PushSubscription).where(PushSubscription.user_id == owner.id)
            )
        ).all()
        for subscription in subscriptions:
            try:
                await asyncio.to_thread(send, subscription, payload, settings)
            except PushRejected as error:
                if error.gone:
                    await session.execute(
                        delete(PushSubscription).where(
                            PushSubscription.id == subscription.id
                        )
                    )
                else:
                    logger.warning("review_notice_push_failed")
                continue
            sent["push"] += 1
        await session.commit()
    return sent
