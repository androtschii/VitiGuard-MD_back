"""Предупреждения в push-уведомления браузера (Web Push, pr-081).

Используется стандарт Web Push с ключами VAPID: браузер сам выбирает
push-сервис, отдельный проект Firebase не нужен. Отметка отправки своя
(`pushed_at`), независимая от Telegram: один канал не гасит другой."""

import asyncio
import json
import uuid
from collections.abc import Callable
from datetime import date

from pywebpush import WebPushException, webpush
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models.alert import Alert
from app.models.push_subscription import PushSubscription
from app.models.user import User
from app.models.vineyard import Vineyard
from app.services.telegram_bot import AlertRow, alerts_text

# Подписку отменили в браузере или она истекла: адрес больше не существует
GONE = {404, 410}


class PushRejected(Exception):
    """Push-сервис не принял сообщение; gone — подписки больше нет."""

    def __init__(self, gone: bool) -> None:
        super().__init__("push rejected")
        self.gone = gone


Deliver = Callable[[PushSubscription, str, Settings], None]


def deliver(subscription: PushSubscription, payload: str, settings: Settings) -> None:
    """Отправка через pywebpush (синхронный вызов, идёт в отдельном потоке)."""
    assert settings.vapid_private_key is not None
    try:
        webpush(
            subscription_info={
                "endpoint": subscription.endpoint,
                "keys": {"p256dh": subscription.p256dh, "auth": subscription.auth},
            },
            data=payload,
            vapid_private_key=settings.vapid_private_key.get_secret_value(),
            vapid_claims={"sub": settings.vapid_subject},
            timeout=settings.external_http_timeout,
        )
    except WebPushException as error:
        status = error.response.status_code if error.response is not None else None
        raise PushRejected(gone=status in GONE) from error


async def send_push_alerts(
    session: AsyncSession,
    settings: Settings,
    today: date,
    send: Deliver = deliver,
) -> int:
    """Новые предупреждения — по одному уведомлению на устройство. Отметка
    ставится, если ушло хотя бы на одно устройство владельца; исчезнувшие
    подписки удаляются. Возвращает число отправленных уведомлений."""
    rows = await session.execute(
        select(
            Alert.id,
            Alert.user_id,
            Vineyard.name,
            Alert.disease,
            Alert.day,
            Alert.level,
        )
        .join(User, User.id == Alert.user_id)
        .join(Vineyard, Vineyard.id == Alert.vineyard_id)
        .where(
            Alert.pushed_at.is_(None),
            Alert.read_at.is_(None),
            Alert.day >= today,
            User.is_active,
        )
        .order_by(Alert.user_id, Vineyard.name, Alert.day, Alert.disease)
    )
    by_user: dict[uuid.UUID, list[tuple[uuid.UUID, AlertRow]]] = {}
    for alert_id, user_id, name, disease, day, level in rows:
        by_user.setdefault(user_id, []).append((alert_id, (name, disease, day, level)))
    if not by_user:
        return 0
    subscriptions = (
        await session.scalars(
            select(PushSubscription).where(PushSubscription.user_id.in_(by_user))
        )
    ).all()

    sent = 0
    for user_id, items in by_user.items():
        payload = json.dumps(
            {
                "title": "VitiGuard MD",
                "body": alerts_text([row for _, row in items], today),
                "url": "/alerts",
            },
            ensure_ascii=False,
        )
        delivered = False
        for subscription in (s for s in subscriptions if s.user_id == user_id):
            try:
                await asyncio.to_thread(send, subscription, payload, settings)
            except PushRejected as error:
                if error.gone:
                    await session.execute(
                        delete(PushSubscription).where(
                            PushSubscription.id == subscription.id
                        )
                    )
                continue
            delivered = True
            sent += 1
        if delivered:
            await session.execute(
                update(Alert)
                .where(Alert.id.in_([alert_id for alert_id, _ in items]))
                .values(pushed_at=func.now())
            )
        await session.commit()
    return sent
