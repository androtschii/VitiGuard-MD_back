from pydantic import Field, HttpUrl

from app.schemas.base import RequestSchema, ResponseSchema


class PushKeys(RequestSchema):
    """Ключи шифрования подписки из PushSubscription.toJSON() браузера."""

    p256dh: str = Field(min_length=1, max_length=200)
    auth: str = Field(min_length=1, max_length=100)


class PushSubscriptionCreate(RequestSchema):
    """Подписка браузера на push-уведомления."""

    endpoint: HttpUrl = Field(description="Адрес push-сервиса браузера")
    keys: PushKeys


class PushKeyRead(ResponseSchema):
    """Открытый ключ VAPID: с ним браузер оформляет подписку."""

    public_key: str
