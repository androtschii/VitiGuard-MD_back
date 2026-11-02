from typing import Annotated

from fastapi import APIRouter, Query, status
from sqlalchemy import delete, select

from app.api.deps import CurrentUserDep, SessionDep, SettingsDep
from app.core.errors import ServiceUnavailableError
from app.models.push_subscription import PushSubscription
from app.schemas.errors import ErrorResponse
from app.schemas.push import PushKeyRead, PushSubscriptionCreate

router = APIRouter(
    prefix="/push",
    tags=["push"],
    responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
)


@router.get(
    "/key",
    summary="Открытый ключ VAPID для подписки браузера",
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse}},
)
async def push_key(settings: SettingsDep, _: CurrentUserDep) -> PushKeyRead:
    if settings.vapid_public_key is None or settings.vapid_private_key is None:
        raise ServiceUnavailableError("Push-уведомления на сервере не настроены")
    return PushKeyRead(public_key=settings.vapid_public_key)


@router.post(
    "/subscriptions",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Подписать этот браузер на push-уведомления",
)
async def subscribe(
    body: PushSubscriptionCreate, user: CurrentUserDep, session: SessionDep
) -> None:
    endpoint = str(body.endpoint)
    row = await session.scalar(
        select(PushSubscription).where(PushSubscription.endpoint == endpoint)
    )
    # Тот же браузер после смены пользователя: подписка переходит к новому
    if row is None:
        row = PushSubscription(endpoint=endpoint, user_id=user.id, p256dh="", auth="")
        session.add(row)
    row.user_id = user.id
    row.p256dh = body.keys.p256dh
    row.auth = body.keys.auth
    await session.commit()


@router.delete(
    "/subscriptions",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Отписать этот браузер",
)
async def unsubscribe(
    endpoint: Annotated[str, Query(max_length=1000)],
    user: CurrentUserDep,
    session: SessionDep,
) -> None:
    await session.execute(
        delete(PushSubscription).where(
            PushSubscription.endpoint == endpoint,
            PushSubscription.user_id == user.id,
        )
    )
    await session.commit()
