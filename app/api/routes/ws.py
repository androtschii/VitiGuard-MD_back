import asyncio
import contextlib
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from redis.asyncio.client import PubSub
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.errors import UnauthorizedError
from app.core.tokens import TokenType, decode_token
from app.integrations.notifier import RedisNotifier
from app.models.alert import Alert
from app.models.user import User

# Уведомления в реальном времени (pr-093). Пока единственное — число
# непрочитанных предупреждений для значка в шапке
router = APIRouter(prefix="/ws", tags=["ws"])


async def unread(sessions: async_sessionmaker[AsyncSession], user_id: uuid.UUID) -> int:
    async with sessions() as session:
        count = await session.scalar(
            select(func.count())
            .select_from(Alert)
            .where(Alert.user_id == user_id, Alert.read_at.is_(None))
        )
    return count or 0


async def authenticate(websocket: WebSocket, token: str) -> uuid.UUID | None:
    settings = websocket.app.state.settings
    try:
        claims = decode_token(token, TokenType.ACCESS, settings)
    except UnauthorizedError:
        return None
    async with websocket.app.state.session_factory() as session:
        user = await session.get(User, claims.subject)
    return user.id if user is not None and user.is_active else None


async def forward(
    websocket: WebSocket,
    pubsub: PubSub,
    sessions: async_sessionmaker[AsyncSession],
    user_id: uuid.UUID,
) -> None:
    async for message in pubsub.listen():
        if message["type"] == "message":
            await websocket.send_json({"unread": await unread(sessions, user_id)})


@router.websocket("/alerts")
async def alerts_socket(websocket: WebSocket, token: str = "") -> None:
    """Число непрочитанных предупреждений: сразу при подключении и каждый раз,
    когда предупреждения меняются. Заголовок Authorization браузер в WebSocket
    не передаёт, поэтому access-токен — в параметре ?token=."""
    user_id = await authenticate(websocket, token)
    if user_id is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await websocket.accept()
    sessions = websocket.app.state.session_factory
    notifier: RedisNotifier = websocket.app.state.notifier
    async with notifier.subscribe(user_id) as pubsub:
        await websocket.send_json({"unread": await unread(sessions, user_id)})
        sender = asyncio.create_task(forward(websocket, pubsub, sessions, user_id))
        try:
            # Клиент ничего не присылает; чтение нужно, чтобы заметить отключение
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender
