import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Query, status
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import CurrentUserDep, NotifierDep, SessionDep
from app.core.errors import NotFoundError
from app.models.alert import Alert
from app.schemas.alert import AlertQuery, AlertRead, ReadAllResult, UnreadCount
from app.schemas.errors import ErrorResponse
from app.schemas.pagination import Page
from app.services.risk import local_today

# Предупреждения о риске болезней (pr-080). Создаются при пересчёте риска
router = APIRouter(
    prefix="/alerts",
    tags=["alerts"],
    responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
)


def to_read(alert: Alert) -> AlertRead:
    return AlertRead(
        id=alert.id,
        created_at=alert.created_at,
        updated_at=alert.updated_at,
        vineyard_id=alert.vineyard_id,
        vineyard_name=alert.vineyard.name,
        disease=alert.disease,
        day=alert.day,
        level=alert.level,
        read_at=alert.read_at,
        forecast=alert.day > local_today(),
    )


@router.get("", summary="Мои предупреждения")
async def list_alerts(
    query: Annotated[AlertQuery, Query()], user: CurrentUserDep, session: SessionDep
) -> Page[AlertRead]:
    where = [Alert.user_id == user.id]
    if query.unread:
        where.append(Alert.read_at.is_(None))
    total = await session.scalar(select(func.count()).select_from(Alert).where(*where))
    rows = await session.scalars(
        select(Alert)
        .where(*where)
        .options(selectinload(Alert.vineyard))
        # Свежие и обновлённые — первыми
        .order_by(Alert.updated_at.desc(), Alert.id.desc())
        .offset(query.offset)
        .limit(query.size)
    )
    return Page.of([to_read(alert) for alert in rows], total or 0, query)


@router.get("/unread-count", summary="Число непрочитанных предупреждений")
async def unread_count(user: CurrentUserDep, session: SessionDep) -> UnreadCount:
    count = await session.scalar(
        select(func.count())
        .select_from(Alert)
        .where(Alert.user_id == user.id, Alert.read_at.is_(None))
    )
    return UnreadCount(count=count or 0)


@router.post(
    "/{alert_id}/read",
    summary="Отметить предупреждение прочитанным",
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
)
async def read_alert(
    alert_id: uuid.UUID,
    user: CurrentUserDep,
    session: SessionDep,
    notifier: NotifierDep,
) -> AlertRead:
    alert = await _owned_alert(session, user.id, alert_id)
    if alert.read_at is None:
        alert.read_at = datetime.now(UTC)
        await session.commit()
        # Значок обновится и в других вкладках и на телефоне
        await notifier.alerts_changed(user.id)
        # После commit объект устарел: перечитываем вместе с участком
        alert = await _owned_alert(session, user.id, alert_id)
    return to_read(alert)


async def _owned_alert(
    session: AsyncSession, user_id: uuid.UUID, alert_id: uuid.UUID
) -> Alert:
    alert = await session.scalar(
        select(Alert)
        .where(Alert.id == alert_id, Alert.user_id == user_id)
        .options(selectinload(Alert.vineyard))
        .execution_options(populate_existing=True)
    )
    # Чужое предупреждение неотличимо от несуществующего
    if alert is None:
        raise NotFoundError("Предупреждение не найдено")
    return alert


@router.post("/read-all", summary="Отметить все предупреждения прочитанными")
async def read_all(
    user: CurrentUserDep, session: SessionDep, notifier: NotifierDep
) -> ReadAllResult:
    updated = await session.scalars(
        update(Alert)
        .where(Alert.user_id == user.id, Alert.read_at.is_(None))
        .values(read_at=datetime.now(UTC))
        .returning(Alert.id)
    )
    count = len(updated.all())
    await session.commit()
    if count:
        await notifier.alerts_changed(user.id)
    return ReadAllResult(updated=count)
