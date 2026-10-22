import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Query, status
from sqlalchemy import func, or_, select

from app.api.deps import AdminDep, SessionDep
from app.core.errors import BadRequestError, NotFoundError
from app.models.user import User, UserRole
from app.repositories.auth import RefreshTokenRepository
from app.schemas.errors import ErrorResponse
from app.schemas.pagination import Page
from app.schemas.user import AdminUserUpdate, UserRead, UserSearchParams

# Управление пользователями и ролями (frontend/pr-091): только администратору
router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
        status.HTTP_403_FORBIDDEN: {"model": ErrorResponse},
    },
)


@router.get("/users", summary="Пользователи системы")
async def list_users(
    admin: AdminDep,
    session: SessionDep,
    params: Annotated[UserSearchParams, Query()],
) -> Page[UserRead]:
    query = select(User)
    search = (params.q or "").strip()
    if search:
        pattern = f"%{search}%"
        query = query.where(
            or_(User.email.ilike(pattern), User.full_name.ilike(pattern))
        )
    total = await session.scalar(select(func.count()).select_from(query.subquery()))
    rows = await session.scalars(
        query.order_by(User.created_at.desc(), User.email)
        .offset(params.offset)
        .limit(params.size)
    )
    return Page[UserRead].of(
        [UserRead.model_validate(user) for user in rows], total or 0, params
    )


@router.patch(
    "/users/{user_id}",
    summary="Роль и доступ пользователя",
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
    },
)
async def update_user(
    user_id: uuid.UUID,
    data: AdminUserUpdate,
    admin: AdminDep,
    session: SessionDep,
) -> UserRead:
    user = await session.get(User, user_id)
    if user is None:
        raise NotFoundError("Пользователь не найден")
    # Администратор не снимает права и не блокирует сам себя: так в системе
    # всегда остаётся хотя бы один администратор
    if user.id == admin.id and (
        (data.role is not None and data.role is not UserRole.ADMIN)
        or data.is_active is False
    ):
        raise BadRequestError("Нельзя снять права администратора с самого себя")
    if data.role is not None:
        user.role = data.role
    if data.is_active is not None:
        user.is_active = data.is_active
        if not data.is_active:
            # Заблокированный выходит со всех устройств сразу
            await RefreshTokenRepository(session).revoke_all_for_user(
                user.id, datetime.now(UTC)
            )
    await session.commit()
    await session.refresh(user)
    return UserRead.model_validate(user)
