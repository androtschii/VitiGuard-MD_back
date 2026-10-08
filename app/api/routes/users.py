from fastapi import APIRouter, status

from app.api.deps import CurrentUserDep, SessionDep
from app.repositories.user import UserRepository
from app.schemas.errors import ErrorResponse
from app.schemas.user import UserRead, UserUpdate

router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
)


@router.get("/me", summary="Мой профиль")
async def read_me(user: CurrentUserDep) -> UserRead:
    return UserRead.model_validate(user)


@router.patch("/me", summary="Изменение профиля")
async def update_me(
    data: UserUpdate, user: CurrentUserDep, session: SessionDep
) -> UserRead:
    updated = await UserRepository(session).update(user, **data.model_dump())
    await session.commit()
    return UserRead.model_validate(updated)
