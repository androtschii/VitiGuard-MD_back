from fastapi import APIRouter, status

from app.api.deps import SessionDep
from app.schemas.auth import RegisterRequest
from app.schemas.errors import ErrorResponse
from app.schemas.user import UserRead
from app.services.auth import register_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    summary="Регистрация",
    responses={status.HTTP_409_CONFLICT: {"model": ErrorResponse}},
)
async def register(data: RegisterRequest, session: SessionDep) -> UserRead:
    user = await register_user(session, data)
    return UserRead.model_validate(user)
