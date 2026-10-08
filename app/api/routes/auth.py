from fastapi import APIRouter, Response, status

from app.api.deps import SessionDep, SettingsDep
from app.core.config import Settings
from app.core.tokens import IssuedToken
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse
from app.schemas.errors import ErrorResponse
from app.schemas.user import UserRead
from app.services.auth import login, register_user

REFRESH_COOKIE = "refresh_token"

router = APIRouter(prefix="/auth", tags=["auth"])


def set_refresh_cookie(
    response: Response, refresh: IssuedToken, settings: Settings
) -> None:
    response.set_cookie(
        REFRESH_COOKIE,
        refresh.token,
        max_age=settings.refresh_token_ttl_days * 24 * 60 * 60,
        # Cookie уходит только на эндпоинты авторизации: остальным запросам
        # refresh-токен не нужен, и чем реже он передаётся, тем меньше мест утечки
        path=f"{settings.api_v1_prefix}{router.prefix}",
        # Скрипт на странице cookie не прочитает (XSS не украдёт токен), а
        # SameSite=Strict не даёт чужому сайту отправить запрос с ней (CSRF)
        httponly=True,
        secure=settings.refresh_cookie_secure,
        samesite="strict",
    )


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    summary="Регистрация",
    responses={status.HTTP_409_CONFLICT: {"model": ErrorResponse}},
)
async def register(data: RegisterRequest, session: SessionDep) -> UserRead:
    user = await register_user(session, data)
    return UserRead.model_validate(user)


@router.post(
    "/login",
    summary="Вход",
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
        status.HTTP_403_FORBIDDEN: {"model": ErrorResponse},
    },
)
async def log_in(
    data: LoginRequest,
    response: Response,
    session: SessionDep,
    settings: SettingsDep,
) -> TokenResponse:
    tokens = await login(session, data, settings)
    set_refresh_cookie(response, tokens.refresh, settings)
    return TokenResponse(access_token=tokens.access.token)
