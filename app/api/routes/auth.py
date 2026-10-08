from typing import Annotated

from fastapi import APIRouter, Cookie, Response, status

from app.api.deps import SessionDep, SettingsDep
from app.core.config import Settings
from app.core.tokens import IssuedToken
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse
from app.schemas.errors import ErrorResponse
from app.schemas.user import UserRead
from app.services.auth import login, logout, refresh_session, register_user

REFRESH_COOKIE = "refresh_token"

router = APIRouter(prefix="/auth", tags=["auth"])


def _cookie_path(settings: Settings) -> str:
    # Cookie уходит только на эндпоинты авторизации: остальным запросам
    # refresh-токен не нужен, и чем реже он передаётся, тем меньше мест утечки
    return f"{settings.api_v1_prefix}{router.prefix}"


def set_refresh_cookie(
    response: Response, refresh: IssuedToken, settings: Settings
) -> None:
    response.set_cookie(
        REFRESH_COOKIE,
        refresh.token,
        max_age=settings.refresh_token_ttl_days * 24 * 60 * 60,
        path=_cookie_path(settings),
        # Скрипт на странице cookie не прочитает (XSS не украдёт токен), а
        # SameSite=Strict не даёт чужому сайту отправить запрос с ней (CSRF)
        httponly=True,
        secure=settings.refresh_cookie_secure,
        samesite="strict",
    )


def delete_refresh_cookie(response: Response, settings: Settings) -> None:
    # Браузер удаляет cookie, только если путь и флаги совпадают с выставленными
    response.delete_cookie(
        REFRESH_COOKIE,
        path=_cookie_path(settings),
        httponly=True,
        secure=settings.refresh_cookie_secure,
        samesite="strict",
    )


RefreshCookie = Annotated[str | None, Cookie(alias=REFRESH_COOKIE)]


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


@router.post(
    "/refresh",
    summary="Обновление access-токена",
    responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
)
async def refresh(
    response: Response,
    session: SessionDep,
    settings: SettingsDep,
    refresh_token: RefreshCookie = None,
) -> TokenResponse:
    tokens = await refresh_session(session, refresh_token, settings)
    set_refresh_cookie(response, tokens.refresh, settings)
    return TokenResponse(access_token=tokens.access.token)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT, summary="Выход")
async def log_out(
    response: Response,
    session: SessionDep,
    settings: SettingsDep,
    refresh_token: RefreshCookie = None,
) -> None:
    await logout(session, refresh_token)
    delete_refresh_cookie(response, settings)
