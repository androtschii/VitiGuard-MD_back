from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Query, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.errors import UnauthorizedError
from app.core.tokens import TokenType, decode_token
from app.models.user import User
from app.repositories.user import UserRepository
from app.schemas.pagination import PageParams


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


SettingsDep = Annotated[Settings, Depends(get_app_settings)]


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Сессия на один запрос. Изменения не сохраняются сами: commit вызывает
    код, который их сделал. При ошибке и в конце запроса всё несохранённое
    откатывается, а соединение возвращается в пул."""
    session_factory: async_sessionmaker[AsyncSession] = (
        request.app.state.session_factory
    )
    async with session_factory() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]

# Параметры страницы берутся из строки запроса: ?page=2&size=20
PageParamsDep = Annotated[PageParams, Query()]

# auto_error=False: без заголовка FastAPI ответил бы 403 без WWW-Authenticate, а
# по стандарту отсутствие учётных данных — это 401
bearer_scheme = HTTPBearer(
    auto_error=False,
    description="Access-токен из ответа POST /auth/login или /auth/refresh",
)
BearerCredentials = Annotated[
    HTTPAuthorizationCredentials | None, Security(bearer_scheme)
]


async def get_current_user(
    credentials: BearerCredentials, session: SessionDep, settings: SettingsDep
) -> User:
    """Пользователь, которому выдан access-токен из заголовка Authorization.

    Пользователь читается из базы на каждый запрос, а не берётся из токена:
    отключённый или удалённый пользователь теряет доступ сразу, а не через
    15 минут, когда истечёт токен, и новая роль действует тоже сразу."""
    if credentials is None:
        raise UnauthorizedError("Требуется вход")
    claims = decode_token(credentials.credentials, TokenType.ACCESS, settings)
    user = await UserRepository(session).get(claims.subject)
    if user is None or not user.is_active:
        raise UnauthorizedError("Недействительный токен")
    return user


CurrentUserDep = Annotated[User, Depends(get_current_user)]
