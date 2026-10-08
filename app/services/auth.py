from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import ForbiddenError, UnauthorizedError
from app.core.security import ahash_password, averify_and_upgrade
from app.core.tokens import (
    IssuedToken,
    create_access_token,
    create_refresh_token,
    hash_token,
)
from app.models.user import User, UserRole
from app.repositories.auth import RefreshTokenRepository
from app.repositories.user import UserRepository
from app.schemas.auth import LoginRequest, RegisterRequest

# Одно сообщение и для неизвестного email, и для неверного пароля: иначе по ответу
# можно проверять, зарегистрирован ли адрес
INVALID_CREDENTIALS = "Неверный email или пароль"


@dataclass(frozen=True)
class AuthSession:
    access: IssuedToken
    refresh: IssuedToken


async def register_user(session: AsyncSession, data: RegisterRequest) -> User:
    """Создаёт пользователя. Занятый email даёт ConflictError (ответ 409).

    Проверка «email свободен» отдельным запросом не нужна и была бы неверной: два
    одновременных запроса прошли бы её оба. Уникальность гарантирует база, а
    репозиторий превращает её нарушение в ConflictError."""
    hashed_password = await ahash_password(data.password)
    user = await UserRepository(session).create(
        email=data.email,
        hashed_password=hashed_password,
        full_name=data.full_name,
        role=UserRole(data.role),
    )
    # Транзакцию фиксирует сервис, которому принадлежит операция (репозиторий
    # только отправляет изменения в базу)
    await session.commit()
    return user


async def login(
    session: AsyncSession, data: LoginRequest, settings: Settings
) -> AuthSession:
    """Проверяет email и пароль и открывает сессию: access-токен и refresh-токен,
    хэш которого сохраняется в базе, чтобы его можно было отозвать."""
    users = UserRepository(session)
    user = await users.get_by_email(data.email)
    # Проверка выполняется и для неизвестного email (на заглушке): время ответа
    # не должно выдавать, есть ли такой пользователь
    valid, new_hash = await averify_and_upgrade(
        data.password, user.hashed_password if user else None
    )
    if user is None or not valid:
        raise UnauthorizedError(INVALID_CREDENTIALS)
    # Об отключённой учётной записи сообщаем только тому, кто знает пароль
    if not user.is_active:
        raise ForbiddenError("Учётная запись отключена")
    if new_hash is not None:
        # Хэш сделан со старыми параметрами Argon2: заменяем на актуальный
        await users.update(user, hashed_password=new_hash)

    access = create_access_token(user.id, user.role, settings)
    refresh = create_refresh_token(user.id, settings)
    await RefreshTokenRepository(session).create(
        user_id=user.id,
        token_hash=hash_token(refresh.token),
        expires_at=refresh.expires_at,
    )
    await session.commit()
    return AuthSession(access=access, refresh=refresh)
