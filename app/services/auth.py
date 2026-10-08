import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import BadRequestError, ForbiddenError, UnauthorizedError
from app.core.security import ahash_password, averify_and_upgrade
from app.core.tokens import (
    IssuedToken,
    TokenType,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_token,
)
from app.models.user import User, UserRole
from app.repositories.auth import PasswordResetTokenRepository, RefreshTokenRepository
from app.repositories.user import UserRepository
from app.schemas.auth import (
    LoginRequest,
    PasswordResetConfirm,
    PasswordResetRequest,
    RegisterRequest,
)

# Одно сообщение и для неизвестного email, и для неверного пароля: иначе по ответу
# можно проверять, зарегистрирован ли адрес
INVALID_CREDENTIALS = "Неверный email или пароль"
INVALID_SESSION = "Сессия недействительна, войдите заново"
INVALID_RESET_LINK = "Ссылка недействительна или устарела, запросите новую"

# Отправка письма со ссылкой: (email, ссылка). В приложении ставит задачу Celery
ResetMailer = Callable[[str, str], None]


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

    auth_session = await _issue_tokens(session, user, settings)
    await session.commit()
    return auth_session


async def refresh_session(
    session: AsyncSession, refresh_token: str | None, settings: Settings
) -> AuthSession:
    """Ротация: старый refresh-токен отзывается, выдаётся новая пара токенов.

    Отозванный токен, пришедший повторно, означает, что его копия у кого-то
    ещё: настоящий владелец уже получил новый. Какая из сторон злоумышленник,
    не понять, поэтому отзываются все сессии пользователя — войти заново
    сможет только тот, кто знает пароль."""
    if not refresh_token:
        raise UnauthorizedError(INVALID_SESSION)
    claims = decode_token(refresh_token, TokenType.REFRESH, settings)
    tokens = RefreshTokenRepository(session)
    stored = await tokens.get_by_hash(hash_token(refresh_token), for_update=True)
    if stored is None or stored.user_id != claims.subject:
        raise UnauthorizedError(INVALID_SESSION)

    now = datetime.now(UTC)
    if stored.revoked_at is not None:
        await tokens.revoke_all_for_user(stored.user_id, now)
        # Отзыв фиксируется до ответа с ошибкой, иначе он откатится вместе с запросом
        await session.commit()
        raise UnauthorizedError(INVALID_SESSION)

    user = await UserRepository(session).get(stored.user_id)
    if user is None or not user.is_active:
        raise UnauthorizedError(INVALID_SESSION)

    await tokens.revoke(stored, now)
    auth_session = await _issue_tokens(session, user, settings)
    await session.commit()
    return auth_session


async def logout(session: AsyncSession, refresh_token: str | None) -> None:
    """Отзывает refresh-токен. Повторный выход и выход без токена — не ошибка:
    результат тот же, сессии больше нет."""
    if not refresh_token:
        return
    tokens = RefreshTokenRepository(session)
    stored = await tokens.get_by_hash(hash_token(refresh_token))
    if stored is not None and stored.revoked_at is None:
        await tokens.revoke(stored, datetime.now(UTC))
        await session.commit()


async def _issue_tokens(
    session: AsyncSession, user: User, settings: Settings
) -> AuthSession:
    access = create_access_token(user.id, user.role, settings)
    refresh = create_refresh_token(user.id, settings)
    await RefreshTokenRepository(session).create(
        user_id=user.id,
        token_hash=hash_token(refresh.token),
        expires_at=refresh.expires_at,
    )
    return AuthSession(access=access, refresh=refresh)


async def request_password_reset(
    session: AsyncSession,
    data: PasswordResetRequest,
    settings: Settings,
    send_mail: ResetMailer,
) -> None:
    """Отправляет ссылку для сброса пароля, если такой активный пользователь есть.

    Вызывающему ничего не сообщается: ответ одинаков для любого адреса, иначе
    форму можно было бы использовать для проверки, зарегистрирован ли email."""
    user = await UserRepository(session).get_by_email(data.email)
    if user is None or not user.is_active:
        return
    now = datetime.now(UTC)
    tokens = PasswordResetTokenRepository(session)
    # Действует только последняя ссылка: старые письма больше не помогут
    await tokens.use_all_for_user(user.id, now)
    token = secrets.token_urlsafe(32)
    await tokens.create(
        user_id=user.id,
        token_hash=hash_token(token),
        expires_at=now + timedelta(minutes=settings.password_reset_ttl_minutes),
    )
    await session.commit()
    send_mail(user.email, f"{settings.frontend_url}/reset-password?token={token}")


async def confirm_password_reset(
    session: AsyncSession, data: PasswordResetConfirm
) -> None:
    """Задаёт новый пароль по ключу из письма. Ключ одноразовый; после смены
    пароля пользователь выходит на всех устройствах — если пароль меняли из-за
    взлома, чужие сессии тоже закрываются."""
    tokens = PasswordResetTokenRepository(session)
    stored = await tokens.get_by_hash_for_update(hash_token(data.token))
    now = datetime.now(UTC)
    if stored is None or stored.used_at is not None or stored.expires_at <= now:
        raise BadRequestError(INVALID_RESET_LINK)
    users = UserRepository(session)
    user = await users.get(stored.user_id)
    if user is None or not user.is_active:
        raise BadRequestError(INVALID_RESET_LINK)

    await users.update(user, hashed_password=await ahash_password(data.new_password))
    await tokens.use_all_for_user(user.id, now)
    await RefreshTokenRepository(session).revoke_all_for_user(user.id, now)
    await session.commit()
