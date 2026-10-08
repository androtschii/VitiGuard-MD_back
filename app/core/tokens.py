import enum
import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from app.core.config import Settings
from app.core.errors import UnauthorizedError
from app.models.user import UserRole

# Алгоритм задан раз и навсегда и передаётся при проверке явно. Если брать его из
# заголовка токена, злоумышленник мог бы подсунуть «alg: none» (подпись не нужна)
# или другой алгоритм, и проверка прошла бы без знания ключа
ALGORITHM = "HS256"
ISSUER = "vitiguard-md"
REQUIRED_CLAIMS = ["exp", "iat", "sub", "iss", "type"]


class TokenType(enum.StrEnum):
    ACCESS = "access"
    REFRESH = "refresh"


@dataclass(frozen=True)
class TokenClaims:
    subject: uuid.UUID
    token_type: TokenType
    issued_at: datetime
    expires_at: datetime
    # Только у access-токена
    role: UserRole | None = None
    # Только у refresh-токена: делает каждый токен уникальным
    token_id: str | None = None


@dataclass(frozen=True)
class IssuedToken:
    token: str
    expires_at: datetime
    token_id: str


def _encode(claims: dict[str, Any], settings: Settings) -> str:
    return jwt.encode(
        claims, settings.secret_key.get_secret_value(), algorithm=ALGORITHM
    )


def create_access_token(
    user_id: uuid.UUID,
    role: UserRole,
    settings: Settings,
    now: datetime | None = None,
) -> IssuedToken:
    issued_at = now or datetime.now(UTC)
    expires_at = issued_at + timedelta(minutes=settings.access_token_ttl_minutes)
    token_id = uuid.uuid4().hex
    token = _encode(
        {
            "iss": ISSUER,
            "sub": str(user_id),
            "type": TokenType.ACCESS,
            "role": role.value,
            "jti": token_id,
            "iat": issued_at,
            "exp": expires_at,
        },
        settings,
    )
    return IssuedToken(token=token, expires_at=expires_at, token_id=token_id)


def create_refresh_token(
    user_id: uuid.UUID, settings: Settings, now: datetime | None = None
) -> IssuedToken:
    issued_at = now or datetime.now(UTC)
    expires_at = issued_at + timedelta(days=settings.refresh_token_ttl_days)
    token_id = uuid.uuid4().hex
    token = _encode(
        {
            "iss": ISSUER,
            "sub": str(user_id),
            "type": TokenType.REFRESH,
            "jti": token_id,
            "iat": issued_at,
            "exp": expires_at,
        },
        settings,
    )
    return IssuedToken(token=token, expires_at=expires_at, token_id=token_id)


def decode_token(
    token: str, expected_type: TokenType, settings: Settings
) -> TokenClaims:
    """Проверяет подпись, издателя, срок и тип токена. Любая проблема — 401."""
    try:
        payload = jwt.decode(
            token,
            settings.secret_key.get_secret_value(),
            algorithms=[ALGORITHM],
            issuer=ISSUER,
            options={"require": REQUIRED_CLAIMS},
        )
        # Токен другого типа не подходит: refresh-токен нельзя использовать как
        # access (он живёт 30 дней и лежит в cookie), и наоборот
        if payload["type"] != expected_type:
            raise ValueError("неверный тип токена")
        role = payload.get("role")
        return TokenClaims(
            subject=uuid.UUID(payload["sub"]),
            token_type=expected_type,
            issued_at=datetime.fromtimestamp(payload["iat"], UTC),
            expires_at=datetime.fromtimestamp(payload["exp"], UTC),
            role=UserRole(role) if role is not None else None,
            token_id=payload.get("jti"),
        )
    except jwt.ExpiredSignatureError as error:
        raise UnauthorizedError("Срок действия токена истёк") from error
    except (jwt.PyJWTError, ValueError, KeyError, TypeError) as error:
        # Причину наружу не отдаём: подробности проверки подсказали бы, что менять
        raise UnauthorizedError("Недействительный токен") from error


def hash_token(token: str) -> str:
    """SHA-256 токена для хранения в БД: утечка таблицы не даёт войти.

    Без соли и без медленного хэша — в отличие от паролей: токен случаен и
    содержит подпись, перебрать его нельзя."""
    return hashlib.sha256(token.encode()).hexdigest()
