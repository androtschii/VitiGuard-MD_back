"""Сервис авторизации без базы: репозитории заменены хранилищем в памяти, а
пароли хэшируются настоящим Argon2."""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import BadRequestError, ForbiddenError, UnauthorizedError
from app.core.security import hash_password
from app.core.tokens import TokenType, decode_token, hash_token
from app.models.auth import PasswordResetToken, RefreshToken
from app.models.user import User, UserRole
from app.schemas.auth import (
    LoginRequest,
    PasswordResetConfirm,
    PasswordResetRequest,
    RegisterRequest,
)
from app.services import auth

PASSWORD = "secret123"
EMAIL = "grower@example.md"


@dataclass
class Store:
    users: dict[uuid.UUID, User] = field(default_factory=dict)
    refresh: list[RefreshToken] = field(default_factory=list)
    resets: list[PasswordResetToken] = field(default_factory=list)
    commits: int = 0
    mail: list[tuple[str, str]] = field(default_factory=list)

    def send_mail(self, to: str, link: str) -> None:
        self.mail.append((to, link))


class FakeSession:
    def __init__(self, store: Store) -> None:
        self.store = store

    async def commit(self) -> None:
        self.store.commits += 1


class FakeUsers:
    def __init__(self, session: FakeSession) -> None:
        self.store = session.store

    async def create(self, **values: Any) -> User:
        user = User(id=uuid.uuid4(), is_active=True, **values)
        self.store.users[user.id] = user
        return user

    async def get(self, user_id: uuid.UUID) -> User | None:
        return self.store.users.get(user_id)

    async def get_by_email(self, email: str) -> User | None:
        users = self.store.users.values()
        return next((user for user in users if user.email == email.lower()), None)

    async def update(self, user: User, **values: Any) -> User:
        for name, value in values.items():
            setattr(user, name, value)
        return user


class FakeRefreshTokens:
    def __init__(self, session: FakeSession) -> None:
        self.store = session.store

    async def create(self, **values: Any) -> RefreshToken:
        token = RefreshToken(id=uuid.uuid4(), revoked_at=None, **values)
        self.store.refresh.append(token)
        return token

    async def get_by_hash(
        self, token_hash: str, *, for_update: bool = False
    ) -> RefreshToken | None:
        tokens = self.store.refresh
        return next((token for token in tokens if token.token_hash == token_hash), None)

    async def revoke(self, token: RefreshToken, now: datetime) -> None:
        token.revoked_at = now

    async def revoke_all_for_user(self, user_id: uuid.UUID, now: datetime) -> None:
        for token in self.store.refresh:
            if token.user_id == user_id and token.revoked_at is None:
                token.revoked_at = now


class FakeResetTokens:
    def __init__(self, session: FakeSession) -> None:
        self.store = session.store

    async def create(self, **values: Any) -> PasswordResetToken:
        token = PasswordResetToken(id=uuid.uuid4(), used_at=None, **values)
        self.store.resets.append(token)
        return token

    async def get_by_hash_for_update(
        self, token_hash: str
    ) -> PasswordResetToken | None:
        tokens = self.store.resets
        return next((token for token in tokens if token.token_hash == token_hash), None)

    async def use_all_for_user(self, user_id: uuid.UUID, now: datetime) -> None:
        for token in self.store.resets:
            if token.user_id == user_id and token.used_at is None:
                token.used_at = now


@pytest.fixture
def store() -> Store:
    return Store()


@pytest.fixture
def session(store: Store, monkeypatch: pytest.MonkeyPatch) -> AsyncSession:
    monkeypatch.setattr(auth, "UserRepository", FakeUsers)
    monkeypatch.setattr(auth, "RefreshTokenRepository", FakeRefreshTokens)
    monkeypatch.setattr(auth, "PasswordResetTokenRepository", FakeResetTokens)
    return cast(AsyncSession, FakeSession(store))


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, environment="test")


def add_user(store: Store, *, active: bool = True) -> User:
    user = User(
        id=uuid.uuid4(),
        email=EMAIL,
        hashed_password=hash_password(PASSWORD),
        role=UserRole.USER,
        is_active=active,
    )
    store.users[user.id] = user
    return user


def credentials(password: str = PASSWORD) -> LoginRequest:
    return LoginRequest(email="Grower@Example.md", password=password)


async def test_register_hashes_password_and_commits(
    session: AsyncSession, store: Store
) -> None:
    user = await auth.register_user(
        session,
        RegisterRequest(
            email="New@Example.md",
            password=PASSWORD,
            full_name="Ион",
            role="agronomist",
        ),
    )

    assert user.email == "new@example.md"
    assert user.role is UserRole.AGRONOMIST
    assert user.hashed_password.startswith("$argon2id$")
    assert store.commits == 1


async def test_login_issues_tokens_and_stores_refresh_hash(
    session: AsyncSession, store: Store, settings: Settings
) -> None:
    user = add_user(store)

    result = await auth.login(session, credentials(), settings)

    access = decode_token(result.access.token, TokenType.ACCESS, settings)
    assert access.subject == user.id
    [stored] = store.refresh
    # В базе только хэш: сам токен знает лишь клиент
    assert stored.token_hash == hash_token(result.refresh.token)
    assert stored.token_hash != result.refresh.token
    assert store.commits == 1


@pytest.mark.parametrize("known", [True, False], ids=["wrong-password", "unknown"])
async def test_login_failure_does_not_reveal_email(
    session: AsyncSession, store: Store, settings: Settings, known: bool
) -> None:
    if known:
        add_user(store)

    with pytest.raises(UnauthorizedError) as error:
        await auth.login(session, credentials("wrong-pass1"), settings)

    assert error.value.detail == auth.INVALID_CREDENTIALS
    assert store.refresh == []


async def test_disabled_account_is_reported_only_with_password(
    session: AsyncSession, store: Store, settings: Settings
) -> None:
    add_user(store, active=False)

    with pytest.raises(ForbiddenError):
        await auth.login(session, credentials(), settings)
    with pytest.raises(UnauthorizedError):
        await auth.login(session, credentials("wrong-pass1"), settings)


async def test_outdated_hash_is_upgraded_on_login(
    session: AsyncSession,
    store: Store,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = add_user(store)

    async def outdated(_: str, __: str | None) -> tuple[bool, str | None]:
        return True, "$argon2id$new"

    monkeypatch.setattr(auth, "averify_and_upgrade", outdated)

    await auth.login(session, credentials(), settings)

    assert user.hashed_password == "$argon2id$new"


async def test_refresh_rotates_tokens(
    session: AsyncSession, store: Store, settings: Settings
) -> None:
    add_user(store)
    first = await auth.login(session, credentials(), settings)

    second = await auth.refresh_session(session, first.refresh.token, settings)

    old, new = store.refresh
    assert old.revoked_at is not None
    assert new.revoked_at is None
    assert new.token_hash == hash_token(second.refresh.token)


async def test_reused_refresh_token_revokes_all_sessions(
    session: AsyncSession, store: Store, settings: Settings
) -> None:
    add_user(store)
    stolen = await auth.login(session, credentials(), settings)
    await auth.login(session, credentials(), settings)
    await auth.refresh_session(session, stolen.refresh.token, settings)

    with pytest.raises(UnauthorizedError):
        await auth.refresh_session(session, stolen.refresh.token, settings)

    assert all(token.revoked_at for token in store.refresh)


async def test_refresh_rejects_missing_and_unknown_tokens(
    session: AsyncSession, store: Store, settings: Settings
) -> None:
    add_user(store)
    issued = await auth.login(session, credentials(), settings)

    for token in (None, ""):
        with pytest.raises(UnauthorizedError):
            await auth.refresh_session(session, token, settings)
    store.refresh.clear()
    with pytest.raises(UnauthorizedError):
        await auth.refresh_session(session, issued.refresh.token, settings)


async def test_refresh_rejects_disabled_user_and_foreign_token(
    session: AsyncSession, store: Store, settings: Settings
) -> None:
    user = add_user(store)
    disabled = await auth.login(session, credentials(), settings)
    foreign = await auth.login(session, credentials(), settings)
    store.refresh[1].user_id = uuid.uuid4()
    user.is_active = False

    for issued in (disabled, foreign):
        with pytest.raises(UnauthorizedError):
            await auth.refresh_session(session, issued.refresh.token, settings)


async def test_logout_revokes_and_is_idempotent(
    session: AsyncSession, store: Store, settings: Settings
) -> None:
    add_user(store)
    issued = await auth.login(session, credentials(), settings)

    await auth.logout(session, issued.refresh.token)
    revoked_at = store.refresh[0].revoked_at
    await auth.logout(session, issued.refresh.token)
    await auth.logout(session, None)

    assert revoked_at is not None
    assert store.refresh[0].revoked_at == revoked_at


async def test_password_reset_keeps_only_latest_link(
    session: AsyncSession, store: Store, settings: Settings
) -> None:
    add_user(store)
    request = PasswordResetRequest(email=EMAIL)

    await auth.request_password_reset(session, request, settings, store.send_mail)
    await auth.request_password_reset(session, request, settings, store.send_mail)

    assert [to for to, _ in store.mail] == [EMAIL, EMAIL]
    assert store.mail[1][1].startswith(f"{settings.frontend_url}/reset-password?token=")
    first, second = store.resets
    assert first.used_at is not None
    assert second.used_at is None


async def test_password_reset_for_unknown_or_disabled_is_silent(
    session: AsyncSession, store: Store, settings: Settings
) -> None:
    add_user(store, active=False)

    for email in ("nobody@example.md", EMAIL):
        await auth.request_password_reset(
            session, PasswordResetRequest(email=email), settings, store.send_mail
        )

    assert store.mail == []
    assert store.resets == []


async def reset_token(session: AsyncSession, store: Store, settings: Settings) -> str:
    await auth.request_password_reset(
        session, PasswordResetRequest(email=EMAIL), settings, store.send_mail
    )
    return store.mail[-1][1].rsplit("token=", 1)[1]


async def test_confirm_reset_changes_password_and_logs_out(
    session: AsyncSession, store: Store, settings: Settings
) -> None:
    user = add_user(store)
    old_hash = user.hashed_password
    await auth.login(session, credentials(), settings)
    token = await reset_token(session, store, settings)

    await auth.confirm_password_reset(
        session, PasswordResetConfirm(token=token, new_password="new-secret9")
    )

    assert user.hashed_password != old_hash
    assert store.resets[0].used_at is not None
    assert all(item.revoked_at for item in store.refresh)
    # Ключ одноразовый
    with pytest.raises(BadRequestError):
        await auth.confirm_password_reset(
            session, PasswordResetConfirm(token=token, new_password="other-secret9")
        )


async def test_confirm_reset_rejects_unknown_expired_and_disabled(
    session: AsyncSession, store: Store, settings: Settings
) -> None:
    user = add_user(store)
    confirm = PasswordResetConfirm(
        token=await reset_token(session, store, settings), new_password="new-secret9"
    )

    with pytest.raises(BadRequestError):
        await auth.confirm_password_reset(
            session, PasswordResetConfirm(token="unknown", new_password="new-secret9")
        )
    store.resets[0].expires_at = datetime.now(UTC) - timedelta(seconds=1)
    with pytest.raises(BadRequestError):
        await auth.confirm_password_reset(session, confirm)

    store.resets[0].expires_at = datetime.now(UTC) + timedelta(hours=1)
    user.is_active = False
    with pytest.raises(BadRequestError):
        await auth.confirm_password_reset(session, confirm)
