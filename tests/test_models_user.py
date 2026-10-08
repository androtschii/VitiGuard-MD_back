from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import RefreshToken, User, UserRole

pytestmark = pytest.mark.integration


async def test_user_defaults(session: AsyncSession) -> None:
    user = User(email="grower@example.md", hashed_password="hash")
    session.add(user)
    await session.flush()
    await session.refresh(user)

    assert user.id.version == 7
    assert user.role is UserRole.USER
    assert user.is_active is True
    assert user.created_at.tzinfo is not None


async def test_email_is_unique(session: AsyncSession) -> None:
    session.add_all(
        [
            User(email="same@example.md", hashed_password="a"),
            User(email="same@example.md", hashed_password="b"),
        ]
    )

    with pytest.raises(IntegrityError, match="uq_users_email"):
        await session.flush()


async def test_unknown_role_rejected(session: AsyncSession) -> None:
    with pytest.raises(IntegrityError, match="ck_users_user_role"):
        await session.execute(
            text(
                "INSERT INTO users (email, hashed_password, role)"
                " VALUES ('root@example.md', 'hash', 'superuser')"
            )
        )


async def test_refresh_tokens_deleted_with_user(session: AsyncSession) -> None:
    user = User(email="token@example.md", hashed_password="hash")
    user.refresh_tokens.append(
        RefreshToken(
            token_hash="a" * 64, expires_at=datetime.now(UTC) + timedelta(days=30)
        )
    )
    session.add(user)
    await session.flush()

    await session.execute(delete(User).where(User.id == user.id))

    count = await session.scalar(select(func.count()).select_from(RefreshToken))
    assert count == 0


async def test_email_must_be_lower_case(session: AsyncSession) -> None:
    session.add(User(email="Grower@Example.MD", hashed_password="hash"))

    with pytest.raises(IntegrityError, match="ck_users_email_lowercase"):
        await session.flush()
