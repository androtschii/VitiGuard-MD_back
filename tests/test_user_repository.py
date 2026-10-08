import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.user import UserRepository

pytestmark = pytest.mark.integration


async def test_get_by_email_finds_user_in_any_letter_case(
    session: AsyncSession,
) -> None:
    users = UserRepository(session)
    user = await users.create(email="grower@example.md", hashed_password="hash")

    assert await users.get_by_email("grower@example.md") is user
    assert await users.get_by_email("Grower@Example.MD") is user


async def test_get_by_email_returns_none_for_unknown_address(
    session: AsyncSession,
) -> None:
    assert await UserRepository(session).get_by_email("nobody@example.md") is None
