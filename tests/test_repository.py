import uuid

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ConflictError, NotFoundError
from app.models import LeafImage, User
from app.repositories.base import Repository
from app.schemas.pagination import PageParams

pytestmark = pytest.mark.integration


class UserRepository(Repository[User]):
    model = User
    not_found_message = "Пользователь не найден"


class LeafImageRepository(Repository[LeafImage]):
    model = LeafImage


@pytest.fixture
def users(session: AsyncSession) -> UserRepository:
    return UserRepository(session)


async def add_users(users: UserRepository, count: int) -> list[User]:
    return [
        await users.create(email=f"user{n}@example.md", hashed_password="hash")
        for n in range(count)
    ]


async def test_create_returns_instance_with_database_defaults(
    users: UserRepository,
) -> None:
    user = await users.create(email="grower@example.md", hashed_password="hash")

    # Обращение к полям без запроса: id и время уже загружены
    assert user.id.version == 7
    assert user.created_at.tzinfo is not None


async def test_get_finds_record_or_returns_none(users: UserRepository) -> None:
    user = await users.create(email="grower@example.md", hashed_password="hash")

    assert await users.get(user.id) is user
    assert await users.get(uuid.uuid4()) is None


async def test_get_or_raise_returns_existing_record(users: UserRepository) -> None:
    user = await users.create(email="grower@example.md", hashed_password="hash")

    assert await users.get_or_raise(user.id) is user


async def test_get_or_raise_uses_not_found_message(users: UserRepository) -> None:
    with pytest.raises(NotFoundError, match="Пользователь не найден"):
        await users.get_or_raise(uuid.uuid4())


async def test_list_returns_page_and_total(users: UserRepository) -> None:
    created = await add_users(users, 5)

    first, total = await users.list(PageParams(page=1, size=2))
    last, _ = await users.list(PageParams(page=3, size=2))

    assert total == 5
    assert [u.id for u in first] == [created[0].id, created[1].id]
    assert [u.id for u in last] == [created[4].id]


async def test_list_filters_and_counts_matching_records(
    users: UserRepository,
) -> None:
    await add_users(users, 4)

    found, total = await users.list(
        PageParams(), User.email.in_(["user1@example.md", "user3@example.md"])
    )

    assert total == 2
    assert [u.email for u in found] == ["user1@example.md", "user3@example.md"]


async def test_list_accepts_custom_order(users: UserRepository) -> None:
    await add_users(users, 3)

    found, _ = await users.list(PageParams(), order_by=[User.email.desc()])

    assert [u.email for u in found] == [
        "user2@example.md",
        "user1@example.md",
        "user0@example.md",
    ]


async def test_list_of_empty_table(users: UserRepository) -> None:
    assert await users.list(PageParams()) == ([], 0)


async def test_update_changes_fields_and_keeps_server_values_readable(
    users: UserRepository,
) -> None:
    user = await users.create(email="grower@example.md", hashed_password="hash")

    updated = await users.update(user, full_name="Андрей")

    assert updated.full_name == "Андрей"
    assert updated.updated_at.tzinfo is not None


async def test_update_rejects_unknown_field(users: UserRepository) -> None:
    user = await users.create(email="grower@example.md", hashed_password="hash")

    with pytest.raises(ValueError, match="full_nmae"):
        await users.update(user, full_nmae="Андрей")


async def test_delete_removes_record(users: UserRepository) -> None:
    user = await users.create(email="grower@example.md", hashed_password="hash")

    await users.delete(user)

    assert await users.get(user.id) is None


async def test_unique_violation_becomes_conflict_and_session_survives(
    users: UserRepository,
) -> None:
    await users.create(email="same@example.md", hashed_password="a")

    with pytest.raises(ConflictError, match="уже существует"):
        await users.create(email="same@example.md", hashed_password="b")

    # После ошибки сессия остаётся рабочей, первая запись на месте
    assert (await users.list(PageParams()))[1] == 1


async def test_failed_update_leaves_instance_as_in_database(
    users: UserRepository,
) -> None:
    await users.create(email="taken@example.md", hashed_password="a")
    user = await users.create(email="mine@example.md", hashed_password="b")

    with pytest.raises(ConflictError):
        await users.update(user, email="taken@example.md")

    await users.session.refresh(user)
    assert user.email == "mine@example.md"


async def test_other_integrity_errors_are_not_hidden_as_conflicts(
    users: UserRepository, session: AsyncSession
) -> None:
    owner = await users.create(email="grower@example.md", hashed_password="hash")

    with pytest.raises(IntegrityError, match="ck_leaf_images_size_positive"):
        await LeafImageRepository(session).create(
            owner_id=owner.id,
            storage_key="leaves/1.jpg",
            content_type="image/jpeg",
            size_bytes=0,
            width=1,
            height=1,
        )
