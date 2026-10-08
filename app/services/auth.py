from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import ahash_password
from app.models.user import User, UserRole
from app.repositories.user import UserRepository
from app.schemas.auth import RegisterRequest


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
