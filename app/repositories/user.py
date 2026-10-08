from sqlalchemy import select

from app.models.user import User
from app.repositories.base import Repository


class UserRepository(Repository[User]):
    model = User
    not_found_message = "Пользователь не найден"
    conflict_message = "Пользователь с таким email уже зарегистрирован"

    async def get_by_email(self, email: str) -> User | None:
        # Email в базе хранится в нижнем регистре, поэтому ищем по нему же
        return await self.session.scalar(
            select(User).where(User.email == email.lower())
        )
