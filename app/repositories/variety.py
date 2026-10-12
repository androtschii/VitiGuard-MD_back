from sqlalchemy import select

from app.models.variety import Variety
from app.repositories.base import Repository


class VarietyRepository(Repository[Variety]):
    model = Variety
    not_found_message = "Сорт не найден"
    conflict_message = "Сорт с таким названием уже есть"

    async def all(self) -> list[Variety]:
        # Справочник — десятки сортов: отдаётся целиком, по алфавиту
        rows = await self.session.scalars(select(Variety).order_by(Variety.name))
        return list(rows)
