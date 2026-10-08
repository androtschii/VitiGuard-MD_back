import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import ColumnElement, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import class_mapper

from app.core.errors import ConflictError, NotFoundError
from app.db.base import UUIDPrimaryKeyMixin
from app.schemas.pagination import PageParams

UNIQUE_VIOLATION = "23505"


class Repository[ModelT: UUIDPrimaryKeyMixin]:
    """Доступ к таблице одной модели: чтение, создание, изменение, удаление.

    Репозиторий только отправляет изменения в базу (flush), но не фиксирует их:
    commit вызывает код, который владеет транзакцией (сервис или эндпоинт). Так
    несколько репозиториев могут работать в одной транзакции."""

    model: type[ModelT]
    not_found_message = "Запись не найдена"
    conflict_message = "Запись с такими данными уже существует"

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, entity_id: uuid.UUID) -> ModelT | None:
        return await self.session.get(self.model, entity_id)

    async def get_or_raise(self, entity_id: uuid.UUID) -> ModelT:
        instance = await self.get(entity_id)
        if instance is None:
            raise NotFoundError(self.not_found_message)
        return instance

    async def list(
        self,
        params: PageParams,
        *where: ColumnElement[bool],
        order_by: Sequence[ColumnElement[Any]] | None = None,
    ) -> tuple[list[ModelT], int]:
        """Страница записей и общее число записей, подходящих под условия.

        Без явной сортировки записи идут по id: UUIDv7 растёт во времени, поэтому
        порядок стабильный и совпадает с порядком создания, а страницы не
        «плывут» между запросами."""
        order = order_by if order_by is not None else [self.model.id]
        rows = await self.session.scalars(
            select(self.model)
            .where(*where)
            .order_by(*order)
            .limit(params.size)
            .offset(params.offset)
        )
        total = await self.session.scalar(
            select(func.count()).select_from(self.model).where(*where)
        )
        return list(rows), total or 0

    async def create(self, **values: Any) -> ModelT:
        instance = self.model(**values)
        async with self._savepoint():
            self.session.add(instance)
        return instance

    async def update(self, instance: ModelT, **values: Any) -> ModelT:
        known = class_mapper(self.model).attrs.keys()
        if unknown := set(values) - set(known):
            # Иначе setattr молча создал бы обычный атрибут, и ошибка осталась бы незамеченной
            raise ValueError(f"У {self.model.__name__} нет полей: {sorted(unknown)}")
        async with self._savepoint():
            for name, value in values.items():
                setattr(instance, name, value)
        # updated_at выставляет база; без refresh его чтение в async-коде упало бы
        await self.session.refresh(instance)
        return instance

    async def delete(self, instance: ModelT) -> None:
        async with self._savepoint():
            await self.session.delete(instance)

    @asynccontextmanager
    async def _savepoint(self) -> AsyncIterator[None]:
        """Изменения внутри блока отправляются в базу при выходе из него.

        Они делаются внутри savepoint, поэтому при ошибке откатывается только он:
        добавленные объекты убираются из сессии, изменённые возвращаются к
        состоянию в базе, и сессия остаётся рабочей. Если flush упал не внутри
        savepoint, сессия остаётся в состоянии «ожидает отката»."""
        try:
            async with self.session.begin_nested():
                yield
        except IntegrityError as error:
            if getattr(error.orig, "sqlstate", None) == UNIQUE_VIOLATION:
                raise ConflictError(self.conflict_message) from error
            raise
