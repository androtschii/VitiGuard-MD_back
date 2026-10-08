import math
from collections.abc import Sequence
from typing import Self

from pydantic import Field, computed_field

from app.schemas.base import RequestSchema, ResponseSchema

MAX_PAGE_SIZE = 100


class PageParams(RequestSchema):
    """Параметры постраничной выборки: ?page=2&size=20."""

    page: int = Field(default=1, ge=1)
    # Верхняя граница защищает базу от запроса «всех записей сразу»
    size: int = Field(default=20, ge=1, le=MAX_PAGE_SIZE)

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size


class Page[T](ResponseSchema):
    items: list[T]
    total: int = Field(ge=0)
    page: int = Field(ge=1)
    size: int = Field(ge=1)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pages(self) -> int:
        return math.ceil(self.total / self.size)

    @classmethod
    def of(cls, items: Sequence[T], total: int, params: PageParams) -> Self:
        return cls(items=list(items), total=total, page=params.page, size=params.size)
