import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

from sqlalchemy import Select, func, select
from sqlalchemy.orm import selectinload

from app.core.errors import NotFoundError
from app.models.diagnosis import DiseaseResult, LeafImage
from app.repositories.base import Repository
from app.schemas.pagination import PageParams


@dataclass(frozen=True)
class DiagnosisFilter:
    disease: str | None = None
    vineyard_id: uuid.UUID | None = None
    # Границы включительно, по календарным дням UTC
    date_from: date | None = None
    date_to: date | None = None
    # Только для очереди агронома: True — с заключением, False — без него
    reviewed: bool | None = None


def _start(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=UTC)


class DiagnosisRepository(Repository[DiseaseResult]):
    model = DiseaseResult
    not_found_message = "Проверка не найдена"

    def _all(self) -> Select[DiseaseResult]:
        # Фото, участок и владелец подгружаются сразу для всей страницы
        return (
            select(DiseaseResult)
            .join(DiseaseResult.image)
            .options(
                selectinload(DiseaseResult.image).selectinload(LeafImage.vineyard),
                selectinload(DiseaseResult.image).selectinload(LeafImage.owner),
                selectinload(DiseaseResult.reviewer),
            )
            # Заключение могло измениться в этой же сессии
            .execution_options(populate_existing=True)
        )

    def _owned(self, owner_id: uuid.UUID) -> Select[DiseaseResult]:
        return self._all().where(LeafImage.owner_id == owner_id)

    async def list_for_owner(
        self, owner_id: uuid.UUID, params: PageParams, where: DiagnosisFilter
    ) -> tuple[list[DiseaseResult], int]:
        return await self._page(self._owned(owner_id), params, where)

    async def list_all(
        self, params: PageParams, where: DiagnosisFilter
    ) -> tuple[list[DiseaseResult], int]:
        """Проверки всех пользователей — очередь агронома."""
        return await self._page(self._all(), params, where)

    async def _page(
        self, query: Select[DiseaseResult], params: PageParams, where: DiagnosisFilter
    ) -> tuple[list[DiseaseResult], int]:
        if where.reviewed is not None:
            query = query.where(
                DiseaseResult.reviewed_at.is_not(None)
                if where.reviewed
                else DiseaseResult.reviewed_at.is_(None)
            )
        if where.disease:
            query = query.where(DiseaseResult.predicted_class == where.disease)
        if where.vineyard_id:
            query = query.where(LeafImage.vineyard_id == where.vineyard_id)
        if where.date_from:
            query = query.where(DiseaseResult.created_at >= _start(where.date_from))
        if where.date_to:
            query = query.where(
                DiseaseResult.created_at < _start(where.date_to + timedelta(days=1))
            )
        total = await self.session.scalar(
            select(func.count()).select_from(query.subquery())
        )
        rows = await self.session.scalars(
            query.order_by(DiseaseResult.created_at.desc(), DiseaseResult.id.desc())
            .offset(params.offset)
            .limit(params.size)
        )
        return list(rows), total or 0

    async def get_for_owner(
        self, owner_id: uuid.UUID, result_id: uuid.UUID
    ) -> DiseaseResult:
        """Чужая проверка неотличима от несуществующей, как и чужой участок."""
        found = await self.session.scalar(
            self._owned(owner_id).where(DiseaseResult.id == result_id)
        )
        if found is None:
            raise NotFoundError(self.not_found_message)
        return found

    async def get_any(self, result_id: uuid.UUID) -> DiseaseResult:
        """Проверка любого пользователя — для агронома и администратора."""
        found = await self.session.scalar(
            self._all().where(DiseaseResult.id == result_id)
        )
        if found is None:
            raise NotFoundError(self.not_found_message)
        return found
