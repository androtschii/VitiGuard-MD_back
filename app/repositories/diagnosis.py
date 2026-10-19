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


def _start(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=UTC)


class DiagnosisRepository(Repository[DiseaseResult]):
    model = DiseaseResult
    not_found_message = "Проверка не найдена"

    def _owned(self, owner_id: uuid.UUID) -> Select[DiseaseResult]:
        # Фото и участок подгружаются сразу для всей страницы
        return (
            select(DiseaseResult)
            .join(DiseaseResult.image)
            .where(LeafImage.owner_id == owner_id)
            .options(selectinload(DiseaseResult.image).selectinload(LeafImage.vineyard))
        )

    async def list_for_owner(
        self, owner_id: uuid.UUID, params: PageParams, where: DiagnosisFilter
    ) -> tuple[list[DiseaseResult], int]:
        query = self._owned(owner_id)
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
