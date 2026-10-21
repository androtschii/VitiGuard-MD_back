import uuid
from dataclasses import asdict

from fastapi import APIRouter, status
from sqlalchemy import select

from app.api.deps import CurrentUserDep, SessionDep
from app.core.errors import BadRequestError, NotFoundError
from app.models.treatment import Treatment
from app.repositories.vineyard import VineyardRepository
from app.schemas.errors import ErrorResponse
from app.schemas.treatment import TreatmentCreate, TreatmentPlanRead, TreatmentRead
from app.services.risk import local_today
from app.services.treatments import vineyard_plan

router = APIRouter(
    prefix="/vineyards",
    tags=["treatments"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
    },
)


@router.get("/{vineyard_id}/treatments", summary="Журнал обработок участка")
async def list_treatments(
    vineyard_id: uuid.UUID, user: CurrentUserDep, session: SessionDep
) -> list[TreatmentRead]:
    await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    rows = await session.scalars(
        select(Treatment)
        .where(Treatment.vineyard_id == vineyard_id)
        .order_by(Treatment.applied_on.desc(), Treatment.created_at.desc())
    )
    return [TreatmentRead.model_validate(row) for row in rows]


@router.post(
    "/{vineyard_id}/treatments",
    status_code=status.HTTP_201_CREATED,
    summary="Записать обработку",
    responses={status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse}},
)
async def create_treatment(
    vineyard_id: uuid.UUID,
    body: TreatmentCreate,
    user: CurrentUserDep,
    session: SessionDep,
) -> TreatmentRead:
    await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    # Журнал — сделанные обработки: запланированная защитой не считается
    if body.applied_on > local_today():
        raise BadRequestError("Дата обработки ещё не наступила")
    treatment = Treatment(vineyard_id=vineyard_id, **body.model_dump())
    session.add(treatment)
    await session.commit()
    await session.refresh(treatment)
    return TreatmentRead.model_validate(treatment)


@router.delete(
    "/{vineyard_id}/treatments/{treatment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить запись об обработке",
)
async def delete_treatment(
    vineyard_id: uuid.UUID,
    treatment_id: uuid.UUID,
    user: CurrentUserDep,
    session: SessionDep,
) -> None:
    await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    treatment = await session.get(Treatment, treatment_id)
    if treatment is None or treatment.vineyard_id != vineyard_id:
        raise NotFoundError("Обработка не найдена")
    await session.delete(treatment)
    await session.commit()


@router.get(
    "/{vineyard_id}/treatments/plan",
    summary="Когда и чем обрабатывать: рекомендации на неделю",
)
async def read_treatment_plan(
    vineyard_id: uuid.UUID, user: CurrentUserDep, session: SessionDep
) -> TreatmentPlanRead:
    await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    result = await vineyard_plan(session, vineyard_id, local_today())
    return TreatmentPlanRead.model_validate(asdict(result))
