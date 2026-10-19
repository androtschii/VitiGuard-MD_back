import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Query, Response, status
from fastapi.concurrency import run_in_threadpool

from app.api.deps import CurrentUserDep, SessionDep, StorageDep
from app.models.diagnosis import DiseaseResult
from app.repositories.diagnosis import DiagnosisFilter, DiagnosisRepository
from app.schemas.diagnosis import (
    DiagnosisQuery,
    DiagnosisRecordRead,
    DiseaseProbability,
)
from app.schemas.errors import ErrorResponse
from app.schemas.pagination import Page

# История проверок по фото (pr-053): журнал предсказаний пользователя — для
# галереи на сайте и для будущего дообучения модели
router = APIRouter(
    prefix="/diagnoses",
    tags=["diagnoses"],
    responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
)
NOT_FOUND: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}
}


def to_record(result: DiseaseResult) -> DiagnosisRecordRead:
    vineyard = result.image.vineyard
    probabilities = sorted(
        (result.probabilities or {}).items(), key=lambda item: item[1], reverse=True
    )
    return DiagnosisRecordRead(
        id=result.id,
        created_at=result.created_at,
        updated_at=result.updated_at,
        vineyard_id=vineyard.id if vineyard else None,
        vineyard_name=vineyard.name if vineyard else None,
        model=result.model_name,
        model_version=result.model_version,
        disease=result.predicted_class or "",
        confidence=result.confidence or 0,
        top=[DiseaseProbability(disease=d, probability=p) for d, p in probabilities],
    )


@router.get("", summary="История проверок по фото")
async def list_diagnoses(
    query: Annotated[DiagnosisQuery, Query()],
    user: CurrentUserDep,
    session: SessionDep,
) -> Page[DiagnosisRecordRead]:
    items, total = await DiagnosisRepository(session).list_for_owner(
        user.id,
        query,
        DiagnosisFilter(
            query.disease, query.vineyard_id, query.date_from, query.date_to
        ),
    )
    return Page.of([to_record(item) for item in items], total, query)


@router.get(
    "/{diagnosis_id}/image",
    summary="Фото проверки",
    response_class=Response,
    responses={
        **NOT_FOUND,
        status.HTTP_200_OK: {"content": {"image/jpeg": {}, "image/png": {}}},
    },
)
async def diagnosis_image(
    diagnosis_id: uuid.UUID,
    user: CurrentUserDep,
    session: SessionDep,
    storage: StorageDep,
) -> Response:
    result = await DiagnosisRepository(session).get_for_owner(user.id, diagnosis_id)
    data = await run_in_threadpool(storage.get, result.image.storage_key)
    # Фото проверки не меняется: браузер может держать его в кэше
    return Response(
        data,
        media_type=result.image.content_type,
        headers={"Cache-Control": "private, max-age=86400"},
    )


@router.delete(
    "/{diagnosis_id}",
    summary="Удаление проверки вместе с фото",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=NOT_FOUND,
)
async def delete_diagnosis(
    diagnosis_id: uuid.UUID,
    user: CurrentUserDep,
    session: SessionDep,
    storage: StorageDep,
) -> None:
    result = await DiagnosisRepository(session).get_for_owner(user.id, diagnosis_id)
    image = result.image
    key = image.storage_key
    # Фото удаляется вместе со всеми его проверками (каскад в базе)
    await session.delete(image)
    await session.commit()
    await run_in_threadpool(storage.delete, key)
