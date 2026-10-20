import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Query, Response, status
from fastapi.concurrency import run_in_threadpool

from app.api.deps import AgronomistDep, CurrentUserDep, SessionDep, StorageDep
from app.core.errors import NotFoundError
from app.models.diagnosis import DiseaseResult
from app.repositories.diagnosis import DiagnosisFilter, DiagnosisRepository
from app.schemas.diagnosis import (
    DiagnosisQuery,
    DiagnosisRecordRead,
    DiseaseProbability,
    Overlay,
    ReviewRead,
    ReviewUpdate,
)
from app.schemas.errors import ErrorResponse
from app.schemas.pagination import Page
from app.services.diagnosis_history import overlay_key

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
    review = None
    if result.expert_class and result.reviewed_at:
        review = ReviewRead(
            disease=result.expert_class,
            comment=result.expert_comment,
            reviewed_at=result.reviewed_at,
            reviewer=(
                (result.reviewer.full_name or result.reviewer.email)
                if result.reviewer
                else None
            ),
        )
    overlays: list[Overlay] = []
    if result.heatmap_key:
        overlays.append("heatmap")
    if result.lesion_key:
        overlays.append("lesion")
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
        severity=result.severity,
        overlays=overlays,
        review=review,
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
    for file_key in (key, overlay_key(key, "heatmap"), overlay_key(key, "lesion")):
        await run_in_threadpool(storage.delete, file_key)


@router.get(
    "/{diagnosis_id}/overlays/{overlay}",
    summary="Тепловая карта или маска поражения проверки",
    response_class=Response,
    responses={**NOT_FOUND, status.HTTP_200_OK: {"content": {"image/png": {}}}},
)
async def diagnosis_overlay(
    diagnosis_id: uuid.UUID,
    overlay: Overlay,
    user: CurrentUserDep,
    session: SessionDep,
    storage: StorageDep,
) -> Response:
    result = await DiagnosisRepository(session).get_for_owner(user.id, diagnosis_id)
    key = result.heatmap_key if overlay == "heatmap" else result.lesion_key
    if key is None:
        raise NotFoundError("Наложение не сохранено")
    data = await run_in_threadpool(storage.get, key)
    return Response(
        data,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=86400"},
    )


# Заключение агронома (pr-052): подтвердить диагноз модели или исправить его.
# Агроном смотрит свои проверки — фото, которые снял в поле сам
@router.put(
    "/{diagnosis_id}/review",
    summary="Заключение агронома по проверке",
    responses={
        **NOT_FOUND,
        status.HTTP_403_FORBIDDEN: {"model": ErrorResponse},
    },
)
async def review_diagnosis(
    diagnosis_id: uuid.UUID,
    data: ReviewUpdate,
    user: AgronomistDep,
    session: SessionDep,
) -> DiagnosisRecordRead:
    repository = DiagnosisRepository(session)
    result = await repository.get_for_owner(user.id, diagnosis_id)
    result.expert_class = data.disease
    result.expert_comment = data.comment
    result.reviewed_by_id = user.id
    result.reviewed_at = datetime.now(UTC)
    await session.commit()
    return to_record(await repository.get_for_owner(user.id, diagnosis_id))


@router.delete(
    "/{diagnosis_id}/review",
    summary="Отзыв заключения агронома",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        **NOT_FOUND,
        status.HTTP_403_FORBIDDEN: {"model": ErrorResponse},
    },
)
async def delete_review(
    diagnosis_id: uuid.UUID, user: AgronomistDep, session: SessionDep
) -> None:
    result = await DiagnosisRepository(session).get_for_owner(user.id, diagnosis_id)
    result.expert_class = None
    result.expert_comment = None
    result.reviewed_by_id = None
    result.reviewed_at = None
    await session.commit()
