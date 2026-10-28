import uuid
from typing import Annotated, Any, Literal

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, File, Query, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from app.api.deps import CurrentUserDep, SessionDep, SettingsDep, StorageDep
from app.api.rate_limit import DIAGNOSE_LIMIT, per_user
from app.core.errors import (
    NotFoundError,
    PayloadTooLargeError,
    ServiceUnavailableError,
)
from app.ml.classifier import LeafClassifier
from app.ml.model_file import ModelFile, describe_model
from app.ml.segmentation import LesionSegmenter
from app.repositories.vineyard import VineyardRepository
from app.schemas.diagnosis import (
    ClassifierModelRead,
    DiagnosisJobRead,
    DiagnosisJobStarted,
    DiagnosisRead,
)
from app.schemas.errors import ErrorResponse
from app.services.diagnosis import analyze_leaf, photo_location, read_leaf_image
from app.services.diagnosis_history import EXTENSIONS, save_diagnosis
from app.tasks.diagnosis import DIAGNOSE
from app.worker.celery_app import celery_app

router = APIRouter(
    prefix="/ai",
    tags=["ai"],
    responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
)

MAX_IMAGE_BYTES = 10 * 1024 * 1024


def get_classifier(request: Request, settings: SettingsDep) -> LeafClassifier:
    """Модель загружается при первом запросе и остаётся в памяти процесса."""
    classifier: LeafClassifier | None = getattr(request.app.state, "classifier", None)
    if classifier is None:
        if not settings.diagnosis_model_path.exists():
            raise ServiceUnavailableError("Модель диагностики не загружена на сервер")
        classifier = LeafClassifier(settings.diagnosis_model_path)
        request.app.state.classifier = classifier
    return classifier


def get_segmenter(request: Request, settings: SettingsDep) -> LesionSegmenter | None:
    """Сегментация необязательна: без её модели диагноз ставится, но без маски."""
    segmenter: LesionSegmenter | None = getattr(request.app.state, "segmenter", None)
    if segmenter is None and settings.lesion_model_path.exists():
        segmenter = LesionSegmenter(settings.lesion_model_path)
        request.app.state.segmenter = segmenter
    return segmenter


def classifier_file(settings: SettingsDep) -> ModelFile:
    path = settings.diagnosis_model_path
    if not path.exists():
        raise ServiceUnavailableError("Модель диагностики не загружена на сервер")
    return describe_model(path)


MODEL_RESPONSES: dict[int | str, dict[str, object]] = {
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse}
}


# Браузер скачивает классификатор один раз и ставит диагноз без сети
# (frontend/pr-071). Версия — начало SHA-256: по ней клиент понимает, что
# сохранённая копия устарела
@router.get(
    "/models/classifier",
    summary="Описание классификатора для работы без сети",
    responses=MODEL_RESPONSES,
)
async def classifier_model(
    settings: SettingsDep, _: CurrentUserDep
) -> ClassifierModelRead:
    model = await run_in_threadpool(classifier_file, settings)
    return ClassifierModelRead(**model.__dict__)


@router.get(
    "/models/classifier/file",
    summary="Файл классификатора (ONNX)",
    response_class=FileResponse,
    responses=MODEL_RESPONSES,
)
async def classifier_model_file(
    settings: SettingsDep, _: CurrentUserDep
) -> FileResponse:
    model = await run_in_threadpool(classifier_file, settings)
    return FileResponse(
        settings.diagnosis_model_path,
        media_type="application/octet-stream",
        filename=f"{model.name}-{model.version}.onnx",
        headers={
            "ETag": f'"{model.version}"',
            "Cache-Control": "private, max-age=86400",
        },
    )


@router.post(
    "/diagnose",
    dependencies=[Depends(per_user(DIAGNOSE_LIMIT))],
    summary="Определение болезни по фото листа",
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_413_CONTENT_TOO_LARGE: {"model": ErrorResponse},
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"model": ErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
    },
)
async def diagnose(
    image: Annotated[
        UploadFile, File(description="Фото листа: JPEG, PNG или WebP до 10 МБ")
    ],
    request: Request,
    settings: SettingsDep,
    user: CurrentUserDep,
    session: SessionDep,
    storage: StorageDep,
    top_k: Annotated[int, Query(ge=1, le=6, description="Сколько классов вернуть")] = 3,
    save: Annotated[
        bool, Query(description="Сохранить фото и результат в историю проверок")
    ] = False,
    vineyard_id: Annotated[
        uuid.UUID | None, Query(description="Участок, на котором снято фото")
    ] = None,
) -> DiagnosisRead:
    data = await image.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise PayloadTooLargeError("Файл больше 10 МБ")
    leaf = read_leaf_image(data)
    classifier = get_classifier(request, settings)
    segmenter = get_segmenter(request, settings)
    if save and vineyard_id is not None:
        # Только свой участок: чужой — как несуществующий
        await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    # Модели занимают процессор на десятки и сотни миллисекунд: в отдельном
    # потоке они не задерживают обработку других запросов
    diagnosis = await run_in_threadpool(
        analyze_leaf, leaf, classifier, segmenter, top_k
    )
    diagnosis.location = photo_location(data)
    # Снимок не похож на лист винограда — в историю он не попадает: там он
    # выглядел бы как настоящая находка болезни
    if save and diagnosis.grape_leaf:
        model = await run_in_threadpool(classifier_file, settings)
        diagnosis.id = await save_diagnosis(
            session,
            storage,
            user.id,
            vineyard_id,
            data,
            leaf,
            diagnosis,
            model.version,
        )
    return diagnosis


def job_state(task_id: str) -> tuple[str, Any]:
    """Состояние задачи и её результат из Redis (синхронный вызов)."""
    result: AsyncResult[Any] = AsyncResult(task_id, app=celery_app)
    return result.state, result.result if result.state == "SUCCESS" else None


JobStatus = Literal["pending", "running", "done", "failed"]
JOB_STATUSES: dict[str, JobStatus] = {"PENDING": "pending", "FAILURE": "failed"}


# Снимок с телефона в 10 МБ анализируется секунды: запрос не держит соединение,
# а ставит задачу воркеру, и клиент спрашивает о результате
@router.post(
    "/diagnose/jobs",
    dependencies=[Depends(per_user(DIAGNOSE_LIMIT))],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Диагностика фото листа в фоне",
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_413_CONTENT_TOO_LARGE: {"model": ErrorResponse},
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"model": ErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
    },
)
async def start_diagnosis(
    image: Annotated[
        UploadFile, File(description="Фото листа: JPEG, PNG или WebP до 10 МБ")
    ],
    settings: SettingsDep,
    user: CurrentUserDep,
    session: SessionDep,
    storage: StorageDep,
    top_k: Annotated[int, Query(ge=1, le=6, description="Сколько классов вернуть")] = 3,
    save: Annotated[
        bool, Query(description="Сохранить фото и результат в историю проверок")
    ] = False,
    vineyard_id: Annotated[
        uuid.UUID | None, Query(description="Участок, на котором снято фото")
    ] = None,
) -> DiagnosisJobStarted:
    data = await image.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise PayloadTooLargeError("Файл больше 10 МБ")
    leaf = read_leaf_image(data)
    await run_in_threadpool(classifier_file, settings)
    if save and vineyard_id is not None:
        await VineyardRepository(session).get_for_owner(user.id, vineyard_id)
    task_id = str(uuid.uuid4())
    key = f"jobs/{user.id}/{task_id}.{EXTENSIONS.get(leaf.format or '', 'jpg')}"
    await run_in_threadpool(storage.put, key, data, "application/octet-stream")
    try:
        celery_app.send_task(
            DIAGNOSE,
            args=[
                str(user.id),
                key,
                top_k,
                save,
                str(vineyard_id) if save and vineyard_id else None,
            ],
            task_id=task_id,
        )
    except Exception:
        await run_in_threadpool(storage.delete, key)
        raise
    return DiagnosisJobStarted(task_id=task_id)


@router.get(
    "/diagnose/jobs/{task_id}",
    summary="Результат диагностики в фоне",
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
)
async def read_diagnosis_job(
    task_id: uuid.UUID, user: CurrentUserDep
) -> DiagnosisJobRead:
    state, result = await run_in_threadpool(job_state, str(task_id))
    if state != "SUCCESS":
        return DiagnosisJobRead(status=JOB_STATUSES.get(state, "running"))
    # Номер задачи знает только тот, кто её поставил, но результат всё равно
    # отдаётся только ему
    if result.get("owner_id") != str(user.id):
        raise NotFoundError("Задача не найдена")
    return DiagnosisJobRead(
        status="done", result=DiagnosisRead.model_validate(result["diagnosis"])
    )
