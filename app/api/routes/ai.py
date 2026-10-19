import uuid
from typing import Annotated

from fastapi import APIRouter, File, Query, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from app.api.deps import CurrentUserDep, SessionDep, SettingsDep, StorageDep
from app.core.errors import PayloadTooLargeError, ServiceUnavailableError
from app.ml.classifier import LeafClassifier
from app.ml.model_file import ModelFile, describe_model
from app.ml.segmentation import LesionSegmenter
from app.repositories.vineyard import VineyardRepository
from app.schemas.diagnosis import ClassifierModelRead, DiagnosisRead
from app.schemas.errors import ErrorResponse
from app.services.diagnosis import analyze_leaf, read_leaf_image
from app.services.diagnosis_history import save_diagnosis

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
    if save:
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
