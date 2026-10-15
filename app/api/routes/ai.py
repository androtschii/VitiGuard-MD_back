from typing import Annotated

from fastapi import APIRouter, File, Query, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.api.deps import CurrentUserDep, SettingsDep
from app.core.errors import PayloadTooLargeError, ServiceUnavailableError
from app.ml.classifier import LeafClassifier
from app.ml.segmentation import LesionSegmenter
from app.schemas.diagnosis import DiagnosisRead
from app.schemas.errors import ErrorResponse
from app.services.diagnosis import analyze_leaf, read_leaf_image

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


@router.post(
    "/diagnose",
    summary="Определение болезни по фото листа",
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
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
    _: CurrentUserDep,
    top_k: Annotated[int, Query(ge=1, le=6, description="Сколько классов вернуть")] = 3,
) -> DiagnosisRead:
    data = await image.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise PayloadTooLargeError("Файл больше 10 МБ")
    leaf = read_leaf_image(data)
    classifier = get_classifier(request, settings)
    segmenter = get_segmenter(request, settings)
    # Модели занимают процессор на десятки и сотни миллисекунд: в отдельном
    # потоке они не задерживают обработку других запросов
    return await run_in_threadpool(analyze_leaf, leaf, classifier, segmenter, top_k)
