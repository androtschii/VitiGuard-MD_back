import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.api.deps import CurrentUserDep, SettingsDep
from app.api.rate_limit import DIAGNOSE_LIMIT, per_user
from app.core.errors import BadRequestError, PayloadTooLargeError
from app.ml.model_file import describe_robot_model
from app.schemas.errors import ErrorResponse
from app.schemas.robot import (
    HardSampleUploadResult,
    RobotTelemetryCreate,
    RobotTelemetryRead,
)
from app.services.diagnosis import read_leaf_image
from app.services.diagnosis_history import EXTENSIONS

router = APIRouter(
    prefix="/robot",
    tags=["robot"],
    responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
)

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
# Идентификатор робота становится именем папки: только буквы, цифры, «-» и «_»,
# иначе «../» в нём вывел бы запись за пределы каталога кадров
ROBOT_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


@router.post(
    "/telemetry",
    summary="Приём телеметрии от бортового компьютера робота-сборщика",
    response_model=RobotTelemetryRead,
)
async def submit_telemetry(
    body: RobotTelemetryCreate,
    settings: SettingsDep,
    _: CurrentUserDep,
) -> RobotTelemetryRead:
    """Принимает телеметрию (батарея, скорость, FPS, задержка) и сообщает о наличии новой модели."""
    update_available = False
    latest_version: str | None = None

    if settings.vine_model_path.exists():
        try:
            meta = describe_robot_model(settings.vine_model_path)
            latest_version = meta.version
            if body.active_model_version != latest_version:
                update_available = True
        except Exception:
            pass

    # Регулировка скорости робота при высокой задержке инференса (> 40 мс)
    recommended_speed = 0.5
    if body.infer_latency_ms > 45.0:
        recommended_speed = 0.3
    elif body.battery_pct < 20.0:
        recommended_speed = 0.2

    return RobotTelemetryRead(
        robot_id=body.robot_id,
        status="ok",
        received_at=datetime.now(UTC),
        recommended_speed_mps=recommended_speed,
        active_model_version=body.active_model_version,
        update_available=update_available,
        latest_model_version=latest_version,
    )


def _save_uploaded_frame(
    dest_path: Path,
    content: bytes,
    queue_log: Path,
    log_entry: dict[str, object],
) -> None:
    dest_path.write_bytes(content)
    with queue_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


@router.post(
    "/active-learning/upload",
    dependencies=[Depends(per_user(DIAGNOSE_LIMIT))],
    status_code=status.HTTP_201_CREATED,
    summary="Выгрузка сложных кадров (Active Learning) с робота на сервер",
    response_model=HardSampleUploadResult,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_413_CONTENT_TOO_LARGE: {"model": ErrorResponse},
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"model": ErrorResponse},
    },
)
async def upload_active_learning_frame(
    image: Annotated[UploadFile, File(description="Снимок со спорной разметкой")],
    settings: SettingsDep,
    robot_id: Annotated[
        str, Form(description="Идентификатор робота", pattern=ROBOT_ID_PATTERN)
    ],
    uncertainty_score: Annotated[
        float, Form(description="Оценка неопределённости TTA")
    ] = 0.5,
    metadata_json: Annotated[
        str, Form(description="JSON предразметки и TTA расхождений")
    ] = "{}",
    _: CurrentUserDep = None,  # type: ignore[assignment]
) -> HardSampleUploadResult:
    """Принимает кадр, где модель лозы была не уверена, и ставит его в очередь на разметку."""
    content = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise PayloadTooLargeError("Файл кадра превышает лимит 15 МБ")
    # Формат — по содержимому, а не по имени файла: иначе под видом кадра
    # можно было бы сохранить на сервер что угодно
    frame = await run_in_threadpool(read_leaf_image, content)

    try:
        meta = json.loads(metadata_json) if metadata_json else {}
    except json.JSONDecodeError as error:
        raise BadRequestError("metadata_json должен быть JSON") from error

    upload_dir = settings.robot_upload_dir / robot_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    sample_id = f"al_{uuid.uuid4().hex[:10]}"
    ext = "." + EXTENSIONS.get(frame.format or "", "jpg")
    dest_path = upload_dir / f"{sample_id}{ext}"

    queue_log = settings.robot_upload_dir / "active_learning_queue.jsonl"
    queue_log.parent.mkdir(parents=True, exist_ok=True)

    log_entry: dict[str, object] = {
        "sample_id": sample_id,
        "robot_id": robot_id,
        "received_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "uncertainty_score": uncertainty_score,
        "file": str(dest_path),
        "meta": meta,
    }
    await run_in_threadpool(
        _save_uploaded_frame, dest_path, content, queue_log, log_entry
    )

    return HardSampleUploadResult(
        sample_id=sample_id,
        status="queued_for_label_studio",
        filename=dest_path.name,
        bytes_received=len(content),
    )
