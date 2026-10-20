import uuid

from fastapi.concurrency import run_in_threadpool
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.integrations.object_storage import Storage
from app.models.diagnosis import DiseaseResult, LeafImage
from app.models.enums import ProcessingStatus
from app.schemas.diagnosis import DiagnosisRead

EXTENSIONS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}
CONTENT_TYPES = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}


def storage_key(owner_id: uuid.UUID, image_id: uuid.UUID, leaf: Image.Image) -> str:
    # Папка владельца: фото одного пользователя легко найти и удалить разом
    extension = EXTENSIONS.get(leaf.format or "", "jpg")
    return f"{owner_id}/{image_id}.{extension}"


async def save_diagnosis(
    session: AsyncSession,
    storage: Storage,
    owner_id: uuid.UUID,
    vineyard_id: uuid.UUID | None,
    data: bytes,
    leaf: Image.Image,
    diagnosis: DiagnosisRead,
    model_version: str,
) -> uuid.UUID:
    """Фото — в хранилище, проверка — в историю (pr-039, pr-053). Если запись
    в базу не удалась, фото удаляется: файлов без записи не остаётся."""
    image = LeafImage(
        owner_id=owner_id,
        vineyard_id=vineyard_id,
        storage_key="",
        content_type="",
        size_bytes=len(data),
        width=leaf.width,
        height=leaf.height,
    )
    session.add(image)
    await session.flush()
    key = storage_key(owner_id, image.id, leaf)
    image.storage_key = key
    image.content_type = CONTENT_TYPES[key.rsplit(".", 1)[1]]
    result = DiseaseResult(
        image_id=image.id,
        model_name=diagnosis.model,
        model_version=model_version,
        status=ProcessingStatus.COMPLETED,
        predicted_class=diagnosis.disease,
        confidence=diagnosis.confidence,
        probabilities={item.disease: item.probability for item in diagnosis.top},
        severity=diagnosis.lesion.severity if diagnosis.lesion else None,
    )
    session.add(result)
    await run_in_threadpool(storage.put, key, data, image.content_type)
    try:
        await session.commit()
    except Exception:
        await run_in_threadpool(storage.delete, key)
        raise
    return result.id
