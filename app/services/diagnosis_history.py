import base64
import uuid

from fastapi.concurrency import run_in_threadpool
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.integrations.object_storage import Storage
from app.ml.overlays import PNG_DATA_URL
from app.models.diagnosis import DiseaseResult, LeafImage
from app.models.enums import ProcessingStatus
from app.schemas.diagnosis import DiagnosisRead

EXTENSIONS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}
CONTENT_TYPES = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}


def storage_key(owner_id: uuid.UUID, image_id: uuid.UUID, leaf: Image.Image) -> str:
    # Папка владельца: фото одного пользователя легко найти и удалить разом
    extension = EXTENSIONS.get(leaf.format or "", "jpg")
    return f"{owner_id}/{image_id}.{extension}"


def overlay_key(photo_key: str, overlay: str) -> str:
    # Наложения лежат рядом с фото: <владелец>/<фото>.heatmap.png
    return f"{photo_key.rsplit('.', 1)[0]}.{overlay}.png"


def png_bytes(data_url: str) -> bytes:
    return base64.b64decode(data_url.removeprefix(PNG_DATA_URL))


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
    """Фото, тепловая карта и маска поражения — в хранилище, проверка — в
    историю. Если запись в базу не удалась, файлы удаляются: файлов без записи
    не остаётся."""
    image = LeafImage(
        owner_id=owner_id,
        vineyard_id=vineyard_id,
        storage_key="",
        content_type="",
        size_bytes=len(data),
        width=leaf.width,
        height=leaf.height,
        latitude=diagnosis.location.lat if diagnosis.location else None,
        longitude=diagnosis.location.lon if diagnosis.location else None,
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
    files = [(key, data, image.content_type)]
    if diagnosis.heatmap:
        result.heatmap_key = overlay_key(key, "heatmap")
        files.append((result.heatmap_key, png_bytes(diagnosis.heatmap), "image/png"))
    if diagnosis.lesion:
        result.lesion_key = overlay_key(key, "lesion")
        files.append((result.lesion_key, png_bytes(diagnosis.lesion.mask), "image/png"))
    session.add(result)
    saved: list[str] = []
    try:
        for file_key, content, content_type in files:
            await run_in_threadpool(storage.put, file_key, content, content_type)
            saved.append(file_key)
        await session.commit()
    except Exception:
        for file_key in saved:
            await run_in_threadpool(storage.delete, file_key)
        raise
    return result.id
