import asyncio
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from celery import shared_task

from app.core.config import Settings, get_settings
from app.db.session import create_engine, create_session_factory
from app.integrations.object_storage import ObjectStorage, Storage
from app.ml.classifier import LeafClassifier
from app.ml.model_file import describe_model
from app.ml.segmentation import LesionSegmenter
from app.services.diagnosis import analyze_leaf, photo_location, read_leaf_image
from app.services.diagnosis_history import save_diagnosis

DIAGNOSE = "ai.diagnose"


# Модели загружаются один раз на процесс воркера и остаются в памяти
@lru_cache(maxsize=2)
def load_classifier(path: Path) -> LeafClassifier:
    return LeafClassifier(path)


@lru_cache(maxsize=2)
def load_segmenter(path: Path) -> LesionSegmenter | None:
    return LesionSegmenter(path) if path.exists() else None


async def run_diagnosis(
    settings: Settings,
    storage: Storage,
    owner_id: str,
    key: str,
    top_k: int,
    save: bool,
    vineyard_id: str | None,
) -> dict[str, Any]:
    """Диагноз по фото из временного ключа хранилища. Фото проверено API до
    постановки задачи; временный файл удаляется в любом случае."""
    data = await asyncio.to_thread(storage.get, key)
    try:
        leaf = read_leaf_image(data)
        diagnosis = await asyncio.to_thread(
            analyze_leaf,
            leaf,
            load_classifier(settings.diagnosis_model_path),
            load_segmenter(settings.lesion_model_path),
            top_k,
        )
        diagnosis.location = photo_location(data)
        if save:
            version = describe_model(settings.diagnosis_model_path).version
            engine = create_engine(settings)
            try:
                async with create_session_factory(engine)() as session:
                    diagnosis.id = await save_diagnosis(
                        session,
                        storage,
                        uuid.UUID(owner_id),
                        uuid.UUID(vineyard_id) if vineyard_id else None,
                        data,
                        leaf,
                        diagnosis,
                        version,
                    )
            finally:
                await engine.dispose()
    finally:
        await asyncio.to_thread(storage.delete, key)
    return {"owner_id": owner_id, "diagnosis": diagnosis.model_dump(mode="json")}


@shared_task(name=DIAGNOSE)
def diagnose_task(
    owner_id: str, key: str, top_k: int, save: bool, vineyard_id: str | None
) -> dict[str, Any]:
    settings = get_settings()
    return asyncio.run(
        run_diagnosis(
            settings,
            ObjectStorage.from_settings(settings),
            owner_id,
            key,
            top_k,
            save,
            vineyard_id,
        )
    )
