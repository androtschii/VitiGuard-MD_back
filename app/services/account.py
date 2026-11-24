"""Сервис управления учётной записью: удаление аккаунта и выгрузка данных (GDPR)."""

from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.errors import ConflictError, UnauthorizedError
from app.core.security import averify_and_upgrade
from app.integrations.object_storage import Storage
from app.models.alert import Alert
from app.models.diagnosis import DiseaseResult, LeafImage
from app.models.field_note import FieldNote
from app.models.harvest import HarvestRecord
from app.models.treatment import Treatment
from app.models.user import User, UserRole
from app.models.vine_check import VineCheck
from app.models.vineyard import Vineyard
from app.repositories.vineyard import VineyardRepository
from app.schemas.user import UserDelete
from app.services.auth import INVALID_CREDENTIALS
from app.services.diagnosis_history import overlay_key

logger = structlog.get_logger(__name__)

EXPORT_NOTES = (
    "Фотографии не включены в архив выгрузки (только идентификаторы записей)."
)


async def delete_account(
    session: AsyncSession,
    user: User,
    data: UserDelete,
    storage: Storage,
) -> None:
    """Удаляет учётную запись пользователя и связанные файлы фото.

    Подтверждается текущим паролем. Нельзя удалить последнего активного администратора.
    Файлы из хранилища удаляются после коммита в БД: если файл не удалится,
    данные пользователя в БД уже исчезли, а сирота не страшен."""
    valid, _ = await averify_and_upgrade(data.password, user.hashed_password)
    if not valid:
        raise UnauthorizedError(INVALID_CREDENTIALS)

    if user.role == UserRole.ADMIN:
        other_admins = await session.scalar(
            select(func.count())
            .select_from(User)
            .where(
                User.role == UserRole.ADMIN,
                User.is_active.is_(True),
                User.id != user.id,
            )
        )
        if not other_admins:
            raise ConflictError("Нельзя удалить последнего администратора")

    # Сбор ключей фото перед удалением пользователя из базы
    leaf_keys = (
        await session.scalars(
            select(LeafImage.storage_key).where(LeafImage.owner_id == user.id)
        )
    ).all()
    file_keys: list[str] = []
    for key in leaf_keys:
        file_keys.extend((key, overlay_key(key, "heatmap"), overlay_key(key, "lesion")))

    vine_check_keys = (
        await session.scalars(
            select(VineCheck.storage_key).where(VineCheck.owner_id == user.id)
        )
    ).all()
    file_keys.extend(vine_check_keys)

    note_keys = (
        await session.scalars(
            select(FieldNote.photo_key)
            .join(Vineyard, Vineyard.id == FieldNote.vineyard_id)
            .where(Vineyard.owner_id == user.id, FieldNote.photo_key.is_not(None))
        )
    ).all()
    file_keys.extend(k for k in note_keys if k is not None)

    unique_file_keys = list(dict.fromkeys(file_keys))

    await session.delete(user)
    await session.commit()

    # Удаление файлов из хранилища после коммита в БД
    deleted_count = 0
    failed_count = 0
    for file_key in unique_file_keys:
        try:
            await run_in_threadpool(storage.delete, file_key)
            deleted_count += 1
        except Exception:
            failed_count += 1

    if failed_count:
        logger.warning(
            "account_storage_files_cleanup_failed",
            count=failed_count,
        )
    logger.info("account_storage_files_deleted", count=deleted_count)


async def export_account_data(
    session: AsyncSession,
    user: User,
) -> dict[str, Any]:
    """Формирует выгрузку данных пользователя в соответствии с Законом № 195/2024."""
    exported_at = datetime.now(UTC)

    # 1. Профиль
    profile = {
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role.value,
        "created_at": user.created_at.isoformat(),
        "notification_settings": {
            "alert_level": user.alert_level.value if user.alert_level else None
        },
        "telegram_linked": user.telegram_chat_id is not None,
    }

    # 2. Участки
    vineyard_views = await VineyardRepository(session).all_for_owner(user.id)
    vineyard_names = {v.vineyard.id: v.vineyard.name for v in vineyard_views}
    user_vineyard_ids = [v.vineyard.id for v in vineyard_views]

    vineyards_data = [
        {
            "id": str(v.vineyard.id),
            "name": v.vineyard.name,
            "variety": v.vineyard.variety.name if v.vineyard.variety else None,
            "area_ha": round(v.area_ha, 4),
            "geometry": v.geometry,
            "planting_year": v.vineyard.planting_year,
            "telegram_alerts": v.vineyard.telegram_alerts,
            "created_at": v.vineyard.created_at.isoformat(),
        }
        for v in vineyard_views
    ]

    # 3. Журналы участков
    treatments_data: list[dict[str, Any]] = []
    harvests_data: list[dict[str, Any]] = []
    notes_data: list[dict[str, Any]] = []

    if user_vineyard_ids:
        treatments = (
            await session.scalars(
                select(Treatment)
                .where(Treatment.vineyard_id.in_(user_vineyard_ids))
                .order_by(Treatment.applied_on.desc(), Treatment.id)
            )
        ).all()
        treatments_data = [
            {
                "id": str(t.id),
                "vineyard_id": str(t.vineyard_id),
                "vineyard": vineyard_names.get(t.vineyard_id),
                "applied_on": t.applied_on.isoformat(),
                "fungicide": t.fungicide.value,
                "product": t.product,
                "waiting_days": t.waiting_days,
            }
            for t in treatments
        ]

        harvests = (
            await session.scalars(
                select(HarvestRecord)
                .where(HarvestRecord.vineyard_id.in_(user_vineyard_ids))
                .order_by(HarvestRecord.season.desc(), HarvestRecord.id)
            )
        ).all()
        harvests_data = [
            {
                "id": str(h.id),
                "vineyard_id": str(h.vineyard_id),
                "vineyard": vineyard_names.get(h.vineyard_id),
                "season": h.season,
                "estimated_kg": h.estimated_kg,
                "actual_kg": h.actual_kg,
                "sugar_brix": h.sugar_brix,
            }
            for h in harvests
        ]

        field_notes = (
            await session.scalars(
                select(FieldNote)
                .where(FieldNote.vineyard_id.in_(user_vineyard_ids))
                .order_by(FieldNote.noted_on.desc(), FieldNote.id)
            )
        ).all()
        notes_data = [
            {
                "id": str(n.id),
                "vineyard_id": str(n.vineyard_id),
                "vineyard": vineyard_names.get(n.vineyard_id),
                "noted_on": n.noted_on.isoformat(),
                "text": n.text,
                "latitude": n.latitude,
                "longitude": n.longitude,
                "has_photo": n.photo_key is not None,
            }
            for n in field_notes
        ]

    # 4. Диагнозы
    diagnoses = (
        await session.scalars(
            select(DiseaseResult)
            .join(LeafImage, DiseaseResult.image_id == LeafImage.id)
            .options(selectinload(DiseaseResult.image).selectinload(LeafImage.vineyard))
            .where(LeafImage.owner_id == user.id)
            .order_by(DiseaseResult.created_at.desc(), DiseaseResult.id)
        )
    ).all()
    diagnoses_data = [
        {
            "id": str(d.id),
            "date": d.created_at.isoformat(),
            "vineyard_id": str(d.image.vineyard_id) if d.image.vineyard_id else None,
            "vineyard": d.image.vineyard.name if d.image.vineyard else None,
            "predicted_class": d.predicted_class,
            "confidence": d.confidence,
            "coordinates": (
                {"latitude": d.image.latitude, "longitude": d.image.longitude}
                if d.image.latitude is not None and d.image.longitude is not None
                else None
            ),
            "agronomist_conclusion": (
                {
                    "disease": d.expert_class,
                    "comment": d.expert_comment,
                }
                if d.expert_class and d.reviewed_at
                else None
            ),
        }
        for d in diagnoses
    ]

    # 5. Проверки лозы
    vine_checks = (
        await session.scalars(
            select(VineCheck)
            .options(selectinload(VineCheck.vineyard))
            .where(VineCheck.owner_id == user.id)
            .order_by(VineCheck.created_at.desc(), VineCheck.id)
        )
    ).all()
    vine_checks_data = [
        {
            "id": str(c.id),
            "date": c.created_at.isoformat(),
            "vineyard_id": str(c.vineyard_id) if c.vineyard_id else None,
            "vineyard": c.vineyard.name if c.vineyard else None,
            "trunks": c.trunks,
            "bunches": c.bunches,
        }
        for c in vine_checks
    ]

    # 6. Предупреждения
    alerts = (
        await session.scalars(
            select(Alert)
            .options(selectinload(Alert.vineyard))
            .where(Alert.user_id == user.id)
            .order_by(Alert.day.desc(), Alert.id)
        )
    ).all()
    alerts_data = [
        {
            "id": str(a.id),
            "date": a.day.isoformat(),
            "vineyard_id": str(a.vineyard_id),
            "vineyard": a.vineyard.name if a.vineyard else None,
            "disease": a.disease.value,
            "level": a.level.value,
            "is_read": a.read_at is not None,
        }
        for a in alerts
    ]

    return {
        "exported_at": exported_at.isoformat(),
        "notes": EXPORT_NOTES,
        "profile": profile,
        "vineyards": vineyards_data,
        "field_logs": {
            "treatments": treatments_data,
            "harvests": harvests_data,
            "notes": notes_data,
        },
        "diagnoses": diagnoses_data,
        "vine_checks": vine_checks_data,
        "alerts": alerts_data,
    }
