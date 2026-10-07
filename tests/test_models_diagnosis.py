import pytest
from geoalchemy2 import WKTElement
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DiagnosisStatus, DiseaseResult, LeafImage, User, Vineyard

pytestmark = pytest.mark.integration

PLOT = "MULTIPOLYGON(((28.86 47.14, 28.87 47.14, 28.87 47.15, 28.86 47.14)))"


def make_image(owner: User, **kwargs: object) -> LeafImage:
    values: dict[str, object] = {
        "storage_key": "leaves/1.jpg",
        "content_type": "image/jpeg",
        "size_bytes": 512_000,
        "width": 1024,
        "height": 768,
    }
    values.update(kwargs)
    return LeafImage(owner=owner, **values)


async def add_image(session: AsyncSession, **kwargs: object) -> LeafImage:
    image = make_image(
        User(email="grower@example.md", hashed_password="hash"), **kwargs
    )
    session.add(image)
    await session.flush()
    return image


async def test_result_defaults_to_pending(session: AsyncSession) -> None:
    image = await add_image(session)
    result = DiseaseResult(image=image, model_name="efficientnet_b0", model_version="1")
    session.add(result)
    await session.flush()
    await session.refresh(result)

    assert result.status is DiagnosisStatus.PENDING
    assert result.predicted_class is None


async def test_completed_result_keeps_probabilities(session: AsyncSession) -> None:
    image = await add_image(session)
    probabilities = {"downy_mildew": 0.91, "powdery_mildew": 0.06, "healthy": 0.03}
    session.add(
        DiseaseResult(
            image=image,
            model_name="efficientnet_b0",
            model_version="1",
            status=DiagnosisStatus.COMPLETED,
            predicted_class="downy_mildew",
            confidence=0.91,
            probabilities=probabilities,
        )
    )
    await session.flush()
    session.expunge_all()

    stored = await session.scalar(select(DiseaseResult))

    assert stored is not None
    assert stored.probabilities == probabilities


@pytest.mark.parametrize(
    ("fields", "constraint"),
    [
        (
            {"status": DiagnosisStatus.COMPLETED},
            "ck_disease_results_completed_has_prediction",
        ),
        (
            {"status": DiagnosisStatus.FAILED, "confidence": 1.5},
            "ck_disease_results_confidence_range",
        ),
    ],
)
async def test_result_constraints(
    session: AsyncSession, fields: dict[str, object], constraint: str
) -> None:
    image = await add_image(session)
    session.add(DiseaseResult(image=image, model_name="m", model_version="1", **fields))

    with pytest.raises(IntegrityError, match=constraint):
        await session.flush()


@pytest.mark.parametrize(
    ("fields", "constraint"),
    [
        ({"size_bytes": 0}, "ck_leaf_images_size_positive"),
        ({"width": 0}, "ck_leaf_images_dimensions_positive"),
    ],
)
async def test_image_constraints(
    session: AsyncSession, fields: dict[str, object], constraint: str
) -> None:
    with pytest.raises(IntegrityError, match=constraint):
        await add_image(session, **fields)


async def test_storage_key_is_unique(session: AsyncSession) -> None:
    owner = User(email="grower@example.md", hashed_password="hash")
    session.add_all([make_image(owner), make_image(owner)])

    with pytest.raises(IntegrityError, match="uq_leaf_images_storage_key"):
        await session.flush()


async def test_image_survives_vineyard_deletion(session: AsyncSession) -> None:
    owner = User(email="grower@example.md", hashed_password="hash")
    vineyard = Vineyard(owner=owner, name="Участок", geom=WKTElement(PLOT, srid=4326))
    image = make_image(owner, vineyard=vineyard)
    session.add(image)
    await session.flush()

    await session.execute(delete(Vineyard).where(Vineyard.id == vineyard.id))
    await session.refresh(image)

    assert image.vineyard_id is None


async def test_images_and_results_deleted_with_owner(session: AsyncSession) -> None:
    image = await add_image(session)
    session.add(DiseaseResult(image=image, model_name="m", model_version="1"))
    await session.flush()

    await session.execute(delete(User).where(User.id == image.owner_id))

    assert await session.scalar(select(func.count()).select_from(LeafImage)) == 0
    assert await session.scalar(select(func.count()).select_from(DiseaseResult)) == 0
