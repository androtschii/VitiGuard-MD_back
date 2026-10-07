from datetime import UTC, date, datetime

import pytest
from geoalchemy2 import WKTElement
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Disease, DiseaseRisk, RiskLevel, User, Vineyard, WeatherData

pytestmark = pytest.mark.integration

PLOT = "MULTIPOLYGON(((28.86 47.14, 28.87 47.14, 28.87 47.15, 28.86 47.14)))"


def make_vineyard() -> Vineyard:
    owner = User(email="grower@example.md", hashed_password="hash")
    return Vineyard(owner=owner, name="Участок", geom=WKTElement(PLOT, srid=4326))


def make_weather(vineyard: Vineyard, **kwargs: object) -> WeatherData:
    values: dict[str, object] = {
        "observed_at": datetime(2026, 5, 20, 6, tzinfo=UTC),
        "temperature_c": 14.2,
        "relative_humidity": 92.0,
        "precipitation_mm": 3.4,
    }
    values.update(kwargs)
    return WeatherData(vineyard=vineyard, **values)


def make_risk(vineyard: Vineyard, **kwargs: object) -> DiseaseRisk:
    values: dict[str, object] = {
        "disease": Disease.DOWNY_MILDEW,
        "target_date": date(2026, 5, 21),
        "method": "three_tens",
        "score": 0.8,
        "level": RiskLevel.HIGH,
    }
    values.update(kwargs)
    return DiseaseRisk(vineyard=vineyard, **values)


async def test_weather_defaults_to_observation(session: AsyncSession) -> None:
    weather = make_weather(make_vineyard())
    session.add(weather)
    await session.flush()
    await session.refresh(weather)

    assert weather.is_forecast is False


async def test_one_weather_row_per_hour(session: AsyncSession) -> None:
    vineyard = make_vineyard()
    session.add_all([make_weather(vineyard), make_weather(vineyard, is_forecast=True)])

    with pytest.raises(IntegrityError, match="uq_weather_data_vineyard_id_observed_at"):
        await session.flush()


@pytest.mark.parametrize(
    ("fields", "constraint"),
    [
        ({"relative_humidity": 101}, "ck_weather_data_humidity_range"),
        ({"precipitation_mm": -0.1}, "ck_weather_data_precipitation_non_negative"),
    ],
)
async def test_weather_constraints(
    session: AsyncSession, fields: dict[str, object], constraint: str
) -> None:
    session.add(make_weather(make_vineyard(), **fields))

    with pytest.raises(IntegrityError, match=constraint):
        await session.flush()


async def test_risk_keeps_factors(session: AsyncSession) -> None:
    factors = {"temperature_c": 12.5, "precipitation_mm": 14.0, "shoot_length_cm": 11}
    session.add(make_risk(make_vineyard(), factors=factors))
    await session.flush()
    session.expunge_all()

    stored = await session.scalar(select(DiseaseRisk))

    assert stored is not None
    assert stored.disease is Disease.DOWNY_MILDEW
    assert stored.level is RiskLevel.HIGH
    assert stored.factors == factors


async def test_methods_stored_side_by_side(session: AsyncSession) -> None:
    vineyard = make_vineyard()
    session.add_all(
        [
            make_risk(vineyard),
            make_risk(vineyard, method="goidanich", score=0.4, level=RiskLevel.MEDIUM),
            make_risk(
                vineyard,
                disease=Disease.POWDERY_MILDEW,
                method="gubler_thomas",
                score=0.2,
                level=RiskLevel.LOW,
            ),
        ]
    )
    await session.flush()

    assert await session.scalar(select(func.count()).select_from(DiseaseRisk)) == 3


async def test_same_method_and_date_rejected(session: AsyncSession) -> None:
    vineyard = make_vineyard()
    session.add_all([make_risk(vineyard), make_risk(vineyard)])

    with pytest.raises(
        IntegrityError, match="uq_disease_risks_vineyard_id_disease_target_date_method"
    ):
        await session.flush()


@pytest.mark.parametrize("score", [-0.1, 1.1])
async def test_score_range(session: AsyncSession, score: float) -> None:
    session.add(make_risk(make_vineyard(), score=score))

    with pytest.raises(IntegrityError, match="ck_disease_risks_score_range"):
        await session.flush()


async def test_weather_and_risks_deleted_with_vineyard(session: AsyncSession) -> None:
    vineyard = make_vineyard()
    session.add_all([make_weather(vineyard), make_risk(vineyard)])
    await session.flush()

    await session.execute(delete(Vineyard).where(Vineyard.id == vineyard.id))

    assert await session.scalar(select(func.count()).select_from(WeatherData)) == 0
    assert await session.scalar(select(func.count()).select_from(DiseaseRisk)) == 0
