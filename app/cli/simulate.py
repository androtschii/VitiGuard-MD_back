"""Учебный участок с полным сезоном данных (pr-099).

    uv run python -m app.cli.simulate --email student@example.md \\
        --password secret123 --scenario outbreak --season 2025 --seed 7

Создаёт (если нужно) учётную запись и новый участок у Крикова и записывает в
базу почасовую погоду сезона, риск по агромоделям и спутниковые снимки с
индексами — дальше всё видно в приложении: риск по дням, график NDVI,
аномалии, PDF-отчёт и выгрузка. Внешние сервисы не нужны."""

import argparse
import asyncio
import random
from dataclasses import dataclass
from datetime import date

from geoalchemy2 import WKTElement
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.security import ahash_password
from app.db.session import create_engine, create_session_factory
from app.integrations.open_meteo import WeatherSample
from app.models.enums import ProcessingStatus, RiskLevel
from app.models.satellite import IndexType, SatelliteScan, VegetationIndex
from app.models.user import User
from app.models.vineyard import Vineyard
from app.repositories.weather import WeatherRepository
from app.services.agromodels import season_risks
from app.services.anomaly import vineyard_anomalies
from app.services.risk import save_risks
from app.services.simulation import Scenario, simulate_scans, simulate_weather

# Около 4 га на склоне у Крикова; зерно сдвигает участок, чтобы участки
# одной группы не лежали друг на друге
CENTRE = (47.14, 28.86)
SIDE = 0.0018


@dataclass(frozen=True)
class Summary:
    vineyard_id: str
    hours: int
    infections: int
    powdery_high_days: int
    scans: int
    anomalies: int


def _square(seed: int) -> WKTElement:
    rng = random.Random(f"place-{seed}")
    lat = CENTRE[0] + rng.uniform(-0.05, 0.05)
    lon = CENTRE[1] + rng.uniform(-0.05, 0.05)
    ring = [
        (lon, lat),
        (lon + SIDE * 1.47, lat),
        (lon + SIDE * 1.47, lat + SIDE),
        (lon, lat + SIDE),
        (lon, lat),
    ]
    points = ", ".join(f"{x:.6f} {y:.6f}" for x, y in ring)
    return WKTElement(f"MULTIPOLYGON((({points})))", srid=4326)


async def simulate(
    session: AsyncSession,
    email: str,
    password: str | None,
    scenario: Scenario,
    season: int,
    seed: int,
) -> Summary:
    owner = await session.scalar(select(User).where(User.email == email.lower()))
    if owner is None:
        if not password:
            raise SystemExit("Новой учётной записи нужен пароль: --password")
        owner = User(
            email=email.lower(),
            hashed_password=await ahash_password(password),
            full_name="Студент",
        )
        session.add(owner)
    vineyard = Vineyard(
        owner=owner,
        name=f"Учебный участок: {scenario}, {season}, зерно {seed}",
        geom=_square(seed),
    )
    session.add(vineyard)
    await session.flush()

    hours = simulate_weather(season, scenario, seed)
    await WeatherRepository(session).upsert(
        vineyard.id,
        [
            WeatherSample(
                observed_at=hour.at,
                temperature_c=hour.temperature_c,
                relative_humidity=hour.relative_humidity,
                precipitation_mm=hour.precipitation_mm,
            )
            for hour in hours
        ],
    )
    risks = season_risks(hours)
    # Весь сезон — прошлое: риск считается по факту, а не по прогнозу
    await save_risks(session, vineyard, risks, date(season, 12, 31))

    scans = simulate_scans(season, scenario, seed)
    for index, scan in enumerate(scans):
        record = SatelliteScan(
            vineyard=vineyard,
            product_id=f"SIM_{seed}_{season}_{index:03d}",
            acquired_at=scan.acquired_at,
            platform="sentinel-2a" if index % 2 == 0 else "sentinel-2b",
            cloud_cover=scan.scene_cloud_cover,
            parcel_cloud_cover=scan.parcel_cloud_cover,
            status=ProcessingStatus.COMPLETED,
        )
        for index_type, value in (
            (IndexType.NDVI, scan.ndvi),
            (IndexType.NDRE, scan.ndre),
        ):
            if value is not None:
                record.indices.append(
                    VegetationIndex(
                        index_type=index_type,
                        mean_value=value,
                        min_value=value - 0.08,
                        max_value=value + 0.08,
                        std_value=0.03,
                    )
                )
        session.add(record)
    await session.commit()

    return Summary(
        vineyard_id=str(vineyard.id),
        hours=len(hours),
        infections=sum(1 for r in risks if r.method == "three_tens" and r.score == 1),
        powdery_high_days=sum(
            1
            for r in risks
            if r.method == "gubler_thomas" and r.level is RiskLevel.HIGH
        ),
        scans=len(scans),
        anomalies=len(await vineyard_anomalies(session, vineyard)),
    )


async def run(
    settings: Settings,
    email: str,
    password: str | None,
    scenario: Scenario,
    season: int,
    seed: int,
) -> Summary:
    engine = create_engine(settings)
    try:
        async with create_session_factory(engine)() as session:
            return await simulate(session, email, password, scenario, season, seed)
    finally:
        await engine.dispose()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True)
    parser.add_argument("--password")
    parser.add_argument(
        "--scenario", type=Scenario, choices=list(Scenario), default=Scenario.normal
    )
    parser.add_argument("--season", type=int, default=date.today().year - 1)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    summary = asyncio.run(
        run(
            get_settings(),
            args.email,
            args.password,
            args.scenario,
            args.season,
            args.seed,
        )
    )
    print(
        f"Участок {summary.vineyard_id}: {summary.hours} ч погоды, "
        f"заражений милдью {summary.infections}, дней высокого риска оидиума "
        f"{summary.powdery_high_days}, снимков {summary.scans}, "
        f"аномалий NDVI {summary.anomalies}"
    )


if __name__ == "__main__":  # pragma: no cover — запуск как скрипта
    main()
