"""Данные для нагрузочного теста (pr-096): пул пользователей с участками и
риском на сегодня, чтобы запросы кабинета возвращали не пустые списки.

    uv run python -m app.cli.load_seed --users 50

Пользователи — load-001@load.vitiguard.md … с одним паролем; повторный запуск
пропускает уже созданных. Участки — квадраты по 4 га, разбросанные по
винодельческим районам, риск — на неделю вперёд разного уровня."""

import argparse
import asyncio
import random
from datetime import timedelta

from geoalchemy2 import WKTElement
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.security import ahash_password
from app.db.session import create_engine, create_session_factory
from app.models.enums import Disease, RiskLevel
from app.models.user import User
from app.models.vineyard import Vineyard
from app.models.weather import DiseaseRisk
from app.services.risk import local_today

PASSWORD = "load-test-123"
DOMAIN = "load.vitiguard.md"
VINEYARDS_PER_USER = 3
RISK_DAYS = 7
# Центры винодельческих зон: Кодры, Штефан-Водэ, Валул луй Траян
CENTRES = ((47.14, 28.86), (46.52, 29.86), (45.90, 28.19))
# Сторона квадрата 200 м: около 4 га
SIDE_DEG = 0.0018


def email(number: int) -> str:
    return f"load-{number:03d}@{DOMAIN}"


def square(lat: float, lon: float) -> WKTElement:
    ring = [
        (lon, lat),
        (lon + SIDE_DEG * 1.47, lat),
        (lon + SIDE_DEG * 1.47, lat + SIDE_DEG),
        (lon, lat + SIDE_DEG),
        (lon, lat),
    ]
    points = ", ".join(f"{x:.6f} {y:.6f}" for x, y in ring)
    return WKTElement(f"MULTIPOLYGON((({points})))", srid=4326)


async def seed(session: AsyncSession, users: int, rng: random.Random) -> int:
    """Создаёт недостающих пользователей; возвращает, сколько создано."""
    existing = set(
        (
            await session.scalars(
                select(User.email).where(User.email.like(f"%@{DOMAIN}"))
            )
        ).all()
    )
    hashed = await ahash_password(PASSWORD)
    today = local_today()
    created = 0
    for number in range(1, users + 1):
        if email(number) in existing:
            continue
        owner = User(email=email(number), hashed_password=hashed, full_name="Нагрузка")
        for index in range(VINEYARDS_PER_USER):
            lat, lon = CENTRES[(number + index) % len(CENTRES)]
            vineyard = Vineyard(
                owner=owner,
                name=f"Участок {index + 1}",
                geom=square(
                    lat + rng.uniform(-0.15, 0.15), lon + rng.uniform(-0.15, 0.15)
                ),
            )
            for offset in range(RISK_DAYS + 1):
                for disease, method in (
                    (Disease.DOWNY_MILDEW, "three_tens"),
                    (Disease.POWDERY_MILDEW, "gubler_thomas"),
                ):
                    level = rng.choice(list(RiskLevel))
                    session.add(
                        DiseaseRisk(
                            vineyard=vineyard,
                            disease=disease,
                            method=method,
                            target_date=today + timedelta(days=offset),
                            score={"low": 0.1, "medium": 0.5, "high": 0.9}[level],
                            level=level,
                            factors={},
                        )
                    )
            session.add(vineyard)
        created += 1
    await session.commit()
    return created


async def run(settings: Settings, users: int) -> int:
    engine = create_engine(settings)
    try:
        async with create_session_factory(engine)() as session:
            return await seed(session, users, random.Random(96))
    finally:
        await engine.dispose()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=50)
    args = parser.parse_args(argv)
    created = asyncio.run(run(get_settings(), args.users))
    print(f"Создано пользователей: {created}, пароль: {PASSWORD}")


if __name__ == "__main__":  # pragma: no cover — запуск как скрипта
    main()
