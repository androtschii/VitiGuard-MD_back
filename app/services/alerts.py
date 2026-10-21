from datetime import date, timedelta
from typing import Any

from sqlalchemy import ColumnElement, case, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute

from app.models.alert import Alert
from app.models.enums import Disease, RiskLevel
from app.models.vineyard import Vineyard
from app.models.weather import DiseaseRisk

# Сегодня и два дня вперёд: прогноз погоды на этот срок надёжнее, а
# предупреждение за день-два оставляет время на обработку
ALERT_HORIZON_DAYS = 2

LEVEL_RANK = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}


def _rank(level: ColumnElement[Any] | InstrumentedAttribute[Any]) -> ColumnElement[int]:
    # Порядок уровней в SQL: строки «low < medium < high» по алфавиту не сравнить
    return case(
        {RiskLevel.HIGH.value: 2, RiskLevel.MEDIUM.value: 1}, value=level, else_=0
    )


async def raise_alerts(
    session: AsyncSession,
    vineyard: Vineyard,
    today: date,
    min_level: RiskLevel = RiskLevel.HIGH,
) -> int:
    """Предупреждения владельцу участка о риске не ниже порога на ближайшие
    дни (pr-080). По болезни берётся худший уровень из её моделей. Возвращает
    число дней с риском, о которых есть предупреждение."""
    rows = await session.execute(
        select(DiseaseRisk.disease, DiseaseRisk.target_date, DiseaseRisk.level).where(
            DiseaseRisk.vineyard_id == vineyard.id,
            DiseaseRisk.target_date >= today,
            DiseaseRisk.target_date <= today + timedelta(days=ALERT_HORIZON_DAYS),
        )
    )
    worst: dict[tuple[Disease, date], RiskLevel] = {}
    for disease, day, level in rows:
        current = worst.get((disease, day), RiskLevel.LOW)
        if LEVEL_RANK[level] > LEVEL_RANK[current]:
            worst[(disease, day)] = level
        else:
            worst.setdefault((disease, day), level)
    risky = [
        {
            "user_id": vineyard.owner_id,
            "vineyard_id": vineyard.id,
            "disease": disease,
            "day": day,
            "level": level,
        }
        for (disease, day), level in worst.items()
        if LEVEL_RANK[level] >= LEVEL_RANK[min_level]
    ]
    if not risky:
        return 0
    statement = insert(Alert).values(risky)
    # Повтор пересчёта ничего не меняет; рост уровня обновляет предупреждение
    # и снова делает его непрочитанным, снижение — не трогает
    await session.execute(
        statement.on_conflict_do_update(
            index_elements=["vineyard_id", "disease", "day"],
            set_={
                "level": statement.excluded.level,
                "read_at": None,
                "updated_at": func.now(),
            },
            where=_rank(statement.excluded.level) > _rank(Alert.level),
        )
    )
    return len(risky)
