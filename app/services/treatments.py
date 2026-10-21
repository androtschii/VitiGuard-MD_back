"""Рекомендации по обработкам (pr-078): когда и каким типом препарата
обрабатывать участок, чтобы он был защищён в дни риска.

Риск по дням дают агромодели (pr-073, pr-074) — по факту и по прогнозу погоды.
Здесь он сопоставляется с журналом обработок: защищён ли участок в день
заражения, не смыл ли защиту дождь, не прошло ли лечебное окно. Сроки — средние
по рекомендациям для винограда; срок по этикетке конкретного препарата важнее."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import Disease, Fungicide, RiskLevel
from app.models.treatment import Treatment
from app.models.weather import DiseaseRisk, WeatherData
from app.services.agromodels import LOCAL_TZ, DailyWeather, HourlyWeather, daily_weather

# Сколько дней держится защита после обработки
PROTECTION_DAYS = {Fungicide.CONTACT: 8, Fungicide.SYSTEMIC: 12, Fungicide.SULFUR: 7}
# От каких болезней защищает: медь и манкоцеб против оидиума почти не работают,
# сера — против милдью
PROTECTS = {
    Fungicide.CONTACT: {Disease.DOWNY_MILDEW},
    Fungicide.SYSTEMIC: {Disease.DOWNY_MILDEW, Disease.POWDERY_MILDEW},
    Fungicide.SULFUR: {Disease.POWDERY_MILDEW},
}
# Контактный препарат смывается, когда после обработки выпало столько осадков
WASH_OFF_MM = 20.0
# Системный препарат останавливает милдью в первые трое суток после заражения
CURATIVE_DAYS = 3
# В жару сера обжигает листья и ягоды: вместо неё — системный препарат
SULFUR_MAX_TEMP = 30.0
OUTLOOK_DAYS = 7


class Action(StrEnum):
    # Защитная обработка до заражения
    PROTECT = "protect"
    # Лечебная обработка системным препаратом после заражения
    CURE = "cure"
    # Осмотр: ожидаются пятна милдью
    INSPECT = "inspect"


@dataclass(frozen=True)
class Recommendation:
    action: Action
    disease: Disease
    fungicide: Fungicide | None
    # Крайний срок: до этого дня включительно
    due_by: date
    # День риска, из-за которого рекомендация
    risk_day: date
    reason: str


@dataclass(frozen=True)
class Plan:
    today: date
    # До какого дня включительно участок защищён от болезни; None — не защищён
    protected_until: dict[Disease, date | None]
    recommendations: list[Recommendation]


def _rain_between(days: dict[date, DailyWeather], start: date, end: date) -> float:
    return sum(
        weather.precipitation_mm for day, weather in days.items() if start <= day <= end
    )


def protection_end(
    treatment: Treatment, disease: Disease, days: dict[date, DailyWeather]
) -> date | None:
    """Последний защищённый день после обработки. Контактный препарат
    действует до дня, когда осадки с обработки набрали 20 мм: в этот день
    он уже смыт."""
    if disease not in PROTECTS[treatment.fungicide]:
        return None
    end = treatment.applied_on + timedelta(
        days=PROTECTION_DAYS[treatment.fungicide] - 1
    )
    if treatment.fungicide is not Fungicide.CONTACT:
        return end
    day = treatment.applied_on
    while day <= end:
        if _rain_between(days, treatment.applied_on, day) >= WASH_OFF_MM:
            return day - timedelta(days=1)
        day += timedelta(days=1)
    return end


def protected_until(
    treatments: Sequence[Treatment],
    disease: Disease,
    days: dict[date, DailyWeather],
    on: date,
) -> date | None:
    """До какого дня участок защищён, если смотреть из дня on."""
    ends = [
        end
        for treatment in treatments
        if treatment.applied_on <= on
        and (end := protection_end(treatment, disease, days)) is not None
        and end >= on
    ]
    return max(ends, default=None)


def plan(
    today: date,
    risks: Sequence[DiseaseRisk],
    treatments: Sequence[Treatment],
    weather: Sequence[DailyWeather],
) -> Plan:
    days = {item.day: item for item in weather}
    by_method: dict[str, dict[date, DiseaseRisk]] = {}
    for risk in risks:
        by_method.setdefault(risk.method, {})[risk.target_date] = risk
    ahead = [today + timedelta(days=offset) for offset in range(OUTLOOK_DAYS + 1)]

    def high(method: str, day: date) -> bool:
        risk = by_method.get(method, {}).get(day)
        return risk is not None and risk.level is RiskLevel.HIGH

    def unprotected(disease: Disease, day: date) -> bool:
        return protected_until(treatments, disease, days, day) is None

    recommendations: list[Recommendation] = []

    # Заражение уже случилось и участок не был защищён: системный препарат,
    # пока не закрылось лечебное окно
    for offset in range(CURATIVE_DAYS, 0, -1):
        day = today - timedelta(days=offset)
        due = day + timedelta(days=CURATIVE_DAYS)
        if high("three_tens", day) and unprotected(Disease.DOWNY_MILDEW, day):
            recommendations.append(
                Recommendation(
                    Action.CURE,
                    Disease.DOWNY_MILDEW,
                    Fungicide.SYSTEMIC,
                    due,
                    day,
                    "downy_infection_past",
                )
            )
            break

    # Заражение по прогнозу: защитить участок до дождя
    infection = next(
        (
            day
            for day in ahead
            if high("three_tens", day) and unprotected(Disease.DOWNY_MILDEW, day)
        ),
        None,
    )
    if infection is not None:
        recommendations.append(
            Recommendation(
                Action.PROTECT,
                Disease.DOWNY_MILDEW,
                # Защита до дождя — контактным; если дождь уже сегодня, успеть
                # можно только системным
                Fungicide.CONTACT if infection > today else Fungicide.SYSTEMIC,
                max(today, infection - timedelta(days=1)),
                infection,
                "downy_infection_forecast",
            )
        )

    symptoms = next((day for day in ahead if high("goidanich", day)), None)
    if symptoms is not None:
        recommendations.append(
            Recommendation(
                Action.INSPECT,
                Disease.DOWNY_MILDEW,
                None,
                symptoms,
                symptoms,
                "downy_symptoms",
            )
        )

    # Высокий индекс оидиума в незащищённый день: сера, а в жару — системный
    powdery = next(
        (
            day
            for day in ahead
            if high("gubler_thomas", day) and unprotected(Disease.POWDERY_MILDEW, day)
        ),
        None,
    )
    if powdery is not None:
        hot = any(
            (forecast := days.get(day)) is not None
            and forecast.max_temp > SULFUR_MAX_TEMP
            for day in (powdery, powdery + timedelta(days=1))
        )
        recommendations.append(
            Recommendation(
                Action.PROTECT,
                Disease.POWDERY_MILDEW,
                Fungicide.SYSTEMIC if hot else Fungicide.SULFUR,
                powdery,
                powdery,
                "powdery_high_hot" if hot else "powdery_high",
            )
        )

    return Plan(
        today=today,
        protected_until={
            disease: protected_until(treatments, disease, days, today)
            for disease in (Disease.DOWNY_MILDEW, Disease.POWDERY_MILDEW)
        },
        recommendations=sorted(recommendations, key=lambda item: item.due_by),
    )


async def vineyard_plan(
    session: AsyncSession, vineyard_id: uuid.UUID, today: date
) -> Plan:
    """План по сохранённому риску, погоде и журналу обработок участка."""
    # Обработки старше самой долгой защиты на сегодня уже не влияют
    since = today - timedelta(days=max(PROTECTION_DAYS.values()) + CURATIVE_DAYS)
    until = today + timedelta(days=OUTLOOK_DAYS)
    risks = await session.scalars(
        select(DiseaseRisk).where(
            DiseaseRisk.vineyard_id == vineyard_id,
            DiseaseRisk.target_date >= since,
            DiseaseRisk.target_date <= until,
        )
    )
    treatments = await session.scalars(
        select(Treatment).where(
            Treatment.vineyard_id == vineyard_id,
            Treatment.applied_on >= since,
            Treatment.applied_on <= today,
        )
    )
    start = datetime.combine(since, time(), LOCAL_TZ).astimezone(UTC)
    end = datetime.combine(until + timedelta(days=1), time(), LOCAL_TZ).astimezone(UTC)
    hours = await session.execute(
        select(
            WeatherData.observed_at,
            WeatherData.temperature_c,
            WeatherData.relative_humidity,
            WeatherData.precipitation_mm,
        )
        .where(
            WeatherData.vineyard_id == vineyard_id,
            WeatherData.observed_at >= start,
            WeatherData.observed_at < end,
        )
        .order_by(WeatherData.observed_at)
    )
    return plan(
        today,
        risks.all(),
        treatments.all(),
        daily_weather(HourlyWeather(*row) for row in hours),
    )
