"""Агрономические модели риска милдью и оидиума (pr-073, pr-074).

Модели работают с суточными сводками почасовой погоды. Сутки — местные
(Europe/Chisinau): «среднесуточная температура» и «часы подряд» в правилах
относятся к календарному дню на участке, а не к суткам по UTC.

Результат — риск на каждый день: оценка 0..1, уровень и факторы, из которых она
получилась. Те же модели размечают исторические метеоданные для обучения
прогнозной модели (ml/pr-014)."""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from itertools import pairwise
from typing import Any
from zoneinfo import ZoneInfo

from app.models.enums import Disease, RiskLevel

LOCAL_TZ = ZoneInfo("Europe/Chisinau")
# День без большей части часов (обрыв ряда) моделям не годится: среднее по
# нескольким ночным часам занизило бы температуру дня
MIN_DAY_HOURS = 20

# Сезон, когда на лозе есть зелёные ткани и модели имеют смысл. Правило «трёх
# десяток» описывает первичное (весеннее) заражение — для него окно до конца
# июня; инкубация и оидиум — до конца сентября. Вне окон модели дают ложные
# сигналы: осенние дожди после листопада заражать уже нечего
SEASON_START = (4, 1)
PRIMARY_INFECTION_END = (6, 30)
SEASON_END = (9, 30)

# «Три десятки»: среднесуточная температура, осадки за 48 ч и длина побегов
THREE_TENS_TEMPERATURE = 10.0
THREE_TENS_RAIN_MM = 10.0
# Длина побегов из метеоданных не следует. Оценка по сумме активных температур
# (база 10 °C с 1 марта): побеги около 10 см — примерно через две-три недели
# после распускания почек, что в Молдове соответствует сумме около 130 °C·сут.
# Это приближение: фенофазу, указанную пользователем, следует ставить выше
SHOOTS_10_CM_GDD = 130.0
GDD_BASE = 10.0

# Модель Гойданича: длительность инкубационного периода милдью (дни) в
# зависимости от среднесуточной температуры. Узлы — приближение кривой
# Гойданича по опубликованным диапазонам: около двух недель в прохладную погоду
# (13 °C), 4-5 дней при 20-25 °C; ниже 10 °C и выше 30 °C развитие практически
# останавливается. Между узлами — линейная интерполяция
GOIDANICH_DAYS = (
    (10.0, 20.0),
    (13.0, 14.0),
    (15.0, 11.0),
    (17.0, 8.5),
    (19.0, 6.5),
    (21.0, 5.5),
    (23.0, 4.5),
    (25.0, 4.0),
    (27.0, 4.5),
    (29.0, 5.5),
    (30.0, 6.0),
)

# Индекс Gubler-Thomas (UC Davis): день «оидиума» — не меньше 6 часов подряд
# при 21-30 °C; жара выше 35 °C подавляет гриб
GT_LOW, GT_HIGH = 21.0, 30.0
GT_HOURS = 6
GT_HEAT = 35.0


@dataclass(frozen=True)
class HourlyWeather:
    at: datetime
    temperature_c: float
    relative_humidity: float
    precipitation_mm: float


@dataclass(frozen=True)
class DailyWeather:
    day: date
    mean_temp: float
    min_temp: float
    max_temp: float
    precipitation_mm: float
    mean_humidity: float
    # Самая длинная серия часов подряд с температурой 21-30 °C
    longest_mild_run: int
    hours_above_35: int


@dataclass(frozen=True)
class RiskDay:
    day: date
    disease: Disease
    method: str
    score: float
    level: RiskLevel
    factors: dict[str, Any] = field(default_factory=dict)


def daily_weather(
    hours: Iterable[HourlyWeather], tz: ZoneInfo = LOCAL_TZ
) -> list[DailyWeather]:
    by_day: dict[date, list[HourlyWeather]] = defaultdict(list)
    for hour in hours:
        by_day[hour.at.astimezone(tz).date()].append(hour)
    days = []
    for day in sorted(by_day):
        items = sorted(by_day[day], key=lambda hour: hour.at)
        if len(items) < MIN_DAY_HOURS:
            continue
        temperatures = [hour.temperature_c for hour in items]
        run = longest = 0
        for value in temperatures:
            run = run + 1 if GT_LOW <= value <= GT_HIGH else 0
            longest = max(longest, run)
        days.append(
            DailyWeather(
                day=day,
                mean_temp=round(sum(temperatures) / len(temperatures), 2),
                min_temp=min(temperatures),
                max_temp=max(temperatures),
                precipitation_mm=round(sum(h.precipitation_mm for h in items), 2),
                mean_humidity=round(
                    sum(h.relative_humidity for h in items) / len(items), 1
                ),
                longest_mild_run=longest,
                hours_above_35=sum(value > GT_HEAT for value in temperatures),
            )
        )
    return days


def in_window(day: date, end: tuple[int, int]) -> bool:
    return date(day.year, *SEASON_START) <= day <= date(day.year, *end)


def degree_days(days: Sequence[DailyWeather]) -> dict[date, float]:
    """Сумма активных температур (база 10 °C) с 1 марта года каждого дня."""
    totals: dict[date, float] = {}
    total = 0.0
    year = None
    for day in days:
        if day.day.year != year:
            year, total = day.day.year, 0.0
        if day.day >= date(day.day.year, 3, 1):
            total += max(0.0, day.mean_temp - GDD_BASE)
        totals[day.day] = round(total, 1)
    return totals


def three_tens(
    days: Sequence[DailyWeather], shoots_ready_from: date | None = None
) -> list[RiskDay]:
    """Правило «трёх десяток»: первичное заражение милдью возможно, когда
    среднесуточная температура не ниже 10 °C, за 48 ч выпало не меньше 10 мм и
    побеги достигли 10 см. shoots_ready_from — дата, с которой побеги длиннее
    10 см по наблюдению; без неё длина оценивается по сумме активных температур."""
    gdd = degree_days(days)
    risks = []
    previous: DailyWeather | None = None
    for day in days:
        if not in_window(day.day, PRIMARY_INFECTION_END):
            previous = day
            continue
        rain = day.precipitation_mm + (
            previous.precipitation_mm
            if previous and (day.day - previous.day).days == 1
            else 0.0
        )
        shoots = (
            day.day >= shoots_ready_from
            if shoots_ready_from is not None
            else gdd[day.day] >= SHOOTS_10_CM_GDD
        )
        infection = (
            day.mean_temp >= THREE_TENS_TEMPERATURE
            and rain >= THREE_TENS_RAIN_MM
            and shoots
        )
        risks.append(
            RiskDay(
                day=day.day,
                disease=Disease.DOWNY_MILDEW,
                method="three_tens",
                score=1.0 if infection else 0.0,
                level=RiskLevel.HIGH if infection else RiskLevel.LOW,
                factors={
                    "mean_temp": day.mean_temp,
                    "rain_48h_mm": round(rain, 1),
                    "degree_days": gdd[day.day],
                    "shoots_10_cm": shoots,
                },
            )
        )
        previous = day
    return risks


def incubation_days(mean_temp: float) -> float | None:
    """Длительность инкубационного периода при постоянной температуре; None —
    вне диапазона развития гриба."""
    first, last = GOIDANICH_DAYS[0], GOIDANICH_DAYS[-1]
    if mean_temp < first[0] or mean_temp > last[0]:
        return None
    for (t1, d1), (t2, d2) in pairwise(GOIDANICH_DAYS):
        if t1 <= mean_temp <= t2:
            return d1 + (d2 - d1) * (mean_temp - t1) / (t2 - t1)
    return last[1]  # pragma: no cover — диапазон покрыт узлами


def goidanich(
    days: Sequence[DailyWeather], infections: Iterable[date]
) -> list[RiskDay]:
    """Модель Гойданича: после заражения каждый день проходит долю
    инкубационного периода, равную 1 / длительность при температуре дня. Когда
    сумма доходит до 100 %, ожидаются пятна и новое спороношение — обработать
    участок нужно до этого. Оценка дня — самая продвинутая из идущих инкубаций."""
    starts = sorted(set(infections))
    active: list[tuple[date, float]] = []
    risks = []
    for day in days:
        if not in_window(day.day, SEASON_END):
            continue
        # Заражение в этот день начинает отсчёт со следующего дня
        progressed = []
        for started, progress in active:
            length = incubation_days(day.mean_temp)
            progressed.append((started, progress + (100.0 / length if length else 0.0)))
        finished = [item for item in progressed if item[1] >= 100]
        active = [item for item in progressed if item[1] < 100]
        active.extend((start, 0.0) for start in starts if start == day.day)
        if finished:
            started, _ = min(finished)
            score, level = 1.0, RiskLevel.HIGH
            factors: dict[str, Any] = {
                "infection_day": started.isoformat(),
                "progress": 100.0,
                "symptoms_expected": True,
            }
        elif active and max(p for _, p in active) > 0:
            started, progress = max(active, key=lambda item: item[1])
            score = round(progress / 100, 3)
            level = RiskLevel.MEDIUM if progress >= 60 else RiskLevel.LOW
            factors = {
                "infection_day": started.isoformat(),
                "progress": round(progress, 1),
                "symptoms_expected": False,
            }
        else:
            score, level, factors = 0.0, RiskLevel.LOW, {}
        factors["mean_temp"] = day.mean_temp
        risks.append(
            RiskDay(
                day=day.day,
                disease=Disease.DOWNY_MILDEW,
                method="goidanich",
                score=score,
                level=level,
                factors=factors,
            )
        )
    return risks


def gubler_thomas(days: Sequence[DailyWeather]) -> list[RiskDay]:
    """Индекс риска оидиума Gubler-Thomas, 0..100. Отсчёт начинается после трёх
    дней подряд с 6 часами подряд при 21-30 °C (эти три дня дают 60 баллов);
    дальше такой день прибавляет 20, день без этого условия вычитает 10, жара
    выше 35 °C вычитает ещё 10. 0-30 — низкий риск, 40-50 — средний, 60-100 —
    высокий: интервалы между обработками нужно сокращать."""
    index = 0
    started = False
    streak = 0
    risks = []
    for day in days:
        if not in_window(day.day, SEASON_END):
            # Новый сезон — индекс с нуля
            index, started, streak = 0, False, 0
            continue
        mild = day.longest_mild_run >= GT_HOURS
        if not started:
            streak = streak + 1 if mild else 0
            if streak >= 3:
                started, index = True, 60
        else:
            index += 20 if mild else -10
            if day.hours_above_35:
                index -= 10
            index = max(0, min(100, index))
        level = (
            RiskLevel.HIGH
            if index >= 60
            else RiskLevel.MEDIUM if index > 30 else RiskLevel.LOW
        )
        risks.append(
            RiskDay(
                day=day.day,
                disease=Disease.POWDERY_MILDEW,
                method="gubler_thomas",
                score=index / 100,
                level=level,
                factors={
                    "index": index,
                    "started": started,
                    "mild_hours": day.longest_mild_run,
                    "hours_above_35": day.hours_above_35,
                },
            )
        )
    return risks


def season_risks(
    hours: Iterable[HourlyWeather], shoots_ready_from: date | None = None
) -> list[RiskDay]:
    """Все три модели по одному ряду погоды."""
    days = daily_weather(hours)
    tens = three_tens(days, shoots_ready_from)
    infections = [risk.day for risk in tens if risk.score == 1]
    return [*tens, *goidanich(days, infections), *gubler_thomas(days)]
