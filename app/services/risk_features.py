"""Признаки прогноза риска на день t — тот же набор и порядок, что в
vitiguard_ml.risk.features (ml/pr-015), на которых обучена модель (ml/pr-018).

В ml признаки строятся по всему архиву сразу через pandas, на сервере нужен
один вектор для участка на сегодня, поэтому здесь — те же правила на чистом
Python без зависимостей: модуль подключается и к ml, где совпадение двух
реализаций проверяет тест. Вход — суточные строки в формате набора данных
ml/pr-014 (app.cli.risk_dataset): пустая разметка — день вне окна модели."""

import math
from collections.abc import Mapping, Sequence
from datetime import date
from itertools import pairwise
from typing import Any

LAGS = range(0, 14)
WEATHER = ("mean_temp", "max_temp", "precipitation_mm", "mean_humidity", "mild_hours")
RAIN_WINDOWS = (2, 3, 7, 14)
MILD_DAY_HOURS = 6
HISTORY_DAYS = max(len(LAGS), *RAIN_WINDOWS)

FEATURES = (
    *(f"{column}_lag{lag}" for column in WEATHER for lag in LAGS),
    *(f"rain_{window}d" for window in RAIN_WINDOWS),
    "temp_7d",
    "humidity_7d",
    "mild_days_7d",
    "degree_days",
    "season_sin",
    "season_cos",
    "gt_index",
    "incubation",
    "infection_today",
    "infections_7d",
)

Row = Mapping[str, Any]


def _label(value: Any) -> float:
    # Вне окна модели состояние — ноль, как fillna(0) в ml
    return 0.0 if value is None or value == "" else float(value)


def _day(value: date | str) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value[:10])


def features(rows: Sequence[Row]) -> dict[str, float]:
    """Признаки последнего дня rows. Строки — сутки подряд по возрастанию, их
    нужно не меньше двух недель: без полной истории лагов строка в ml
    отбрасывается, и модель таких не видела."""
    if len(rows) < HISTORY_DAYS:
        raise ValueError(f"Нужна история не меньше {HISTORY_DAYS} суток")
    days = [_day(row["date"]) for row in rows[-HISTORY_DAYS:]]
    if any((later - earlier).days != 1 for earlier, later in pairwise(days)):
        raise ValueError("Сутки истории должны идти подряд")
    today = rows[-1]
    week = rows[-7:]
    values: dict[str, float] = {}
    for column in WEATHER:
        for lag in LAGS:
            values[f"{column}_lag{lag}"] = float(rows[-1 - lag][column])
    for window in RAIN_WINDOWS:
        values[f"rain_{window}d"] = math.fsum(
            float(row["precipitation_mm"]) for row in rows[-window:]
        )
    values["temp_7d"] = math.fsum(float(row["mean_temp"]) for row in week) / 7
    values["humidity_7d"] = math.fsum(float(row["mean_humidity"]) for row in week) / 7
    values["mild_days_7d"] = float(
        sum(float(row["mild_hours"]) >= MILD_DAY_HOURS for row in week)
    )
    values["degree_days"] = float(today["degree_days"])
    day_of_year = days[-1].timetuple().tm_yday
    values["season_sin"] = math.sin(2 * math.pi * day_of_year / 365.25)
    values["season_cos"] = math.cos(2 * math.pi * day_of_year / 365.25)
    values["gt_index"] = _label(today["gubler_thomas"])
    values["incubation"] = _label(today["goidanich_progress"])
    values["infection_today"] = _label(today["three_tens"])
    values["infections_7d"] = math.fsum(_label(row["three_tens"]) for row in week)
    return values


def vector(rows: Sequence[Row]) -> list[float]:
    """Признаки в порядке столбцов, на которых обучена модель."""
    values = features(rows)
    return [values[name] for name in FEATURES]
