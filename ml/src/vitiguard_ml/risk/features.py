"""Признаки и цели прогноза риска (ml/pr-015).

Строка — точка и день t. Признаки строятся только по дням не позже t: погода
последних двух недель, суммы осадков, часы мягкой погоды, сумма активных
температур, время года и состояние агромоделей на день t. Цели — состояние
моделей через h дней (h = 1..7): будет ли первичное заражение милдью по
«трём десяткам» и будет ли высокий риск оидиума (индекс Gubler-Thomas ≥ 60).

Прогноз погоды в признаки не входит: модель предсказывает риск по одной
истории, а сравнение с базовыми линиями показывает, насколько это вообще
возможно (ml/pr-018)."""

from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = range(1, 8)
LAGS = range(0, 14)
WEATHER = ("mean_temp", "max_temp", "precipitation_mm", "mean_humidity", "mild_hours")
GT_HIGH = 60
TARGETS = ("downy", "powdery")


def load_daily(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    return frame.sort_values(["point", "date"]).reset_index(drop=True)


def _point_features(group: pd.DataFrame) -> pd.DataFrame:
    """Признаки одной точки; в group дни идут подряд и по возрастанию."""
    features: dict[str, pd.Series] = {}
    for column in WEATHER:
        for lag in LAGS:
            features[f"{column}_lag{lag}"] = group[column].shift(lag)
    rain = group["precipitation_mm"]
    for window in (2, 3, 7, 14):
        features[f"rain_{window}d"] = rain.rolling(window).sum()
    features["temp_7d"] = group["mean_temp"].rolling(7).mean()
    features["humidity_7d"] = group["mean_humidity"].rolling(7).mean()
    features["mild_days_7d"] = (group["mild_hours"] >= 6).rolling(7).sum()
    features["degree_days"] = group["degree_days"]
    day_of_year = group["date"].dt.dayofyear
    features["season_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["season_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    # Состояние агромоделей на день t: вне окна моделей — ноль
    features["gt_index"] = group["gubler_thomas"].fillna(0)
    features["incubation"] = group["goidanich_progress"].fillna(0)
    infection = group["three_tens"].fillna(0)
    features["infection_today"] = infection
    features["infections_7d"] = infection.rolling(7).sum()
    for horizon in HORIZONS:
        downy = group["three_tens"].shift(-horizon)
        powdery = group["gubler_thomas"].shift(-horizon)
        # Пусто — день t + h вне окна модели: такая пара не обучает и не проверяет
        features[f"downy_h{horizon}"] = downy
        features[f"powdery_h{horizon}"] = (powdery >= GT_HIGH).where(powdery.notna())
    return pd.DataFrame(features, index=group.index)


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in daily.groupby("point", sort=False):
        frame = _point_features(group)
        frame.insert(0, "point", group["point"])
        frame.insert(1, "region", group["region"])
        frame.insert(2, "date", group["date"])
        parts.append(frame)
    # Первые две недели точки без полной истории лагов — отбрасываются
    return pd.concat(parts).dropna(subset=["mean_temp_lag13", "rain_14d"])


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in ("point", "region", "date")
        and not column.startswith(("downy_h", "powdery_h"))
    ]


def target_column(target: str, horizon: int) -> str:
    return f"{target}_h{horizon}"
