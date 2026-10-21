"""Признаки сервера (pr-075) совпадают с признаками обучения: модуль бэкенда
без зависимостей читается по пути и сравнивается с build_features."""

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from vitiguard_ml.risk.features import (  # noqa: E402
    build_features,
    feature_columns,
    load_daily,
)

SERVER = Path(__file__).parents[2] / "app" / "services" / "risk_features.py"


def server_features() -> Any:
    spec = importlib.util.spec_from_file_location("server_risk_features", SERVER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def daily_csv(path: Path) -> Path:
    """Одна точка, 2019 год: случайная погода и разметка в окнах моделей."""
    rng = np.random.default_rng(1)
    dates = pd.date_range("2019-01-01", "2019-12-31")
    count = len(dates)
    spring = (dates.month >= 4) & (dates.month <= 6)
    season = (dates.month >= 4) & (dates.month <= 9)
    temp = rng.normal(15, 6, count).round(2)
    frame = pd.DataFrame(
        {
            "point": "cricova",
            "region": "codru",
            "date": dates.strftime("%Y-%m-%d"),
            "mean_temp": temp,
            "min_temp": temp - 5,
            "max_temp": temp + 5,
            "precipitation_mm": (
                rng.exponential(5, count) * (rng.random(count) < 0.3)
            ).round(2),
            "mean_humidity": rng.uniform(40, 95, count).round(1),
            "mild_hours": rng.integers(0, 12, count),
            "hours_above_35": 0,
            "degree_days": np.cumsum(np.maximum(temp - 10, 0)).round(1),
            "three_tens": np.where(spring, rng.integers(0, 2, count), np.nan),
            "goidanich_progress": np.where(season, rng.uniform(0, 100, count), np.nan),
            "symptoms_expected": np.where(season, 0, np.nan),
            "gubler_thomas": np.where(season, rng.integers(0, 11, count) * 10, np.nan),
        }
    )
    frame.to_csv(path, index=False)
    return path


def test_server_builds_the_same_features(tmp_path: Path) -> None:
    server = server_features()
    daily = load_daily(daily_csv(tmp_path / "daily.csv"))
    frame = build_features(daily)

    assert feature_columns(frame) == list(server.FEATURES)
    # Начало окна «трёх десяток», середина лета и день сразу после окон
    for day in ("2019-04-03", "2019-07-15", "2019-10-02"):
        history = daily[daily["date"] <= day].tail(server.HISTORY_DAYS)
        rows = [
            {key: "" if pd.isna(value) else value for key, value in row.items()}
            for row in history.to_dict("records")
        ]
        expected = frame[frame["date"] == day].iloc[0][list(server.FEATURES)]

        assert server.vector(rows) == pytest.approx(expected.astype(float).tolist())
