import math
from datetime import UTC, date, datetime, timedelta

import pytest

from app.cli.risk_dataset import Point, label_rows
from app.services.agromodels import HourlyWeather
from app.services.risk_features import FEATURES, HISTORY_DAYS, features, vector


def row(day: date, **values: object) -> dict[str, object]:
    return {
        "date": day.isoformat(),
        "mean_temp": 15.0,
        "max_temp": 20.0,
        "precipitation_mm": 0.0,
        "mean_humidity": 70.0,
        "mild_hours": 0,
        "degree_days": 100.0,
        "three_tens": "",
        "goidanich_progress": "",
        "gubler_thomas": "",
        **values,
    }


def fortnight(end: date = date(2025, 5, 20)) -> list[dict[str, object]]:
    return [row(end - timedelta(days=13 - i)) for i in range(14)]


def test_feature_set_matches_training() -> None:
    # 5 рядов погоды по 14 лагов, 4 суммы осадков и 10 признаков дня
    assert len(FEATURES) == 84
    assert len(set(FEATURES)) == 84
    assert FEATURES[:2] == ("mean_temp_lag0", "mean_temp_lag1")
    assert FEATURES[-1] == "infections_7d"
    assert HISTORY_DAYS == 14


def test_lags_and_windows_look_back_from_today() -> None:
    rows = fortnight()
    for index, item in enumerate(rows):
        item["precipitation_mm"] = float(index)
        item["mean_temp"] = 10.0 + index
    rows[-1].update(three_tens=1, gubler_thomas=40, goidanich_progress=35.5)
    rows[-3].update(three_tens=1, mild_hours=7)
    rows[-8].update(three_tens=1)

    values = features(rows)

    assert values["mean_temp_lag0"] == 23
    assert values["mean_temp_lag13"] == 10
    assert values["precipitation_mm_lag1"] == 12
    assert values["rain_2d"] == 13 + 12
    assert values["rain_14d"] == sum(range(14))
    assert values["temp_7d"] == pytest.approx(sum(range(17, 24)) / 7)
    assert values["mild_days_7d"] == 1
    assert values["gt_index"] == 40
    assert values["incubation"] == 35.5
    assert values["infection_today"] == 1
    # Заражение восемь дней назад в недельную сумму не входит
    assert values["infections_7d"] == 2


def test_outside_model_windows_state_is_zero() -> None:
    values = features(fortnight(date(2025, 1, 20)))

    assert values["gt_index"] == values["incubation"] == 0
    assert values["infection_today"] == values["infections_7d"] == 0
    assert values["season_sin"] == pytest.approx(math.sin(2 * math.pi * 20 / 365.25))


def test_vector_follows_training_order() -> None:
    rows = fortnight()
    rows[-1]["mean_temp"] = 21.5

    assert vector(rows)[FEATURES.index("mean_temp_lag0")] == 21.5
    assert len(vector(rows)) == len(FEATURES)


def test_history_must_be_full_and_continuous() -> None:
    with pytest.raises(ValueError, match="14"):
        features(fortnight()[1:])
    gap = fortnight()
    del gap[5]
    with pytest.raises(ValueError, match="подряд"):
        features([row(date(2025, 5, 1)), *gap])


def test_rows_of_dataset_command_are_accepted() -> None:
    # Строки набора данных ml/pr-014: те же функции дают признаки и на сервере
    start = datetime(2025, 5, 1, tzinfo=UTC)
    hours = [
        HourlyWeather(start + timedelta(hours=i), 22.0, 80.0, 1.0 if i < 24 else 0.0)
        for i in range(24 * 16)
    ]
    rows = label_rows(Point("cricova", "codru", 47.14, 28.86), hours)

    values = features(rows)

    assert values["mean_temp_lag0"] == 22
    assert values["mild_days_7d"] == 7
    assert values["gt_index"] == 100
