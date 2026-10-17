import json
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("lightgbm")
pytest.importorskip("torch")

from vitiguard_ml.risk import experiment  # noqa: E402
from vitiguard_ml.risk.features import build_features, load_daily  # noqa: E402
from vitiguard_ml.risk.models import (  # noqa: E402
    class_weight,
    climatology,
    persistence,
)


def synthetic_daily(path: Path, years: range = range(2018, 2023)) -> Path:
    """Две точки: сезонная температура, случайные дожди; «три десятки» — по
    дождю за 48 ч в апреле — июне, индекс оидиума растёт к середине лета."""
    rng = np.random.default_rng(0)
    rows = []
    for point in ("north", "south"):
        dates = pd.date_range(f"{years.start}-01-01", f"{years.stop - 1}-12-31")
        day_of_year = dates.dayofyear.to_numpy()
        temp = (
            11
            - 13 * np.cos(2 * np.pi * day_of_year / 365)
            + rng.normal(0, 2, len(dates))
        )
        rain = rng.exponential(6.0, len(dates)) * (rng.random(len(dates)) < 0.3)
        rain_48h = rain + np.concatenate([[0.0], rain[:-1]])
        spring = (dates.month >= 4) & (dates.month <= 6)
        season = (dates.month >= 4) & (dates.month <= 9)
        gt = np.clip((temp - 12) * 6, 0, 100)
        for i, date in enumerate(dates):
            rows.append(
                {
                    "point": point,
                    "region": "codru",
                    "date": date.date().isoformat(),
                    "mean_temp": round(temp[i], 2),
                    "min_temp": round(temp[i] - 5, 2),
                    "max_temp": round(temp[i] + 5, 2),
                    "precipitation_mm": round(rain[i], 2),
                    "mean_humidity": 70.0,
                    "mild_hours": int(temp[i] > 20) * 7,
                    "hours_above_35": 0,
                    "degree_days": float(max(0, day_of_year[i] - 60)),
                    "three_tens": int(rain_48h[i] >= 10) if spring[i] else "",
                    "goidanich_progress": 0.0 if season[i] else "",
                    "symptoms_expected": 0 if season[i] else "",
                    "gubler_thomas": round(gt[i]) if season[i] else "",
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_features_look_only_back_and_targets_look_ahead(tmp_path: Path) -> None:
    daily = load_daily(synthetic_daily(tmp_path / "daily.csv"))
    frame = build_features(daily)

    row = frame[(frame["point"] == "north") & (frame["date"] == "2019-05-10")].iloc[0]
    day = daily.set_index(["point", "date"])
    today = day.loc[("north", pd.Timestamp("2019-05-10"))]
    tomorrow = day.loc[("north", pd.Timestamp("2019-05-11"))]
    week = day.loc[("north", pd.Timestamp("2019-05-17"))]
    assert row["mean_temp_lag0"] == today["mean_temp"]
    assert (
        row["precipitation_mm_lag1"]
        == day.loc[("north", pd.Timestamp("2019-05-09"))]["precipitation_mm"]
    )
    assert row["downy_h1"] == tomorrow["three_tens"]
    assert row["powdery_h7"] == (week["gubler_thomas"] >= 60)
    # Цель вне окна модели — пусто, а не «нет события»
    july = frame[(frame["point"] == "north") & (frame["date"] == "2019-07-10")].iloc[0]
    assert np.isnan(july["downy_h1"])
    # Первые две недели точки без полной истории не используются
    first = frame.groupby("point")["date"].min()
    assert (first == pd.Timestamp("2018-01-14")).all()


def test_baselines() -> None:
    frame = pd.DataFrame(
        {
            "infection_today": [0.0, 1.0],
            "gt_index": [20.0, 80.0],
            "date": pd.to_datetime(["2020-05-01", "2020-05-02"]),
            "target": [0.0, 1.0],
        }
    )

    assert persistence(frame, "downy").tolist() == [0, 1]
    assert persistence(frame, "powdery").tolist() == [0.2, 0.8]
    assert climatology(frame, frame, "target", window=0).tolist() == [0, 1]
    # Дней рядом в обучении нет — общая частота
    far = frame.assign(date=pd.to_datetime(["2020-12-01", "2020-12-02"]))
    assert climatology(frame, far, "target", window=0).tolist() == [0.5, 0.5]


def test_class_weight() -> None:
    assert class_weight(0.1) == pytest.approx(9)
    assert class_weight(0.0) == 1
    assert class_weight(0.7) == 1


def test_threshold_and_metrics() -> None:
    labels = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.7, 0.4, 0.9])

    threshold = experiment.best_threshold(labels, scores)
    result = experiment.metrics(labels, scores, threshold)

    assert 0.4 < threshold <= 0.7
    assert result == {
        "pr_auc": 1.0,
        "roc_auc": 1.0,
        "f1": 1.0,
        "positive_rate": 0.5,
        "count": 6,
    }


def test_walk_forward_experiment(tmp_path: Path) -> None:
    data = synthetic_daily(tmp_path / "daily.csv")

    results = experiment.run(data, tmp_path / "out", test_years=(2021, 2022), epochs=2)

    saved = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert saved["test_years"] == [2021, 2022]
    for target in ("downy", "powdery"):
        models = results["targets"][target]["models"]
        assert set(models) == {"persistence", "climatology", "lightgbm", "lstm"}
        assert set(models["lightgbm"]) == {str(h) for h in range(1, 8)}
        assert 0 <= models["lstm"]["1"]["pr_auc"] <= 1
    # Дождь за 48 ч — главный признак завтрашнего заражения
    top = list(results["targets"]["downy"]["top_features"])[:3]
    assert any(name.startswith(("precipitation", "rain")) for name in top)
    assert (tmp_path / "out" / "pr_auc_by_horizon.png").exists()
    assert (tmp_path / "out" / "f1_by_horizon.png").exists()
