"""Сравнение моделей прогноза риска (ml/pr-018).

    python -m vitiguard_ml.risk.experiment --data data/raw/risk/daily.csv --out results/risk

Проверка по годам (walk-forward): для каждого тестового года 2021-2025 модели
обучаются на всех предыдущих годах, последний из них — проверочный (ранняя
остановка и порог F1). Так модель никогда не видит будущего. Метрики считаются
по прогнозам всех тестовых лет вместе, отдельно для каждого горизонта."""

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from vitiguard_ml.risk.features import (
    HORIZONS,
    TARGETS,
    build_features,
    feature_columns,
    load_daily,
    target_column,
)
from vitiguard_ml.risk.models import (
    SEQUENCE_DAYS,
    climatology,
    persistence,
    predict_boosting,
    predict_lstm,
    sequences,
    train_boosting,
    train_lstm,
)

TEST_YEARS = range(2021, 2026)
MODELS = ("persistence", "climatology", "lightgbm", "lstm")
TITLES = {
    "downy": "Милдью: заражение по «трём десяткам»",
    "powdery": "Оидиум: индекс Gubler-Thomas ≥ 60",
}


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Порог с наибольшим F1 на проверочном году."""
    candidates = np.unique(np.round(scores, 3))
    if len(candidates) > 200:
        candidates = np.quantile(scores, np.linspace(0.01, 0.99, 200))
    best, best_f1 = 0.5, -1.0
    for threshold in candidates:
        score = f1_score(labels, scores >= threshold, zero_division=0)
        if score > best_f1:
            best, best_f1 = float(threshold), score
    return best


def metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    return {
        "pr_auc": round(float(average_precision_score(labels, scores)), 4),
        "roc_auc": round(float(roc_auc_score(labels, scores)), 4),
        "f1": round(float(f1_score(labels, scores >= threshold, zero_division=0)), 4),
        "positive_rate": round(float(labels.mean()), 4),
        "count": len(labels),
    }


def with_targets(part: pd.DataFrame, target: str) -> pd.DataFrame:
    """Строки, где известна цель хотя бы на одном горизонте."""
    columns = [target_column(target, h) for h in HORIZONS]
    known: pd.DataFrame = part[part[columns].notna().any(axis=1)]
    return known


def run(
    data: Path,
    out: Path,
    test_years: range | tuple[int, ...] = TEST_YEARS,
    epochs: int = 25,
) -> dict[str, Any]:
    daily = load_daily(data)
    frame = build_features(daily)
    # LSTM нужна история в SEQUENCE_DAYS дней — те же строки и для остальных
    # моделей, чтобы сравнение шло на одних и тех же днях
    start = daily.groupby("point")["date"].transform("min") + pd.Timedelta(
        days=SEQUENCE_DAYS
    )
    frame = frame[frame["date"] >= start.loc[frame.index]]
    daily_by_index = daily.set_index(daily.index)
    daily_by_index = daily_by_index.assign(
        season_sin=np.sin(2 * np.pi * daily["date"].dt.dayofyear / 365.25),
        season_cos=np.cos(2 * np.pi * daily["date"].dt.dayofyear / 365.25),
        gt_index=daily["gubler_thomas"].fillna(0),
        incubation=daily["goidanich_progress"].fillna(0),
        infection_today=daily["three_tens"].fillna(0),
    )
    columns = feature_columns(frame)
    year = frame["date"].dt.year

    collected: dict[tuple[str, str, int], dict[str, list[np.ndarray]]] = {}
    importance: dict[str, list[dict[str, float]]] = {target: [] for target in TARGETS}
    for target in TARGETS:
        for test_year in test_years:
            train_all = frame[year < test_year]
            fit = train_all[train_all["date"].dt.year < test_year - 1]
            valid = train_all[train_all["date"].dt.year == test_year - 1]
            test = frame[year == test_year]
            print(
                f"{target} {test_year}: обучение {len(fit)}, проверка {len(valid)}, тест {len(test)}",
                flush=True,
            )

            # LSTM — одна сеть на все горизонты
            fit_seq, valid_seq, test_seq = (
                with_targets(part, target) for part in (fit, valid, test)
            )
            lstm = train_lstm(
                sequences(daily_by_index, fit_seq, fit_seq.index),
                fit_seq,
                sequences(daily_by_index, valid_seq, valid_seq.index),
                valid_seq,
                target,
                epochs=epochs,
            )
            lstm_valid = predict_lstm(
                lstm, sequences(daily_by_index, valid_seq, valid_seq.index)
            )
            lstm_test = predict_lstm(
                lstm, sequences(daily_by_index, test_seq, test_seq.index)
            )

            for horizon in HORIZONS:
                column = target_column(target, horizon)
                fit_h = fit.dropna(subset=[column])
                valid_h = valid.dropna(subset=[column])
                test_h = test.dropna(subset=[column])
                if test_h.empty or valid_h[column].nunique() < 2:
                    continue
                boosting = train_boosting(fit_h, valid_h, columns, column)
                if horizon == 1:
                    importance[target].append(boosting.importance)
                lstm_v = pd.Series(
                    lstm_valid[:, horizon - 1], index=valid_seq.index
                ).loc[valid_h.index]
                lstm_t = pd.Series(lstm_test[:, horizon - 1], index=test_seq.index).loc[
                    test_h.index
                ]
                scores = {
                    "persistence": (
                        persistence(valid_h, target),
                        persistence(test_h, target),
                    ),
                    "climatology": (
                        climatology(fit, valid_h, column),
                        climatology(train_all, test_h, column),
                    ),
                    "lightgbm": (
                        predict_boosting(boosting, valid_h, columns),
                        predict_boosting(boosting, test_h, columns),
                    ),
                    "lstm": (lstm_v.to_numpy(), lstm_t.to_numpy()),
                }
                y_valid = valid_h[column].to_numpy(dtype=int)
                y_test = test_h[column].to_numpy(dtype=int)
                for name, (valid_scores, test_scores) in scores.items():
                    threshold = (
                        0.5
                        if name == "persistence"
                        else best_threshold(y_valid, valid_scores)
                    )
                    entry = collected.setdefault(
                        (target, name, horizon),
                        {"labels": [], "scores": [], "hits": []},
                    )
                    entry["labels"].append(y_test)
                    entry["scores"].append(test_scores)
                    entry["hits"].append((test_scores >= threshold).astype(int))

    results: dict[str, Any] = {"test_years": list(test_years), "targets": {}}
    for target in TARGETS:
        table: dict[str, dict[str, Any]] = {}
        for name in MODELS:
            per_horizon = {}
            for horizon in HORIZONS:
                pooled = collected.get((target, name, horizon))
                if not pooled:
                    continue
                labels = np.concatenate(pooled["labels"])
                pooled_scores = np.concatenate(pooled["scores"])
                hits = np.concatenate(pooled["hits"])
                values = metrics(labels, pooled_scores, 0.5)
                values["f1"] = round(float(f1_score(labels, hits, zero_division=0)), 4)
                per_horizon[str(horizon)] = values
            table[name] = per_horizon
        mean_importance = (
            pd.DataFrame(importance[target]).mean().sort_values(ascending=False)
        )
        results["targets"][target] = {
            "models": table,
            "top_features": {
                k: round(float(v), 4) for k, v in mean_importance.head(15).items()
            },
        }

    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot(results, out)
    return results


def plot(results: dict[str, Any], out: Path) -> None:
    for metric, title in (("pr_auc", "PR-AUC"), ("f1", "F1")):
        fig, axes = plt.subplots(1, len(TARGETS), figsize=(11, 3.8))
        for ax, target in zip(axes, TARGETS, strict=True):
            models = results["targets"][target]["models"]
            for name in MODELS:
                horizons = sorted(int(h) for h in models[name])
                ax.plot(
                    horizons,
                    [models[name][str(h)][metric] for h in horizons],
                    marker="o",
                    label=name,
                )
            ax.set_title(TITLES[target], fontsize=10)
            ax.set_xlabel("Горизонт, дней")
            ax.set_ylabel(title)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{metric}_by_horizon.png", dpi=160)
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/risk/daily.csv")
    parser.add_argument("--out", default="results/risk")
    args = parser.parse_args(argv)
    # LightGBM предупреждает о будущем изменении проверки eval_set — на расчёт
    # это не влияет, а вывод засоряет
    warnings.filterwarnings("ignore", category=FutureWarning, module="lightgbm")
    results = run(Path(args.data), Path(args.out))
    for target, entry in results["targets"].items():
        print(f"== {target}")
        for name, horizons in entry["models"].items():
            row = " ".join(
                f"h{h}: {v['pr_auc']:.2f}/{v['f1']:.2f}" for h, v in horizons.items()
            )
            print(f"  {name:<12} PR-AUC/F1  {row}")


if __name__ == "__main__":
    main()
