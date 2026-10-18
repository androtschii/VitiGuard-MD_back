"""Аномалии NDVI и NDRE на участках: Isolation Forest против LOF (ml/pr-019).

    python -m vitiguard_ml.anomaly.experiment --data data/raw/ndvi/observations.csv

Размеченных «больных» участков на снимках нет, поэтому методы сравниваются на
встроенных аномалиях: в реальные ряды индексов в случайные снимки вносится
просадка NDVI известной величины (с соответствующей просадкой NDRE и ростом
неоднородности участка), и проверяется, какую долю таких снимков метод
находит и сколько ложных тревог даёт. Признаки снимка — отклонения от
собственной сезонной кривой участка и от соседних участков в те же дни: общая
для района засуха тогда не тревога, а просадка одного участка — тревога."""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

SELF_WINDOW_DAYS = 20
PEER_WINDOW_DAYS = 3
DROPS = (0.05, 0.10, 0.15, 0.20)
SHARE = 0.05
FEATURES = (
    "ndvi_self",
    "ndre_self",
    "ndvi_peer",
    "ndre_peer",
    "ndvi_std",
    "parcel_cloud",
)


def load_observations(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    # Снимки, закрытые облаками над участком, индексов не имеют
    clear = frame.dropna(subset=["ndvi_mean", "ndre_mean"])
    return clear.sort_values(["vineyard", "date"]).reset_index(drop=True)


def _self_residual(group: pd.DataFrame, column: str) -> pd.Series:
    """Отклонение от медианы своих снимков ± SELF_WINDOW_DAYS дней, без
    самого снимка: одна просадка не сдвигает собственную норму."""
    dates = group["date"].to_numpy()
    values = group[column].to_numpy()
    residual = np.full(len(group), np.nan)
    window = np.timedelta64(SELF_WINDOW_DAYS, "D")
    for i, date in enumerate(dates):
        near = (np.abs(dates - date) <= window) & (np.arange(len(group)) != i)
        if near.sum() >= 2:
            residual[i] = values[i] - np.median(values[near])
    return pd.Series(residual, index=group.index)


def _peer_residual(frame: pd.DataFrame, column: str) -> pd.Series:
    """Отклонение от медианы других участков ± PEER_WINDOW_DAYS дней."""
    dates = frame["date"].to_numpy()
    vineyards = frame["vineyard"].to_numpy()
    values = frame[column].to_numpy()
    residual = np.full(len(frame), np.nan)
    window = np.timedelta64(PEER_WINDOW_DAYS, "D")
    for i in range(len(frame)):
        near = (np.abs(dates - dates[i]) <= window) & (vineyards != vineyards[i])
        if near.sum() >= 3:
            residual[i] = values[i] - np.median(values[near])
    return pd.Series(residual, index=frame.index)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.copy()
    for column, name in (("ndvi_mean", "ndvi"), ("ndre_mean", "ndre")):
        features[f"{name}_self"] = pd.concat(
            _self_residual(group, column) for _, group in frame.groupby("vineyard")
        )
        features[f"{name}_peer"] = _peer_residual(frame, column)
    return features.dropna(subset=list(FEATURES))


def inject(
    frame: pd.DataFrame, drop: float, share: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Просадка NDVI около drop (от половины до полутора) в доле снимков share:
    NDRE проседает слабее, а неоднородность участка растёт — болезнь поражает
    участок пятнами."""
    damaged = frame.copy()
    chosen = rng.random(len(frame)) < share
    count = int(chosen.sum())
    # Величина у каждого снимка своя: одинаковые просадки собрались бы в
    # пространстве признаков в плотное облако, которое LOF принял бы за норму
    size = drop * rng.uniform(0.5, 1.5, count)
    damaged["injected"] = chosen
    damaged["drop"] = 0.0
    damaged.loc[chosen, "drop"] = size
    damaged.loc[chosen, "ndvi_mean"] -= size
    damaged.loc[chosen, "ndre_mean"] -= size * rng.uniform(0.4, 0.8, count)
    damaged.loc[chosen, "ndvi_std"] += size * rng.uniform(0.1, 0.5, count)
    return damaged


def scores(features: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    """Оценка аномальности снимков: больше — подозрительнее."""
    x = StandardScaler().fit_transform(features[list(FEATURES)])
    forest = IsolationForest(n_estimators=300, contamination="auto", random_state=seed)
    forest.fit(x)
    lof = LocalOutlierFactor(n_neighbors=20)
    lof.fit(x)
    # Базовая линия: насколько NDVI ниже своей нормы и соседей, в z-оценках
    drop = -(x[:, 0] + x[:, 2])
    return {
        "isolation_forest": -forest.score_samples(x),
        "lof": -lof.negative_outlier_factor_,
        "zscore": drop,
    }


def evaluate(labels: np.ndarray, value: np.ndarray, share: float) -> dict[str, float]:
    """PR-AUC, ROC-AUC и доля найденных при тревогах на долю share снимков."""
    alarms = value >= np.quantile(value, 1 - share)
    return {
        "pr_auc": round(float(average_precision_score(labels, value)), 4),
        "roc_auc": round(float(roc_auc_score(labels, value)), 4),
        "recall_at_alarms": round(float(alarms[labels].mean()), 4),
        "precision_at_alarms": round(float(labels[alarms].mean()), 4),
    }


def run(data: Path, out: Path, repeats: int = 10) -> dict[str, Any]:
    clean = load_observations(data)
    results: dict[str, Any] = {
        "observations": len(clean),
        "vineyards": int(clean["vineyard"].nunique()),
        "share": SHARE,
        "drops": {},
    }
    for drop in DROPS:
        runs: dict[str, list[dict[str, float]]] = {}
        for seed in range(repeats):
            damaged = inject(clean, drop, SHARE, np.random.default_rng(seed))
            features = build_features(damaged)
            labels = features["injected"].to_numpy(dtype=bool)
            for name, value in scores(features, seed).items():
                runs.setdefault(name, []).append(evaluate(labels, value, SHARE))
        results["drops"][str(drop)] = {
            name: {
                key: round(float(np.mean([r[key] for r in values])), 4)
                for key in values[0]
            }
            for name, values in runs.items()
        }
        print(
            f"просадка {drop}: "
            + ", ".join(
                f"{n} PR-AUC {v['pr_auc']}"
                for n, v in results["drops"][str(drop)].items()
            ),
            flush=True,
        )

    # Реальные данные без встроенных аномалий: что метод считает необычным
    features = build_features(clean.assign(injected=False))
    natural = scores(features, 0)["isolation_forest"]
    flagged = features.assign(score=natural).nlargest(10, "score")
    results["natural_top"] = [
        {
            "vineyard": str(row.vineyard),
            "date": row.date.date().isoformat(),
            "ndvi": float(row.ndvi_mean),
            "ndvi_self": round(float(row.ndvi_self), 3),
            "ndvi_peer": round(float(row.ndvi_peer), 3),
            "parcel_cloud": float(row.parcel_cloud),
        }
        for row in flagged.itertuples()
    ]
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot(results, out)
    plot_example(features.assign(score=natural), out)
    return results


def plot(results: dict[str, Any], out: Path) -> None:
    drops = [float(d) for d in results["drops"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    labels = {
        "isolation_forest": "Isolation Forest",
        "lof": "LOF",
        "zscore": "z-оценка просадки",
    }
    for name, label in labels.items():
        values = [results["drops"][str(d)][name] for d in drops]
        axes[0].plot(drops, [v["pr_auc"] for v in values], marker="o", label=label)
        axes[1].plot(
            drops, [v["recall_at_alarms"] for v in values], marker="o", label=label
        )
    axes[0].set_title("PR-AUC")
    axes[1].set_title(f"Доля найденных при тревогах на {SHARE:.0%} снимков")
    for ax in axes:
        ax.set_xlabel("Просадка NDVI")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "detection_by_drop.png", dpi=160)
    plt.close(fig)


def plot_example(features: pd.DataFrame, out: Path) -> None:
    """Ряд участка с самым подозрительным снимком и тревогами на нём."""
    worst = features.nlargest(1, "score").iloc[0]["vineyard"]
    series = features[features["vineyard"] == worst]
    threshold = features["score"].quantile(1 - SHARE)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    # Каждый сезон отдельной линией: зимой снимков нет, и линия через зиму
    # показывала бы несуществующие значения
    for i, (_, season) in enumerate(series.groupby(series["date"].dt.year)):
        ax.plot(
            season["date"],
            season["ndvi_mean"],
            marker=".",
            color="#622ac3",
            label="NDVI участка" if i == 0 else None,
        )
    alarms = series[series["score"] >= threshold]
    ax.scatter(
        alarms["date"],
        alarms["ndvi_mean"],
        s=80,
        facecolors="none",
        edgecolors="#b91c1c",
        linewidths=2,
        label="тревога Isolation Forest",
    )
    ax.set_title(f"Участок {worst}")
    ax.set_ylabel("NDVI")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "example_series.png", dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/ndvi/observations.csv")
    parser.add_argument("--out", default="results/anomaly")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args(argv)
    run(Path(args.data), Path(args.out), args.repeats)


if __name__ == "__main__":
    main()
