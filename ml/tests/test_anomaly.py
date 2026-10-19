import json
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")

from vitiguard_ml.anomaly import experiment  # noqa: E402


def synthetic(path: Path, vineyards: int = 12, days: int = 120) -> Path:
    """Сезонная кривая NDVI (рост, плато, спад) с шумом; снимок раз в 3 дня."""
    rng = np.random.default_rng(0)
    rows = []
    dates = pd.date_range("2026-04-01", periods=days // 3, freq="3D")
    for v in range(vineyards):
        offset = rng.normal(0, 0.03)
        for i, date in enumerate(dates):
            ndvi = (
                0.3
                + 0.4 * np.sin(np.pi * i / len(dates))
                + offset
                + rng.normal(0, 0.015)
            )
            rows.append(
                {
                    "vineyard": f"way/{v}",
                    "point": "cricova",
                    "area_ha": 5.0,
                    "date": date.date().isoformat(),
                    "platform": "sentinel-2a",
                    "scene_cloud": 5.0,
                    "parcel_cloud": 0.0,
                    "parcel_pixels": 500,
                    "clear_pixels": 500,
                    "ndvi_mean": round(ndvi, 4),
                    "ndvi_std": 0.05,
                    "ndvi_min": 0.1,
                    "ndvi_max": 0.9,
                    "ndre_mean": round(ndvi * 0.6, 4),
                    "ndre_std": 0.03,
                }
            )
    # Облачный снимок без индексов — отбрасывается при загрузке
    rows.append({**rows[0], "date": "2026-09-01", "ndvi_mean": None, "ndre_mean": None})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_residuals_exclude_the_snapshot_and_its_own_vineyard(tmp_path: Path) -> None:
    frame = experiment.load_observations(synthetic(tmp_path / "obs.csv"))
    one = frame[frame["vineyard"] == "way/0"].copy()
    one.loc[one.index[10], "ndvi_mean"] -= 0.3

    self_residual = experiment._self_residual(one, "ndvi_mean")

    # Просадка видна в собственном отклонении и не сдвигает норму соседних снимков
    assert self_residual.iloc[10] < -0.25
    assert abs(self_residual.iloc[9]) < 0.1
    peers = experiment._peer_residual(frame, "ndvi_mean")
    assert peers.notna().all()
    assert len(frame) == 12 * 40


def test_injection_hits_the_requested_share(tmp_path: Path) -> None:
    frame = experiment.load_observations(synthetic(tmp_path / "obs.csv"))

    damaged = experiment.inject(frame, 0.2, 0.25, np.random.default_rng(1))

    chosen = damaged["injected"]
    assert 0.15 < chosen.mean() < 0.35
    difference = frame["ndvi_mean"] - damaged["ndvi_mean"]
    assert difference[chosen].between(0.1, 0.3).all()
    assert np.allclose(difference[chosen], damaged["drop"][chosen])
    assert np.allclose(difference[~chosen], 0)


def test_experiment_finds_large_drops(tmp_path: Path) -> None:
    results = experiment.run(
        synthetic(tmp_path / "obs.csv"), tmp_path / "out", repeats=2
    )

    saved = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert saved["vineyards"] == 12
    assert set(saved["drops"]) == {"0.05", "0.1", "0.15", "0.2"}
    strong = results["drops"]["0.2"]
    assert set(strong) == {"isolation_forest", "lof", "zscore"}
    # Просадка 0,2 при шуме 0,015 видна всем методам куда лучше случайности
    for method in strong.values():
        assert method["pr_auc"] > 0.5
    assert len(results["natural_top"]) == 10
    assert (tmp_path / "out" / "detection_by_drop.png").exists()
    assert (tmp_path / "out" / "example_series.png").exists()
