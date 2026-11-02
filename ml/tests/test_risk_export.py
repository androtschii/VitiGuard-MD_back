"""Экспорт модели риска для сервера (pr-076) на маленьком искусственном архиве."""

import json
from pathlib import Path

import numpy as np
import pytest

for module in ("pandas", "lightgbm", "onnxmltools", "onnxruntime", "torch", "sklearn"):
    pytest.importorskip(module)
pytest.importorskip("matplotlib")

import pandas as pd  # noqa: E402

from vitiguard_ml.risk.export_server import train_and_export  # noqa: E402


def fake_archive(path: Path) -> Path:
    """Три года по двум точкам: заражение — после дождя больше 8 мм в апреле-июне."""
    rng = np.random.default_rng(0)
    parts = []
    for point in ("a", "b"):
        dates = pd.date_range("2023-01-01", "2025-12-31", freq="D")
        rain = rng.gamma(0.6, 6, len(dates)).round(1)
        window = dates.month.isin([4, 5, 6])
        three_tens = np.where(window, (rain > 8).astype(int), np.nan)
        parts.append(
            pd.DataFrame(
                {
                    "point": point,
                    "region": "codru",
                    "date": dates,
                    "mean_temp": 15 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365),
                    "max_temp": 22 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365),
                    "precipitation_mm": rain,
                    "mean_humidity": rng.uniform(50, 95, len(dates)),
                    "mild_hours": rng.integers(0, 12, len(dates)),
                    "degree_days": np.arange(len(dates)) % 365 * 2.0,
                    "three_tens": three_tens,
                    "goidanich_progress": np.where(window, 0.2, np.nan),
                    "gubler_thomas": np.where(window, 40.0, np.nan),
                }
            )
        )
    pd.concat(parts).to_csv(path, index=False)
    return path


def test_export_matches_lightgbm_decisions(tmp_path: Path) -> None:
    meta = train_and_export(fake_archive(tmp_path / "daily.csv"), tmp_path / "out")

    assert (tmp_path / "out" / "risk.onnx").exists()
    saved = json.loads((tmp_path / "out" / "risk.json").read_text(encoding="utf-8"))
    assert saved == meta
    assert meta["target"] == "downy_h1"
    assert meta["valid_year"] == 2025
    assert meta["trained_years"] == [2023, 2024]
    assert "rain_2d" in meta["features"]
    assert meta["onnx_check"]["same_decision"] >= 0.99
