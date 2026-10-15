from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("matplotlib")

from vitiguard_ml.experiments import by_source, load_results, save_results
from vitiguard_ml.training import Sample

CLASSES = ["healthy", "powdery_mildew"]


def test_by_source_splits_metrics_per_source() -> None:
    samples = [
        Sample(Path("a.jpg"), 0, "plantvillage"),
        Sample(Path("b.jpg"), 1, "plantvillage"),
        Sample(Path("c.jpg"), 0, "ngld"),
        Sample(Path("d.jpg"), 1, "ngld"),
    ]
    reports = by_source(samples, [0, 1, 0, 1], [0, 1, 1, 1], CLASSES)

    assert set(reports) == {"ngld", "plantvillage"}
    assert reports["plantvillage"]["accuracy"] == 1.0
    assert reports["ngld"]["accuracy"] == 0.5


def test_results_round_trip_keeps_previous_stages(tmp_path: Path) -> None:
    assert load_results(tmp_path) == {}
    save_results(tmp_path, {"classification": {"resnet50": {"Здоровый": 1}}})
    results = load_results(tmp_path)
    results["export"] = {"model": "resnet50"}
    save_results(tmp_path, results)

    assert load_results(tmp_path) == {
        "classification": {"resnet50": {"Здоровый": 1}},
        "export": {"model": "resnet50"},
    }
