import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from vitiguard_ml.robot.nightly import (
    NightlyRetrainer,
    count_new_samples,
    evaluate_gate,
)


def test_evaluate_gate_first_baseline() -> None:
    cand = {"map50_95": 0.40, "trunk_map50_95": 0.35, "bunch_map50_95": 0.45}
    decision = evaluate_gate(cand, None)
    assert decision.accepted
    assert "Базовая модель" in decision.reason


def test_evaluate_gate_valid_improvement() -> None:
    base = {"map50_95": 0.400, "trunk_map50_95": 0.350, "bunch_map50_95": 0.450}
    # Общий рост на 0.02, ствол вырос, гроздь просела на 0.005 (в пределах 0.01)
    cand = {"map50_95": 0.420, "trunk_map50_95": 0.380, "bunch_map50_95": 0.445}
    decision = evaluate_gate(cand, base)
    assert decision.accepted
    assert decision.overall_diff == 0.02
    assert "Принято" in decision.reason


def test_evaluate_gate_rejected_no_overall_gain() -> None:
    base = {"map50_95": 0.400, "trunk_map50_95": 0.350, "bunch_map50_95": 0.450}
    cand = {"map50_95": 0.395, "trunk_map50_95": 0.360, "bunch_map50_95": 0.440}
    decision = evaluate_gate(cand, base)
    assert not decision.accepted
    assert "не превысила" in decision.reason


def test_evaluate_gate_rejected_class_drop() -> None:
    base = {"map50_95": 0.400, "trunk_map50_95": 0.350, "bunch_map50_95": 0.450}
    # Общий рост есть, но класс trunk просел на 0.02 (> 0.01)
    cand = {"map50_95": 0.410, "trunk_map50_95": 0.330, "bunch_map50_95": 0.490}
    decision = evaluate_gate(cand, base)
    assert not decision.accepted
    assert "trunk" in decision.reason
    assert "просел" in decision.reason


def test_count_new_samples(tmp_path: Path) -> None:
    log_file = tmp_path / "ingestion_log.jsonl"
    records = [
        {"timestamp": "2026-11-01T10:00:00Z", "status": "added", "image": "img1.jpg"},
        {"timestamp": "2026-11-02T10:00:00Z", "status": "added", "image": "img2.jpg"},
        {
            "timestamp": "2026-11-02T11:00:00Z",
            "status": "rejected_test_leak",
            "image": "img3.jpg",
        },
        {"timestamp": "2026-11-03T10:00:00Z", "status": "added", "image": "img4.jpg"},
    ]
    with log_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Без ограничения даты: 3 добавленных
    assert count_new_samples(log_file, None) == 3
    # После 2026-11-02: 1 добавленный
    assert count_new_samples(log_file, "2026-11-02T10:30:00Z") == 1
    # Несуществующий лог: 0
    assert count_new_samples(tmp_path / "absent.jsonl", None) == 0


def test_runner_skips_when_no_new_data(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data" / "data.yaml"
    data_yaml.parent.mkdir(parents=True)
    data_yaml.write_text("names: {0: trunk, 1: bunch}\n", encoding="utf-8")

    out_dir = tmp_path / "results"
    out_dir.mkdir(parents=True)
    registry_file = out_dir / "registry.json"
    registry_file.write_text(
        json.dumps(
            [
                {
                    "version": "v001",
                    "date": "2026-11-03T12:00:00Z",
                    "accepted": True,
                    "weights": str(tmp_path / "v001.pt"),
                }
            ]
        ),
        encoding="utf-8",
    )

    runner = NightlyRetrainer(data_yaml=data_yaml, out_dir=out_dir)
    result = runner.run_nightly(force=False)

    assert result["status"] == "skipped"
    assert result["new_samples"] == 0


def test_runner_executes_pipeline_with_deploy(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data" / "data.yaml"
    data_yaml.parent.mkdir(parents=True)
    data_yaml.write_text("names: {0: trunk, 1: bunch}\n", encoding="utf-8")

    out_dir = tmp_path / "results"
    deploy_dir = tmp_path / "robot_models"

    mock_yolo_instance = MagicMock()
    mock_val_metrics = MagicMock()
    mock_val_metrics.box.map = 0.45
    mock_val_metrics.box.map50 = 0.65
    mock_val_metrics.box.ap_class_index = np.array([0, 1])
    mock_val_metrics.box.maps = np.array([0.42, 0.48])
    mock_yolo_instance.val.return_value = mock_val_metrics

    fake_onnx = tmp_path / "fake.onnx"
    fake_onnx.write_text("fake onnx", encoding="utf-8")
    mock_yolo_instance.export.return_value = str(fake_onnx)

    def mock_train(*args: Any, **kwargs: Any) -> None:
        # Имитация создания весов
        run_weights = out_dir / "runs" / "v001" / "weights"
        run_weights.mkdir(parents=True, exist_ok=True)
        (run_weights / "best.pt").write_text("fake best weights", encoding="utf-8")

    mock_yolo_instance.train.side_effect = mock_train

    with patch("importlib.import_module") as mock_import:
        mock_ultralytics = MagicMock()
        mock_ultralytics.YOLO.return_value = mock_yolo_instance
        mock_import.return_value = mock_ultralytics

        runner = NightlyRetrainer(
            data_yaml=data_yaml,
            out_dir=out_dir,
            deploy_dir=deploy_dir,
        )
        result = runner.run_nightly(epochs=2, force=True)

    assert result["status"] == "completed"
    assert result["accepted"] is True
    assert result["version"] == "v001"

    # Реестр обновлён
    registry = json.loads((out_dir / "registry.json").read_text(encoding="utf-8"))
    assert len(registry) == 1
    assert registry[0]["accepted"] is True
    assert "onnx" in registry[0]

    # Модель автоматически выкачена на робота через DeploymentManager
    assert (deploy_dir / "current.onnx").exists()
    assert (deploy_dir / "current.json").exists()
    info = json.loads((deploy_dir / "current.json").read_text(encoding="utf-8"))
    assert info["version"] == "v001"
