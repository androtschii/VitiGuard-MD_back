import json
from pathlib import Path

import pytest

from vitiguard_ml.robot.deploy import DeploymentManager


def test_deployment_manager_deploy_and_rollback(tmp_path: Path) -> None:
    # Создаём фиктивный ONNX файл v001 и v002
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "v001.onnx").write_bytes(b"dummy_onnx_v1")
    (models_dir / "v002.onnx").write_bytes(b"dummy_onnx_v2")

    registry = [
        {
            "version": "v001",
            "date": "2026-11-01T10:00:00Z",
            "accepted": True,
            "weights": str(models_dir / "v001.pt"),
            "onnx": str(models_dir / "v001.onnx"),
            "test": {"map50_95": 0.38},
        },
        {
            "version": "v002",
            "date": "2026-11-02T10:00:00Z",
            "accepted": True,
            "weights": str(models_dir / "v002.pt"),
            "onnx": str(models_dir / "v002.onnx"),
            "test": {"map50_95": 0.42},
        },
    ]

    registry_file = tmp_path / "registry.json"
    registry_file.write_text(json.dumps(registry), encoding="utf-8")

    deploy_dir = tmp_path / "deployed"
    mgr = DeploymentManager(registry_path=registry_file, deploy_dir=deploy_dir)

    # 1. Первая выкатка: должна выкатить v002 (как лучшую принятую)
    res1 = mgr.deploy()
    assert res1["version"] == "v002"
    assert mgr.current_onnx.read_bytes() == b"dummy_onnx_v2"
    assert mgr.status()["deployed"]
    assert not mgr.status()["can_rollback"]

    # 2. Вторая выкатка: выкатываем v001 явно
    res2 = mgr.deploy(version="v001")
    assert res2["version"] == "v001"
    assert mgr.current_onnx.read_bytes() == b"dummy_onnx_v1"
    assert mgr.previous_onnx.read_bytes() == b"dummy_onnx_v2"
    assert mgr.status()["can_rollback"]

    # 3. Откат: должен вернуть v002
    res_roll = mgr.rollback()
    assert res_roll["version"] == "v002"
    assert mgr.current_onnx.read_bytes() == b"dummy_onnx_v2"

    # 4. Проверка журнала
    log_lines = mgr.deploy_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(log_lines) == 3


def test_deployment_manager_rejects_unaccepted_version(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "v001.onnx").write_bytes(b"dummy")

    registry = [
        {
            "version": "v001",
            "accepted": False,
            "onnx": str(models_dir / "v001.onnx"),
        }
    ]
    registry_file = tmp_path / "registry.json"
    registry_file.write_text(json.dumps(registry), encoding="utf-8")

    mgr = DeploymentManager(registry_path=registry_file, deploy_dir=tmp_path / "deploy")

    with pytest.raises(ValueError, match="не была принята"):
        mgr.deploy(version="v001")
