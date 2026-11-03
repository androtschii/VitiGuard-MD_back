import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from vitiguard_ml.robot.edge_sync import EdgeSyncClient


def test_edge_sync_get_active_version(tmp_path: Path) -> None:
    client = EdgeSyncClient(
        server_url="http://testserver",
        token="test_token",
        robot_id="robot-01",
        deploy_dir=tmp_path,
    )
    assert client._get_active_version() == "v000"

    (tmp_path / "current.json").write_text(json.dumps({"version": "v005"}))
    assert client._get_active_version() == "v005"


def test_edge_sync_send_telemetry(tmp_path: Path) -> None:
    client = EdgeSyncClient(
        server_url="http://testserver",
        token="test_token",
        robot_id="robot-01",
        deploy_dir=tmp_path,
    )

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(
        {
            "robot_id": "robot-01",
            "status": "ok",
            "update_available": False,
        }
    ).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = client.send_telemetry(battery_pct=90.0)

    assert res["status"] == "ok"
    assert res["robot_id"] == "robot-01"


def test_edge_sync_model_update(tmp_path: Path) -> None:
    client = EdgeSyncClient(
        server_url="http://testserver",
        token="test_token",
        robot_id="robot-01",
        deploy_dir=tmp_path,
    )

    fake_onnx_bytes = b"sample-onnx-content-for-edge-sync"
    server_sha = hashlib.sha256(fake_onnx_bytes).hexdigest()[:16]

    meta_mock = MagicMock()
    meta_mock.read.return_value = json.dumps({"version": server_sha}).encode("utf-8")
    meta_mock.__enter__.return_value = meta_mock

    file_mock = MagicMock()
    file_mock.read.side_effect = [fake_onnx_bytes, b""]
    file_mock.__enter__.return_value = file_mock

    with patch("urllib.request.urlopen", side_effect=[meta_mock, file_mock]):
        res = client.check_and_update_model()

    assert res["status"] == "updated"
    assert res["new_version"] == server_sha
    assert (tmp_path / "current.onnx").exists()
    assert (tmp_path / "current.json").exists()
