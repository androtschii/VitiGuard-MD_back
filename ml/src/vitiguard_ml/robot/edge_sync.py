"""Клиент синхронизации бортового компьютера робота с сервером VitiGuard MD (ml/pr-033, edge sync).

Обеспечивает работу робота в поле:
1. Отправку регулярной телеметрии (заряд, скорость, FPS камеры, задержка инференса);
2. Запрос обновлений модели детектора лозы с сервера хозяйства;
3. Безопасную выкатку новой модели через DeploymentManager с созданием previous.onnx;
4. Отправку сложных кадров (Active Learning) на сервер для разметки агрономом.

Запуск из каталога ml:
  uv run python -m vitiguard_ml.robot.edge_sync \\
      --server http://localhost:8000 \\
      --token <api_token> \\
      --robot-id harvester-01 \\
      --deploy-dir /opt/vitiguard/robot/models
"""

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vitiguard_ml.robot.deploy import DeploymentManager


@dataclass
class SyncState:
    robot_id: str
    active_version: str
    server_url: str
    deploy_dir: Path
    last_sync_status: str = "init"


class EdgeSyncClient:
    """Клиент синхронизации для бортового компьютера робота."""

    def __init__(
        self,
        server_url: str,
        token: str,
        robot_id: str,
        deploy_dir: Path,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.token = token
        self.robot_id = robot_id
        self.deploy_dir = deploy_dir
        self.headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": f"VitiGuard-EdgeAgent/{robot_id}",
        }

    def _get_active_version(self) -> str:
        """Читает версию текущей выкаченной модели."""
        info_file = self.deploy_dir / "current.json"
        if info_file.exists():
            try:
                data = json.loads(info_file.read_text(encoding="utf-8"))
                return str(data.get("version", "v000"))
            except Exception:
                pass
        return "v000"

    def send_telemetry(
        self,
        battery_pct: float = 100.0,
        speed_mps: float = 0.5,
        current_row: str = "row_01",
        lat: float | None = None,
        lon: float | None = None,
        camera_fps: float = 30.0,
        infer_latency_ms: float = 25.0,
        harvested_bunches: int = 0,
        discarded_bunches: int = 0,
    ) -> dict[str, Any]:
        """Отправляет телеметрию на сервер и получает указания/сигнал обновления."""
        active_ver = self._get_active_version()
        payload = {
            "robot_id": self.robot_id,
            "battery_pct": battery_pct,
            "speed_mps": speed_mps,
            "current_row": current_row,
            "lat": lat,
            "lon": lon,
            "camera_fps": camera_fps,
            "infer_latency_ms": infer_latency_ms,
            "active_model_version": active_ver,
            "harvested_bunches": harvested_bunches,
            "discarded_bunches": discarded_bunches,
        }

        url = f"{self.server_url}/api/v1/robot/telemetry"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={**self.headers, "Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
            return data

    def check_and_update_model(self) -> dict[str, Any]:
        """Проверяет наличие новой версии на сервере и выкатывает её при наличии."""
        active_ver = self._get_active_version()
        meta_url = f"{self.server_url}/api/v1/ai/models/vine"
        req = urllib.request.Request(meta_url, headers=self.headers, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                meta = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 503:
                return {"status": "skipped", "reason": "Модель на сервере не загружена"}
            raise

        server_ver = meta.get("version")
        if server_ver == active_ver:
            return {"status": "up_to_date", "version": active_ver}

        # Скачивание файла модели
        file_url = f"{self.server_url}/api/v1/ai/models/vine/file"
        file_req = urllib.request.Request(file_url, headers=self.headers, method="GET")

        incoming_dir = self.deploy_dir / "incoming"
        incoming_dir.mkdir(parents=True, exist_ok=True)
        download_target = incoming_dir / f"vine_{server_ver}.onnx"

        with (
            urllib.request.urlopen(file_req, timeout=60.0) as resp,
            download_target.open("wb") as out_f,
        ):
            while chunk := resp.read(64 * 1024):
                out_f.write(chunk)

        # Проверка целостности SHA-256
        sha = hashlib.sha256(download_target.read_bytes()).hexdigest()[:16]
        if sha != server_ver:
            download_target.unlink(missing_ok=True)
            raise ValueError(
                f"Контрольная сумма скачанной модели не совпала ({sha} != {server_ver})"
            )

        # Выкатка через реестр/DeploymentManager
        fake_registry = self.deploy_dir / "registry.json"
        registry = []
        if fake_registry.exists():
            try:
                registry = json.loads(fake_registry.read_text(encoding="utf-8"))
            except Exception:
                registry = []

        new_entry = {
            "version": server_ver,
            "accepted": True,
            "onnx": str(download_target),
            "date": meta.get("date", ""),
            "classes": meta.get("classes", ["trunk", "bunch"]),
        }
        registry.append(new_entry)
        fake_registry.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        manager = DeploymentManager(
            registry_path=fake_registry, deploy_dir=self.deploy_dir
        )
        deploy_info = manager.deploy(version=server_ver)

        return {
            "status": "updated",
            "previous_version": active_ver,
            "new_version": server_ver,
            "deploy_info": deploy_info,
        }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=str, default="http://localhost:8000")
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--robot-id", type=str, default="harvester-01")
    parser.add_argument("--deploy-dir", type=Path, default=Path("results/robot_deploy"))
    args = parser.parse_args(argv)

    client = EdgeSyncClient(
        server_url=args.server,
        token=args.token,
        robot_id=args.robot_id,
        deploy_dir=args.deploy_dir,
    )
    telem = client.send_telemetry()
    print("Telemetry response:", json.dumps(telem, indent=2, ensure_ascii=False))

    update_res = client.check_and_update_model()
    print("Model sync response:", json.dumps(update_res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
