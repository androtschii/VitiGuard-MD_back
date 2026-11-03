"""Управление версиями и выкатка ONNX на компьютер робота (ml/pr-033).

Обеспечивает выкатку принятых версий детектора лозы на бортовой компьютер
робота с возможностью мгновенного отката на предыдущую рабочую версию:
- читает реестр registry.json;
- находит последнюю принятую версию или явно запрошенную;
- сохраняет резервную копию previous.onnx перед обновлением;
- выкатывает current.onnx и метаданные current.json;
- ведёт журнал всех выкаток и откатов в deploy_log.jsonl.
"""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vitiguard_ml.robot.vine import best_entry, load_registry


class DeploymentManager:
    """Управляет выкаткой моделей на компьютер робота и откатами."""

    def __init__(self, registry_path: Path, deploy_dir: Path) -> None:
        self.registry_path = registry_path
        self.deploy_dir = deploy_dir
        self.current_onnx = deploy_dir / "current.onnx"
        self.current_json = deploy_dir / "current.json"
        self.previous_onnx = deploy_dir / "previous.onnx"
        self.previous_json = deploy_dir / "previous.json"
        self.deploy_log = deploy_dir / "deploy_log.jsonl"

        self.deploy_dir.mkdir(parents=True, exist_ok=True)

    def _log_action(self, action: str, details: dict[str, Any]) -> None:
        entry = {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "action": action,
            **details,
        }
        with self.deploy_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def deploy(self, version: str | None = None) -> dict[str, Any]:
        """Выкатывает версию модели на робота. Если version не указан — берётся лучшая принятая."""
        registry = load_registry(self.registry_path)
        if not registry:
            raise FileNotFoundError(
                f"Реестр версий не найден или пуст: {self.registry_path}"
            )

        target: dict[str, Any] | None = None
        if version:
            for item in registry:
                if item.get("version") == version:
                    target = item
                    break
            if target is None:
                raise ValueError(f"Версия {version} не найдена в реестре")
            if not target.get("accepted", False):
                raise ValueError(f"Версия {version} не была принята правилами отбора")
        else:
            target = best_entry(registry)
            if target is None:
                raise ValueError("В реестре нет ни одной принятой версии модели")

        source_onnx_str = target.get("onnx")
        if not source_onnx_str:
            raise FileNotFoundError(
                f"У версии {target['version']} отсутствует экспортированный ONNX-файл"
            )

        source_onnx = Path(source_onnx_str)
        if not source_onnx.exists():
            raise FileNotFoundError(f"Файл ONNX не найден на диске: {source_onnx}")

        # Резервное копирование текущей версии перед обновлением
        prev_version: str | None = None
        if self.current_onnx.exists() and self.current_json.exists():
            try:
                prev_info = json.loads(self.current_json.read_text(encoding="utf-8"))
                prev_version = prev_info.get("version")
            except Exception:
                prev_version = "unknown"
            shutil.copyfile(self.current_onnx, self.previous_onnx)
            shutil.copyfile(self.current_json, self.previous_json)

        # Выкатка нового файла
        shutil.copyfile(source_onnx, self.current_onnx)
        deploy_meta = {
            "version": target["version"],
            "deployed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "source_onnx": str(source_onnx),
            "image_size": target.get("image_size", 640),
            "metrics": target.get("test", {}),
            "epochs": target.get("epochs", 0),
        }
        self.current_json.write_text(
            json.dumps(deploy_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        self._log_action(
            "deploy",
            {
                "version": target["version"],
                "previous_version": prev_version,
                "metrics": target.get("test", {}),
            },
        )
        return deploy_meta

    def rollback(self) -> dict[str, Any]:
        """Откатывает модель на предыдущую рабочую версию."""
        if not self.previous_onnx.exists() or not self.previous_json.exists():
            raise FileNotFoundError("Предыдущая версия для отката отсутствует")

        prev_info: dict[str, Any] = json.loads(
            self.previous_json.read_text(encoding="utf-8")
        )
        current_version = "unknown"
        if self.current_json.exists():
            try:
                curr_info = json.loads(self.current_json.read_text(encoding="utf-8"))
                current_version = curr_info.get("version", "unknown")
            except Exception:
                pass

        # Восстанавливаем предыдущую версию
        shutil.copyfile(self.previous_onnx, self.current_onnx)
        shutil.copyfile(self.previous_json, self.current_json)

        self._log_action(
            "rollback",
            {
                "rolled_back_from": current_version,
                "restored_version": prev_info.get("version"),
            },
        )
        return prev_info

    def status(self) -> dict[str, Any]:
        """Возвращает информацию о текущей выкатке."""
        if not self.current_json.exists() or not self.current_onnx.exists():
            return {"deployed": False, "can_rollback": False}

        current_meta: dict[str, Any] = json.loads(
            self.current_json.read_text(encoding="utf-8")
        )
        return {
            "deployed": True,
            "version": current_meta.get("version"),
            "deployed_at": current_meta.get("deployed_at"),
            "metrics": current_meta.get("metrics"),
            "can_rollback": self.previous_onnx.exists() and self.previous_json.exists(),
        }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("results/vine/registry.json"),
        help="Путь к реестру версий модели",
    )
    parser.add_argument(
        "--deploy-dir",
        type=Path,
        default=Path("results/robot/deployed"),
        help="Каталог выкатки на компьютере робота",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    deploy_parser = subparsers.add_parser("deploy", help="Выкатить версию на робота")
    deploy_parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Имя версии (v001, v002...), по умолчанию лучшая",
    )

    subparsers.add_parser("rollback", help="Откатиться на предыдущую версию")
    subparsers.add_parser("status", help="Показать статус текущей модели на роботе")

    args = parser.parse_args(argv)
    mgr = DeploymentManager(registry_path=args.registry, deploy_dir=args.deploy_dir)

    if args.command == "deploy":
        res = mgr.deploy(version=args.version)
        print(f"Выкачена модель {res['version']} ({res['deployed_at']})")
    elif args.command == "rollback":
        res = mgr.rollback()
        print(f"Выполнен откат на модель {res['version']}")
    elif args.command == "status":
        print(json.dumps(mgr.status(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
