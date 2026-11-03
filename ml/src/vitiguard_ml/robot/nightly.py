"""Ночное дообучение модели лозы от вчерашних весов и гейт качества (ml/pr-032).

Каждую ночь после пополнения обучающего набора (ml/pr-031):
1. Проверяется наличие новых размеченных кадров со времени последней сборки.
2. Новая модель дообучается от лучших принятых весов (warm start).
3. Производится валидация на неизменной эталонной выборке (test split).
4. Гейт качества (acceptance gate):
   - Новая версия принимается ТОЛЬКО если средняя точность (mAP50-95) выросла;
   - И ни один класс (ствол, гроздь) не просел более чем на 0.01 (1 пункт).
5. При успехе:
   - Модель экспортируется в ONNX;
   - Заносится в реестр (registry.json) со статусом accepted=True;
   - При указании каталога робота (--deploy-dir) автоматически обновляется
     модель через DeploymentManager (ml/pr-033) с сохранением предыдущей.
6. При неудаче:
   - В реестр заносится причина отклонения (rejection_reason);
   - Выкатка не производится, робот остаётся на проверенной версии.

Запуск из каталога ml:
  uv run --group train python -m vitiguard_ml.robot.nightly \\
      --data data/processed/vine/data.yaml \\
      --out results/vine \\
      --deploy-dir /opt/vitiguard/robot/models
"""

import argparse
import importlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vitiguard_ml.data.vine import CLASSES
from vitiguard_ml.robot.deploy import DeploymentManager
from vitiguard_ml.robot.vine import (
    MAX_CLASS_DROP,
    START_MODEL,
    best_entry,
    class_scores,
    load_registry,
)


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reason: str
    overall_diff: float
    class_diffs: dict[str, float]


def evaluate_gate(
    candidate: dict[str, float],
    baseline: dict[str, float] | None,
    max_class_drop: float = MAX_CLASS_DROP,
) -> GateDecision:
    """Проверяет строгое правило приёмки модели лозы по каждому классу."""
    if baseline is None:
        return GateDecision(
            accepted=True,
            reason="Базовая модель: первая версия в реестре принимается автоматически",
            overall_diff=0.0,
            class_diffs=dict.fromkeys(CLASSES, 0.0),
        )

    cand_map = candidate.get("map50_95", 0.0)
    base_map = baseline.get("map50_95", 0.0)
    overall_diff = round(cand_map - base_map, 4)

    class_diffs: dict[str, float] = {}
    for cls in CLASSES:
        cand_c = candidate.get(f"{cls}_map50_95", 0.0)
        base_c = baseline.get(f"{cls}_map50_95", 0.0)
        class_diffs[cls] = round(cand_c - base_c, 4)

    if overall_diff <= 0.0:
        return GateDecision(
            accepted=False,
            reason=(
                f"Отклонено: средняя mAP50-95 ({cand_map:.4f}) не превысила "
                f"лучший показатель ({base_map:.4f}, разница {overall_diff:+.4f})"
            ),
            overall_diff=overall_diff,
            class_diffs=class_diffs,
        )

    for cls, diff in class_diffs.items():
        if diff < -max_class_drop:
            return GateDecision(
                accepted=False,
                reason=(
                    f"Отклонено: класс '{cls}' просел на {abs(diff):.4f} "
                    f"(допустимо не более {max_class_drop:.4f})"
                ),
                overall_diff=overall_diff,
                class_diffs=class_diffs,
            )

    return GateDecision(
        accepted=True,
        reason=(
            f"Принято: общая точность выросла на {overall_diff:+.4f} "
            f"({base_map:.4f} -> {cand_map:.4f}), стабильность классов подтверждена"
        ),
        overall_diff=overall_diff,
        class_diffs=class_diffs,
    )


def count_new_samples(ingestion_log: Path, since_iso: str | None) -> int:
    """Считает число новых кадров, добавленных в обучающий набор после since_iso."""
    if not ingestion_log.exists():
        return 0

    count = 0
    with ingestion_log.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if item.get("status") != "added":
                    continue
                if since_iso is None or item.get("timestamp", "") > since_iso:
                    count += 1
            except json.JSONDecodeError:
                continue
    return count


class NightlyRetrainer:
    """Автоматизированный конвейер ночного дообучения и выкатки модели."""

    def __init__(
        self,
        data_yaml: Path,
        out_dir: Path,
        deploy_dir: Path | None = None,
        max_class_drop: float = MAX_CLASS_DROP,
    ) -> None:
        self.data_yaml = data_yaml
        self.out_dir = out_dir
        self.deploy_dir = deploy_dir
        self.max_class_drop = max_class_drop
        self.registry_path = out_dir / "registry.json"
        self.models_dir = out_dir / "models"
        self.runs_dir = out_dir / "runs"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def run_nightly(
        self,
        epochs: int = 10,
        image_size: int = 640,
        batch: int = 8,
        workers: int = 2,
        force: bool = False,
        cpu: bool = False,
    ) -> dict[str, Any]:
        """Выполняет ночной цикл дообучения."""
        registry = load_registry(self.registry_path)
        best = best_entry(registry)

        # 1. Проверка наличия новых данных
        dataset_root = self.data_yaml.parent
        log_path = dataset_root / "ingestion_log.jsonl"
        last_date = best.get("date") if best else None
        new_samples = count_new_samples(log_path, last_date)

        if not force and best is not None and new_samples == 0:
            return {
                "status": "skipped",
                "reason": "Новых размеченных данных с последней сборки не обнаружено",
                "active_version": best.get("version"),
                "new_samples": 0,
            }

        start_weights = best["weights"] if best else START_MODEL
        version_num = len(registry) + 1
        version_name = f"v{version_num:03d}"

        yolo = importlib.import_module("ultralytics").YOLO
        model = yolo(start_weights)

        # 2. Обучение
        model.train(
            data=str(self.data_yaml),
            epochs=epochs,
            imgsz=image_size,
            batch=batch,
            seed=42,
            deterministic=True,
            project=str(self.runs_dir),
            name=version_name,
            exist_ok=True,
            workers=workers,
            device="cpu" if cpu else None,
        )

        trained_weights = self.runs_dir / version_name / "weights" / "best.pt"
        if not trained_weights.exists():
            # Запасной путь на случай, если лучший чекпоинт назван last.pt
            trained_weights = self.runs_dir / version_name / "weights" / "last.pt"

        # 3. Валидация на неизменной тестовой выборке
        val_metrics = yolo(str(trained_weights)).val(
            data=str(self.data_yaml),
            split="test",
            imgsz=image_size,
            plots=False,
        )
        scores = class_scores(val_metrics)

        # 4. Проверка критериев приёмки
        best_test = best.get("test") if best else None
        decision = evaluate_gate(scores, best_test, self.max_class_drop)

        # 5. Сохранение чекпоинта
        dest_weights = self.models_dir / f"{version_name}.pt"
        shutil.copyfile(trained_weights, dest_weights)

        entry: dict[str, Any] = {
            "version": version_name,
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "started_from": str(start_weights),
            "epochs": epochs,
            "image_size": image_size,
            "data": str(self.data_yaml),
            "new_samples_trained": new_samples,
            "test": scores,
            "accepted": decision.accepted,
            "decision_reason": decision.reason,
            "weights": str(dest_weights),
        }

        # 6. Экспорт и запись в реестр
        if decision.accepted:
            exported_onnx = yolo(str(dest_weights)).export(
                format="onnx", imgsz=image_size
            )
            entry["onnx"] = str(exported_onnx)

        registry.append(entry)
        self.registry_path.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 7. Автоматическая выкатка на робота при принятии
        if decision.accepted and self.deploy_dir is not None:
            manager = DeploymentManager(
                registry_path=self.registry_path,
                deploy_dir=self.deploy_dir,
            )
            deploy_info = manager.deploy(version=version_name)
            entry["deploy_info"] = deploy_info

        return {
            "status": "completed",
            "version": version_name,
            "accepted": decision.accepted,
            "reason": decision.reason,
            "scores": scores,
            "new_samples": new_samples,
        }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/vine/data.yaml"),
        help="Путь к data.yaml датасета",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/vine"),
        help="Каталог вывода результатов и реестра моделей",
    )
    parser.add_argument(
        "--deploy-dir",
        type=Path,
        default=None,
        help="Каталог выкатки моделей робота (DeploymentManager)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args(argv)
    runner = NightlyRetrainer(
        data_yaml=args.data,
        out_dir=args.out,
        deploy_dir=args.deploy_dir,
    )
    result = runner.run_nightly(
        epochs=args.epochs,
        image_size=args.image_size,
        batch=args.batch,
        workers=args.workers,
        force=args.force,
        cpu=args.cpu,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
