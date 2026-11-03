"""Единый бортовой конвейер инференса и принятия решений реального времени (ml/sprint-26).

Объединяет все этапы бортового зрения робота в один цикл обработки кадра (< 50 мс):
1. Проверка целостности потока сенсоров (RGB и Depth).
2. Оценка здоровья каждой обнаруженной грозди (bunch_health): отсев серой гнили (Botrytis).
3. Оценка спектральной зрелости и сахаристости °Brix (maturity): отсев недозрелых ягод.
4. Вычисление 3D-точки среза плодоножки и нормали (cutting).
5. Проверка коллизий со стволами лозы и планирование путевых точек манипулятора (kinematics).
6. Автоматическое принятие решения по каждой грозди:
   - HARVEST (сбор в бункер);
   - SKIP_UNRIPE (оставить дозревать);
   - SKIP_INFECTED (опасность порчи урожая);
   - SKIP_OBSTACLE (риск поломки манипулятора о шпалеру/ствол);
   - SKIP_UNREACHABLE (вне кинематической зоны досягаемости);
   - SKIP_UNSAFE_STEM (небезопасный срез ножки).
7. Мониторинг задержки (latency p50/p95/p99) и аварийная остановка (emergency stop).
"""

import enum
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from vitiguard_ml.robot.bunch_health import (
    BunchHealthAssessment,
    assess_bunch_health,
)
from vitiguard_ml.robot.cutting import (
    CameraIntrinsics,
    CutTarget,
    compute_cut_target,
)
from vitiguard_ml.robot.kinematics import (
    TrajectoryPlan,
    TrunkObstacle,
    generate_harvest_trajectory,
)
from vitiguard_ml.robot.maturity import (
    MaturityAssessment,
    evaluate_maturity,
)


class HarvestDecision(enum.StrEnum):
    """Решение бортовой системы комбайна по обнаруженной грозди."""

    HARVEST = "harvest"
    SKIP_UNRIPE = "skip_unripe"
    SKIP_INFECTED = "skip_infected"
    SKIP_OBSTACLE = "skip_obstacle"
    SKIP_UNREACHABLE = "skip_unreachable"
    SKIP_UNSAFE_STEM = "skip_unsafe_stem"


@dataclass(frozen=True)
class BunchDecisionVerdict:
    """Полное агрономическое и кинематическое заключение по отдельной грозди."""

    bunch_id: int
    bbox_xyxy: tuple[int, int, int, int]
    health: BunchHealthAssessment
    maturity: MaturityAssessment
    cut_target: CutTarget | None
    trajectory: TrajectoryPlan | None
    decision: HarvestDecision
    decision_reason: str


@dataclass
class EdgeInferenceMetrics:
    """Статистика производительности бортового цикла инференса."""

    frame_count: int = 0
    total_time_ms: float = 0.0
    latency_history_ms: list[float] = field(default_factory=list)

    def record(self, latency_ms: float) -> None:
        self.frame_count += 1
        self.total_time_ms += latency_ms
        self.latency_history_ms.append(latency_ms)
        if len(self.latency_history_ms) > 1000:
            self.latency_history_ms.pop(0)

    @property
    def mean_latency_ms(self) -> float:
        return (
            self.total_time_ms / max(1, self.frame_count)
            if self.frame_count > 0
            else 0.0
        )

    @property
    def fps(self) -> float:
        return 1000.0 / max(1.0, self.mean_latency_ms)

    def percentiles(self) -> dict[str, float]:
        if not self.latency_history_ms:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        arr = np.array(self.latency_history_ms)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }


@dataclass(frozen=True)
class FrameProcessingResult:
    """Итог обработки видеокадра с камеры комбайна."""

    frame_id: int | str
    timestamp: float
    latency_ms: float
    trunks_count: int
    bunches_count: int
    harvest_candidates: list[BunchDecisionVerdict]
    skipped_bunches: list[BunchDecisionVerdict]
    emergency_stop: bool
    emergency_stop_reason: str | None


class EdgeInferencePipeline:
    """Единый оптимизированный конвейер бортового зрения комбайна."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics | None = None,
        rot_matrix: np.ndarray | None = None,
        trans_vector: np.ndarray | None = None,
        variety: str = "cabernet_sauvignon",
        min_sugar_brix: float = 18.0,
        max_rot_fraction: float = 0.15,
        max_allowed_latency_ms: float = 60.0,
    ) -> None:
        self.intrinsics = intrinsics or CameraIntrinsics(
            fx=615.0, fy=615.0, cx=320.0, cy=240.0
        )
        self.rot_matrix = rot_matrix
        self.trans_vector = trans_vector
        self.variety = variety
        self.min_sugar_brix = min_sugar_brix
        self.max_rot_fraction = max_rot_fraction
        self.max_allowed_latency_ms = max_allowed_latency_ms

        self.metrics = EdgeInferenceMetrics()

    def process_frame(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        bunch_masks: Sequence[np.ndarray],
        trunk_obstacles: Sequence[TrunkObstacle] | None = None,
        frame_id: int | str = 0,
    ) -> FrameProcessingResult:
        """Обрабатывает один синхронизированный кадр (RGB + карта глубины)."""
        t0 = time.perf_counter()
        now = time.time()

        # 1. Проверка целостности сенсоров (Safety & Sanity Checks)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return FrameProcessingResult(
                frame_id=frame_id,
                timestamp=now,
                latency_ms=0.0,
                trunks_count=0,
                bunches_count=0,
                harvest_candidates=[],
                skipped_bunches=[],
                emergency_stop=True,
                emergency_stop_reason="Некорректная размерность RGB-потока камеры",
            )

        if depth_m.ndim != 2 or depth_m.shape[:2] != rgb.shape[:2]:
            return FrameProcessingResult(
                frame_id=frame_id,
                timestamp=now,
                latency_ms=0.0,
                trunks_count=0,
                bunches_count=0,
                harvest_candidates=[],
                skipped_bunches=[],
                emergency_stop=True,
                emergency_stop_reason="Несовпадение габаритов RGB и Depth-карты камеры",
            )

        # Проверка отвала дальномера (менее 10% достоверных точек)
        valid_depth_ratio = float(np.mean(depth_m > 0.05))
        if valid_depth_ratio < 0.10:
            return FrameProcessingResult(
                frame_id=frame_id,
                timestamp=now,
                latency_ms=0.0,
                trunks_count=0,
                bunches_count=0,
                harvest_candidates=[],
                skipped_bunches=[],
                emergency_stop=True,
                emergency_stop_reason="Сбой лидара/камеры глубины: менее 10% достоверных точек",
            )

        obstacles = list(trunk_obstacles or [])
        harvest_candidates: list[BunchDecisionVerdict] = []
        skipped_bunches: list[BunchDecisionVerdict] = []

        # 2. Поочерёдный анализ обнаруженных масок гроздей
        for i, mask in enumerate(bunch_masks):
            y_idx, x_idx = np.where(mask > 0)
            if len(y_idx) == 0:
                continue

            ymin, xmin = int(np.min(y_idx)), int(np.min(x_idx))
            ymax, xmax = int(np.max(y_idx)), int(np.max(x_idx))
            bbox = (xmin, ymin, xmax, ymax)

            # Вырезка фрагмента грозди
            crop_rgb = rgb[ymin : ymax + 1, xmin : xmax + 1]
            if crop_rgb.size == 0 or crop_rgb.shape[0] < 5 or crop_rgb.shape[1] < 5:
                continue
            crop_pil = Image.fromarray(crop_rgb)

            # 2a. Проверка здоровья (серая гниль Botrytis)
            health = assess_bunch_health(crop_pil, rot_threshold=self.max_rot_fraction)
            if not health.should_harvest or health.rot_ratio > self.max_rot_fraction:
                skipped_bunches.append(
                    BunchDecisionVerdict(
                        bunch_id=i,
                        bbox_xyxy=bbox,
                        health=health,
                        maturity=MaturityAssessment(
                            variety=self.variety,
                            is_red=True,
                            color_index=0.0,
                            sugar_brix=0.0,
                            stage="unripe",
                            harvest_suitability="too_early",
                            mean_lab=(0.0, 0.0, 0.0),
                            metrics={},
                        ),
                        cut_target=None,
                        trajectory=None,
                        decision=HarvestDecision.SKIP_INFECTED,
                        decision_reason=f"Поражение серой гнилью ({health.rot_ratio * 100:.1f} % > {self.max_rot_fraction * 100:.1f} %)",
                    )
                )
                continue

            # 2b. Проверка зрелости и сахаристости
            maturity = evaluate_maturity(crop_pil, variety=self.variety)
            if (
                maturity.stage not in ("harvest_ready", "overripe")
                or maturity.sugar_brix < self.min_sugar_brix
            ):
                skipped_bunches.append(
                    BunchDecisionVerdict(
                        bunch_id=i,
                        bbox_xyxy=bbox,
                        health=health,
                        maturity=maturity,
                        cut_target=None,
                        trajectory=None,
                        decision=HarvestDecision.SKIP_UNRIPE,
                        decision_reason=f"Недостаточная сахаристость ({maturity.sugar_brix:.1f} °Brix < {self.min_sugar_brix:.1f} °Brix)",
                    )
                )
                continue

            # 2c. Определение 3D-точки среза плодоножки
            cut_target = compute_cut_target(mask, depth_m, self.intrinsics)
            if cut_target is None or not cut_target.safe_to_cut:
                skipped_bunches.append(
                    BunchDecisionVerdict(
                        bunch_id=i,
                        bbox_xyxy=bbox,
                        health=health,
                        maturity=maturity,
                        cut_target=cut_target,
                        trajectory=None,
                        decision=HarvestDecision.SKIP_UNSAFE_STEM,
                        decision_reason="Небезопасное положение плодоножки (слишком близко к ягодам или вне поля зрения)",
                    )
                )
                continue

            # 2d. Планирование кинематической траектории манипулятора
            trajectory = generate_harvest_trajectory(
                cut_point_cam=cut_target.point_3d,
                normal_cam=cut_target.approach_vector,
                obstacles=obstacles,
                rot_matrix=self.rot_matrix,
                trans_vector=self.trans_vector,
            )

            if not trajectory.is_collision_free:
                skipped_bunches.append(
                    BunchDecisionVerdict(
                        bunch_id=i,
                        bbox_xyxy=bbox,
                        health=health,
                        maturity=maturity,
                        cut_target=cut_target,
                        trajectory=trajectory,
                        decision=HarvestDecision.SKIP_OBSTACLE,
                        decision_reason="Обнаружена опасная коллизия со стволом или опорой шпалеры",
                    )
                )
            elif not trajectory.is_reachable:
                skipped_bunches.append(
                    BunchDecisionVerdict(
                        bunch_id=i,
                        bbox_xyxy=bbox,
                        health=health,
                        maturity=maturity,
                        cut_target=cut_target,
                        trajectory=trajectory,
                        decision=HarvestDecision.SKIP_UNREACHABLE,
                        decision_reason="Точка реза находится вне рабочей кинематической зоны манипулятора",
                    )
                )
            elif trajectory.is_valid:
                harvest_candidates.append(
                    BunchDecisionVerdict(
                        bunch_id=i,
                        bbox_xyxy=bbox,
                        health=health,
                        maturity=maturity,
                        cut_target=cut_target,
                        trajectory=trajectory,
                        decision=HarvestDecision.HARVEST,
                        decision_reason=f"Кондиция подтверждена ({maturity.sugar_brix:.1f} °Brix), траектория свободна",
                    )
                )
            else:
                skipped_bunches.append(
                    BunchDecisionVerdict(
                        bunch_id=i,
                        bbox_xyxy=bbox,
                        health=health,
                        maturity=maturity,
                        cut_target=cut_target,
                        trajectory=trajectory,
                        decision=HarvestDecision.SKIP_UNSAFE_STEM,
                        decision_reason="Отказ планирования траектории среза",
                    )
                )

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        self.metrics.record(latency_ms)

        emergency = latency_ms > (self.max_allowed_latency_ms * 3.0)
        emergency_reason = (
            f"Критическая задержка цикла инференса ({latency_ms:.1f} мс > 3x порога)"
            if emergency
            else None
        )

        return FrameProcessingResult(
            frame_id=frame_id,
            timestamp=now,
            latency_ms=round(latency_ms, 2),
            trunks_count=len(obstacles),
            bunches_count=len(bunch_masks),
            harvest_candidates=harvest_candidates,
            skipped_bunches=skipped_bunches,
            emergency_stop=emergency,
            emergency_stop_reason=emergency_reason,
        )

    def benchmark(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        bunch_masks: Sequence[np.ndarray],
        num_runs: int = 15,
    ) -> dict[str, float]:
        """Запускает серию прогонов для замера стабильности задержки и FPS."""
        latencies: list[float] = []
        for i in range(num_runs):
            res = self.process_frame(rgb, depth_m, bunch_masks, frame_id=f"bench_{i}")
            latencies.append(res.latency_ms)

        arr = np.array(latencies)
        return {
            "mean_ms": round(float(np.mean(arr)), 2),
            "std_ms": round(float(np.std(arr)), 2),
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "fps": round(1000.0 / max(0.1, float(np.mean(arr))), 1),
        }
