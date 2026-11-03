"""Сбор и фильтрация кадров за день с камеры робота (ml/pr-029).

Каждый день робот сканирует ряды виноградника. Скрипт принимает поток кадров
(RGB + карта глубины + метаданные положения), отбрасывает смазанные кадры
(по дисперсии Лапласиана) и кадры-повторы (по перцептивному dHash),
формируя структурированный набор для предразметки и дообучения.
"""

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from vitiguard_ml.data.images import difference_hash, hamming


@dataclass(frozen=True)
class RobotPose:
    """Координаты и ориентация робота на винограднике."""

    row_id: str
    x: float
    y: float
    heading_deg: float


@dataclass(frozen=True)
class CollectedFrameMeta:
    """Метаданные сохранённого кадра."""

    frame_id: str
    timestamp: str
    pose: RobotPose
    blur_score: float
    dhash: str
    rgb_file: str
    depth_file: str | None


def laplacian_variance(image_rgb: np.ndarray) -> float:
    """Вычисляет оценку резкости через дисперсию дискретного оператора Лапласа."""
    if image_rgb.ndim == 3:
        gray = (
            0.299 * image_rgb[..., 0]
            + 0.587 * image_rgb[..., 1]
            + 0.114 * image_rgb[..., 2]
        ).astype(np.float32)
    else:
        gray = image_rgb.astype(np.float32)

    # Дискретное ядро Лапласа 3x3: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    padded = np.pad(gray, 1, mode="edge")
    lap = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * gray
    )
    return float(np.var(lap))


def is_blurry(image_rgb: np.ndarray, min_variance: float = 80.0) -> tuple[bool, float]:
    """Проверяет, смазан ли снимок от вибрации или движения робота."""
    score = laplacian_variance(image_rgb)
    return score < min_variance, score


class FrameCollector:
    """Менеджер отбора и сохранения дневных сессий робота."""

    def __init__(
        self,
        output_dir: Path,
        min_variance: float = 80.0,
        max_hamming: int = 5,
    ) -> None:
        self.output_dir = output_dir
        self.rgb_dir = output_dir / "rgb"
        self.depth_dir = output_dir / "depth"
        self.manifest_file = output_dir / "manifest.jsonl"
        self.min_variance = min_variance
        self.max_hamming = max_hamming

        self.seen_hashes: set[str] = set()
        self.stats = {
            "total_received": 0,
            "saved": 0,
            "rejected_blurry": 0,
            "rejected_duplicate": 0,
        }

        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)

    def process_frame(
        self,
        rgb: np.ndarray,
        pose: RobotPose,
        depth: np.ndarray | None = None,
        frame_id: str | None = None,
        timestamp: str | None = None,
    ) -> CollectedFrameMeta | None:
        """Обрабатывает входящий кадр: проверяет резкость и новизну, сохраняет на диск."""
        self.stats["total_received"] += 1

        # 1. Проверка на размытие
        blurry, score = is_blurry(rgb, min_variance=self.min_variance)
        if blurry:
            self.stats["rejected_blurry"] += 1
            return None

        # 2. Проверка на дубликат по dHash
        pil_img = Image.fromarray(rgb)
        h = difference_hash(pil_img)
        if any(
            hamming(h, existing) <= self.max_hamming for existing in self.seen_hashes
        ):
            self.stats["rejected_duplicate"] += 1
            return None

        # 3. Сохранение
        if frame_id is None:
            now_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
            frame_id = f"frame_{now_str}"

        if timestamp is None:
            timestamp = datetime.now(UTC).isoformat(timespec="seconds")

        rgb_path = self.rgb_dir / f"{frame_id}.jpg"
        pil_img.save(rgb_path, quality=90)

        depth_filename: str | None = None
        if depth is not None:
            depth_filename = f"{frame_id}.npy"
            np.save(self.depth_dir / depth_filename, depth.astype(np.float32))

        meta = CollectedFrameMeta(
            frame_id=frame_id,
            timestamp=timestamp,
            pose=pose,
            blur_score=round(score, 2),
            dhash=h,
            rgb_file=rgb_path.name,
            depth_file=depth_filename,
        )

        with self.manifest_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(meta), ensure_ascii=False) + "\n")

        self.seen_hashes.add(h)
        self.stats["saved"] += 1
        return meta
