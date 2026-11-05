"""Оценка урожайности виноградника и пространственная карта сахаристости (ml/pr-027, yield mapping).

При проезде робота-сборщика по рядам камеры фиксируют грозди с высокой частотой кадров.
Для построения точной карты урожая:
1. Пространственная дедупликация (spatial clustering):
   - Объединяет повторные детекции одной и той же грозди на соседних кадрах
     по 3D-координатам относительно траектории движения робота (радиус кластеризации 15 см);
   - Устраняет многократный переучёт (overcounting) гроздей;
2. Оценка массы грозди и сахаристости:
   - Масса грозди оценивается по площади проекции ягод в пикселях с поправкой на расстояние;
   - Накопление средней сахаристости (°Brix) и доли больных гроздей;
3. Формирование пространственной ГИС-сетки (GeoJSON FeatureCollection):
   - Сетка ячеек вдоль шпалеры с расчётом плотности урожая (т/га),
     среднего сахара (°Brix) и очагов поражения гнилями для дифференцированного сбора.

Запуск из каталога ml:
  uv run python -m vitiguard_ml.robot.yield_estimation --input telemetry.jsonl --out yield_map.geojson
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Плотность посадки по умолчанию: междурядье 2.5 м, расстояние в ряду 1.0 м (4000 кустов/га)
DEFAULT_ROW_SPACING_M = 2.5
DEFAULT_BUNCH_WEIGHT_G = 180.0  # Средняя масса грозди для технических сортов


@dataclass(frozen=True)
class RawDetection:
    frame_id: str
    camera_x_m: float
    camera_y_m: float
    depth_m: float
    robot_lat: float
    robot_lon: float
    row_id: str
    box_area_norm: float
    sugar_brix: float
    is_diseased: bool


@dataclass
class ClusteredBunch:
    bunch_id: str
    row_id: str
    world_lat: float
    world_lon: float
    est_weight_g: float
    sugar_brix: float
    is_diseased: bool
    detection_count: int


@dataclass
class YieldMapCell:
    cell_id: str
    center_lat: float
    center_lon: float
    row_id: str
    total_bunches: int
    yield_t_per_ha: float
    mean_brix: float
    disease_rate_pct: float


def cluster_detections(
    detections: list[RawDetection],
    spatial_radius_m: float = 0.15,
) -> list[ClusteredBunch]:
    """Группирует повторные детекции одной грозди на соседних кадрах."""
    if not detections:
        return []

    # Группировка по рядам
    by_row: dict[str, list[RawDetection]] = {}
    for det in detections:
        by_row.setdefault(det.row_id, []).append(det)

    clustered: list[ClusteredBunch] = []
    bunch_idx = 0

    for row_id, row_dets in by_row.items():
        # Преобразование координат в метрическое приближение
        # 1 градус широты ~ 111 320 м, 1 градус долготы ~ 111 320 * cos(lat)
        ref_lat = row_dets[0].robot_lat
        cos_lat = math.cos(math.radians(ref_lat))
        m_per_lat = 111320.0
        m_per_lon = 111320.0 * cos_lat

        pts_m = np.array(
            [
                [
                    (d.robot_lat - ref_lat) * m_per_lat + d.camera_x_m,
                    (d.robot_lon - row_dets[0].robot_lon) * m_per_lon + d.camera_y_m,
                    d.depth_m,
                ]
                for d in row_dets
            ]
        )

        visited = np.zeros(len(row_dets), dtype=bool)

        for i in range(len(row_dets)):
            if visited[i]:
                continue
            visited[i] = True
            cluster_indices = [i]

            for j in range(i + 1, len(row_dets)):
                if visited[j]:
                    continue
                dist = float(np.linalg.norm(pts_m[i] - pts_m[j]))
                if dist <= spatial_radius_m:
                    visited[j] = True
                    cluster_indices.append(j)

            bunch_idx += 1
            # Усреднение характеристик кластера
            c_dets = [row_dets[idx] for idx in cluster_indices]
            mean_lat = float(np.mean([d.robot_lat for d in c_dets]))
            mean_lon = float(np.mean([d.robot_lon for d in c_dets]))
            mean_brix = float(np.mean([d.sugar_brix for d in c_dets]))
            any_diseased = any(d.is_diseased for d in c_dets)

            # Оценка массы: пропорциональна площади проекции и квадрату расстояния
            mean_area = float(np.mean([d.box_area_norm for d in c_dets]))
            mean_depth = float(np.mean([d.depth_m for d in c_dets]))
            # Калибровка размера (при depth=1м нормализованная площадь 0.05 ~ 180г)
            est_weight = DEFAULT_BUNCH_WEIGHT_G * (mean_area / 0.05) * (mean_depth**1.5)
            est_weight = max(50.0, min(500.0, est_weight))

            clustered.append(
                ClusteredBunch(
                    bunch_id=f"b_{row_id}_{bunch_idx:04d}",
                    row_id=row_id,
                    world_lat=round(mean_lat, 7),
                    world_lon=round(mean_lon, 7),
                    est_weight_g=round(est_weight, 1),
                    sugar_brix=round(mean_brix, 1),
                    is_diseased=any_diseased,
                    detection_count=len(cluster_indices),
                )
            )

    return clustered


def aggregate_yield_grid(
    bunches: list[ClusteredBunch],
    row_spacing_m: float = DEFAULT_ROW_SPACING_M,
    cell_length_m: float = 10.0,
) -> list[YieldMapCell]:
    """Агрегирует уникальные грозди в ячейки вдоль шпалеры с расчётом т/га."""
    if not bunches:
        return []

    # Группировка по рядам
    by_row: dict[str, list[ClusteredBunch]] = {}
    for b in bunches:
        by_row.setdefault(b.row_id, []).append(b)

    cells: list[YieldMapCell] = []
    cell_idx = 0

    for row_id, r_bunches in by_row.items():
        # Сортировка гроздей по широте/долготе вдоль ряда
        r_bunches.sort(key=lambda x: (x.world_lat, x.world_lon))

        # Разбиение на участки по cell_length_m
        chunk_size = max(1, int(len(r_bunches) / max(1, len(r_bunches) // 15)))
        for i in range(0, len(r_bunches), chunk_size):
            chunk = r_bunches[i : i + chunk_size]
            cell_idx += 1

            total_bunches = len(chunk)
            total_weight_kg = sum(b.est_weight_g for b in chunk) / 1000.0
            mean_brix = float(np.mean([b.sugar_brix for b in chunk]))
            diseased_count = sum(1 for b in chunk if b.is_diseased)
            disease_rate = (
                (diseased_count / total_bunches * 100.0) if total_bunches else 0.0
            )

            # Расчёт урожайности в т/га:
            # Площадь участка = cell_length_m * row_spacing_m (в м^2)
            # В 1 га = 10 000 м^2
            area_m2 = cell_length_m * row_spacing_m
            yield_t_per_ha = (total_weight_kg / 1000.0) / (area_m2 / 10000.0)

            mean_lat = float(np.mean([b.world_lat for b in chunk]))
            mean_lon = float(np.mean([b.world_lon for b in chunk]))

            cells.append(
                YieldMapCell(
                    cell_id=f"cell_{row_id}_{cell_idx:03d}",
                    center_lat=round(mean_lat, 7),
                    center_lon=round(mean_lon, 7),
                    row_id=row_id,
                    total_bunches=total_bunches,
                    yield_t_per_ha=round(yield_t_per_ha, 2),
                    mean_brix=round(mean_brix, 1),
                    disease_rate_pct=round(disease_rate, 1),
                )
            )

    return cells


def export_geojson(cells: list[YieldMapCell], out_path: Path) -> dict[str, Any]:
    """Экспортирует ячейки урожайности в формат GeoJSON FeatureCollection."""
    features = []
    for c in cells:
        # Полигональный прямоугольник ячейки вокруг центра (ширина ~2м, длина ~10м)
        delta_lat = 0.00005
        delta_lon = 0.00007
        coords = [
            [
                [c.center_lon - delta_lon, c.center_lat - delta_lat],
                [c.center_lon + delta_lon, c.center_lat - delta_lat],
                [c.center_lon + delta_lon, c.center_lat + delta_lat],
                [c.center_lon - delta_lon, c.center_lat + delta_lat],
                [c.center_lon - delta_lon, c.center_lat - delta_lat],
            ]
        ]
        feat = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": coords},
            "properties": {
                "cell_id": c.cell_id,
                "row_id": c.row_id,
                "total_bunches": c.total_bunches,
                "yield_t_per_ha": c.yield_t_per_ha,
                "mean_sugar_brix": c.mean_brix,
                "disease_rate_pct": c.disease_rate_pct,
            },
        }
        features.append(feat)

    geojson_data = {
        "type": "FeatureCollection",
        "features": features,
    }
    out_path.write_text(
        json.dumps(geojson_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return geojson_data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("yield_map.geojson"))
    args = parser.parse_args(argv)

    # Генерация демонстрационного проезда
    sample_dets = [
        RawDetection(
            frame_id=f"f_{i:03d}",
            camera_x_m=0.02 * (i % 3),
            camera_y_m=0.01,
            depth_m=0.9,
            robot_lat=47.1234 + i * 0.00001,
            robot_lon=28.5678,
            row_id="row_01",
            box_area_norm=0.05,
            sugar_brix=21.2 + (i % 2) * 0.4,
            is_diseased=(i % 15 == 0),
        )
        for i in range(50)
    ]

    clustered = cluster_detections(sample_dets)
    grid = aggregate_yield_grid(clustered)
    export_geojson(grid, args.out)
    print(f"Экспортировано {len(grid)} ячеек урожайности в {args.out}")


if __name__ == "__main__":
    main()
