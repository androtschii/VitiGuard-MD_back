"""Обработка мультиспектральных ортофотопланов БПЛА и детекция выпадов лозы (sprint-30).

Реализует:
1. Расчёт ультравысокоточных вегетационных индексов по каналам БПЛА:
   - NDRE (Normalized Difference Red Edge): (NIR - RedEdge) / (NIR + RedEdge);
   - GNDVI (Green NDVI): (NIR - Green) / (NIR + Green);
   - NDVI (NIR - Red) / (NIR + Red).
2. Анализ профиля листового полога вдоль шпалерного ряда (transect profile).
3. Детекцию выпадов лозы (missing vines): определение пропусков в шпалере (> 1.2 м),
   где индекс вегетации опускается ниже порога растительности лозы.
4. Расчёт точных GPS-координат каждого выпавшего куста для весенней подсадки.
5. Кластеризацию зон хлороза и раннего стресса листового аппарата.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MultispectralPixel:
    """Значения каналов мультиспектрального сенсора дрона в точке ряда."""

    offset_m: float  # Смещение от начала ряда (м)
    green: float  # Коэффициент отражения зелёного канала (0..1)
    red: float  # Красный канал (0..1)
    red_edge: float  # RedEdge канал ~730 нм (0..1)
    nir: float  # Ближний инфракрасный канал ~840 нм (0..1)

    @property
    def ndvi(self) -> float:
        denom = self.nir + self.red
        return (self.nir - self.red) / denom if denom > 1e-4 else 0.0

    @property
    def ndre(self) -> float:
        denom = self.nir + self.red_edge
        return (self.nir - self.red_edge) / denom if denom > 1e-4 else 0.0

    @property
    def gndvi(self) -> float:
        denom = self.nir + self.green
        return (self.nir - self.green) / denom if denom > 1e-4 else 0.0


@dataclass(frozen=True)
class MissingVineGap:
    """Обнаруженный пропуск (выпад куста) в шпалерном ряду."""

    gap_id: str
    row_id: str
    start_offset_m: float
    end_offset_m: float
    gap_length_m: float
    estimated_missing_vines: int
    gps_coordinate: tuple[float, float]  # (lat, lon) центра выпада


@dataclass(frozen=True)
class CanopyStressZone:
    """Зона снижения хлорофилла / хлороза по индексу NDRE."""

    zone_id: str
    row_id: str
    start_offset_m: float
    end_offset_m: float
    mean_ndre: float
    mean_ndvi: float
    stress_severity: str  # mild | moderate | severe


class UAVOrthomosaicAnalyzer:
    """Анализатор мультиспектральных данных пролёта БПЛА."""

    def __init__(
        self,
        canopy_ndvi_threshold: float = 0.40,
        nominal_vine_spacing_m: float = 1.25,
        stress_ndre_threshold: float = 0.28,
    ) -> None:
        self.canopy_ndvi_threshold = canopy_ndvi_threshold
        self.nominal_vine_spacing_m = nominal_vine_spacing_m
        self.stress_ndre_threshold = stress_ndre_threshold

    def interpolate_gps_coordinate(
        self,
        start_point: tuple[float, float],
        end_point: tuple[float, float],
        fraction: float,
    ) -> tuple[float, float]:
        """Линейная интерполяция GPS координат вдоль прямолинейного ряда."""
        lat = start_point[0] + (end_point[0] - start_point[0]) * fraction
        lon = start_point[1] + (end_point[1] - start_point[1]) * fraction
        return (round(lat, 6), round(lon, 6))

    def detect_missing_vines_in_row(
        self,
        row_id: str,
        row_length_m: float,
        start_point: tuple[float, float],
        end_point: tuple[float, float],
        transect_pixels: Sequence[MultispectralPixel],
    ) -> list[MissingVineGap]:
        """Ищет разрывы полога, превышающие шаг посадки, где нет здоровой лозы."""
        if not transect_pixels or row_length_m <= 0:
            return []

        sorted_pixels = sorted(transect_pixels, key=lambda p: p.offset_m)
        gaps: list[MissingVineGap] = []

        in_gap = False
        gap_start = 0.0

        for p in sorted_pixels:
            is_canopy = p.ndvi >= self.canopy_ndvi_threshold

            if not is_canopy and not in_gap:
                in_gap = True
                gap_start = p.offset_m
            elif is_canopy and in_gap:
                in_gap = False
                gap_end = p.offset_m
                length = gap_end - gap_start

                # Пропуск считается выпадом, если превышает 80% номинального шага посадки
                if length >= self.nominal_vine_spacing_m * 0.8:
                    missing_count = max(1, round(length / self.nominal_vine_spacing_m))
                    mid_offset = (gap_start + gap_end) / 2.0
                    fraction = min(1.0, max(0.0, mid_offset / row_length_m))
                    gps = self.interpolate_gps_coordinate(
                        start_point, end_point, fraction
                    )

                    gaps.append(
                        MissingVineGap(
                            gap_id=f"{row_id}_gap_{len(gaps) + 1}",
                            row_id=row_id,
                            start_offset_m=round(gap_start, 2),
                            end_offset_m=round(gap_end, 2),
                            gap_length_m=round(length, 2),
                            estimated_missing_vines=missing_count,
                            gps_coordinate=gps,
                        )
                    )

        # Если ряд заканчивается пропуском
        if in_gap:
            gap_end = sorted_pixels[-1].offset_m
            length = gap_end - gap_start
            if length >= self.nominal_vine_spacing_m * 0.8:
                missing_count = max(1, round(length / self.nominal_vine_spacing_m))
                mid_offset = (gap_start + gap_end) / 2.0
                fraction = min(1.0, max(0.0, mid_offset / row_length_m))
                gps = self.interpolate_gps_coordinate(start_point, end_point, fraction)

                gaps.append(
                    MissingVineGap(
                        gap_id=f"{row_id}_gap_{len(gaps) + 1}",
                        row_id=row_id,
                        start_offset_m=round(gap_start, 2),
                        end_offset_m=round(gap_end, 2),
                        gap_length_m=round(length, 2),
                        estimated_missing_vines=missing_count,
                        gps_coordinate=gps,
                    )
                )

        return gaps

    def detect_canopy_stress(
        self,
        row_id: str,
        transect_pixels: Sequence[MultispectralPixel],
    ) -> list[CanopyStressZone]:
        """Ищет очаги хлороза и раннего водного/хлорофилльного стресса по NDRE."""
        # Рассматриваем только живой полог (NDVI >= порог)
        canopy_pixels = [
            p for p in transect_pixels if p.ndvi >= self.canopy_ndvi_threshold
        ]
        if not canopy_pixels:
            return []

        stress_zones: list[CanopyStressZone] = []
        current_cluster: list[MultispectralPixel] = []

        for p in canopy_pixels:
            if p.ndre < self.stress_ndre_threshold:
                current_cluster.append(p)
            else:
                if len(current_cluster) >= 3:
                    stress_zones.append(
                        self._make_stress_zone(
                            row_id, current_cluster, len(stress_zones) + 1
                        )
                    )
                current_cluster = []

        if len(current_cluster) >= 3:
            stress_zones.append(
                self._make_stress_zone(row_id, current_cluster, len(stress_zones) + 1)
            )

        return stress_zones

    def _make_stress_zone(
        self,
        row_id: str,
        cluster: list[MultispectralPixel],
        idx: int,
    ) -> CanopyStressZone:
        mean_ndre = sum(p.ndre for p in cluster) / len(cluster)
        mean_ndvi = sum(p.ndvi for p in cluster) / len(cluster)

        if mean_ndre < 0.18:
            severity = "severe"
        elif mean_ndre < 0.24:
            severity = "moderate"
        else:
            severity = "mild"

        return CanopyStressZone(
            zone_id=f"{row_id}_stress_{idx}",
            row_id=row_id,
            start_offset_m=round(cluster[0].offset_m, 2),
            end_offset_m=round(cluster[-1].offset_m, 2),
            mean_ndre=round(mean_ndre, 3),
            mean_ndvi=round(mean_ndvi, 3),
            stress_severity=severity,
        )

    def summarize_replanting_order(
        self,
        field_id: str,
        variety_name: str,
        gaps: Sequence[MissingVineGap],
        total_nominal_vines: int,
    ) -> dict[str, Any]:
        """Формирует производственную спецификацию на досадку саженцев."""
        total_missing = sum(g.estimated_missing_vines for g in gaps)
        safe_nominal = max(1, total_nominal_vines)
        loss_pct = (total_missing / safe_nominal) * 100.0

        return {
            "field_id": field_id,
            "variety_name": variety_name,
            "total_nominal_vines": safe_nominal,
            "total_missing_vines": total_missing,
            "mortality_rate_pct": round(loss_pct, 2),
            "replanting_order_vines_count": math.ceil(
                total_missing * 1.05
            ),  # +5% страховой резерв
            "gaps_count": len(gaps),
            "geojson_missing_points": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [
                                g.gps_coordinate[1],
                                g.gps_coordinate[0],
                            ],  # [lon, lat]
                        },
                        "properties": {
                            "gap_id": g.gap_id,
                            "row_id": g.row_id,
                            "missing_vines": g.estimated_missing_vines,
                            "gap_length_m": g.gap_length_m,
                        },
                    }
                    for g in gaps
                ],
            },
        }
