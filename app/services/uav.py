"""Сервис обработки мультиспектральных данных БПЛА и формирования ведомостей подсадки (sprint-30)."""

from typing import Any

from app.schemas.uav import (
    CanopyStressZoneDTO,
    MissingVineGapDTO,
    ReplantingOrderRequest,
    ReplantingOrderResponse,
    UAVAnalysisRequest,
    UAVAnalysisResponse,
)
from app.services.uav_orthomosaic import (
    CanopyStressZone,
    MissingVineGap,
    MultispectralPixel,
    UAVOrthomosaicAnalyzer,
)


def analyze_uav_transects(request: UAVAnalysisRequest) -> UAVAnalysisResponse:
    """Анализирует рядные мультиспектральные трансекты БПЛА.

    Обнаруживает выпады кустов и зоны хлорозного стресса.
    """
    analyzer = UAVOrthomosaicAnalyzer(
        canopy_ndvi_threshold=request.canopy_ndvi_threshold,
        nominal_vine_spacing_m=request.nominal_vine_spacing_m,
    )

    all_gaps: list[MissingVineGap] = []
    all_stress: list[CanopyStressZone] = []

    for transect in request.transects:
        domain_pixels = [
            MultispectralPixel(
                offset_m=p.offset_m,
                green=p.green,
                red=p.red,
                red_edge=p.red_edge,
                nir=p.nir,
            )
            for p in transect.pixels
        ]

        # 1. Поиск выпадов лозы
        row_gaps = analyzer.detect_missing_vines_in_row(
            row_id=transect.row_id,
            row_length_m=transect.row_length_m,
            start_point=transect.start_point,
            end_point=transect.end_point,
            transect_pixels=domain_pixels,
        )
        all_gaps.extend(row_gaps)

        # 2. Поиск стресса хлорофилла (NDRE)
        row_stress = analyzer.detect_canopy_stress(
            row_id=transect.row_id,
            transect_pixels=domain_pixels,
        )
        all_stress.extend(row_stress)

    total_missing = sum(g.estimated_missing_vines for g in all_gaps)

    gap_dtos = [
        MissingVineGapDTO(
            gap_id=g.gap_id,
            row_id=g.row_id,
            start_offset_m=g.start_offset_m,
            end_offset_m=g.end_offset_m,
            gap_length_m=g.gap_length_m,
            estimated_missing_vines=g.estimated_missing_vines,
            gps_coordinate=g.gps_coordinate,
        )
        for g in all_gaps
    ]

    stress_dtos = [
        CanopyStressZoneDTO(
            zone_id=s.zone_id,
            row_id=s.row_id,
            start_offset_m=s.start_offset_m,
            end_offset_m=s.end_offset_m,
            mean_ndre=s.mean_ndre,
            mean_ndvi=s.mean_ndvi,
            stress_severity=s.stress_severity,
        )
        for s in all_stress
    ]

    geojson_features: list[dict[str, Any]] = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [g.gps_coordinate[1], g.gps_coordinate[0]],  # [lon, lat]
            },
            "properties": {
                "gap_id": g.gap_id,
                "row_id": g.row_id,
                "missing_vines": g.estimated_missing_vines,
                "gap_length_m": g.gap_length_m,
            },
        }
        for g in all_gaps
    ]

    return UAVAnalysisResponse(
        field_id=request.field_id,
        missing_gaps=gap_dtos,
        stress_zones=stress_dtos,
        total_missing_vines=total_missing,
        gaps_count=len(gap_dtos),
        geojson_gaps={
            "type": "FeatureCollection",
            "features": geojson_features,
        },
    )


def generate_replanting_specification(
    request: ReplantingOrderRequest,
) -> ReplantingOrderResponse:
    """Генерирует официальную производственную спецификацию заказа саженцев."""
    domain_gaps = [
        MissingVineGap(
            gap_id=g.gap_id,
            row_id=g.row_id,
            start_offset_m=g.start_offset_m,
            end_offset_m=g.end_offset_m,
            gap_length_m=g.gap_length_m,
            estimated_missing_vines=g.estimated_missing_vines,
            gps_coordinate=g.gps_coordinate,
        )
        for g in request.gaps
    ]

    analyzer = UAVOrthomosaicAnalyzer()
    summary = analyzer.summarize_replanting_order(
        field_id=request.field_id,
        variety_name=request.variety_name,
        gaps=domain_gaps,
        total_nominal_vines=request.total_nominal_vines,
    )

    return ReplantingOrderResponse(
        field_id=summary["field_id"],
        variety_name=summary["variety_name"],
        total_nominal_vines=summary["total_nominal_vines"],
        total_missing_vines=summary["total_missing_vines"],
        mortality_rate_pct=summary["mortality_rate_pct"],
        replanting_order_vines_count=summary["replanting_order_vines_count"],
        gaps_count=summary["gaps_count"],
        geojson_missing_points=summary["geojson_missing_points"],
    )
