"""Тесты алгоритмов БПЛА-анализа, детекции выпадов лозы и расчёта NDRE (sprint-30)."""

import pytest

from app.services.uav_orthomosaic import (
    MultispectralPixel,
    UAVOrthomosaicAnalyzer,
)


def test_multispectral_pixel_indices() -> None:
    """Проверка расчёта NDVI, NDRE и GNDVI."""
    # Здоровая зрелая листва: высокое отражение в NIR (0.55), умеренное в RedEdge (0.20), низкое в Red (0.08)
    pixel = MultispectralPixel(
        offset_m=10.0,
        green=0.12,
        red=0.08,
        red_edge=0.20,
        nir=0.55,
    )
    # NDVI = (0.55 - 0.08) / (0.55 + 0.08) = 0.47 / 0.63 ~ 0.746
    assert pixel.ndvi == pytest.approx(0.746, abs=1e-2)
    # NDRE = (0.55 - 0.20) / (0.55 + 0.20) = 0.35 / 0.75 ~ 0.467
    assert pixel.ndre == pytest.approx(0.467, abs=1e-2)
    # GNDVI = (0.55 - 0.12) / (0.55 + 0.12) = 0.43 / 0.67 ~ 0.642
    assert pixel.gndvi == pytest.approx(0.642, abs=1e-2)


def test_detect_missing_vines_gap() -> None:
    """Проверка обнаружения разрыва шпалеры (выпавших кустов)."""
    analyzer = UAVOrthomosaicAnalyzer(
        canopy_ndvi_threshold=0.40,
        nominal_vine_spacing_m=1.25,
    )

    # Моделируем ряд 15 м:
    # 0..4м: здоровая лоза (NDVI ~0.7)
    # 4..7.5м: пустота / голая земля (пропуск 3.5 м -> ~3 куста, NDVI ~0.15)
    # 7.5..15м: здоровая лоза
    pixels: list[MultispectralPixel] = []
    for off in [0.0, 1.0, 2.0, 3.0, 4.0]:
        pixels.append(MultispectralPixel(off, 0.1, 0.08, 0.2, 0.55))
    for off in [4.5, 5.0, 6.0, 7.0]:
        pixels.append(
            MultispectralPixel(off, 0.15, 0.25, 0.28, 0.30)
        )  # земля: NIR 0.30, Red 0.25 -> NDVI ~0.09
    for off in [7.5, 9.0, 11.0, 13.0, 15.0]:
        pixels.append(MultispectralPixel(off, 0.1, 0.08, 0.2, 0.55))

    start_pt = (47.140000, 28.860000)
    end_pt = (47.141500, 28.860000)

    gaps = analyzer.detect_missing_vines_in_row(
        row_id="row_01",
        row_length_m=15.0,
        start_point=start_pt,
        end_point=end_pt,
        transect_pixels=pixels,
    )

    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.row_id == "row_01"
    assert gap.start_offset_m == 4.5
    assert gap.end_offset_m == 7.5
    assert gap.gap_length_m == pytest.approx(3.0, abs=0.1)
    assert gap.estimated_missing_vines in (2, 3)

    # GPS координата должна лежать между началом и концом
    assert start_pt[0] < gap.gps_coordinate[0] < end_pt[0]


def test_detect_canopy_stress_ndre() -> None:
    """Проверка выявления очага хлороза/стресса по NDRE."""
    analyzer = UAVOrthomosaicAnalyzer(
        canopy_ndvi_threshold=0.35,
        stress_ndre_threshold=0.25,
    )

    # Создаём кластер из 4 точек с угнетённым синтезом хлорофилла:
    # NDVI ещё высокий (0.45), но RedEdge растёт и NDRE падает до 0.15
    pixels = [
        MultispectralPixel(0.0, 0.1, 0.08, 0.2, 0.55),  # здоровый
        MultispectralPixel(1.0, 0.1, 0.08, 0.2, 0.55),
        # Стрессовый очаг хлороза
        MultispectralPixel(
            3.0, 0.15, 0.12, 0.40, 0.50
        ),  # NDRE = (0.5-0.4)/(0.5+0.4) = 0.11
        MultispectralPixel(4.0, 0.15, 0.12, 0.38, 0.50),  # NDRE = 0.136
        MultispectralPixel(5.0, 0.15, 0.12, 0.39, 0.50),  # NDRE = 0.123
        MultispectralPixel(6.0, 0.15, 0.12, 0.40, 0.50),
        # Возврат в норму
        MultispectralPixel(8.0, 0.1, 0.08, 0.2, 0.55),
    ]

    stress_zones = analyzer.detect_canopy_stress("row_stress", pixels)
    assert len(stress_zones) == 1
    zone = stress_zones[0]
    assert zone.start_offset_m == 3.0
    assert zone.end_offset_m == 6.0
    assert zone.mean_ndre < 0.20
    assert zone.stress_severity == "severe"


def test_summarize_replanting_order() -> None:
    """Проверка ведомости заказа саженцев с 5% резервом."""
    from app.services.uav_orthomosaic import MissingVineGap

    analyzer = UAVOrthomosaicAnalyzer()
    gaps = [
        MissingVineGap("g1", "r1", 5.0, 8.0, 3.0, 2, (47.14, 28.86)),
        MissingVineGap("g2", "r2", 12.0, 18.0, 6.0, 5, (47.142, 28.862)),
    ]

    summary = analyzer.summarize_replanting_order(
        field_id="cricova_plot_4",
        variety_name="Feteasca Regala",
        gaps=gaps,
        total_nominal_vines=1000,
    )

    assert summary["total_missing_vines"] == 7
    assert summary["mortality_rate_pct"] == 0.7
    # 7 * 1.05 = 7.35 -> ceil = 8 саженцев
    assert summary["replanting_order_vines_count"] == 8
    assert summary["geojson_missing_points"]["type"] == "FeatureCollection"
    assert len(summary["geojson_missing_points"]["features"]) == 2
