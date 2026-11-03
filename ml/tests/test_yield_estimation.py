from pathlib import Path

from vitiguard_ml.robot.yield_estimation import (
    RawDetection,
    aggregate_yield_grid,
    cluster_detections,
    export_geojson,
)


def test_cluster_detections_merges_close_detections() -> None:
    # 3 кадра подряд видят одну и ту же гроздь со смещением в пару сантиметров
    dets = [
        RawDetection(
            frame_id="f1",
            camera_x_m=0.01,
            camera_y_m=0.02,
            depth_m=0.85,
            robot_lat=47.100000,
            robot_lon=28.800000,
            row_id="row_01",
            box_area_norm=0.05,
            sugar_brix=21.0,
            is_diseased=False,
        ),
        RawDetection(
            frame_id="f2",
            camera_x_m=0.03,
            camera_y_m=0.02,
            depth_m=0.86,
            robot_lat=47.100000,
            robot_lon=28.800000,
            row_id="row_01",
            box_area_norm=0.05,
            sugar_brix=21.2,
            is_diseased=False,
        ),
        RawDetection(
            frame_id="f3",
            camera_x_m=0.02,
            camera_y_m=0.03,
            depth_m=0.84,
            robot_lat=47.100000,
            robot_lon=28.800000,
            row_id="row_01",
            box_area_norm=0.05,
            sugar_brix=21.1,
            is_diseased=False,
        ),
    ]

    clusters = cluster_detections(dets, spatial_radius_m=0.15)
    assert len(clusters) == 1
    assert clusters[0].detection_count == 3
    assert clusters[0].sugar_brix == 21.1
    assert clusters[0].est_weight_g > 0


def test_cluster_detections_separates_distant_bunches() -> None:
    # Две грозди на расстоянии 1 метра друг от друга
    dets = [
        RawDetection(
            frame_id="f1",
            camera_x_m=0.0,
            camera_y_m=0.0,
            depth_m=0.8,
            robot_lat=47.100000,
            robot_lon=28.800000,
            row_id="row_01",
            box_area_norm=0.04,
            sugar_brix=20.5,
            is_diseased=False,
        ),
        RawDetection(
            frame_id="f5",
            camera_x_m=1.0,  # 1 метр в сторону
            camera_y_m=0.0,
            depth_m=0.8,
            robot_lat=47.100000,
            robot_lon=28.800000,
            row_id="row_01",
            box_area_norm=0.04,
            sugar_brix=22.0,
            is_diseased=True,
        ),
    ]

    clusters = cluster_detections(dets, spatial_radius_m=0.20)
    assert len(clusters) == 2
    assert clusters[0].is_diseased is False
    assert clusters[1].is_diseased is True


def test_aggregate_yield_grid_and_geojson_export(tmp_path: Path) -> None:
    dets = [
        RawDetection(
            frame_id=f"f_{i}",
            camera_x_m=0.0,
            camera_y_m=0.0,
            depth_m=0.9,
            robot_lat=47.1000 + i * 0.0001,
            robot_lon=28.8000,
            row_id="row_01",
            box_area_norm=0.05,
            sugar_brix=21.0,
            is_diseased=(i % 5 == 0),
        )
        for i in range(10)
    ]
    clusters = cluster_detections(dets, spatial_radius_m=0.10)
    cells = aggregate_yield_grid(clusters, row_spacing_m=2.5, cell_length_m=10.0)

    assert len(cells) > 0
    assert cells[0].yield_t_per_ha > 0
    assert cells[0].mean_brix == 21.0

    out_file = tmp_path / "yield.geojson"
    geo = export_geojson(cells, out_file)
    assert out_file.exists()
    assert geo["type"] == "FeatureCollection"
    assert len(geo["features"]) == len(cells)
    props = geo["features"][0]["properties"]
    assert "yield_t_per_ha" in props
    assert "mean_sugar_brix" in props
    assert "disease_rate_pct" in props
