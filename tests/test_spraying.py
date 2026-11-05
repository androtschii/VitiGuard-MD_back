"""Тесты сервиса дифференцированного внесения СЗР и генерации задач ISO-XML (sprint-28)."""

import uuid
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from app.api.deps import get_current_user
from app.main import create_app
from app.models.user import User
from app.schemas.spraying import (
    IsoXmlExportRequest,
    SprayerEquipmentDTO,
    SpraySectorDTO,
)
from app.services.spraying import (
    calculate_flow_rate_l_min,
    calculate_trv,
    compute_sector_dosage,
    export_task_data_isoxml,
    generate_vra_prescription,
)


def _mock_agronomist() -> User:
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.email = "agronomist@vitiguard.md"
    user.role = "agronomist"
    user.is_active = True
    return user


def test_calculate_trv() -> None:
    """Проверка расчёта Tree Row Volume (TRV, м³/га)."""
    # Стандартный полог: высота 1.4 м, толщина 0.45 м, междурядье 2.5 м
    trv = calculate_trv(canopy_height_m=1.4, canopy_width_m=0.45, row_width_m=2.5)
    assert trv == pytest.approx(2520.0, rel=1e-2)

    # Низкий редкий полог (молодой виноградник): высота 0.8 м, толщина 0.3 м, междурядье 3.0 м
    trv_young = calculate_trv(canopy_height_m=0.8, canopy_width_m=0.3, row_width_m=3.0)
    assert trv_young == pytest.approx(800.0, rel=1e-2)


def test_calculate_flow_rate_l_min() -> None:
    """Проверка пересчёта нормы внесения в минутный расход форсунок."""
    # 300 л/га при 6 км/ч и междурядье 2.5 м: Q = (300 * 6 * 2.5) / 600 = 7.5 л/мин
    flow = calculate_flow_rate_l_min(
        application_rate_l_ha=300.0, travel_speed_kmh=6.0, boom_width_m=2.5
    )
    assert flow == pytest.approx(7.5, rel=1e-2)


def test_compute_sector_dosage_and_limits() -> None:
    """Проверка мультипликатора дозировки и ограничения min/max."""
    equipment = SprayerEquipmentDTO(
        sprayer_id="sp-1",
        nominal_application_rate_l_ha=350.0,
        min_application_rate_l_ha=200.0,
        max_application_rate_l_ha=500.0,
        boom_width_m=2.5,
    )

    # 1. Сильно разреженный полог с низким NDVI и нулевым риском болезни
    sparse_sector = SpraySectorDTO(
        sector_id="sec-sparse",
        row_number=1,
        length_m=100.0,
        start_lat=47.14,
        start_lon=28.86,
        end_lat=47.15,
        end_lon=28.86,
        canopy_height_m=0.6,
        canopy_width_m=0.2,
        ndvi_index=0.2,
        disease_pressure_score=0.0,
    )
    rate, mult, _, status = compute_sector_dosage(sparse_sector, equipment)
    assert mult < 1.0
    assert rate == equipment.min_application_rate_l_ha
    assert status == "capped_min"

    # 2. Густой полог с высоким риском вспышки милдью
    dense_sick_sector = SpraySectorDTO(
        sector_id="sec-dense",
        row_number=2,
        length_m=100.0,
        start_lat=47.14,
        start_lon=28.87,
        end_lat=47.15,
        end_lon=28.87,
        canopy_height_m=2.2,
        canopy_width_m=0.9,
        ndvi_index=0.9,
        disease_pressure_score=0.9,
    )
    rate_dense, mult_dense, _, status_dense = compute_sector_dosage(
        dense_sick_sector, equipment
    )
    assert mult_dense > 1.2
    assert rate_dense == equipment.max_application_rate_l_ha
    assert status_dense == "capped_max"


def test_generate_vra_prescription_report() -> None:
    """Проверка формирования карты предписаний и отчёта экономии ресурсов."""
    equipment = SprayerEquipmentDTO(
        sprayer_id="sp-test",
        nominal_application_rate_l_ha=400.0,
        min_application_rate_l_ha=150.0,
        max_application_rate_l_ha=600.0,
        boom_width_m=2.5,
        chemical_concentration_g_l=10.0,
        chemical_price_per_kg_mdl=500.0,
    )

    sectors = [
        SpraySectorDTO(
            sector_id="s1",
            row_number=1,
            length_m=200.0,
            start_lat=47.1,
            start_lon=28.8,
            end_lat=47.102,
            end_lon=28.8,
            canopy_height_m=1.0,
            canopy_width_m=0.3,
            ndvi_index=0.4,
            disease_pressure_score=0.1,
        ),
        SpraySectorDTO(
            sector_id="s2",
            row_number=2,
            length_m=200.0,
            start_lat=47.1,
            start_lon=28.801,
            end_lat=47.102,
            end_lon=28.801,
            canopy_height_m=1.8,
            canopy_width_m=0.6,
            ndvi_index=0.75,
            disease_pressure_score=0.5,
        ),
    ]

    res = generate_vra_prescription(equipment, sectors)

    assert len(res.prescription_zones) == 2
    assert res.summary.treated_area_ha == pytest.approx(
        0.1, rel=1e-2
    )  # 2 ряда по 200м * 2.5м = 1000 м² = 0.1 га
    assert res.summary.liquid_saved_liters >= 0.0
    assert res.summary.cost_savings_mdl >= 0.0
    assert res.geojson_feature_collection["type"] == "FeatureCollection"
    assert len(res.geojson_feature_collection["features"]) == 2


def test_export_task_data_isoxml() -> None:
    """Проверка экспорта в XML по стандарту ISO 11783-10."""
    req = IsoXmlExportRequest(
        task_name="Fungicide_Spray_May2026",
        field_name="Plot_Purcari_Cabernet",
        equipment=SprayerEquipmentDTO(sprayer_id="berthoud-01"),
        sectors=[
            SpraySectorDTO(
                sector_id="p1",
                row_number=5,
                length_m=150.0,
                start_lat=47.1,
                start_lon=28.8,
                end_lat=47.101,
                end_lon=28.8,
            )
        ],
    )
    xml_str = export_task_data_isoxml(req)
    assert xml_str.startswith("<?xml")

    # Валидация парсинга XML
    root = ET.fromstring(xml_str)
    assert root.tag == "ISO11783_TaskData"
    assert root.get("VersionMajor") == "4"

    pfd = root.find("PFD")
    assert pfd is not None
    assert pfd.get("C") == "Plot_Purcari_Cabernet"

    tsk = root.find("TSK")
    assert tsk is not None
    assert tsk.get("B") == "Fungicide_Spray_May2026"

    pzn = tsk.find("PZN")
    assert pzn is not None
    pdv = pzn.find("PDV")
    assert pdv is not None
    assert pdv.get("A") == "0001"  # DDI volume rate


def test_spraying_api_endpoints() -> None:
    """Тест эндпоинтов API /api/v1/spraying."""
    app = create_app()
    user = _mock_agronomist()
    app.dependency_overrides[get_current_user] = lambda: user

    client = TestClient(app)

    payload = {
        "equipment": {
            "sprayer_id": "sprayer-alpha",
            "nominal_application_rate_l_ha": 350.0,
            "min_application_rate_l_ha": 150.0,
            "max_application_rate_l_ha": 550.0,
            "boom_width_m": 2.5,
            "travel_speed_kmh": 6.0,
        },
        "sectors": [
            {
                "sector_id": "sec-01",
                "row_number": 1,
                "length_m": 120.0,
                "start_lat": 47.14,
                "start_lon": 28.86,
                "end_lat": 47.142,
                "end_lon": 28.86,
                "canopy_height_m": 1.4,
                "canopy_width_m": 0.45,
                "ndvi_index": 0.65,
                "disease_pressure_score": 0.3,
            }
        ],
    }

    # 1. Генерация карты VRA
    res = client.post("/api/v1/spraying/generate-vra-map", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "prescription_zones" in data
    assert "summary" in data
    assert len(data["prescription_zones"]) == 1

    # 2. Оценка экономии
    res_sav = client.post("/api/v1/spraying/savings-estimate", json=payload)
    assert res_sav.status_code == 200
    sav_data = res_sav.json()
    assert "liquid_saved_liters" in sav_data
    assert "cost_savings_mdl" in sav_data

    # 3. Экспорт в ISO-XML
    iso_payload = {
        "task_name": "Spray_Sprint28",
        "field_name": "Field_10",
        **payload,
    }
    res_iso = client.post("/api/v1/spraying/export-isoxml", json=iso_payload)
    assert res_iso.status_code == 200
    iso_data = res_iso.json()
    assert "ISO11783_TaskData" in iso_data["xml_content"]
    assert iso_data["zones_count"] == 1

    # 4. Скачивание образца XML
    res_sample = client.get("/api/v1/spraying/isoxml/download-sample")
    assert res_sample.status_code == 200
    assert "application/xml" in res_sample.headers["content-type"]
