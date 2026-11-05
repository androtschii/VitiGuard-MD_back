"""Сервис дифференцированного внесения средств защиты растений (VRA) и расчёта задач опрыскивания (sprint-28).

Реализует:
1. Модель расчёта объёма листового полога (Tree Row Volume, TRV, м³/га).
2. Мультифакторную оптимизацию нормы внесения на основе TRV, спутникового/дронового NDVI и инфекционного давления патогенов.
3. Расчёт минутного расхода форсунок (л/мин) при заданной рабочей скорости агрегата.
4. Формирование карт предписаний в GeoJSON и экспорт в стандарт точного земледелия ISO 11783-10 (TaskData XML).
5. Аналитику сокращения пестицидной нагрузки, экономии воды и финансовых затрат хозяйства.
"""

import xml.etree.ElementTree as ET
from typing import Any

from app.schemas.spraying import (
    IsoXmlExportRequest,
    PrescriptionZoneDTO,
    SprayerEquipmentDTO,
    SpraySectorDTO,
    VRAOptimizationReportDTO,
    VRAPrescriptionResponse,
)

# Эталонный объём пологовой стены для стандартного виноградника при междурядье 2.5 м (м³/га)
REFERENCE_TRV_M3_HA = 2520.0


def calculate_trv(
    canopy_height_m: float,
    canopy_width_m: float,
    row_width_m: float,
) -> float:
    """Рассчитывает удельный объём вегетативной массы (Tree Row Volume, TRV, м³/га).

    Формула: TRV = (10 000 * H_canopy * W_canopy) / Row_Spacing
    """
    row_width_clamped = max(0.8, row_width_m)
    trv = (
        10000.0 * max(0.1, canopy_height_m) * max(0.05, canopy_width_m)
    ) / row_width_clamped
    return round(trv, 1)


def calculate_flow_rate_l_min(
    application_rate_l_ha: float,
    travel_speed_kmh: float,
    boom_width_m: float,
) -> float:
    """Пересчитывает гектарную норму внесения в минутный расход опрыскивателя (л/мин).

    Формула ISO: Q (л/мин) = (Rate (л/га) * Speed (км/ч) * Width (м)) / 600
    """
    flow = (application_rate_l_ha * travel_speed_kmh * boom_width_m) / 600.0
    return round(flow, 2)


def compute_sector_dosage(
    sector: SpraySectorDTO,
    equipment: SprayerEquipmentDTO,
) -> tuple[float, float, float, str]:
    """Рассчитывает скорректированную норму внесения (л/га) для сектора.

    Возвращает:
        (target_rate_l_ha, multiplier, trv_m3_ha, status)
    """
    trv = calculate_trv(
        canopy_height_m=sector.canopy_height_m,
        canopy_width_m=sector.canopy_width_m,
        row_width_m=equipment.boom_width_m,
    )

    # 1. Поправочный коэффициент по биомассе TRV
    k_trv = trv / REFERENCE_TRV_M3_HA

    # 2. Поправочный коэффициент по вегетационному индексу NDVI
    k_ndvi = 0.5 + 0.8 * sector.ndvi_index

    # 3. Поправочный коэффициент по инфекционному фону (оидиум / милдью / ботритис)
    k_risk = 1.0 + 0.5 * (sector.disease_pressure_score - 0.3)

    # Комплексный весовой мультипликатор
    multiplier = 0.45 * k_trv + 0.35 * k_ndvi + 0.20 * k_risk
    multiplier = max(0.35, min(1.6, multiplier))

    raw_rate = equipment.nominal_application_rate_l_ha * multiplier

    status = "ok"
    if raw_rate < equipment.min_application_rate_l_ha:
        target_rate = equipment.min_application_rate_l_ha
        status = "capped_min"
    elif raw_rate > equipment.max_application_rate_l_ha:
        target_rate = equipment.max_application_rate_l_ha
        status = "capped_max"
    else:
        target_rate = raw_rate

    effective_multiplier = target_rate / equipment.nominal_application_rate_l_ha
    return (
        round(target_rate, 1),
        round(effective_multiplier, 3),
        trv,
        status,
    )


def generate_sector_polygon(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    buffer_width_deg: float = 0.00002,
) -> dict[str, Any]:
    """Формирует полигональный коридор вокруг отрезка ряда для GeoJSON."""
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [start_lon - buffer_width_deg, start_lat - buffer_width_deg],
                [end_lon - buffer_width_deg, end_lat - buffer_width_deg],
                [end_lon + buffer_width_deg, end_lat + buffer_width_deg],
                [start_lon + buffer_width_deg, start_lat + buffer_width_deg],
                [start_lon - buffer_width_deg, start_lat - buffer_width_deg],
            ]
        ],
    }


def generate_vra_prescription(
    equipment: SprayerEquipmentDTO,
    sectors: list[SpraySectorDTO],
) -> VRAPrescriptionResponse:
    """Формирует полную карту дифференцированного внесения СЗР."""
    zones: list[PrescriptionZoneDTO] = []
    features: list[dict[str, Any]] = []

    total_area_ha = 0.0
    uniform_liquid_liters = 0.0
    vra_liquid_liters = 0.0

    for s in sectors:
        target_rate, multiplier, trv, status = compute_sector_dosage(s, equipment)
        flow_rate = calculate_flow_rate_l_min(
            application_rate_l_ha=target_rate,
            travel_speed_kmh=equipment.travel_speed_kmh,
            boom_width_m=equipment.boom_width_m,
        )

        # Площадь сектора в га: (длина * ширина захвата) / 10 000
        sector_area_ha = (s.length_m * equipment.boom_width_m) / 10000.0
        total_area_ha += sector_area_ha

        uniform_liquid_liters += (
            sector_area_ha * equipment.nominal_application_rate_l_ha
        )
        vra_liquid_liters += sector_area_ha * target_rate

        ai_kg_ha = round(
            (target_rate * equipment.chemical_concentration_g_l) / 1000.0, 3
        )

        polygon_geom = generate_sector_polygon(
            s.start_lat, s.start_lon, s.end_lat, s.end_lon
        )

        zone_dto = PrescriptionZoneDTO(
            sector_id=s.sector_id,
            target_rate_l_ha=target_rate,
            dosage_multiplier=multiplier,
            target_flow_rate_l_min=flow_rate,
            tree_row_volume_m3_ha=trv,
            active_ingredient_rate_kg_ha=ai_kg_ha,
            status=status,
            geometry_geojson=polygon_geom,
        )
        zones.append(zone_dto)

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "sector_id": s.sector_id,
                    "row_number": s.row_number,
                    "target_rate_l_ha": target_rate,
                    "dosage_multiplier": multiplier,
                    "flow_l_min": flow_rate,
                    "trv_m3_ha": trv,
                    "status": status,
                },
                "geometry": polygon_geom,
            }
        )

    liquid_saved = max(0.0, uniform_liquid_liters - vra_liquid_liters)
    liquid_saved_pct = (liquid_saved / max(1.0, uniform_liquid_liters)) * 100.0

    uniform_chem_kg = (
        uniform_liquid_liters * equipment.chemical_concentration_g_l
    ) / 1000.0
    vra_chem_kg = (vra_liquid_liters * equipment.chemical_concentration_g_l) / 1000.0
    chem_saved_kg = max(0.0, uniform_chem_kg - vra_chem_kg)
    chem_saved_pct = (chem_saved_kg / max(0.001, uniform_chem_kg)) * 100.0
    cost_savings = chem_saved_kg * equipment.chemical_price_per_kg_mdl

    report = VRAOptimizationReportDTO(
        treated_area_ha=round(total_area_ha, 3),
        uniform_liquid_total_liters=round(uniform_liquid_liters, 1),
        vra_liquid_total_liters=round(vra_liquid_liters, 1),
        liquid_saved_liters=round(liquid_saved, 1),
        liquid_saved_pct=round(liquid_saved_pct, 1),
        uniform_chemical_total_kg=round(uniform_chem_kg, 2),
        vra_chemical_total_kg=round(vra_chem_kg, 2),
        chemical_saved_kg=round(chem_saved_kg, 2),
        chemical_saved_pct=round(chem_saved_pct, 1),
        cost_savings_mdl=round(cost_savings, 2),
    )

    geojson_collection = {
        "type": "FeatureCollection",
        "features": features,
    }

    return VRAPrescriptionResponse(
        prescription_zones=zones,
        summary=report,
        geojson_feature_collection=geojson_collection,
    )


def export_task_data_isoxml(request: IsoXmlExportRequest) -> str:
    """Генерирует XML-документ стандарта ISO 11783-10 (TaskData).

    Поддерживается бортовыми терминалами опрыскивателей с шиной ISOBUS (ISO-XML).
    """
    root = ET.Element(
        "ISO11783_TaskData",
        {
            "VersionMajor": "4",
            "VersionMinor": "3",
            "ManagementSoftwareManufacturer": "VitiGuard MD",
            "ManagementSoftwareVersion": "2.4.0",
            "DataTransferOrigin": "1",
        },
    )

    # 1. Заказчик и поле (Partfield)
    pfd = ET.SubElement(
        root,
        "PFD",
        {
            "A": "PFD1",
            "C": request.field_name,
            "D": f"{len(request.sectors)}",
        },
    )

    # 2. Устройство (Device / Sprayer)
    dvc = ET.SubElement(
        root,
        "DVC",
        {
            "A": "DVC1",
            "B": request.equipment.sprayer_id,
        },
    )
    # Ширина штанги / междурядье (Device Element)
    ET.SubElement(
        dvc,
        "DET",
        {
            "A": "DET1",
            "B": "1",
            "C": f"{int(request.equipment.boom_width_m * 1000)}",  # в мм
        },
    )

    # 3. Технологическая задача (Task)
    tsk = ET.SubElement(
        root,
        "TSK",
        {
            "A": "TSK1",
            "B": request.task_name,
            "E": pfd.get("A", "PFD1"),
            "G": "1",  # Статус: запланирована (Planned)
        },
    )

    # 4. Сетка предписания норм внесения (Grid / Treatment Zones)
    treatment = generate_vra_prescription(request.equipment, request.sectors)

    # Карты зон обработки (PZN: Prescription Zone)
    for idx, zone in enumerate(treatment.prescription_zones, start=1):
        pzn = ET.SubElement(
            tsk,
            "PZN",
            {
                "A": f"{idx}",
                "B": f"Zone_{zone.sector_id}",
            },
        )
        # PDV: Process Data Variable — DDI 0x0001 (Setpoint Application Rate L/m2)
        # 1 л/га = 0.0001 л/м² = 1e-4
        l_per_m2 = zone.target_rate_l_ha / 10000.0
        ET.SubElement(
            pzn,
            "PDV",
            {
                "A": "0001",  # DDI: Volume Per Area
                "B": f"{int(l_per_m2 * 1000000)}",  # микролитры/м² для точного представления
            },
        )

    raw_bytes: bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return raw_bytes.decode("utf-8")
