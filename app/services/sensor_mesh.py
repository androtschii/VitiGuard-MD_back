"""Сервис беспроводной микроклиматической сети (LoRaWAN) и предупреждения заморозков/града (sprint-29).

Реализует:
1. Пространственную интерполяцию микроклимата виноградника с учётом гипсометрии и экспозиции склонов (IDW + гипсометрический градиент).
2. Выявление «морозных карманов» (frost pockets) — понижений рельефа со скоплением холодного воздуха.
3. Расчёт температуры точки росы и температуры смоченного термометра (wet-bulb temperature) листа винограда.
4. Детектор температурной инверсии и рекомендации по защите почек (ветровые машины, дождевание, задымление).
5. Прогноз конвективного градового риска по скорости барометрического спада.
"""

import math

from app.schemas.sensor_mesh import (
    FrostRiskReportDTO,
    HailStormRiskReportDTO,
    SensorTelemetryDTO,
    SpatialPointDTO,
    VineyardMeshStatusResponse,
)


def calculate_dew_point(temp_c: float, humidity_pct: float) -> float:
    """Точка росы по формуле Магнуса-Тетенса (°C)."""
    rh_clamped = max(1.0, min(100.0, humidity_pct))
    a = 17.27
    b = 237.7
    gamma = (a * temp_c) / (b + temp_c) + math.log(rh_clamped / 100.0)
    dew_point = (b * gamma) / (a - gamma)
    return round(dew_point, 2)


def calculate_wet_bulb_temp(temp_c: float, humidity_pct: float) -> float:
    """Температура смоченного термометра (wet-bulb temperature) по формуле R. Stull (°C).

    Определяет температуру испаряющейся влаги с поверхности листа.
    """
    rh = max(1.0, min(100.0, humidity_pct))
    t = temp_c

    tw = (
        t * math.atan(0.151977 * math.sqrt(rh + 8.313659))
        + math.atan(t + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * (rh**1.5) * math.atan(0.023101 * rh)
        - 4.686035
    )
    return float(round(tw, 2))


def interpolate_spatial_point(
    target_lat: float,
    target_lon: float,
    target_elevation_m: float,
    sensors: list[SensorTelemetryDTO],
    lapse_rate_c_per_m: float = 0.0065,
) -> tuple[float, float, bool]:
    """Интерполяция температуры и влажности методом обратно взвешенных расстояний (IDW) с высотной поправкой.

    Возвращает: (interpolated_temp_c, interpolated_humidity_pct, is_frost_pocket)
    """
    if not sensors:
        return (15.0, 60.0, False)

    weights: list[float] = []
    temps_adjusted: list[float] = []
    humidities: list[float] = []

    for s in sensors:
        # Расстояние в градусах
        dist = math.hypot(target_lat - s.lat, target_lon - s.lon)
        dist = max(1e-5, dist)
        weight = 1.0 / (dist**2)
        weights.append(weight)

        # Поправка на высоту (высотный градиент): в низинах ночью может возникать инверсия
        dz = target_elevation_m - s.elevation_m
        temp_adj = s.canopy_temperature_c - (dz * lapse_rate_c_per_m)
        temps_adjusted.append(temp_adj)
        humidities.append(s.canopy_humidity_pct)

    sum_weights = sum(weights)
    interp_temp = (
        sum(t * w for t, w in zip(temps_adjusted, weights, strict=False)) / sum_weights
    )
    interp_rh = (
        sum(h * w for h, w in zip(humidities, weights, strict=False)) / sum_weights
    )

    # Проверка на «морозный карман»: точка ниже среднего уровня сенсоров, холоднее и влажнее
    avg_elevation = sum(s.elevation_m for s in sensors) / len(sensors)
    is_frost_pocket = (target_elevation_m < avg_elevation - 15.0) and (interp_rh > 85.0)

    # Если это морозный карман в ясную тихую ночь, температура дополнительно проседает на 1.0..2.0 °C
    if is_frost_pocket:
        interp_temp -= 1.2

    return (
        round(interp_temp, 2),
        round(max(0.0, min(100.0, interp_rh)), 1),
        is_frost_pocket,
    )


def build_mesh_status(
    vineyard_id: str,
    sensors: list[SensorTelemetryDTO],
) -> VineyardMeshStatusResponse:
    """Формирует текущий сводный отчёт сенсорной сети и интерполированную сетку."""
    if not sensors:
        return VineyardMeshStatusResponse(
            vineyard_id=vineyard_id,
            active_sensors_count=0,
            sensors=[],
            mean_canopy_temp_c=0.0,
            mean_canopy_humidity_pct=0.0,
            mean_leaf_wetness_pct=0.0,
            spatial_grid=[],
        )

    mean_t = sum(s.canopy_temperature_c for s in sensors) / len(sensors)
    mean_rh = sum(s.canopy_humidity_pct for s in sensors) / len(sensors)
    mean_lws = sum(s.leaf_wetness_pct for s in sensors) / len(sensors)

    # Строим компактную регулярную сетку 3x3 вокруг датчиков
    lats = [s.lat for s in sensors]
    lons = [s.lon for s in sensors]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    avg_elev = sum(s.elevation_m for s in sensors) / len(sensors)

    grid: list[SpatialPointDTO] = []
    lat_steps = [min_lat, (min_lat + max_lat) / 2.0, max_lat]
    lon_steps = [min_lon, (min_lon + max_lon) / 2.0, max_lon]

    for plat in lat_steps:
        for plon in lon_steps:
            t_int, rh_int, frost_pocket = interpolate_spatial_point(
                target_lat=plat,
                target_lon=plon,
                target_elevation_m=avg_elev,
                sensors=sensors,
            )
            grid.append(
                SpatialPointDTO(
                    lat=round(plat, 5),
                    lon=round(plon, 5),
                    elevation_m=round(avg_elev, 1),
                    interpolated_temp_c=t_int,
                    interpolated_humidity_pct=rh_int,
                    frost_pocket_risk=frost_pocket,
                )
            )

    return VineyardMeshStatusResponse(
        vineyard_id=vineyard_id,
        active_sensors_count=len(sensors),
        sensors=sensors,
        mean_canopy_temp_c=round(mean_t, 2),
        mean_canopy_humidity_pct=round(mean_rh, 1),
        mean_leaf_wetness_pct=round(mean_lws, 1),
        spatial_grid=grid,
    )


def evaluate_frost_risk(
    vineyard_id: str,
    sensors: list[SensorTelemetryDTO],
    phenology_stage: str = "BBCH 09 (Распускание почек)",
) -> FrostRiskReportDTO:
    """Анализирует угрозу весенних заморозков для зеленых почек и побегов винограда."""
    if not sensors:
        return FrostRiskReportDTO(
            vineyard_id=vineyard_id,
            risk_level="none",
            min_canopy_temp_c=10.0,
            wet_bulb_temp_c=10.0,
            has_thermal_inversion=False,
            inversion_delta_c=0.0,
            critical_phenology_stage=phenology_stage,
            frost_pocket_sensors=[],
            recommended_mitigation=[],
            estimated_window_hours=0.0,
        )

    min_sensor = min(sensors, key=lambda s: s.canopy_temperature_c)
    max_elev_sensor = max(sensors, key=lambda s: s.elevation_m)
    min_elev_sensor = min(sensors, key=lambda s: s.elevation_m)

    min_t = min_sensor.canopy_temperature_c
    tw = calculate_wet_bulb_temp(min_t, min_sensor.canopy_humidity_pct)

    # Проверка термической инверсии (на возвышенности теплее, чем в низине)
    inversion_delta = (
        max_elev_sensor.canopy_temperature_c - min_elev_sensor.canopy_temperature_c
    )
    has_inversion = inversion_delta >= 1.5

    # Детекция морозных карманов
    pockets: list[str] = [
        s.sensor_id
        for s in sensors
        if s.elevation_m <= min_elev_sensor.elevation_m + 5.0
        and s.canopy_temperature_c <= 1.0
    ]

    # Градация риска заморозков
    # Критический порог для набухших/распустившихся почек (BBCH 05-09): Tw <= -0.5 °C
    mitigations: list[str] = []
    if tw <= -2.5:
        risk_level = "critical"
        est_hours = 5.0
    elif tw <= -0.5:
        risk_level = "moderate"
        est_hours = 3.5
    elif tw <= 1.5:
        risk_level = "low"
        est_hours = 1.0
    else:
        risk_level = "none"
        est_hours = 0.0

    if risk_level in ("moderate", "critical"):
        if has_inversion:
            mitigations.append(
                f"Запустить ветровые машины (wind machines): сильная инверсия (+{inversion_delta:.1f} °C) "
                "позволит перемешать тёплый верхний слой воздуха с приземным."
            )
        else:
            mitigations.append(
                "Активировать противозаморозковое дождевание (sprinklers): тепло фазового перехода воды в лёд "
                "сохранит температуру почек на уровне 0 °C."
            )
            mitigations.append(
                "Использовать стационарные обогреватели / дымовые шашки в выявленных морозных карманах."
            )

    return FrostRiskReportDTO(
        vineyard_id=vineyard_id,
        risk_level=risk_level,
        min_canopy_temp_c=round(min_t, 2),
        wet_bulb_temp_c=round(tw, 2),
        has_thermal_inversion=has_inversion,
        inversion_delta_c=round(max(0.0, inversion_delta), 2),
        critical_phenology_stage=phenology_stage,
        frost_pocket_sensors=pockets,
        recommended_mitigation=mitigations,
        estimated_window_hours=est_hours,
    )


def evaluate_hail_risk(
    vineyard_id: str,
    telemetry: SensorTelemetryDTO,
    pressure_trend_1h_hpa: float,
) -> HailStormRiskReportDTO:
    """Оценивает риск конвективного градоопасного шторма по барометрическому градиенту и теплу кроны."""
    # Быстрый спад давления (> 2.0 гПа/ч) в сочетании с высокой температурой (> 25 °C) и влажностью (> 65 %)
    t = telemetry.canopy_temperature_c
    rh = telemetry.canopy_humidity_pct
    drop = abs(min(0.0, pressure_trend_1h_hpa))

    convective_index = (t * 0.4) + (rh * 0.3) + (drop * 10.0)

    if drop >= 3.0 and t >= 26.0 and rh >= 70.0:
        level = "severe"
        msg = "Штормовое предупреждение: взрывной конвективный фронт с вероятностью крупного града > 80%."
        netting = True
    elif drop >= 2.0 and t >= 23.0:
        level = "high"
        msg = "Высокая вероятность шквала и локального градобития. Закройте защитные сетки."
        netting = True
    elif drop >= 1.2:
        level = "moderate"
        msg = "Умеренная грозовая активность. Рекомендуется мониторинг радарных данных."
        netting = False
    else:
        level = "low"
        msg = "Атмосферное давление стабильно. Риск града минимален."
        netting = False

    return HailStormRiskReportDTO(
        vineyard_id=vineyard_id,
        hail_risk_level=level,
        pressure_drop_rate_hpa_h=round(drop, 2),
        convective_energy_index=round(convective_index, 1),
        warning_message=msg,
        netting_action_required=netting,
    )
