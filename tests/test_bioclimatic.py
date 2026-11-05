"""Тесты биоклиматических индексов, фенологии BBCH и окна сбора урожая (sprint-25)."""

import uuid
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock

from starlette.testclient import TestClient

from app.api.deps import get_current_user
from app.main import create_app
from app.models.user import User
from app.models.variety import BerryColor, Variety
from app.models.vineyard import Vineyard
from app.repositories.vineyard import VineyardView
from app.services.agromodels import DailyWeather
from app.services.bioclimatic import (
    VARIETY_CATALOG,
    BBCHStage,
    compute_cool_night_index,
    compute_huglin_index,
    compute_selyaninov_htc,
    compute_winkler_index,
    estimate_phenology_stage,
    get_variety_profile,
    predict_harvest_window,
)


def _make_daily(
    day: date,
    mean_temp: float,
    min_temp: float = 12.0,
    max_temp: float = 26.0,
    precipitation_mm: float = 0.0,
) -> DailyWeather:
    return DailyWeather(
        day=day,
        mean_temp=mean_temp,
        min_temp=min_temp,
        max_temp=max_temp,
        precipitation_mm=precipitation_mm,
        mean_humidity=65.0,
        longest_mild_run=8,
        hours_above_35=0,
    )


def test_variety_catalog_coverage() -> None:
    expected_varieties = {
        "feteasca_regala",
        "feteasca_alba",
        "feteasca_neagra",
        "rara_neagra",
        "viorica",
        "chardonnay",
        "sauvignon_blanc",
        "cabernet_sauvignon",
        "merlot",
        "pinot_noir",
    }
    assert expected_varieties.issubset(VARIETY_CATALOG.keys())

    # Проверка нормализации названий и диакритики
    profile1 = get_variety_profile("Fetească Neagră")
    assert profile1.code == "feteasca_neagra"
    assert profile1.berry_color == "red"

    profile2 = get_variety_profile("rara-neagra")
    assert profile2.code == "rara_neagra"

    profile_unknown = get_variety_profile("Unknown Grape")
    assert profile_unknown.code == "feteasca_regala"


def test_estimate_phenology_stage_lifecycle() -> None:
    # 1. Покой лозы
    p0 = estimate_phenology_stage(50.0, "feteasca_regala")
    assert p0.current_bbch == int(BBCHStage.DORMANCY)
    assert p0.next_bbch == int(BBCHStage.BUDBURST)
    assert p0.gdd_to_next == 60.0
    assert not p0.is_season_complete

    # 2. Распускание почек
    p1 = estimate_phenology_stage(110.0, "feteasca_regala")
    assert p1.current_bbch == int(BBCHStage.BUDBURST)
    assert p1.next_bbch == int(BBCHStage.SHOOT_GROWTH)
    assert p1.progress_percentage == 0.0

    # 3. Полное цветение
    p2 = estimate_phenology_stage(420.0, "feteasca_regala")
    assert p2.current_bbch == int(BBCHStage.FLOWERING)
    assert p2.next_bbch == int(BBCHStage.FRUIT_SET)

    # 4. Веaison (начало созревания)
    p3 = estimate_phenology_stage(1065.0, "feteasca_regala")
    assert p3.current_bbch == int(BBCHStage.VERAISON)
    assert p3.next_bbch == int(BBCHStage.BERRY_SOFTENING)
    assert 40.0 < p3.progress_percentage < 60.0

    # 5. Полная техническая зрелость
    p4 = estimate_phenology_stage(1400.0, "feteasca_regala")
    assert p4.current_bbch == int(BBCHStage.HARVEST_MATURITY)
    assert p4.next_bbch is None
    assert p4.is_season_complete
    assert p4.progress_percentage == 100.0


def test_compute_winkler_index() -> None:
    # 200 дней в вегетационный период с mean_temp = 18.0 °C -> (18 - 10) * 200 = 1600 GDD (Region II)
    start = date(2026, 4, 1)
    days = [_make_daily(start + timedelta(days=i), mean_temp=18.0) for i in range(200)]
    winkler = compute_winkler_index(days)
    assert winkler.gdd_sum == 1600.0
    assert winkler.region_class == "Region II"

    # Жаркий климат (mean_temp = 20.0) -> (20 - 10) * 200 = 2000 GDD (Region IV)
    hot_days = [
        _make_daily(start + timedelta(days=i), mean_temp=20.0) for i in range(200)
    ]
    winkler_hot = compute_winkler_index(hot_days)
    assert winkler_hot.gdd_sum == 2000.0
    assert winkler_hot.region_class == "Region IV"


def test_compute_huglin_index() -> None:
    # 100 дней c T_mean=20, T_max=28 -> ((10 + 18)/2) * 1.04 = 14 * 1.04 = 14.56 * 100 = 1456 (HI-1)
    start = date(2026, 5, 1)
    days = [
        _make_daily(start + timedelta(days=i), mean_temp=20.0, max_temp=28.0)
        for i in range(100)
    ]
    huglin = compute_huglin_index(days, latitude=47.0)
    assert 1450.0 <= huglin.hi_value <= 1460.0
    assert "HI-1" in huglin.thermal_class


def test_compute_cool_night_index() -> None:
    # 30 дней сентября с min_temp 11.5 °C
    start = date(2026, 9, 1)
    days = [
        _make_daily(start + timedelta(days=i), mean_temp=18.0, min_temp=11.5)
        for i in range(30)
    ]
    ci = compute_cool_night_index(days)
    assert ci.ci_temperature == 11.5
    assert "CI+2" in ci.night_class
    assert "антоцианов" in ci.aroma_color_potential

    # Без сентября
    empty_ci = compute_cool_night_index([])
    assert empty_ci.ci_temperature == 0.0
    assert "Нет данных" in empty_ci.night_class


def test_compute_selyaninov_htc() -> None:
    # 100 активных дней с T_mean = 20.0 (°C) -> sum(T) = 2000
    # 1. Засуха: осадков всего 60 мм -> HTC = 60 / (0.1 * 2000) = 0.30 (очень засушливая)
    start = date(2026, 5, 1)
    dry_days = [
        _make_daily(start + timedelta(days=i), mean_temp=20.0, precipitation_mm=0.6)
        for i in range(100)
    ]
    htc_dry = compute_selyaninov_htc(dry_days)
    assert htc_dry.htc_value == 0.3
    assert "Очень засушливая" in htc_dry.moisture_class

    # 2. Умеренно влажная зона: осадков 220 мм -> HTC = 220 / 200 = 1.10 (достаточное увлажнение)
    wet_days = [
        _make_daily(start + timedelta(days=i), mean_temp=20.0, precipitation_mm=2.2)
        for i in range(100)
    ]
    htc_wet = compute_selyaninov_htc(wet_days)
    assert htc_wet.htc_value == 1.1
    assert "достаточного увлажнения" in htc_wet.moisture_class


def test_predict_harvest_window_favorable() -> None:
    # Благоприятный прогноз погоды на 14 дней
    start = date(2026, 9, 10)
    forecast = [
        _make_daily(
            start + timedelta(days=i),
            mean_temp=22.0,
            max_temp=26.0,
            precipitation_mm=0.0,
        )
        for i in range(14)
    ]
    # Текущий сахар 19.5, цель 20.5 (Feteasca Regala)
    pred = predict_harvest_window(
        accumulated_gdd=1250.0,
        forecast_days=forecast,
        variety="feteasca_regala",
        current_sugar_brix=19.5,
        target_sugar_brix=20.5,
    )
    assert pred.optimal_window_start is not None
    assert pred.optimal_window_end is not None
    assert pred.harvest_urgency in ("low", "medium")
    assert len(pred.predicted_brix_trajectory) == 14
    assert pred.predicted_brix_trajectory[-1]["predicted_brix"] >= 20.5


def test_predict_harvest_window_rain_and_frost_warnings() -> None:
    start = date(2026, 9, 10)
    # День 3: ливень 25 мм
    # День 8: заморозок -1.5 °C
    forecast = []
    for i in range(14):
        d = start + timedelta(days=i)
        precip = 25.0 if i == 3 else 0.0
        t_min = -1.5 if i == 8 else 10.0
        forecast.append(
            _make_daily(
                d,
                mean_temp=16.0,
                min_temp=t_min,
                max_temp=20.0,
                precipitation_mm=precip,
            )
        )

    pred = predict_harvest_window(
        accumulated_gdd=1300.0,
        forecast_days=forecast,
        variety="feteasca_neagra",
        current_sugar_brix=23.0,
        target_sugar_brix=23.5,
    )
    assert pred.harvest_urgency == "critical"
    assert any("заморозка" in r for r in pred.risks_detected)
    assert any("Обильные осадки" in r for r in pred.risks_detected)


def test_bioclimatic_api_endpoints() -> None:
    from collections.abc import AsyncIterator
    from unittest.mock import patch

    from sqlalchemy.ext.asyncio import AsyncSession

    from app.api.deps import get_session

    app = create_app()
    fake_user_id = uuid.uuid4()
    mock_user = MagicMock(spec=User)
    mock_user.id = fake_user_id
    mock_user.email = "viticulturist@vitiguard.md"
    app.dependency_overrides[get_current_user] = lambda: mock_user

    async def override_get_session() -> AsyncIterator[AsyncMock]:
        yield AsyncMock(spec=AsyncSession)

    app.dependency_overrides[get_session] = override_get_session

    client = TestClient(app)

    # 1. Каталог сортов
    resp = client.get("/api/v1/bioclimatic/varieties")
    assert resp.status_code == 200
    varieties = resp.json()
    assert len(varieties) >= 10
    codes = [v["code"] for v in varieties]
    assert "feteasca_regala" in codes
    assert "cabernet_sauvignon" in codes

    # 2. Моки сессии и репозитория виноградника
    fake_vineyard_id = uuid.uuid4()
    mock_variety = MagicMock(spec=Variety)
    mock_variety.name = "Feteasca Regala"
    mock_variety.berry_color = BerryColor.WHITE

    mock_vineyard = MagicMock(spec=Vineyard)
    mock_vineyard.id = fake_vineyard_id
    mock_vineyard.variety = mock_variety

    mock_view = VineyardView(
        vineyard=mock_vineyard,
        geometry={"type": "Polygon", "coordinates": []},
        area_ha=5.2,
    )

    mock_days = [
        _make_daily(date(2026, 4, 1) + timedelta(days=i), mean_temp=20.0, min_temp=12.0)
        for i in range(120)
    ]

    with (
        patch(
            "app.repositories.vineyard.VineyardRepository.get_for_owner",
            new_callable=AsyncMock,
        ) as mock_get_owner,
        patch(
            "app.api.routes.bioclimatic._load_vineyard_daily_weather",
            new_callable=AsyncMock,
        ) as mock_load_weather,
    ):
        mock_get_owner.return_value = mock_view
        mock_load_weather.return_value = mock_days

        # GET phenology-estimate
        r_pheno = client.get(
            f"/api/v1/vineyards/{fake_vineyard_id}/bioclimatic/phenology-estimate"
        )
        assert r_pheno.status_code == 200
        pheno_data = r_pheno.json()
        assert "current_bbch" in pheno_data
        assert pheno_data["accumulated_gdd"] > 0
        assert pheno_data["variety_name"] == "Fetească Regală"

        # GET indices
        r_indices = client.get(
            f"/api/v1/vineyards/{fake_vineyard_id}/bioclimatic/indices"
        )
        assert r_indices.status_code == 200
        indices_data = r_indices.json()
        assert "winkler" in indices_data
        assert "huglin" in indices_data
        assert "cool_night" in indices_data
        assert "selyaninov" in indices_data

        # POST harvest-window
        r_harvest = client.post(
            f"/api/v1/vineyards/{fake_vineyard_id}/bioclimatic/harvest-window",
            json={"target_sugar_brix": 21.0, "current_sugar_brix": 19.0},
        )
        assert r_harvest.status_code == 200
        harvest_data = r_harvest.json()
        assert "predicted_brix_trajectory" in harvest_data
        assert "agronomic_recommendation" in harvest_data
        assert harvest_data["target_sugar_brix"] == 21.0
