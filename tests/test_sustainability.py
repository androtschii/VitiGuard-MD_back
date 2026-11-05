"""Тесты сервиса углеродного баланса OIV и экологического аудита виноградника (sprint-31)."""

import uuid
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from app.api.deps import get_current_user
from app.main import create_app
from app.models.user import User
from app.schemas.sustainability import (
    CarbonBalanceRequest,
    OrganicAuditRequest,
    VineyardPracticesDTO,
)
from app.services.sustainability import (
    audit_organic_practices,
    calculate_carbon_breakdown,
    calculate_n2o_fertilizer_emissions,
    calculate_woody_sequestration,
    compute_carbon_balance,
)


def _mock_agronomist() -> User:
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.email = "agronomist@vitiguard.md"
    user.role = "agronomist"
    user.is_active = True
    return user


def test_calculate_woody_sequestration() -> None:
    """Проверка расчёта депонирования углерода в многолетней древесине лозы."""
    # 3333 кустов/га, зрелая лоза 10 лет -> ~2.05 т CO2e/га/год
    seq = calculate_woody_sequestration(vines_per_ha=3333, vine_age_years=10)
    assert 1.90 <= seq <= 2.20

    # Молодая лоза 2 года (пониженный возрастной коэффициент)
    seq_young = calculate_woody_sequestration(vines_per_ha=3333, vine_age_years=2)
    assert seq_young < seq


def test_calculate_n2o_fertilizer_emissions() -> None:
    """Проверка выбросов закиси азота по методике IPCC Tier 1."""
    # 50 кг N/га -> 50 * 0.01 * (44/28) * 273 / 1000 = 0.214 т CO2e/га
    emissions = calculate_n2o_fertilizer_emissions(50.0)
    assert emissions == pytest.approx(0.214, abs=1e-2)


def test_calculate_carbon_breakdown_sink_and_credits() -> None:
    """Регенеративный виноградник с травами, мульчированием и электророботами является нетто-поглотителем CO2."""
    practices = VineyardPracticesDTO(
        area_ha=10.0,
        vines_per_ha=3500,
        vine_age_years=12,
        inter_row_cover_crop="permanent_grass",  # +0.95 т
        pruning_residue_management="mulched_in_situ",  # +0.55 т
        diesel_tractor_liters_per_ha=80.0,
        electric_robot_operations=True,  # сокращает солярку на 65%
        nitrogen_fertilizer_kg_n_ha=20.0,
    )

    req = CarbonBalanceRequest(
        vineyard_id="v_regenerative",
        practices=practices,
        carbon_credit_price_eur_ton=40.0,
    )
    res = compute_carbon_balance(req)

    assert res.is_carbon_sink is True
    assert (
        res.breakdown_per_ha.total_sequestration_tco2e_ha
        > res.breakdown_per_ha.total_emissions_tco2e_ha
    )
    assert res.breakdown_per_ha.net_carbon_balance_tco2e_ha > 2.0  # сильный нетто-сток
    assert res.total_farm_net_tco2e_year > 20.0  # на 10 га суммарно > 20 т CO2e
    assert res.potential_carbon_credits_eur > 800.0  # 20т * 40€ = 800+ €


def test_calculate_carbon_breakdown_diesel_polluter() -> None:
    """Традиционный виноградник с черным паром, сжиганием лозы и высоким расходом солярки."""
    practices = VineyardPracticesDTO(
        area_ha=5.0,
        vines_per_ha=2500,
        vine_age_years=4,
        inter_row_cover_crop="bare_soil",  # 0.0 т поглощения
        pruning_residue_management="burned",  # -0.2 т штраф
        diesel_tractor_liters_per_ha=260.0,  # ~0.70 т эмиссий солярки
        electric_robot_operations=False,
        nitrogen_fertilizer_kg_n_ha=120.0,  # ~0.51 т эмиссий N2O
    )

    breakdown = calculate_carbon_breakdown(practices)
    assert breakdown.total_emissions_tco2e_ha > breakdown.total_sequestration_tco2e_ha
    assert breakdown.net_carbon_balance_tco2e_ha < 0.0


def test_audit_organic_practices() -> None:
    """Проверка экологического аудита на соответствие нормам ЕС."""
    # 1. Образцовый органический виноградник
    compliant_practices = VineyardPracticesDTO(
        area_ha=5.0,
        copper_applied_kg_ha=2.5,  # норма < 4.0
        synthetic_herbicides_used=False,
        inter_row_cover_crop="legumes",
        pruning_residue_management="mulched_in_situ",
    )
    audit_ok = audit_organic_practices(
        OrganicAuditRequest(vineyard_id="v-org", practices=compliant_practices)
    )
    assert audit_ok.is_organic_certified_ready is True
    assert audit_ok.compliance_score_pct >= 90.0
    assert audit_ok.copper_limit_status == "ok"
    assert len(audit_ok.deficiencies) == 0

    # 2. Нарушение органического регламента (превышение меди 4.8 кг и глифосат)
    violating_practices = VineyardPracticesDTO(
        area_ha=5.0,
        copper_applied_kg_ha=4.8,  # превышение норматива
        synthetic_herbicides_used=True,  # запрещено
        inter_row_cover_crop="bare_soil",
        pruning_residue_management="burned",
    )
    audit_fail = audit_organic_practices(
        OrganicAuditRequest(vineyard_id="v-fail", practices=violating_practices)
    )
    assert audit_fail.is_organic_certified_ready is False
    assert audit_fail.copper_limit_status == "exceeded"
    assert audit_fail.compliance_score_pct < 50.0
    assert len(audit_fail.deficiencies) >= 3


def test_sustainability_api_endpoints() -> None:
    """Тест эндпоинтов API /api/v1/sustainability."""
    app = create_app()
    user = _mock_agronomist()
    app.dependency_overrides[get_current_user] = lambda: user

    client = TestClient(app)

    practices_payload = {
        "area_ha": 8.0,
        "vines_per_ha": 3333,
        "vine_age_years": 8,
        "inter_row_cover_crop": "permanent_grass",
        "pruning_residue_management": "mulched_in_situ",
        "diesel_tractor_liters_per_ha": 90.0,
        "electric_robot_operations": True,
        "nitrogen_fertilizer_kg_n_ha": 25.0,
        "copper_applied_kg_ha": 2.8,
        "synthetic_herbicides_used": False,
    }

    # 1. Расчёт углеродного баланса
    res_carbon = client.post(
        "/api/v1/sustainability/carbon-balance",
        json={
            "vineyard_id": "cricova_plot_12",
            "practices": practices_payload,
            "carbon_credit_price_eur_ton": 35.0,
        },
    )
    assert res_carbon.status_code == 200, res_carbon.text
    carbon_data = res_carbon.json()
    assert carbon_data["is_carbon_sink"] is True
    assert "breakdown_per_ha" in carbon_data
    assert carbon_data["electric_robot_diesel_reduction_pct"] == 65.0

    # 2. Органический аудит
    res_audit = client.post(
        "/api/v1/sustainability/organic-audit",
        json={
            "vineyard_id": "cricova_plot_12",
            "practices": practices_payload,
        },
    )
    assert res_audit.status_code == 200, res_audit.text
    audit_data = res_audit.json()
    assert audit_data["is_organic_certified_ready"] is True
    assert audit_data["copper_limit_status"] == "ok"

    # 3. Региональные бенчмарки
    res_bench = client.get("/api/v1/sustainability/benchmarks")
    assert res_bench.status_code == 200
    bench_data = res_bench.json()
    assert len(bench_data) >= 3
    regions = [b["region_name"] for b in bench_data]
    assert any("Пуркарь" in r for r in regions)
