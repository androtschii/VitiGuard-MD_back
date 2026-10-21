from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.models.enums import Disease, Fungicide, RiskLevel
from app.models.treatment import Treatment
from app.services.anomaly import Anomaly
from app.services.report import (
    IndexPoint,
    LeafCheck,
    ReportData,
    RiskDay,
    render_report,
)
from app.services.risk import local_today
from app.services.treatments import Action, Plan, Recommendation

RunSql = Callable[[str], list[tuple[object, ...]]]
TODAY = date(2026, 6, 10)


def full_report() -> ReportData:
    return ReportData(
        name="Криково, северный склон",
        area_ha=13.6,
        variety="Фетяска албэ",
        planting_year=2015,
        today=TODAY,
        created_at=datetime(2026, 6, 10, 9, 30, tzinfo=UTC),
        risk=[
            RiskDay(TODAY, RiskLevel.HIGH, RiskLevel.MEDIUM),
            RiskDay(TODAY + timedelta(days=1), RiskLevel.LOW, None),
        ],
        plan=Plan(
            today=TODAY,
            protected_until={Disease.DOWNY_MILDEW: TODAY, Disease.POWDERY_MILDEW: None},
            recommendations=[
                Recommendation(
                    Action.PROTECT,
                    Disease.POWDERY_MILDEW,
                    Fungicide.SULFUR,
                    TODAY,
                    TODAY,
                    "powdery_high",
                ),
                Recommendation(
                    Action.INSPECT,
                    Disease.DOWNY_MILDEW,
                    None,
                    TODAY,
                    TODAY,
                    "downy_symptoms",
                ),
            ],
        ),
        treatments=[
            Treatment(
                applied_on=TODAY, fungicide=Fungicide.CONTACT, product="Купроксат"
            ),
            Treatment(applied_on=TODAY, fungicide=Fungicide.SULFUR, product=None),
        ],
        indices=[
            IndexPoint(date(2026, 5, 1), 0.52, 0.31, 0),
            IndexPoint(date(2026, 5, 6), None, None, 100),
            IndexPoint(date(2026, 5, 11), 0.61, 0.35, None),
        ],
        anomalies=[
            Anomaly(
                datetime(2026, 5, 11, tzinfo=UTC), 0.4, 0.2, None, 3.1, "suspected"
            ),
            Anomaly(
                datetime(2026, 5, 16, tzinfo=UTC), 0.4, 0.2, 0.19, 3.3, "confirmed"
            ),
        ],
        checks=[
            LeafCheck(TODAY, "милдью", 0.91, "милдью"),
            LeafCheck(TODAY, "—", None, None),
        ],
    )


def empty_report() -> ReportData:
    return ReportData(
        name="Новый участок",
        area_ha=1.0,
        variety=None,
        planting_year=None,
        today=TODAY,
        created_at=datetime(2026, 6, 10, 9, 30, tzinfo=UTC),
        risk=[RiskDay(TODAY, None, None)],
        plan=Plan(
            TODAY, {Disease.DOWNY_MILDEW: None, Disease.POWDERY_MILDEW: None}, []
        ),
        treatments=[],
        indices=[],
        anomalies=[],
        checks=[],
    )


@pytest.mark.parametrize("data", [full_report(), empty_report()], ids=["full", "empty"])
def test_report_is_a_pdf_with_cyrillic_font(data: ReportData) -> None:
    pdf = render_report(data)

    assert pdf.startswith(b"%PDF-")
    assert pdf.rstrip().endswith(b"%%EOF")
    # Шрифт с кириллицей встроен: без него русский текст был бы квадратиками
    assert b"DejaVuSans" in pdf


def login(client: TestClient, email: str = "grower@example.md") -> dict[str, str]:
    credentials = {"email": email, "password": "secret123"}
    client.post("/api/v1/auth/register", json={**credentials, "full_name": "Ион"})
    token = client.post("/api/v1/auth/login", json=credentials).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.integration
def test_report_endpoint_collects_vineyard_state(
    api_client: TestClient, run_sql: RunSql, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.api.routes import reports

    collected: list[ReportData] = []

    def capture(data: ReportData) -> bytes:
        collected.append(data)
        return render_report(data)

    monkeypatch.setattr(reports, "render_report", capture)
    headers = login(api_client)
    ring = [[28.86, 47.14], [28.862, 47.14], [28.862, 47.141], [28.86, 47.14]]
    vineyard = api_client.post(
        "/api/v1/vineyards",
        headers=headers,
        json={
            "name": "Криково",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "planting_year": 2015,
        },
    ).json()["id"]
    today = local_today()
    owner = api_client.get("/api/v1/users/me", headers=headers).json()["id"]
    run_sql(
        "INSERT INTO disease_risks (vineyard_id, disease, target_date, method, score, "
        f"level) VALUES ('{vineyard}', 'downy_mildew', '{today}', 'three_tens', 1, "
        f"'high'), ('{vineyard}', 'downy_mildew', '{today}', 'goidanich', 0, 'low')"
    )
    run_sql(
        "WITH scan AS (INSERT INTO satellite_scans (vineyard_id, product_id, "
        "acquired_at, cloud_cover, parcel_cloud_cover, status) VALUES "
        f"('{vineyard}', 'S2_report', '{today - timedelta(days=3)}T09:17:00+00', 0, 0, "
        "'completed') RETURNING id) INSERT INTO vegetation_indices (scan_id, "
        "index_type, mean_value, min_value, max_value, std_value) "
        "SELECT id, 'ndvi', 0.7, 0.6, 0.8, 0.05 FROM scan"
    )
    run_sql(
        "WITH image AS (INSERT INTO leaf_images (owner_id, vineyard_id, storage_key, "
        f"content_type, size_bytes, width, height) VALUES ('{owner}', '{vineyard}', "
        "'leaf/report.jpg', 'image/jpeg', 10, 224, 224) RETURNING id) "
        "INSERT INTO disease_results (image_id, model_name, model_version, status, "
        "predicted_class, confidence) SELECT id, 'efficientnet_b0', 'v1', "
        "'completed', 'downy_mildew', 0.91 FROM image"
    )
    api_client.post(
        f"/api/v1/vineyards/{vineyard}/treatments",
        headers=headers,
        json={"applied_on": today.isoformat(), "fungicide": "contact"},
    )

    response = api_client.get(
        f"/api/v1/vineyards/{vineyard}/report.pdf", headers=headers
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"] == (
        f'attachment; filename="vitiguard-report-{today.isoformat()}.pdf"'
    )
    assert response.content.startswith(b"%PDF-")
    [data] = collected
    assert (data.name, data.planting_year, data.variety) == ("Криково", 2015, None)
    assert data.area_ha > 0
    # Худшее из моделей милдью — высокий; оидиума ещё нет
    assert (data.risk[0].downy, data.risk[0].powdery) == (RiskLevel.HIGH, None)
    assert len(data.risk) == 8
    assert data.plan.protected_until[Disease.DOWNY_MILDEW] is not None
    assert [point.ndvi for point in data.indices] == [pytest.approx(0.7)]
    [check] = data.checks
    assert check.verdict == "милдью"
    assert check.confidence == pytest.approx(0.91)
    assert [t.fungicide for t in data.treatments] == [Fungicide.CONTACT]

    stranger = login(api_client, "stranger@example.md")
    url = f"/api/v1/vineyards/{vineyard}/report.pdf"
    assert api_client.get(url, headers=stranger).status_code == 404
    assert api_client.get(url).status_code == 401


@pytest.mark.integration
def test_report_shows_variety_and_current_season_only(
    api_client: TestClient, run_sql: RunSql, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.api.routes import reports

    collected: list[ReportData] = []

    def capture(data: ReportData) -> bytes:
        collected.append(data)
        return b"%PDF-"

    monkeypatch.setattr(reports, "render_report", capture)
    headers = login(api_client)
    [(variety_id, variety_name)] = run_sql(
        "SELECT id, name FROM varieties ORDER BY name LIMIT 1"
    )
    ring = [[28.86, 47.14], [28.862, 47.14], [28.862, 47.141], [28.86, 47.14]]
    vineyard = api_client.post(
        "/api/v1/vineyards",
        headers=headers,
        json={
            "name": "Криково",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "variety_id": str(variety_id),
        },
    ).json()["id"]
    last_year = local_today().year - 1
    run_sql(
        "WITH scan AS (INSERT INTO satellite_scans (vineyard_id, product_id, "
        "acquired_at, cloud_cover, parcel_cloud_cover, status) VALUES "
        f"('{vineyard}', 'S2_old', '{last_year - 1}-07-01T09:17:00+00', 0, 0, "
        "'completed') RETURNING id) INSERT INTO vegetation_indices (scan_id, "
        "index_type, mean_value, min_value, max_value, std_value) "
        "SELECT id, 'ndvi', 0.7, 0.6, 0.8, 0.05 FROM scan"
    )

    api_client.get(f"/api/v1/vineyards/{vineyard}/report.pdf", headers=headers)

    assert collected[0].variety == variety_name
    # Снимок позапрошлого года в отчёт сезона не попадает
    assert collected[0].indices == []
