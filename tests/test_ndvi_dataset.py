import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from raster_scene import CLOUD, PARCEL, make_scene

from app.cli import ndvi_dataset
from app.core.errors import ExternalServiceError
from app.integrations.earth_search import Band, EarthScene, EarthSearchClient

FEATURE: dict[str, Any] = {
    "type": "Feature",
    "properties": {"id": "way/1", "point": "cricova", "area_ha": 1.2},
    "geometry": PARCEL,
}
SEASONS = [(datetime(2026, 4, 1, tzinfo=UTC), datetime(2026, 10, 15, tzinfo=UTC))]


def scenes(tmp_path: Path) -> list[EarthScene]:
    clear = make_scene(tmp_path)
    cloudy = make_scene(
        tmp_path,
        scl=CLOUD,
        product_id="S2B_35TPN_20260603_0_L2A",
        acquired_at=datetime(2026, 6, 3, 9, 7, tzinfo=UTC),
    )
    broken = make_scene(
        tmp_path,
        product_id="S2A_35TPN_20260715_1_L2A",
        acquired_at=datetime(2026, 7, 15, 9, 17, tzinfo=UTC),
    )
    broken.bands["B08"] = Band(href=str(tmp_path / "missing.tif"))
    return [clear, cloudy, broken]


class FakeClient:
    def __init__(self, found: list[EarthScene]) -> None:
        self.found = found
        self.bodies: list[dict[str, Any]] = []

    async def search(self, *_: Any, extra_body: dict[str, Any]) -> list[EarthScene]:
        self.bodies.append(extra_body)
        return self.found


async def test_collect_reads_indices_and_counts_failures(tmp_path: Path) -> None:
    client = FakeClient(scenes(tmp_path))

    rows, failed = await ndvi_dataset.collect(
        client,  # type: ignore[arg-type]
        [FEATURE],
        SEASONS,
        max_cloud=40,
        workers=2,
    )

    assert failed == 1
    assert client.bodies == [{"query": {"eo:cloud_cover": {"lte": 40}}}]
    by_date = {row["date"]: row for row in rows}
    clear = by_date["2026-08-29"]
    assert clear["vineyard"] == "way/1"
    assert clear["point"] == "cricova"
    assert clear["ndvi_mean"] == pytest.approx(0.714, abs=1e-3)
    assert clear["parcel_cloud"] == 0
    cloudy = by_date["2026-06-03"]
    assert cloudy["ndvi_mean"] == ""
    assert cloudy["parcel_cloud"] == 100


def test_main_writes_sorted_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    found = scenes(tmp_path)[:2]
    vineyards = tmp_path / "vineyards.geojson"
    vineyards.write_text(
        json.dumps({"type": "FeatureCollection", "features": [FEATURE]}),
        encoding="utf-8",
    )

    async def fake_search(*_: Any, **__: Any) -> list[EarthScene]:
        return found

    monkeypatch.setattr(EarthSearchClient, "search", fake_search)
    out = tmp_path / "out" / "observations.csv"

    ndvi_dataset.main(
        ["--vineyards", str(vineyards), "--out", str(out), "--seasons", "2026"]
    )

    with out.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert tuple(rows[0]) == ndvi_dataset.COLUMNS
    assert [row["date"] for row in rows] == ["2026-06-03", "2026-08-29"]


def test_main_resumes_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    second = {**FEATURE, "properties": {**FEATURE["properties"], "id": "way/2"}}
    vineyards = tmp_path / "vineyards.geojson"
    vineyards.write_text(
        json.dumps({"type": "FeatureCollection", "features": [FEATURE, second]}),
        encoding="utf-8",
    )
    out = tmp_path / "observations.csv"
    # Первый виноградник уже собран прошлым запуском
    collected: dict[str, object] = {**dict.fromkeys(ndvi_dataset.COLUMNS, "")}
    collected["vineyard"] = "way/1"
    ndvi_dataset.write_csv([collected], out)
    found = scenes(tmp_path)[:1]
    searched: list[Any] = []

    async def fake_search(
        _: Any, geometry: Any, *__: Any, **___: Any
    ) -> list[EarthScene]:
        searched.append(geometry)
        return found

    monkeypatch.setattr(EarthSearchClient, "search", fake_search)

    ndvi_dataset.main(
        ["--vineyards", str(vineyards), "--out", str(out), "--seasons", "2026"]
    )

    assert len(searched) == 1
    with out.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [row["vineyard"] for row in rows] == ["way/1", "way/2"]


class FlakyClient(FakeClient):
    def __init__(self, found: list[EarthScene], failures: int) -> None:
        super().__init__(found)
        self.failures = failures

    async def search(self, *args: Any, extra_body: dict[str, Any]) -> list[EarthScene]:
        if self.failures:
            self.failures -= 1
            raise ExternalServiceError("Каталог недоступен")
        return await super().search(*args, extra_body=extra_body)


async def test_collect_retries_when_catalog_fails(tmp_path: Path) -> None:
    client = FlakyClient(scenes(tmp_path)[:1], failures=2)

    rows, failed = await ndvi_dataset.collect_with_retries(
        client, FEATURE, SEASONS, 40, 1, pause=0  # type: ignore[arg-type]
    )

    assert (len(rows), failed) == (1, 0)


async def test_collect_gives_up_after_attempts(tmp_path: Path) -> None:
    client = FlakyClient(scenes(tmp_path)[:1], failures=5)

    with pytest.raises(ExternalServiceError):
        await ndvi_dataset.collect_with_retries(
            client, FEATURE, SEASONS, 40, 1, attempts=2, pause=0  # type: ignore[arg-type]
        )
