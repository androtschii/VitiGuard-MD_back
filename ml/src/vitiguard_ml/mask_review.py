"""Выборочная ручная проверка масок поражения OpenCV (ml/pr-012).

U-Net учится на масках OpenCV, поэтому её IoU показывает согласие с правилами,
а не с глазом специалиста. Команда собирает страницу со случайными листьями:
человек отмечает, верно ли выделено поражение, и выгружает ответы в CSV.
Вторая команда считает долю верных масок по каждому классу.

Запуск из каталога ml:
  uv run --group train python -m vitiguard_ml.mask_review page --count 60
  uv run python -m vitiguard_ml.mask_review summary results/mask_review/answers.csv
"""

import argparse
import base64
import csv
import html
import io
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

VERDICTS = ("correct", "partly", "wrong")
SIDE = 256


def overlay(path: Path) -> tuple[str, float]:
    """Снимок с выделенным поражением (красным) и контуром листа; доля поражения."""
    from vitiguard_ml.masks import lesion_masks

    rgb = np.asarray(Image.open(path).convert("RGB").resize((SIDE, SIDE)))
    masks = lesion_masks(rgb)
    picture = rgb.astype(np.float32)
    picture[masks.lesion] = picture[masks.lesion] * 0.4 + np.array([230, 30, 30]) * 0.6
    picture[~masks.leaf] *= 0.35
    buffer = io.BytesIO()
    Image.fromarray(picture.astype(np.uint8)).save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode(), float(masks.severity)


def pick(manifest: Path, count: int, seed: int = 7) -> list[dict[str, str]]:
    """Поровну снимков каждого класса, без повторов."""
    with manifest.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    by_label: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    rng = random.Random(seed)
    per_label = max(1, count // len(by_label))
    picked: list[dict[str, str]] = []
    for label in sorted(by_label):
        items = by_label[label]
        picked.extend(rng.sample(items, min(per_label, len(items))))
    return picked


PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Проверка масок поражения</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 16px; color: #1c1917; background: #fafaf9; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }}
figure {{ margin: 0; background: #fff; border: 1px solid #e7e5e4; border-radius: 8px; padding: 8px; }}
img {{ width: 100%; border-radius: 4px; }}
label {{ margin-right: 8px; font-size: 14px; }}
button {{ margin: 16px 0; padding: 10px 16px; background: #622ac3; color: #fff; border: 0; border-radius: 6px; font-size: 15px; }}
</style></head><body>
<h1>Проверка масок поражения</h1>
<p>Красным выделено то, что OpenCV считает поражением, затемнено — не лист.
Отметьте каждый снимок и нажмите «Сохранить ответы».</p>
<div class="grid">{items}</div>
<button id="save">Сохранить ответы</button>
<script>
const rows = {rows};
document.getElementById('save').onclick = () => {{
  const lines = ['path,label,severity,verdict'];
  for (const row of rows) {{
    const checked = document.querySelector(`input[name="${{row.id}}"]:checked`);
    lines.push([row.path, row.label, row.severity, checked ? checked.value : ''].join(','));
  }}
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([lines.join('\\n')], {{ type: 'text/csv' }}));
  link.download = 'answers.csv';
  link.click();
}};
</script></body></html>
"""


def build_page(data: Path, out: Path, count: int) -> Path:
    rows = []
    items = []
    for index, row in enumerate(pick(data / "manifest.csv", count)):
        image, severity = overlay(data / row["path"])
        item_id = f"m{index}"
        rows.append(
            {
                "id": item_id,
                "path": row["path"],
                "label": row["label"],
                "severity": round(severity, 3),
            }
        )
        choices = " ".join(
            f'<label><input type="radio" name="{item_id}" value="{verdict}"> '
            f"{title}</label>"
            for verdict, title in zip(
                VERDICTS, ("верно", "частично", "неверно"), strict=True
            )
        )
        items.append(
            f'<figure><img alt="" src="data:image/jpeg;base64,{image}">'
            f"<figcaption>{html.escape(row['label'])}, поражение "
            f"{severity:.0%}<br>{choices}</figcaption></figure>"
        )
    out.mkdir(parents=True, exist_ok=True)
    page = out / "index.html"
    page.write_text(
        PAGE.format(items="".join(items), rows=json.dumps(rows)), encoding="utf-8"
    )
    return page


def summarize(answers: Path) -> dict[str, dict[str, float]]:
    """Доля верных, частично верных и неверных масок по классам и в целом.
    Снимки без отметки не учитываются."""
    with answers.open(encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["verdict"] in VERDICTS]
    groups: dict[str, Counter[str]] = {}
    for row in rows:
        groups.setdefault(row["label"], Counter())[row["verdict"]] += 1
        groups.setdefault("all", Counter())[row["verdict"]] += 1
    return {
        label: {
            "checked": float(sum(counts.values())),
            **{
                verdict: round(counts[verdict] / sum(counts.values()), 3)
                for verdict in VERDICTS
            },
        }
        for label, counts in sorted(groups.items())
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    page = commands.add_parser("page")
    page.add_argument("--data", type=Path, default=Path("data/processed"))
    page.add_argument("--out", type=Path, default=Path("results/mask_review"))
    page.add_argument("--count", type=int, default=60)
    summary = commands.add_parser("summary")
    summary.add_argument("answers", type=Path)
    args = parser.parse_args(argv)
    if args.command == "page":
        print(build_page(args.data, args.out, args.count))
    else:
        print(json.dumps(summarize(args.answers), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
