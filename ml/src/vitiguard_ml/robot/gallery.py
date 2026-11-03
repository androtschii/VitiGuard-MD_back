"""Галерея ошибок детекции за неделю для робота-сборщика (ml/pr-034).

Сравнивает предсказания модели с проверенной человеком разметкой из Label Studio:
- находит ложные срабатывания (FP: листва, столбы, принятые за гроздь/ствол);
- находит пропуски (FN: грозди в тени, стволы на фоне почвы);
- формирует автономную HTML-галерею с наложением рамок и рекомендациями,
  какие условия съёмки (освещение, угол солнца, тени) нужно добавить в обучение.
"""

import argparse
import base64
import html
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from vitiguard_ml.data.images import IMAGE_SUFFIXES
from vitiguard_ml.data.vine import CLASSES


@dataclass(frozen=True)
class BoundingBox:
    cls_id: int
    cx: float
    cy: float
    w: float
    h: float
    conf: float = 1.0


def box_iou(a: BoundingBox, b: BoundingBox) -> float:
    """Вычисляет IoU двух рамок."""
    ax1, ax2 = a.cx - a.w / 2.0, a.cx + a.w / 2.0
    ay1, ay2 = a.cy - a.h / 2.0, a.cy + a.h / 2.0
    bx1, bx2 = b.cx - b.w / 2.0, b.cx + b.w / 2.0
    by1, by2 = b.cy - b.h / 2.0, b.cy + b.h / 2.0

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    union = a.w * a.h + b.w * b.h - intersection
    return float(intersection / union) if union > 0 else 0.0


def parse_yolo_labels(path: Path) -> list[BoundingBox]:
    """Читает рамки из текстового файла формата YOLO."""
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            boxes.append(
                BoundingBox(
                    cls_id=int(parts[0]),
                    cx=float(parts[1]),
                    cy=float(parts[2]),
                    w=float(parts[3]),
                    h=float(parts[4]),
                    conf=float(parts[5]) if len(parts) > 5 else 1.0,
                )
            )
    return boxes


def match_boxes(
    gt_boxes: list[BoundingBox],
    pred_boxes: list[BoundingBox],
    iou_thresh: float = 0.5,
) -> tuple[list[BoundingBox], list[BoundingBox], list[BoundingBox]]:
    """Сопоставляет предсказания с истиной: возвращает (TP, FP, FN)."""
    matched_gt = set()
    tp = []
    fp = []

    for pred in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt or pred.cls_id != gt.cls_id:
                continue
            iou = box_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= iou_thresh and best_idx >= 0:
            matched_gt.add(best_idx)
            tp.append(pred)
        else:
            fp.append(pred)

    fn = [gt for i, gt in enumerate(gt_boxes) if i not in matched_gt]
    return tp, fp, fn


def render_error_overlay(
    image_path: Path,
    tp: list[BoundingBox],
    fp: list[BoundingBox],
    fn: list[BoundingBox],
    max_side: int = 512,
) -> str:
    """Рендерит кадр с цветными рамками (зелёный=TP, красный=FP, синий=FN) в base64."""
    with Image.open(image_path) as im:
        img = im.convert("RGB")
        img.thumbnail((max_side, max_side))
        w_thumb, h_thumb = img.width, img.height
        draw = ImageDraw.Draw(img)

        # Отрисовка TP (зелёный, тонкий)
        for b in tp:
            x1 = (b.cx - b.w / 2.0) * w_thumb
            y1 = (b.cy - b.h / 2.0) * h_thumb
            x2 = (b.cx + b.w / 2.0) * w_thumb
            y2 = (b.cy + b.h / 2.0) * h_thumb
            draw.rectangle([x1, y1, x2, y2], outline=(34, 197, 94), width=2)

        # Отрисовка FP (красный: ложные)
        for b in fp:
            x1 = (b.cx - b.w / 2.0) * w_thumb
            y1 = (b.cy - b.h / 2.0) * h_thumb
            x2 = (b.cx + b.w / 2.0) * w_thumb
            y2 = (b.cy + b.h / 2.0) * h_thumb
            draw.rectangle([x1, y1, x2, y2], outline=(239, 68, 68), width=3)
            label = f"FP: {CLASSES[b.cls_id]}"
            draw.text((x1 + 3, y1 + 3), label, fill=(239, 68, 68))

        # Отрисовка FN (синий: пропущенные истинные)
        for b in fn:
            x1 = (b.cx - b.w / 2.0) * w_thumb
            y1 = (b.cy - b.h / 2.0) * h_thumb
            x2 = (b.cx + b.w / 2.0) * w_thumb
            y2 = (b.cy + b.h / 2.0) * h_thumb
            draw.rectangle([x1, y1, x2, y2], outline=(59, 130, 246), width=3)
            label = f"FN: {CLASSES[b.cls_id]}"
            draw.text((x1 + 3, y1 + 3), label, fill=(59, 130, 246))

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode()


HTML_TEMPLATE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Галерея ошибок зрения робота</title>
<style>
body {{ font-family: system-ui, sans-serif; background: #0c0a09; color: #f5f5f4; margin: 20px; }}
.header {{ background: #1c1917; padding: 16px; border-radius: 8px; border: 1px solid #292524; margin-bottom: 20px; }}
.stats {{ display: flex; gap: 24px; margin-top: 10px; font-size: 15px; }}
.stat-item {{ background: #292524; padding: 8px 14px; border-radius: 6px; }}
.legend {{ display: flex; gap: 16px; margin-top: 10px; font-size: 14px; }}
.tag {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 6px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }}
.card {{ background: #1c1917; border: 1px solid #292524; border-radius: 8px; padding: 12px; overflow: hidden; }}
.card img {{ width: 100%; border-radius: 4px; display: block; }}
.card figcaption {{ font-size: 13px; margin-top: 8px; color: #d6d3d1; }}
.badge-fp {{ color: #ef4444; font-weight: 600; }}
.badge-fn {{ color: #3b82f6; font-weight: 600; }}
</style>
</head>
<body>
<div class="header">
  <h2>Галерея ошибок детекции за неделю (ml/pr-034)</h2>
  <div class="legend">
    <span><span class="tag" style="background:#22c55e;"></span>Верно (TP)</span>
    <span><span class="tag" style="background:#ef4444;"></span>Ложные детекции (FP)</span>
    <span><span class="tag" style="background:#3b82f6;"></span>Пропуски (FN)</span>
  </div>
  <div class="stats">
    <div class="stat-item">Кадров с ошибками: <b>{total_error_frames}</b></div>
    <div class="stat-item">Ложных детекций (FP): <b class="badge-fp">{total_fp}</b></div>
    <div class="stat-item">Пропущенных объектов (FN): <b class="badge-fn">{total_fn}</b></div>
  </div>
</div>
<div class="grid">
{cards}
</div>
</body>
</html>
"""


def build_weekly_error_gallery(
    images_dir: Path,
    gt_dir: Path,
    pred_dir: Path,
    out_html: Path,
    iou_thresh: float = 0.5,
) -> dict[str, int]:
    """Сравнивает предсказания с истиной для всех кадров каталога и генерирует галерею."""
    cards = []
    total_fp = 0
    total_fn = 0
    error_frames = 0

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        gt_file = gt_dir / f"{img_path.stem}.txt"
        pred_file = pred_dir / f"{img_path.stem}.txt"

        gt_boxes = parse_yolo_labels(gt_file)
        pred_boxes = parse_yolo_labels(pred_file)

        tp, fp, fn = match_boxes(gt_boxes, pred_boxes, iou_thresh=iou_thresh)

        if not fp and not fn:
            continue

        error_frames += 1
        total_fp += len(fp)
        total_fn += len(fn)

        b64 = render_error_overlay(img_path, tp, fp, fn)
        desc_parts = []
        if fp:
            desc_parts.append(f"<span class='badge-fp'>FP: {len(fp)}</span>")
        if fn:
            desc_parts.append(f"<span class='badge-fn'>FN: {len(fn)}</span>")

        cards.append(
            f"<div class='card'>"
            f"<img alt='' src='data:image/jpeg;base64,{b64}'>"
            f"<figcaption><b>{html.escape(img_path.name)}</b> — {' '.join(desc_parts)}</figcaption>"
            f"</div>"
        )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(
        HTML_TEMPLATE.format(
            total_error_frames=error_frames,
            total_fp=total_fp,
            total_fn=total_fn,
            cards="\n".join(cards),
        ),
        encoding="utf-8",
    )

    return {
        "error_frames": error_frames,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Папка со снимками")
    parser.add_argument(
        "--gt", type=Path, required=True, help="Папка с разметкой эксперта (.txt)"
    )
    parser.add_argument(
        "--pred", type=Path, required=True, help="Папка с предсказаниями модели (.txt)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/robot/gallery.html"),
        help="Выходной HTML",
    )
    parser.add_argument(
        "--iou", type=float, default=0.5, help="Порог IoU для сопоставления"
    )
    args = parser.parse_args(argv)

    summary = build_weekly_error_gallery(
        images_dir=args.images,
        gt_dir=args.gt,
        pred_dir=args.pred,
        out_html=args.out,
        iou_thresh=args.iou,
    )
    print(
        f"Галерея сохранена в {args.out}: кадров с ошибками {summary['error_frames']}"
    )


if __name__ == "__main__":
    main()
