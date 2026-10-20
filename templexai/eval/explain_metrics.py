"""Explanation metrics.

GROUND-TRUTH-FREE (run on every build; agent-only mode):
  * deletion / insertion faithfulness AUC for Grad-CAM vs a random-saliency baseline
  * component-occlusion sensitivity vs equal-area random boxes
  * SHAP - knowledge-graph alignment vs a random-component baseline, and SHAP component-deletion AUC vs random order
  * Grad-CAM / detector-box agreement (pointing game, energy share) - AGREEMENT with the detector, never correctness
GOLD-BASED (implemented + unit-tested on synthetic data; run only when data/boxes/gold/*.txt exists):
  * pointing game, energy pointing, explanation IoU at top 10/20/30 % pixels, detector mAP50 / mAP50-95
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

IMG = 192
FRACS = np.linspace(0, 1, 11)


def auc(ys: np.ndarray, xs: np.ndarray = FRACS) -> float:
    return float(np.trapezoid(ys, xs))


def rank_masks(sal: np.ndarray, fracs: np.ndarray = FRACS) -> list[np.ndarray]:
    """Boolean masks of the top-f fraction of pixels of `sal` (ties broken stably)."""
    flat = sal.ravel()
    order = np.argsort(-flat, kind="stable")
    out = []
    for f in fracs:
        m = np.zeros(flat.size, dtype=bool)
        m[order[: int(round(f * flat.size))]] = True
        out.append(m.reshape(sal.shape))
    return out


def deletion_insertion_curves(prob_fn, x: np.ndarray, sal: np.ndarray, baseline_del: np.ndarray, baseline_ins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x: (3,H,W) normalised image. prob_fn: batch (B,3,H,W) -> probability of the originally predicted class (B,).
    Deletion: top-f pixels replaced by baseline_del. Insertion: start from baseline_ins, top-f pixels restored from x."""
    masks = rank_masks(sal)
    dele = np.stack([np.where(m[None], baseline_del, x) for m in masks]).astype(np.float32)
    ins = np.stack([np.where(m[None], x, baseline_ins) for m in masks]).astype(np.float32)
    p = prob_fn(np.concatenate([dele, ins]))
    return p[: len(masks)], p[len(masks):]


def equal_area_random_box(box_crop, rng: np.random.Generator) -> list[float]:
    x0, y0, x1, y1 = box_crop
    w, h = x1 - x0, y1 - y0
    nx = rng.uniform(0, max(1e-6, IMG - w))
    ny = rng.uniform(0, max(1e-6, IMG - h))
    return [nx, ny, nx + w, ny + h]


def temple_bootstrap(values: np.ndarray, temple_ids: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95 % CI resampling TEMPLES (images of one temple are correlated)."""
    rng = np.random.default_rng(seed)
    ids = np.unique(temple_ids)
    by = {t: values[temple_ids == t] for t in ids}
    means = []
    for _ in range(n):
        pick = rng.choice(ids, len(ids))
        v = np.concatenate([by[t] for t in pick])
        means.append(np.nanmean(v))
    return float(np.nanmean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------- gold-based
def pointing_hit(cam: np.ndarray, boxes_crop: list[list[float]]) -> bool:
    if not boxes_crop:
        return False
    y, x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in boxes_crop)


def energy_inside(cam: np.ndarray, boxes_crop: list[list[float]]) -> float:
    from templexai.xai.geometry import box_mask

    tot = float(cam.sum()) + 1e-12
    if not boxes_crop:
        return 0.0
    m = np.zeros(cam.shape, dtype=bool)
    for b in boxes_crop:
        m |= box_mask(b, cam.shape)
    return float(cam[m].sum() / tot)


def explanation_iou(cam: np.ndarray, boxes_crop: list[list[float]], top: float) -> float:
    from templexai.xai.geometry import box_mask

    if not boxes_crop:
        return 0.0
    gt = np.zeros(cam.shape, dtype=bool)
    for b in boxes_crop:
        gt |= box_mask(b, cam.shape)
    pred = rank_masks(cam, np.array([top]))[0]
    return float((pred & gt).sum() / max(1, (pred | gt).sum()))


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def average_precision(preds: list[dict], gts: dict[str, list[list[float]]], iou_thr: float) -> float:
    """Single-class AP (all-point interpolation). preds: [{image, box, score}], gts: image -> boxes."""
    npos = sum(len(v) for v in gts.values())
    if npos == 0:
        return float("nan")
    used = {k: [False] * len(v) for k, v in gts.items()}
    tp, fp = [], []
    for p in sorted(preds, key=lambda d: -d["score"]):
        g = gts.get(p["image"], [])
        best, bi = 0.0, -1
        for i, gb in enumerate(g):
            io = _iou(p["box"], gb)
            if io > best:
                best, bi = io, i
        if best >= iou_thr and not used[p["image"]][bi]:
            used[p["image"]][bi] = True
            tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    if not tp:
        return 0.0
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec, prec = tp / npos, tp / np.maximum(tp + fp, 1)
    mrec, mpre = np.concatenate([[0], rec, [1]]), np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def detector_map(preds: list[dict], gts: dict[str, list[dict]], classes: list[str]) -> dict:
    """preds: [{image, label, box, score}], gts: image -> [{label, box}] in the same pixel frame. Returns mAP50 and mAP50-95."""
    thrs = np.arange(0.5, 0.96, 0.05)
    per_class = {}
    for c in classes:
        gc = {k: [g["box"] for g in v if g["label"] == c] for k, v in gts.items()}
        pc = [{"image": p["image"], "box": p["box"], "score": p["score"]} for p in preds if p["label"] == c]
        aps = [average_precision(pc, gc, t) for t in thrs]
        per_class[c] = {"AP50": aps[0], "AP50_95": float(np.nanmean(aps)) if not np.all(np.isnan(aps)) else float("nan")}
    return {"mAP50": float(np.nanmean([v["AP50"] for v in per_class.values()])),
            "mAP50_95": float(np.nanmean([v["AP50_95"] for v in per_class.values()])), "per_class": per_class}


def has_gold(gold_dir: str | Path = "data/boxes/gold") -> bool:
    d = Path(gold_dir)
    return d.exists() and (any(d.glob("*.txt")) or any(d.glob("*.json")))


def load_gold(gold_dir: str | Path = "data/boxes/gold") -> dict[str, list[dict]]:
    """Gold boxes by image id -> [{label, box}] with boxes NORMALISED (x0,y0,x1,y1).
    Accepts YOLO txt (one per image: class cx cy w h) and Label Studio JSON exports (rectanglelabels, percent coordinates)."""
    import json

    from templexai.interfaces import COMPONENTS

    out: dict[str, list[dict]] = {}
    d = Path(gold_dir)
    for f in d.glob("*.txt"):
        rows = []
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
                rows.append({"label": COMPONENTS[c], "box": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]})
        out[f.stem] = rows
    for f in d.glob("*.json"):
        for task in json.loads(f.read_text()):
            img = Path(task.get("data", {}).get("image", "")).stem
            rows = []
            for ann in task.get("annotations", [])[:1]:
                for r in ann.get("result", []):
                    if r.get("type") == "rectanglelabels" and r["value"]["rectanglelabels"]:
                        v = r["value"]
                        lab = v["rectanglelabels"][0]
                        if lab in COMPONENTS:
                            rows.append({"label": lab, "box": [v["x"] / 100, v["y"] / 100, (v["x"] + v["width"]) / 100, (v["y"] + v["height"]) / 100]})
            if img:
                out[img] = rows
    return out
