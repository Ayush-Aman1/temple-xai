"""Component feature vectors from detector boxes (fixed order, documented in knowledge/taxonomy.yaml).

Per component (7, in COMPONENTS order): present, max_score, count, total_rel_area, centroid_y of the largest box
(0..1, 0=top), aspect (w/h) of the largest box. -> 42 dims. Pure numpy so it runs on Kaggle and locally.
"""
from __future__ import annotations

import numpy as np

COMPONENTS = ["shikhara", "vimana", "gopuram", "mandapa", "pillar", "jagati_base", "amalaka"]
TRADITIONS = ["Nagara", "Dravida", "Deccan_Vesara"]
PER_COMPONENT = ["present", "max_score", "count", "total_rel_area", "centroid_y_largest", "aspect_largest"]
DIM = len(COMPONENTS) * len(PER_COMPONENT)


def feature_names() -> list[str]:
    return [f"{c}:{f}" for c in COMPONENTS for f in PER_COMPONENT]


def component_vector(dets: list[dict], width: int, height: int, thresholds: dict[str, float] | None = None) -> np.ndarray:
    """dets: [{label, box_xyxy, score}]. thresholds: optional per-class score cut-off."""
    v = np.zeros(DIM, dtype=np.float32)
    area_img = float(width * height) or 1.0
    for ci, comp in enumerate(COMPONENTS):
        thr = (thresholds or {}).get(comp, 0.0)
        ds = [d for d in dets if d["label"] == comp and d["score"] >= thr]
        if not ds:
            continue
        areas = []
        for d in ds:
            x0, y0, x1, y1 = d["box_xyxy"]
            areas.append(max(0.0, x1 - x0) * max(0.0, y1 - y0))
        big = int(np.argmax(areas))
        x0, y0, x1, y1 = ds[big]["box_xyxy"]
        base = ci * len(PER_COMPONENT)
        v[base:base + 6] = [1.0, max(d["score"] for d in ds), float(len(ds)), min(1.0, sum(areas) / area_img),
                            ((y0 + y1) / 2) / height, (x1 - x0) / max(1e-6, (y1 - y0))]
    return v


def knowledge_prior(vec: np.ndarray, support: dict[str, dict[str, int]]) -> np.ndarray:
    """3-dim evidence prior over TRADITIONS: sum over present components of normalised knowledge support.

    support comes from configs/knowledge_support.yaml (fetched-sentence counts, MACHINE_COMPILED). Components with no
    tradition-specific support contribute nothing. Returns a probability vector (uniform if no evidence).
    """
    score = np.zeros(len(TRADITIONS), dtype=np.float64)
    for ci, comp in enumerate(COMPONENTS):
        base = ci * len(PER_COMPONENT)
        if vec[base] < 0.5:
            continue
        row = np.array([support.get(comp, {}).get(t, 0) for t in TRADITIONS], dtype=np.float64)
        if row.sum() > 0:
            score += vec[base + 1] * row / row.sum()
    if score.sum() <= 0:
        return np.full(len(TRADITIONS), 1.0 / len(TRADITIONS))
    return score / score.sum()
