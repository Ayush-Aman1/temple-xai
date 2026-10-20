"""J1: zero-shot component boxes with Grounding DINO (IDEA-Research/grounding-dino-tiny) on every image.

One text-prompted pass per component (several English phrasings joined with " . "). Raw detections >= 0.15 are stored;
per-class thresholds are tuned WITHOUT ground truth: threshold_c = max(0.25, 60th percentile of the per-image max score
of class c over CLIP-tagged exterior_full images), i.e. a class fires on roughly the top 40 % of exterior-full images.
This is a documented recall proxy, not an accuracy claim. Writes <out>/boxes_zeroshot.json and labelstudio_import.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

for p in [Path(__file__).resolve().parents[3], *Path("/kaggle/input").glob("*/code")]:
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

if Path("/kaggle").exists():  # pin library versions to the ones the code was written against
    import subprocess

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.20.2", "pillow"], check=False)

MODEL = "IDEA-Research/grounding-dino-tiny"
RAW_MIN = 0.15
PROMPTS = {
    "shikhara": ["temple tower", "curvilinear tower", "temple spire"],
    "vimana": ["pyramidal temple tower", "stepped temple tower"],
    "gopuram": ["gateway tower", "temple entrance tower"],
    "mandapa": ["pillared hall", "temple porch with pillars"],
    "pillar": ["stone pillar", "carved stone column"],
    "jagati_base": ["raised platform", "temple base"],
    "amalaka": ["ribbed stone disc on top of tower"],
}


def phrase_token_spans(tokenizer, text: str, phrases: list[str]) -> list[list[int]]:
    """Token indices covered by each phrase inside `text` (via tokenizer offsets)."""
    enc = tokenizer(text, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    spans, cursor = [], 0
    for ph in phrases:
        c0 = text.index(ph, cursor)
        c1 = c0 + len(ph)
        cursor = c1
        spans.append([i for i, (a, b) in enumerate(offs) if b > a and a >= c0 and b <= c1])
    return spans


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float = 0.7) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]
        xx0 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx1 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
        iou = inter / (area(boxes[i]) + area(boxes[rest]) - inter + 1e-9)
        order = rest[iou < thr]
    return keep


def find_root(arg):
    if arg:
        return Path(arg)
    for c in Path("/kaggle/input").glob("*"):
        if (c / "splits.csv").exists():
            return c
    return Path("data")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default="/kaggle/working/out")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compute-label", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all-types", action="store_true")
    a = ap.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from templexai.device import pick_device

    root, out = find_root(a.data_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(a.device)
    df = pd.read_csv(root / "meta" / "images.csv")
    sp = pd.read_csv(root / "splits.csv")
    df = df.merge(sp[["temple_id", "split"]], on="temple_id")
    if "image_type" in df and not a.all_types:   # classifiers/detector only use exterior images
        df = df[df.image_type.astype(str).str.startswith("exterior")]
    if a.smoke:
        df = df.head(8)
    elif a.limit:
        df = df.head(a.limit)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL).to(device).eval()
    global TEXT, COMP_SPANS
    all_phr = [p for v in PROMPTS.values() for p in v]
    TEXT = " . ".join(all_phr) + " ."          # ONE forward pass per image; phrases scored from their own token spans
    spans = phrase_token_spans(proc.tokenizer, TEXT, all_phr)
    COMP_SPANS, k = {}, 0
    for comp, v in PROMPTS.items():
        COMP_SPANS[comp] = spans[k:k + len(v)]
        k += len(v)
    res: dict = {}
    t0 = time.time()
    for n, r in enumerate(df.itertuples(), 1):
        im = Image.open(root / r.file).convert("RGB")
        W, H = im.size
        inp = proc(images=im, text=TEXT, return_tensors="pt").to(device)
        with torch.no_grad():
            o = model(**inp)
        prob = o.logits[0].sigmoid().float().cpu().numpy()              # (queries, tokens)
        cx, cy, bw, bh = o.pred_boxes[0].float().cpu().numpy().T
        xyxy = np.stack([(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H], 1)
        dets = []
        for comp, idxs in COMP_SPANS.items():
            sc = np.max([prob[:, sp].max(1) for sp in idxs if sp], axis=0)   # best phrase of this component per query
            sel = np.where(sc >= RAW_MIN)[0]
            if len(sel):
                for k in nms(xyxy[sel], sc[sel]):
                    q = sel[k]
                    dets.append({"label": comp, "box_xyxy": [round(float(v), 1) for v in xyxy[q]], "score": round(float(sc[q]), 4)})
        res[r.image_id] = {"temple_id": r.temple_id, "split": r.split, "width": W, "height": H,
                           "image_type": getattr(r, "image_type", None), "detections": dets}
        if n % 100 == 0:
            print(f"J1 {n}/{len(df)} {time.time()-t0:.0f}s", flush=True)
    full = [v for v in res.values() if str(v["image_type"]) == "exterior_full"] or list(res.values())
    thr = {}
    for comp in PROMPTS:
        mx = [max([d["score"] for d in v["detections"] if d["label"] == comp], default=0.0) for v in full]
        thr[comp] = round(max(0.25, float(np.percentile(mx, 60))), 4)
    res["_thresholds"] = thr
    res["_meta"] = {"model": MODEL, "raw_min": RAW_MIN, "prompts": PROMPTS, "method": "single forward pass, phrase score = max token prob over the phrase span, per-class NMS 0.7", "threshold_rule": "max(0.25, p60 of per-image max score on exterior_full)",
                    "detector_mode": "DISTILLED_SOURCE_ZEROSHOT", "n_images": len(df), "device": device}
    (out / "boxes_zeroshot.json").write_text(json.dumps(res))
    # Label Studio pre-annotations (only used by the optional corrections add-on); TRAIN images only
    ls = []
    for iid, v in res.items():
        if iid.startswith("_") or v["split"] != "train":
            continue
        results = [{"from_name": "label", "to_name": "image", "type": "rectanglelabels", "original_width": v["width"],
                    "original_height": v["height"], "value": {"x": d["box_xyxy"][0] / v["width"] * 100, "y": d["box_xyxy"][1] / v["height"] * 100,
                    "width": (d["box_xyxy"][2] - d["box_xyxy"][0]) / v["width"] * 100, "height": (d["box_xyxy"][3] - d["box_xyxy"][1]) / v["height"] * 100,
                    "rectanglelabels": [d["label"]]}, "score": d["score"]} for d in v["detections"] if d["score"] >= thr[d["label"]]]
        ls.append({"data": {"image": f"{iid}.jpg"}, "predictions": [{"model_version": "grounding-dino-tiny", "result": results}]})
    (out / "labelstudio_import.json").write_text(json.dumps(ls))
    print("J1 done", len(df), "images; thresholds", thr)


if __name__ == "__main__":
    main()
