"""Build the ready-to-use kits for the optional human add-ons under addons/ (see HUMAN_ADDONS.md).

gold/            150 TEST-split exterior images (stratified by tradition, <=6 per temple) + Label Studio config (7 classes, NO pre-labels)
corrections/     ~500 TRAIN-split exterior images with the zero-shot (J1) pre-labels as Label Studio predictions
labels_audit/    temple_styles_review.csv (rule label + sources) and make_human_labels.py
taxonomy_review/ checklist.csv (one row per entry and source) and instructions
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from templexai.data.load import exterior_frame
from templexai.detect.cached import CachedBoxes
from templexai.interfaces import COMPONENTS, TRADITIONS
from templexai.knowledge.taxonomy import load_taxonomy

OUT = Path("addons")
LS_CONFIG = """<View>
  <Image name="image" value="$image" zoom="true"/>
  <RectangleLabels name="label" toName="image">
{labels}
  </RectangleLabels>
</View>
"""
COLORS = ["#d97706", "#0d9488", "#7c3aed", "#e11d48", "#2563eb", "#65a30d", "#ea580c"]


def ls_config() -> str:
    return LS_CONFIG.format(labels="\n".join(f'    <Label value="{c}" background="{COLORS[i]}"/>' for i, c in enumerate(COMPONENTS)))


def stratified(df: pd.DataFrame, n: int, per_temple: int, seed: int = 42) -> pd.DataFrame:
    """~n images, round-robin across traditions, at most `per_temple` per temple, deterministic."""
    rng = np.random.default_rng(seed)
    pools = {}
    for t in TRADITIONS:
        sub = df[df.tradition == t].sample(frac=1, random_state=seed)
        pools[t] = sub.groupby("temple_id").head(per_temple).reset_index(drop=True)
    picked, idx = [], {t: 0 for t in TRADITIONS}
    while len(picked) < n and any(idx[t] < len(pools[t]) for t in TRADITIONS):
        for t in TRADITIONS:
            if idx[t] < len(pools[t]) and len(picked) < n:
                picked.append(pools[t].iloc[idx[t]]); idx[t] += 1
    return pd.DataFrame(picked).reset_index(drop=True)


def copy_images(df: pd.DataFrame, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for r in df.itertuples():
        shutil.copy2(Path("data") / r.file, dest / f"{r.image_id}.jpg")


def gold() -> None:
    d = OUT / "gold"
    df = exterior_frame("data")
    df = df[(df.split == "test") & (df.image_type == "exterior_full")]
    pick = stratified(df, 150, per_temple=6)
    copy_images(pick, d / "images")
    pick[["image_id", "temple_id", "tradition", "file", "licence", "author", "url"]].to_csv(d / "gold_image_list.csv", index=False)
    (d / "label_config.xml").write_text(ls_config())
    tasks = [{"data": {"image": f"/data/local-files/?d=addons/gold/images/{r.image_id}.jpg"}} for r in pick.itertuples()]  # NO predictions on purpose
    (d / "import_tasks.json").write_text(json.dumps(tasks, indent=1))
    print("gold:", len(pick), pick.tradition.value_counts().to_dict())


def corrections() -> None:
    d = OUT / "corrections"
    df = exterior_frame("data")
    df = df[df.split == "train"]
    pick = stratified(df, 500, per_temple=3)
    copy_images(pick, d / "images")
    cb = CachedBoxes("data/derived/boxes_zeroshot.json")
    tasks = []
    for r in pick.itertuples():
        rec = cb.boxes[r.image_id]
        W, H = rec["width"], rec["height"]
        res = []
        for k, det in enumerate(rec["detections"]):
            if det["score"] < cb.thresholds.get(det["label"], 0.25):
                continue
            x0, y0, x1, y1 = det["box_xyxy"]
            res.append({"id": f"b{k}", "type": "rectanglelabels", "from_name": "label", "to_name": "image", "original_width": W, "original_height": H,
                        "image_rotation": 0, "value": {"x": 100 * x0 / W, "y": 100 * y0 / H, "width": 100 * (x1 - x0) / W, "height": 100 * (y1 - y0) / H,
                                                       "rotation": 0, "rectanglelabels": [det["label"]]}, "score": det["score"]})
        tasks.append({"data": {"image": f"/data/local-files/?d=addons/corrections/images/{r.image_id}.jpg"},
                      "predictions": [{"model_version": "grounding-dino-tiny (J1 zero-shot)", "result": res}]})
    (d / "label_config.xml").write_text(ls_config())
    (d / "import_with_prelabels.json").write_text(json.dumps(tasks))
    pick[["image_id", "temple_id", "tradition", "file"]].to_csv(d / "correction_image_list.csv", index=False)
    print("corrections:", len(pick))


def labels_audit() -> None:
    d = OUT / "labels_audit"
    d.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv("data/labels/temple_styles_rule.csv")
    lab = lab[lab.status == "LABELLED"]
    tem = pd.read_csv("data/meta/temples.csv")[["temple_id", "name", "enwiki_title", "state"]]
    sp = pd.read_csv("data/splits.csv")[["temple_id", "split"]]
    rows = []
    for r in lab.itertuples():
        for ev in json.loads(r.evidence_json):
            if ev.get("mapped_to"):
                rows.append({"temple_id": r.temple_id, "rule_label": r.tradition, "source": ev["source"], "source_url": ev["url"], "raw_value": ev["raw_value"], "rule_matched": ev["rule"]})
    d_df = pd.DataFrame(rows).merge(tem, on="temple_id").merge(sp, on="temple_id")
    one = d_df.groupby("temple_id").agg(name=("name", "first"), state=("state", "first"), split=("split", "first"), rule_label=("rule_label", "first"),
                                        sources=("source", lambda s: " | ".join(s)), source_urls=("source_url", lambda s: " | ".join(s)), raw_values=("raw_value", lambda s: " | ".join(map(str, s)))).reset_index()
    one["human_tradition"] = ""; one["human_source"] = ""; one["notes"] = ""
    one = one.sample(frac=1, random_state=42).reset_index(drop=True)  # random order: audit the first 50 for a sample estimate
    one.to_csv(d / "temple_styles_review.csv", index=False)
    (d / "make_human_labels.py").write_text(MAKE_HUMAN)
    print("labels_audit:", len(one))


MAKE_HUMAN = '''"""Turn the reviewed CSV into data/labels/temple_styles_human.csv (columns temple_id, tradition, source).
Usage (from the repo root):  python addons/labels_audit/make_human_labels.py
A row counts only if human_tradition is one of Nagara / Dravida / Deccan_Vesara / EXCLUDE AND human_source is non-empty.
Rows you leave blank are NOT overridden (the rule-based label stays). Also prints the agreement rate with the rule-based labels."""
import csv
import math
from pathlib import Path

ALLOWED = {"Nagara", "Dravida", "Deccan_Vesara", "EXCLUDE"}
ALIAS = {"deccan-vesara": "Deccan_Vesara", "deccan/vesara": "Deccan_Vesara", "deccan_vesara": "Deccan_Vesara"}
src = Path("addons/labels_audit/temple_styles_review.csv")
rows, bad, agree, n = [], [], 0, 0
for r in csv.DictReader(src.open()):
    t = (r["human_tradition"] or "").strip()
    if not t:
        continue
    t = ALIAS.get(t.lower(), t)
    if t not in ALLOWED or not (r["human_source"] or "").strip():
        bad.append(r["temple_id"]); continue
    rows.append({"temple_id": r["temple_id"], "tradition": t, "source": r["human_source"].strip()})
    n += 1; agree += t == r["rule_label"]
Path("data/labels").mkdir(parents=True, exist_ok=True)
with open("data/labels/temple_styles_human.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["temple_id", "tradition", "source"]); w.writeheader(); w.writerows(rows)
print(f"wrote {len(rows)} human labels; skipped {len(bad)} invalid rows {bad[:10]}")
if n:
    p = agree / n; se = math.sqrt(p * (1 - p) / n)
    print(f"agreement with rule-based labels: {agree}/{n} = {p:.1%} (normal-approx 95% CI {max(0, p - 1.96 * se):.1%}-{min(1, p + 1.96 * se):.1%})")
'''


def taxonomy_review() -> None:
    d = OUT / "taxonomy_review"
    d.mkdir(parents=True, exist_ok=True)
    tax = load_taxonomy()
    rows = []
    for kind in ("components", "traditions"):
        for name, e in tax[kind].items():
            for s in e["sources"]:
                rows.append({"kind": kind[:-1], "entry": name, "definition": e["definition"], "visual_cues": "; ".join(e["visual_cues"]), "source_page": s["page"],
                             "source_url": s["url"], "quote": s["quote"], "ok_(y/n)": "", "correction": "", "better_source": ""})
    pd.DataFrame(rows).to_csv(d / "checklist.csv", index=False)
    print("taxonomy_review:", len(rows), "rows")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    gold(); corrections(); labels_audit(); taxonomy_review()


if __name__ == "__main__":
    main()
