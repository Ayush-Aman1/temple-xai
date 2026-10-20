"""Collect training outputs -> artifacts/<job>/, artifacts/MANIFEST.json, artifacts/TRAINING_REPORT.md.

All numbers come from metrics files written by the jobs. Run: python -m templexai.artifacts.collect [--data-root data --art artifacts --raw artifacts/raw]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def sha(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "uncommitted"
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--art", default="artifacts")
    ap.add_argument("--raw", default=None)
    a = ap.parse_args()
    root, art = Path(a.data_root), Path(a.art)
    raw = Path(a.raw) if a.raw else art / "raw"
    art.mkdir(exist_ok=True)
    labels = pd.read_csv(root / "labels" / "temple_styles_rule.csv")
    splits = pd.read_csv(root / "splits.csv")
    imgs = pd.read_csv(root / "meta" / "images.csv")
    label_mode = "HUMAN_VERIFIED" if (labels.label_mode == "HUMAN_VERIFIED").any() else "RULE_BASED"
    ks = Path("configs/knowledge_rules.yaml")
    knowledge_mode = "HUMAN_REVIEWED" if Path("knowledge_reviewed.yaml").exists() else "MACHINE_COMPILED"
    data_hash = sha([root / "splits.csv", root / "labels" / "temple_styles_rule.csv"])
    gold = "AVAILABLE" if Path("data/boxes/gold").exists() and any(Path("data/boxes/gold").glob("*.json")) else "UNAVAILABLE"
    commit, date = git_commit(), pd.Timestamp.now().strftime("%Y-%m-%d")
    manifest, md = [], ["# Temple XAI training report", "",
                        f"Generated {date} from metrics files only (git {commit}).", ""]

    def prov(compute: str, detector_mode: str = "n/a") -> dict:
        return {"label_mode": label_mode, "knowledge_mode": knowledge_mode, "detector_mode": detector_mode, "gold_eval": gold,
                "compute": compute, "data_hash": data_hash, "git_commit": commit, "date": date}

    # ---- data tables
    lab = labels[labels.status == "LABELLED"]
    md += ["## Dataset", "", f"label_mode={label_mode}; temples considered={len(labels)}; labelled={len(lab)}", "",
           "Label status: " + json.dumps(labels.status.value_counts().to_dict()), "",
           "| split | " + " | ".join(sorted(splits.tradition.unique())) + " | temples | images (all types) |", "|---" * (len(splits.tradition.unique()) + 3) + "|"]
    img_by_t = imgs.groupby("temple_id").size()
    for s in ["train", "val", "test"]:
        g = splits[splits.split == s]
        cnt = [int((g.tradition == t).sum()) for t in sorted(splits.tradition.unique())]
        md.append(f"| {s} | " + " | ".join(map(str, cnt)) + f" | {len(g)} | {int(img_by_t.reindex(g.temple_id).fillna(0).sum())} |")
    if "image_type" in imgs:
        md += ["", "Image types: " + json.dumps(imgs.image_type.value_counts().to_dict()), ""]
    md += ["Licences: " + json.dumps(imgs.licence.str.extract(r"(CC0|CC BY-SA|CC BY|[Pp]ublic domain|PD)", expand=False).fillna("other").value_counts().head(8).to_dict()), ""]
    manifest.append({"path": str(root / "splits.csv"), "kind": "split", **prov("n/a")})

    # ---- J1
    det_mode = "DISTILLED"
    j1 = raw / "j1_zeroshot"
    if (j1 / "boxes_zeroshot.json").exists():
        (art / "j1").mkdir(exist_ok=True)
        for f in ["boxes_zeroshot.json", "labelstudio_import.json"]:
            if (j1 / f).exists():
                shutil.copy2(j1 / f, art / "j1" / f)
        b = json.loads((j1 / "boxes_zeroshot.json").read_text())
        thr = b.get("_thresholds", {})
        ids = [k for k in b if not k.startswith("_")]
        per = {c: float(np.mean([any(d["label"] == c and d["score"] >= thr[c] for d in b[i]["detections"]) for i in ids])) for c in thr}
        comp = json.loads((j1 / "compute.json").read_text())["compute"] if (j1 / "compute.json").exists() else "unknown"
        md += ["## J1 zero-shot component boxes (Grounding DINO tiny)", "", f"images={len(ids)}; compute={comp}", "",
               "| component | threshold | share of images with a detection |", "|---|---|---|"]
        md += [f"| {c} | {thr[c]} | {per[c]:.3f} |" for c in thr]
        md += ["", "Thresholds are a recall proxy (no ground truth); detections are evidence, not verified boxes.", ""]
        manifest.append({"path": "artifacts/j1/boxes_zeroshot.json", "kind": "zero-shot boxes", "metrics": {"share_with_detection": per},
                         **prov(comp, "ZEROSHOT")})

    # ---- J2
    j2 = raw / "j2_classifiers"
    if (j2 / "summary.json").exists():
        s = pd.DataFrame(json.loads((j2 / "summary.json").read_text()))
        comp = json.loads((j2 / "compute.json").read_text())["compute"] if (j2 / "compute.json").exists() else s.compute.iloc[0]
        s["test_acc"] = s.test.map(lambda d: d.get("accuracy")); s["test_f1"] = s.test.map(lambda d: d.get("macro_f1"))
        agg = s.groupby(["variant", "backbone"]).agg(n_seeds=("seed", "count"), acc_mean=("test_acc", "mean"), acc_std=("test_acc", "std"),
                                                     f1_mean=("test_f1", "mean"), f1_std=("test_f1", "std"), val_f1=("best_val_macro_f1", "mean")).reset_index()
        md += ["## J2 style classifiers (temple-disjoint test split)", "", f"compute={comp}; n_test_images={int(s.test.map(lambda d: d.get('n', 0)).max())}", "",
               "| variant | backbone | seeds | test acc (mean±std) | test macro-F1 (mean±std) | val macro-F1 |", "|---|---|---|---|---|---|"]
        md += [f"| {r.variant} | {r.backbone} | {r.n_seeds} | {r.acc_mean:.3f}±{0 if np.isnan(r.acc_std) else r.acc_std:.3f} | {r.f1_mean:.3f}±{0 if np.isnan(r.f1_std) else r.f1_std:.3f} | {r.val_f1:.3f} |" for r in agg.itertuples()]
        md += ["", "Variants: A image CNN; B +component features (zero-shot boxes); C +knowledge-prior features; D +evidence-alignment loss.", ""]
        for (v, bb), g in s.groupby(["variant", "backbone"]):
            best = g.sort_values("best_val_macro_f1").iloc[-1]
            dst = art / "j2" / f"{v}_{bb}"
            dst.mkdir(parents=True, exist_ok=True)
            for f in ["best.pt", "metrics.json", "predictions.csv", "confusion.png"]:
                if (j2 / best.run / f).exists():
                    shutil.copy2(j2 / best.run / f, dst / f)
            manifest.append({"path": str(dst), "kind": "classifier", "run": best.run, "selected_by": "best val macro-F1 over seeds",
                             "metrics": best.test, **prov(comp, "ZEROSHOT" if v != "A" else "n/a")})
        shutil.copy2(j2 / "summary.json", art / "j2" / "summary.json")

    # ---- J3
    j3 = raw / "j3_detector"
    if (j3 / "metrics.json").exists():
        m = json.loads((j3 / "metrics.json").read_text())
        det_mode = m["detector_mode"]
        (art / "j3").mkdir(exist_ok=True)
        shutil.copy2(j3 / "metrics.json", art / "j3" / "metrics.json")
        if (j3 / "detector_best.pt").exists():
            shutil.copy2(j3 / "detector_best.pt", art / "j3" / "detector_best.pt")
        ag = m["agreement_with_zeroshot_val"]
        md += ["## J3 distilled component detector (YOLO)", "", f"detector_mode={det_mode}; gold_eval={m['gold_eval']}; compute={m['compute']}", "",
               f"Agreement with zero-shot boxes on val temples: mAP50={ag['mAP50']:.3f}, mAP50-95={ag['mAP50_95']:.3f}, P={ag['precision']:.3f}, R={ag['recall']:.3f}. "
               "This is agreement with pseudo-labels, not accuracy; detector mAP vs. gold boxes is available with the gold-set add-on.", ""]
        manifest.append({"path": "artifacts/j3/detector_best.pt", "kind": "detector", "metrics": ag, **prov(m["compute"], det_mode)})

    # ---- J0
    if (raw / "j0_gpu_check" / "gpu.json").exists():
        shutil.copy2(raw / "j0_gpu_check" / "gpu.json", art / "gpu.json")

    md += ["## Method (plain words)", "",
           f"- Style labels: {label_mode} - rule-based from Wikidata P149, the Wikipedia infobox and Commons categories; any disagreement excludes the temple (docs/METHOD.md).",
           f"- Knowledge layer: {knowledge_mode} - sentences fetched from Wikipedia, each with URL and short quote (docs/knowledge_sources.md).",
           f"- Component detector: {det_mode} - YOLO trained on zero-shot Grounding DINO boxes.",
           f"- Explanation ground truth (gold boxes): {gold}.", ""]
    (art / "MANIFEST.json").write_text(json.dumps({"generated": date, "items": manifest}, indent=1))
    (art / "TRAINING_REPORT.md").write_text("\n".join(md) + "\n")
    print(f"collect: {len(manifest)} manifest items -> {art}/MANIFEST.json, TRAINING_REPORT.md")


if __name__ == "__main__":
    main()
