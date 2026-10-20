"""J3: distilled component detector. Small Ultralytics YOLO trained on J1 zero-shot boxes (train-split exterior images,
boxes above the J1 per-class thresholds). detector_mode=DISTILLED. If <data-root>/boxes_corrected/ holds Label Studio
YOLO exports they are used instead (HUMAN_CORRECTED). Validation metrics on the val split are AGREEMENT with the
zero-shot boxes, not accuracy. Gold-box evaluation runs only if <data-root>/boxes_gold/ exists (else gold_eval=UNAVAILABLE).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

for p in [Path(__file__).resolve().parents[3], *Path("/kaggle/input").glob("*/code")]:
    sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402

if Path("/kaggle").exists():  # pin library versions to the ones the code was written against
    import subprocess

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.183"], check=False)

CLASSES = ["shikhara", "vimana", "gopuram", "mandapa", "pillar", "jagati_base", "amalaka"]
MIN_CONF_FLOOR = 0.25


def find_root(arg):
    if arg:
        return Path(arg)
    for c in Path("/kaggle/input").glob("*"):
        if (c / "splits.csv").exists():
            return c
    return Path("data")


def write_split(root: Path, boxes: dict, thr: dict, ids: list[str], files: dict, dest: Path) -> int:
    (dest / "images").mkdir(parents=True, exist_ok=True)
    (dest / "labels").mkdir(parents=True, exist_ok=True)
    n = 0
    for iid in ids:
        v = boxes[iid]
        lines = []
        for d in v["detections"]:
            if d["score"] < max(thr.get(d["label"], MIN_CONF_FLOOR), MIN_CONF_FLOOR):
                continue
            x0, y0, x1, y1 = d["box_xyxy"]
            W, H = v["width"], v["height"]
            lines.append(f"{CLASSES.index(d['label'])} {(x0+x1)/2/W:.6f} {(y0+y1)/2/H:.6f} {(x1-x0)/W:.6f} {(y1-y0)/H:.6f}")
        if not lines:
            continue
        src = root / files[iid]
        dst = dest / "images" / f"{iid}.jpg"
        if not dst.exists():
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)
        (dest / "labels" / f"{iid}.txt").write_text("\n".join(lines))
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default="/kaggle/working/out")
    ap.add_argument("--boxes", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--compute-label", default="kaggle_gpu")
    a = ap.parse_args()
    from ultralytics import YOLO
    from templexai.device import pick_device

    root, out = find_root(a.data_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(a.device)
    bp = Path(a.boxes) if a.boxes else next(iter([*Path("/kaggle/input").glob("*/boxes_zeroshot.json"), root / "derived" / "boxes_zeroshot.json"]), root / "derived" / "boxes_zeroshot.json")
    boxes = json.loads(bp.read_text())
    thr = boxes.pop("_thresholds")
    boxes.pop("_meta", None)
    imgs = pd.read_csv(root / "meta" / "images.csv")
    files = dict(zip(imgs.image_id, imgs.file))
    ext = set(imgs[imgs.image_type.astype(str).str.startswith("exterior")].image_id) if "image_type" in imgs else set(imgs.image_id)
    tr = [i for i, v in boxes.items() if v["split"] == "train" and i in ext]
    va = [i for i, v in boxes.items() if v["split"] == "val" and i in ext]
    if a.smoke:
        tr, va = tr[:12], va[:6] or tr[:4]
    work = out / "yolo_data"
    n_tr = write_split(root, boxes, thr, tr, files, work / "train")
    n_va = write_split(root, boxes, thr, va, files, work / "val")
    mode = "DISTILLED"
    corrected = root / "boxes_corrected"
    if corrected.exists() and any(corrected.glob("**/*.txt")):
        mode = "HUMAN_CORRECTED"  # corrected YOLO export overrides the distilled labels (add-on hook)
        for lab in corrected.glob("**/labels/*.txt"):
            if (work / "train" / "labels" / lab.name).exists():
                shutil.copy2(lab, work / "train" / "labels" / lab.name)
    (work / "data.yaml").write_text(json.dumps({"path": str(work.resolve()), "train": "train/images", "val": "val/images",
                                                "names": {i: c for i, c in enumerate(CLASSES)}}))
    print(f"J3 train={n_tr} val={n_va} mode={mode} device={device}", flush=True)
    model = YOLO(a.weights)
    dev = 0 if device == "cuda" else device
    model.train(data=str(work / "data.yaml"), epochs=1 if a.smoke else a.epochs, imgsz=320 if a.smoke else a.imgsz, batch=8 if a.smoke else 16,
                device=dev, project=str((out / "yolo_runs").resolve()), name="distilled", exist_ok=True, seed=0, deterministic=True, plots=False, workers=2, verbose=False)
    best = (out / "yolo_runs" / "distilled" / "weights" / "best.pt").resolve()
    if not best.exists():
        best = best.with_name("last.pt")
    shutil.copy2(best, out / "detector_best.pt")
    m = YOLO(str(out / "detector_best.pt")).val(data=str(work / "data.yaml"), imgsz=320 if a.smoke else a.imgsz, device=dev, plots=False, verbose=False)
    gold = (root / "boxes_gold").exists()
    metrics = {"detector_mode": mode, "gold_eval": "AVAILABLE" if gold else "UNAVAILABLE", "compute": a.compute_label if device == "cuda" else "local_" + device,
               "agreement_with_zeroshot_val": {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map), "precision": float(m.box.mp), "recall": float(m.box.mr),
                                              "note": "agreement with zero-shot pseudo-labels on val temples, NOT accuracy"},
               "n_train_images": n_tr, "n_val_images": n_va, "thresholds": thr, "weights_init": a.weights, "epochs": 1 if a.smoke else a.epochs}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    shutil.rmtree(work, ignore_errors=True)
    print("J3 done", metrics["agreement_with_zeroshot_val"])


if __name__ == "__main__":
    main()
