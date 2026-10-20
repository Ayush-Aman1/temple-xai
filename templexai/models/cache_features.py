"""P4 caching: run the distilled YOLO detector and the fine-tuned image backbone ONCE over all exterior images of labelled temples.

Outputs (data/derived/): boxes_yolo.json (same schema as boxes_zeroshot.json), feats_<run>.npy (pooled CNN features, row order =
feats_index.csv). Training of the fusion heads then never re-runs detection (guide P4).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from templexai.data.load import exterior_frame
from templexai.detect.yolo import YoloDetector
from templexai.device import pick_device
from templexai.models.net import load_run

IMG = 192


def eval_tf():
    return T.Compose([T.Resize(IMG + 32), T.CenterCrop(IMG), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--run", default="artifacts/j2/A_efficientnet_b0")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    root = Path(a.data_root)
    df = exterior_frame(root)
    if a.smoke:
        df = df.head(24)
    out = root / "derived"
    out.mkdir(exist_ok=True)
    dev = pick_device("auto")
    det = YoloDetector(device=dev)
    net = load_run(a.run, dev)
    tf = eval_tf()
    boxes: dict = {"_thresholds": det.thresholds, "_meta": {"detector_mode": det.mode}}
    feats = []
    for i, r in enumerate(df.itertuples()):
        im = Image.open(root / r.file).convert("RGB")
        boxes[r.image_id] = {"temple_id": r.temple_id, "split": r.split, "width": im.width, "height": im.height, "detections": det.detect(im)}
        with torch.no_grad():
            feats.append(net.pooled(tf(im)[None].to(dev)).cpu().numpy()[0])
        if i % 500 == 0:
            print(i, len(df), flush=True)
    name = Path(a.run).name
    (out / ("boxes_yolo_smoke.json" if a.smoke else "boxes_yolo.json")).write_text(json.dumps(boxes))
    np.save(out / f"feats_{name}{'_smoke' if a.smoke else ''}.npy", np.stack(feats).astype(np.float32))
    df[["image_id"]].to_csv(out / f"feats_index{'_smoke' if a.smoke else ''}.csv", index=False)
    print("cached", len(df))


if __name__ == "__main__":
    main()
