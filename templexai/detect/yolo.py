"""ComponentDetector backed by the distilled Ultralytics YOLO weights (artifacts/j3/detector_best.pt)."""
from __future__ import annotations

import json
from pathlib import Path

from templexai.interfaces import COMPONENTS, Detection


class YoloDetector:
    """detect(PIL image) -> [{label, box_xyxy, score}]. Weights: AGPL-3.0 (Ultralytics); see CREDITS.md."""

    def __init__(self, weights: str = "artifacts/j3/detector_best.pt", metrics: str = "artifacts/j3/metrics.json",
                 imgsz: int = 416, conf: float = 0.1, device: str = "cpu"):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.imgsz, self.conf, self.device = imgsz, conf, device
        self.thresholds: dict[str, float] = json.loads(Path(metrics).read_text()).get("thresholds", {}) if Path(metrics).exists() else {}
        self.mode = json.loads(Path(metrics).read_text()).get("detector_mode", "DISTILLED") if Path(metrics).exists() else "DISTILLED"

    def detect(self, img) -> list[Detection]:
        r = self.model.predict(img, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        out: list[Detection] = []
        for b in r.boxes:
            out.append({"label": COMPONENTS[int(b.cls)], "box_xyxy": [round(float(v), 1) for v in b.xyxy[0].tolist()], "score": round(float(b.conf), 4)})
        return out
