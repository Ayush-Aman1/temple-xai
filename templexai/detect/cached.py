"""Read-only detector over a cached boxes JSON ({image_id: {width,height,detections}}); used for the zero-shot boxes."""
from __future__ import annotations

import json
from pathlib import Path


class CachedBoxes:
    def __init__(self, path: str | Path):
        d = json.loads(Path(path).read_text())
        self.thresholds: dict[str, float] = d.pop("_thresholds", {})
        d.pop("_meta", None)
        self.boxes = d

    def get(self, image_id: str) -> dict | None:
        return self.boxes.get(image_id)
