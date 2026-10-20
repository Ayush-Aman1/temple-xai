"""Mapping between original-image pixels and the model's 192x192 centre crop (Resize(224) + CenterCrop(192))."""
from __future__ import annotations

import numpy as np
import torchvision.transforms as T

IMG, RESIZE = 192, 224
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
EVAL_TF = T.Compose([T.Resize(RESIZE), T.CenterCrop(IMG), T.ToTensor(), NORM])


def crop_params(w: int, h: int) -> tuple[float, float, float]:
    """scale, offset_x, offset_y such that crop_xy = orig_xy * scale - offset."""
    s = RESIZE / min(w, h)
    return s, (round(w * s) - IMG) / 2.0, (round(h * s) - IMG) / 2.0


def box_to_crop(box, w: int, h: int) -> list[float]:
    s, ox, oy = crop_params(w, h)
    x0, y0, x1, y1 = box
    return [float(np.clip(x0 * s - ox, 0, IMG)), float(np.clip(y0 * s - oy, 0, IMG)), float(np.clip(x1 * s - ox, 0, IMG)), float(np.clip(y1 * s - oy, 0, IMG))]


def box_mask(box_crop, shape=(IMG, IMG)) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = [int(round(v)) for v in box_crop]
    m[y0:y1, x0:x1] = True
    return m


def cam_to_original(cam: np.ndarray, w: int, h: int) -> np.ndarray:
    """Place a (192,192) CAM back on an (h, w) canvas (zero outside the area the model saw)."""
    import cv2

    s, ox, oy = crop_params(w, h)
    cw, ch = max(1, int(round(IMG / s))), max(1, int(round(IMG / s)))
    small = cv2.resize(cam.astype(np.float32), (cw, ch), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((h, w), dtype=np.float32)
    x0, y0 = int(round(ox / s)), int(round(oy / s))
    xs, ys = max(0, x0), max(0, y0)
    xe, ye = min(w, x0 + cw), min(h, y0 + ch)
    out[ys:ye, xs:xe] = small[ys - y0:ye - y0, xs - x0:xe - x0]
    return out
