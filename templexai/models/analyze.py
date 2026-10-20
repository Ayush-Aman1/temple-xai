"""End-to-end analysis of one image with the loaded Bundle: detection -> fusion prediction -> Grad-CAM -> SHAP -> evidence report."""
from __future__ import annotations

import base64
import io

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from templexai.interfaces import TRADITIONS
from templexai.knowledge.explain import build_evidence, render_report
from templexai.knowledge.taxonomy import load_taxonomy
from templexai.models.component_features import COMPONENTS, component_vector
from templexai.models.serving import Bundle
from templexai.xai.geometry import EVAL_TF, IMG, box_mask, box_to_crop, cam_to_original
from templexai.xai.gradcam import compute_cam_batch
from templexai.xai.shap_components import component_shap


def prepare(raw: bytes | Image.Image) -> Image.Image:
    im = Image.open(io.BytesIO(raw)) if isinstance(raw, (bytes, bytearray)) else raw
    return ImageOps.exif_transpose(im).convert("RGB")


def overlay_png(im: Image.Image, cam_full: np.ndarray, alpha: float = 0.5) -> str:
    """Heat-map PNG (transparent where the model did not look) as base64."""
    heat = cv2.applyColorMap((np.clip(cam_full, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    a = (np.clip(cam_full, 0, 1) * 255 * alpha * 1.6).clip(0, 255).astype(np.uint8)
    rgba = np.dstack([heat, a])
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def analyze(B: Bundle, im: Image.Image, with_cam: bool = True) -> dict:
    w, h = im.size
    dets_all = B.detector.detect(im)
    thr = B.detector.thresholds
    dets = [d for d in dets_all if d["score"] >= thr.get(d["label"], 0.0)]
    cvec = component_vector(dets, w, h)
    comp_t = torch.tensor(cvec[None], dtype=torch.float32)
    x = EVAL_TF(im)[None].to(B.device)
    B.net.comp = comp_t
    with torch.no_grad():
        P = torch.softmax(B.net(x), -1)[0].cpu().numpy()
    cls = int(P.argmax())
    pred = {"tradition": TRADITIONS[cls], "probability": float(P[cls]), "probabilities": {t: float(p) for t, p in zip(TRADITIONS, P)}}
    cam_png = None
    cam_energy: dict[str, float] = {}
    if with_cam:
        cam = compute_cam_batch(B.net, x, comp_t, [cls])[0]
        cam_png = overlay_png(im, cam_to_original(cam, w, h))
        for d in dets:
            bm = box_mask(box_to_crop(d["box_xyxy"], w, h))
            cam_energy[d["label"]] = max(cam_energy.get(d["label"], 0.0), float(cam[bm].sum() / (cam.sum() + 1e-12)))
    shap_c = component_shap(B.gb.predict_proba, cvec, cls)
    parts_p = B.gb.predict_proba(cvec[None])[0]
    ev = build_evidence(pred, dets, shap_c, cam_energy)
    return {"image": {"width": w, "height": h}, "prediction": pred,
            "components": [{**c, "box_xyxy": c["box_xyxy"]} for c in ev["components"]],
            "all_detections": dets, "gradcam_png_base64": cam_png,
            "shap": {"by_component": shap_c, "explained_class": TRADITIONS[cls], "model": "components-only gradient boosting",
                     "parts_model_prediction": TRADITIONS[int(parts_p.argmax())], "parts_model_probabilities": {t: float(p) for t, p in zip(TRADITIONS, parts_p)}},
            "evidence": ev, "report_markdown": render_report(ev), "caveat": ev["caveat"], "provenance": B.provenance}
