"""Load the trained artifacts once: serving model = fine-tuned image backbone + fusion head (P4) on distilled-YOLO component
vectors; SHAP on the components-only gradient-boosting model; knowledge from the taxonomy. Selection rule is in configs/serve.yaml."""
from __future__ import annotations

import json
import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from templexai.detect.yolo import YoloDetector
from templexai.device import pick_device
from templexai.interfaces import TRADITIONS
from templexai.models.fusion import Head
from templexai.eval.explain_metrics import has_gold
from templexai.models.net import load_run
from templexai.xai.gradcam import FusionNet


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


@dataclass
class Bundle:
    net: FusionNet
    detector: YoloDetector
    gb: object
    explainer: object
    device: str
    provenance: dict


def load_bundle(cfg_path: str = "configs/serve.yaml", device: str | None = None) -> Bundle:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    dev = pick_device(device or cfg.get("device", "auto"))
    base = load_run(cfg["backbone_run"], "cpu")
    ck = torch.load(Path(cfg["p4_dir"]) / "fusion.pt", map_location="cpu", weights_only=False)  # our own artifact (numpy mu/sd)
    head = Head(ck["d_in"])
    head.load_state_dict(ck["state"])
    net = FusionNet(base.bb, head.eval(), ck["mu"], ck["sd"]).to(dev).eval()
    det = YoloDetector(weights=cfg["detector_weights"], metrics=cfg["detector_metrics"], device="cpu" if dev == "mps" else dev)
    gb = pickle.load(open(Path(cfg["p4_dir"]) / "parts_gb.pkl", "rb"))
    from templexai.knowledge.taxonomy import load_taxonomy

    man = json.loads(Path("artifacts/MANIFEST.json").read_text())["items"][0]
    prov = {"label_mode": man.get("label_mode", "RULE_BASED"), "knowledge_mode": load_taxonomy()["knowledge_mode"], "detector_mode": det.mode,
            "gold_eval": "AVAILABLE" if has_gold() else "UNAVAILABLE",
            "models": {"classifier": f"{cfg['backbone_run']} backbone + {cfg['p4_dir']}/fusion.pt", "detector": cfg["detector_weights"],
                       "parts_model": f"{cfg['p4_dir']}/parts_gb.pkl"},
            "data_hash": man.get("data_hash"), "git_commit": git_commit()}
    return Bundle(net, det, gb, None, dev, prov)
