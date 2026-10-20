"""Classifier network shared by training (kaggle/jobs/j2_classifiers) and serving: timm backbone + optional component/prior head."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from templexai.models.component_features import DIM

TRAD = ["Nagara", "Dravida", "Deccan_Vesara"]


class Net(nn.Module):
    """Same architecture as the J2 `Net` (state_dict compatible). variant A: image only; B: +components; C/D: +knowledge prior."""

    def __init__(self, backbone: str, variant: str, pretrained: bool = False):
        super().__init__()
        import timm

        self.variant = variant
        self.bb = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.bb.eval()
        with torch.no_grad():
            d = int(self.bb(torch.zeros(2, 3, 64, 64)).shape[1])
        self.d = d
        extra = DIM if variant in "BCD" else 0
        extra += 3 if variant in "CD" else 0
        self.head = nn.Sequential(nn.Linear(d + extra, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 3)) if extra else nn.Linear(d, 3)

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        return self.bb(x)

    def forward(self, x, f=None, p=None):
        h = self.bb(x)
        if self.variant in "BCD":
            h = torch.cat([h, f] + ([p] if self.variant in "CD" else []), 1)
        return self.head(h)


def load_run(run_dir: str | Path, device: str = "cpu") -> Net:
    """Load artifacts/j2/<variant>_<backbone>/best.pt (the val-selected seed). Name format: <variant>_<backbone>."""
    run_dir = Path(run_dir)
    import re

    variant, backbone = re.sub(r"_s\d+$", "", run_dir.name).split("_", 1)
    net = Net(backbone, variant)
    net.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    return net.to(device).eval()
