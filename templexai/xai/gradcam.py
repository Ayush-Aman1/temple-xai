"""Grad-CAM over the serving model (fine-tuned backbone + fusion head). Library: pytorch-grad-cam (MIT).

The component vector is held fixed while CAM is computed, so the map explains the IMAGE branch of the fusion model.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class FusionNet(nn.Module):
    """backbone -> [pooled ; standardised component vector] -> Head. `comp` must be set before forward()."""

    def __init__(self, backbone: nn.Module, head: nn.Module, mu: np.ndarray, sd: np.ndarray):
        super().__init__()
        self.bb, self.head = backbone, head
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(sd, dtype=torch.float32))
        self.comp: torch.Tensor | None = None  # (B, 42) raw component vectors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.bb(x)
        z = torch.cat([h, self.comp.to(h.device).expand(h.shape[0], -1)], 1)
        return self.head((z - self.mu) / self.sd)


def compute_cam(model: FusionNet, x: torch.Tensor, comp: torch.Tensor, cls: int) -> np.ndarray:
    """x: (1,3,H,W); comp: (1,42). Returns CAM (H,W) in [0,1]."""
    model.comp = comp
    with GradCAM(model=model, target_layers=[model.bb.bn2]) as cam:
        out = cam(input_tensor=x, targets=[ClassifierOutputTarget(cls)])
    return out[0]


def compute_cam_batch(model: FusionNet, x: torch.Tensor, comp: torch.Tensor, cls: list[int]) -> np.ndarray:
    model.comp = comp
    with GradCAM(model=model, target_layers=[model.bb.bn2]) as cam:
        out = cam(input_tensor=x, targets=[ClassifierOutputTarget(c) for c in cls])
    return out
