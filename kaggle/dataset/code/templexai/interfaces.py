"""Shared contracts: StyleModel and ComponentDetector."""
from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable

TRADITIONS = ["Nagara", "Dravida", "Deccan_Vesara"]
COMPONENTS = ["shikhara", "vimana", "gopuram", "mandapa", "pillar", "jagati_base", "amalaka"]


class Detection(TypedDict):
    label: str
    box_xyxy: list[float]
    score: float


@runtime_checkable
class StyleModel(Protocol):
    """Classifier contract."""

    target_layer: object

    def predict(self, img) -> "list[float]":
        """Return class probabilities in TRADITIONS order."""

    def features(self, img):
        """Return the pooled feature tensor."""


@runtime_checkable
class ComponentDetector(Protocol):
    """Detector contract; implementations: zeroshot, yolo (selected by config only)."""

    def detect(self, img) -> list[Detection]:
        """Return detections for one image."""
