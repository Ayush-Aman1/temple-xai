"""Seed every RNG we use."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> int:
    """Seed python, numpy and torch (if importable)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass
    return seed
