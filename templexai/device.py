"""Device selection: cuda, then mps, then cpu."""
from __future__ import annotations

import torch


def pick_device(preferred: str | None = None) -> str:
    """Return a torch device string. `preferred` ("cuda"/"mps"/"cpu"/"auto"/None) is honoured if available."""
    if preferred and preferred != "auto":
        if preferred == "cuda" and not torch.cuda.is_available():
            return pick_device(None)
        if preferred == "mps" and not torch.backends.mps.is_available():
            return pick_device(None)
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
