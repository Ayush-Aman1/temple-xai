"""Convenience loaders over data/ (splits, image metadata, exterior-only frame used by every model)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from templexai.interfaces import TRADITIONS


def exterior_frame(root: str | Path = "data") -> pd.DataFrame:
    """One row per exterior image of a labelled temple: image_id, temple_id, file, licence, author, url, tradition, split, y."""
    root = Path(root)
    sp = pd.read_csv(root / "splits.csv")
    im = pd.read_csv(root / "meta" / "images.csv")
    im = im[im.image_type.astype(str).str.startswith("exterior")]
    df = im.merge(sp, on="temple_id").reset_index(drop=True)
    df["y"] = df.tradition.map({t: i for i, t in enumerate(TRADITIONS)})
    return df
