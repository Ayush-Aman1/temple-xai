"""a5: stratified temple-group 70/15/15 split, fixed seed -> <data-root>/splits.csv. NEVER overwrites an existing file.

Only LABELLED temples that have >=1 exterior image are split; excluded/unlabelled temples are in no split.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import add_data_root_arg

SEED = 42
FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}


def split_temples(temples: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """temples: columns temple_id, tradition. Per-tradition shuffled allocation (stratified, group = temple)."""
    rng = np.random.default_rng(seed)
    rows = []
    for trad, g in temples.groupby("tradition"):
        ids = sorted(g.temple_id)
        rng.shuffle(ids)
        n = len(ids)
        n_val = max(1, round(n * FRACS["val"])) if n >= 3 else 0
        n_test = max(1, round(n * FRACS["test"])) if n >= 3 else 0
        for i, t in enumerate(ids):
            sp = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            rows.append({"temple_id": t, "tradition": trad, "split": sp})
    return pd.DataFrame(rows).sort_values("temple_id").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_data_root_arg(ap)
    a = ap.parse_args()
    root = Path(a.data_root)
    out = root / "splits.csv"
    if out.exists():
        print(f"a5: {out} already exists - NOT regenerating (integrity rule 1)")
        return
    lab = pd.read_csv(root / "labels" / "temple_styles_rule.csv")
    lab = lab[lab.status == "LABELLED"][["temple_id", "tradition"]]
    imgs = pd.read_csv(root / "meta" / "images.csv")
    ext = imgs[imgs.get("image_type", pd.Series(dtype=str)).astype(str).str.startswith("exterior")]
    if a.smoke and ext.empty:
        ext = imgs
    t = lab[lab.temple_id.isin(set(ext.temple_id))]
    sp = split_temples(t)
    sp.to_csv(out, index=False)
    print("a5:", sp.groupby(["split", "tradition"]).size().unstack(fill_value=0).to_dict())


if __name__ == "__main__":
    main()
