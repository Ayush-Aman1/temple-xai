"""a6: build kaggle/dataset/ (images + meta + splits + labels + dataset-metadata.json, private).

Only images of temples that are in splits.csv are packaged. Slug is templexai-data.
The Kaggle username is read from the CLI config at run time (never the token).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import yaml

from .common import add_data_root_arg


def kaggle_username() -> str | None:
    """Username from KAGGLE_USERNAME, kaggle.json, or `kaggle config view`. Never reads the secret key."""
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    p = Path.home() / ".kaggle" / "kaggle.json"
    if p.exists():
        try:
            return json.loads(p.read_text()).get("username")
        except Exception:
            return None
    import subprocess
    try:
        out = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True, timeout=30).stdout
        for line in out.splitlines():
            if "username" in line.lower():
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_data_root_arg(ap)
    ap.add_argument("--out", default="kaggle/dataset")
    a = ap.parse_args()
    root, out = Path(a.data_root), Path(a.out)
    if a.smoke:
        print("a6: smoke mode - packaging skipped (no uploads in smoke)")
        return
    sp = pd.read_csv(root / "splits.csv")
    imgs = pd.read_csv(root / "meta" / "images.csv")
    imgs = imgs[imgs.temple_id.isin(set(sp.temple_id))]
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)
    for _, r in imgs.iterrows():
        dst = out / "images" / Path(r.file).relative_to("images")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / r.file, dst)
    imgs.to_csv(out / "meta" / "images.csv", index=False)
    pd.read_csv(root / "meta" / "temples.csv").to_csv(out / "meta" / "temples.csv", index=False)
    shutil.copy2(root / "splits.csv", out / "splits.csv")
    (out / "labels").mkdir()
    shutil.copy2(root / "labels" / "temple_styles_rule.csv", out / "labels" / "temple_styles_rule.csv")
    shutil.copytree("templexai", out / "code" / "templexai", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (out / "derived").mkdir(exist_ok=True)
    shutil.copy2("configs/knowledge_support.yaml", out / "derived" / "knowledge_support.yaml")
    user = kaggle_username() or yaml.safe_load(Path("configs/compute.yaml").read_text()).get("kaggle_user")
    if not user:
        raise SystemExit("a6: Kaggle username not found")
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "templexai-data", "id": f"{user}/templexai-data",
        "licenses": [{"name": "other"}], "isPrivate": True,
        "subtitle": "Temple XAI: Commons temple images with per-image licence metadata"}, indent=1))
    print(f"a6: packaged {len(imgs)} images -> {out} as {user}/templexai-data")


if __name__ == "__main__":
    main()
