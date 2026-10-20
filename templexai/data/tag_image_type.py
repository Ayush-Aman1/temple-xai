"""a4: CLIP zero-shot image-type tag (exterior_full / exterior_partial / detail / interior) -> images.csv columns
image_type, image_type_score. Model: openai/clip-vit-base-patch32 (transformers CLIPModel)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from templexai.device import pick_device

from .common import add_data_root_arg

MODEL = "openai/clip-vit-base-patch32"
PROMPTS = {
    "exterior_full": "a photo of the whole exterior of a Hindu temple building showing its tower",
    "exterior_partial": "a photo of part of the outside of a Hindu temple, a wall, entrance or corner",
    "detail": "a close-up photo of a carving, sculpture or architectural detail",
    "interior": "a photo of the inside of a temple, a dark hall with pillars or a shrine",
}


def _feat(o):
    """transformers>=5 returns an output object (pooler_output = projected features); older versions a tensor."""
    return o if isinstance(o, torch.Tensor) else o.pooler_output


@torch.no_grad()
def tag(paths: list[Path], device: str, batch: int = 32) -> list[tuple[str, float]]:
    model = CLIPModel.from_pretrained(MODEL).to(device).eval()
    proc = CLIPProcessor.from_pretrained(MODEL)
    names = list(PROMPTS)
    t = proc(text=list(PROMPTS.values()), return_tensors="pt", padding=True).to(device)
    tf = _feat(model.get_text_features(**t))
    tf = tf / tf.norm(dim=-1, keepdim=True)
    out: list[tuple[str, float]] = []
    for i in range(0, len(paths), batch):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
        x = proc(images=ims, return_tensors="pt").to(device)
        f = _feat(model.get_image_features(**x))
        f = f / f.norm(dim=-1, keepdim=True)
        probs = (100 * f @ tf.T).softmax(-1).cpu()
        for p in probs:
            j = int(p.argmax())
            out.append((names[j], float(p[j])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_data_root_arg(ap)
    a = ap.parse_args()
    root = Path(a.data_root)
    df = pd.read_csv(root / "meta" / "images.csv")
    dev = pick_device("auto")
    res = tag([root / f for f in df.file], dev)
    df["image_type"] = [r[0] for r in res]
    df["image_type_score"] = [round(r[1], 4) for r in res]
    df.to_csv(root / "meta" / "images.csv", index=False)
    print("a4:", df.image_type.value_counts().to_dict(), "device", dev)


if __name__ == "__main__":
    main()
