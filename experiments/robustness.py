"""P8 robustness on the test split (exterior images): brightness/contrast, random crops, occlusion patches, blur, downscaling and
background masking. Models: image-only A (EfficientNet-B0, 3 seeds, mean ± std) and the serving fusion model (re-runs the distilled detector
on every perturbed image). Output: experiments/results/robustness.csv, robustness.md, robustness.png (macro-F1 / accuracy per condition)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter

from templexai.data.load import exterior_frame
from templexai.device import pick_device
from templexai.eval.metrics import cls_metrics
from templexai.models.component_features import component_vector
from templexai.models.net import load_run
from templexai.models.serving import load_bundle
from templexai.seed import seed_everything
from templexai.xai.geometry import EVAL_TF

OUT = Path("experiments/results")


def perturb(name: str, im: Image.Image, dets: list[dict], rng: np.random.Generator) -> Image.Image | None:
    w, h = im.size
    if name == "clean":
        return im
    if name == "brightness_0.5":
        return ImageEnhance.Brightness(im).enhance(0.5)
    if name == "brightness_1.5":
        return ImageEnhance.Brightness(im).enhance(1.5)
    if name == "contrast_0.5":
        return ImageEnhance.Contrast(im).enhance(0.5)
    if name == "contrast_1.5":
        return ImageEnhance.Contrast(im).enhance(1.5)
    if name == "random_crop_50pct":
        cw, ch = int(w * 0.7), int(h * 0.7)
        x, y = rng.integers(0, w - cw + 1), rng.integers(0, h - ch + 1)
        return im.crop((x, y, x + cw, y + ch))
    if name == "occlusion_patch_25pct":
        pw, ph = w // 2, h // 2
        x, y = rng.integers(0, w - pw + 1), rng.integers(0, h - ph + 1)
        o = im.copy()
        o.paste((124, 116, 104), (x, y, x + pw, y + ph))
        return o
    if name == "gaussian_blur":
        return im.filter(ImageFilter.GaussianBlur(radius=max(w, h) / 100))
    if name == "downscale_64px":
        return im.resize((64, int(64 * h / w)) if w >= h else (int(64 * w / h), 64)).resize((w, h))
    if name == "background_masked":
        if not dets:
            return None
        o = Image.new("RGB", im.size, (124, 116, 104))
        for d in dets:
            b = tuple(int(v) for v in d["box_xyxy"])
            o.paste(im.crop(b), b)
        return o
    raise ValueError(name)


CONDS = ["clean", "brightness_0.5", "brightness_1.5", "contrast_0.5", "contrast_1.5", "random_crop_50pct", "occlusion_patch_25pct", "gaussian_blur", "downscale_64px", "background_masked"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    seed_everything(0)
    rng = np.random.default_rng(0)
    dev = pick_device("auto")
    B = load_bundle(device=dev)
    A = [load_run(f"artifacts/raw/j2_classifiers/A_efficientnet_b0_s{s}", dev) for s in range(3)]
    df = exterior_frame("data")
    df = df[df.split == "test"].reset_index(drop=True)
    if a.limit:
        df = df.head(a.limit)
    rows = []
    store: dict[str, dict[str, list]] = {c: {"y": [], "fusion": [], "A0": [], "A1": [], "A2": []} for c in CONDS}
    for k, r in enumerate(df.itertuples()):
        im0 = Image.open(Path("data") / r.file).convert("RGB")
        d0 = [d for d in B.detector.detect(im0) if d["score"] >= B.detector.thresholds.get(d["label"], 0)]
        for c in CONDS:
            im = perturb(c, im0, d0, rng)
            if im is None:
                continue
            w, h = im.size
            dets = d0 if c == "clean" else [d for d in B.detector.detect(im) if d["score"] >= B.detector.thresholds.get(d["label"], 0)]
            x = EVAL_TF(im)[None].to(dev)
            B.net.comp = torch.tensor(component_vector(dets, w, h)[None])
            with torch.no_grad():
                store[c]["fusion"].append(int(B.net(x).argmax()))
                for s, m in enumerate(A):
                    store[c][f"A{s}"].append(int(m(x).argmax()))
            store[c]["y"].append(int(r.y))
        if k % 100 == 0:
            print(k, len(df), flush=True)
    for c in CONDS:
        y = np.array(store[c]["y"])
        for name in ("fusion", "A0", "A1", "A2"):
            m = cls_metrics(y, np.array(store[c][name]))
            rows.append({"condition": c, "model": name, "n": m["n"], "acc": m["accuracy"], "macro_f1": m["macro_f1"]})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "robustness_per_run.csv", index=False)
    d["model"] = d.model.str.replace(r"^A\d$", "A_image_only", regex=True)
    g = d.groupby(["condition", "model"]).agg(n=("n", "first"), acc_mean=("acc", "mean"), acc_std=("acc", "std"), f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std")).reset_index()
    g.to_csv(OUT / "robustness.csv", index=False)
    L = ["# Robustness (test split, exterior images)", "", "A_image_only: EfficientNet-B0, mean ± std over 3 seeds. fusion: serving model (detector re-run on each perturbed image). "
         "`background_masked` keeps only images with ≥1 detection (n shown).", "", "| condition | model | n | accuracy | macro-F1 |", "|---|---|---|---|---|"]
    for r in g.itertuples():
        sd = lambda v: "" if np.isnan(v) else f" ± {v:.3f}"  # noqa: E731
        L.append(f"| {r.condition} | {r.model} | {r.n} | {r.acc_mean:.3f}{sd(r.acc_std)} | {r.f1_mean:.3f}{sd(r.f1_std)} |")
    (OUT / "robustness.md").write_text("\n".join(L) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.4))
    for i, (m, sub) in enumerate(g.groupby("model")):
        sub = sub.set_index("condition").loc[CONDS]
        ax.bar(np.arange(len(CONDS)) + i * 0.4, sub.f1_mean, 0.4, yerr=sub.f1_std.fillna(0), label=m, capsize=2)
    ax.set_xticks(np.arange(len(CONDS)) + 0.2, CONDS, rotation=35, ha="right", fontsize=7); ax.set_ylabel("macro-F1"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "robustness.png", dpi=130)
    print("robustness done")


if __name__ == "__main__":
    main()
