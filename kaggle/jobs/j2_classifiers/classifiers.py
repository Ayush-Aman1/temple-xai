"""J2: style classifiers. Baselines (EfficientNet-B0, MobileNetV3) and ablation variants A-D x seeds.

A  image CNN only (both backbones)            B  CNN + component features (J1 zero-shot boxes)
C  B + knowledge-prior features               D  C + evidence-alignment loss (KL to the knowledge prior)
Temple-disjoint split from splits.csv only; exterior images only for training/eval. Same file runs on Kaggle
(cuda) and on the Mac (mps/cpu): --device auto. Writes <out>/<variant>_<backbone>_s<seed>/{metrics.json,best.pt,
predictions.csv,confusion.png} and <out>/summary.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

for p in [Path(__file__).resolve().parents[3], *Path("/kaggle/input").glob("*/code")]:
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

if Path("/kaggle").exists():  # pin library versions to the ones the code was written against
    import subprocess

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm==1.0.30", "pyyaml"], check=False)

TRAD = ["Nagara", "Dravida", "Deccan_Vesara"]


def pip_install(pkgs: list[str]) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)


def find_root(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    for c in Path("/kaggle/input").glob("*"):
        if (c / "splits.csv").exists():
            return c
    return Path("data")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default="/kaggle/working/out")
    ap.add_argument("--boxes", default=None, help="boxes_zeroshot.json from J1 (default: <data-root>/derived/boxes_zeroshot.json)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--compute-label", default="kaggle_gpu")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    import torch
    import torch.nn as nn
    import timm
    from PIL import Image
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as T
    import yaml
    from templexai.device import pick_device
    from templexai.models.component_features import DIM, component_vector, knowledge_prior
    from templexai.seed import seed_everything

    root = find_root(a.data_root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(a.device)
    print("device", device, "root", root, flush=True)

    sp = pd.read_csv(root / "splits.csv")
    imgs = pd.read_csv(root / "meta" / "images.csv")
    imgs = imgs[imgs.image_type.astype(str).str.startswith("exterior")] if "image_type" in imgs else imgs
    df = imgs.merge(sp, on="temple_id")
    df["y"] = df.tradition.map({t: i for i, t in enumerate(TRAD)})
    boxes_path = Path(a.boxes) if a.boxes else next(iter([*Path("/kaggle/input").glob("*/boxes_zeroshot.json"), root / "derived" / "boxes_zeroshot.json"]), root / "derived" / "boxes_zeroshot.json")
    boxes = json.loads(boxes_path.read_text()) if boxes_path.exists() else {}
    thr = boxes.get("_thresholds", {}) if boxes else {}
    support = {}
    for cand in [root / "derived" / "knowledge_support.yaml", Path("configs/knowledge_support.yaml")]:
        if cand.exists():
            support = yaml.safe_load(cand.read_text())["support"]
            break
    if a.smoke:
        df = pd.concat([df[df.split == sp].head(20) for sp in ("train", "val", "test")]).reset_index(drop=True)

    def feats(row) -> np.ndarray:
        rec = boxes.get(row.image_id)
        if not rec:
            return np.zeros(DIM, dtype=np.float32)
        return component_vector(rec["detections"], rec["width"], rec["height"], thr)

    F = np.stack([feats(r) for r in df.itertuples()]) if len(df) else np.zeros((0, DIM), dtype=np.float32)
    P = np.stack([knowledge_prior(f, support) for f in F]).astype(np.float32) if len(df) and support else np.full((len(df), 3), 1 / 3, dtype=np.float32)
    df = df.reset_index(drop=True)

    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    S = a.img_size
    tf_train = T.Compose([T.RandomResizedCrop(S, scale=(0.6, 1.0)), T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), norm])
    tf_eval = T.Compose([T.Resize(S + 32), T.CenterCrop(S), T.ToTensor(), norm])

    class DS(Dataset):
        def __init__(self, idx, train):
            self.idx, self.tf = idx, tf_train if train else tf_eval

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            j = self.idx[i]
            im = Image.open(root / df.file[j]).convert("RGB")
            return self.tf(im), torch.from_numpy(F[j]), torch.from_numpy(P[j]), int(df.y[j]), j

    class Net(nn.Module):
        def __init__(self, backbone, variant):
            super().__init__()
            self.variant = variant
            self.bb = timm.create_model(backbone, pretrained=True, num_classes=0)
            self.bb.eval()
            with torch.no_grad():
                d = int(self.bb(torch.zeros(2, 3, 64, 64)).shape[1])  # real pooled size (num_features is wrong for MobileNetV3)
            self.bb.train()
            extra = DIM if variant in "BCD" else 0
            extra += 3 if variant in "CD" else 0
            self.head = nn.Sequential(nn.Linear(d + extra, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 3)) if extra else nn.Linear(d, 3)

        def forward(self, x, f, p):
            h = self.bb(x)
            if self.variant in "BCD":
                parts = [h, f] + ([p] if self.variant in "CD" else [])
                h = torch.cat(parts, 1)
            return self.head(h)

    tr = np.where(df.split == "train")[0]
    va = np.where(df.split == "val")[0]
    te = np.where(df.split == "test")[0]
    assert not (set(df.temple_id[tr]) & set(df.temple_id[va])) and not (set(df.temple_id[tr]) & set(df.temple_id[te])) \
        and not (set(df.temple_id[va]) & set(df.temple_id[te])), "temple overlap between splits"
    print("no-overlap assertion passed:", len(tr), len(va), len(te), flush=True)
    counts = np.bincount(df.y[tr], minlength=3).astype(np.float32)
    w = torch.tensor(counts.sum() / np.maximum(counts, 1) / 3, dtype=torch.float32).to(device)

    def evaluate(model, idx):
        model.eval()
        ys, ps, probs, ids = [], [], [], []
        with torch.no_grad():
            for x, f, p, y, j in DataLoader(DS(idx, False), batch_size=a.batch, num_workers=a.workers):
                lo = model(x.to(device), f.to(device), p.to(device))
                pr = lo.softmax(-1).cpu()
                ys += y.tolist(); ps += pr.argmax(1).tolist(); probs += pr.tolist(); ids += j.tolist()
        return np.array(ys), np.array(ps), np.array(probs), ids

    def metrics(y, p):
        if len(y) == 0:
            return {"n": 0}
        return {"n": int(len(y)), "accuracy": float(accuracy_score(y, p)),
                "macro_f1": float(f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0)),
                "weighted_f1": float(f1_score(y, p, average="weighted", labels=[0, 1, 2], zero_division=0)),
                "per_class": classification_report(y, p, labels=[0, 1, 2], target_names=TRAD, output_dict=True, zero_division=0),
                "confusion": confusion_matrix(y, p, labels=[0, 1, 2]).tolist()}

    runs = [("A", "efficientnet_b0"), ("A", "mobilenetv3_large_100"), ("B", "efficientnet_b0"),
            ("C", "efficientnet_b0"), ("D", "efficientnet_b0")]
    summary = []
    for variant, backbone in runs:
        for seed in a.seeds:
            name = f"{variant}_{backbone}_s{seed}"
            rd = out / name
            if (rd / "metrics.json").exists():
                summary.append(json.loads((rd / "metrics.json").read_text())["summary"])
                continue
            rd.mkdir(parents=True, exist_ok=True)
            seed_everything(seed)
            t0 = time.time()
            model = Net(backbone, variant).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.epochs))
            crit = nn.CrossEntropyLoss(weight=w)
            best, best_f1, bad, log = None, -1.0, 0, []
            for ep in range(a.epochs):
                model.train()
                tot = 0.0
                for x, f, p, y, _ in DataLoader(DS(tr, True), batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=len(tr) > a.batch):
                    x, f, p, y = x.to(device), f.to(device), p.to(device), y.to(device)
                    lo = model(x, f, p)
                    loss = crit(lo, y)
                    if variant == "D":  # evidence alignment: pull predictions toward the knowledge prior where evidence exists
                        has = (p.max(1).values > 1 / 3 + 1e-3).float()
                        kl = (p * (torch.log(p.clamp_min(1e-8)) - torch.log_softmax(lo, -1))).sum(1)
                        loss = loss + 0.2 * (kl * has).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot += float(loss) * len(y)
                sched.step()
                yv, pv, _, _ = evaluate(model, va)
                f1v = float(f1_score(yv, pv, average="macro", labels=[0, 1, 2], zero_division=0)) if len(yv) else 0.0
                log.append({"epoch": ep, "train_loss": tot / max(1, len(tr)), "val_macro_f1": f1v})
                print(name, log[-1], flush=True)
                if f1v > best_f1:
                    best_f1, bad = f1v, 0
                    best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 3:
                        break
            model.load_state_dict(best)
            torch.save(best, rd / "best.pt")
            yt, pt, probs, ids = evaluate(model, te)
            yv, pv, _, _ = evaluate(model, va)
            pd.DataFrame({"image_id": df.image_id[ids].values, "temple_id": df.temple_id[ids].values, "y": yt, "pred": pt,
                          **{f"p_{t}": probs[:, i] for i, t in enumerate(TRAD)}}).to_csv(rd / "predictions.csv", index=False)
            summ = {"run": name, "variant": variant, "backbone": backbone, "seed": seed, "best_val_macro_f1": best_f1,
                    "test": {k: v for k, v in metrics(yt, pt).items() if k in ("n", "accuracy", "macro_f1", "weighted_f1")},
                    "device": device, "compute": a.compute_label, "epochs_run": len(log), "seconds": round(time.time() - t0, 1),
                    "img_size": S, "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
            (rd / "metrics.json").write_text(json.dumps({"summary": summ, "test": metrics(yt, pt), "val": metrics(yv, pv), "log": log}, indent=1))
            summary.append(summ)
            try:
                import matplotlib; matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                cm = np.array(metrics(yt, pt).get("confusion", np.zeros((3, 3))))
                fig, ax = plt.subplots(figsize=(3.5, 3)); ax.imshow(cm, cmap="Blues")
                ax.set_xticks(range(3), TRAD, rotation=30); ax.set_yticks(range(3), TRAD)
                for i in range(3):
                    for j in range(3):
                        ax.text(j, i, cm[i, j], ha="center", va="center")
                fig.tight_layout(); fig.savefig(rd / "confusion.png", dpi=120); plt.close(fig)
            except Exception as e:  # plotting must never fail a run
                print("plot skipped", e)
            del model
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("J2 done:", len(summary), "runs")


if __name__ == "__main__":
    main()
