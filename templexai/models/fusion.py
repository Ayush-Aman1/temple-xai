"""P4 fusion models on cached features (no detector/backbone re-runs).

- cnn_only   : MLP on the fine-tuned image backbone's pooled features (control).
- fusion     : MLP on [CNN features ; 42-d component vector from the DISTILLED YOLO boxes]   ("Model B").
- parts_gb   : gradient boosting on the component vector alone (EXPLANet-style part-based classifier).
- parts_mlp  : small MLP on the component vector alone.
Seeds 0-2, early stopping on val macro-F1, temple-disjoint split from data/splits.csv. Outputs artifacts/p4/.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier

from templexai.data.load import exterior_frame
from templexai.detect.cached import CachedBoxes
from templexai.eval.metrics import cls_metrics, mcnemar
from templexai.models.component_features import DIM, component_vector
from templexai.seed import seed_everything

DER = Path("data/derived")


class Head(nn.Module):
    """2-layer MLP over concatenated inputs (guide P4)."""

    def __init__(self, d_in: int, hidden: int = 256, drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(0.2), nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden, 3))

    def forward(self, x):
        return self.net(x)


def load_cached(root: str = "data", run: str = "A_efficientnet_b0", boxes_file: str = "boxes_yolo.json"):
    """Returns frame df, CNN feats (N,d), component matrix (N,42) built with the cached detector's thresholds."""
    df = exterior_frame(root)
    idx = pd.read_csv(DER / "feats_index.csv").image_id.tolist()
    feats_all = np.load(DER / f"feats_{run}.npy")
    pos = {i: k for k, i in enumerate(idx)}
    feats = feats_all[[pos[i] for i in df.image_id]]
    cb = CachedBoxes(DER / boxes_file)
    comp = np.stack([component_vector(cb.boxes[i]["detections"], cb.boxes[i]["width"], cb.boxes[i]["height"], cb.thresholds) for i in df.image_id])
    return df, feats.astype(np.float32), comp.astype(np.float32)


def standardiser(x: np.ndarray):
    mu, sd = x.mean(0), x.std(0) + 1e-6
    return mu, sd


def train_head(X, y, tr, va, seed, epochs=60, lr=1e-3, device="cpu"):
    seed_everything(seed)
    mu, sd = standardiser(X[tr])
    Z = torch.tensor((X - mu) / sd, dtype=torch.float32)
    Y = torch.tensor(y)
    m = Head(X.shape[1]).to(device)
    counts = np.bincount(y[tr], minlength=3).astype(np.float32)
    w = torch.tensor(counts.sum() / np.maximum(counts, 1) / 3).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
    best, best_f1, bad = None, -1.0, 0
    tri = torch.tensor(tr)
    for ep in range(epochs):
        m.train()
        perm = tri[torch.randperm(len(tri))]
        for k in range(0, len(perm), 128):
            b = perm[k:k + 128]
            loss = nn.functional.cross_entropy(m(Z[b].to(device)), Y[b].to(device), weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pv = m(Z[va].to(device)).argmax(1).cpu().numpy()
        f1 = cls_metrics(y[va], pv)["macro_f1"]
        if f1 > best_f1:
            best_f1, bad, best = f1, 0, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    m.load_state_dict(best)
    m.eval()
    return m.cpu(), (mu, sd), best_f1


def predict_head(m, norm, X):
    mu, sd = norm
    with torch.no_grad():
        return torch.softmax(m(torch.tensor((X - mu) / sd, dtype=torch.float32)), -1).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/p4")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df, F, C = load_cached()
    y = df.y.values
    tr, va, te = (np.where(df.split == s)[0] for s in ("train", "val", "test"))
    assert not (set(df.temple_id[tr]) & set(df.temple_id[te])) and not (set(df.temple_id[tr]) & set(df.temple_id[va])) \
        and not (set(df.temple_id[va]) & set(df.temple_id[te])), "temple overlap"
    print("no-overlap assertion passed", len(tr), len(va), len(te), flush=True)
    inputs = {"cnn_only": F, "fusion": np.hstack([F, C]), "parts_mlp": C}
    res, preds = {}, {}
    for name, X in inputs.items():
        runs = []
        for s in (0, 1, 2):
            m, norm, f1v = train_head(X, y, tr, va, s)
            P = predict_head(m, norm, X)
            runs.append({"seed": s, "val_macro_f1": f1v, "test": cls_metrics(y[te], P[te].argmax(1)), "P": P, "m": m, "norm": norm})
        sel = max(runs, key=lambda r: r["val_macro_f1"])  # selection by validation only
        torch.save({"state": sel["m"].state_dict(), "mu": sel["norm"][0], "sd": sel["norm"][1], "d_in": X.shape[1], "seed": sel["seed"]}, out / f"{name}.pt")
        res[name] = {"seeds": [{k: r[k] for k in ("seed", "val_macro_f1", "test")} for r in runs],
                     "mean_test_acc": float(np.mean([r["test"]["accuracy"] for r in runs])), "std_test_acc": float(np.std([r["test"]["accuracy"] for r in runs])),
                     "mean_test_macro_f1": float(np.mean([r["test"]["macro_f1"] for r in runs])), "std_test_macro_f1": float(np.std([r["test"]["macro_f1"] for r in runs])),
                     "selected_seed": sel["seed"]}
        preds[name] = sel["P"]
        print(name, res[name]["mean_test_acc"], res[name]["mean_test_macro_f1"], flush=True)
    seed_everything(0)
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=0)
    gbs = []
    for s in (0, 1, 2):
        g = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=s)
        wts = (np.bincount(y[tr], minlength=3).sum() / np.maximum(np.bincount(y[tr], minlength=3), 1) / 3)[y[tr]]
        g.fit(C[tr], y[tr], sample_weight=wts)
        P = g.predict_proba(C)
        gbs.append({"seed": s, "val_macro_f1": cls_metrics(y[va], P[va].argmax(1))["macro_f1"], "test": cls_metrics(y[te], P[te].argmax(1)), "P": P, "g": g})
    sel = max(gbs, key=lambda r: r["val_macro_f1"])
    pickle.dump(sel["g"], open(out / "parts_gb.pkl", "wb"))
    res["parts_gb"] = {"seeds": [{k: r[k] for k in ("seed", "val_macro_f1", "test")} for r in gbs],
                       "mean_test_acc": float(np.mean([r["test"]["accuracy"] for r in gbs])), "std_test_acc": float(np.std([r["test"]["accuracy"] for r in gbs])),
                       "mean_test_macro_f1": float(np.mean([r["test"]["macro_f1"] for r in gbs])), "std_test_macro_f1": float(np.std([r["test"]["macro_f1"] for r in gbs])),
                       "selected_seed": sel["seed"]}
    preds["parts_gb"] = sel["P"]
    # reference: original J2 model A (fine-tuned end to end) on the same test images
    pa = pd.read_csv("artifacts/j2/A_efficientnet_b0/predictions.csv").set_index("image_id")
    te_ids = df.image_id[te].tolist()
    pa = pa.loc[te_ids]
    res["A_finetuned_ref"] = {"test": cls_metrics(pa.y.values, pa.pred.values)}
    res["mcnemar_fusion_vs_cnn_only"] = mcnemar(y[te], preds["cnn_only"][te].argmax(1), preds["fusion"][te].argmax(1))
    res["mcnemar_fusion_vs_A_finetuned"] = mcnemar(y[te], pa.pred.values, preds["fusion"][te].argmax(1))
    res["compare_note"] = "mcnemar 'b' = first model right & second wrong; 'c' = first wrong & second right"
    res["detector_mode"] = "DISTILLED"
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    pd.DataFrame({"image_id": df.image_id[te].values, "temple_id": df.temple_id[te].values, "y": y[te],
                  **{f"pred_{k}": v[te].argmax(1) for k, v in preds.items()},
                  **{f"p_fusion_{t}": preds["fusion"][te][:, i] for i, t in enumerate(["Nagara", "Dravida", "Deccan_Vesara"])}}).to_csv(out / "test_predictions.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
