"""P8 ablation: A-D x 3 seeds (J2 runs), mean +- std, same temple-disjoint split; plus the P4 frozen-feature heads.
Reads only metrics files written by code. Output: experiments/results/ablation.csv, ablation.md, ablation.png"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("experiments/results")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s = json.loads(Path("artifacts/raw/j2_classifiers/summary.json").read_text())
    rows = []
    for r in s:
        rows.append({"model": f"{r['variant']} {r['backbone']}", "variant": r["variant"], "backbone": r["backbone"], "seed": r["seed"], "acc": r["test"]["accuracy"],
                     "macro_f1": r["test"]["macro_f1"], "val_macro_f1": r["best_val_macro_f1"]})
    d = pd.DataFrame(rows)
    g = d.groupby(["variant", "backbone"]).agg(acc_mean=("acc", "mean"), acc_std=("acc", "std"), f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"), n_seeds=("seed", "count")).reset_index()
    p4 = Path("artifacts/p4/metrics.json")
    extra = []
    if p4.exists():
        m = json.loads(p4.read_text())
        for k in ("cnn_only", "fusion", "parts_gb", "parts_mlp"):
            if k in m:
                extra.append({"variant": "P4", "backbone": k, "acc_mean": m[k]["mean_test_acc"], "acc_std": m[k]["std_test_acc"], "f1_mean": m[k]["mean_test_macro_f1"],
                              "f1_std": m[k]["std_test_macro_f1"], "n_seeds": len(m[k]["seeds"])})
    g = pd.concat([g, pd.DataFrame(extra)], ignore_index=True)
    g.to_csv(OUT / "ablation.csv", index=False)
    d.to_csv(OUT / "ablation_per_seed.csv", index=False)
    L = ["# Ablation (test split, temple-disjoint; mean ± std over seeds)", "",
         "A image CNN; B +component features (zero-shot boxes); C +knowledge-prior features; D +evidence-alignment loss. P4 rows: heads trained on cached frozen features "
         "(`cnn_only`, `fusion` = CNN ⊕ distilled-YOLO component vector, `parts_*` = components only).", "",
         "| variant | backbone / head | seeds | accuracy | macro-F1 |", "|---|---|---|---|---|"]
    for r in g.itertuples():
        L.append(f"| {r.variant} | {r.backbone} | {r.n_seeds} | {r.acc_mean:.3f} ± {0 if np.isnan(r.acc_std) else r.acc_std:.3f} | {r.f1_mean:.3f} ± {0 if np.isnan(r.f1_std) else r.f1_std:.3f} |")
    (OUT / "ablation.md").write_text("\n".join(L) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.2))
    lab = [f"{r.variant}\n{r.backbone[:12]}" for r in g.itertuples()]
    ax.bar(range(len(g)), g.f1_mean, yerr=g.f1_std.fillna(0), capsize=3)
    ax.set_xticks(range(len(g)), lab, fontsize=6); ax.set_ylabel("test macro-F1"); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(OUT / "ablation.png", dpi=130)
    print((OUT / "ablation.md").read_text())


if __name__ == "__main__":
    main()
