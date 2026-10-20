"""Run the explanation evaluation on the TEST split -> experiments/results/explanations/.

Ground-truth-free metrics always; gold-based metrics only if data/boxes/gold/*.txt exists (else status = unavailable).
Image-branch explanations: the component vector is held fixed while pixels are perturbed (documented in docs/METHOD.md).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from templexai.data.load import exterior_frame
from templexai.detect.cached import CachedBoxes
from templexai.eval import explain_metrics as M
from templexai.knowledge.taxonomy import diagnostic_links, load_taxonomy
from templexai.models.component_features import COMPONENTS, PER_COMPONENT, TRADITIONS, component_vector
from templexai.models.serving import load_bundle
from templexai.seed import seed_everything
from templexai.xai.geometry import EVAL_TF, IMG, box_mask, box_to_crop
from templexai.xai.gradcam import compute_cam_batch
from templexai.xai.shap_components import component_shap

OUT = Path("experiments/results/explanations")


def best_boxes(dets, thr, w, h):
    """Best box per component above its threshold, in 192-crop coords. -> {label: (box_crop, score)}."""
    out = {}
    for d in dets:
        if d["score"] < thr.get(d["label"], 0.0):
            continue
        if d["label"] not in out or d["score"] > out[d["label"]][1]:
            out[d["label"]] = (box_to_crop(d["box_xyxy"], w, h), d["score"])
    return {k: v for k, v in out.items() if (v[0][2] - v[0][0]) * (v[0][3] - v[0][1]) > 4}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    seed_everything(a.seed)
    rng = np.random.default_rng(a.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    B = load_bundle(device=a.device)
    dev, net = B.device, B.net
    tax = load_taxonomy()
    links = diagnostic_links(tax)
    df = exterior_frame("data")
    df = df[df.split == "test"].reset_index(drop=True)
    if a.limit:
        df = df.head(a.limit)
    cb = CachedBoxes("data/derived/boxes_yolo.json")
    mean_pix = torch.zeros(3, IMG, IMG).numpy()  # grey = ImageNet mean in normalised space

    def prob_fn_factory(comp_t, cls):
        def f(batch: np.ndarray) -> np.ndarray:
            net.comp = comp_t
            with torch.no_grad():
                return torch.softmax(net(torch.tensor(batch).to(dev)), -1)[:, cls].cpu().numpy()
        return f

    rows, occl_rows, shap_rows, curves = [], [], [], {"del_cam": [], "ins_cam": [], "del_rand": [], "ins_rand": [], "shap_del": [], "shap_rand": []}
    gb_X, gb_ids = [], []
    for k, r in enumerate(df.itertuples()):
        im = Image.open(Path("data") / r.file).convert("RGB")
        w, h = im.size
        rec = cb.boxes[r.image_id]
        cvec = component_vector(rec["detections"], rec["width"], rec["height"], cb.thresholds)
        comp_t = torch.tensor(cvec[None], dtype=torch.float32)
        x = EVAL_TF(im)[None]
        net.comp = comp_t
        with torch.no_grad():
            P = torch.softmax(net(x.to(dev)), -1)[0].cpu().numpy()
        cls = int(P.argmax())
        cam = compute_cam_batch(net, x.to(dev), comp_t, [cls])[0]
        xn = x[0].numpy()
        pf = prob_fn_factory(comp_t, cls)
        # random saliency at the CAM's own granularity
        rnd = np.random.default_rng(rng.integers(1 << 31)).random((6, 6)).astype(np.float32)
        import cv2
        rsal = cv2.resize(rnd, (IMG, IMG), interpolation=cv2.INTER_LINEAR)
        blur = cv2.GaussianBlur(xn.transpose(1, 2, 0), (0, 0), 12).transpose(2, 0, 1)
        dc, ic = M.deletion_insertion_curves(pf, xn, cam, mean_pix, blur)
        dr, ir = M.deletion_insertion_curves(pf, xn, rsal, mean_pix, blur)
        for key, v in zip(("del_cam", "ins_cam", "del_rand", "ins_rand"), (dc, ic, dr, ir)):
            curves[key].append(v)
        row = {"image_id": r.image_id, "temple_id": r.temple_id, "y": int(r.y), "pred": cls, "p_pred": float(P[cls]),
               "del_auc_cam": M.auc(dc), "ins_auc_cam": M.auc(ic), "del_auc_rand": M.auc(dr), "ins_auc_rand": M.auc(ir)}
        # component boxes + occlusion
        bb = best_boxes(rec["detections"], cb.thresholds, rec["width"], rec["height"])
        sc = {c: (v[0], v[1]) for c, v in bb.items()}
        occl_imgs, meta = [], []
        for c, (box, score) in sc.items():
            for kind, bx in (("component", box), ("random", M.equal_area_random_box(box, rng))):
                z = xn.copy()
                m = box_mask(bx)
                z[:, m] = 0.0
                occl_imgs.append(z); meta.append((c, kind))
        if occl_imgs:
            pr = pf(np.stack(occl_imgs).astype(np.float32))
            for (c, kind), p in zip(meta, pr):
                occl_rows.append({"image_id": r.image_id, "temple_id": r.temple_id, "component": c, "kind": kind, "pred": TRADITIONS[cls],
                                  "delta": float(P[cls] - p), "linked_to_pred": TRADITIONS[cls] in links.get(c, set())})
        # CAM vs detector agreement
        allm = np.zeros((IMG, IMG), dtype=bool)
        for c, (box, _) in sc.items():
            allm |= box_mask(box)
        linked = [v[0] for c, v in sc.items() if TRADITIONS[cls] in links.get(c, set())]
        row.update({"n_components": len(sc), "cam_energy_in_boxes": float(cam[allm].sum() / (cam.sum() + 1e-12)) if allm.any() else np.nan,
                    "box_area_share": float(allm.mean()), "point_in_any_box": bool(M.pointing_hit(cam, [v[0] for v in sc.values()])) if sc else np.nan,
                    "point_in_linked_box": bool(M.pointing_hit(cam, linked)) if linked else np.nan,
                    "linked_area_share": float(np.logical_or.reduce([box_mask(b) for b in linked]).mean()) if linked else np.nan})
        rows.append(row)
        gb_X.append(cvec); gb_ids.append((r.image_id, r.temple_id))
        if k % 100 == 0:
            print(k, len(df), flush=True)

    ex = pd.DataFrame(rows)
    ex.to_csv(OUT / "per_image.csv", index=False)
    pd.DataFrame(occl_rows).to_csv(OUT / "occlusion_per_component.csv", index=False)

    # ---- SHAP (components-only gradient boosting) : alignment + deletion
    Xg = np.stack(gb_X)
    Pg = B.gb.predict_proba(Xg)
    clsg = Pg.argmax(1)
    kf = len(PER_COMPONENT)
    for i in range(len(Xg)):
        present = [c for j, c in enumerate(COMPONENTS) if Xg[i, j * kf] > 0.5]
        if not present:
            continue
        sh = component_shap(B.gb.predict_proba, Xg[i], int(clsg[i]))
        top = max(present, key=lambda c: sh[c])
        tg = TRADITIONS[clsg[i]]
        linked_present = [c for c in present if tg in links.get(c, set())]
        shap_rows.append({"image_id": gb_ids[i][0], "temple_id": gb_ids[i][1], "pred_parts": tg, "n_present": len(present), "top_shap_component": top,
                          "top_is_linked": tg in links.get(top, set()), "random_expectation": len(linked_present) / len(present),
                          "any_linked_present": bool(linked_present)})
        # SHAP deletion: zero components in descending SHAP order vs random order
        order_s = sorted(present, key=lambda c: -sh[c])
        order_r = list(rng.permutation(present))
        for key, order in (("shap_del", order_s), ("shap_rand", order_r)):
            ps = [Pg[i, clsg[i]]]
            v = Xg[i].copy()
            for c in order:
                j = COMPONENTS.index(c)
                v[j * kf:(j + 1) * kf] = 0
                ps.append(B.gb.predict_proba(v[None])[0, clsg[i]])
            xs = np.linspace(0, 1, len(ps))
            curves[key].append(np.interp(M.FRACS, xs, ps))
    sh_df = pd.DataFrame(shap_rows)
    sh_df.to_csv(OUT / "shap_alignment_per_image.csv", index=False)

    # ---- aggregate
    tid = ex.temple_id.values

    def ci(v, ids=tid):
        v = np.asarray(v, dtype=float)
        ok = ~np.isnan(v)
        m, lo, hi = M.temple_bootstrap(v[ok], np.asarray(ids)[ok])
        return {"mean": m, "ci95": [lo, hi], "n": int(ok.sum())}

    summ = {"n_images": len(ex), "n_temples": int(ex.temple_id.nunique()), "compute": dev, "split": "test", "gold_eval": "UNAVAILABLE",
            "faithfulness": {"deletion_auc_gradcam": ci(ex.del_auc_cam), "deletion_auc_random": ci(ex.del_auc_rand),
                             "insertion_auc_gradcam": ci(ex.ins_auc_cam), "insertion_auc_random": ci(ex.ins_auc_rand),
                             "deletion_gradcam_lower_than_random_share": float((ex.del_auc_cam < ex.del_auc_rand).mean()),
                             "insertion_gradcam_higher_than_random_share": float((ex.ins_auc_cam > ex.ins_auc_rand).mean()),
                             "note": "deletion: lower AUC = more faithful; insertion: higher = more faithful. Probability of the originally predicted class."}}
    oc = pd.DataFrame(occl_rows)
    if len(oc):
        comp = oc[oc.kind == "component"].reset_index(drop=True)
        rnd = oc[oc.kind == "random"].reset_index(drop=True)
        diff = comp.delta.values - rnd.delta.values
        summ["component_occlusion"] = {"mean_drop_component": ci(comp.delta, comp.temple_id), "mean_drop_random_equal_area": ci(rnd.delta, rnd.temple_id),
                                       "paired_difference": ci(diff, comp.temple_id), "component_drop_gt_random_share": float((diff > 0).mean()),
                                       "mean_drop_if_component_linked_to_prediction": float(comp[comp.linked_to_pred].delta.mean()) if comp.linked_to_pred.any() else None,
                                       "mean_drop_if_not_linked": float(comp[~comp.linked_to_pred].delta.mean()) if (~comp.linked_to_pred).any() else None}
        pivot = comp.pivot_table(index="component", columns="pred", values="delta", aggfunc=["mean", "count"])
        pivot.to_csv(OUT / "occlusion_by_component_x_tradition.csv")
    if len(sh_df):
        ok = sh_df[sh_df.any_linked_present]
        summ["shap_knowledge_alignment"] = {"top_shap_component_is_linked_to_predicted_tradition": ci(ok.top_is_linked.astype(float), ok.temple_id),
                                            "random_component_baseline": ci(ok.random_expectation, ok.temple_id), "n_images_with_linked_component": int(len(ok)),
                                            "n_images_with_components": int(len(sh_df)),
                                            "note": "SHAP of the components-only gradient-boosting model; baseline = chance that a random present component is linked."}
        summ["shap_component_deletion"] = {"auc_by_shap_order": float(np.mean([M.auc(c) for c in curves["shap_del"]])),
                                           "auc_random_order": float(np.mean([M.auc(c) for c in curves["shap_rand"]])),
                                           "note": "parts-model class probability as components are removed; lower AUC = SHAP order is more faithful."}
    pm = ex.dropna(subset=["point_in_any_box"])
    pl = ex.dropna(subset=["point_in_linked_box"])
    en = ex.dropna(subset=["cam_energy_in_boxes"])
    summ["gradcam_detector_agreement"] = {"note": "AGREEMENT of Grad-CAM with the detector's boxes, not correctness (no gold boxes).",
                                          "cam_peak_in_any_detected_box": ci(pm.point_in_any_box.astype(float), pm.temple_id), "box_area_share_chance": ci(pm.box_area_share, pm.temple_id),
                                          "cam_peak_in_linked_box": ci(pl.point_in_linked_box.astype(float), pl.temple_id), "linked_area_share_chance": ci(pl.linked_area_share, pl.temple_id),
                                          "cam_energy_in_boxes": ci(en.cam_energy_in_boxes, en.temple_id), "box_area_share_for_energy": ci(en.box_area_share, en.temple_id)}
    # ---- gold hook
    gold = M.load_gold()
    if gold:
        summ["gold_eval"] = "AVAILABLE"
        summ["gold"] = run_gold(gold, df, ex, B, cb)
    else:
        summ["gold"] = {"status": "available with the gold-set add-on (HUMAN_ADDONS.md): pointing game, energy pointing, explanation IoU, detector mAP vs gold boxes"}
    summ["provenance"] = B.provenance
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    plot(curves, summ)
    write_md(summ)
    print("explanations done", OUT)


def run_gold(gold, df, ex, B, cb) -> dict:
    """Gold-based metrics: detector mAP on gold-boxed test images; pointing / energy / IoU for CAM vs diagnostic gold boxes."""
    from templexai.xai.gradcam import compute_cam_batch  # noqa: F401

    ids = [i for i in df.image_id if i in gold]
    if not ids:
        return {"status": "gold directory present but no gold-boxed test images matched"}
    preds, gts = [], {}
    for i in ids:
        rec = cb.boxes[i]
        w, h = rec["width"], rec["height"]
        gts[i] = [{"label": g["label"], "box": [g["box"][0] * w, g["box"][1] * h, g["box"][2] * w, g["box"][3] * h]} for g in gold[i]]
        preds += [{"image": i, "label": d["label"], "box": d["box_xyxy"], "score": d["score"]} for d in rec["detections"]]
    det = M.detector_map(preds, gts, COMPONENTS)
    return {"n_gold_images": len(ids), "detector_vs_gold": det}


def plot(curves, summ) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    for key, lab in (("del_cam", "Grad-CAM"), ("del_rand", "random")):
        ax[0].plot(M.FRACS, np.mean(curves[key], 0), label=lab)
    ax[0].set_title("Deletion (lower is better)"); ax[0].set_xlabel("fraction of pixels removed"); ax[0].set_ylabel("p(predicted class)"); ax[0].legend()
    for key, lab in (("ins_cam", "Grad-CAM"), ("ins_rand", "random")):
        ax[1].plot(M.FRACS, np.mean(curves[key], 0), label=lab)
    ax[1].set_title("Insertion (higher is better)"); ax[1].set_xlabel("fraction of pixels inserted"); ax[1].legend()
    if curves["shap_del"]:
        for key, lab in (("shap_del", "SHAP order"), ("shap_rand", "random order")):
            ax[2].plot(M.FRACS, np.mean(curves[key], 0), label=lab)
        ax[2].set_title("Component deletion, parts model"); ax[2].set_xlabel("fraction of components removed"); ax[2].legend()
    fig.tight_layout(); fig.savefig(OUT / "faithfulness_curves.png", dpi=130); plt.close(fig)


def write_md(s) -> None:
    f = s["faithfulness"]
    fmt = lambda d: f"{d['mean']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"  # noqa: E731
    L = [f"# Explanation evaluation (test split, {s['n_images']} images, {s['n_temples']} temples; gold_eval={s['gold_eval']})", "",
         "Numbers below are copied from `summary.json`; intervals are 95 % temple-bootstrap CIs. All metrics are ground-truth-free.", "",
         "## Faithfulness of Grad-CAM (image branch)", "",
         f"- Deletion AUC (lower = better): Grad-CAM {fmt(f['deletion_auc_gradcam'])} vs random {fmt(f['deletion_auc_random'])}; Grad-CAM lower in {f['deletion_gradcam_lower_than_random_share']:.0%} of images.",
         f"- Insertion AUC (higher = better): Grad-CAM {fmt(f['insertion_auc_gradcam'])} vs random {fmt(f['insertion_auc_random'])}; Grad-CAM higher in {f['insertion_gradcam_higher_than_random_share']:.0%} of images.", ""]
    if "component_occlusion" in s:
        o = s["component_occlusion"]
        L += ["## Component-occlusion sensitivity", "",
              f"- Mean drop in p(predicted class) when a detected component box is greyed out: {fmt(o['mean_drop_component'])}; equal-area random box: {fmt(o['mean_drop_random_equal_area'])}.",
              f"- Paired difference (component - random): {fmt(o['paired_difference'])}; component larger in {o['component_drop_gt_random_share']:.0%} of pairs.",
              "- Per component x tradition: `occlusion_by_component_x_tradition.csv`.", ""]
    if "shap_knowledge_alignment" in s:
        k = s["shap_knowledge_alignment"]
        d = s["shap_component_deletion"]
        L += ["## SHAP - knowledge-graph alignment (components-only model)", "",
              f"- Top-SHAP component is cited as evidence for the predicted tradition: {fmt(k['top_shap_component_is_linked_to_predicted_tradition'])} vs random-component chance {fmt(k['random_component_baseline'])} ({k['n_images_with_linked_component']} images).",
              f"- Component-deletion AUC by SHAP order {d['auc_by_shap_order']:.3f} vs random order {d['auc_random_order']:.3f} (lower = SHAP order more faithful).", ""]
    g = s["gradcam_detector_agreement"]
    L += ["## Grad-CAM / detector agreement (not correctness)", "",
          f"- CAM peak inside any detected component box: {fmt(g['cam_peak_in_any_detected_box'])} (box area share, i.e. chance: {fmt(g['box_area_share_chance'])}).",
          f"- CAM peak inside a box of a component linked to the predicted tradition: {fmt(g['cam_peak_in_linked_box'])} (chance: {fmt(g['linked_area_share_chance'])}).",
          f"- Share of CAM energy inside detected boxes: {fmt(g['cam_energy_in_boxes'])} (area share: {fmt(g['box_area_share_for_energy'])}).", "",
          "## Gold-based metrics", "", str(s["gold"] if isinstance(s["gold"], dict) and "status" not in s["gold"] else s["gold"]["status"] if "status" in s["gold"] else s["gold"]), ""]
    (OUT / "explanations.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
