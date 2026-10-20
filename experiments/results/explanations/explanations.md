# Explanation evaluation (test split, 940 images, 54 temples; gold_eval=UNAVAILABLE)

Numbers below are copied from `summary.json`; intervals are 95 % temple-bootstrap CIs. All metrics are ground-truth-free.

## Faithfulness of Grad-CAM (image branch)

- Deletion AUC (lower = better): Grad-CAM 0.527 [0.496, 0.556] vs random 0.758 [0.730, 0.784]; Grad-CAM lower in 91% of images.
- Insertion AUC (higher = better): Grad-CAM 0.864 [0.841, 0.887] vs random 0.757 [0.717, 0.793]; Grad-CAM higher in 84% of images.

## Component-occlusion sensitivity

- Mean drop in p(predicted class) when a detected component box is greyed out: 0.220 [0.187, 0.255]; equal-area random box: 0.201 [0.170, 0.232].
- Paired difference (component - random): 0.019 [0.006, 0.036]; component larger in 43% of pairs.
- Per component x tradition: `occlusion_by_component_x_tradition.csv`.

## SHAP - knowledge-graph alignment (components-only model)

- Top-SHAP component is cited as evidence for the predicted tradition: 0.771 [0.729, 0.809] vs random-component chance 0.728 [0.702, 0.755] (620 images).
- Component-deletion AUC by SHAP order 0.405 vs random order 0.440 (lower = SHAP order more faithful).

## Grad-CAM / detector agreement (not correctness)

- CAM peak inside any detected component box: 0.768 [0.717, 0.819] (box area share, i.e. chance: 0.617 [0.556, 0.672]).
- CAM peak inside a box of a component linked to the predicted tradition: 0.804 [0.744, 0.854] (chance: 0.616 [0.554, 0.669]).
- Share of CAM energy inside detected boxes: 0.706 [0.651, 0.755] (area share: 0.617 [0.558, 0.672]).

## Gold-based metrics

available with the gold-set add-on (HUMAN_ADDONS.md): pointing game, energy pointing, explanation IoU, detector mAP vs gold boxes
