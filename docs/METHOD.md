# Method (agent-only build: label_mode=RULE_BASED, knowledge_mode=MACHINE_COMPILED, detector_mode=DISTILLED, gold_eval=UNAVAILABLE)

Written for the report. Numbers appear only in `artifacts/TRAINING_REPORT.md` (generated from metrics files).

## 1. Temple list
Wikidata SPARQL: items that are (instance of / subclass of)* Hindu temple (Q842402), country India (Q668), with a Commons
category (P373). Item/property IDs were checked against the Wikidata API. Fields: coordinates (P625), state, English Wikipedia
title, architectural style (P149). Code: `templexai/data/wikidata_temples.py`.

## 2. Style labels (RULE_BASED protocol)
Evidence per temple, each stored with URL, raw value and retrieval date (`data/labels/temple_styles_rule.csv`, column `evidence_json`):
1. Wikidata P149 label(s);
2. English-Wikipedia infobox `architecture` / `architecture_style` value (via MediaWiki API, redirects resolved);
3. Commons category names (parents and, for names containing "architecture", grandparents) - only names that contain
   "architecture", "style" or "ratna" are considered, so location categories never act as evidence.

Mapping to the operational taxonomy (Nagara / Dravida / Deccan-Vesara) is `configs/style_map.yaml`. A rule is kept only if the
Wikipedia page named in the rule, fetched at run time, contains a verification keyword; the quoted sentence (<= 25 words) is in
`docs/style_map_evidence.md`. Rules whose page did not support them (Nayak, Vijayanagara) are dropped, i.e. give no evidence.
Generic values ("Hindu temple architecture", "rock-cut architecture") give no evidence. Values of regional styles outside the
three-class taxonomy (Bengal terracotta ratna/chala, Kerala, Jain, Mughal ...) map to EXCLUDE.

Decision rule: at least one mapped source; all mapped sources agree -> labelled; any disagreement or any EXCLUDE source -> temple
excluded; no mapped source -> unlabelled. The rule is never relaxed. A human file `data/labels/temple_styles_human.csv`
(columns temple_id, tradition, source) overrides per temple and sets label_mode=HUMAN_VERIFIED.

Limits (stated in the report): labels are an operational taxonomy built from open sources, not scholarly ground truth; class
counts are imbalanced (Dravida dominates), handled by class-weighted loss and macro-F1.

## 3. Images
Commons API, per temple category (+ up to 30 direct sub-categories). Licences kept: public domain, CC0, CC BY, CC BY-SA (any
version; NC/ND/GFDL-only rejected). <= 40 per temple, longest side 1024 px, perceptual-hash (pHash) dedupe at distance <= 6.
Every image keeps source URL, licence, author, sha1. Image type (exterior_full / exterior_partial / detail / interior) by CLIP
ViT-B/32 zero-shot; training and evaluation use exterior images only.

## 4. Split
Stratified temple-group split 70/15/15, seed 42, written once to `data/splits.csv`, never regenerated. Every training script
asserts no temple_id overlap before training.

## 5. Zero-shot components (J1)
Grounding DINO tiny, ONE text-prompted forward pass per image with all phrasings; each phrase is scored from its own token span (max token probability), a component takes its best phrase, per-class NMS 0.7; exterior images only (`kaggle/jobs/j1_zeroshot/zeroshot.py`).
Per-class thresholds are a ground-truth-free recall proxy (the 60th percentile of per-image max score on exterior_full images,
floor 0.25). Detections are evidence, not verified boxes; they never enter `data/boxes/gold/`.

## 6. Knowledge rules (MACHINE_COMPILED)
`templexai/knowledge/compile_rules.py` fetches Wikipedia pages and keeps a (component, tradition) rule only if one fetched sentence
mentions both; each rule stores URL, retrieval date and a quote of <= 25 words (`docs/knowledge_sources.md`,
`configs/knowledge_rules.yaml`). Rules are co-occurrence evidence, not verdicts; classifier variants learn their own weights.

## 7. Classifiers (J2) and ablation
EfficientNet-B0 and MobileNetV3-Large (timm, ImageNet weights), class-weighted cross-entropy, early stopping on val macro-F1.
Variants: A image only; B + component vector (J1); C + knowledge-prior vector; D + KL alignment of predictions to the prior
where evidence exists. 3 seeds each; mean and std over seeds; best seed chosen by validation macro-F1 only.

## 8. Distilled detector (J3)
Small Ultralytics YOLO trained on J1 boxes (train-split exterior images, boxes above J1 thresholds). Validation numbers are
agreement with the zero-shot pseudo-labels, not accuracy. Corrected boxes (add-on) replace the distilled ones automatically.

## 9. Explanation evaluation (BUILD phase)
Ground-truth-free: deletion/insertion faithfulness, component-occlusion sensitivity, SHAP-knowledge alignment, Grad-CAM agreement
with detector boxes (agreement, not correctness). Gold-box metrics are shown as "available with the gold-set add-on".

## 10. Knowledge layer: compilation and validation (BUILD, MACHINE_COMPILED)
Full-text snapshots of the cited Wikipedia articles are saved in `data/knowledge_sources/<Title>.json` (URL, retrieval date, licence, text).
`templexai/knowledge/taxonomy_spec.yaml` lists, for each of the 7 components and 3 traditions, a definition and visual cues written in the
project's own words as paraphrases of the quoted sentences, the quotes (≤ 25 words, verbatim) with their page, and `links` (component →
tradition evidence, each with its own quote). `python -m templexai.knowledge.compile_taxonomy` resolves URLs/dates from the snapshots, writes
`templexai/knowledge/taxonomy.yaml` and `configs/diagnostic_links.yaml`, and runs `templexai/knowledge/validate.py`, which fails if an entry has no
definition, cues, source, URL or quote, if a quote exceeds 25 words, or if a quote does not appear verbatim in its snapshot (tested on the real file).
The agent selected the quoted sentences after reading the fetched text; the definitions are therefore only as good as those sentences
(e.g. Wikipedia's *Vimana* article is about mythological flying vehicles, so the architectural sense is sourced from *Shikhara* and *Dravidian architecture*).
Per-entry override: `templexai/knowledge/taxonomy_reviewed.yaml` → that entry becomes HUMAN_REVIEWED. The older co-occurrence file
`configs/knowledge_rules.yaml` / `knowledge_support.yaml` (used for the C/D classifier features during training) is unchanged, so no retraining was needed;
explanations and the SHAP-alignment metric use the curated, quote-backed `configs/diagnostic_links.yaml` instead.
Explanations (`knowledge/explain.py`) build a structured evidence object and a template report with numbered citations; the class is read-only. The optional
LLM-phrasing module is not built and no paid API is used.

## 11. Serving model, fusion and part-based model (P4)
`templexai/models/cache_features.py` runs the distilled YOLO detector and the fine-tuned image backbone (J2 variant A, EfficientNet-B0, selected seed) once over
all exterior images of labelled temples (`data/derived/boxes_yolo.json`, `feats_*.npy`). `templexai/models/fusion.py` trains, on those caches (3 seeds, early stopping on
val macro-F1, temple-disjoint split asserted): a CNN-only head, the **fusion** head (CNN features ⊕ 42-d component vector from the distilled detector), and components-only
models (gradient boosting, MLP). The serving model is the fusion head by design (components are part of the decision, as in the guide), chosen before looking at test results;
the results table reports it next to the CNN-only control with McNemar tests so the comparison is visible either way. Caveat: the backbone was fine-tuned on the training
images, so cached training features are optimistic; the head is regularised (dropout, weight decay, early stopping on validation temples).
The classes are imbalanced (Dravida dominates); macro-F1 is the headline metric and the part-based models are not competitive with the CNN, which limits what
their SHAP values can tell us (stated in the UI text: SHAP explains the part-based model).

## 12. Explanation evaluation protocol (P5, ground-truth-free)
Run by `python -m templexai.eval.run_explanations` on the test split; image-branch explanations keep the component vector fixed while pixels change.
- **Deletion / insertion**: pixels ranked by Grad-CAM (pytorch-grad-cam on `bn2`, the last feature map) are replaced by the grey mean (deletion) or restored onto a blurred copy (insertion) in 10 % steps;
  AUC of the predicted-class probability; baseline = a random saliency map at the CAM's own 6×6 granularity.
- **Component occlusion**: grey out the best box of each detected component and an equal-area random box; paired difference of probability drops.
- **SHAP alignment**: exact Shapley values over the present components (`xai/shap_components.py`) of the components-only gradient-boosting model; is the top component one that
  `configs/diagnostic_links.yaml` links to the predicted tradition? Baseline = share of present components that are linked (a random pick). Also component-deletion AUC in SHAP order vs random order.
- **Grad-CAM / detector agreement**: CAM peak and CAM energy inside detected boxes vs the box area share; reported as agreement with the detector, never as correctness.
Confidence intervals resample temples (images of one temple are correlated). Gold-based metrics (pointing game, explanation IoU, detector mAP) are implemented and unit-tested on synthetic data and run
automatically when `data/boxes/gold/` holds boxes.

## 13. Ablation and robustness (P8)
`experiments/ablation.py` tabulates A–D × 3 seeds from the J2 metrics files. `experiments/robustness.py` perturbs the test images (brightness, contrast, 70 % random crop, 25 %-area occlusion patch,
blur, downscale to 64 px, background masking outside detected boxes) and evaluates the image-only model A (3 seeds) and the serving fusion model (detector re-run per perturbed image). Variants B–D need
zero-shot (Grounding DINO) boxes, which are not recomputed under perturbation, so they are not part of the robustness table.

## 14. Limitations
- Labels are rule-based from open sources and the taxonomy is operational; the test split contains few Nagara and very few Deccan-Vesara temples, so per-class numbers are noisy.
- The detector is distilled from a zero-shot detector; its quality is unknown without gold boxes, and the zero-shot component boxes are weak evidence (see the detector agreement numbers).
- The component-based evidence is weak in this build (see the part-based results and the occlusion table); the CNN carries most of the prediction.
- Knowledge entries are compiled from Wikipedia only; they are paraphrases of cited sentences, not scholarly review.
- Training ran on the Mac (MPS) at reduced image size/epochs because the Kaggle GPU queue did not start (BLOCKERS.md).
- Image-level statistics (McNemar) ignore temple clustering; bootstrap CIs resample temples.

## 15. Optional human validation
See `HUMAN_ADDONS.md`; ready-made kits are in `addons/` (gold boxes, label audit, taxonomy review, box corrections). Each add-on plugs in via `prompts/03_HUMAN_ADDONS.md` without code changes
(labels → `data/labels/temple_styles_human.csv`, taxonomy → `templexai/knowledge/taxonomy_reviewed.yaml`, gold → `data/boxes/gold/`, corrections → `data/boxes/corrected/`).
