# PROGRESS — Temple XAI (agent-only build)

[TRAIN] items (this phase). [BUILD] items are for prompts/02_BUILD.md.

- [x] P0 scaffold: package, device/seed, interfaces, schemas, taxonomy skeleton, pytest config
- [x] P1 / a1 temples (Wikidata) · a2 rule-based labels (cited, conflicts excluded) · a3 Commons images (licences) · a4 CLIP image type · a5 temple-disjoint split · a6 Kaggle package
- [x] Style-map verification against fetched Wikipedia pages (docs/style_map_evidence.md) + knowledge rules (docs/knowledge_sources.md)  *(replaces the human taxonomy step)*
- [x] J0 GPU check · J1 zero-shot boxes (Grounding DINO) · J2 classifiers A–D × 3 seeds (+ MobileNetV3 baseline) · J3 distilled YOLO detector  — code + SMOKE passed
- [x] Full pipeline run DONE (local MPS; Kaggle GPU queue stalled >1 h, see BLOCKERS.md) → artifacts/MANIFEST.json, TRAINING_REPORT.md
- [x] [BUILD] P7 taxonomy compiled from fetched snapshots + validator + explain.py (cited report)
- [x] [BUILD] P4 fusion model proper (cached features; CNN-only control, fusion, part-based GB/MLP; McNemar)
- [x] [BUILD] P5 explanation evaluation, ground-truth-free (+ gold-based metrics implemented/tested, auto-run when data/boxes/gold exists)
- [x] [BUILD] P8 ablation A–D and robustness (script + results in experiments/results/)
- [x] [BUILD] P9 FastAPI (/analyze, /taxonomy, /health, /gallery, /method) with provenance, TestClient tests
- [x] [BUILD] P10 Vite+React+TS+Tailwind UI (upload, boxes, Grad-CAM slider, SHAP chart, cited report, Method panel, gallery)
- [x] [BUILD] add-on kits (addons/), Makefile, README, DEMO, METHOD sections 10–15, CREDITS
- Human-wait steps replaced by agent-only defaults: P1 step 2 (labels) → RULE_BASED; P3 gold boxes → skipped (gold_eval=UNAVAILABLE); P6 corrected boxes → DISTILLED.

## Compute estimates (GPU, Kaggle T4)
J1 ≈ 1–2 h (images × 7 prompts), J2 ≈ 1–2 h (5 variants × 3 seeds, 3 epochs), J3 ≈ 0.5 h → well under 20 GPU-h/week.
