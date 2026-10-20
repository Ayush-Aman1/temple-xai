Run the BUILD phase of Temple XAI end to end, autonomously, in AGENT-ONLY mode. Read CLAUDE.md, PROGRESS.md, BLOCKERS.md, docs/METHOD.md, artifacts/MANIFEST.json and artifacts/TRAINING_REPORT.md first.
- If runs/train_all/STATUS is not DONE: tell me in two lines and stop.
- Resume from the first unchecked [BUILD] task in PROGRESS.md.
Do not ask me for confirmation between tasks. Blockers → BLOCKERS.md + tools/notify.sh + documented fallback, then continue.

PHASE GOAL
A complete, working, fully evaluated system with no human input: every [BUILD] task in the guide, in order, wired to the trained artifacts, so on this Mac `make test` is green and `make demo` runs everything. Where a guide task waits on a human, build the agent-only version from CLAUDE.md AND the hook that lets the human input replace it later with no code change. Commit after each task.

WORK LOOP — for each task
Re-read its module prompt (via docs/md/INDEX.txt) → implement that scope (plus the agent-only replacement where needed) → tests → run → fix → tick PROGRESS.md → commit "P<n>: …". Anything > ~5 minutes (evaluation sweeps, robustness, Grad-CAM over the test split): `bash tools/bg.sh <label> "<cmd>"`, keep building other parts, check runs/<label>/STATUS once when you need it; if still running at the end, finish the rest and end your turn — re-running this prompt resumes.

WHAT MUST EXIST AT THE END (the guide's specs win on details)
1. Knowledge layer (guide P7), MACHINE_COMPILED:
   - Fetch open sources (Wikipedia articles on Nagara, Dravidian, Vesara/Deccan, Hoysala, Chalukya, Kalinga architecture and on each component; Wikimedia/Wikisource or other openly licensed texts you can actually retrieve). Save fetched text snapshots under data/knowledge_sources/ with URL + date.
   - taxonomy.yaml: for each tradition and component — definition, visual_cues, related components, and sources (URL, retrieved date, quote ≤ 25 words supporting the entry). Write definitions in your own words strictly from those quotes; nothing from memory. Every entry has knowledge_mode: MACHINE_COMPILED.
   - The validator from the guide enforces: every entry has ≥ 1 source with URL + quote; quotes appear verbatim in the saved snapshot. Tests run the validator on the real file.
   - Expand configs/knowledge_rules.yaml (component → tradition evidence) the same way; keep it consistent with what J2 used, or re-run only the affected evaluation (not training).
   - explain.py + template report exactly as the guide specifies, citing taxonomy sources inline. The optional LLM phrasing module stays OFF by default; the demo needs no paid API.
   - Hook: if taxonomy_reviewed.yaml exists it overrides per entry → HUMAN_REVIEWED.
2. Explanation evaluation without ground truth (the agent-only replacement for the gold-set metrics) — implement, test, run on the test split, write to experiments/results/explanations/:
   - Deletion / insertion faithfulness curves and AUC for Grad-CAM (and SHAP where applicable), vs a random-saliency baseline.
   - Component-occlusion sensitivity: mask each detected component box (grey fill), measure the drop in predicted-class probability; aggregate per component × tradition; compare with masking random boxes of equal area.
   - SHAP–knowledge-graph alignment (X-NeSyL-style, per the guide): how often top SHAP components for a prediction are ones the knowledge rules link to the predicted tradition, vs a shuffled baseline.
   - Grad-CAM / detector-box agreement (pointing-game style) — reported as "agreement with detector", never as correctness.
   - Gold-based metrics (explanation IoU, pointing game vs gold, detector mAP): implement + unit-test on a synthetic fixture; they run automatically when data/boxes/gold/ exists; otherwise reports say "available with the gold-set add-on (HUMAN_ADDONS.md)".
3. Ablation + robustness (guide P8): from saved artifacts on this Mac (MPS) — A–D over 3 seeds (mean ± std), brightness/contrast, crops, occlusion patches, blur/downscale, background masking. Tables + plots to experiments/results/; markdown summary quoting only those files.
4. Backend (guide P9): FastAPI as specified (POST /analyze, GET /taxonomy, GET /health, plus whatever the guide lists). Every response includes provenance {label_mode, knowledge_mode, detector_mode, gold_eval, model ids, git commit}. Models load once at startup. TestClient tests with a sample test-split image.
5. Frontend / demo (guide P10 onward; if the stack is open: Vite + React + TypeScript): upload → predicted tradition + probabilities, component boxes overlay, Grad-CAM toggle, SHAP contribution chart, evidence report with cited sources, image attribution (licence/author), and a small "Method" panel explaining in plain words how labels, knowledge and detector were produced (reads provenance; updates automatically when add-ons are used).
6. docs/METHOD.md completed for the report: labelling protocol + coverage table, knowledge compilation method, distillation, evaluation protocol, limitations, and a short "Optional human validation" section pointing to HUMAN_ADDONS.md.
7. CREDITS.md: every repo, model, dataset and text source with URL + licence.
8. Add-on kits in addons/ (so the optional human work is quick whenever someone does it), each with a README.md giving exact steps for the installed tool versions:
   - addons/gold/: 150 test-split exterior images stratified by tradition (list + copies or paths), a Label Studio labeling config with the 7 component classes, and import instructions. Exports go to data/boxes/gold/.
   - addons/corrections/: ~500 train-split images with the J1 pre-labels as a Label Studio import file. Exports go to data/boxes/corrected/.
   - addons/labels_audit/: temple_styles_review.csv = rule-based label + every source URL + raw value per temple, with empty columns `human_tradition`, `human_source`, `notes`; plus a script that turns the reviewed CSV into data/labels/temple_styles_human.csv.
   - addons/taxonomy_review/: a checklist generated from taxonomy.yaml (one row per entry with its sources) and instructions for producing taxonomy_reviewed.yaml.

FINISH
- Makefile targets: setup, test, demo (backend + frontend), eval, report, retrain (pipelines/train_all.sh via tools/bg.sh).
- README.md: what it is, quick start, Mermaid architecture diagram, results tables generated from experiments/results/, method summary, credits. DEMO.md: a 5-minute demo script with 5 test-split images.
- Update the "Add-on status" table at the bottom of HUMAN_ADDONS.md.
- Fresh-shell check: `make setup && make test && make demo` (stop once /health passes).
- `bash tools/notify.sh "Temple XAI: BUILD DONE" "make demo"`. Final reply: what's done, how to run it, headline results (copied from results files), and which optional add-ons would add what.
