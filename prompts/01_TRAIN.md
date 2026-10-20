Run the TRAIN phase of Temple XAI end to end, autonomously, in AGENT-ONLY mode (no human input needed). Read CLAUDE.md first and follow it exactly — especially "Build strategy: agent-only first". Do not ask me for confirmation between steps. If something is blocked: record it in BLOCKERS.md, run tools/notify.sh, take the documented fallback, keep going.

PHASE GOAL
With zero paid services and zero human input, produce: the dataset (licences tracked, fixed temple-disjoint split, rule-based cited labels), zero-shot component boxes, the trained style classifiers including the ablation runs, and the distilled component detector — collected in artifacts/ with artifacts/MANIFEST.json and artifacts/TRAINING_REPORT.md — then notify me. Every stage must also accept the optional human inputs listed in CLAUDE.md and prefer them automatically if they exist. You write and smoke-test the code in this session; all heavy work runs in ONE detached background pipeline; you end your turn once it is launched.

STEP 0 — Resume or start
1. If PROGRESS.md exists, read it plus BLOCKERS.md and runs/train_all/STATUS (if present):
   - DONE → verify MANIFEST covers every [TRAIN] deliverable; reply "ready for prompt 02" with a 5-line summary; stop.
   - RUNNING → summarise the last 20 lines of runs/train_all/log.txt; stop.
   - FAILED:* → read only the last 80 log lines, fix the cause, re-run that stage's smoke test, relaunch with the same bg.sh command (finished stages are skipped), notify, stop.
   - No STATUS → continue from the first unchecked [TRAIN] item.
2. Fresh start: `git init` (if needed); `bash tools/docx_to_md.sh`; read docs/md/INDEX.txt; then read by heading: purpose/tiers, what already exists, data sources, component annotations, download checklist, pick list, build order, master context prompt, every module prompt (P0 …).
   Create PROGRESS.md with one checkbox per guide task tagged [TRAIN] (creates data, labels, cached features or trained/zero-shot model artifacts: data pipeline, zero-shot detection, feature caching, CNN baselines, YOLO detector, ablation training runs) or [BUILD] (everything else). Where a guide task waits on a human, note the agent-only replacement from CLAUDE.md next to it. Do P0 scaffolding + [TRAIN] work only.

STEP 1 — Environment
- .venv, Python 3.11; requirements-train.txt pinned; check `torch.backends.mps.is_available()`.
- Kaggle: `kaggle --version` and `kaggle kernels list --mine --page-size 1`. Auth fails → BLOCKER "Kaggle token missing — see START_HERE.md", notify, write `compute: local` in configs/compute.yaml; else `compute: kaggle`. Read the username from the CLI config; never print or copy the token.

STEP 2 — Write the code (the guide's details win on conflicts, except where CLAUDE.md overrides)
A. Data pipeline in templexai/data/, one CLI per stage, all honouring --data-root (data/, or data_smoke/ in smoke mode). Polite to Wikimedia: descriptive User-Agent with a contact placeholder, ≤ 5 requests/s, exponential backoff, response cache in data/cache/.
   a1 temples: Wikidata SPARQL for Hindu temples in India → data/meta/temples.csv (temple_id, name, qid, commons_category, enwiki_title, lat, lon, state, wikidata_style_qids, wikidata_style_labels).
   a2 labels (RULE_BASED protocol — document it in docs/METHOD.md):
      - Evidence sources per temple: (1) Wikidata P149 architectural style; (2) the English Wikipedia infobox "architecture style"/"architecture" field, fetched via the Wikipedia API; (3) Commons category membership in style categories (e.g. categories for Dravidian / Nagara / Hoysala / Chalukya / Kalinga architecture).
      - configs/style_map.yaml (you create it): maps each style value to Nagara / Dravida / Deccan-Vesara / EXCLUDE, with a comment giving the original label. Generic values ("Hindu temple architecture", "Indian architecture") → no evidence. Regional styles map only where the mapping is standard (e.g. Hoysala, Western Chalukya → Deccan-Vesara; Kalinga, Maru-Gurjara → Nagara; Pallava, Chola → Dravida); document each mapping with a fetched source in docs/METHOD.md.
      - Decision rule: at least one mapped source; all mapped sources agree → label; any disagreement → EXCLUDE. Store per temple: tradition, every source URL + the raw value used, retrieved date.
      - Output data/labels/temple_styles_rule.csv. If data/labels/temple_styles_human.csv exists, it overrides per temple and those rows get label_mode=HUMAN_VERIFIED.
      - Coverage check: print counts per tradition. Target 150–250 temples. If below ~120, or any tradition < 25 temples, widen the Wikidata query (subclasses of Hindu temple, Jain temples only if the guide allows) and record what you did; never relax the agreement rule.
   a3 images: Commons API per category → PD, CC0, CC BY, CC BY-SA only; ≤ 40 per temple; perceptual-hash dedupe; longest side 1024 px; data/meta/images.csv (image_id, temple_id, file, url, licence, author, sha1, phash).
   a4 image type: CLIP zero-shot (exterior-full / exterior-partial / detail / interior) on MPS/CPU → column in images.csv. Train on exterior images; keep the rest for robustness tests.
   a5 split: stratified temple-group split 70/15/15, fixed seed → <data-root>/splits.csv. Never touch it if it exists. Excluded/unlabelled temples are not in any split.
   a6 package: kaggle/dataset/ = images + meta + splits + labels + dataset-metadata.json (private, slug templexai-data).
B. Job scripts, one folder each in kaggle/jobs/<job>/: kernel-metadata.json (kernel_type script, language python, is_private true, enable_gpu true, enable_internet true, dataset_sources ["<user>/templexai-data"]) + ONE python script writing to --out (/kaggle/working/out on Kaggle): weights, metrics.json, logs, plots. Every script supports --smoke (≈ 20 images, 1 epoch), --data-root and --device so the same file runs on Kaggle (cuda) and on this Mac (mps/cpu). Pin versions in the script's pip line.
   J0 gpu_check → out/gpu.json (torch.cuda.is_available(), device name).
   J1 zero-shot components: the guide's zero-shot detector(s) (e.g. Grounding DINO tiny and/or OWLv2 via transformers — verify the installed API) on all images, with per-class score thresholds tuned on the CLIP-tagged exterior images for sensible recall → out/boxes_zeroshot.json, plus out/labelstudio_import.json (only used by the optional corrections add-on).
   J2 classifiers: the guide's baseline CNN(s) and ablation variants A–D × 3 seeds on the fixed split. Knowledge-informed variants use the component→tradition evidence table in configs/knowledge_rules.yaml (create it now with each rule citing a fetched source; P7 later expands it) plus component-presence features from J1. Save per-run metrics (accuracy, macro-F1, per-class report, confusion matrix) and the best checkpoint per variant.
   J3 detector (DISTILLED): small Ultralytics YOLO trained on J1 boxes from train-split exterior images only (filter low-confidence boxes; document thresholds). If data/boxes/corrected/ has Label Studio exports, use those instead → HUMAN_CORRECTED. Report agreement with J1 on validation images (as agreement, not accuracy); evaluate on data/boxes/gold/ only if it exists.
   GPU budget: estimate hours per job; keep the total ≤ ~20 GPU-hours/week; split anything > 11 h with checkpoints. Record estimates in PROGRESS.md.
C. pipelines/train_all.sh — idempotent stages with markers in runs/train_all/stages/; activates .venv; reads configs/compute.yaml; SMOKE=1 = ≈ 5 temples, --smoke everywhere, data_smoke/ + artifacts_smoke/, local only, no uploads. Order:
   a1 → a2 → a3 → a4 → a5 → (kaggle) a6 via tools/kaggle_dataset.sh → J0 via tools/kaggle_run.sh (no GPU → BLOCKER "Kaggle GPU unavailable (phone verification?)", notify, switch remaining jobs to local) → J1 → J2 → J3 → `python -m templexai.artifacts.collect`.
   Local mode = same scripts on MPS with reduced epochs/image size, recorded as compute=local_mps.
   collect: copy weights to artifacts/<job>/; artifacts/MANIFEST.json (per artifact: path, metrics, label_mode, knowledge_mode, detector_mode, gold_eval, compute, data hash of splits+labels, git commit, date); artifacts/TRAINING_REPORT.md (tables generated from metrics files; label coverage table; method summary).
   Success → bash tools/notify.sh "Temple XAI: training DONE" "Next: prompts/02_BUILD.md". Failure → bash tools/notify.sh "Temple XAI: training FAILED" "<stage> — see runs/train_all/log.txt", exit non-zero.

STEP 3 — Verify before launching
`pytest -q` green; `SMOKE=1 bash pipelines/train_all.sh` passes end to end (call it directly); delete runs/train_all/, data_smoke/, artifacts_smoke/; commit.

STEP 4 — Launch and end your turn
`bash tools/bg.sh train_all "bash pipelines/train_all.sh"`; `bash tools/notify.sh "Temple XAI: training launched" "<expected duration>"`; update PROGRESS.md; commit; reply with what is running, expected duration and how to check (`cat runs/train_all/STATUS`, `tail -n 40 runs/train_all/log.txt`). Then STOP — do not wait or poll.
