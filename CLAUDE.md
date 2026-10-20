# Temple XAI — standing rules for Claude Code

## Project in one paragraph
Explainable classification of Indian temple architectural traditions (operational taxonomy:
Nagara / Dravida / Deccan–Vesara) from photographs. The model detects architectural components
(shikhara, vimana, gopuram, mandapa, pillar, jagati/base, amalaka), uses them as evidence next to a
CNN, and explains each prediction with SHAP, Grad-CAM and a cited knowledge layer. Evaluation is
temple-disjoint. Closest prior work: X-NeSyL / EXPLANet (MonuMAI). Python package: `templexai`.

## Build strategy: agent-only first, human add-ons later
The first build must be a COMPLETE, working, fully evaluated system with no human input. Human work
(verified labels, reviewed taxonomy, gold boxes, corrected boxes) is an optional upgrade described in
`HUMAN_ADDONS.md`; each add-on plugs in independently later via `prompts/03_HUMAN_ADDONS.md` without
code changes. Design every component so human inputs, when present, simply override the agent-made ones.

| Part | Agent-only default (first build) | Human add-on (optional) |
|---|---|---|
| Style labels | Rule-based protocol: Wikidata P149 + Wikipedia infobox + Commons style categories, each label cited; conflicts → EXCLUDE. `label_mode=RULE_BASED` | `data/labels/temple_styles_human.csv` → `HUMAN_VERIFIED` |
| Knowledge layer | Compiled from fetched open sources, every claim with URL + short quote. `knowledge_mode=MACHINE_COMPILED` | Member 1 review → `HUMAN_REVIEWED` |
| Detector | YOLO distilled from zero-shot (Grounding DINO / OWLv2) boxes. `detector_mode=DISTILLED` | Corrected boxes → `HUMAN_CORRECTED` |
| Explanation evaluation | Ground-truth-free metrics: deletion/insertion faithfulness, component-occlusion sensitivity, SHAP–knowledge-graph alignment, Grad-CAM agreement with detector boxes (reported as agreement, not correctness). `gold_eval=UNAVAILABLE` | Gold boxes → explanation IoU, pointing game, detector mAP. `gold_eval=AVAILABLE` |

## Source of truth
- `docs/md/*.md`, converted from `docs/*.docx` (+ `docs/project.pdf`) by `bash tools/docx_to_md.sh`.
  The build guide's tasks (P0, P1, …) define the work; its "master context prompt" is part of these rules.
- Conflicts: on scope and technical detail the guide wins. On integrity, cost and the agent-only-first
  strategy above, this file wins — it deliberately supersedes the guide where the guide requires a
  human step before something can work (labels, taxonomy content, corrected boxes, gold-based metrics).
- Read the guide by section via `docs/md/INDEX.txt` (~150 lines around a heading), never all at once.

## Integrity rules (non-negotiable)
1. `data/splits.csv` is created once (temple-disjoint 70/15/15, fixed seed) and never regenerated.
   Smoke runs use `data_smoke/`.
2. Nothing scholarly comes from your memory. Every label and every knowledge-layer claim cites a
   source you actually fetched (URL + retrieved date + quote of at most 25 words). No source → no claim.
3. Zero-shot or distilled boxes never enter `data/boxes/gold/`.
4. Every result carries provenance: `label_mode`, `knowledge_mode`, `detector_mode`, `gold_eval`,
   `compute`, data hash, git commit. The UI and reports show a short "Method" panel stating these
   modes in plain words. Agent-only results are real results under a documented protocol — present
   them as such, not as "provisional". Only metrics that need gold boxes are shown as
   "available with the gold-set add-on".
5. Numbers in reports and READMEs come only from metrics files written by code.
6. Every image keeps licence, author and source URL (PD, CC0, CC BY, CC BY-SA only). Every
   third-party repo/model/dataset/text source gets a `CREDITS.md` line with URL + licence.

## Cost rules
- No paid services. GPU = Kaggle free tier via the `kaggle` CLI (`tools/kaggle_run.sh`,
  `tools/kaggle_dataset.sh`); fallback = this Mac (torch MPS). No paid LLM APIs.
- Never block the session on long work: anything likely > ~5 minutes runs via
  `bash tools/bg.sh <label> "<command>"` (detached, notifies on finish). No polling loops; check
  `runs/<label>/STATUS` once, when you need the result.
- Small tool output: logs to files, `tail -n 60`, quiet flags; summarise big files in 5 lines of Python.
- Smoke-test every stage (`SMOKE=1` / `--smoke`) before full or GPU runs. Work sequentially.

## Engineering rules
- Python 3.11 venv at `.venv` (uv if installed), pinned `requirements*.txt`; scripts activate `.venv`.
- Before using transformers, ultralytics, torch, shap, captum or kaggle APIs, check the INSTALLED
  version's API. Don't code from memory.
- Config-driven, no absolute paths, seeds everywhere, type hints + docstrings, small modules.
- Human-input hooks: loaders check for the human files listed in the table above and prefer them
  automatically; provenance modes update from what was actually used.
- pytest for every module; `pytest -q` before ticking a task. Idempotent pipelines with
  `runs/<pipeline>/stages/<stage>.done` markers.
- Commit after each task: `git add -A && git commit -m "P<n>: <summary>"`. Never force-push.

## State files — keep current
`PROGRESS.md` (guide tasks tagged [TRAIN]/[BUILD] + status), `BLOCKERS.md`,
`docs/METHOD.md` (the labelling protocol, knowledge compilation method, distillation and evaluation
protocol — written for the report), `artifacts/MANIFEST.json`, and the "Add-on status" table at the
bottom of `HUMAN_ADDONS.md` (update which inputs exist; never edit the instructions above it).

## Notifications
`bash tools/notify.sh "<title>" "<message>"` at pipeline launch / done / failed, phase complete, and
blockers that need a human.

## End of every turn
Reply in ≤ 25 lines: tasks done, tests + results, background jobs (label + how to check), blockers,
next step.
