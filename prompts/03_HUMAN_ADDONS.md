One or more optional human add-ons from HUMAN_ADDONS.md are now in the repo. Integrate exactly the ones present, autonomously. Read CLAUDE.md, PROGRESS.md, HUMAN_ADDONS.md, docs/METHOD.md and artifacts/MANIFEST.json first.

If runs/addons/STATUS exists: RUNNING → summarise the last 20 log lines and stop; FAILED:* → read the last 80 lines of runs/addons/log.txt, fix, smoke-test, relaunch with the same command, stop; DONE → go to step 4.

1. Detect and validate what's present (report problems in BLOCKERS.md; skip an invalid add-on, continue with the valid ones):
   a. data/labels/temple_styles_human.csv — every row has tradition ∈ {Nagara, Dravida, Deccan-Vesara, EXCLUDE} and a non-empty source; temple_ids exist.
   b. taxonomy_reviewed.yaml — passes the taxonomy validator.
   c. data/boxes/gold/ — Label Studio exports parse; every image is in the TEST split; class names match.
   d. data/boxes/corrected/ — exports parse; every image is in the TRAIN split.
2. Work out the minimum that must re-run:
   a (labels) → labels stage, package/upload, J2, J3, collect, then all evaluation. NEVER regenerate data/splits.csv; temples now EXCLUDED drop out; record per-split counts before/after in PROGRESS.md.
   b (taxonomy) → no retraining; re-run knowledge-dependent evaluation and reports only.
   c (gold boxes) → no retraining; run the gold-based metrics.
   d (corrected boxes) → J3 only, then evaluation.
   Remove only the matching stage markers.
3. If anything needs training: smoke-test, then `bash tools/bg.sh addons "bash pipelines/train_all.sh"`, notify "Temple XAI: add-on retrain launched", update PROGRESS.md, commit, reply briefly, STOP. If nothing needs training, go straight to step 4.
4. Run `make eval report`. Confirm MANIFEST and API provenance now show the upgraded modes (HUMAN_VERIFIED / HUMAN_REVIEWED / HUMAN_CORRECTED / gold_eval=AVAILABLE as applicable) and the UI Method panel reflects them. If the label add-on covers only a sample of temples, also compute the agreement rate between rule-based and human labels (with a 95% confidence interval) and add it to docs/METHOD.md as the measured label-error rate. Regenerate README results and docs/METHOD.md sections from the results files; update the "Add-on status" table in HUMAN_ADDONS.md; commit; notify "Temple XAI: add-ons integrated"; reply with a before/after table of headline metrics (copied from results files).
