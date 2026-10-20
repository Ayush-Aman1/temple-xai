# Add-on 1 — Gold test boxes (≈ 2.5 h)

Draw boxes for the 7 component classes on **150 test-split images** (`gold_image_list.csv`; tradition counts are as balanced as the test split allows — test has only a handful of Nagara and Deccan temples). This unlocks explanation IoU, pointing game vs ground truth and real detector mAP (`gold_eval=AVAILABLE`). **No retraining.**

**Rule:** draw from scratch. `import_tasks.json` deliberately has *no pre-labels*; never start from the zero-shot boxes or the metrics would measure the zero-shot detector.

Label Studio was **not installed** in the build environment, so these steps follow its documented Local Files workflow (written for Label Studio 1.x; check yours with `label-studio --version`). If a menu name differs in your version, the intent is: create a project, paste the labelling config, connect the folder as Local Files storage, import the tasks.

## Steps
1. `pip install label-studio`, then from the **repo root**:
   ```bash
   export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
   export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$PWD"
   label-studio start
   ```
2. Create a project "temple-gold". *Labeling Interface → Code*: paste `addons/gold/label_config.xml`.
3. *Settings → Cloud Storage → Add Source Storage → Local Files*: absolute path = `<repo>/addons/gold/images`, import method "Files" → *Sync*. (Or *Import* → upload `import_tasks.json`.)
4. Box every visible shikhara / vimana / gopuram / mandapa / pillar / jagati_base / amalaka. Box one representative pillar per group if there are dozens. Skip classes that are not visible.
5. *Export → JSON* (not JSON-MIN). Save it as `data/boxes/gold/gold_export.json` (create the folder if needed).
6. Run `claude --model sonnet "$(cat prompts/03_HUMAN_ADDONS.md)"`.

The loader (`templexai.eval.explain_metrics.load_gold`) also accepts YOLO `.txt` files named `<image_id>.txt`.
