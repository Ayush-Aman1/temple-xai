# Optional human add-ons for Temple XAI

The first build is complete and evaluated without any human input. Each add-on below upgrades one part
of it and can be done independently, in any order, by anyone on the team. After adding one or more,
run:

```bash
claude --model sonnet "$(cat prompts/03_HUMAN_ADDONS.md)"
```

It validates what you added, re-runs only what's affected, and updates results, the API provenance
and the UI "Method" panel automatically. The agent prepares a ready-made kit for each add-on in
`addons/` during the build (each kit has its own README with exact steps).

| Priority | Add-on | Effort | What it unlocks | Where the output goes |
|---|---|---|---|---|
| 1 | **Gold test boxes** — draw component boxes on 150 test images in Label Studio | ~2.5 h (≈ 1 min/image) | Explanation correctness metrics: Grad-CAM-to-component IoU, pointing game vs ground truth, real detector mAP. This is the strongest evidence for the project's explainability claim. No retraining. | `data/boxes/gold/` (Label Studio JSON export) |
| 2 | **Label audit** — check the rule-based style label of ~50 random temples (or all) against the cited sources | 30 min (50) – 3 h (all) | A measured label-error rate for the report (50) or fully human-verified labels (all). Retrains classifiers if labels change. | `addons/labels_audit/temple_styles_review.csv` → script → `data/labels/temple_styles_human.csv` |
| 3 | **Taxonomy review** — Member 1 reads each machine-compiled entry and its sources, corrects or approves | 1–2 h | Knowledge layer marked HUMAN_REVIEWED; stronger IKS section. No retraining. | `taxonomy_reviewed.yaml` |
| 4 | **Box corrections** — fix the zero-shot pre-labels on ~500 training images | 3–5 h | Detector trained on human-corrected boxes instead of distilled ones (usually a modest accuracy gain). Retrains the detector only. | `data/boxes/corrected/` |

## Rules that keep the add-ons valid
- Gold boxes are drawn from scratch on the images in `addons/gold/` (test split). Never start them
  from the pre-labels, or the metrics would measure the zero-shot detector instead of the model.
- Corrections are only for the training images in `addons/corrections/`.
- Every human label needs a source (book, paper, ASI page or a URL) in the `source` column.
- Don't edit `data/splits.csv`.

## Add-on status
(The agent keeps this table current.)

| Add-on | Present | Integrated | Notes |
|---|---|---|---|
| Gold test boxes | no | no | kit ready: `addons/gold/` (150 test images, no pre-labels) |
| Label audit / human labels | no | no | kit ready: `addons/labels_audit/` (362 labelled temples, random order) |
| Taxonomy review | no | no | kit ready: `addons/taxonomy_review/` (33 rows); loader hook `templexai/knowledge/taxonomy_reviewed.yaml` |
| Box corrections | no | no | kit ready: `addons/corrections/` (500 train images with zero-shot pre-labels) |
