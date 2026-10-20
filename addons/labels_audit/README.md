# Add-on 2 — Label audit (30 min for 50 temples, ~3 h for all 362)

`temple_styles_review.csv` has one row per labelled temple in **random order**: the rule-based label, every source URL that produced it, and the raw value found there. Columns to fill: `human_tradition`, `human_source`, `notes`.

## Steps
1. Open the CSV in a spreadsheet. For each temple (start at the top; the first 50 are a random sample), open the source URLs / your own reference and decide the tradition.
2. `human_tradition` ∈ `Nagara`, `Dravida`, `Deccan_Vesara`, `EXCLUDE` (use EXCLUDE for hybrids / out-of-taxonomy temples). `human_source` is **required** (book + page, paper, ASI page or a URL). Leave a row blank to keep the rule-based label.
3. Save the CSV, then from the repo root: `python addons/labels_audit/make_human_labels.py`. It writes `data/labels/temple_styles_human.csv` and prints the agreement rate with a 95 % CI — that is the measured label-error rate for the report.
4. Run `claude --model sonnet "$(cat prompts/03_HUMAN_ADDONS.md)"`. If only a sample was audited the agent reports the agreement rate; if labels changed it retrains the classifiers (the split file is never regenerated).
