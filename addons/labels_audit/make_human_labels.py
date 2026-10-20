"""Turn the reviewed CSV into data/labels/temple_styles_human.csv (columns temple_id, tradition, source).
Usage (from the repo root):  python addons/labels_audit/make_human_labels.py
A row counts only if human_tradition is one of Nagara / Dravida / Deccan_Vesara / EXCLUDE AND human_source is non-empty.
Rows you leave blank are NOT overridden (the rule-based label stays). Also prints the agreement rate with the rule-based labels."""
import csv
import math
from pathlib import Path

ALLOWED = {"Nagara", "Dravida", "Deccan_Vesara", "EXCLUDE"}
ALIAS = {"deccan-vesara": "Deccan_Vesara", "deccan/vesara": "Deccan_Vesara", "deccan_vesara": "Deccan_Vesara"}
src = Path("addons/labels_audit/temple_styles_review.csv")
rows, bad, agree, n = [], [], 0, 0
for r in csv.DictReader(src.open()):
    t = (r["human_tradition"] or "").strip()
    if not t:
        continue
    t = ALIAS.get(t.lower(), t)
    if t not in ALLOWED or not (r["human_source"] or "").strip():
        bad.append(r["temple_id"]); continue
    rows.append({"temple_id": r["temple_id"], "tradition": t, "source": r["human_source"].strip()})
    n += 1; agree += t == r["rule_label"]
Path("data/labels").mkdir(parents=True, exist_ok=True)
with open("data/labels/temple_styles_human.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["temple_id", "tradition", "source"]); w.writeheader(); w.writerows(rows)
print(f"wrote {len(rows)} human labels; skipped {len(bad)} invalid rows {bad[:10]}")
if n:
    p = agree / n; se = math.sqrt(p * (1 - p) / n)
    print(f"agreement with rule-based labels: {agree}/{n} = {p:.1%} (normal-approx 95% CI {max(0, p - 1.96 * se):.1%}-{min(1, p + 1.96 * se):.1%})")
