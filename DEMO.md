# 5-minute demo script

Start: `make demo` (backend :8000, frontend :5173), open http://127.0.0.1:5173. Everything runs offline once models are cached.
All five images are test-split photographs of temples the models never trained on; click them in the gallery strip at the bottom of the page (the `image_id` is in the page's gallery data and in `artifacts/p4/test_predictions.csv`).

| # | image_id | what to show |
|---|---|---|
| 1 | `00cac2c86e25` (Dravida, correct) | Upload flow result: tradition + probability bars, component boxes (toggle each), Grad-CAM slider, the cited report. Point at the numbered references and the operational-taxonomy note. |
| 2 | `02091377d7d2` (Nagara, correct) | Different tradition: shikhara / amalaka evidence, SHAP bar chart (what the part-based model used), "cited as evidence for …" tags. |
| 3 | `00fe87c138ad` (Deccan / Vesara, correct) | The hybrid class: jagati / pillar evidence; mention this class has the fewest temples. |
| 4 | `028feeb500a6` (Dravida, predicted Nagara — wrong) | Explain the miss from the evidence: which components were (not) found, where Grad-CAM looked, what the part-based model says on its own (shown under the SHAP chart). |
| 5 | `156c1f85b6f6` (Deccan / Vesara, predicted Nagara — wrong) | Honest failure: hybrid temple read as northern; label is an operational grouping. |

Close with the **Method** panel (label_mode / knowledge_mode / detector_mode / gold_eval in plain words) and `experiments/results/explanations/explanations.md` (the ground-truth-free explanation metrics), then mention the optional add-ons in `HUMAN_ADDONS.md`.

Your own photo: drop any exterior temple photograph on the upload box.
