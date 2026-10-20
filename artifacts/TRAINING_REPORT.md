# Temple XAI training report

Generated 2026-10-21 from metrics files only (git 3bc1a27).

## Dataset

label_mode=RULE_BASED; temples considered=1607; labelled=368

Label status: {"NO_EVIDENCE": 1061, "LABELLED": 368, "EXCLUDED_OUT_OF_TAXONOMY": 165, "EXCLUDED_CONFLICT": 13}

| split | Deccan_Vesara | Dravida | Nagara | temples | images (all types) |
|---|---|---|---|---|---|
| train | 19 | 190 | 45 | 254 | 6592 |
| val | 4 | 41 | 9 | 54 | 1276 |
| test | 4 | 41 | 9 | 54 | 1094 |

Image types: {"exterior_partial": 4026, "exterior_full": 3771, "interior": 729, "detail": 436}

Licences: {"CC BY-SA": 7172, "CC0": 809, "CC BY": 744, "Public domain": 237}

## J1 zero-shot component boxes (Grounding DINO tiny)

images=7691; compute=local_mps

| component | threshold | share of images with a detection |
|---|---|---|
| shikhara | 0.3114 | 0.322 |
| vimana | 0.3454 | 0.266 |
| gopuram | 0.2797 | 0.321 |
| mandapa | 0.3489 | 0.513 |
| pillar | 0.3233 | 0.375 |
| jagati_base | 0.3274 | 0.394 |
| amalaka | 0.317 | 0.379 |

Thresholds are a recall proxy (no ground truth); detections are evidence, not verified boxes.

## J2 style classifiers (temple-disjoint test split)

compute=local_mps; n_test_images=940

| variant | backbone | seeds | test acc (mean±std) | test macro-F1 (mean±std) | val macro-F1 |
|---|---|---|---|---|---|
| A | efficientnet_b0 | 3 | 0.814±0.003 | 0.688±0.013 | 0.753 |
| A | mobilenetv3_large_100 | 3 | 0.812±0.013 | 0.697±0.016 | 0.716 |
| B | efficientnet_b0 | 3 | 0.815±0.015 | 0.682±0.024 | 0.738 |
| C | efficientnet_b0 | 3 | 0.824±0.019 | 0.711±0.018 | 0.722 |
| D | efficientnet_b0 | 3 | 0.822±0.017 | 0.694±0.027 | 0.736 |

Variants: A image CNN; B +component features (zero-shot boxes); C +knowledge-prior features; D +evidence-alignment loss.

## J3 distilled component detector (YOLO)

detector_mode=DISTILLED; gold_eval=UNAVAILABLE; compute=local_mps

Agreement with zero-shot boxes on val temples: mAP50=0.392, mAP50-95=0.302, P=0.414, R=0.452. This is agreement with pseudo-labels, not accuracy; detector mAP vs. gold boxes is available with the gold-set add-on.

## Method (plain words)

- Style labels: RULE_BASED - rule-based from Wikidata P149, the Wikipedia infobox and Commons categories; any disagreement excludes the temple (docs/METHOD.md).
- Knowledge layer: MACHINE_COMPILED - sentences fetched from Wikipedia, each with URL and short quote (docs/knowledge_sources.md).
- Component detector: DISTILLED - YOLO trained on zero-shot Grounding DINO boxes.
- Explanation ground truth (gold boxes): UNAVAILABLE.

