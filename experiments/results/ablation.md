# Ablation (test split, temple-disjoint; mean ± std over seeds)

A image CNN; B +component features (zero-shot boxes); C +knowledge-prior features; D +evidence-alignment loss. P4 rows: heads trained on cached frozen features (`cnn_only`, `fusion` = CNN ⊕ distilled-YOLO component vector, `parts_*` = components only).

| variant | backbone / head | seeds | accuracy | macro-F1 |
|---|---|---|---|---|
| A | efficientnet_b0 | 3 | 0.814 ± 0.003 | 0.688 ± 0.013 |
| A | mobilenetv3_large_100 | 3 | 0.812 ± 0.013 | 0.697 ± 0.016 |
| B | efficientnet_b0 | 3 | 0.815 ± 0.015 | 0.682 ± 0.024 |
| C | efficientnet_b0 | 3 | 0.824 ± 0.019 | 0.711 ± 0.018 |
| D | efficientnet_b0 | 3 | 0.822 ± 0.017 | 0.694 ± 0.027 |
| P4 | cnn_only | 3 | 0.844 ± 0.003 | 0.716 ± 0.008 |
| P4 | fusion | 3 | 0.830 ± 0.006 | 0.714 ± 0.002 |
| P4 | parts_gb | 3 | 0.541 ± 0.003 | 0.450 ± 0.001 |
| P4 | parts_mlp | 3 | 0.548 ± 0.018 | 0.386 ± 0.004 |
