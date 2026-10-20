# Robustness (test split, exterior images)

A_image_only: EfficientNet-B0, mean ± std over 3 seeds. fusion: serving model (detector re-run on each perturbed image). `background_masked` keeps only images with ≥1 detection (n shown).

| condition | model | n | accuracy | macro-F1 |
|---|---|---|---|---|
| background_masked | A_image_only | 802 | 0.676 ± 0.075 | 0.570 ± 0.062 |
| background_masked | fusion | 802 | 0.751 | 0.639 |
| brightness_0.5 | A_image_only | 940 | 0.814 ± 0.012 | 0.680 ± 0.025 |
| brightness_0.5 | fusion | 940 | 0.831 | 0.714 |
| brightness_1.5 | A_image_only | 940 | 0.809 ± 0.009 | 0.682 ± 0.009 |
| brightness_1.5 | fusion | 940 | 0.804 | 0.686 |
| clean | A_image_only | 940 | 0.814 ± 0.003 | 0.688 ± 0.013 |
| clean | fusion | 940 | 0.827 | 0.717 |
| contrast_0.5 | A_image_only | 940 | 0.799 ± 0.030 | 0.666 ± 0.037 |
| contrast_0.5 | fusion | 940 | 0.829 | 0.707 |
| contrast_1.5 | A_image_only | 940 | 0.811 ± 0.003 | 0.687 ± 0.005 |
| contrast_1.5 | fusion | 940 | 0.801 | 0.685 |
| downscale_64px | A_image_only | 940 | 0.756 ± 0.043 | 0.487 ± 0.033 |
| downscale_64px | fusion | 940 | 0.793 | 0.452 |
| gaussian_blur | A_image_only | 940 | 0.715 ± 0.065 | 0.459 ± 0.014 |
| gaussian_blur | fusion | 940 | 0.777 | 0.396 |
| occlusion_patch_25pct | A_image_only | 940 | 0.685 ± 0.103 | 0.546 ± 0.095 |
| occlusion_patch_25pct | fusion | 940 | 0.776 | 0.620 |
| random_crop_50pct | A_image_only | 940 | 0.779 ± 0.018 | 0.654 ± 0.027 |
| random_crop_50pct | fusion | 940 | 0.803 | 0.683 |
