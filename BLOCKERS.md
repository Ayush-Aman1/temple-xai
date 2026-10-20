# BLOCKERS
- (info) PyPI downloads were very slow/interrupted during setup; resolved by installing torch first.
- 2026-10-20 21:37:29 Kaggle job j0_gpu_check failed; reran locally (compute=local_mps)
- 2026-10-20 21:37:31 Kaggle GPU kernel queued >1h without starting (free-tier queue); switched remaining jobs to compute=local (MPS), reduced epochs/image size. Dataset ayushaman707/templexai-data is uploaded if you want to rerun on Kaggle later.
- 2026-10-21 (BUILD) shap 0.51 TreeExplainer rejects scikit-learn's multiclass GradientBoostingClassifier -> exact Shapley values over the 7 components implemented in templexai/xai/shap_components.py (tested for efficiency/dummy axioms); documented in CREDITS.md and METHOD.md.
- 2026-10-21 (BUILD) Label Studio not installed locally -> add-on READMEs follow its documented Local Files workflow (written for 1.x) and say so; not tested against a live instance.
