"""SHAP-style attribution on the components-only model (gradient boosting on the 42-d component vector).

Players = the 7 components. Value of a coalition S = P(class | vector with only the components in S kept, others zeroed, i.e. "absent").
Shapley values are computed EXACTLY over the components present in the image (<= 2^7 coalitions), which is what SHAP estimates.
(shap.TreeExplainer does not support scikit-learn's multiclass GradientBoostingClassifier in the installed shap 0.51; the exact
computation is cheap here and checked against the efficiency property in tests.)
"""
from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np

from templexai.models.component_features import COMPONENTS, PER_COMPONENT

K = len(PER_COMPONENT)


def component_shap(predict_proba, vec: np.ndarray, cls: int) -> dict[str, float]:
    """Exact Shapley value of each PRESENT component for P(cls). Absent components get 0 (omitted)."""
    present = [i for i in range(len(COMPONENTS)) if vec[i * K] > 0.5]
    n = len(present)
    if n == 0:
        return {}
    subsets = [s for r in range(n + 1) for s in combinations(range(n), r)]
    X = np.zeros((len(subsets), len(vec)), dtype=vec.dtype)
    for row, s in enumerate(subsets):
        for j in s:
            ci = present[j]
            X[row, ci * K:(ci + 1) * K] = vec[ci * K:(ci + 1) * K]
    v = dict(zip(subsets, predict_proba(X)[:, cls]))
    out = {}
    for j in range(n):
        phi = 0.0
        for s in subsets:
            if j in s:
                continue
            w = factorial(len(s)) * factorial(n - len(s) - 1) / factorial(n)
            phi += w * (v[tuple(sorted(s + (j,)))] - v[s])
        out[COMPONENTS[present[j]]] = float(phi)
    return out
