"""Classification metrics + McNemar test (exact binomial, no scipy needed beyond stdlib)."""
from __future__ import annotations

from math import comb

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def cls_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0)),
            "weighted_f1": float(f1_score(y, p, average="weighted", labels=[0, 1, 2], zero_division=0)),
            "confusion": confusion_matrix(y, p, labels=[0, 1, 2]).tolist()}


def mcnemar(y, p_a, p_b) -> dict:
    """Exact two-sided McNemar on paired correctness. b = A right & B wrong, c = A wrong & B right."""
    y, a, b_ = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    b = int(((a == y) & (b_ != y)).sum())
    c = int(((a != y) & (b_ == y)).sum())
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    k = min(b, c)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"b": b, "c": c, "p_value": float(p), "note": "image-level; images of one temple are correlated, so p is optimistic"}
