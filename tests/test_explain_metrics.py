import numpy as np

from templexai.eval import explain_metrics as M
from templexai.eval.metrics import mcnemar


def test_rank_masks_monotone():
    sal = np.random.default_rng(0).random((8, 8))
    ms = M.rank_masks(sal)
    assert ms[0].sum() == 0 and ms[-1].all()
    assert all(a.sum() <= b.sum() for a, b in zip(ms, ms[1:]))


def test_deletion_insertion_oracle_beats_random():
    # prob = mean brightness of left half: an oracle saliency (left half) must delete faster than a wrong one
    H = M.IMG
    x = np.ones((3, H, H), dtype=np.float32)
    f = lambda b: b[:, :, :, : H // 2].mean((1, 2, 3))  # noqa: E731
    good = np.zeros((H, H)); good[:, : H // 2] = 1
    bad = 1 - good
    base = np.zeros_like(x)
    dg, ig = M.deletion_insertion_curves(f, x, good, base, base)
    db, ib = M.deletion_insertion_curves(f, x, bad, base, base)
    assert M.auc(dg) < M.auc(db) and M.auc(ig) > M.auc(ib)


def test_gold_metrics_on_synthetic_fixture():
    cam = np.zeros((M.IMG, M.IMG)); cam[40:80, 40:80] = 1
    inside, outside = [[30, 30, 90, 90]], [[120, 120, 180, 180]]
    assert M.pointing_hit(cam, inside) and not M.pointing_hit(cam, outside)
    assert abs(M.energy_inside(cam, inside) - 1.0) < 1e-9 and M.energy_inside(cam, outside) == 0.0
    assert M.explanation_iou(cam, [[40, 40, 80, 80]], 0.0434) > 0.9
    gts = {"a": [{"label": "pillar", "box": [10, 10, 50, 50]}], "b": [{"label": "pillar", "box": [0, 0, 20, 20]}]}
    perfect = [{"image": "a", "label": "pillar", "box": [10, 10, 50, 50], "score": .9}, {"image": "b", "label": "pillar", "box": [0, 0, 20, 20], "score": .8}]
    r = M.detector_map(perfect, gts, ["pillar"])
    assert abs(r["mAP50"] - 1) < 1e-6 and abs(r["mAP50_95"] - 1) < 1e-6
    wrong = [{"image": "a", "label": "pillar", "box": [100, 100, 150, 150], "score": .9}]
    assert M.detector_map(wrong, gts, ["pillar"])["mAP50"] == 0


def test_temple_bootstrap_and_mcnemar():
    v = np.array([1., 1., 0., 0.]); t = np.array(["a", "a", "b", "b"])
    m, lo, hi = M.temple_bootstrap(v, t, n=200)
    assert lo <= m <= hi
    r = mcnemar([0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0])
    assert r["b"] == 0 and r["c"] == 2 and 0 < r["p_value"] <= 1


def test_load_gold_both_formats(tmp_path):
    import json

    (tmp_path / "abc.txt").write_text("4 0.5 0.5 0.2 0.4\n")
    (tmp_path / "export.json").write_text(json.dumps([{"data": {"image": "/data/local-files/?d=gold/images/xyz.jpg"}, "annotations": [{"result": [
        {"type": "rectanglelabels", "value": {"x": 10, "y": 20, "width": 30, "height": 40, "rectanglelabels": ["gopuram"]}}]}]}]))
    g = M.load_gold(tmp_path)
    assert g["abc"][0]["label"] == "pillar" and abs(g["abc"][0]["box"][0] - 0.4) < 1e-9
    assert g["xyz"][0]["label"] == "gopuram" and abs(g["xyz"][0]["box"][2] - 0.4) < 1e-9
    assert M.has_gold(tmp_path) and not M.has_gold(tmp_path / "nope")
