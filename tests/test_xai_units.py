import numpy as np

from templexai.models.component_features import COMPONENTS, PER_COMPONENT, component_vector
from templexai.xai.geometry import IMG, box_to_crop, cam_to_original
from templexai.xai.shap_components import component_shap

K = len(PER_COMPONENT)


def test_shapley_efficiency_and_dummy():
    # toy model: P(class0) = 0.2 + 0.5 * [shikhara present] + 0.1 * [amalaka present]
    def pp(X):
        p0 = 0.2 + 0.5 * (X[:, 0] > 0.5) + 0.1 * (X[:, 6 * K] > 0.5)
        return np.stack([p0, 1 - p0, np.zeros_like(p0)], 1)

    dets = [{"label": "shikhara", "box_xyxy": [0, 0, 10, 10], "score": .9}, {"label": "amalaka", "box_xyxy": [0, 0, 5, 5], "score": .8},
            {"label": "pillar", "box_xyxy": [0, 0, 5, 5], "score": .7}]
    v = component_vector(dets, 100, 100)
    s = component_shap(pp, v, 0)
    assert abs(s["shikhara"] - 0.5) < 1e-9 and abs(s["amalaka"] - 0.1) < 1e-9 and abs(s["pillar"]) < 1e-9  # dummy player gets 0
    assert abs(sum(s.values()) - (pp(v[None])[0, 0] - pp(np.zeros((1, len(v))))[0, 0])) < 1e-9  # efficiency
    assert component_shap(pp, np.zeros(len(v), dtype=np.float32), 0) == {}


def test_geometry_roundtrip():
    w, h = 400, 300
    b = box_to_crop([0, 0, w, h], w, h)
    assert 0 <= b[0] <= b[2] <= IMG and 0 <= b[1] <= b[3] <= IMG
    cam = np.ones((IMG, IMG), dtype=np.float32)
    full = cam_to_original(cam, w, h)
    assert full.shape == (h, w) and full.max() > 0.99
