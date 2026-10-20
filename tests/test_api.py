import base64
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("fastapi")
ART = Path("artifacts/p4/fusion.pt")
pytestmark = pytest.mark.skipif(not ART.exists(), reason="trained artifacts not present (run make setup / pipelines)")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from templexai.api.main import app

    with TestClient(app) as c:
        yield c


def test_health_and_provenance(client):
    j = client.get("/health").json()
    assert j["status"] == "ok"
    for k in ("label_mode", "knowledge_mode", "detector_mode", "gold_eval", "models", "git_commit"):
        assert k in j["provenance"]


def test_taxonomy(client):
    j = client.get("/taxonomy").json()
    assert set(j["components"]) >= {"shikhara", "gopuram"} and "operational" in j["caveat"]


def test_analyze_test_image(client):
    t = pd.read_csv("artifacts/p4/test_predictions.csv")
    iid = t.image_id.iloc[0]
    meta = pd.read_csv("data/meta/images.csv").set_index("image_id")
    raw = (Path("data") / meta.loc[iid].file).read_bytes()
    r = client.post("/analyze", files={"file": ("x.jpg", raw, "image/jpeg")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["prediction"]["tradition"] in ("Nagara", "Dravida", "Deccan_Vesara")
    assert abs(sum(j["prediction"]["probabilities"].values()) - 1) < 1e-5
    assert base64.b64decode(j["gradcam_png_base64"])[:4] == b"\x89PNG"
    assert "operational taxonomy" in j["report_markdown"] and "References" in j["report_markdown"]
    assert j["provenance"]["detector_mode"] and "shap" in j
    assert all(len(c["box_xyxy"]) == 4 for c in j["components"])


def test_bad_upload(client):
    assert client.post("/analyze", files={"file": ("x.txt", b"not an image", "text/plain")}).status_code == 400


def test_gallery_and_guard(client):
    g = client.get("/gallery").json()
    assert g["correct"] and g["incorrect"] and g["correct"][0]["licence"]
    assert client.get("/gallery/image/../../etc/passwd").status_code in (404, 422)
    assert client.get("/gallery/image/notanid").status_code == 404
    iid = g["incorrect"][0]["image_id"]
    r = client.post(f"/gallery/analyze/{iid}")
    assert r.status_code == 200 and r.json()["attribution"]["source_url"].startswith("http")
