from pathlib import Path

import yaml

from templexai.knowledge import validate as V
from templexai.knowledge.explain import build_evidence, render_report
from templexai.knowledge.taxonomy import diagnostic_links, load_taxonomy


def test_real_taxonomy_valid():
    assert V.validate() == []


def test_validator_catches_fake_quote(tmp_path):
    tax = yaml.safe_load(V.TAXONOMY.read_text())
    tax["components"]["gopuram"]["sources"][0]["quote"] = "this sentence is not in the snapshot"
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(tax))
    assert any("not verbatim" in e for e in V.validate(p))


def test_validator_catches_missing_source(tmp_path):
    tax = yaml.safe_load(V.TAXONOMY.read_text())
    tax["components"]["pillar"]["sources"] = []
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(tax))
    assert any("no sources" in e for e in V.validate(p))


def test_reviewed_override(tmp_path):
    rev = tmp_path / "r.yaml"
    rev.write_text(yaml.safe_dump({"components": {"gopuram": {"definition": "reviewed text"}}}))
    t = load_taxonomy(rev)
    assert t["components"]["gopuram"]["knowledge_mode"] == "HUMAN_REVIEWED"
    assert t["components"]["pillar"]["knowledge_mode"] == "MACHINE_COMPILED"
    assert t["knowledge_mode"] == "PARTLY_HUMAN_REVIEWED"


def test_links_and_report():
    assert "Dravida" in diagnostic_links()["gopuram"]
    pred = {"tradition": "Dravida", "probability": 0.8, "probabilities": {"Nagara": 0.1, "Dravida": 0.8, "Deccan_Vesara": 0.1}}
    dets = [{"label": "gopuram", "box_xyxy": [0, 0, 10, 10], "score": 0.7}, {"label": "amalaka", "box_xyxy": [0, 0, 5, 5], "score": 0.4}]
    ev = build_evidence(pred, dets, {"gopuram": 0.2, "amalaka": -0.1})
    rep = render_report(ev)
    assert "Dravida" in rep and "[1]" in rep and "operational taxonomy" in rep and "References" in rep
    assert all(r["url"].startswith("https://") for r in ev["sources"])
    assert ev["prediction"]["tradition"] == "Dravida"  # evidence never changes the class
