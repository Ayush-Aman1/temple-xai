"""Load the taxonomy; taxonomy_reviewed.yaml (human add-on) overrides per entry and marks that entry HUMAN_REVIEWED."""
from __future__ import annotations

from pathlib import Path

import yaml

KDIR = Path(__file__).parent
REVIEWED = KDIR / "taxonomy_reviewed.yaml"


def load_taxonomy(reviewed: Path | None = None) -> dict:
    tax = yaml.safe_load((KDIR / "taxonomy.yaml").read_text())
    rev_path = REVIEWED if reviewed is None else reviewed
    n_over = 0
    if rev_path.exists():
        rev = yaml.safe_load(rev_path.read_text()) or {}
        for kind in ("components", "traditions"):
            for name, e in (rev.get(kind) or {}).items():
                if name in tax[kind]:
                    tax[kind][name] = {**tax[kind][name], **e, "knowledge_mode": "HUMAN_REVIEWED"}
                    n_over += 1
    total = len(tax["components"]) + len(tax["traditions"])
    tax["knowledge_mode"] = "MACHINE_COMPILED" if n_over == 0 else ("HUMAN_REVIEWED" if n_over >= total else "PARTLY_HUMAN_REVIEWED")
    return tax


def diagnostic_links(tax: dict | None = None) -> dict[str, set[str]]:
    """component -> set of traditions it is cited as evidence for (the knowledge graph used by the metrics)."""
    tax = tax or load_taxonomy()
    return {c: {l["tradition"] for l in e.get("links", [])} for c, e in tax["components"].items()}
