"""Compile knowledge/taxonomy.yaml from taxonomy_spec.yaml + fetched snapshots (knowledge_mode=MACHINE_COMPILED).

Resolves each source's page title to its snapshot URL and retrieval date, keeps the existing display names / detector
prompts, writes the file and runs the validator. Also writes configs/diagnostic_links.yaml (component -> tradition links,
each with its cited sentence), the knowledge graph used by the explanation metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from templexai.knowledge.validate import SNAP, TAXONOMY, validate

SPEC = Path(__file__).with_name("taxonomy_spec.yaml")
PER_COMPONENT = ["present", "max_score", "count", "total_rel_area", "centroid_y_largest", "aspect_largest"]


def _snapshots() -> dict[str, dict]:
    return {d["title"]: d for d in (json.loads(f.read_text()) for f in SNAP.glob("*.json"))}


def _resolve(items: list[dict], snaps: dict[str, dict]) -> list[dict]:
    out = []
    for it in items:
        s = snaps[it["page"]]
        row = {k: v for k, v in it.items() if k != "page"}
        out.append({**row, "page": it["page"], "url": s["url"], "retrieved": s["retrieved"], "licence": s["licence"]})
    return out


def main() -> None:
    spec = yaml.safe_load(SPEC.read_text())
    old = yaml.safe_load(TAXONOMY.read_text())
    snaps = _snapshots()
    tax = {"knowledge_mode": "MACHINE_COMPILED",
           "note": "Compiled from data/knowledge_sources snapshots; definitions paraphrase the quoted sentences. "
                   "taxonomy_reviewed.yaml (if present) overrides per entry -> HUMAN_REVIEWED.",
           "operational_taxonomy_caveat": "Nagara / Dravida / Deccan-Vesara is an operational grouping used by this project; "
                                          "real temples mix traits and scholars debate the categories.",
           "components": {}, "traditions": {}}
    links = []
    for kind in ("components", "traditions"):
        for name, e in spec[kind].items():
            prev = old[kind].get(name, {})
            ent = {"display_name": prev.get("display_name", name), "knowledge_mode": "MACHINE_COMPILED",
                   "definition": e["definition"], "visual_cues": e["visual_cues"], "related": e.get("related", []),
                   "sources": _resolve(e["sources"], snaps)}
            if kind == "components":
                ent["prompts"] = prev.get("prompts", [])
                ent["links"] = _resolve(e.get("links", []), snaps)
                for l in ent["links"]:
                    links.append({"component": name, "tradition": l["tradition"], "url": l["url"], "retrieved": l["retrieved"],
                                  "quote": l["quote"]})
            tax[kind][name] = ent
    tax["component_feature_order"] = PER_COMPONENT
    TAXONOMY.write_text(yaml.safe_dump(tax, sort_keys=False, allow_unicode=True, width=120))
    Path("configs/diagnostic_links.yaml").write_text(yaml.safe_dump(
        {"knowledge_mode": "MACHINE_COMPILED", "note": "component -> tradition evidence links, each with the fetched sentence supporting it",
         "links": links}, sort_keys=False, allow_unicode=True, width=120))
    errs = validate()
    print(f"taxonomy: {len(tax['components'])} components, {len(tax['traditions'])} traditions, {len(links)} links;",
          "valid" if not errs else f"{len(errs)} PROBLEMS")
    for e in errs:
        print(" -", e)
    raise SystemExit(1 if errs else 0)


if __name__ == "__main__":
    main()
