"""Structured evidence object per prediction + template-based report with inline citations.

The class comes from the vision model only; nothing here (and no optional phrasing module) can change it.
"""
from __future__ import annotations

from templexai.knowledge.taxonomy import diagnostic_links, load_taxonomy

CAVEAT = ("Tradition labels follow the project's operational taxonomy (Nagara / Dravida / Deccan-Vesara), built from open sources. "
          "Real temples mix traits and scholars debate these categories; treat the output as evidence for a grouping, not a verdict.")


def build_evidence(pred: dict, detections: list[dict], shap_by_component: dict[str, float] | None = None,
                   cam_energy: dict[str, float] | None = None, tax: dict | None = None, max_components: int = 5) -> dict:
    """pred: {tradition, probability, probabilities{trad: p}}. detections: [{label, box_xyxy, score}].

    Returns {prediction, components[], shap_top[], taxonomy{tradition, components}, caveat, sources[]}; `sources` is the numbered
    reference list the report cites.
    """
    tax = tax or load_taxonomy()
    links = diagnostic_links(tax)
    best: dict[str, dict] = {}
    for d in detections:
        if d["label"] not in best or d["score"] > best[d["label"]]["score"]:
            best[d["label"]] = d
    comps = []
    for lab, d in sorted(best.items(), key=lambda kv: -kv[1]["score"])[:max_components]:
        comps.append({"label": lab, "display_name": tax["components"][lab]["display_name"], "score": d["score"], "box_xyxy": d["box_xyxy"],
                      "supports_prediction": pred["tradition"] in links.get(lab, set()),
                      "linked_traditions": sorted(links.get(lab, set())),
                      "shap": (shap_by_component or {}).get(lab), "cam_energy": (cam_energy or {}).get(lab)})
    shap_top = sorted(((k, v) for k, v in (shap_by_component or {}).items()), key=lambda kv: -abs(kv[1]))[:5]
    refs: list[dict] = []

    def ref(src: dict) -> int:
        key = (src["url"], src["quote"])
        for i, r in enumerate(refs):
            if (r["url"], r["quote"]) == key:
                return i + 1
        refs.append({"url": src["url"], "page": src["page"], "retrieved": src["retrieved"], "quote": src["quote"], "licence": src.get("licence", "")})
        return len(refs)

    t = tax["traditions"][pred["tradition"]]
    trad_entry = {"name": t["display_name"], "definition": t["definition"], "visual_cues": t["visual_cues"],
                  "refs": [ref(s) for s in t["sources"][:2]], "knowledge_mode": t["knowledge_mode"]}
    comp_entries = []
    for c in comps:
        e = tax["components"][c["label"]]
        c_refs = [ref(l) for l in e.get("links", []) if l["tradition"] == pred["tradition"]] or [ref(e["sources"][0])]
        comp_entries.append({"label": c["label"], "name": e["display_name"], "definition": e["definition"], "refs": c_refs,
                             "knowledge_mode": e["knowledge_mode"]})
    return {"prediction": pred, "components": comps, "shap_top": [{"component": k, "value": v} for k, v in shap_top],
            "taxonomy": {"tradition": trad_entry, "components": comp_entries}, "caveat": CAVEAT, "sources": refs}


def render_report(ev: dict) -> str:
    """Markdown report. Every taxonomy statement carries [n] citations resolved in the References list."""
    p = ev["prediction"]
    tr = ev["taxonomy"]["tradition"]
    cite = lambda refs: "".join(f"[{n}]" for n in refs)  # noqa: E731
    lines = [f"## Predicted tradition: {tr['name']} ({p['probability']:.0%})", "",
             "Probabilities: " + ", ".join(f"{k} {v:.0%}" for k, v in p["probabilities"].items()) + ".", "",
             f"**{tr['name']}** — {tr['definition']} {cite(tr['refs'])}", ""]
    if ev["components"]:
        lines += ["### Detected components", ""]
        entries = {e["label"]: e for e in ev["taxonomy"]["components"]}
        for c in ev["components"]:
            e = entries[c["label"]]
            tie = (f"cited as evidence for {tr['name']}" if c["supports_prediction"]
                   else ("not cited as specific to " + tr['name'] if c["linked_traditions"] else "not specific to a single tradition in the sources"))
            extra = f"; SHAP contribution {c['shap']:+.3f}" if c["shap"] is not None else ""
            lines.append(f"- **{e['name']}** (detector score {c['score']:.2f}{extra}) — {e['definition']} {cite(e['refs'])} Here: {tie}.")
        lines.append("")
    else:
        lines += ["No architectural components were detected above threshold, so this prediction rests on the image model alone.", ""]
    if ev["shap_top"]:
        lines += ["### What drove the component-based explanation (SHAP)", "",
                  ", ".join(f"{s['component']} ({s['value']:+.3f})" for s in ev["shap_top"]) + ".", ""]
    lines += [f"> {ev['caveat']}", "", "### References", ""]
    for i, r in enumerate(ev["sources"], 1):
        lines.append(f"{i}. [{r['page']}]({r['url']}) — retrieved {r['retrieved']}, {r['licence']}: “{r['quote']}”")
    return "\n".join(lines)
