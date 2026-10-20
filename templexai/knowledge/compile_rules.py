"""Compile component -> tradition evidence rules from FETCHED Wikipedia text (knowledge_mode=MACHINE_COMPILED).

For every candidate (component, tradition) pair we look for a sentence in the fetched pages that mentions BOTH
a component term and a tradition term. Only pairs with such a sentence become rules; each rule stores the page
URL, retrieved date and a quote of <= 25 words. No fetched sentence -> no rule (integrity rule 2).
Output: configs/knowledge_rules.yaml and docs/knowledge_sources.md.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from templexai.data.common import get_json, today

API = "https://en.wikipedia.org/w/api.php"
PAGES = ["Hindu temple architecture", "Shikhara", "Vimana", "Gopuram", "Mandapa", "Amalaka", "Jagati (architecture)",
         "Dravidian architecture", "Nagara (architecture)", "Vesara", "Hoysala architecture", "Kalinga architecture",
         "Pillar"]
COMPONENT_TERMS = {
    "shikhara": ["shikhara", "shikara", "sikhara", "curvilinear tower"],
    "vimana": ["vimana", "vimanam"],
    "gopuram": ["gopuram", "gopura"],
    "mandapa": ["mandapa", "mandapam", "mantapa"],
    "pillar": ["pillar", "pillared", "column"],
    "jagati_base": ["jagati", "platform", "plinth", "adhisthana"],
    "amalaka": ["amalaka"],
}
TRADITION_TERMS = {
    "Nagara": ["nagara", "north indian"],
    "Dravida": ["dravida", "dravidian", "south indian"],
    "Deccan_Vesara": ["vesara", "karnata dravida", "deccan", "hoysala", "chalukya"],
}


def fetch(title: str, cache: Path) -> tuple[str, str]:
    d = get_json(API, {"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1, "titles": title,
                       "format": "json", "formatversion": 2}, cache)
    p = d["query"]["pages"][0]
    return p.get("extract", "") or "", "https://en.wikipedia.org/wiki/" + p.get("title", title).replace(" ", "_")


def trim(sent: str) -> str:
    w = sent.split()
    return " ".join(w[:25]) + (" …" if len(w) > 25 else "")


def main(cache_dir: str = "data/cache/wikipedia_extracts") -> None:
    texts = {t: fetch(t, Path(cache_dir)) for t in PAGES}
    rules, seen = [], set()
    for page, (text, url) in texts.items():
        for sent in re.split(r"(?<=[.!?])\s+", text):
            low = sent.lower()
            if len(sent.split()) > 70:
                continue
            for comp, cterms in COMPONENT_TERMS.items():
                if not any(re.search(rf"\b{re.escape(t)}", low) for t in cterms):
                    continue
                for trad, tterms in TRADITION_TERMS.items():
                    if not any(re.search(rf"\b{re.escape(t)}", low) for t in tterms):
                        continue
                    key = (comp, trad, page)
                    if key in seen:
                        continue
                    seen.add(key)
                    rules.append({"component": comp, "tradition": trad, "url": url, "page": page,
                                  "retrieved": today(), "quote": trim(sent.replace("\n", " "))})
    out = {"knowledge_mode": "MACHINE_COMPILED", "retrieved": today(),
           "note": "Co-occurrence rules from fetched Wikipedia sentences; each is evidence that a component is "
                   "discussed with a tradition, not a verdict. Used as soft features by the knowledge-informed "
                   "variants; weights are learned from data, never taken from this file.",
           "rules": rules}
    Path("configs").mkdir(exist_ok=True)
    Path("configs/knowledge_rules.yaml").write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True))
    # component x tradition support matrix (count of distinct supporting pages) -> used as prior in variant C
    support: dict[str, dict[str, int]] = {c: {t: 0 for t in TRADITION_TERMS} for c in COMPONENT_TERMS}
    for r in rules:
        support[r["component"]][r["tradition"]] += 1
    Path("configs/knowledge_support.yaml").write_text(yaml.safe_dump(
        {"knowledge_mode": "MACHINE_COMPILED", "retrieved": today(), "support": support}, sort_keys=False))
    md = [f"# Knowledge sources (machine-compiled, retrieved {today()})", "",
          "| component | tradition | page | quote (≤25 words) |", "|---|---|---|---|"]
    for r in rules:
        md.append(f"| {r['component']} | {r['tradition']} | [{r['page']}]({r['url']}) | {r['quote'].replace('|', '/')} |")
    Path("docs").mkdir(exist_ok=True)
    Path("docs/knowledge_sources.md").write_text("\n".join(md) + "\n")
    print(f"knowledge: {len(rules)} rules from {len(PAGES)} pages; support={support}")


if __name__ == "__main__":
    main()
