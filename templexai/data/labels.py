"""a2: RULE_BASED style labels with cited evidence -> <data-root>/labels/temple_styles_rule.csv.

Evidence per temple: (1) Wikidata P149 labels (already in temples.csv), (2) English Wikipedia infobox
"architecture"/"architecture_style" value, (3) Commons parent-category names (2 levels).
Decision: >=1 source maps to a tradition and all mapped sources agree -> label; any disagreement, or any
source mapping to EXCLUDE -> EXCLUDE. Human labels in <data-root>/labels/temple_styles_human.csv override
per temple (label_mode=HUMAN_VERIFIED). Every row stores each source URL, raw value and retrieval date.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from .common import add_data_root_arg, get_json, today
from .style_map import StyleMap

WIKI_API = "https://en.wikipedia.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
INFOBOX_FIELDS = re.compile(r"^\s*\|\s*(architecture|architecture[ _]style|architectural[ _]style|style)\s*=\s*(.*)$",
                            re.I | re.M)


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def fetch_infobox_values(titles: list[str], cache: Path) -> dict[str, list[str]]:
    """enwiki title -> list of raw infobox architecture strings (after redirect resolution)."""
    out: dict[str, list[str]] = {}
    for batch in _chunks(titles, 20):
        d = get_json(WIKI_API, {"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
                                "redirects": 1, "titles": "|".join(batch), "format": "json",
                                "formatversion": 2}, cache, timeout=120)
        q = d["query"]
        alias: dict[str, str] = {}
        for k in ("normalized", "redirects"):
            for m in q.get(k, []):
                alias[m["to"]] = m["from"]
        for page in q["pages"]:
            if "revisions" not in page:
                continue
            text = page["revisions"][0]["slots"]["main"]["content"]
            vals = [m.group(2).strip() for m in INFOBOX_FIELDS.finditer(text)]
            vals = [re.sub(r"<ref[^>]*?(/>|>.*?</ref>)", "", v, flags=re.S) for v in vals]
            title = page["title"]
            # walk the alias chain back to the original requested title(s)
            origin = title
            seen = 0
            while origin in alias and seen < 5:
                origin = alias[origin]
                seen += 1
            for t in {origin, title}:
                out.setdefault(t, []).extend(v for v in vals if v)
    return out


def fetch_commons_parents(categories: list[str], cache: Path) -> dict[str, list[str]]:
    """Commons category name -> parent category names (batch of 40)."""
    out: dict[str, list[str]] = {}
    for batch in _chunks(categories, 40):
        params = {"action": "query", "prop": "categories", "cllimit": "max", "redirects": 1,
                  "titles": "|".join("Category:" + c for c in batch), "format": "json", "formatversion": 2}
        cont: dict = {}
        alias: dict[str, str] = {}
        while True:
            d = get_json(COMMONS_API, {**params, **cont}, cache, timeout=120)
            q = d["query"]
            for k in ("normalized", "redirects"):
                for m in q.get(k, []):
                    alias[m["to"]] = m["from"]
            for p in q["pages"]:
                names = [c["title"].removeprefix("Category:") for c in p.get("categories", [])]
                key = p["title"].removeprefix("Category:")
                orig = alias.get(p["title"], p["title"]).removeprefix("Category:")
                out.setdefault(key, []).extend(names)
                if orig != key:
                    out.setdefault(orig, []).extend(names)
            if "continue" in d:
                cont = d["continue"]
            else:
                break
    return out


def build(temples: pd.DataFrame, smap: StyleMap, cache_root: Path) -> pd.DataFrame:
    retrieved = today()
    titles = sorted({t for t in temples.enwiki_title.dropna()})
    infobox = fetch_infobox_values(titles, cache_root / "wikipedia") if titles else {}
    cats = sorted(set(temples.commons_category.dropna()))
    parents1 = fetch_commons_parents(cats, cache_root / "commons_cat")
    grand = sorted({p for ps in parents1.values() for p in ps} - set(cats))
    parents2 = fetch_commons_parents(grand, cache_root / "commons_cat")
    rows = []
    for t in temples.itertuples():
        ev: list[dict] = []

        def add(source: str, url: str, raw: str) -> None:
            trad, rule = smap.classify(raw)
            ev.append({"source": source, "url": url, "raw_value": raw, "mapped_to": trad, "rule": rule,
                       "retrieved": retrieved})

        for lbl in [x for x in str(t.wikidata_style_labels or "").split(";") if x and x != "nan"]:
            add("wikidata_P149", f"https://www.wikidata.org/wiki/{t.qid}", lbl)
        if isinstance(t.enwiki_title, str):
            url = "https://en.wikipedia.org/wiki/" + quote(t.enwiki_title.replace(" ", "_"))
            for v in dict.fromkeys(infobox.get(t.enwiki_title, [])):
                add("wikipedia_infobox", url, v)
        if isinstance(t.commons_category, str):
            curl = "https://commons.wikimedia.org/wiki/Category:" + quote(t.commons_category.replace(" ", "_"))
            names = list(parents1.get(t.commons_category, []))
            for p in list(names):
                names += [g for g in parents2.get(p, []) if "architecture" in g.lower()]
            for n in dict.fromkeys(names):
                if not re.search(r"architecture|style|ratna", n, re.I):  # location/topic categories are not style evidence
                    continue
                trad, rule = smap.classify(n)
                if trad is not None:  # only record category names that carry style evidence
                    ev.append({"source": "commons_category", "url": curl, "raw_value": n, "mapped_to": trad,
                               "rule": rule, "retrieved": retrieved})
        mapped = [e["mapped_to"] for e in ev if e["mapped_to"]]
        if not mapped:
            status, trad = "NO_EVIDENCE", None
        elif "EXCLUDE" in mapped:
            status, trad = "EXCLUDED_OUT_OF_TAXONOMY", None
        elif len(set(mapped)) > 1:
            status, trad = "EXCLUDED_CONFLICT", None
        else:
            status, trad = "LABELLED", mapped[0]
        rows.append({"temple_id": t.temple_id, "tradition": trad, "status": status, "label_mode": "RULE_BASED",
                     "n_sources": len(mapped), "sources": ";".join(sorted({e["source"] for e in ev if e["mapped_to"]})),
                     "evidence_json": json.dumps(ev, ensure_ascii=False), "retrieved": retrieved})
    return pd.DataFrame(rows)


def apply_human_override(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        return df
    h = pd.read_csv(path)
    need = {"temple_id", "tradition", "source"}
    if not need <= set(h.columns):
        raise ValueError(f"{path} needs columns {need}")
    h = h[h.source.notna() & (h.source.astype(str).str.strip() != "")]  # unsourced human labels are ignored
    m = df.set_index("temple_id")
    for r in h.itertuples():
        if r.temple_id in m.index:
            m.loc[r.temple_id, ["tradition", "status", "label_mode"]] = [
                None if r.tradition == "exclude" else r.tradition,
                "EXCLUDED_HUMAN" if r.tradition == "exclude" else "LABELLED", "HUMAN_VERIFIED"]
    return m.reset_index()


def coverage(df: pd.DataFrame) -> str:
    lab = df[df.status == "LABELLED"]
    lines = [f"temples={len(df)}", f"labelled={len(lab)}", "per tradition: " + json.dumps(lab.tradition.value_counts().to_dict()),
             "status: " + json.dumps(df.status.value_counts().to_dict())]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_data_root_arg(ap)
    ap.add_argument("--cache", default="data/cache")
    a = ap.parse_args()
    root = Path(a.data_root)
    temples = pd.read_csv(root / "meta" / "temples.csv")
    smap = StyleMap()
    df = build(temples, smap, Path(a.cache))
    df = apply_human_override(df, Path("data/labels/temple_styles_human.csv"))
    (root / "labels").mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "labels" / "temple_styles_rule.csv", index=False)
    print("a2 " + coverage(df).replace("\n", "\na2 "))


if __name__ == "__main__":
    main()
