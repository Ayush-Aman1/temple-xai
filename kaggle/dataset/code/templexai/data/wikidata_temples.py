"""a1: Hindu temples in India from Wikidata -> <data-root>/meta/temples.csv.

IDs verified against the Wikidata API (not from memory): Q842402 Hindu temple, Q668 India,
P17 country, P31 instance of, P279 subclass of, P149 architectural style, P373 Commons category,
P625 coordinate location, P131 located in the administrative entity, Q12443800 state of India.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from .common import add_data_root_arg, get_json, today

SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

QUERY = """
SELECT ?t ?tLabel ?cat ?style ?styleLabel ?coord ?stateLabel ?article WHERE {
  ?t wdt:P31/wdt:P279* wd:Q842402 ; wdt:P17 wd:Q668 ; wdt:P373 ?cat .
  OPTIONAL { ?t wdt:P149 ?style }
  OPTIONAL { ?t wdt:P625 ?coord }
  OPTIONAL { ?t wdt:P131+ ?state . ?state wdt:P31 wd:Q12443800 }
  OPTIONAL { ?article schema:about ?t ; schema:isPartOf <https://en.wikipedia.org/> }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
} LIMIT 20000
"""


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def _parse_coord(wkt: str | None) -> tuple[float | None, float | None]:
    if not wkt:
        return None, None
    m = re.match(r"Point\(([-\d.eE]+) ([-\d.eE]+)\)", wkt)
    return (float(m.group(2)), float(m.group(1))) if m else (None, None)


def fetch(cache_dir: Path) -> pd.DataFrame:
    data = get_json(SPARQL_URL, {"query": QUERY, "format": "json"}, cache_dir, timeout=180)
    rows: dict[str, dict] = {}
    for b in data["results"]["bindings"]:
        qid = _qid(b["t"]["value"])
        r = rows.setdefault(qid, {
            "qid": qid, "name": b.get("tLabel", {}).get("value", qid),
            "commons_category": b["cat"]["value"], "enwiki_title": None,
            "lat": None, "lon": None, "state": None,
            "wikidata_style_qids": set(), "wikidata_style_labels": set()})
        if "style" in b:
            r["wikidata_style_qids"].add(_qid(b["style"]["value"]))
            r["wikidata_style_labels"].add(b["styleLabel"]["value"])
        if "coord" in b and r["lat"] is None:
            r["lat"], r["lon"] = _parse_coord(b["coord"]["value"])
        if "stateLabel" in b and r["state"] is None:
            r["state"] = b["stateLabel"]["value"]
        if "article" in b and r["enwiki_title"] is None:
            r["enwiki_title"] = requests_unquote(b["article"]["value"].rsplit("/wiki/", 1)[-1])
    out = []
    for i, r in enumerate(sorted(rows.values(), key=lambda x: x["qid"])):
        r["temple_id"] = f"T{r['qid'][1:]}"
        r["wikidata_style_qids"] = ";".join(sorted(r["wikidata_style_qids"]))
        r["wikidata_style_labels"] = ";".join(sorted(r["wikidata_style_labels"]))
        out.append(r)
    cols = ["temple_id", "name", "qid", "commons_category", "enwiki_title", "lat", "lon", "state",
            "wikidata_style_qids", "wikidata_style_labels"]
    return pd.DataFrame(out)[cols]


def requests_unquote(s: str) -> str:
    from urllib.parse import unquote
    return unquote(s).replace("_", " ")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_data_root_arg(ap)
    ap.add_argument("--cache", default="data/cache")
    a = ap.parse_args()
    root = Path(a.data_root)
    df = fetch(Path(a.cache) / "wikidata")
    if a.smoke:  # tiny diverse subset: temples whose Wikidata P149 maps to each tradition (real labels, no fabrication)
        picks = [("Dravidian", 3), ("Kalinga", 2), ("Badami Chalukya", 1)]
        parts = [df[df.wikidata_style_labels.str.contains(k, na=False)].head(n) for k, n in picks]
        df = pd.concat(parts).drop_duplicates("temple_id")
    (root / "meta").mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "meta" / "temples.csv", index=False)
    print(f"a1: {len(df)} temples -> {root/'meta'/'temples.csv'} "
          f"({(df.wikidata_style_labels != '').sum()} with Wikidata P149) retrieved {today()}")


if __name__ == "__main__":
    main()
