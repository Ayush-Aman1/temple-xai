"""Validator for taxonomy.yaml: every entry has a definition, visual cues and >=1 source with URL + quote, and every quote
(<= 25 words) appears verbatim in the saved snapshot of its page (data/knowledge_sources/)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "data" / "knowledge_sources"
TAXONOMY = Path(__file__).with_name("taxonomy.yaml")


def _norm(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", s).strip()


def snapshot_text(url: str, snap_dir: Path = SNAP) -> str | None:
    """Text of the snapshot whose url matches, else None."""
    for f in snap_dir.glob("*.json"):
        d = json.loads(f.read_text())
        if d["url"] == url:
            return _norm(d["text"])
    return None


def validate(path: Path = TAXONOMY, snap_dir: Path = SNAP) -> list[str]:
    """Return a list of problems (empty = valid)."""
    tax = yaml.safe_load(Path(path).read_text())
    errs: list[str] = []
    for kind in ("components", "traditions"):
        for name, e in tax[kind].items():
            tag = f"{kind}/{name}"
            if not e.get("definition"):
                errs.append(f"{tag}: no definition")
            if not e.get("visual_cues"):
                errs.append(f"{tag}: no visual_cues")
            if not e.get("sources"):
                errs.append(f"{tag}: no sources")
            if not e.get("knowledge_mode"):
                errs.append(f"{tag}: no knowledge_mode")
            for s in list(e.get("sources", [])) + list(e.get("links", [])):
                if not s.get("url") or not s.get("quote"):
                    errs.append(f"{tag}: source without url/quote")
                    continue
                if len(s["quote"].split()) > 25:
                    errs.append(f"{tag}: quote > 25 words: {s['quote'][:40]}")
                txt = snapshot_text(s["url"], snap_dir)
                if txt is None:
                    errs.append(f"{tag}: no snapshot for {s['url']}")
                elif _norm(s["quote"]) not in txt:
                    errs.append(f"{tag}: quote not verbatim in snapshot: {s['quote'][:50]}")
    return errs


if __name__ == "__main__":
    problems = validate()
    print("OK" if not problems else "\n".join(problems))
    raise SystemExit(1 if problems else 0)
