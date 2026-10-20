"""Load configs/style_map.yaml and classify raw style strings."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

DEFAULT = Path("configs/style_map.yaml")
VERIFIED = Path("configs/style_map_verified.yaml")  # written by verify_style_map; lists verified rule ids


def normalise(s: str) -> str:
    s = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", s)  # wikilink -> display text
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\{\{[^}]*\}\}", " ", s)
    s = s.replace("_", " ").lower()
    return re.sub(r"\s+", " ", s).strip(" ,;.*")


class StyleMap:
    def __init__(self, path: Path = DEFAULT, verified_path: Path = VERIFIED, require_verified: bool = True):
        cfg = yaml.safe_load(Path(path).read_text())
        self.generic = {normalise(g) for g in cfg["generic"]}
        self.rules = cfg["rules"]
        self.taxonomy = cfg["taxonomy"]
        self.verified: set[str] | None = None
        if require_verified and Path(verified_path).exists():
            self.verified = set(yaml.safe_load(Path(verified_path).read_text())["verified_rule_ids"])

    def classify(self, raw: str) -> tuple[str | None, str | None]:
        """Return (tradition|'EXCLUDE'|None, rule_id). None = no evidence (generic/unmapped/unverified)."""
        n = normalise(raw)
        if not n or n in self.generic:
            return None, None
        for r in self.rules:
            if any(m in n for m in r["match"]):
                if self.verified is not None and r["id"] not in self.verified:
                    return None, r["id"]
                return r["tradition"], r["id"]
        return None, None
