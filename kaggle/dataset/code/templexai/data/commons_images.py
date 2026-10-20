"""a3: Commons images for labelled temples -> <data-root>/images/ + <data-root>/meta/images.csv.

Licences kept: public domain, CC0, CC BY, CC BY-SA only. <=40 images per temple, longest side 1024 px,
perceptual-hash dedupe (distance <= 6). Per-temple results are stored in meta/images_parts/ so re-runs resume.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import re
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image

from .common import add_data_root_arg, download_file, get_json, sha1_file

API = "https://commons.wikimedia.org/w/api.php"
MAX_PER_TEMPLE = 40
CANDIDATES = 70
PHASH_DIST = 6
OK_MIME = {"image/jpeg", "image/png"}


def licence_ok(short: str, usage_terms: str = "") -> bool:
    """True for PD / CC0 / CC BY / CC BY-SA (any version). NC, ND, GFDL-only, fair use are rejected."""
    s = f"{short} {usage_terms}".lower()
    if re.search(r"\b(nc|nd)\b|non-?commercial|no.?deriv|fair use|gfdl only|all rights reserved", s):
        return False
    if re.search(r"public domain|\bpd\b|\bpd-|cc0|cc-zero", s):
        return True
    return bool(re.search(r"cc[- ]by(-sa)?[- ]\d|attribution(-sharealike)?\s+\d", s))


def strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def list_files(category: str, cache: Path, with_subcats: bool = True) -> list[str]:
    titles: list[str] = []
    cats = [category]
    if with_subcats:
        d = get_json(API, {"action": "query", "list": "categorymembers", "cmtitle": f"Category:{category}",
                           "cmtype": "subcat", "cmlimit": 30, "format": "json"}, cache)
        cats += [m["title"].removeprefix("Category:") for m in d["query"]["categorymembers"]]
    for c in cats:
        cont: dict = {}
        while len(titles) < 400:
            d = get_json(API, {"action": "query", "list": "categorymembers", "cmtitle": f"Category:{c}",
                               "cmtype": "file", "cmlimit": 200, "format": "json", **cont}, cache)
            titles += [m["title"] for m in d["query"]["categorymembers"]]
            if "continue" in d:
                cont = d["continue"]
            else:
                break
    return list(dict.fromkeys(titles))


def _batch_info(titles: list[str], cache: Path) -> list[dict]:
    d = get_json(API, {"action": "query", "prop": "imageinfo", "titles": "|".join(titles),
                       "iiprop": "url|extmetadata|mime|size", "iiurlwidth": 1024, "iiurlheight": 1024,
                       "format": "json", "formatversion": 2}, cache, timeout=120)
    out = []
    for p in d.get("query", {}).get("pages", []):
        ii = (p.get("imageinfo") or [None])[0]
        if not ii:
            continue
        em = ii.get("extmetadata", {})
        lic = em.get("LicenseShortName", {}).get("value", "")
        terms = em.get("UsageTerms", {}).get("value", "")
        if ii.get("mime") not in OK_MIME or not licence_ok(lic, terms):
            continue
        if min(ii.get("width", 0), ii.get("height", 0)) < 300:
            continue
        out.append({"title": p["title"], "thumb": ii.get("thumburl") or ii["url"], "page": ii["descriptionurl"],
                    "licence": strip_html(lic) or strip_html(terms), "author": strip_html(em.get("Artist", {}).get("value", "")) or "unknown",
                    "width": ii.get("width"), "height": ii.get("height")})
    return out


def image_infos(titles: list[str], cache: Path) -> list[dict]:
    """Batched imageinfo; if a batch fails (e.g. one un-normalisable title) retry its titles one by one and skip the bad ones."""
    out = []
    for i in range(0, len(titles), 40):
        batch = titles[i:i + 40]
        try:
            out += _batch_info(batch, cache)
        except RuntimeError:
            for t in batch:
                try:
                    out += _batch_info([t], cache)
                except RuntimeError:
                    continue
    return out


def collect_temple(temple_id: str, category: str, root: Path, cache: Path) -> list[dict]:
    part = root / "meta" / "images_parts" / f"{temple_id}.csv"
    if part.exists():
        return pd.read_csv(part).to_dict("records")
    infos = image_infos(list_files(category, cache)[:CANDIDATES], cache)
    kept: list[dict] = []
    hashes: list[imagehash.ImageHash] = []
    img_dir = root / "images" / temple_id
    for inf in infos:
        if len(kept) >= MAX_PER_TEMPLE:
            break
        image_id = hashlib.sha1(inf["title"].encode()).hexdigest()[:12]
        dest = img_dir / f"{image_id}.jpg"
        tmp = dest.with_suffix(".dl")
        if not dest.exists():
            if not download_file(inf["thumb"], tmp):
                continue
            try:
                with Image.open(tmp) as im:
                    im = im.convert("RGB")
                    im.thumbnail((1024, 1024))
                    im.save(dest, "JPEG", quality=90)
            except Exception:  # corrupt download
                tmp.unlink(missing_ok=True)
                continue
            tmp.unlink(missing_ok=True)
        with Image.open(dest) as im:
            ph = imagehash.phash(im)
        if any(ph - h <= PHASH_DIST for h in hashes):
            dest.unlink(missing_ok=True)
            continue
        hashes.append(ph)
        kept.append({"image_id": image_id, "temple_id": temple_id, "file": str(dest.relative_to(root)),
                     "url": inf["page"], "licence": inf["licence"], "author": inf["author"],
                     "sha1": sha1_file(dest), "phash": str(ph), "commons_title": inf["title"]})
    part.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(kept, columns=["image_id", "temple_id", "file", "url", "licence", "author", "sha1", "phash",
                                "commons_title"]).to_csv(part, index=False)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_data_root_arg(ap)
    ap.add_argument("--cache", default="data/cache")
    a = ap.parse_args()
    root = Path(a.data_root)
    temples = pd.read_csv(root / "meta" / "temples.csv").set_index("temple_id")
    lab = pd.read_csv(root / "labels" / "temple_styles_rule.csv")
    ids = lab[lab.status == "LABELLED"].temple_id.tolist()
    if a.smoke and not ids:  # nothing labelled in the smoke subset: still exercise the image path
        ids = lab.temple_id.tolist()[:3]
    if a.smoke:
        ids = ids[:5]
    rows: list[dict] = []
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=8) as ex:  # network latency dominates; request starts stay <= 5/s globally
        futs = [ex.submit(collect_temple, tid, temples.loc[tid, "commons_category"], root, Path(a.cache) / "commons") for tid in ids]
        for n, f in enumerate(as_completed(futs), 1):
            rows += f.result()
            if n % 10 == 0:
                print(f"a3 {n}/{len(ids)} temples, {len(rows)} images", flush=True)
    rows.sort(key=lambda r: (r["temple_id"], r["image_id"]))
    df = pd.DataFrame(rows)
    df.to_csv(root / "meta" / "images.csv", index=False)
    print(f"a3: {len(df)} images from {df.temple_id.nunique() if len(df) else 0} temples -> {root/'meta'/'images.csv'}")


if __name__ == "__main__":
    main()
