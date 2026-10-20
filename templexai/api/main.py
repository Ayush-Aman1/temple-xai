"""FastAPI backend. Models load once at startup. Run: uvicorn templexai.api.main:app"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

from templexai.knowledge.taxonomy import load_taxonomy
from templexai.models.analyze import analyze, prepare
from templexai.models.serving import load_bundle

DATA = Path("data")
MAX_BYTES = 15 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = load_bundle()
    app.state.taxonomy = load_taxonomy()
    meta = pd.read_csv(DATA / "meta" / "images.csv").set_index("image_id")
    app.state.meta = meta
    p = Path("artifacts/p4/test_predictions.csv")
    app.state.test = pd.read_csv(p) if p.exists() else pd.DataFrame()
    yield


app = FastAPI(title="Temple XAI", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])


def _attr(app, image_id: str) -> dict:
    r = app.state.meta.loc[image_id]
    return {"image_id": image_id, "licence": r.licence, "author": None if pd.isna(r.author) else r.author, "source_url": r.url}


@app.get("/health")
def health():
    b = app.state.bundle
    return {"status": "ok", "device": b.device, "provenance": b.provenance}


@app.get("/taxonomy")
def taxonomy():
    t = app.state.taxonomy
    return {"knowledge_mode": t["knowledge_mode"], "caveat": t["operational_taxonomy_caveat"], "components": t["components"], "traditions": t["traditions"]}


@app.post("/analyze")
async def analyze_upload(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "image larger than 15 MB")
    try:
        im = prepare(raw)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"not a readable image: {e}") from e
    return analyze(app.state.bundle, im)


@app.get("/gallery")
def gallery(per_group: int = 6):
    """Test-split examples: correct and incorrect predictions of the serving fusion model, with attribution."""
    t = app.state.test
    if t.empty:
        return {"correct": [], "incorrect": []}
    trad = ["Nagara", "Dravida", "Deccan_Vesara"]
    t = t.assign(ok=t.y == t.pred_fusion)

    def pick(df):
        out = []
        for c in range(3):  # spread across traditions, one image per temple
            sub = df[df.y == c].drop_duplicates("temple_id").sort_values("image_id").head(max(1, per_group // 3))
            for r in sub.itertuples():
                out.append({**_attr(app, r.image_id), "true": trad[r.y], "predicted": trad[r.pred_fusion]})
        return out

    return {"correct": pick(t[t.ok]), "incorrect": pick(t[~t.ok])}


def _test_image(image_id: str) -> Path:
    t = app.state.test
    if t.empty or image_id not in set(t.image_id):
        raise HTTPException(404, "not a test-split image")
    return DATA / app.state.meta.loc[image_id].file


@app.get("/gallery/image/{image_id}")
def gallery_image(image_id: str):
    return FileResponse(_test_image(image_id))


@app.post("/gallery/analyze/{image_id}")
def gallery_analyze(image_id: str):
    res = analyze(app.state.bundle, prepare(Image.open(_test_image(image_id))))
    res["attribution"] = _attr(app, image_id)
    row = app.state.test[app.state.test.image_id == image_id].iloc[0]
    res["true_tradition"] = ["Nagara", "Dravida", "Deccan_Vesara"][int(row.y)]
    return res


@app.get("/method")
def method():
    """Provenance + headline metrics (read from the metrics files written by code)."""
    out = {"provenance": app.state.bundle.provenance}
    for k, p in (("p4", "artifacts/p4/metrics.json"), ("explanations", "experiments/results/explanations/summary.json")):
        if Path(p).exists():
            out[k] = json.loads(Path(p).read_text())
    return out
