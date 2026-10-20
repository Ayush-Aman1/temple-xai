"""pydantic schemas for the CSV contracts (temples, images, splits)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

Tradition = Literal["Nagara", "Dravida", "Deccan_Vesara"]


class TempleRow(BaseModel):
    temple_id: str
    name: str
    qid: str
    commons_category: str
    enwiki_title: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    state: Optional[str] = None
    wikidata_style_qids: Optional[str] = None
    wikidata_style_labels: Optional[str] = None


class ImageRow(BaseModel):
    image_id: str
    temple_id: str
    file: str
    url: str
    licence: str
    author: str
    sha1: str
    phash: str
    image_type: Optional[str] = None
    image_type_score: Optional[float] = None


class SplitRow(BaseModel):
    temple_id: str
    tradition: Tradition
    split: Literal["train", "val", "test"]
