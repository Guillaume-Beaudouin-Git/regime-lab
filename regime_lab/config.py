"""Paths and environment configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
MANIFESTS = DATA / "manifests"
CACHE = DATA / "cache"

for _d in (RAW, MANIFESTS, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

FRED_API_KEY: str | None = os.getenv("FRED_API_KEY") or None

#: First date the research sample may reference.
SAMPLE_START = "1990-01-01"
