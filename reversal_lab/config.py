"""Paths and store for this project.

The point-in-time contract, the coverage guard and the statistical machinery all
come from the companion project, which is installed as a dependency and frozen.
What does not come from it is the data directory: writing here would put this
study's downloads inside a repository whose results are published and whose
manifests are part of its audit trail.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from regime_lab.data.pit import validate

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
MANIFESTS = DATA / "manifests"
CACHE = DATA / "cache"
for _d in (RAW, MANIFESTS, CACHE):
    _d.mkdir(parents=True, exist_ok=True)


def write(frame: pd.DataFrame, source: str, name: str, *, origin: str = "") -> Path:
    """Persist a point-in-time frame here, with a content-hashed manifest."""
    frame = validate(frame)
    target = RAW / source / f"{name}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)

    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (MANIFESTS / f"{source}__{name}.json").write_text(
        json.dumps(
            {
                "source": source,
                "name": name,
                "origin": origin,
                "rows": int(len(frame)),
                "period_min": frame["period"].min().date().isoformat(),
                "period_max": frame["period"].max().date().isoformat(),
                "sha256": digest,
                "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n"
    )
    return target


def read(source: str, name: str) -> pd.DataFrame:
    """Load a stored point-in-time frame."""
    return validate(pd.read_parquet(RAW / source / f"{name}.parquet"))


def load_many(items: list[tuple[str, str]]) -> pd.DataFrame:
    """Concatenate several stored frames."""
    return validate(pd.concat([read(s, n) for s, n in items], ignore_index=True))
