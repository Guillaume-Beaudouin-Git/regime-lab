"""Immutable parquet store with a fetch manifest.

Every write records a manifest entry: the content hash, the row count, the
period range and the retrieval timestamp. A reviewer can therefore tell whether
two runs saw the same bytes, which is the minimum bar for a reproducible study.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from regime_lab.config import MANIFESTS, RAW
from regime_lab.data.pit import validate


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(frame: pd.DataFrame, source: str, name: str, *, origin: str = "") -> Path:
    """Persist a point-in-time frame and record its manifest entry."""
    frame = validate(frame)
    target = RAW / source / f"{name}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)

    entry = {
        "source": source,
        "name": name,
        "path": str(target.relative_to(RAW.parent)),
        "origin": origin,
        "rows": int(len(frame)),
        "series": sorted(frame["series_id"].dropna().unique().tolist()),
        "period_min": frame["period"].min().date().isoformat(),
        "period_max": frame["period"].max().date().isoformat(),
        "available_at_max": frame["available_at"].max().isoformat(),
        "sha256": _digest(target),
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    manifest = MANIFESTS / f"{source}__{name}.json"
    manifest.write_text(json.dumps(entry, indent=2) + "\n")
    return target


def read(source: str, name: str) -> pd.DataFrame:
    """Load a stored point-in-time frame."""
    return validate(pd.read_parquet(RAW / source / f"{name}.parquet"))


def load_many(items: list[tuple[str, str]]) -> pd.DataFrame:
    """Concatenate several stored frames into one point-in-time frame."""
    frames = [read(source, name) for source, name in items]
    return validate(pd.concat(frames, ignore_index=True))
