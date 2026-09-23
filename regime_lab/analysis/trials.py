"""Append-only log of every configuration that was evaluated.

The deflated Sharpe ratio needs the number of trials *and* the variance of their
Sharpe ratios. Counting trials by hand is not definable — is a debugging run a
trial? — so nothing is counted by hand: every evaluation writes a row here, with
a hash of its configuration and a timestamp, and the deflation reads the file.

The log is also the honest record of how much searching happened, which is the
part of a backtest that is usually invisible.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from regime_lab.config import DATA

TRIALS = DATA / "trials.parquet"


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def log(family: str, config: dict, metrics: dict, *, path: Path = TRIALS) -> None:
    """Append one evaluated configuration."""
    row = {
        "logged_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "family": family,
        "config_hash": config_hash(config),
        "config": json.dumps(config, sort_keys=True, default=str),
        **{f"m_{k}": v for k, v in metrics.items()},
    }
    frame = pd.DataFrame([row])
    if path.exists():
        frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
    _text_where_mixed(frame).to_parquet(path, index=False)


def _text_where_mixed(frame: pd.DataFrame) -> pd.DataFrame:
    """Store as text any column that holds both text and non-text values.

    One family may log a metric as a number and another as text under the same name
    (``note``: a flag ``1.0`` from one family, a sentence from another). Parquet needs
    one type per column, and the append would otherwise fail and lose the row. Missing
    values stay missing; every other value keeps its content, written as text.
    """
    out = frame.copy()
    for column in out.columns:
        values = out[column]
        if values.dtype != object:
            continue
        present = values.dropna()
        kinds = {isinstance(v, str) for v in present}
        if kinds == {True, False}:
            out[column] = values.map(lambda v: v if pd.isna(v) else str(v))
    return out


def read(path: Path = TRIALS) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def summary(path: Path = TRIALS) -> dict[str, float]:
    """Inputs the deflated Sharpe ratio needs, measured rather than assumed."""
    frame = read(path)
    if frame.empty or "m_sharpe" not in frame:
        return {"n_trials": 0, "sharpe_var": float("nan")}
    sharpes = frame["m_sharpe"].dropna()
    return {
        "n_trials": int(len(frame)),
        "n_distinct": int(frame["config_hash"].nunique()),
        "sharpe_var": float(sharpes.var(ddof=1)) if len(sharpes) > 1 else float("nan"),
        "sharpe_max": float(sharpes.max()) if len(sharpes) else float("nan"),
    }
