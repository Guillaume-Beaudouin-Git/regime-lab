"""Concurrent writers must not lose rows of the trials log."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from regime_lab.analysis import trials


def _write(args: tuple[str, int]) -> None:
    path, k = args
    trials.log("lock_test", {"k": k}, {"sharpe": float(k)}, path=Path(path))


def test_parallel_logs_keep_every_row(tmp_path: Path) -> None:
    path = tmp_path / "trials.parquet"
    with ProcessPoolExecutor(max_workers=6) as pool:
        list(pool.map(_write, [(str(path), k) for k in range(30)]))
    frame = pd.read_parquet(path)
    assert len(frame) == 30
    assert frame["config_hash"].nunique() == 30
