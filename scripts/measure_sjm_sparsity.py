"""How many features does the sparse jump model (A') actually keep?

``max_features=10`` is the ``max_feats`` of the reference implementation, the square
of the L1 bound on the lasso weights ``w`` (with ``||w||_2 = 1``). It bounds the
*effective* number of features, ``(sum w)^2 / sum w^2``, at 10; it does not bound
the number of non-zero weights. Presentations had said both "≈ 10 effective" and
"at most 10": this measures which is true.

One fit per calibration window of the published run (training ends every fourth
refit), at the penalty that run used there, read from the 2026-09-09 calibration rows
of ``data/trials.parquet`` (grid-median fallback 20 when no candidate was valid).
Descriptive only: no returns are scored, no trial is logged.

Usage: .venv/bin/python scripts/measure_sjm_sparsity.py
Output: docs/artifacts/sjm_sparsity.txt
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from regime_lab.config import CACHE
from regime_lab.models.calibrate import LAMBDA_GRID
from regime_lab.models.jump import JumpRegimes
from regime_lab.strategies.book import base_book

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "artifacts" / "sjm_sparsity.txt"
RUN_DAY = "2026-09-09"  # UTC day of the calibration rows behind data/cache/states.parquet


def penalties_used() -> list[tuple[str, float]]:
    """(training end, penalty) for each calibration of A' in the published run."""
    trials = pd.read_parquet(ROOT / "data" / "trials.parquet")
    rows = trials[(trials["family"] == "sparse_jump") & trials["logged_at"].str.startswith(RUN_DAY)]
    out = []
    for end, group in rows.groupby(rows["config"].map(lambda c: json.loads(c)["train_end"])):
        valid = group[np.isfinite(group["m_sharpe"])]
        if valid.empty:
            out.append((end, float(np.median(LAMBDA_GRID))))
        else:
            best = valid.loc[valid["m_sharpe"].idxmax(), "config"]
            out.append((end, float(json.loads(best)["lambda"])))
    return out


def main() -> None:
    features = pd.read_parquet(CACHE / "features.parquet")
    book = base_book()
    lines = [
        "Sparse jump model (A'), max_features = 10: weights of the lasso step, one fit per",
        f"calibration window of the published run (calibration rows logged {RUN_DAY} UTC).",
        "",
        f"{'train end':<12}{'lambda':>8}{'non-zero':>10}{'effective':>11}{'top-10 share':>14}"
        "  largest weights",
    ]
    counts, shares = [], []
    for end, penalty in penalties_used():
        train = features.loc[:end].dropna()
        model = JumpRegimes(jump_penalty=penalty, max_features=10.0)
        model.fit(train, book.reindex(train.index))
        w = pd.Series(np.asarray(model.model_.w, dtype=float), index=model.features_)
        if not np.isfinite(w).all():
            raise SystemExit(f"{end}: non-finite weights, no reading")
        ranked = w.sort_values(ascending=False)
        nonzero = int((w > 1e-8).sum())
        effective = float(w.sum() ** 2 / (w**2).sum())
        share = float((ranked.iloc[:10] ** 2).sum() / (w**2).sum())
        counts.append(nonzero)
        shares.append(share)
        top = ", ".join(f"{k} {v:.2f}" for k, v in ranked.head(5).items())
        lines.append(f"{end:<12}{penalty:>8g}{nonzero:>10}{effective:>11.1f}{share:>14.0%}  {top}")
        print(lines[-1], flush=True)
    lines += [
        "",
        f"non-zero weights: {min(counts)} to {max(counts)} of {features.shape[1]}; "
        f"the ten largest carry {min(shares):.0%} to {max(shares):.0%} of sum w^2.",
        "Reading: the effective number of features is 10 by construction; the number of",
        "features with a non-zero weight is not bounded by 10.",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
