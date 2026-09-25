"""Export the out-of-sample states of the five models as a versioned CSV.

data/ is not versioned, so the states behind the handout's classification table and
its figures would otherwise only exist after a 15-20 minute run of
scripts/run_phase2.py. This writes them next to the other committed readings.

Columns: date, then one column per family, 1 = calm, 0 = stress (states ordered by
training volatility, filtered online, as in data/cache/states.parquet). Days run from
2002-04-01, the first out-of-sample day; US holidays carry the previous state.

Usage: .venv/bin/python scripts/export_states.py
Output: docs/artifacts/etats_hors_echantillon.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from regime_lab.config import CACHE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "artifacts" / "etats_hors_echantillon.csv"
COLUMNS = {"A' sparse jump": "A_prime_sparse_jump", "A  jump": "A_jump",
           "B  filtered HMM": "B_hmm_filtre", "C  gradient boost": "C_gradient_boosting",
           "C' HAR-RV": "C_prime_har_rv"}


def main() -> None:
    states = pd.read_parquet(CACHE / "states.parquet").loc["2002-04-01":, list(COLUMNS)]
    states = states.dropna(how="all").rename(columns=COLUMNS)
    if states.isna().any().any():
        raise SystemExit("missing states inside the out-of-sample window")
    out = states.astype(int)
    out.index.name = "date"
    out.to_csv(OUT, date_format="%Y-%m-%d")
    span = f"{out.index[0]:%Y-%m-%d} to {out.index[-1]:%Y-%m-%d}"
    print(f"{OUT.relative_to(ROOT)}: {len(out)} days, {span}")


if __name__ == "__main__":
    main()
