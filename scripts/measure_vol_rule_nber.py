"""The one-line volatility rules, scored against NBER on the models' own days.

Layer 1 of the study (docs/RESULTS_FINAL.md) scores the five classifiers against the
NBER recession label but leaves the programme's one-line control out of that table.
This puts it in, on exactly the same 6,377 out-of-sample days (April 2002 to
September 2026, US holidays included, as in data/cache/states.parquet), with the same
scoring function, so that "does the model beat a one-line rule at recognising
recessions?" has a number.

Two rules, both causal, both on the 21-session realised volatility of the S&P 500:
stress when it is above its expanding median (`volatility_quantile_placebo`), or above
its expanding 80th percentile (`crisis.volatility_tail_rule`). The rule is computed on
trading sessions and carried over US holidays, like the models' states.

The five families are rescored in the same pass: their kappas must match the published
ones, otherwise the script stops. Descriptive only: no returns, no trial logged.

Usage: .venv/bin/python scripts/measure_vol_rule_nber.py
Output: docs/artifacts/temoin_nber.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_b1_regime_coupling as b1  # noqa: E402

from regime_lab.config import CACHE  # noqa: E402
from regime_lab.data import store  # noqa: E402
from regime_lab.evaluation.predictive import volatility_quantile_placebo  # noqa: E402
from regime_lab.evaluation.reliability import external_validation  # noqa: E402
from regime_lab.extensions.crisis import volatility_tail_rule  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "artifacts" / "temoin_nber.txt"
OOS_START = "2002-04-01"
PUBLISHED_KAPPA = {"A  jump": 0.49, "A' sparse jump": 0.53, "B  filtered HMM": 0.24,
                   "C  gradient boost": 0.12, "C' HAR-RV": 0.17}


def score(name: str, labels: pd.Series, nber: pd.Series) -> dict:
    s = labels.dropna()
    ref = nber.reindex(s.index)
    rec = (ref == 1.0).to_numpy()
    stress = (s == s.min()).to_numpy()
    out = external_validation(s, ref)
    years = (s.index[-1] - s.index[0]).days / 365.25
    switches = int((s.diff().abs() > 0).sum())
    return {
        "name": name, "n": len(s), "ba": 100 * out["balanced_accuracy"], "kappa": out["kappa"],
        "recall": 100 * stress[rec].mean(), "spec": 100 * (~stress[~rec]).mean(),
        "stress": 100 * stress.mean(), "per_year": switches / years,
        "precision": 100 * rec[stress].mean(), "run": len(s) / (switches + 1),
    }


def main() -> None:
    states = pd.read_parquet(CACHE / "states.parquet").loc[OOS_START:]
    index = states["A' sparse jump"].dropna().index
    raw = store.read("references", "ref_nber").set_index("period")["value"]
    nber = pd.Series(raw.reindex(index.to_period("M").to_timestamp()).to_numpy(), index=index)

    spx = b1.us_prices("eq_us_large")
    spx.index = pd.to_datetime(spx.index)
    returns = spx.pct_change()
    rules = {
        "rule: vol above expanding median": volatility_quantile_placebo(returns),
        "rule: vol above expanding 80th pct": volatility_tail_rule(returns),
    }

    rows = []
    for column in PUBLISHED_KAPPA:
        row = score(column, states[column].reindex(index), nber)
        if abs(row["kappa"] - PUBLISHED_KAPPA[column]) > 0.006:
            want = PUBLISHED_KAPPA[column]
            raise SystemExit(f"{column}: kappa {row['kappa']:.3f}, published {want}")
        rows.append(row)
    for name, rule in rules.items():
        carried = rule.reindex(rule.index.union(index)).ffill().reindex(index)
        rows.append(score(name, carried, nber))

    for r in rows:
        if not np.isfinite(r["ba"]):
            raise SystemExit(f"{r['name']}: non-finite score, no reading")

    lines = [
        "One-line volatility rules against NBER, on the models' out-of-sample days",
        f"{index[0]:%Y-%m-%d} to {index[-1]:%Y-%m-%d}, {len(index)} days, "
        f"{int((nber == 1.0).sum())} recession days; S&P 500 21-session realised volatility.",
        "",
        f"{'label':<38}{'bal.acc':>8}{'kappa':>7}{'recall':>8}{'spec':>7}{'stress':>8}{'switch/yr':>10}"
        f"{'rec|stress':>11}{'mean run':>9}",
    ]
    for r in rows:
        lines.append(f"{r['name']:<38}{r['ba']:>7.1f}%{r['kappa']:>7.2f}{r['recall']:>7.1f}%"
                     f"{r['spec']:>6.1f}%{r['stress']:>7.1f}%{r['per_year']:>10.2f}"
                     f"{r['precision']:>10.1f}%{r['run']:>9.0f}")
    lines += ["", "Switches per calendar year; rec|stress = share of stress days that fall in a",
              "recession; mean run in days. The five model rows reproduce the published kappas."]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
