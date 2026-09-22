"""Two Sigma plan — the signal library, measured in position space. §1 P1, P2, P5.

`PRESPEC_TWOSIGMA.md` is a DRAFT: §1 discloses figures produced by scripts that were
never committed. This script reproduces the library's share of them from committed
code, `regime_lab/selection/library.py`, and sets each against the disclosed value.
A mismatch is printed as a mismatch. Nothing here is tuned to meet a disclosed number.

    P1   46-instrument universe, eleven candidates: mean |pairwise position corr|,
         effective rank, first eigenvalue, components for 90%, three quoted pairs.
         The declared result that the approach is NOT transposable there.
    P2   49-industry panel, twelve candidates: mean |corr|, signed mean, effective
         rank, components for 90%, IND_ACCEL against IND_MOM_12_1. Admission-critical.
    P5   turnover: the equal-weight blend and IND_REV_1W, each signal's |Δw| per year.

THE BLINDNESS RULE. The pre-registration is not locked, so this script computes no
quantity from which the tree's answer could be read. It never multiplies a weight by a
return: `factors_5` is read only for `ff_mkt-rf`, the regressor of IND_LOWBETA. What
it prints is correlations between weight vectors, their eigenvalues, turnover |Δw|,
and turnover priced at the §5 schedule — a cost, not a performance.

THE GUARD. If any reading is not finite, the script prints no reproduction verdict:
a broken instrument is not a mismatch, and letting it read as one is the failure this
programme keeps finding in its own work.

Usage:
    .venv/bin/python scripts/measure_twosigma_library.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime_lab.config import CACHE, RAW
from regime_lab.selection.library import (
    EXCLUDED,
    INDUSTRY_CANDIDATES,
    LIBRARY,
    TREND_MIN_LEGS,
    Geometry,
    annual_turnover,
    equal_weight_blend,
    geometry,
    industry_library,
    load_panel,
    positions,
    trend_universe_scores,
)
from regime_lab.selection.protocol import COST_BPS

RULE = "=" * 78
#: Axis 4 of the rule on a seventh device: a conditioned object below four effective
#: dimensions shares the six refutations' narrowness.
AXIS_4_FLOOR = 4.0

#: The pre-commit script printed its strongest pairs with round(c, 2); the draft wrote
#: two of those two-decimal values with a third, zero, decimal.
ROUNDED_PAIR = "matches at two decimals: a round(c, 2) printout written with three"

INDUSTRY_PANEL = RAW / "panels" / "industry_49.parquet"
FACTORS_PANEL = RAW / "panels" / "factors_5.parquet"
TREND_UNIVERSE = CACHE / "trend_universe_m1.parquet"


@dataclass(frozen=True)
class Check:
    """One disclosed figure against its reproduction, compared at the draft's precision."""

    label: str
    disclosed: float
    decimals: int
    measured: float
    admission_critical: bool
    note: str = ""

    @property
    def match(self) -> bool:
        return abs(self.measured - self.disclosed) <= 0.5 * 10.0**-self.decimals + 1e-12

    def row(self) -> str:
        flag = "MATCH" if self.match else "MISMATCH"
        star = "*" if self.admission_critical else " "
        shown = self.decimals + 1 if self.decimals else 0
        return (f" {star} {self.label:<52} {self.disclosed:>9.{self.decimals}f} "
                f"{self.measured:>11.{shown}f}   {flag}")


def matrix(g: Geometry) -> str:
    short = {n: n.replace("IND_", "") for n in g.correlation.index}
    return g.correlation.rename(index=short, columns=short).round(2).to_string()


def describe(title: str, g: Geometry) -> None:
    print(f"\n   {title}")
    print(f"      signals {len(g.correlation)} | mean corr {g.mean_corr:+.3f} | mean |corr| "
          f"{g.mean_abs_corr:.3f} | max |corr| {g.max_abs_corr:.3f}")
    print(f"      effective rank {g.effective_rank:.2f} of {len(g.correlation)} | "
          f"{g.components_90} components for 90% of the trace")
    print(f"      eigenvalues {np.round(g.eigenvalues, 2).tolist()}")


def industry_section(weights: dict[str, pd.DataFrame]) -> dict[str, Geometry]:
    print(f"\n{RULE}\nP2.  THE 49-INDUSTRY PANEL — twelve candidates, position space\n")
    twelve = geometry(weights)
    s = twelve.sessions
    print(f"   sample   {s.min():%Y-%m-%d} to {s.max():%Y-%m-%d}, {len(s):,} sessions, "
          f"{twelve.years:.2f} years, every signal on all 49 legs")
    print("\n   mean daily cross-sectional correlation of the weight vectors:\n")
    print("   " + matrix(twelve).replace("\n", "\n   "))
    describe("twelve candidates (the disclosed P2 set)", twelve)
    print("      strongest pairs: " + "; ".join(
        f"{a}/{b} {c:+.3f}" for a, b, c in twelve.strongest_pairs(4)))

    eleven_names = [n for n in INDUSTRY_CANDIDATES if n != "IND_ACCEL"]
    eleven = geometry(weights, eleven_names)
    ten = geometry(weights, LIBRARY)
    print("\n   exclusions, §4, made before contact:")
    for name, reason in EXCLUDED.items():
        print(f"      {name:<11} {reason}")
    describe("eleven: IND_ACCEL dropped (PLAN.md's 'about 8')", eleven)
    describe("ten: the inferential library of §4 — NOT the set §1 P2 measured", ten)
    clears = "clears" if ten.effective_rank >= AXIS_4_FLOOR else "does NOT clear"
    print(f"      axis 4: {ten.effective_rank:.2f} effective dimensions {clears} the floor "
          f"of {AXIS_4_FLOOR:.0f}")
    return {"twelve": twelve, "eleven": eleven, "ten": ten}


def trend_section(prices: pd.DataFrame) -> Geometry:
    print(f"\n{RULE}\nP1.  THE 46-INSTRUMENT UNIVERSE — eleven candidates, position space\n")
    coverage = prices.notna().mean()
    thin = sorted(coverage[coverage < 0.9].index)
    print(f"   universe {prices.shape[1]} instruments, {prices.index.min():%Y-%m-%d} to "
          f"{prices.index.max():%Y-%m-%d}; {len(thin)} under 90% coverage")
    weights = {k: positions(v) for k, v in trend_universe_scores(prices).items()}
    g = geometry(weights, min_legs=TREND_MIN_LEGS)
    s = g.sessions
    print(f"   sample   {s.min():%Y-%m-%d} to {s.max():%Y-%m-%d}, {len(s):,} sessions with "
          f">= {TREND_MIN_LEGS} instruments carrying all eleven, {g.years:.2f} years")
    print("\n   mean daily cross-sectional correlation of the weight vectors:\n")
    print("   " + matrix(g).replace("\n", "\n   "))
    describe("eleven candidates", g)
    print("      strongest pairs: " + "; ".join(
        f"{a}/{b} {c:+.3f}" for a, b, c in g.strongest_pairs(5)))
    return g


def turnover_section(weights: dict[str, pd.DataFrame], sessions: pd.DatetimeIndex) -> dict:
    print(f"\n{RULE}\nP5.  TURNOVER — |Δw| per year on the common sample, priced at §5\n")
    held = {k: v.loc[sessions] for k, v in weights.items()}
    header = "   ".join(f"{b:>4.0f} bp" for b in COST_BPS.values())
    print(f"   {'signal':<30} {'x/yr':>7}   cost %/yr at {header}")

    def line(label: str, turnover: float) -> None:
        costs = "   ".join(f"{turnover * b / 100:>7.2f}" for b in COST_BPS.values())
        print(f"   {label:<30} {turnover:>7.1f}                {costs}")

    per_signal = {k: annual_turnover(w) for k, w in held.items()}
    for name, t in sorted(per_signal.items(), key=lambda kv: -kv[1]):
        tag = "  (excluded)" if name in EXCLUDED else ""
        line(name + tag, t)

    eleven_names = [n for n in INDUSTRY_CANDIDATES if n != "IND_ACCEL"]
    blend_11 = annual_turnover(equal_weight_blend(held, eleven_names))
    blend_10 = annual_turnover(equal_weight_blend(held, LIBRARY))
    print()
    line("blend, 11 incl. IND_REV_1W", blend_11)
    line("blend, the 10-signal control", blend_10)
    print("\n   The disclosed 21.6x/yr was measured on the eleven-signal blend, which still")
    print("   holds IND_REV_1W. §4 declares the control to be the blend of the ten.")
    return {"per_signal": per_signal, "blend_11": blend_11, "blend_10": blend_10}


def main() -> None:
    missing = [p for p in (INDUSTRY_PANEL, FACTORS_PANEL, TREND_UNIVERSE) if not p.exists()]
    if missing:
        raise SystemExit("missing input: " + ", ".join(str(p.name) for p in missing))

    print(RULE)
    print("TWO SIGMA PLAN — THE SIGNAL LIBRARY, POSITION SPACE ONLY")
    print(RULE)
    print("\n   No weight is multiplied by a return anywhere in this script.")

    industries = load_panel("industry_49")
    market = load_panel("factors_5")["ff_mkt-rf"]
    weights = industry_library(industries, market, INDUSTRY_CANDIDATES)
    ind = industry_section(weights)
    trend = trend_section(pd.read_parquet(TREND_UNIVERSE).sort_index())
    cost = turnover_section(weights, ind["twelve"].sessions)

    rev_1w = cost["per_signal"]["IND_REV_1W"]
    ten, twelve = ind["ten"], ind["twelve"]
    checks = [
        Check("P1  mean |pairwise position corr|", 0.336, 3, trend.mean_abs_corr, False),
        Check("P1  effective rank", 3.86, 2, trend.effective_rank, True),
        Check("P1  first eigenvalue (of 11)", 4.82, 2, float(trend.eigenvalues[0]), False),
        Check("P1  components for 90% of the trace", 6, 0, trend.components_90, False),
        Check("P1  TSMOM_1M / XSREV_1M", -0.990, 3, trend.pair("TSMOM_1M", "XSREV_1M"), False),
        Check("P1  TSMOM_12_1 / XSMOM_12_1", 0.930, 3,
              trend.pair("TSMOM_12_1", "XSMOM_12_1"), False, ROUNDED_PAIR),
        Check("P1  TSMOM_3M / BRKOUT_100", 0.840, 3, trend.pair("TSMOM_3M", "BRKOUT_100"), False,
              ROUNDED_PAIR),
        Check("P2  mean |pairwise position corr|, 12", 0.101, 3, twelve.mean_abs_corr, False),
        Check("P2  signed mean pairwise corr, 12", 0.013, 3, twelve.mean_corr, False),
        Check("P2  effective rank, 12 candidates", 8.37, 2, twelve.effective_rank, True),
        Check("§2  effective rank, the 10-signal library", 8.37, 2, ten.effective_rank, True,
              "§2 attaches the 12-signal figure to the 10-signal library"),
        Check("P2  components for 90% of the trace, 12", 8, 0, twelve.components_90, False),
        Check("P2  IND_ACCEL / IND_MOM_12_1", -0.912, 3,
              twelve.pair("IND_ACCEL", "IND_MOM_12_1"), False),
        Check("§3  common sample, sessions", 7946, 0, len(twelve.sessions), False),
        Check("§3  common sample, years", 31.57, 2, twelve.years, False),
        Check("P5  blend turnover, 11 incl. IND_REV_1W (x/yr)", 21.6, 1, cost["blend_11"], False),
        Check("P5  blend turnover, the 10-signal control (x/yr)", 21.6, 1, cost["blend_10"],
              False, "the draft calls 21.6 the control's; the control holds ten"),
        Check("P5  blend cost at 5 bp, 11 (%/yr)", 1.08, 2, cost["blend_11"] * 5 / 100, False),
        Check("P5  IND_REV_1W turnover (x/yr)", 156.7, 1, rev_1w, False),
        Check("P5  IND_REV_1W cost at 5 bp (%/yr)", 7.83, 2, rev_1w * 5 / 100, False),
    ]

    print(f"\n{RULE}\nREPRODUCTION — disclosed against measured, at the draft's precision\n")
    if not all(np.isfinite(c.measured) for c in checks):
        broken = [c.label for c in checks if not np.isfinite(c.measured)]
        print("   A READING IS NOT FINITE: " + ", ".join(broken))
        print("   No reproduction verdict. A broken instrument is not a mismatch.")
        print(f"\n{RULE}")
        sys.exit(1)

    print(f"   {'':<54} {'disclosed':>9} {'measured':>11}")
    for c in checks:
        print(c.row())
        if c.note:
            print(f"        {c.note}")
    failed = [c for c in checks if not c.match]
    critical = [c for c in failed if c.admission_critical]
    print(f"\n   {len(checks) - len(failed)} of {len(checks)} match; "
          f"{len(critical)} admission-critical mismatch(es).   * = admission-critical")
    clears = ten.effective_rank >= AXIS_4_FLOOR
    print(f"   Axis 4 on the 10-signal library: {ten.effective_rank:.2f} effective dimensions, "
          f"{'above' if clears else 'BELOW'} the floor of {AXIS_4_FLOOR:.0f}.")
    print(f"\n{RULE}")


if __name__ == "__main__":
    main()
