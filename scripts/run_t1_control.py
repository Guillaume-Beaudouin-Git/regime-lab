"""T1 — the volatility-quantile placebo at the level of the strategy book.

The charter (§ 05, control 1) reads:

    Placebo par quantile de volatilite. Le modele bat-il "volatilite realisee
    a 20 jours au-dessus de son seuil median" ? Valide seulement si le vecteur
    de variables contient le bloc non-vol du § 03.

and § 08, stop rule 1:

    si aucune famille ne bat le placebo de volatilite (T1) avec un ecart
    superieur a l'effet minimum detectable sur au moins trois plis sur cinq :
    on ne va pas en phase 5, on redige le resultat nul et sa decomposition.

The repository implements that comparison at the level of conditional moments
(``predictive.incremental_information``), which is control T4, not T1. T1 is the
same comparison one layer up: the frozen state -> position rule of
``models/mapping.py`` applied to the fitted state, against the identical rule
applied to a state that is nothing but a volatility quantile.

Specification, fixed here before any number was read
----------------------------------------------------
* Book: ``strategies.book.base_book()`` — 60/40, IN EXCESS of the three-month
  bill, the same object every other script evaluates.
* Real leg: ``mapping.state_to_position(states[family])`` — hold the book in the
  calm state, flat otherwise, signal at t-1 traded at t.
* Placebo leg: exactly the same rule applied to
  ``1{ realised W-session volatility of the book < its own expanding median }``.
  W = 20 sessions (the charter's literal wording) for the primary reading;
  W = 21 (the repository's existing default in ``predictive``) as a declared
  sensitivity. The median is expanding with a 252-session burn-in, never
  full-sample, and it is computed on the WHOLE book history from 1990, which is
  causal and avoids truncating the evaluation window (see section 5.4).
* Primary statistic: difference in annualised Sharpe, overlay minus placebo.
* Inference: HAC lag-6 t-statistic on the daily difference series (project rule)
  and a paired stationary block bootstrap (Politis-Romano) through
  ``analysis.bootstrap.paired_sharpe_difference``, mean block 63 sessions.
  Never iid.
* Power: minimum detectable effect from ``analysis.power``, same block, alpha
  0.05, power 0.80. An observed gap inside the MDE is UNDERPOWERED, never a pass.
* Folds: the charter's five, taken from ``protocol.walk_forward`` exactly as
  ``run_phase2`` builds them, so the fold edges are the study's own.
* Costs: both legs trade, so both are charged, at the three declared levels.
* Multiplicity: five families, so a Sidak-corrected MDE is reported alongside.

This file writes nothing anywhere. It reads the caches that already exist.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from regime_lab.analysis.bootstrap import paired_sharpe_difference
from regime_lab.analysis.power import minimum_detectable_sharpe_difference
from regime_lab.config import CACHE
from regime_lab.models.mapping import apply_overlay, state_to_position
from regime_lab.models.protocol import walk_forward
from regime_lab.strategies.book import base_book
from regime_lab.strategies.costs import COST_BPS, charge

warnings.filterwarnings("ignore")

RULE = "=" * 86
GAMMA = 5.0
MEAN_BLOCK = 63
DRAWS_FULL = 2_000
DRAWS_FOLD = 1_500
HAC_LAG = 6
Z_MDE = float(stats.norm.ppf(0.975) + stats.norm.ppf(0.80))


# ----------------------------------------------------------------- primitives


def annual_sharpe(x: pd.Series | np.ndarray) -> float:
    v = np.asarray(x, dtype=float)
    s = v.std(ddof=1)
    return float(v.mean() / s * np.sqrt(252)) if s > 0 else np.nan


def ceq(x: pd.Series) -> float:
    v = np.asarray(x, dtype=float)
    return float(v.mean()) * 252 - 0.5 * GAMMA * float(v.var(ddof=1)) * 252


def max_drawdown(x: pd.Series) -> float:
    curve = (1 + pd.Series(x)).cumprod()
    return float((curve / curve.cummax() - 1).min())


def hac_t(difference: pd.Series, lag: int = HAC_LAG) -> tuple[float, float]:
    """Annualised mean of the daily difference and its HAC lag-6 t-statistic."""
    y = np.asarray(difference.dropna(), dtype=float)
    model = sm.OLS(y, np.ones((len(y), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    return float(model.params[0] * 252), float(model.tvalues[0])


def vol_quantile_state(
    returns: pd.Series, *, window: int, q: float = 0.5, min_periods: int = 252
) -> pd.Series:
    """One when realised volatility sits below its own expanding q-quantile.

    Same orientation as a fitted state: the higher label is the calm one, so
    ``state_to_position`` applies to it unchanged. Causal by construction.
    """
    realised = returns.rolling(window).std() * np.sqrt(252)
    threshold = realised.expanding(min_periods=min_periods).quantile(q)
    return (realised < threshold).astype(float).where(threshold.notna()).rename("placebo")


def compare(a: pd.Series, b: pd.Series, *, draws: int, mean_block: int = MEAN_BLOCK) -> dict:
    """One paired comparison of two overlay return series on a common index."""
    common = a.dropna().index.intersection(b.dropna().index)
    x, y = a.loc[common], b.loc[common]
    distribution = paired_sharpe_difference(
        x.to_numpy(), y.to_numpy(), mean_block=mean_block, draws=draws, seed=0
    )
    se = float(distribution.std(ddof=1))
    observed = annual_sharpe(x) - annual_sharpe(y)
    centred = distribution - distribution.mean()
    mean_diff, t_stat = hac_t(x - y)
    return {
        "n": int(len(common)),
        "start": common.min(),
        "end": common.max(),
        "sharpe_a": annual_sharpe(x),
        "sharpe_b": annual_sharpe(y),
        "delta": observed,
        "se": se,
        "mde": Z_MDE * se,
        "ci_low": float(np.percentile(distribution, 2.5)),
        "ci_high": float(np.percentile(distribution, 97.5)),
        "p_boot": float(np.mean(np.abs(centred) >= abs(observed))),
        "mean_diff": mean_diff,
        "t_hac": t_stat,
        "ceq_a": ceq(x),
        "ceq_b": ceq(y),
        "detected": bool(abs(observed) > Z_MDE * se),
    }


def legs(base: pd.Series, state: pd.Series, index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    position = state_to_position(state).reindex(index).fillna(0.0)
    return apply_overlay(base, position).dropna(), position


# ----------------------------------------------------------------------- main


def main() -> None:
    states = pd.read_parquet(CACHE / "states.parquet")
    offline = pd.read_parquet(CACHE / "states_offline.parquet")
    features = pd.read_parquet(CACHE / "features.parquet").dropna()

    book = base_book()  # excess of cash
    oos = states.dropna(how="all").index
    base = book.reindex(oos).dropna()
    oos = base.index

    placebo_full = vol_quantile_state(book, window=20)  # full history, causal
    placebo_21 = vol_quantile_state(book, window=21)
    placebo_oos_only = vol_quantile_state(base, window=20)  # restricted, for 5.4

    print(RULE)
    print("T1  volatility-quantile placebo AT STRATEGY LEVEL")
    print(RULE)
    print(f"\n   book          60/40 in excess of cash, {len(book):,} sessions "
          f"{book.index.min():%Y-%m} to {book.index.max():%Y-%m}")
    print(f"   evaluation    {len(oos):,} out-of-sample sessions "
          f"{oos.min():%Y-%m-%d} to {oos.max():%Y-%m-%d}")
    print("   placebo       1 = 20-session realised vol below its expanding median "
          "(252-session burn-in)")
    print(f"   placebo NaN inside the evaluation window: "
          f"{int(placebo_full.reindex(oos).isna().sum())} of {len(oos)}")
    print("   rule          mapping.state_to_position, identical on both legs, "
          "signal t-1 traded t")
    print(f"   bootstrap     stationary block, mean block {MEAN_BLOCK}, "
          f"{DRAWS_FULL} draws, paired index path")
    print(f"   HAC lag       {HAC_LAG}   |   MDE z = {Z_MDE:.4f} (alpha 0.05, power 0.80)")

    # ---------------------------------------------------------------- 1. legs
    print("\n" + RULE)
    print("\n1. THE LEGS\n")
    print(f"   {'leg':<22} {'Sharpe':>7} {'vol':>7} {'maxDD':>8} {'CEQ':>8} "
          f"{'exposed':>8} {'turnover':>9}")
    print("   " + "-" * 74)

    def describe(name: str, series: pd.Series, position: pd.Series | None) -> None:
        exposure = "" if position is None else f"{position.reindex(series.index).mean():>7.1%}"
        turn = "" if position is None else f"{position.diff().abs().mean():>9.4f}"
        print(f"   {name:<22} {annual_sharpe(series):>7.2f} "
              f"{series.std(ddof=1) * np.sqrt(252):>6.1%} {max_drawdown(series):>8.1%} "
              f"{ceq(series):>7.2%} {exposure:>8} {turn:>9}")

    describe("base 60/40 (hold)", base, None)
    placebo_leg, placebo_pos = legs(base, placebo_full, oos)
    describe("PLACEBO vol quantile", placebo_leg, placebo_pos)
    family_legs: dict[str, tuple[pd.Series, pd.Series]] = {}
    for family in states.columns:
        leg, pos = legs(base, states[family], oos)
        family_legs[family] = (leg, pos)
        describe(family, leg, pos)

    # ------------------------------------------------------- 2. full sample T1
    print("\n" + RULE)
    print("\n2. T1 ON THE WHOLE OUT-OF-SAMPLE BLOCK  (overlay minus placebo)\n")
    print(f"   {'family':<18} {'SR ovl':>7} {'SR plc':>7} {'dSR':>7} {'MDE':>6} "
          f"{'95% CI':>17} {'p boot':>7} {'dMean':>8} {'t HAC6':>7}  verdict")
    print("   " + "-" * 100)

    full: dict[str, dict] = {}
    for family in states.columns:
        res = compare(family_legs[family][0], placebo_leg, draws=DRAWS_FULL)
        full[family] = res
        verdict = "DETECTED" if res["detected"] else "inside the noise"
        print(f"   {family:<18} {res['sharpe_a']:>7.2f} {res['sharpe_b']:>7.2f} "
              f"{res['delta']:>+7.2f} {res['mde']:>6.2f} "
              f"[{res['ci_low']:>+6.2f},{res['ci_high']:>+6.2f}] {res['p_boot']:>7.3f} "
              f"{res['mean_diff']:>+7.2%} {res['t_hac']:>7.2f}  {verdict}")

    print("\n   A gap smaller than the MDE is underpowered, not a null. It is never a pass.")

    # ------------------------------------------------------------- 3. by fold
    folds = walk_forward(features.index, min_train_years=10, folds=5)
    print("\n" + RULE)
    print("\n3. THE FIVE FOLDS  (stop rule 1 needs a positive gap above the MDE on 3 of 5)\n")
    for f in folds:
        print(f"   {f}")
    print()
    print(f"   {'family':<18} " + "".join(f"{'fold ' + str(f.index):>16}" for f in folds)
          + f"{'passes':>9}")
    print("   " + "-" * 100)

    fold_pass: dict[str, int] = {}
    fold_detail: dict[str, list[dict]] = {}
    for family in states.columns:
        cells, rows, passes = [], [], 0
        for f in folds:
            window = oos[(oos >= f.eval_start) & (oos < f.eval_end)]
            res = compare(
                family_legs[family][0].reindex(window).dropna(),
                placebo_leg.reindex(window).dropna(),
                draws=DRAWS_FOLD,
            )
            rows.append(res)
            ok = res["delta"] > 0 and abs(res["delta"]) > res["mde"]
            passes += int(ok)
            cells.append(f"{res['delta']:>+7.2f}/{res['mde']:>5.2f}{'*' if ok else ' '}")
        fold_pass[family] = passes
        fold_detail[family] = rows
        print(f"   {family:<18} " + "".join(f"{c:>16}" for c in cells) + f"{passes:>7}/5")

    print("\n   cells read  dSR / MDE ; * marks a positive gap larger than the fold MDE")
    print(f"\n   STOP RULE 1: max passes over families = {max(fold_pass.values())} of 5 "
          f"(threshold 3 of 5)")
    if max(fold_pass.values()) < 3:
        print("   -> no family beats the volatility placebo on three folds of five.")

    # ------------------------------------------------------------- 4. costs
    print("\n" + RULE)
    print("\n4. COSTS  both legs trade, so both are charged\n")
    print(f"   {'family':<18} {'turn ovl':>9} {'turn plc':>9} {'gross dSR':>10} "
          + "".join(f"{k + ' ' + str(int(v)) + 'bp':>18}" for k, v in COST_BPS.items()))
    print("   " + "-" * 100)
    for family in states.columns:
        leg, pos = family_legs[family]
        row = (f"   {family:<18} {pos.diff().abs().mean():>9.4f} "
               f"{placebo_pos.diff().abs().mean():>9.4f} {full[family]['delta']:>+10.2f}")
        for bps in COST_BPS.values():
            net_a = charge(leg, pos, bps=bps)
            net_b = charge(placebo_leg, placebo_pos, bps=bps)
            common = net_a.index.intersection(net_b.index)
            row += f" {annual_sharpe(net_a.loc[common]) - annual_sharpe(net_b.loc[common]):>+17.2f}"
        print(row)

    # ------------------------------------------------------ 5. sensitivities
    print("\n" + RULE)
    print("\n5. SENSITIVITIES  declared before the primary was read\n")

    print("   5.1  placebo window 21 sessions instead of the charter's 20\n")
    leg21, pos21 = legs(base, placebo_21, oos)
    print(f"        placebo Sharpe {annual_sharpe(leg21):.2f}  "
          f"exposure {pos21.mean():.1%}  turnover {pos21.diff().abs().mean():.4f}")
    for family in states.columns:
        res = compare(family_legs[family][0], leg21, draws=DRAWS_FOLD)
        print(f"        {family:<18} dSR {res['delta']:>+6.2f}  MDE {res['mde']:.2f}  "
              f"t HAC6 {res['t_hac']:>+5.2f}  "
              f"{'DETECTED' if res['detected'] else 'inside the noise'}")

    print("\n   5.2  states_offline (detection latency) instead of states\n")
    for family in offline.columns:
        leg_off, _ = legs(base, offline[family], oos)
        res = compare(leg_off, placebo_leg, draws=DRAWS_FOLD)
        print(f"        {family:<18} SR {res['sharpe_a']:>5.2f}  dSR {res['delta']:>+6.2f}  "
              f"MDE {res['mde']:.2f}  t HAC6 {res['t_hac']:>+5.2f}  "
              f"{'DETECTED' if res['detected'] else 'inside the noise'}")

    print("\n   5.3  placebo matched on exposure instead of on the median")
    print("        (the quantile is read from the family's own realised exposure, which uses")
    print("         hindsight; it is deliberately generous to the placebo, and it overlaps T3)\n")
    for family in states.columns:
        q = float(family_legs[family][1].mean())
        matched = vol_quantile_state(book, window=20, q=q)
        leg_m, pos_m = legs(base, matched, oos)
        res = compare(family_legs[family][0], leg_m, draws=DRAWS_FOLD)
        print(f"        {family:<18} q {q:.3f}  placebo exposure {pos_m.mean():.1%}  "
              f"SR plc {res['sharpe_b']:>5.2f}  dSR {res['delta']:>+6.2f}  "
              f"MDE {res['mde']:.2f}  {'DETECTED' if res['detected'] else 'inside the noise'}")

    print("\n   5.4  the truncation trap: placebo built on the evaluation window only\n")
    leg_trunc, pos_trunc = legs(base, placebo_oos_only, oos)
    common_trunc = leg_trunc.dropna().index
    print(f"        expanding median restricted to the OOS block starts the placebo at "
          f"{placebo_oos_only.dropna().index.min():%Y-%m-%d}")
    print(f"        common sample {len(common_trunc):,} sessions against "
          f"{len(oos):,} — {len(oos) - len(common_trunc)} sessions lost")
    for family in states.columns:
        res = compare(family_legs[family][0], leg_trunc, draws=DRAWS_FOLD)
        print(f"        {family:<18} dSR {res['delta']:>+6.2f} on {res['n']:,} sessions "
              f"(full-history placebo gave {full[family]['delta']:>+6.2f} on "
              f"{full[family]['n']:,})")

    print("\n   5.5  block length of the bootstrap\n")
    for block in (21, 63, 126, 252):
        row = f"        mean block {block:>3}   "
        for family in states.columns:
            res = compare(family_legs[family][0], placebo_leg, draws=DRAWS_FOLD, mean_block=block)
            row += f"{family.split()[0]:>4} MDE {res['mde']:.2f}  "
        print(row)

    # ---------------------------------------------------- 6. multiple testing
    print("\n" + RULE)
    print("\n6. MULTIPLICITY  five families tested against one placebo\n")
    k = len(states.columns)
    alpha_sidak = 1 - (1 - 0.05) ** (1 / k)
    z_sidak = float(stats.norm.ppf(1 - alpha_sidak / 2) + stats.norm.ppf(0.80))
    print(f"   Sidak alpha for {k} tests: {alpha_sidak:.4f}   z = {z_sidak:.3f} "
          f"against {Z_MDE:.3f} uncorrected\n")
    for family in states.columns:
        res = full[family]
        mde_c = z_sidak * res["se"]
        print(f"   {family:<18} dSR {res['delta']:>+6.2f}  MDE {res['mde']:.2f} -> "
              f"corrected {mde_c:.2f}  "
              f"{'DETECTED' if abs(res['delta']) > mde_c else 'inside the noise'}")

    # ------------------------------------------------------- 7. verification
    print("\n" + RULE)
    print("\n7. VERIFICATION  my MDE against analysis.power on the same inputs\n")
    for family in list(states.columns)[:2]:
        leg = family_legs[family][0]
        common = leg.index.intersection(placebo_leg.index)
        ref = minimum_detectable_sharpe_difference(
            leg.loc[common].to_numpy(), placebo_leg.loc[common].to_numpy(),
            mean_block=MEAN_BLOCK, draws=DRAWS_FULL,
        )
        print(f"   {family:<18} power.py mde {ref.mde:.6f}  observed {ref.observed:+.6f}  |  "
              f"here {full[family]['mde']:.6f}  {full[family]['delta']:+.6f}")

    print("\n   years of data needed to bring each gap to detectability")
    years = len(oos) / 252
    for family in states.columns:
        res = full[family]
        needed = years * (res["mde"] / abs(res["delta"])) ** 2 if res["delta"] != 0 else np.inf
        print(f"   {family:<18} |dSR| {abs(res['delta']):.3f}  MDE {res['mde']:.3f}  "
              f"shortfall x{(res['mde'] / abs(res['delta'])) ** 2:>6.1f}  "
              f"{needed:>7.0f} years needed against {years:.0f} available")

    print("\n" + RULE)


if __name__ == "__main__":
    sys.exit(main())
