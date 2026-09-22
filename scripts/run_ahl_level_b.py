"""AHL level B — a correlation break as a CONSTRUCTION input. B1 decides.

The construction is frozen in `docs/PRESPEC_AHL.md` §5 and §10, locked at `92e4e9e`:

    latent B     ||C63 - C252||_F / sqrt(k(k-1)) on the 28-instrument block complete
                 from the sample start, residualised on VOL63 by causal expanding OLS
                 (756 minimum)
    detector B   two-sided Page CUSUM, k = 1.0, h = 5.0, reset at alarm
    dwell        no state change within 21 sessions of the previous one

B1 asks: **does a break predict a rise in risk-model error** — the gap between the
book's predicted and realised volatility? A variance claim, the one channel where this
programme holds positive evidence. KB1: HAC t below the tree-corrected z of 2.638, or
the wrong sign, and B1 is dead and the tree goes to level C.

THE OPERATIONALISATION. §10 names the question and not the statistic, so everything
below was fixed in this file, and committed, before any number of B1 existed. Where a
choice was open, the one inherited from committed code was taken.

    VOL63        cross-sectional median of the 46 instruments' 63-session annualised
                 realised volatility. Not defined in any committed file; this is the
                 definition that reproduces the disclosed ER63 correlation of §1 Q1 to
                 the third decimal (+0.096 / -0.043 against +0.096 / -0.044).
    CUSUM input  the residual, standardised on an expanding window with a 252-session
                 minimum — detector A's standardisation, §5.
    returns      simple returns, as everywhere else in the book. CORRECTED AFTER THE
                 MDE: the pre-lock script, found later in a temporary scratchpad, used
                 LOG returns, and that alone reproduces the disclosed 7.26 alarms a
                 year exactly. The earlier claim in this docstring that 7.26 came from
                 a non-causal standardisation was wrong: that script standardises
                 causally, as this one does. On log returns the threshold is 52.2% of
                 the mean error instead of 49.6%, so the verdict does not depend on it.
    dwell        the CUSUM resets at every crossing; a crossing within 21 sessions of
                 the last ACCEPTED alarm resets it but is not accepted
    the book     the M3 headline book of `extensions/vehicle.py`, 46 instruments, the
                 committed construction, net of headline costs and in excess of cash
    its model    the incumbent risk model, the one §5 names: the 63-session rolling sd
                 of book returns, lagged one session. The multiplier is set so that this
                 model predicts exactly the 10% target; the cap binds on 0.00% of
                 sessions, so the book's predicted volatility IS 10%
    the error    e_t = |log(RV_{t+1..t+21} / 0.10)|, RV = sqrt(252 * mean r^2) over the
                 next 21 sessions. How far the book misses its own target. This is the
                 quantity B2's secondary asks the reset covariance to tighten
    the break    B_t = 1 when an accepted alarm fired on one of sessions t-20..t. Known
                 at the close of t; the error it is asked to predict starts at t+1
    B1           beta in e_t = alpha + beta * B_t + u_t. Stated direction: beta > 0
    the t        overlapping 21-session windows make HAC lag 6 too short, so the
                 decision t is the smaller in magnitude of HAC lag 6 (the hard rule) and
                 HAC lag 20 (the overlap). Never the larger

THE MDE, by the procedure §9 fixes and level A applied: stationary block bootstrap over
the date dimension, blocks 21 / 63 / 126, plus a null that rotates the break series
circularly against the error series, preserving its run structure exactly. MDE =
(1.960 + 0.842) x sd x 1.346. The decision threshold is the widest of the four.

THE CRITERION, WRITTEN BEFORE THE MDE IS PRODUCED:

    If the tree-corrected threshold exceeds a quarter of the unconditional mean error,
    B1 is undecidable on this sample: it is written as such and the tree goes to C,
    half a day and zero trials. A break that raises the book's average volatility
    miss by 25% is the upper end of what a mundane break — seven to nine a year, and
    orthogonal to volatility by construction — could plausibly do. An instrument that
    cannot see that cannot see anything plausible.

    Declared expectation, from arithmetic and not from data: the error's sd near 0.2,
    about a hundred independent 42-session stretches, B on about half the sample, give
    a standard error near 0.04 and a corrected MDE near 0.15 — 60% to 75% of a mean
    error of 0.20-0.25. **The expected outcome is UNDECIDABLE.**

    Seen AFTER the criterion and the expectation were written, BEFORE any MDE: the
    instrument's own descriptors. 4,760 usable sessions, 2008-05-13 to 2026-08-12.
    The CUSUM crosses 166 times, 9.1 a year, but the crossings cluster, and the dwell
    leaves **48 accepted alarms, 2.63 a year** — not the 7.26 §1 disclosed. B is on
    21.2% of sessions, not half. Mean error 0.255, sd 0.196. Nothing above is changed
    in their light; they are written here so the change of footing is visible.
    2.63 a year still clears the ~2-a-year clock of the rule on a seventh device, but
    by a quarter and not by the factor of 3.6 §2 of the prespec implied for B.

THE DECISION, if the question is decidable:

    beta not finite                  no verdict
    beta <= 0                        WRONG SIGN, B1 refuted, go to C
    decision t < 2.638               KB1, B1 refuted, go to C
    beta < threshold                 UNDERPOWERED, never a pass, B1 refuted, go to C
    otherwise                        B1 survives KB1, and must then survive KY

KY AT B1, the matched placebo, declared here because the known killer is volatility. The
same detector — same k, h, reset, dwell and window — run on standardised VOL63 instead
of the residualised distance gives V_t. B1 must survive V_t as a covariate: beta in
e = alpha + beta B + gamma V + u stays positive with a decision t of at least 2.638.
If it does not, the break is a volatility break in disguise and B1 is refused.

**The discipline is mechanical, as at level A.** Plain, this script measures the
instrument and prints no statistic. `--read` prints B1, which is a trial logged to
`data/trials.parquet`, and it should be run once.

Usage:
    .venv/bin/python scripts/run_ahl_level_b.py            # instrument only
    .venv/bin/python scripts/run_ahl_level_b.py --read     # B1, a trial
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from regime_lab.analysis import trials
from regime_lab.analysis.bootstrap import stationary_indices
from regime_lab.config import CACHE
from regime_lab.extensions.vehicle import vol_targeted_book

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_m3_evaluation import active_window, cash_rate_daily, net_excess  # noqa: E402

RULE = "=" * 78

SAMPLE_START = pd.Timestamp("2003-07-17")   # settled 2026-09-23, strict reading
BLOCK_SIZE = 28
SHORT, LONG = 63, 252
RESID_MIN = 756
STANDARDISE_MIN = 252
CUSUM_K, CUSUM_H = 1.0, 5.0
MIN_DWELL = 21
WINDOW = 21
HORIZON = 21
TARGET = 0.10
DRAWS = 2_000
ROTATIONS = 400
BLOCKS = (21, 63, 126)
Z_MDE = 1.959964 + 0.841621
TREE_CORRECTION = 1.346   # Holm-Bonferroni over the six primary tests, PRESPEC §11
Z_STRICT = 2.638
DECIDABLE_SHARE = 0.25


def correlation_distance(returns: pd.DataFrame) -> pd.Series:
    """||C63 - C252||_F / sqrt(k(k-1)) on a block with no missing value."""
    x = returns.to_numpy(float)
    k = x.shape[1]
    out = np.full(len(x), np.nan)
    for i in range(LONG, len(x)):
        long_window = x[i - LONG + 1 : i + 1]
        if np.isnan(long_window).any():
            continue
        c_short = np.corrcoef(long_window[-SHORT:].T)
        c_long = np.corrcoef(long_window.T)
        out[i] = np.linalg.norm(c_short - c_long) / np.sqrt(k * (k - 1))
    return pd.Series(out, index=returns.index, name="distance")


def expanding_residual(y: pd.Series, x: pd.Series, minimum: int) -> pd.Series:
    """y minus its causal expanding OLS fit on x: coefficients use data through t only."""
    both = pd.concat([y, x], axis=1).dropna()
    yv, xv = both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()
    n = np.arange(1, len(yv) + 1)
    sx, sy = np.cumsum(xv), np.cumsum(yv)
    sxx, sxy = np.cumsum(xv * xv), np.cumsum(xv * yv)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (sxy - sx * sy / n) / (sxx - sx * sx / n)
    intercept = (sy - slope * sx) / n
    resid = yv - (intercept + slope * xv)
    resid[n < minimum] = np.nan
    return pd.Series(resid, index=both.index).reindex(y.index)


def standardise(x: pd.Series) -> pd.Series:
    live = x.dropna()
    z = (live - live.expanding(STANDARDISE_MIN).mean()) / live.expanding(STANDARDISE_MIN).std()
    return z.reindex(x.index)


def page_cusum(z: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Two-sided Page CUSUM with reset at every crossing, and the 21-session dwell.

    Returns the accepted alarms, their direction (+1 upward, -1 downward), and every
    crossing before the dwell, so the time constant can be reported both ways."""
    up = down = 0.0
    last = -MIN_DWELL
    accepted = np.zeros(len(z), dtype=bool)
    crossed = np.zeros(len(z), dtype=bool)
    direction = np.zeros(len(z))
    for i, value in enumerate(z.to_numpy()):
        if np.isnan(value):
            continue
        up = max(0.0, up + value - CUSUM_K)
        down = max(0.0, down - value - CUSUM_K)
        if up > CUSUM_H or down > CUSUM_H:
            crossed[i] = True
            if i - last >= MIN_DWELL:
                accepted[i], last = True, i
                direction[i] = 1.0 if up > CUSUM_H else -1.0
            up = down = 0.0
    live = z.notna()
    return (
        pd.Series(accepted, index=z.index).where(live),
        pd.Series(direction, index=z.index).where(live),
        pd.Series(crossed, index=z.index).where(live),
    )


def post_break(alarms: pd.Series) -> pd.Series:
    """1 on the WINDOW sessions starting at an accepted alarm, known at their close."""
    return alarms.fillna(0.0).rolling(WINDOW, min_periods=1).max().where(alarms.notna())


def forward_rv(returns: pd.Series) -> pd.Series:
    """Annualised realised volatility over sessions t+1 .. t+HORIZON, stamped at t."""
    mean_square = (returns**2).rolling(HORIZON).mean().shift(-HORIZON)
    return np.sqrt(252.0 * mean_square)


def beta(e: np.ndarray, b: np.ndarray) -> float:
    on, off = b == 1.0, b == 0.0
    if on.sum() < 2 or off.sum() < 2:
        return np.nan
    return float(e[on].mean() - e[off].mean())


def hac_fit(e: pd.Series, regressors: pd.DataFrame, lags: int):
    design = sm.add_constant(regressors.astype(float))
    return sm.OLS(e.astype(float), design).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def decision_t(e: pd.Series, regressors: pd.DataFrame, name: str) -> tuple[float, float, float]:
    """(t at lag 6, t at lag 20, the smaller in magnitude)."""
    t6 = float(hac_fit(e, regressors, 6).tvalues[name])
    t20 = float(hac_fit(e, regressors, HORIZON - 1).tvalues[name])
    return t6, t20, (t6 if abs(t6) < abs(t20) else t20)


def build() -> dict:
    prices = pd.read_parquet(CACHE / "trend_universe_m1.parquet").loc[SAMPLE_START:]
    returns = prices.pct_change()
    block = prices.columns[prices.notna().all()]
    if len(block) != BLOCK_SIZE:
        raise SystemExit(f"the complete block holds {len(block)} instruments, §3 says 28")

    vol63 = (returns.rolling(SHORT).std() * np.sqrt(252)).median(axis=1).rename("vol63")
    distance = correlation_distance(returns[block])
    residual = expanding_residual(distance, vol63, RESID_MIN)
    z_break = standardise(residual)
    alarms, direction, crossings = page_cusum(z_break)

    z_vol = standardise(vol63.where(distance.notna()))
    vol_alarms, _, _ = page_cusum(z_vol)

    built = vol_targeted_book(prices)
    live = active_window(built["returns"]).index
    net = net_excess(
        built["returns"].reindex(live), built["weights"].reindex(live),
        cash_rate_daily(live), column="headline",
    )
    rv = forward_rv(net)

    gross = (built["weights"] * returns).sum(axis=1)
    diagonal = (built["weights"] ** 2 * returns**2).sum(axis=1)
    cross = gross**2 - diagonal
    cross_share = (
        cross.rolling(HORIZON).mean().shift(-HORIZON)
        / (gross**2).rolling(HORIZON).mean().shift(-HORIZON)
    )

    frame = pd.DataFrame({
        "e": np.abs(np.log(rv / TARGET)),
        "signed": np.log(rv / TARGET),
        "b": post_break(alarms),
        "v": post_break(vol_alarms),
        "direction": direction.replace(0.0, np.nan).ffill(limit=WINDOW - 1),
        "cross_share": cross_share,
        "vol63": vol63,
        "residual": residual,
        "distance": distance,
    })
    usable = frame[["e", "b", "v"]].notna().all(axis=1) & np.isfinite(frame["e"])
    return {
        "frame": frame.loc[usable],
        "alarms": alarms,
        "vol_alarms": vol_alarms,
        "direction": direction,
        "crossings": crossings,
        "cap_binds": built["cap_binds"],
        "block": block,
        "multiplier": built["multiplier"],
    }


def main() -> None:
    read_the_answer = "--read" in sys.argv
    pieces = build()
    frame = pieces["frame"]
    dates = frame.index
    alarms = pieces["alarms"].loc[dates.min() : dates.max()]
    years = (dates.max() - dates.min()).days / 365.25

    print(RULE)
    print("AHL LEVEL B — A CORRELATION BREAK AS A CONSTRUCTION INPUT")
    print(RULE)
    print(f"\n   sample      {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}, "
          f"{len(dates):,} usable sessions")
    print(f"   block       {len(pieces['block'])} instruments complete from "
          f"{SAMPLE_START:%Y-%m-%d}; the book holds all 46")
    j = frame[["residual", "vol63"]].dropna()
    print(f"   latent      residual vs VOL63: Pearson {j['residual'].corr(j['vol63']):+.3f}, "
          f"Spearman {j['residual'].corr(j['vol63'], method='spearman'):+.3f}  "
          f"(disclosed pre-lock +0.009 / +0.075)")
    n_alarms = int(alarms.sum())
    ups = int((pieces["direction"].loc[dates.min() : dates.max()] > 0).sum())
    print(f"   detector    CUSUM k {CUSUM_K}, h {CUSUM_H}, dwell {MIN_DWELL}: {n_alarms} accepted "
          f"alarms, {n_alarms / years:.2f} a year ({ups} upward, {n_alarms - ups} downward)")
    n_cross = int(pieces["crossings"].loc[dates.min() : dates.max()].sum())
    print(f"               before the dwell: {n_cross} crossings, {n_cross / years:.2f} a year "
          f"(pre-lock disclosure: 7.26, on log returns)")
    n_vol = int(pieces["vol_alarms"].loc[dates.min() : dates.max()].sum())
    print(f"   placebo V   same detector on VOL63: {n_vol} alarms, {n_vol / years:.2f} a year")
    share_b, share_v = float(frame["b"].mean()), float(frame["v"].mean())
    print(f"   B on        {share_b:.1%} of sessions; V on {share_v:.1%}; "
          f"overlap corr {frame['b'].corr(frame['v']):+.3f}")
    cap = float(pieces["cap_binds"].reindex(dates).mean())
    print(f"   cap         binds on {cap:.2%} of sessions — the model predicts the target")
    mean_error = float(frame["e"].mean())
    print(f"   error       unconditional mean |log(RV21 / 10%)| = {mean_error:.4f}, "
          f"sd {frame['e'].std():.4f}")

    e, b = frame["e"].to_numpy(), frame["b"].to_numpy()
    rng = np.random.default_rng(20260922)
    print(f"\n{RULE}\n1.  THE INSTRUMENT — MDE under the null, no statistic read\n")
    print(f"   {'mean block':>11} {'se':>10} {'MDE raw':>10} {'MDE corrected':>15}")
    readings = []
    for block in BLOCKS:
        draws = np.empty(DRAWS)
        for d in range(DRAWS):
            idx = stationary_indices(len(e), block, rng)
            draws[d] = beta(e[idx], b[idx])
        se = float(np.nanstd(draws, ddof=1))
        raw, corrected = Z_MDE * se, Z_MDE * se * TREE_CORRECTION
        readings.append(corrected)
        print(f"   {block:>11} {se:>10.5f} {raw:>10.4f} {corrected:>15.4f}")

    offsets = rng.integers(252, len(e) - 252, size=ROTATIONS)
    null = np.array([beta(e, np.roll(b, o)) for o in offsets])
    sd_null = float(np.nanstd(null, ddof=1))
    readings.append(Z_MDE * sd_null * TREE_CORRECTION)
    print(f"\n   rotation null, {ROTATIONS} draws: sd {sd_null:.5f}, MDE corrected "
          f"{readings[-1]:.4f}, centred at {np.nanmean(null):+.4f}")

    if any(not np.isfinite(v) for v in [*readings, mean_error]):
        print("\n   A READING IS NOT FINITE. No verdict: a broken instrument is not an")
        print("   undecidable question, and letting it read as one is the failure this")
        print(f"   programme keeps finding in its own work.\n{RULE}")
        return

    threshold = max(readings)
    ceiling = DECIDABLE_SHARE * mean_error
    print(f"\n   decision threshold, tree-corrected   {threshold:.4f}")
    print(f"   criterion, 25% of the mean error      {ceiling:.4f}")
    print(f"   threshold as a share of the mean error  {threshold / mean_error:.1%}")
    decidable = threshold <= ceiling
    if not decidable:
        print("\n   => UNDECIDABLE. The instrument cannot resolve a break that raises the")
        print("      book's average volatility miss by a quarter. B1 is not read; it is")
        print("      written as undecidable and the tree goes to level C. Zero trials.")
        print(f"\n{RULE}")
        return
    print("\n   => DECIDABLE.")

    if not read_the_answer:
        print(f"\n{RULE}\n2.  B1 NOT READ\n")
        print("   Printing the statistic is B1, and B1 is a trial against")
        print("   data/trials.parquet. Re-run with --read, once, to spend it.")
        print(f"\n{RULE}")
        return

    read(frame, threshold, null, mean_error, n_alarms / years)


def read(frame: pd.DataFrame, threshold: float, null: np.ndarray, mean_error: float,
         alarms_per_year: float) -> None:
    e, b = frame["e"].to_numpy(), frame["b"].to_numpy()
    observed = beta(e, b)
    t6, t20, t_dec = decision_t(frame["e"], frame[["b"]], "b")
    percentile = float((null < observed).mean())

    print(f"\n{RULE}\n2.  B1 — THE TRIAL\n")
    print(f"   beta observed            {observed:+.4f}  ({observed / mean_error:+.1%} of the "
          f"mean error)")
    print(f"   HAC t, lag 6 / lag 20    {t6:+.2f} / {t20:+.2f}   decision t {t_dec:+.2f}")
    print(f"   decision threshold       {threshold:.4f}   strict z {Z_STRICT}")
    print(f"   rotation null            percentile {percentile:.1%}")
    print("   stated direction         positive (a break raises the book's volatility miss)")

    if not np.isfinite(observed) or not np.isfinite(t_dec):
        verdict = "NOT FINITE — no verdict"
    elif observed <= 0:
        verdict = "WRONG SIGN => B1 refuted, proceed to level C"
    elif t_dec < Z_STRICT:
        verdict = "KB1 — the t is below the corrected z => B1 refuted, proceed to level C"
    elif observed < threshold:
        verdict = "BELOW THE THRESHOLD => UNDERPOWERED, never a pass; B1 refuted, go to C"
    else:
        verdict = "B1 SURVIVES KB1 — now KY"
    print(f"\n   => {verdict}")

    controlled = hac_fit(frame["e"], frame[["b", "v"]], 6)
    c6, c20, c_dec = decision_t(frame["e"], frame[["b", "v"]], "b")
    v_alone = beta(e, frame["v"].to_numpy())
    print(f"\n{RULE}\n3.  KY — THE MATCHED VOLATILITY PLACEBO\n")
    print(f"   V alone                  beta {v_alone:+.4f}")
    print(f"   B with V as covariate    beta {controlled.params['b']:+.4f}, decision t "
          f"{c_dec:+.2f} (lag 6 {c6:+.2f}, lag 20 {c20:+.2f})")
    print(f"   V with B as covariate    gamma {controlled.params['v']:+.4f}")
    ky_holds = controlled.params["b"] > 0 and c_dec >= Z_STRICT
    if verdict.startswith("B1 SURVIVES"):
        verdict = ("B1 SURVIVES KB1 AND KY" if ky_holds
                   else "REFUSED BY KY — a volatility break in disguise; go to C")
        print(f"\n   => {verdict}")
    else:
        print("\n   reported, not decisive: B1 already fell at KB1")

    print(f"\n{RULE}\n4.  DECOMPOSITION — the mechanism, measured rather than inferred\n")
    on, off = frame["b"] == 1.0, frame["b"] == 0.0
    print(f"   {'':34}{'after a break':>15}{'otherwise':>12}")
    print(f"   {'mean |log(RV / 10%)|':34}{frame.loc[on, 'e'].mean():>15.4f}"
          f"{frame.loc[off, 'e'].mean():>12.4f}")
    print(f"   {'mean log(RV / 10%), signed':34}{frame.loc[on, 'signed'].mean():>+15.4f}"
          f"{frame.loc[off, 'signed'].mean():>+12.4f}")
    print(f"   {'cross-term share of variance':34}{frame.loc[on, 'cross_share'].mean():>15.3f}"
          f"{frame.loc[off, 'cross_share'].mean():>12.3f}")
    print(f"   {'VOL63 at t':34}{frame.loc[on, 'vol63'].mean():>15.4f}"
          f"{frame.loc[off, 'vol63'].mean():>12.4f}")
    print(f"   {'sessions':34}{int(on.sum()):>15,}{int(off.sum()):>12,}")
    for label, sign in (("upward alarms", 1.0), ("downward alarms", -1.0)):
        sel = (frame["direction"] == sign).to_numpy()
        keep = sel | off.to_numpy()
        print(f"   {label:<34}beta {beta(e[keep], b[keep]):+.4f} over {int(sel.sum()):,} sessions")
    mid = frame.index[len(frame) // 2]
    for label, part in (("first half", frame.loc[:mid]), ("second half", frame.loc[mid:])):
        print(f"   {label:<34}beta {beta(part['e'].to_numpy(), part['b'].to_numpy()):+.4f}  "
              f"{part.index.min():%Y-%m} to {part.index.max():%Y-%m}")

    trials.log(
        "ahl_level_b",
        {
            "test": "B1",
            "prespec": "docs/PRESPEC_AHL.md, LOCKED 92e4e9e",
            "panel": "trend_universe_m1",
            "sample_start": str(SAMPLE_START.date()),
            "latent": "||C63-C252||_F/sqrt(k(k-1)), 28 block, resid on VOL63 (median 46), "
                      "expanding OLS 756",
            "detector": "two-sided Page CUSUM k1 h5 reset, input expanding-standardised 252",
            "dwell": MIN_DWELL,
            "window": WINDOW,
            "horizon": HORIZON,
            "error": "|log(RV21 fwd / 0.10)|, M3 headline net excess",
        },
        {
            "beta": observed,
            "t_lag6": t6,
            "t_lag20": t20,
            "t_decision": t_dec,
            "threshold": threshold,
            "mean_error": mean_error,
            "rotation_percentile": percentile,
            "beta_controlled": float(controlled.params["b"]),
            "t_controlled": c_dec,
            "alarms_per_year": alarms_per_year,
            "sessions": int(len(frame)),
            "verdict": verdict,
        },
    )
    print(f"\n   logged to data/trials.parquet ({trials.summary()['n_distinct']} distinct)")
    print(f"\n{RULE}")


if __name__ == "__main__":
    main()
