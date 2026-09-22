"""AHL level A — speed selection on a trend/chop change point.

The construction is frozen in `5_Plans_de_recherche/ahl/PRESPEC_AHL.md` §5 and nothing
here is a choice made after seeing a result.

    latent A     ER63 = |log P_t - log P_{t-63}| / sum|d log P|, cross-sectional median
    detector A   BOCPD, Normal-inverse-gamma (mu0=0, kappa0=1, alpha0=1, beta0=1),
                 hazard 1/250, run-length truncation 600, input standardised on an
                 expanding window with a 252-session minimum
    state A      "consolidating" when the posterior-mean ER63 since the last changepoint
                 is below its expanding median (756-session minimum)
    dwell        no state change within 21 sessions of the previous one
    speeds       trending carries S = 12-minus-1; consolidating carries F = (63, 5)

A1, which decides, asks whether the state shifts the relative directional accuracy of F
over S, in the stated direction:

    delta = [hit(F | consolidating) - hit(S | consolidating)]
          - [hit(F | trending)      - hit(S | trending)]

weighted by risk-scaled exposure, over the 46-instrument panel.

**The discipline is mechanical here rather than remembered.** Running this script plain
measures the instrument and prints no statistic. Printing delta is A1, and A1 is a trial
against `data/trials.parquet`, so it requires `--read` and it should be run once. The
first version of the day-one instrument returned NaN on every reading and printed a
verdict anyway; a script that cannot accidentally read its own answer is the cheaper
guard.

Usage:
    .venv/bin/python scripts/run_ahl_level_a.py            # instrument only
    .venv/bin/python scripts/run_ahl_level_a.py --read     # A1, a trial
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats as st

from regime_lab.analysis.bootstrap import stationary_indices
from regime_lab.config import CACHE

RULE = "=" * 78

SAMPLE_START = pd.Timestamp("2003-07-17")   # settled 2026-09-23, strict reading
SLOW = (252, 21)
FAST = (63, 5)
VOL_WINDOW = ER_WINDOW = 63
HAZARD = 1.0 / 250.0
TRUNCATION = 600
MIN_DWELL = 21
MEDIAN_MIN = 756
DRAWS = 2_000
BLOCKS = (21, 63, 126)
Z_MDE = 1.959964 + 0.841621
TREE_CORRECTION = 1.346   # Holm-Bonferroni over the six primary tests, PRESPEC §11


def efficiency_ratio(prices: pd.DataFrame) -> pd.Series:
    log_prices = np.log(prices.where(prices > 0))
    net = (log_prices - log_prices.shift(ER_WINDOW)).abs()
    gross = log_prices.diff().abs().rolling(ER_WINDOW).sum()
    return (net / gross.replace(0.0, np.nan)).median(axis=1)


def build_state(prices: pd.DataFrame) -> tuple[pd.Series, pd.Series, int]:
    """State A, exactly as §5 freezes it. Returns the lagged state, ER63, and the
    number of changepoints, so the time constant can be reported alongside."""
    ratio = efficiency_ratio(prices).dropna()
    standardised = (
        (ratio - ratio.expanding(252).mean()) / ratio.expanding(252).std()
    ).dropna()

    run_length = _bocpd(standardised.to_numpy(float))
    breaks = np.r_[False, np.diff(run_length) < 0]

    # Posterior-mean ER63 since the last changepoint: the running mean of the segment.
    segment = np.empty(len(standardised))
    start = 0
    values = ratio.reindex(standardised.index).to_numpy(float)
    for i in range(len(values)):
        if breaks[i]:
            start = i
        segment[i] = np.nanmean(values[start : i + 1])
    segment = pd.Series(segment, index=standardised.index)

    median = segment.expanding(MEDIAN_MIN).median()
    raw = (segment > median).astype(float).where(median.notna())

    # Minimum dwell: once the state moves it is held for MIN_DWELL sessions.
    held = raw.copy()
    last_change, current = -MIN_DWELL, np.nan
    for i, value in enumerate(raw.to_numpy()):
        if np.isnan(value):
            continue
        if np.isnan(current) or value != current and i - last_change >= MIN_DWELL:
            current, last_change = value, i
        held.iloc[i] = current

    return held.shift(1), ratio, int(breaks.sum())


def _bocpd(x: np.ndarray) -> np.ndarray:
    """Adams & MacKay (2007) with the conjugate prior §5 names. Returns the MAP run
    length, whose drops are the changepoints."""
    mu, kappa = np.array([0.0]), np.array([1.0])
    alpha, beta = np.array([1.0]), np.array([1.0])
    R = np.zeros(TRUNCATION + 1)
    R[0] = 1.0
    out = np.zeros(len(x), dtype=int)
    for t, value in enumerate(x):
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        pred = st.t.pdf((value - mu) / scale, 2 * alpha) / scale
        m = len(pred)
        new = np.zeros(TRUNCATION + 1)
        new[1 : min(m + 1, TRUNCATION + 1)] = (R[:m] * pred * (1 - HAZARD))[: min(m, TRUNCATION)]
        new[0] = float((R[:m] * pred * HAZARD).sum())
        total = new.sum()
        R = new / total if total > 0 else new
        mu_next = np.concatenate([[0.0], (kappa * mu + value) / (kappa + 1)])
        kappa_next = np.concatenate([[1.0], kappa + 1.0])
        alpha_next = np.concatenate([[1.0], alpha + 0.5])
        beta_next = np.concatenate([[1.0], beta + kappa * (value - mu) ** 2 / (2 * (kappa + 1))])
        keep = min(len(mu_next), TRUNCATION + 1)
        mu, kappa = mu_next[:keep], kappa_next[:keep]
        alpha, beta = alpha_next[:keep], beta_next[:keep]
        out[t] = int(np.argmax(R))
    return out


def panel_pieces(prices: pd.DataFrame):
    returns = prices.pct_change()
    sigma = returns.rolling(VOL_WINDOW).std() * np.sqrt(252)
    weight = (0.10 / sigma).clip(upper=3.0).shift(1)

    def hits(lookback: int, skip: int) -> pd.DataFrame:
        signal = np.sign(prices.shift(skip) / prices.shift(lookback) - 1.0).shift(1)
        realised = np.sign(returns)
        keep = signal.notna() & realised.notna() & signal.ne(0.0)
        return (signal == realised).where(keep).astype(float)

    return hits(*FAST), hits(*SLOW), weight


def delta(hit_fast, hit_slow, weight, state, dates) -> float:
    hf, hs = hit_fast.loc[dates], hit_slow.loc[dates]
    w, s = weight.loc[dates], state.loc[dates]

    def rate(hits: pd.DataFrame, mask: pd.Series) -> float:
        wm = w.where(hits.notna()).mul(mask.astype(float), axis=0)
        wm = wm.where(wm > 0)
        total = np.nansum(wm.to_numpy())
        if total <= 0:
            return np.nan
        return float(np.nansum((hits * wm).to_numpy()) / total)

    chop, trend = s.eq(0.0), s.eq(1.0)
    return (rate(hf, chop) - rate(hs, chop)) - (rate(hf, trend) - rate(hs, trend))


def main() -> None:
    read_the_answer = "--read" in sys.argv

    prices = pd.read_parquet(CACHE / "trend_universe_m1.parquet").loc[SAMPLE_START:]
    state, ratio, breaks = build_state(prices)
    hit_fast, hit_slow, weight = panel_pieces(prices)

    usable = state.notna() & weight.notna().any(axis=1) & hit_slow.notna().any(axis=1)
    dates = prices.index[usable.reindex(prices.index, fill_value=False)]

    years = (dates.max() - dates.min()).days / 365.25
    print(RULE)
    print("AHL LEVEL A — SPEED SELECTION ON A TREND/CHOP CHANGE POINT")
    print(RULE)
    print(f"\n   sample      {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}, "
          f"{len(dates):,} usable sessions, {prices.shape[1]} instruments")
    print(f"   detector    BOCPD hazard 1/250, {breaks} changepoints, "
          f"{breaks / years:.2f} a year")
    print(f"   dwell       {MIN_DWELL} sessions minimum")
    share = float(state.loc[dates].eq(1.0).mean())
    print(f"   state       trending on {share:.1%} of sessions, consolidating on "
          f"{1 - share:.1%}")

    rng = np.random.default_rng(20260923)
    print(f"\n{RULE}\n1.  THE INSTRUMENT — MDE under the null, no statistic read\n")
    print(f"   {'mean block':>11} {'se':>10} {'MDE raw':>10} {'MDE corrected':>15}")
    readings = []
    for block in BLOCKS:
        draws = np.array([
            delta(hit_fast, hit_slow, weight, state, dates[stationary_indices(len(dates), block, rng)])
            for _ in range(DRAWS)
        ])
        se = float(np.nanstd(draws, ddof=1))
        raw, corrected = Z_MDE * se * 100.0, Z_MDE * se * 100.0 * TREE_CORRECTION
        readings.append(corrected)
        print(f"   {block:>11} {se:>10.5f} {raw:>10.2f} {corrected:>15.2f}")

    offsets = rng.integers(252, len(dates) - 252, size=400)
    values = state.loc[dates].to_numpy()
    null = np.array([
        delta(hit_fast, hit_slow, weight, pd.Series(np.roll(values, o), index=dates), dates)
        for o in offsets
    ])
    sd_null = float(np.nanstd(null, ddof=1))
    readings.append(Z_MDE * sd_null * 100.0 * TREE_CORRECTION)
    print(f"\n   rotation null, 400 draws: sd {sd_null:.5f}, MDE corrected "
          f"{Z_MDE * sd_null * 100 * TREE_CORRECTION:>.2f}, centred at "
          f"{np.nanmean(null) * 100:+.3f} points")

    if any(not np.isfinite(v) for v in readings):
        print("\n   A READING IS NOT FINITE. No verdict: a broken instrument is not an")
        print("   undecidable question, and letting it read as one is the failure this")
        print(f"   programme keeps finding in its own work.\n{RULE}")
        return

    threshold = max(readings)
    print(f"\n   decision threshold, tree-corrected: {threshold:.2f} points")

    if not read_the_answer:
        print(f"\n{RULE}\n2.  A1 NOT READ\n")
        print("   Printing the statistic is A1, and A1 is a trial against")
        print("   data/trials.parquet. Re-run with --read, once, to spend it.")
        print(f"\n{RULE}")
        return

    print(f"\n{RULE}\n2.  A1 — THE TRIAL\n")
    observed = delta(hit_fast, hit_slow, weight, state, dates) * 100.0
    print(f"   delta observed          {observed:+.3f} points")
    print(f"   decision threshold      {threshold:.3f} points")
    print("   stated direction        positive (F better in chop, S better in trend)")
    if not np.isfinite(observed):
        verdict = "NOT FINITE — no verdict"
    elif observed <= 0:
        verdict = "WRONG SIGN => A1 refuted, proceed to level B"
    elif observed < threshold:
        verdict = "BELOW THE THRESHOLD => UNDERPOWERED, never a pass; A1 refuted, go to B"
    else:
        verdict = "ABOVE THE THRESHOLD => A1 survives its own falsification"
    print(f"\n   => {verdict}")
    print(f"\n{RULE}")


if __name__ == "__main__":
    main()
