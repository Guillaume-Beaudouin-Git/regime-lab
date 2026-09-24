"""Phase B of `docs/PRESPEC_LONGHIST.md`: stopping UMD in the long SJM's stress, 1937-2026.

EVERYTHING BELOW IS FIXED BY THE PRESPEC, COMMITTED (440c796) BEFORE ANY STATE EXISTED
AND BEFORE ANY RETURN CONDITIONED ON A STATE WAS READ.

The object (§6.1)
    Ken French's daily momentum factor UMD, long-short, an excess return as it stands,
    held at weight min(0.10 / sigma_63, 3), sigma_63 the realised volatility of UMD over
    the 63 sessions ending the session before (`extensions.crisis.vol_target_weight`).
    Zero cost; the annual turnover of the weight is reported.

The labels (§4), each formed at the close of d-1 and traded on d
    SJM long        the online state of `scripts/longhist_fit.py`
    asym k=10 / 5   its stress, released after 10 (5) sessions of RV21 < RV63
    median rule     21-session market volatility above its expanding median
    80th rule       ... above its expanding 80th percentile
    DM panic        market below its level 24 calendar months earlier AND 126-session
                    variance above its expanding median (Daniel and Moskowitz)
    the market is the CRSP value-weighted total return (Ken French), simple returns.

The family (§6.3): 3 tests at alpha 0.05/3
    B1  Sharpe(stop SJM long)  - Sharpe(alone)        controls: median, 80th, DM
    B2  Sharpe(stop asym k=10) - Sharpe(alone)        controls: median, 80th, DM
    B3  Sharpe(stop SJM long)  - Sharpe(stop 80th)    no control (it is the control)
    plus one sensitivity row, never deciding: B2 with k = 5.

The measures (§6.4)
    MDE     `selection.protocol.blinded_mde` on the two legs, demeaned; blocks 21, 63,
            126; 2,000 draws; seed 0; read at alpha 0.05/3 and power 0.80; the largest
    t       HAC lag 6 of the daily difference (`paired_hac_t`); critical 2.394
    placebo the lagged label rotated circularly, 400 rotations of at least 252
            sessions, seed 20260924; percentile of delta among the rotated deltas
    folds   5 consecutive blocks of equal session count; delta per fold
The verdict is `extensions.longhist.verdict` (§6.5). A non-finite value: no verdict.

Discipline, as in `scripts/run_crisis_coupling.py`: run plain, the script prints the
instrument (sample, coverage, stress shares, MDE) and no return conditioned on a
label. `--read` prints the results and logs 4 rows (family `longhist_umd`) to
`data/trials.parquet`; it refuses to run if that family is already in the log.

Usage:
    .venv/bin/python scripts/longhist_umd.py          # instrument only
    .venv/bin/python scripts/longhist_umd.py --read   # the reading, once
"""

from __future__ import annotations

import hashlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from regime_lab.analysis import trials
from regime_lab.config import CACHE, RAW
from regime_lab.data import store
from regime_lab.evaluation.predictive import volatility_quantile_placebo
from regime_lab.extensions import crisis
from regime_lab.extensions import longhist as lh
from regime_lab.selection.protocol import blinded_mde, mde_at, paired_hac_t

warnings.filterwarnings("ignore")
RULE = "=" * 78
FAMILY = "longhist_umd"
OOS_START = pd.Timestamp("1937-01-01")
END = pd.Timestamp("2026-07-31")
A_PRIME_START = pd.Timestamp("2002-04-01")
N_TESTS = 3
ALPHA = 0.05 / N_TESTS
T_CRITICAL = float(stats.norm.ppf(1.0 - ALPHA / 2.0))
BLOCKS = (21, 63, 126)
DRAWS = 2000
ROTATIONS = 400
MIN_SHIFT = 252
SEED_PLACEBO = 20260924
FOLDS = 5
SUBPERIODS = {"1937-1962": ("1937-01-01", "1962-12-31"),
              "1963-2001": ("1963-01-01", "2001-12-31"),
              "2002-2026": ("2002-01-01", "2026-07-31")}
LABELS = ("SJM long", "asym k=10", "asym k=5", "median rule", "80th rule", "DM panic")
CONTROLS = ("median rule", "80th rule", "DM panic")
TESTS = {
    # name: (coupled label, reference: "alone" or a label, controls, role)
    "B1": ("SJM long", "alone", CONTROLS, "primary"),
    "B2": ("asym k=10", "alone", CONTROLS, "primary"),
    "B3": ("SJM long", "80th rule", (), "primary"),
    "S_k5": ("asym k=5", "alone", CONTROLS, "sensitivity"),
}
INPUTS = (
    RAW / "crisis" / "french_umd.parquet",
    RAW / "longhist" / "ff3_daily.parquet",
    CACHE / "longhist_states.parquet",
    CACHE / "states.parquet",
)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def build() -> dict:
    umd_frame = store.read("crisis", "french_umd")
    unit = pd.Series(umd_frame["value"].to_numpy(float),
                     index=pd.DatetimeIndex(umd_frame["period"])).sort_index().rename("UMD")
    weight = crisis.vol_target_weight(unit)

    simple = lh.market_returns(lh.load_wide("ff3_daily"))["total"]
    price = (1.0 + simple).cumprod()
    sjm = pd.read_parquet(CACHE / "longhist_states.parquet")["state"]
    sjm = sjm[sjm.index >= OOS_START]
    raw = {
        "SJM long": sjm,
        "asym k=10": lh.asymmetric_exit(sjm, simple, k=10),
        "asym k=5": lh.asymmetric_exit(sjm, simple, k=5),
        "median rule": volatility_quantile_placebo(simple),
        "80th rule": crisis.volatility_tail_rule(simple),
        "DM panic": lh.panic_state(price, simple),
    }
    index = unit.index[(unit.index >= OOS_START) & (unit.index <= END)]
    lagged = {k: crisis.lagged_state(v, index) for k, v in raw.items()}
    ready = weight.reindex(index).notna()
    for v in lagged.values():
        ready &= v.notna()
    window = index[ready.to_numpy()]
    a_prime = pd.read_parquet(CACHE / "states.parquet")["A' sparse jump"]
    return {
        "index": window,
        "unit": unit.reindex(window),
        "weight": weight.reindex(window),
        "labels": {k: v.reindex(window) for k, v in lagged.items()},
        "a_prime": crisis.lagged_state(a_prime, window),
        "gaps": int((~ready.loc[window.min():]).sum()),
    }


def arm(p: dict, label: str | None) -> tuple[pd.Series, pd.Series]:
    """(daily return, weight) of UMD alone (``label`` None) or stopped by ``label``."""
    w = p["weight"] if label is None else crisis.couple(p["weight"], p["labels"][label], "stop")
    return (w * p["unit"]).rename(label or "alone"), w


def legs(p: dict, test: str) -> tuple[pd.Series, pd.Series]:
    coupled, reference = TESTS[test][0], TESTS[test][1]
    a = arm(p, coupled)[0]
    b = arm(p, None)[0] if reference == "alone" else arm(p, reference)[0]
    return a, b


def max_drawdown(x: pd.Series) -> float:
    curve = (1.0 + x.fillna(0.0)).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def describe(x: pd.Series, w: pd.Series) -> dict:
    return {"sharpe": lh.sharpe(x), "ann_return": float(x.mean() * 252),
            "ann_vol": float(x.std(ddof=1) * np.sqrt(252)), "max_dd": max_drawdown(x),
            "turnover": crisis.annual_turnover(w), "exposed": float((w > 0).mean())}


# ---------------------------------------------------------------------------
# the instrument
# ---------------------------------------------------------------------------
def instrument(p: dict) -> dict[str, float]:
    print(RULE)
    print("LONGHIST PHASE B — UMD stopped in stress, 1937-2026 — INSTRUMENT")
    print(RULE)
    print("\n0.  INPUTS (SHA-256)\n")
    for path in INPUTS:
        print(f"   {sha256(path)[:16]}  {path.relative_to(RAW.parent)}")

    idx = p["index"]
    alone = arm(p, None)[0]
    nan = int(alone.isna().sum()) + sum(int(v.isna().sum()) for v in p["labels"].values())
    print(f"\n{RULE}\n1.  SAMPLE, COVERAGE, SIZING — no mean, no Sharpe\n")
    print(f"   {idx.min():%Y-%m-%d} -> {idx.max():%Y-%m-%d}, {len(idx):,} sessions, "
          f"{nan} missing values, {p['gaps']} sessions dropped inside the window")
    w = p["weight"]
    print(f"   weight: median {w.median():.2f}, 95th pct {w.quantile(0.95):.2f}, at the cap "
          f"{(w >= crisis.MAX_LEVERAGE).mean():.1%}; realised vol of the alone arm "
          f"{alone.std() * np.sqrt(252):.1%}")
    print(f"\n   {'lagged label':<14} {'stress':>7} {'episodes':>9} {'tr/yr':>6}")
    for name, lab in p["labels"].items():
        stress = lab.eq(lh.STRESS).astype(int)
        episodes = int((stress.diff() == 1).sum() + stress.iloc[0])
        print(f"   {name:<14} {lab.eq(lh.STRESS).mean():>7.1%} {episodes:>9} "
              f"{lh.transitions_per_year(lab):>6.2f}")
    common = p["a_prime"].dropna()
    print(f"   A' (2002+, descriptive only): {common.index.min():%Y-%m-%d} -> "
          f"{common.index.max():%Y-%m-%d}, stress {common.eq(lh.STRESS).mean():.1%}")

    print(f"\n{RULE}\n2.  MINIMUM DETECTABLE EFFECT — demeaned legs, alpha 0.05/{N_TESTS} = "
          f"{ALPHA:.4f}, power 0.80\n")
    mdes: dict[str, float] = {}
    for test in TESTS:
        a, b = legs(p, test)
        values = []
        for block in BLOCKS:
            res = blinded_mde(a.to_numpy(), b.to_numpy(), mean_block=block, draws=DRAWS, seed=0)
            values.append(mde_at(res, ALPHA))
        mdes[test] = max(values)
        joined = " / ".join(f"{v:.3f}" for v in values)
        coupled, reference, _, role = TESTS[test]
        print(f"   {test:<5} {coupled} vs {reference:<11} ({role:<11}) MDE {joined}  ->  "
              f"threshold {mdes[test]:.3f}")
    print(f"\n   HAC t critical value |t| >= {T_CRITICAL:.3f}; placebo {ROTATIONS} rotations "
          f">= {MIN_SHIFT} sessions; {FOLDS} folds, at least 3 positive")
    if nan or not all(np.isfinite(v) for v in mdes.values()):
        print("\n   A READING IS NOT FINITE. No verdict.")
        return {}
    return mdes


# ---------------------------------------------------------------------------
# the reading
# ---------------------------------------------------------------------------
def rotation_null(p: dict, test: str, shifts: np.ndarray) -> np.ndarray:
    coupled, reference = TESTS[test][0], TESTS[test][1]
    b = arm(p, None)[0] if reference == "alone" else arm(p, reference)[0]
    base = lh.sharpe(b)
    values = p["labels"][coupled].to_numpy()
    out = []
    for k in shifts:
        rotated = pd.Series(np.roll(values, k), index=p["index"])
        w = crisis.couple(p["weight"], rotated, "stop")
        out.append(lh.sharpe(w * p["unit"]) - base)
    return np.array(out)


def already_read() -> bool:
    log = trials.read()
    return not log.empty and (log["family"] == FAMILY).any()


def read(p: dict, mdes: dict[str, float]) -> None:
    idx = p["index"]
    print(f"\n{RULE}\n3.  THE ARMS — {idx.min():%Y-%m-%d} -> {idx.max():%Y-%m-%d}, zero cost\n")
    arms = {"alone": arm(p, None)}
    arms.update({f"stop ({k})": arm(p, k) for k in LABELS})
    table = pd.DataFrame({k: describe(*v) for k, v in arms.items()}).T
    print(f"   {'arm':<22} {'Sharpe':>7} {'return':>8} {'vol':>7} {'max DD':>8} "
          f"{'turnover':>9} {'exposed':>8}")
    for name, r in table.iterrows():
        print(f"   {name:<22} {r['sharpe']:>+7.3f} {r['ann_return']:>+8.2%} {r['ann_vol']:>7.2%} "
              f"{r['max_dd']:>8.1%} {r['turnover']:>9.2f} {r['exposed']:>8.1%}")

    alone = arms["alone"][0]
    print("\n   alone, by lagged label (descriptive):")
    for name in ("SJM long", "asym k=10", "80th rule", "DM panic"):
        stress = p["labels"][name].eq(lh.STRESS)
        print(f"     {name:<11} stress {stress.sum():>6,} sessions Sharpe "
              f"{lh.sharpe(alone[stress]):>+6.2f}   calm {(~stress).sum():>6,} Sharpe "
              f"{lh.sharpe(alone[~stress]):>+6.2f}")

    rng = np.random.default_rng(SEED_PLACEBO)
    shifts = rng.integers(MIN_SHIFT, len(idx) - MIN_SHIFT, size=ROTATIONS)
    print(f"\n{RULE}\n4.  THE DECISIONS (§6.5)\n")
    for test, (coupled, reference, controls, role) in TESTS.items():
        a, b = legs(p, test)
        delta = lh.sharpe(a) - lh.sharpe(b)
        t = paired_hac_t(a, b)
        null = rotation_null(p, test, shifts)
        pct = float((null < delta).mean()) if np.isfinite(null).all() else float("nan")
        folds = lh.fold_deltas(a, b, folds=FOLDS)
        positive = int(sum(d > 0 for d in folds))
        ctrl = {c: lh.sharpe(arm(p, c)[0]) - lh.sharpe(alone) for c in controls}
        v = lh.verdict(delta, mdes[test], t, T_CRITICAL, pct, positive, ctrl)
        if not all(np.isfinite(folds)):
            v = "NOT FINITE — no verdict"
        subs = {name: lh.sharpe(a.loc[s:e]) - lh.sharpe(b.loc[s:e])
                for name, (s, e) in SUBPERIODS.items()}
        mean_diff = float((a - b).mean() * 252)
        print(f"   {test} ({role}) {coupled} vs {reference}")
        print(f"      delta {delta:+.3f}  MDE {mdes[test]:.3f}  t HAC6 {t:+.2f}  placebo pct "
              f"{pct:.1%} (median {np.median(null):+.3f}, p95 {np.quantile(null, 0.95):+.3f})")
        print("      folds " + " ".join(f"{d:+.3f}" for d in folds) + f"  ({positive}/5 > 0)")
        if ctrl:
            print("      controls (delta of the same stop): " + ", ".join(
                f"{c} {d:+.3f}" for c, d in ctrl.items()))
        print("      sub-periods: " + ", ".join(f"{k} {d:+.3f}" for k, d in subs.items()))
        print(f"      annual mean of the daily difference {mean_diff:+.2%} (the t above tests it)")
        shown = v if role == "primary" else f"{v}  [SENSITIVITY — does not decide]"
        print(f"      => {shown}\n")
        extra = {}
        if test == "B1":
            common = p["a_prime"].notna() & (idx >= A_PRIME_START)
            cw = p["weight"][common]
            a_stop = crisis.couple(cw, p["a_prime"][common], "stop") * p["unit"][common]
            l_stop = crisis.couple(cw, p["labels"]["SJM long"][common], "stop") * p["unit"][common]
            base_c = lh.sharpe(cw * p["unit"][common])
            extra = {"a_prime_window": f"{idx[common].min().date()} -> {idx[common].max().date()}",
                     "a_prime_delta": lh.sharpe(a_stop) - base_c,
                     "long_delta_same_window": lh.sharpe(l_stop) - base_c}
            raw_alone = p["unit"]
            raw_stop = raw_alone.where(p["labels"]["SJM long"].ne(lh.STRESS), 0.0)
            extra.update({"raw_1x_alone": lh.sharpe(raw_alone), "raw_1x_stop": lh.sharpe(raw_stop),
                          "raw_1x_maxdd_alone": max_drawdown(raw_alone),
                          "raw_1x_maxdd_stop": max_drawdown(raw_stop)})
            print(f"      descriptive, {extra['a_prime_window']}: "
                  f"stop by A' delta {extra['a_prime_delta']:+.3f}, "
                  f"stop by the long SJM delta {extra['long_delta_same_window']:+.3f}")
            print(f"      descriptive, raw UMD at weight 1: alone Sharpe "
                  f"{extra['raw_1x_alone']:+.3f} max DD {extra['raw_1x_maxdd_alone']:.1%}; "
                  f"stopped by the long SJM {extra['raw_1x_stop']:+.3f} max DD "
                  f"{extra['raw_1x_maxdd_stop']:.1%}\n")
        trials.log(
            FAMILY,
            {"test": test, "role": role, "coupled": coupled, "reference": reference,
             "object": "UMD daily, Ken French", "sizing": "min(0.10/sigma63, 3)",
             "coupling": "stop", "cost": "zero", "alpha": f"0.05/{N_TESTS}",
             "state": "longhist reduced SJM, online, lag 1",
             "sample": [str(idx.min().date()), str(idx.max().date())]},
            {"sharpe": lh.sharpe(a), "sharpe_reference": lh.sharpe(b), "delta": delta,
             "threshold": mdes[test], "t_hac": t, "placebo_pct": pct,
             "folds": " ".join(f"{d:+.4f}" for d in folds),
             "folds_positive": positive, **{f"control_{c}": d for c, d in ctrl.items()},
             **{f"sub_{k}": d for k, d in subs.items()}, "mean_diff_annual": mean_diff,
             "sessions": int(len(idx)), **extra, "verdict": v},
        )
    print(f"   logged {len(TESTS)} rows to data/trials.parquet, family {FAMILY} "
          f"({trials.summary()['n_distinct']} distinct configurations)")
    print(RULE)


def main() -> None:
    reading = "--read" in sys.argv
    if reading and already_read():
        print(f"REFUSED: family {FAMILY} is already in data/trials.parquet. One reading only.")
        return
    p = build()
    mdes = instrument(p)
    if not mdes:
        return
    if not reading:
        print(f"\n{RULE}\nNOT READ — re-run with --read, once.\n{RULE}")
        return
    read(p, mdes)


if __name__ == "__main__":
    main()
