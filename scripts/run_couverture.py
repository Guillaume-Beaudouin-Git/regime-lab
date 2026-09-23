"""Which hedge, not only when: the stock-bond correlation regime (idea 3, `couverture`).

The protocol is `docs/PRESPEC_COUVERTURE.md`. EVERYTHING BELOW IS FIXED, AND COMMITTED,
BEFORE ANY RETURN OR VARIANCE CONDITIONED ON THE LABEL IS READ. A plain run prints the
instrument only; `--read` performs the reading, once, logs six rows (the five tests
and the W reference, family `couverture`) and refuses to run if that family is already
in the trials log.

The label
    L_t = 1 when the correlation between ^GSPC log returns and the negative change of
    the 10-year yield (^TNX), over the last 63 sessions where both exist, has been
    positive long enough: a 21-session hysteresis, fixed in the advisor's note before
    this study. 1 = "positive regime" (bonds fall with equities: gold), 0 = negative
    regime (bonds hedge: bonds). No parameter is estimated; the label is causal.

Question V — the variance (two tests, 1991-2026)
    y_t = log sum_{j=t+1..t+21} r_j^2 for a stock-bond portfolio r:
        V1  the programme's frozen 60/40 (^GSPC, 10-year bond from ^TNX at duration
            7.5, monthly rebalanced), in excess of cash;
        V2  equal risk: 0.5 x (^GSPC and the same bond, each at min(0.10/sigma63, 3)).
    base   y ~ 1 + log K_t + vol_rank_t + log RV21_t       (K = VIX implied variance,
           vol_rank = causal expanding rank of ^GSPC 21-session volatility, RV21 = the
           portfolio's own 21-session realised variance)
    full   base + L_t
    PREDICTS      coef > 0, HAC-21 t >= z(1 - alpha/2), rotation placebo >= 95 %, and
                  a positive out-of-sample R2 gain (expanding refits every 252 rows
                  after 2,520, trained only on realised targets)
    NOT SHOWN     coef > 0 and t >= z, but the placebo or the out-of-sample gain fails
    UNDERPOWERED  coef > 0, t < z — never a success
    DOES NOT PREDICT  coef <= 0
    Qualifier, same thresholds: does L still add once the portfolio's 63-session
    variance (the label's own window) joins the base? It decides the wording, not the
    verdict.

Question S — the Sharpe (three tests, 2002-10 -> 2026-07, zero cost)
    Legs as in `run_safe_haven_switch.py`: EQ (Ken French Mkt-RF), TLT and GOLD (GC=F)
    in excess of cash, each at min(0.10/sigma63, 3) known at T-1. The pocket is chosen
    by the label known at T-1: TLT when L = 0, GOLD when L = 1.
        P  permanent pocket: 0.5 EQ + 0.5 pocket
        W  the switch: EQ when the sparse jump state (lag 1) is calm, pocket in stress
    Tests: P_TLT, P_GOLD, P_MIX (fixed 50/50 TLT/gold pocket). USEFUL for the idea
    only if all three are USEFUL (an intersection-union claim: the label beats every
    fixed pocket, not the one that happened to lose).
    W is a REFERENCE, not a test: the instrument shows that the label picks gold on
    one of the 832 stress sessions, so W is the fixed-TLT switch already read in
    `docs/RESULTS_REFUGE.md`. Its reading is printed and logged, with no verdict.
    delta = Sharpe(label-chosen) - Sharpe(fixed comparator); MDE blinded, blocks
    21/63/126, the largest, alpha 0.05/5; the verdict of `run_crisis_coupling.verdict`
    with the rotation placebo of L, the median and 80th-percentile volatility rules as
    selectors (high volatility -> TLT), and a third witness, the VIX rule: USEFUL also
    needs delta above the VIX-rule delta, otherwise NOT SHOWN.

Family: 5 tests (V1, V2, P_TLT, P_GOLD, P_MIX), Bonferroni alpha = 0.05/5. Nothing is
added after reading.

Usage
    .venv/bin/python scripts/run_couverture.py          # instrument only
    .venv/bin/python scripts/run_couverture.py --read   # the reading, once
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_crisis_coupling import (  # noqa: E402
    MIN_SHIFT,
    ROTATIONS,
    WINDOWS,
    cohen_kappa,
    max_drawdown,
    series,
    sha256,
    sharpe,
    verdict,
)
from run_m3_evaluation import cash_rate_daily  # noqa: E402

from regime_lab.analysis import trials  # noqa: E402
from regime_lab.config import CACHE, RAW  # noqa: E402
from regime_lab.evaluation.predictive import volatility_quantile_placebo  # noqa: E402
from regime_lab.extensions import couverture as cv  # noqa: E402
from regime_lab.extensions import crisis  # noqa: E402
from regime_lab.selection.protocol import blinded_mde, mde_at, paired_hac_t  # noqa: E402
from regime_lab.strategies.base import bond_return, sixty_forty  # noqa: E402

RULE = "=" * 78
FAMILY = "couverture"
STATE_COLUMN = "A' sparse jump"
OOS_START = pd.Timestamp("2002-04-01")
N_TESTS = 5
ALPHA = 0.05 / N_TESTS
Z = float(stats.norm.ppf(1.0 - ALPHA / 2.0))
BLOCKS = (21, 63, 126)
DRAWS = 2000
HORIZON = cv.HORIZON
MIN_TRAIN = 2520
REFIT = 252
PRIMARY = ("log_k", "vol_rank", "log_rv21")
HARDENED = (*PRIMARY, "log_rv63")
PORTFOLIOS = {"V1_6040": "frozen 60/40", "V2_equal_risk": "equal-risk stock-bond"}
SHARPE_TESTS = (("P_TLT", "P", 1.0), ("P_GOLD", "P", 0.0), ("P_MIX", "P", 0.5))
WITNESSES = ("median", "80th", "vix")
WORST_SHARE = 0.05
INPUTS = (
    RAW / "prices" / "cross_asset.parquet",
    RAW / "panels" / "factors_5.parquet",
    CACHE / "trend_universe_m1.parquet",
    RAW / "macro" / "rate_cash_3m.parquet",
    CACHE / "states.parquet",
)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def market() -> dict[str, pd.Series]:
    path = RAW / "prices" / "cross_asset.parquet"
    spx = series(path, "eq_us_large")
    y10 = series(path, "bond_us_10y").reindex(spx.index)
    vix = series(path, "vol_vix").reindex(spx.index)
    log = np.log(spx).diff()
    return {"spx": spx, "y10": y10, "vix": vix, "log": log, "simple": spx.pct_change(),
            "rv21": log.rolling(21).std() * np.sqrt(252)}


def the_label(m: dict[str, pd.Series]) -> tuple[pd.Series, pd.Series]:
    """The correlation level and the regime label, on the sessions where both exist."""
    corr = cv.rolling_correlation(m["log"], -m["y10"].diff())
    return corr, cv.hysteresis(corr > 0, cv.HYSTERESIS).rename("label")


def variance_portfolios(m: dict[str, pd.Series]) -> dict[str, pd.Series]:
    idx = m["spx"].index
    cash = cash_rate_daily(idx)
    y10 = m["y10"].ffill()
    panel = pd.DataFrame({"eq_us_large": m["spx"], "bond_us_10y": y10})
    v1 = (sixty_forty(panel) - cash).dropna()
    eq = m["simple"] - cash
    bd = bond_return(y10, "bond_us_10y") - cash
    v2 = 0.5 * (crisis.vol_target_weight(eq.dropna()) * eq
                + crisis.vol_target_weight(bd.dropna()) * bd)
    return {"V1_6040": v1, "V2_equal_risk": v2.dropna()}


def variance_frame(r: pd.Series, m: dict, corr: pd.Series, label: pd.Series,
                   states: pd.Series) -> pd.DataFrame:
    idx = r.index
    frame = pd.DataFrame({
        "y": cv.forward_log_variance(r),
        "log_k": np.log(crisis.monthly_variance(m["vix"].reindex(idx))),
        "vol_rank": cv.expanding_rank(m["rv21"]).reindex(idx),
        "log_rv21": cv.trailing_log_variance(r, window=21),
        "log_rv63": cv.trailing_log_variance(r, window=63),
        "label": cv.asof(label, idx),
        "rho": cv.asof(corr, idx),
    }).dropna()
    frame["state"] = cv.asof(states, frame.index)
    return frame


def hedge_units() -> dict[str, pd.Series]:
    eq = series(RAW / "panels" / "factors_5.parquet", "ff_mkt-rf") / 100.0
    prices = pd.read_parquet(CACHE / "trend_universe_m1.parquet")
    out = {"EQ": eq}
    for leg, ticker in (("TLT", "TLT"), ("GOLD", "GC=F")):
        px = prices[ticker].dropna()
        out[leg] = px.pct_change() - cash_rate_daily(px.index)
    return {k: v.dropna().rename(k) for k, v in out.items()}


def build_hedge(m: dict, label: pd.Series, states: pd.Series) -> dict:
    u = hedge_units()
    index = u["EQ"].index
    r = {k: v.reindex(index).fillna(0.0) for k, v in u.items()}
    w = {k: crisis.vol_target_weight(v).reindex(index) for k, v in u.items()}
    labels = {
        "corr": crisis.lagged_state(label, index),
        "state": crisis.lagged_state(states, index),
        "median": crisis.lagged_state(volatility_quantile_placebo(m["simple"]), index),
        "80th": crisis.lagged_state(crisis.volatility_tail_rule(m["simple"]), index),
        "vix": crisis.lagged_state(cv.vix_rule(m["vix"]), index),
    }
    ready = pd.Series(True, index=index)
    for s in (*w.values(), *labels.values()):
        ready &= s.notna()
    end = min(states.dropna().index.max(), index.max())
    start = max(OOS_START, ready[ready].index.min())
    window = index[(index >= start) & (index <= end)]
    return {
        "index": window,
        "returns": pd.DataFrame({"EQ": r["EQ"], "BOND": r["TLT"], "GOLD": r["GOLD"]}
                                ).reindex(window),
        "w": {k: v.reindex(window) for k, v in w.items()},
        "labels": {k: v.reindex(window) for k, v in labels.items()},
        "not_ready_inside": int((~ready.reindex(window)).sum()),
    }


def arm(p: dict, obj: str, bond_share, *, gold_on: bool = True) -> tuple[pd.Series, float]:
    """The book of object ``obj`` ('P' or 'W') with the pocket's bond share given."""
    w = p["w"]
    if obj == "P":
        equity_on, pocket_on = 0.5, 0.5
    else:
        stress = p["labels"]["state"].eq(crisis.STRESS).astype(float)
        equity_on, pocket_on = 1.0 - stress, stress
    gold = w["GOLD"] if gold_on else 0.0 * w["GOLD"]
    held = cv.held_weights(w["EQ"], w["TLT"], gold, bond_share, equity_on=equity_on,
                           pocket_on=pocket_on)
    return cv.book(p["returns"], held)


def chosen_share(label: pd.Series) -> pd.Series:
    """Label 0 -> bonds, label 1 -> gold: the bond share is ``1 - label``."""
    return 1.0 - label


# ---------------------------------------------------------------------------
# the instrument
# ---------------------------------------------------------------------------
def instrument(m: dict, corr: pd.Series, label: pd.Series, states: pd.Series,
               frames: dict[str, pd.DataFrame], p: dict) -> dict | None:
    print(RULE)
    print("WHICH HEDGE, NOT ONLY WHEN — THE STOCK-BOND CORRELATION REGIME — INSTRUMENT")
    print(RULE)
    print("\n0.  INPUTS (SHA-256)\n")
    for path in INPUTS:
        print(f"   {sha256(path)[:16]}  {path.relative_to(RAW.parent)}")

    print(f"\n{RULE}\n1.  THE LABEL — a property of the label, no return, no variance read\n")
    advisor = m["log"].rolling(63).corr(-m["y10"].diff()).dropna()
    print(f"   ^GSPC sessions {len(m['spx']):,}; of which without a 10-year yield (bond "
          f"holidays) {int(m['y10'].isna().sum())}")
    print(f"   correlation defined on {len(corr):,} sessions "
          f"({corr.index[0]:%Y-%m-%d} -> {corr.index[-1]:%Y-%m-%d}); the advisor's "
          f"align-then-roll construction kept {len(advisor):,}")
    print(f"   positive share: level {(corr > 0).mean():.1%}, label {label.mean():.1%}")
    raw = cv.transitions_per_year((corr > 0).astype(float))
    print(f"   transitions per year: raw sign {raw:.2f}, label (hysteresis "
          f"{cv.HYSTERESIS}) {cv.transitions_per_year(label):.2f}")
    by_year = label.groupby(label.index.year).mean()
    print("   label share positive, by year:")
    years = [f"{y}:{v:.0%}" for y, v in by_year.items()]
    for i in range(0, len(years), 10):
        print("     " + " ".join(years[i:i + 10]))

    print(f"\n{RULE}\n2.  THE OPPOSABLE RULE — the four axes, measured before any reading\n")
    j = pd.concat({"rho": corr, "label": label, "rv": m["rv21"], "vix": m["vix"]},
                  axis=1).dropna()
    print("   axis 1, the latent (Spearman, 1990-2026, same session):")
    for col in ("rho", "label"):
        print(f"     {col:<6} with ^GSPC RV21 {stats.spearmanr(j[col], j['rv'])[0]:+.2f}   "
              f"with the VIX {stats.spearmanr(j[col], j['vix'])[0]:+.2f}")
    s = pd.concat({"state": states, "rv": m["rv21"], "vix": m["vix"]}, axis=1)
    s = s.loc[OOS_START:].dropna()
    print(f"     A' state with ^GSPC RV21 {stats.spearmanr(s['state'], s['rv'])[0]:+.2f}   "
          f"with the VIX {stats.spearmanr(s['state'], s['vix'])[0]:+.2f}   (2002-2026, "
          "1 = calm)")
    lab = p["labels"]
    print(f"   agreement of the lagged label with each selector, Cohen's kappa on the "
          f"hedge window ({p['index'].min():%Y-%m-%d} -> {p['index'].max():%Y-%m-%d};")
    print("   every label oriented 1 -> gold, 0 -> bonds; the sparse jump state 1 = calm):")
    for name in ("median", "80th", "vix", "state"):
        k = cohen_kappa(lab["corr"].eq(1.0).to_numpy(), lab[name].eq(1.0).to_numpy())
        print(f"     {name:<7} kappa {k:+.2f}   share 1: {lab[name].mean():.1%}")
    print(f"   axis 2, the clock: {cv.transitions_per_year(label):.2f} transitions per year "
          "(not crossed: under ~2)")
    print("   axis 3, the usage: P chooses between two hedges; W chooses inside a stress "
          "switch; V is a forecast input")
    units = pd.DataFrame({k: p["w"][k] * p["returns"][c] for k, c in
                          (("EQ", "EQ"), ("TLT", "BOND"), ("GOLD", "GOLD"))})
    print(f"   axis 4, the object: participation ratio of EQ, TLT, GOLD (sized, "
          f"unconditional) {cv.participation_ratio(units.corr().to_numpy()):.2f} of 3 "
          "(not crossed)")

    stress = lab["state"].eq(crisis.STRESS)
    gold = lab["corr"].eq(1.0)
    print("\n   where the label can act (label only):")
    print(f"     P: sessions on gold {gold.mean():.1%} of {len(gold):,}; label changes in "
          f"the window {int((lab['corr'].diff().abs() > 0).sum())}")
    print(f"     W: stress sessions {int(stress.sum()):,}; of which on gold "
          f"{int((stress & gold).sum()):,} ({(gold[stress]).mean():.1%})")
    runs = (stress != stress.shift()).cumsum()[stress]
    for _, g in runs.groupby(runs):
        d = g.index
        print(f"       stress run {d[0]:%Y-%m-%d} -> {d[-1]:%Y-%m-%d}  {len(d):>4} sessions, "
              f"gold {int(gold.reindex(d).sum()):>4}")

    print(f"\n{RULE}\n3.  QUESTION V — design and blinded detection threshold, alpha "
          f"0.05/{N_TESTS} = {ALPHA:.4f}, |t| >= {Z:.2f}\n")
    bad = False
    v_mde = {}
    for name, frame in frames.items():
        nan = int(frame[list(HARDENED) + ["y", "label"]].isna().sum().sum())
        bad |= nan > 0
        mde = cv.regression_mde(frame["y"].to_numpy(), frame[list(PRIMARY)].to_numpy(),
                                frame["label"].to_numpy(), lags=HORIZON, alpha=ALPHA)
        v_mde[name] = mde
        oos_first = frame.index[MIN_TRAIN] if len(frame) > MIN_TRAIN else None
        print(f"   {name:<14} {frame.index.min():%Y-%m-%d} -> {frame.index.max():%Y-%m-%d}"
              f"  n {len(frame):,}  label positive {frame['label'].mean():.1%}  NaN {nan}")
        print(f"   {'':<14} detectable coefficient {mde['beta']:.3f} log points, incremental "
              f"R2 {100 * mde['incremental_r2']:.2f} pt (80 % power); out of sample from "
              f"{oos_first:%Y-%m-%d}")

    print(f"\n{RULE}\n4.  QUESTION S — sample and blinded MDE, demeaned legs, alpha "
          f"0.05/{N_TESTS}\n")
    idx = p["index"]
    print(f"   {idx.min():%Y-%m-%d} -> {idx.max():%Y-%m-%d}, {len(idx):,} sessions, not ready "
          f"inside the window: {p['not_ready_inside']}")
    bad |= p["not_ready_inside"] > 0
    for leg, w in p["w"].items():
        print(f"   {leg:<5} weight p50 {w.median():.2f}  p95 {w.quantile(0.95):.2f}  at cap "
              f"{(w >= crisis.MAX_LEVERAGE).mean():.1%}")
    s_mde = {}
    for test, obj, fixed in SHARPE_TESTS:
        chosen, _ = arm(p, obj, chosen_share(lab["corr"]))
        comp, _ = arm(p, obj, fixed)
        bad |= bool(chosen.isna().any() or comp.isna().any())
        values = [mde_at(blinded_mde(chosen.to_numpy(), comp.to_numpy(), mean_block=b,
                                     draws=DRAWS, seed=0), ALPHA) for b in BLOCKS]
        s_mde[test] = max(values)
        print(f"   {test:<7} MDE " + " / ".join(f"{v:.3f}" for v in values)
              + f"  ->  threshold {s_mde[test]:.3f}")
    finite = [m_["beta"] for m_ in v_mde.values()] + list(s_mde.values())
    if bad or not all(np.isfinite(finite)):
        print("\n   A READING IS NOT FINITE. No verdict.")
        return None
    return {"v": v_mde, "s": s_mde}


# ---------------------------------------------------------------------------
# the reading
# ---------------------------------------------------------------------------
def verdict_v(coef: float, t: float, pct: float, gain: float) -> str:
    if not all(np.isfinite([coef, t, pct, gain])):
        return "NOT FINITE — no verdict"
    if coef > 0 and t >= Z:
        if pct >= 0.95 and gain > 0:
            return "PREDICTS — beyond volatility and the VIX, in and out of sample"
        return "NOT SHOWN — significant, fails the placebo or the out-of-sample gain"
    if coef > 0:
        return "UNDERPOWERED — right sign, below the threshold, never a success"
    return "DOES NOT PREDICT — the label does not raise the forecast risk"


def verdict_s(delta: float, mde: float, t: float, pct: float, d: dict[str, float]) -> str:
    v = verdict(delta, mde, t, pct, d["median"], d["80th"])
    if v.startswith("USEFUL") and not delta > d["vix"]:
        return "NOT SHOWN — above the MDE, fails the VIX rule"
    return v


def read_v(frames: dict[str, pd.DataFrame], mdes: dict) -> None:
    rng = np.random.default_rng(20260925)
    print(f"\n{RULE}\nQUESTION V — does the correlation regime forecast a stock-bond "
          "portfolio's risk beyond volatility and the VIX?\n")
    for name, frame in frames.items():
        y = frame["y"].to_numpy()
        lab = frame["label"].to_numpy()
        prim = frame[list(PRIMARY)].to_numpy()
        hard = frame[list(HARDENED)].to_numpy()
        real = cv.incremental_fit(y, prim, lab, lags=HORIZON)
        shifts = rng.integers(MIN_SHIFT, len(y) - MIN_SHIFT, size=ROTATIONS)
        null = np.array([cv.incremental_r2(y, prim, np.roll(lab, k)) for k in shifts])
        pct = float((null < real["incremental"]).mean())
        oos = cv.out_of_sample_gain(y, prim, lab, min_train=MIN_TRAIN, refit=REFIT)
        h = cv.incremental_fit(y, hard, lab, lags=HORIZON)
        h_oos = cv.out_of_sample_gain(y, hard, lab, min_train=MIN_TRAIN, refit=REFIT)
        monthly = frame.iloc[::HORIZON]
        sens = cv.incremental_fit(monthly["y"].to_numpy(), monthly[list(PRIMARY)].to_numpy(),
                                  monthly["label"].to_numpy(), lags=6)
        rho = cv.incremental_fit(y, prim, frame["rho"].to_numpy(), lags=HORIZON)
        late = frame.loc[OOS_START:].dropna(subset=["state"])
        y_l, p_l = late["y"].to_numpy(), late[list(PRIMARY)].to_numpy()
        st = cv.incremental_fit(y_l, p_l, late["state"].to_numpy(), lags=HORIZON)
        lab_late = cv.incremental_fit(y_l, p_l, late["label"].to_numpy(), lags=HORIZON)
        both = cv.incremental_fit(y_l, np.column_stack([p_l, late["state"].to_numpy()]),
                                  late["label"].to_numpy(), lags=HORIZON)
        v = verdict_v(real["coef"], real["t"], pct, oos["gain"])
        beyond63 = (h["coef"] > 0 and h["t"] >= Z and h_oos["gain"] > 0)
        print(f"   {name} — {PORTFOLIOS[name]}, n {real['n']:,}, R2 base {real['r2_base']:.1%}")
        detectable = 100 * mdes[name]["incremental_r2"]
        print(f"     label: coef {real['coef']:+.3f}  t HAC21 {real['t']:+.2f}  incremental R2 "
              f"{100 * real['incremental']:.2f} pt  (detectable {detectable:.2f} pt)  "
              f"placebo pct {pct:.1%} (p95 {100 * np.quantile(null, 0.95):.2f} pt)")
        print(f"     out of sample ({oos['n']:,} rows): R2 gain {100 * oos['gain']:+.2f} pt  "
              f"Clark-West t {oos['cw_t']:+.2f}")
        print(f"     + 63-session variance in the base: coef {h['coef']:+.3f}  t {h['t']:+.2f}  "
              f"incremental {100 * h['incremental']:.2f} pt  OOS gain {100 * h_oos['gain']:+.2f} pt"
              f"  -> beyond it: {'yes' if beyond63 else 'no'}")
        print(f"     sensitivity, every 21st session, HAC 6: coef {sens['coef']:+.3f}  t "
              f"{sens['t']:+.2f}  n {sens['n']}")
        print(f"     descriptive, the continuous correlation instead of the label: coef "
              f"{rho['coef']:+.3f}  t {rho['t']:+.2f}  incremental "
              f"{100 * rho['incremental']:.2f} pt")
        print(f"     2002-2026 ({late.shape[0]:,} rows): A' state beyond the same base "
              f"{100 * st['incremental']:.2f} pt (t {st['t']:+.2f}); label "
              f"{100 * lab_late['incremental']:.2f} pt (t {lab_late['t']:+.2f}); label beyond "
              f"base + state {100 * both['incremental']:.2f} pt (t {both['t']:+.2f})")
        print(f"     => {v}")
        trials.log(
            FAMILY,
            {"test": name, "portfolio": PORTFOLIOS[name],
             "target": "log forward 21-session realised variance",
             "controls": list(PRIMARY), "regressor": "correlation regime (63, hysteresis 21)",
             "hac": HORIZON, "alpha": f"0.05/{N_TESTS}",
             "sample": [str(frame.index.min().date()), str(frame.index.max().date())]},
            {"coef": real["coef"], "t_hac": real["t"], "incremental_r2": real["incremental"],
             "mde_incremental_r2": mdes[name]["incremental_r2"], "placebo_pct": pct,
             "oos_gain": oos["gain"], "cw_t": oos["cw_t"], "hardened_coef": h["coef"],
             "hardened_t": h["t"], "hardened_incremental_r2": h["incremental"],
             "hardened_oos_gain": h_oos["gain"], "beyond_rv63": bool(beyond63),
             "t_monthly": sens["t"], "state_incremental_r2": st["incremental"],
             "label_beyond_state_incremental_r2": both["incremental"],
             "sessions": real["n"], "verdict": v},
        )


def describe(x: pd.Series, turnover: float) -> dict:
    monthly = (1.0 + x).groupby(x.index.to_period("M")).prod() - 1.0
    row = {"sharpe": sharpe(x), "ann_return": float(x.mean() * 252),
           "ann_vol": float(x.std(ddof=1) * np.sqrt(252)), "max_dd": max_drawdown(x),
           "worst_month": float(monthly.min()), "turnover": turnover}
    for name, (a, b) in WINDOWS.items():
        part = x.loc[a:b]
        row[name] = float((1.0 + part).prod() - 1.0) if len(part) else float("nan")
    return row


def monthly(x: pd.Series) -> pd.Series:
    return (1.0 + x).groupby(x.index.to_period("M")).prod() - 1.0


def read_s(p: dict, mdes: dict) -> None:
    lab = p["labels"]
    fmt = {"sharpe": "{:+.2f}", "ann_return": "{:+.1%}", "ann_vol": "{:.1%}",
           "max_dd": "{:.1%}", "worst_month": "{:+.1%}", "turnover": "{:.1f}",
           **{c: "{:+.1%}" for c in WINDOWS}}
    eq_alone = p["w"]["EQ"] * p["returns"]["EQ"]
    rows = {"EQ alone": describe(eq_alone, crisis.annual_turnover(p["w"]["EQ"]))}
    arms = {}
    for obj in ("P", "W"):
        arms[(obj, "corr")] = arm(p, obj, chosen_share(lab["corr"]))
        arms[(obj, "TLT")] = arm(p, obj, 1.0)
        arms[(obj, "GOLD")] = arm(p, obj, 0.0)
        arms[(obj, "MIX")] = arm(p, obj, 0.5)
    for wname in WITNESSES:
        arms[("P", wname)] = arm(p, "P", chosen_share(lab[wname]))
    arms[("P", "TLT/cash ref.")] = arm(p, "P", chosen_share(lab["corr"]), gold_on=False)
    for (obj, name), (x, turn) in arms.items():
        rows[f"{obj} {name}"] = describe(x, turn)
    table = pd.DataFrame(rows).T
    idx = p["index"]
    print(f"\n{RULE}\nQUESTION S — {idx.min():%Y-%m-%d} -> {idx.max():%Y-%m-%d}, zero cost, "
          "excess of cash\n")
    print(f"   {'arm':<17}" + "".join(f"{c:>9}" for c in table.columns))
    for name, r in table.iterrows():
        print(f"   {name:<17}" + "".join(f"{fmt[c].format(r[c]):>9}" for c in table.columns))

    gold = lab["corr"].eq(1.0)
    print("\n   each leg alone, by lagged label (descriptive; 0 = negative regime, bonds):")
    for leg, col in (("EQ", "EQ"), ("TLT", "BOND"), ("GOLD", "GOLD")):
        x = p["w"][leg] * p["returns"][col]
        print(f"     {leg:<5} label 1 Sharpe {sharpe(x[gold]):+.2f} ({x[gold].mean() * 252:+.1%}"
              f"/yr, {int(gold.sum()):,} sessions)   label 0 Sharpe {sharpe(x[~gold]):+.2f} "
              f"({x[~gold].mean() * 252:+.1%}/yr)")

    eq_m = monthly(eq_alone)
    worst = eq_m[eq_m <= eq_m.quantile(WORST_SHARE)].index
    print(f"\n   the advisor's draft statistic, reported, not deciding: the {len(worst)} worst "
          f"months of EQ ({WORST_SHARE:.0%})")
    for name in ("corr", "TLT", "GOLD", "MIX"):
        pm = monthly(arms[("P", name)][0]).reindex(worst)
        print(f"     P {name:<5} mean {pm.mean():+.2%}  std {pm.std():.2%}")

    rng = np.random.default_rng(20260926)
    shifts = rng.integers(MIN_SHIFT, len(idx) - MIN_SHIFT, size=ROTATIONS)
    values = lab["corr"].to_numpy()
    rotated = [arm(p, "P", chosen_share(pd.Series(np.roll(values, k), index=idx)))[0]
               for k in shifts]
    null = np.array([sharpe(x) for x in rotated])
    mix_var = monthly(arms[("P", "MIX")][0]).reindex(worst).var()
    real_ratio = monthly(arms[("P", "corr")][0]).reindex(worst).var() / mix_var
    null_ratio = np.array([monthly(x).reindex(worst).var() / mix_var for x in rotated])
    draft_pct = float((null_ratio < real_ratio).mean())
    print(f"     variance ratio P corr / P MIX {real_ratio:.3f}; share of rotations below it "
          f"{draft_pct:.1%} ({ROTATIONS} rotations; the draft asked <= 5 %)")

    print("\n   decision:")
    for test, obj, fixed in SHARPE_TESTS:
        chosen = arms[(obj, "corr")][0]
        comp = arms[(obj, {1.0: "TLT", 0.0: "GOLD", 0.5: "MIX"}[fixed])][0]
        base = sharpe(comp)
        delta = sharpe(chosen) - base
        t = paired_hac_t(chosen, comp)
        nd = null - base
        pct = float((nd < delta).mean()) if np.isfinite(nd).all() else float("nan")
        d = {wname: sharpe(arms[(obj, wname)][0]) - base for wname in WITNESSES}
        ddd = max_drawdown(chosen) - max_drawdown(comp)
        v = verdict_s(delta, mdes[test], t, pct, d)
        print(f"   {test:<7} delta {delta:+.3f}  MDE {mdes[test]:.3f}  t {t:+.2f}  placebo pct "
              f"{pct:.1%} (p5 {np.quantile(nd, 0.05):+.3f}, p95 {np.quantile(nd, 0.95):+.3f})")
        print(f"           witnesses: median {d['median']:+.3f}, 80th {d['80th']:+.3f}, "
              f"VIX {d['vix']:+.3f}   change in max DD {ddd:+.1%}")
        print(f"           => {v}")
        extra = ({"draft_variance_ratio": real_ratio, "draft_placebo_pct": draft_pct}
                 if test == "P_MIX" else {})
        trials.log(
            FAMILY,
            {"test": test, "object": "0.5 EQ + 0.5 pocket",
             "comparator": f"bond share {fixed}", "selector": "correlation regime, lag 1",
             "cost": "zero", "sizing": "min(0.10/sigma63, 3) per leg",
             "alpha": f"0.05/{N_TESTS}", "sample": [str(idx.min().date()),
                                                   str(idx.max().date())]},
            {"sharpe": sharpe(chosen), "sharpe_comparator": base, "delta": delta,
             "threshold": mdes[test], "t_hac": t, "placebo_pct": pct,
             "delta_median_rule": d["median"], "delta_tail_rule": d["80th"],
             "delta_vix_rule": d["vix"], "delta_max_dd": ddd,
             "turnover": arms[(obj, "corr")][1], "sessions": int(len(idx)), "verdict": v,
             **extra},
        )

    stress = lab["state"].eq(crisis.STRESS)
    on_gold = int((stress & gold).sum())
    w_sharpe = {name: sharpe(arms[("W", name)][0]) for name in ("corr", "TLT", "GOLD", "MIX")}
    print(f"\n   W, reference, not a test: the label picks gold on {on_gold} of "
          f"{int(stress.sum())} stress sessions")
    print("     Sharpe " + "   ".join(f"W {k} {v:+.2f}" for k, v in w_sharpe.items()))
    trials.log(
        FAMILY,
        {"test": "W_reference", "object": "EQ calm / pocket in A' stress",
         "selector": "correlation regime, lag 1", "cost": "zero",
         "sizing": "min(0.10/sigma63, 3) per leg",
         "sample": [str(idx.min().date()), str(idx.max().date())]},
        {"sharpe": w_sharpe["corr"], "sharpe_tlt_switch": w_sharpe["TLT"],
         "sharpe_gold_switch": w_sharpe["GOLD"], "sharpe_mix_switch": w_sharpe["MIX"],
         "stress_sessions_on_gold": on_gold, "sessions": int(len(idx)),
         "verdict": "REFERENCE — not a test: the label and the stress state do not overlap"},
    )


def main() -> None:
    m = market()
    corr, label = the_label(m)
    states = pd.read_parquet(CACHE / "states.parquet")[STATE_COLUMN]
    frames = {name: variance_frame(r, m, corr, label, states)
              for name, r in variance_portfolios(m).items()}
    p = build_hedge(m, label, states)
    mdes = instrument(m, corr, label, states, frames, p)
    if mdes is None:
        return
    if "--read" not in sys.argv:
        print(f"\n{RULE}\nNOT READ — re-run with --read, once.\n{RULE}")
        return
    done = trials.read()
    if not done.empty and (done["family"] == FAMILY).any():
        print(f"\n{RULE}\nREFUSED — family '{FAMILY}' is already in the trials log: the "
              f"reading happens once.\n{RULE}")
        return
    read_v(frames, mdes["v"])
    read_s(p, mdes["s"])
    print(f"\n   logged to data/trials.parquet ({trials.summary()['n_distinct']} distinct)")
    print(RULE)


if __name__ == "__main__":
    main()
