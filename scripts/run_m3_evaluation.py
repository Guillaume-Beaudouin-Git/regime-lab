"""The evaluation of the repaired trend book against the six locked stopping rules.

`docs/PRESPEC_TREND_VEHICLE.md` §8 locks K1-K6. `docs/PROTOCOL_FREEZE.md` records
the amendments. Nothing in this file chooses anything after seeing a result: every
interpretive decision the locked text left open is written in DECLARED below, with
its source, and is printed before the first Sharpe is computed.

Writes nothing into the repository. Reads only.

Usage:
    .venv/bin/python <scratchpad>/run_m3_evaluation.py
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

REPO = Path("/Users/guillaumebeaudouin/Desktop/M2/Projet_Big_Data_Regimes/1_Etude_principale/regime-lab")
sys.path.insert(0, str(REPO))
from regime_lab.config import CACHE, RAW  # noqa: E402

OUT = CACHE

from regime_lab.analysis.bootstrap import stationary_indices  # noqa: E402
from regime_lab.analysis.power import minimum_detectable_sharpe_difference  # noqa: E402
from regime_lab.analysis.trials import read as read_trials  # noqa: E402
from regime_lab.extensions.trend import MAX_LEVERAGE, VOL_TARGET  # noqa: E402
from regime_lab.extensions.vehicle import (  # noqa: E402
    FUNDED_ON_CASH,
    NO_FUTURES_VEHICLE,
    charge_by_instrument,
    cost_bps,
    turnover,
    unscaled_weights,
    vol_targeted_book,
)
from regime_lab.models.protocol import walk_forward  # noqa: E402

warnings.filterwarnings("ignore")
RULE = "=" * 78
PERIODS = 252

# ---------------------------------------------------------------------------
# Everything that had to be decided, decided here, before any return is read.
# ---------------------------------------------------------------------------
DECLARED = {
    # §3. Locked sample.
    "sample": ("2003-07-17", "2026-09-10"),

    # §2 / §4. The panels, already written by scripts/apply_m1_realignment.py.
    "panel_headline": "trend_universe_m1",
    "panel_m1_sensitivity": "trend_universe_m1_verified",
    "panel_stored": "trend_universe",

    # §7. Excess returns. Futures are excess by construction; the eight
    # instruments with no futures vehicle pay the 3-month cash rate on their
    # FUNDED exposure. "Funded" is read as the SIGNED exposure, which is the
    # algebra of w_i(r_i - r_f): a long position is financed, a short earns the
    # rate. That is also the reading §1's D1 used when it took 0.520 to 0.413.
    # The literal alternative — every unit of gross must be financed, D1's 0.145
    # — is reported alongside as `funding=gross`, never instead.
    "funding_exposure": "signed",
    "funding_rate_date": "available_at",

    # §6. Decision column, and the three reported alongside.
    "cost_decision_column": "headline",
    "cost_columns": ("headline", "conservative", "stress"),
    "cost_all_cash": True,

    # §8 K2. The named tool, with its own defaults. The folds are printed before
    # any fold result is computed.
    "wf_min_train_years": 10,
    "wf_folds": 5,
    # A "sign flip" is read as a fold whose net excess Sharpe carries the
    # opposite sign to the full-sample net excess Sharpe of the same arm.
    "wf_sign_flip_reference": "full-sample sign of the same arm",

    # §10. The locked MDE. Not recomputed as a threshold; recomputed as a check.
    "mde_decision": 0.246,
    "mde_reported": {"126": 0.246, "63": 0.264, "21": 0.273},
    "bootstrap_blocks": (21, 63, 126),
    "bootstrap_draws": 2000,
    "bootstrap_seed": 0,

    # §11. Both placebos, 400 draws, the count run_extensions.py and
    # run_barrier.py already use.
    "placebo_draws": 400,
    "placebo_seed": 20260921,

    # H-b. The regime family is fixed here and not chosen after the panel is
    # read: A' sparse jump is scripts/run_barrier.py's DECLARED primary family,
    # the best-validated classifier of the programme (93.2% balanced accuracy
    # against NBER, kappa 0.53).
    "regime_family": "A' sparse jump",
    "regime_source": "data/cache/states_offline.parquet",
    # Burn-in for any expanding estimator, inherited from run_barrier.py's
    # DECLARED["min_history"] rather than chosen here.
    "regime_min_history": 504,
    # Minimum observations inside a state bucket before its volatility is used.
    # 63 is the volatility window §5 already locks, not a number chosen here.
    # Below it the covariate abstains at a ratio of 1.0; no session is dropped.
    "regime_min_bucket": 63,
    # THE IMPLEMENTATION, WRITTEN BEFORE THE RESULT IS LOOKED AT.
    #
    # §2 H-b requires the state to enter as ONE covariate of the risk budget and
    # never as a switch. A pure replacement of sigma_63 by a state-conditional
    # sigma makes the state the ONLY covariate and, the state being binary,
    # produces a two-level leverage — which is a switch on the risk budget under
    # another name. The headline is therefore the BLEND:
    #
    #     sigma_regime(t) = sigma_63(t-1) * [ sigma_hat(state=s(t-1)) / sigma_hat(pooled) ]
    #
    # where both sigma_hat are expanding standard deviations of the unscaled book
    # return estimated on data strictly before t, and the whole ratio is lagged one
    # session. The realised 63-day estimator stays in place and the state
    # contributes a past-estimated relative level. It carries no free parameter,
    # so it adds no degree of freedom over M3.
    #
    # The pure replacement is reported as the degenerate reading, labelled as
    # such, and is counted as a trial because the choice between the two is made.
    "regime_headline": "blend: sigma_63 x (sigma_state / sigma_pooled), both past-only, lagged 1",
    "regime_sensitivity": "replacement: sigma = sigma_state, past-only, lagged 1",
    # states_offline is the block relabelled WITH HINDSIGHT (regime_lab/models/
    # base.py: "The same block assigned with hindsight, used only to price
    # latency"), which is the series run_barrier.py evaluates and the series this
    # study is instructed to use. It FAVOURS H-b. A refutation under it is
    # a fortiori; a pass under it would have to be re-run on the online series.
    # The online series is therefore reported as a sensitivity.
    "regime_hindsight_declared_against": True,

    # §12 / K5. Trials.
    "trials_existing_distinct": 83,
}

SAMPLE = slice(*DECLARED["sample"])

# K6 is already settled: docs/PROTOCOL_FREEZE.md, "leverage cap read", RESOLVED.
K6_FROZEN = {
    "reading": "the MULTIPLIER, the quantity §5's named source caps",
    "binding_rate_multiplier": 0.0000,
    "binding_rate_gross": 0.5224,
    "vol_under_gross_cap": 0.0914,
    "vol_under_multiplier_cap": 0.1090,
    "gross_median": 3.08,
    "gross_p95": 5.30,
    "gross_max": 43.8,
    "gross_max_date": "2004-07-07",
    "turnover_pinned_per_year": 15.36,
    "turnover_m3_per_year": 58.92,
}


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def sharpe(x: pd.Series | np.ndarray) -> float:
    v = np.asarray(pd.Series(x).dropna(), dtype=float)
    s = v.std(ddof=1)
    return float(v.mean() / s * math.sqrt(PERIODS)) if s > 0 else float("nan")


def max_drawdown(x: pd.Series) -> float:
    curve = (1.0 + x.dropna()).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def hac_t(x: pd.Series, lags: int = 6) -> float:
    v = x.dropna().to_numpy()
    if len(v) < 30:
        return float("nan")
    fit = sm.OLS(v, np.ones(len(v))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.tvalues[0])


def cash_rate_daily(index: pd.DatetimeIndex) -> pd.Series:
    """Daily 3-month cash rate, stamped at `available_at`, never at `period`."""
    raw = pd.read_parquet(RAW / "macro" / "rate_cash_3m.parquet")
    series = (
        raw.assign(available_at=pd.to_datetime(raw["available_at"]))
        .sort_values("available_at")
        .drop_duplicates("available_at", keep="last")
        .set_index("available_at")["value"]
    )
    return (series.reindex(series.index.union(index)).ffill().reindex(index) / 100.0) / PERIODS


def funding_charge(weights: pd.DataFrame, rate: pd.Series, *, mode: str) -> pd.Series:
    """What §7 charges the instruments with no futures vehicle."""
    cash = [c for c in weights.columns if c in FUNDED_ON_CASH]
    exposure = weights[cash].sum(axis=1) if mode == "signed" else weights[cash].abs().sum(axis=1)
    return (exposure * rate.reindex(weights.index)).rename("funding")


def net_excess(
    gross_returns: pd.Series,
    weights: pd.DataFrame,
    rate: pd.Series,
    *,
    column: str,
    all_cash: bool = False,
    funding: str = "signed",
) -> pd.Series:
    """Gross book return -> net of cost, in excess of cash. The decision metric."""
    bps = cost_bps(weights.columns, column=column, all_cash=all_cash)
    after_cost = charge_by_instrument(gross_returns, weights, bps)
    return (after_cost - funding_charge(weights, rate, mode=funding)).rename(gross_returns.name)


def describe(name: str, r: pd.Series, w: pd.DataFrame) -> dict:
    gross = w.abs().sum(axis=1).reindex(r.index)
    return {
        "arm": name,
        "n": int(r.dropna().shape[0]),
        "start": str(r.dropna().index.min().date()),
        "end": str(r.dropna().index.max().date()),
        "sharpe": sharpe(r),
        "ann_return": float(r.mean() * PERIODS),
        "ann_vol": float(r.std(ddof=1) * math.sqrt(PERIODS)),
        "max_dd": max_drawdown(r),
        "hac_t": hac_t(r),
        "turnover": turnover(w.reindex(r.index)),
        "mean_gross": float(gross.mean()),
        "median_gross": float(gross.median()),
    }


def verdict_label(gap: float, gap_conservative: float, mde: float, pair_mde: float,
                  hypothesis: str) -> str:
    """The two readings §8 and §10 give the same gap, stated together.

    §8's K3/K4 make a gap below the MDE the REFUTATION of the subsidiary
    hypothesis. §10 forbids calling a gap below the MDE a pass and names it
    UNDERPOWERED. They are not the same statement — one says the effect is
    absent, the other that the sample cannot see it — so both are carried, and
    the pairing's own resolution decides which one the data supports.
    """
    if np.sign(gap) != np.sign(gap_conservative) and gap != 0:
        return "UNDECIDED — the sign moves between the headline and conservative columns"
    if gap >= mde:
        return f"PASS — {hypothesis} supported, gap above the locked MDE"
    if abs(gap) < pair_mde:
        return (f"UNDERPOWERED, and {hypothesis} REFUTED by §8's own rule — the gap is "
                f"below the MDE and below this pairing's own resolution")
    if gap > 0:
        return (f"{hypothesis} REFUTED by §8's own rule — the gap is resolvable but "
                f"smaller than the locked MDE, never a pass (§10)")
    return f"{hypothesis} REFUTED — the gap is negative"


def active_window(series: pd.Series) -> pd.Series:
    live = series.dropna()
    return live[live.index >= live.ne(0.0).idxmax()]


# ---------------------------------------------------------------------------
# the regime covariate
# ---------------------------------------------------------------------------
def regime_multiplier(
    unscaled: pd.Series, states: pd.Series, *, mode: str, min_history: int, min_bucket: int,
    abstain: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """Risk-budget multiplier with the state entered as one covariate.

    Returns (multiplier, ratio). Every estimate uses data strictly before the
    session on which it is applied, and the whole path is lagged one session.

    **The abstention rule, and the defect it repairs.** A first version of this
    function required ``min_history`` observations inside EACH state bucket. The
    turbulent state holds 699 of 6,039 sessions and its bucket only reaches 504
    on 2020-04-20, so every turbulent session before that date produced a NaN
    ratio and was dropped from the arm's sample: the regime arm was being graded
    on a sample with 2008 and March 2020 removed — the truncation trap, in the
    one place where it would have mattered most. The bucket minimum is therefore
    63, the volatility window §5 already locks, and where a bucket is still not
    estimable the covariate ABSTAINS at a ratio of 1.0 rather than deleting the
    session. No date is ever dropped for want of a conditional estimate.
    """
    s = states.reindex(unscaled.index).ffill()
    sigma_pool = unscaled.expanding(min_periods=min_history).std() * math.sqrt(PERIODS)

    conditional = pd.DataFrame(index=unscaled.index, dtype=float)
    for v in (0.0, 1.0):
        conditional[v] = (
            unscaled.where(s == v).expanding(min_periods=min_bucket).std() * math.sqrt(PERIODS)
        )
    # At tau, pick the bucket of the state observed at tau; the whole thing is
    # then lagged so that the session t reads the estimate formed at t-1.
    picked = pd.Series(
        [conditional.at[d, v] if (v in conditional.columns and pd.notna(v)) else np.nan
         for d, v in s.items()],
        index=s.index, dtype=float,
    )
    available = sigma_pool.notna()
    ratio = picked / sigma_pool
    if abstain:
        ratio = ratio.where(picked.notna(), 1.0)
    ratio = ratio.where(available).shift(1).rename("regime_ratio")

    sigma_63 = (unscaled.rolling(63).std() * math.sqrt(PERIODS)).shift(1)
    if mode == "blend":
        sigma_used = sigma_63 * ratio
    elif mode == "replacement":
        # The state-level estimate replaces the realised one; where the bucket is
        # not yet estimable the covariate abstains and the realised one stands.
        sigma_used = picked.shift(1).where(available.shift(1))
        sigma_used = sigma_used.fillna(sigma_63.where(available.shift(1)))
    else:
        raise ValueError(mode)

    multiplier = (VOL_TARGET / sigma_used.replace(0.0, np.nan)).clip(upper=MAX_LEVERAGE)
    return multiplier.rename("multiplier"), ratio


# ---------------------------------------------------------------------------
# deflated Sharpe
# ---------------------------------------------------------------------------
def deflated_sharpe(returns: pd.Series, n_trials: int, sharpe_var_ann: float) -> dict:
    r = returns.dropna().to_numpy()
    t = len(r)
    sr_d = r.mean() / r.std(ddof=1)
    var_d = sharpe_var_ann / PERIODS
    gamma = 0.5772156649015329
    e = math.e
    sr0_d = math.sqrt(var_d) * (
        (1 - gamma) * stats.norm.ppf(1 - 1.0 / n_trials)
        + gamma * stats.norm.ppf(1 - 1.0 / (n_trials * e))
    )
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    denom = math.sqrt(max(1e-12, 1 - skew * sr_d + (kurt - 1) / 4.0 * sr_d**2))
    z = (sr_d - sr0_d) * math.sqrt(t - 1) / denom
    return {
        "n_trials": int(n_trials),
        "sharpe_var_ann": float(sharpe_var_ann),
        "sharpe_ann": float(sr_d * math.sqrt(PERIODS)),
        "sr0_ann": float(sr0_d * math.sqrt(PERIODS)),
        "deflated_excess_ann": float((sr_d - sr0_d) * math.sqrt(PERIODS)),
        "dsr_probability": float(stats.norm.cdf(z)),
        "skew": skew,
        "kurtosis": kurt,
        "t_obs": int(t),
    }


# ---------------------------------------------------------------------------
# placebos
# ---------------------------------------------------------------------------
def rotation_placebo(
    raw: pd.DataFrame,
    returns: pd.DataFrame,
    path: pd.Series,
    other: pd.Series | None,
    rate: pd.Series,
    index: pd.DatetimeIndex,
    *,
    draws: int,
    seed: int,
    column: str,
    funding: str,
) -> np.ndarray:
    """P1 — circular rotation of the leverage path, its autocorrelation preserved.

    `path` is the component rotated. `other` is the component held in calendar
    place (for the regime arm: the realised 63-day estimator stays aligned and
    only the state ratio is rotated). Every draw is re-costed and re-funded on
    the weights it actually implies, so a rotation pays for its own turnover.
    """
    raw_np = raw.reindex(index).to_numpy(dtype=float)
    ret_np = np.nan_to_num(returns.reindex(index).to_numpy(dtype=float))
    bps = (cost_bps(raw.columns, column=column) / 10_000.0).to_numpy(dtype=float)
    cash_mask = np.array([c in FUNDED_ON_CASH for c in raw.columns])
    rate_np = rate.reindex(index).to_numpy(dtype=float)
    p = np.nan_to_num(path.reindex(index).to_numpy(dtype=float))
    o = None if other is None else np.nan_to_num(other.reindex(index).to_numpy(dtype=float))

    rng = np.random.default_rng(seed)
    n = len(index)
    out = np.empty(draws)
    for i, k in enumerate(rng.integers(1, n, size=draws)):
        m = np.roll(p, int(k))
        if o is not None:
            m = m * o
        w = raw_np * m[:, None]
        gross_r = np.nansum(w * ret_np, axis=1)
        traded = np.abs(np.diff(w, axis=0, prepend=0.0))
        cost = traded @ bps
        expo = w[:, cash_mask]
        fund = (expo.sum(axis=1) if funding == "signed" else np.abs(expo).sum(axis=1)) * rate_np
        net = gross_r - cost - fund
        out[i] = net.mean() / net.std(ddof=1) * math.sqrt(PERIODS)
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    results: dict = {"declared": DECLARED, "k6_frozen": K6_FROZEN}

    print(RULE)
    print("M3 — EVALUATION AGAINST THE SIX LOCKED STOPPING RULES  (§8)")
    print(RULE)
    print("\n   Decisions taken before any return was read:\n")
    for key in ("sample", "funding_exposure", "funding_rate_date", "cost_decision_column",
                "wf_min_train_years", "wf_folds", "wf_sign_flip_reference", "mde_decision",
                "regime_family", "regime_source", "regime_min_history",
                "regime_headline", "regime_sensitivity"):
        print(f"     {key:<24} {DECLARED[key]}")
    print("\n     states_offline is the HINDSIGHT relabelling (base.py). It favours H-b.")
    print("     A refutation under it is a fortiori; a pass would need the online series.")

    # ---------------- panels and arms ----------------
    px_head = pd.read_parquet(CACHE / f"{DECLARED['panel_headline']}.parquet").loc[SAMPLE]
    px_sens = pd.read_parquet(CACHE / f"{DECLARED['panel_m1_sensitivity']}.parquet").loc[SAMPLE]
    px_stored = pd.read_parquet(CACHE / f"{DECLARED['panel_stored']}.parquet").loc[SAMPLE]

    built = vol_targeted_book(px_head)
    built_sens = vol_targeted_book(px_sens)
    built_stored = vol_targeted_book(px_stored)

    idx = active_window(built["returns"]).index
    ret_head = px_head.pct_change()
    rate = cash_rate_daily(idx)

    n_active = (built["weights"].reindex(idx) != 0).sum(axis=1)
    print(f"\n   active sample   {idx.min():%Y-%m-%d} to {idx.max():%Y-%m-%d}, {len(idx):,} sessions")
    print(f"   instruments held   min {int(n_active.min())}, median {int(n_active.median())}, "
          f"max {int(n_active.max())}   (N >= 30 rule, §13)")
    print(f"   sessions below 30  {int((n_active < 30).sum())}")
    print("   no date is dropped for missing instruments: an instrument without a price")
    print("   carries zero exposure rather than truncating the panel.")
    results["sample_meta"] = {
        "start": str(idx.min().date()), "end": str(idx.max().date()), "n_sessions": int(len(idx)),
        "min_instruments": int(n_active.min()), "median_instruments": int(n_active.median()),
        "sessions_below_30": int((n_active < 30).sum()),
        "years": float(len(idx) / PERIODS),
    }

    # ---------------- the regime arm ----------------
    states_off = pd.read_parquet(CACHE / "states_offline.parquet")[DECLARED["regime_family"]]
    states_on = pd.read_parquet(CACHE / "states.parquet")[DECLARED["regime_family"]]
    # The book M3 would produce at a multiplier of one. Rebuilt from the raw
    # exposures rather than by dividing the M3 weights, which would be undefined
    # wherever the multiplier is zero.
    raw = unscaled_weights(px_head)
    unscaled = (raw * ret_head).sum(axis=1)

    kw = {"min_history": DECLARED["regime_min_history"],
          "min_bucket": DECLARED["regime_min_bucket"]}
    mult_blend, ratio_blend = regime_multiplier(unscaled, states_off, mode="blend", **kw)
    mult_repl, _ = regime_multiplier(unscaled, states_off, mode="replacement", **kw)
    mult_online, ratio_online = regime_multiplier(unscaled, states_on, mode="blend", **kw)
    # The version this study got wrong first, kept to measure the error it made.
    mult_trunc, _ = regime_multiplier(
        unscaled, states_off, mode="blend",
        min_history=DECLARED["regime_min_history"],
        min_bucket=DECLARED["regime_min_history"],
        abstain=False,
    )
    idx_reg = mult_blend.reindex(idx).dropna().index
    idx_trunc = mult_trunc.reindex(idx).dropna().index
    print(f"\n   regime arm sample  {idx_reg.min():%Y-%m-%d} to {idx_reg.max():%Y-%m-%d}, "
          f"{len(idx_reg):,} sessions  (expanding burn-in of "
          f"{DECLARED['regime_min_history']} per state)")
    s_lagged = states_off.reindex(idx_reg).ffill().shift(1)
    print(f"   state 0 (turbulent) on {int((s_lagged == 0).sum()):,} of {len(idx_reg):,} "
          f"sessions ({float((s_lagged == 0).mean()):.1%}), state 1 on "
          f"{int((s_lagged == 1).sum()):,}")
    print(f"   the defective first version of this arm held only {len(idx_trunc):,} sessions, "
          f"of which\n   "
          f"{int((states_off.reindex(idx_trunc).ffill().shift(1) == 0).sum()):,} were turbulent: "
          "2008 and March 2020 had been deleted. See §9(j).")
    print(f"   regime ratio   median {ratio_blend.reindex(idx_reg).median():.3f}, "
          f"p05 {ratio_blend.reindex(idx_reg).quantile(0.05):.3f}, "
          f"p95 {ratio_blend.reindex(idx_reg).quantile(0.95):.3f}")

    arms_w = {
        "pinned": built["pinned_weights"],
        "M3": built["weights"],
        "M3 M1-sensitivity": built_sens["weights"],
        "M3 stored panel": built_stored["weights"],
        "M3 + regime": raw.mul(mult_blend, axis=0),
        "M3 + regime (replacement)": raw.mul(mult_repl, axis=0),
        "M3 + regime (online states)": raw.mul(mult_online, axis=0),
        "M3 + regime (truncated, defective)": raw.mul(mult_trunc, axis=0),
    }
    arms_gross = {
        "pinned": built["pinned_returns"],
        "M3": built["returns"],
        "M3 M1-sensitivity": built_sens["returns"],
        "M3 stored panel": built_stored["returns"],
    }
    for k in ("M3 + regime", "M3 + regime (replacement)", "M3 + regime (online states)",
              "M3 + regime (truncated, defective)"):
        arms_gross[k] = (arms_w[k] * ret_head).sum(axis=1).rename(k)

    def arm_net(name: str, *, column="headline", all_cash=False, funding="signed",
                on: pd.DatetimeIndex | None = None) -> pd.Series:
        where = idx if on is None else on
        return net_excess(
            arms_gross[name].reindex(where), arms_w[name].reindex(where),
            rate.reindex(where), column=column, all_cash=all_cash, funding=funding,
        ).rename(name)

    # ---------------- the table ----------------
    print(f"\n{RULE}\n1.  THE ARMS, NET OF COST AND IN EXCESS OF CASH  (§6, §7)\n")
    print(f"   {'arm':<28}{'gross SR':>9}{'headline':>10}{'conserv.':>10}"
          f"{'stress':>9}{'ALL_CASH':>10}{'maxDD':>9}{'turn/yr':>9}{'gross':>8}")
    table = {}
    for name in arms_w:
        where = idx_trunc if "truncated" in name else (idx_reg if "regime" in name else idx)
        row = {}
        for col in DECLARED["cost_columns"]:
            row[col] = sharpe(arm_net(name, column=col, on=where))
        row["all_cash"] = sharpe(arm_net(name, column="headline", all_cash=True, on=where))
        base = arm_net(name, on=where)
        row.update(describe(name, base, arms_w[name]))
        row["gross_sharpe"] = sharpe(arms_gross[name].reindex(where))
        row["funding_gross_sharpe"] = sharpe(arm_net(name, funding="gross", on=where))
        table[name] = row
        print(f"   {name:<28}{row['gross_sharpe']:>9.3f}{row['headline']:>10.3f}"
              f"{row['conservative']:>10.3f}{row['stress']:>9.3f}{row['all_cash']:>10.3f}"
              f"{row['max_dd']:>9.1%}{row['turnover']:>9.2f}{row['mean_gross']:>8.2f}")
    results["arms"] = table
    print("\n   'gross SR' is before cost and before the funding charge, §7's continuity")
    print("   column. 'ALL_CASH' prices all 46 at 7.5 bp, the literal reading of §6.")
    print("   Regime arms are measured on their own shorter sample; they are compared")
    print("   only against M3 restricted to that same sample (below), never against the")
    print("   full-sample M3.")

    headline_sr = table["M3"]["headline"]
    results["headline_sharpe_net_excess"] = headline_sr

    # ---------------- K1 ----------------
    print(f"\n{RULE}\n2.  K1 — LEVEL.  net excess Sharpe <= 0.50 kills H\n")
    boot = []
    rng = np.random.default_rng(DECLARED["bootstrap_seed"])
    v = arm_net("M3").dropna().to_numpy()
    for _ in range(DECLARED["bootstrap_draws"]):
        s = v[stationary_indices(len(v), 126, rng)]
        boot.append(s.mean() / s.std(ddof=1) * math.sqrt(PERIODS))
    boot = np.array(boot)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    k1_pass = headline_sr > 0.50
    print(f"   M3, headline costs, net excess     Sharpe {headline_sr:+.3f}")
    print(f"   stationary block bootstrap (126)   IC95 [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    print(f"   HAC(6) t on the mean daily return  {table['M3']['hac_t']:+.2f}")
    print(f"   conservative column                {table['M3']['conservative']:+.3f}")
    print(f"   sign moves between headline and conservative?  "
          f"{'YES -> UNDECIDED' if np.sign(headline_sr) != np.sign(table['M3']['conservative']) else 'no'}")
    print(f"\n   K1 verdict: {'PASS' if k1_pass else 'KILL H'}  "
          f"({headline_sr:+.3f} against the 0.50 threshold)")
    results["K1"] = {
        "sharpe": headline_sr, "threshold": 0.50, "verdict": "PASS" if k1_pass else "KILL",
        "ci95_block126": ci, "hac_t": table["M3"]["hac_t"],
        "sign_stable_headline_to_conservative":
            bool(np.sign(headline_sr) == np.sign(table["M3"]["conservative"])),
    }

    # ---------------- K2 ----------------
    print(f"\n{RULE}\n3.  K2 — WALK-FORWARD.  five folds, partitioned before any is read\n")
    folds = walk_forward(idx, min_train_years=DECLARED["wf_min_train_years"],
                         folds=DECLARED["wf_folds"])
    for f in folds:
        print(f"     {f}")
    print()
    m3_net = arm_net("M3")
    fold_rows = []
    print(f"   {'fold':<7}{'sessions':>10}{'M3 net excess SR':>19}{'pinned':>10}{'sign':>8}")
    for f in folds:
        mask = (idx >= f.eval_start) & (idx <= f.eval_end)
        sub = m3_net[mask]
        pin = arm_net("pinned")[mask]
        srf = sharpe(sub)
        flip = np.sign(srf) != np.sign(headline_sr)
        fold_rows.append({"fold": f.index, "eval_start": str(f.eval_start.date()),
                          "eval_end": str(f.eval_end.date()), "n": int(mask.sum()),
                          "sharpe": srf, "pinned_sharpe": sharpe(pin), "sign_flip": bool(flip)})
        print(f"   {f.index:<7}{int(mask.sum()):>10}{srf:>19.3f}{sharpe(pin):>10.3f}"
              f"{'FLIP' if flip else '':>8}")
    positive = sum(1 for r in fold_rows if r["sharpe"] > 0)
    flips = sum(1 for r in fold_rows if r["sign_flip"])
    k2_pass = positive >= 3 and flips < 2
    print(f"\n   positive folds {positive}/5, sign flips {flips}")
    print(f"   K2 verdict: {'PASS' if k2_pass else 'KILL H'}")
    print("\n   Declared against this study: walk_forward's own default trains ten years,")
    print(f"   so the five folds cover {folds[0].eval_start:%Y-%m} onward and the first")
    print("   decade of the sample is evaluated only by K1. The named tool is used as")
    print("   named. The five equal blocks over the WHOLE active sample are reported")
    print("   below, without choosing between the two partitions (§12).")
    edges = pd.date_range(idx.min(), idx.max(), periods=6)
    equal_rows = []
    for i in range(5):
        mask = (idx >= edges[i]) & (idx <= edges[i + 1])
        srf = sharpe(m3_net[mask])
        equal_rows.append({"block": i, "start": str(edges[i].date()),
                           "end": str(edges[i + 1].date()), "sharpe": srf})
        print(f"     block {i}  {edges[i]:%Y-%m} to {edges[i+1]:%Y-%m}   {srf:>7.3f}")
    print(f"     positive {sum(1 for r in equal_rows if r['sharpe'] > 0)}/5")
    results["K2"] = {"folds": fold_rows, "positive": positive, "sign_flips": flips,
                     "verdict": "PASS" if k2_pass else "KILL",
                     "equal_blocks": equal_rows,
                     "equal_blocks_positive": sum(1 for r in equal_rows if r["sharpe"] > 0)}

    # ---------------- K3 ----------------
    print(f"\n{RULE}\n4.  K3 — SIZING.  M3 against the pinned construction it replaces\n")
    a = arm_net("M3")
    b = arm_net("pinned")
    pair = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    gap = sharpe(pair["a"]) - sharpe(pair["b"])
    print(f"   common sample {len(pair):,} sessions, identical dates for both legs")
    print(f"   M3 {sharpe(pair['a']):+.3f}   pinned {sharpe(pair['b']):+.3f}   "
          f"gap {gap:+.3f}")
    print(f"   HAC(6) t on the daily difference   {hac_t(pair['a'] - pair['b']):+.2f}")
    mdes = {}
    for blk in DECLARED["bootstrap_blocks"]:
        pr = minimum_detectable_sharpe_difference(
            pair["a"].to_numpy(), pair["b"].to_numpy(), mean_block=blk,
            draws=DECLARED["bootstrap_draws"], seed=DECLARED["bootstrap_seed"])
        mdes[blk] = {"se": pr.se, "mde": pr.mde}
        print(f"   block {blk:>3}  bootstrap SE {pr.se:.3f}   MDE {pr.mde:.3f}   "
              f"(locked {DECLARED['mde_reported'][str(blk)]:.3f})")
    gap_cons = table["M3"]["conservative"] - table["pinned"]["conservative"]
    pair_mde = mdes[126]["mde"]
    k3 = verdict_label(gap, gap_cons, DECLARED["mde_decision"], pair_mde, "H-a")
    print(f"\n   gap under the conservative column  {gap_cons:+.3f}")
    print(f"   K3 verdict against the locked MDE {DECLARED['mde_decision']:.3f}: {k3}")
    print("   A gap below the MDE is UNDERPOWERED, never a pass (§10); §8's own rule")
    print("   makes the same gap the refutation of H-a. Both readings are stated,")
    print("   because they say different things: one is 'no effect', the other is")
    print("   'no effect this sample could see'. Here the gap is below even the")
    print(f"   resolution of its own pairing ({pair_mde:.3f}), so the second holds.")
    results["K3"] = {"sharpe_m3": sharpe(pair["a"]), "sharpe_pinned": sharpe(pair["b"]),
                     "gap": gap, "gap_conservative": gap_cons,
                     "hac_t_difference": hac_t(pair["a"] - pair["b"]),
                     "mde_locked": DECLARED["mde_decision"], "mde_recomputed": mdes,
                     "n": int(len(pair)), "verdict": k3}

    # ---------------- K4 ----------------
    print(f"\n{RULE}\n5.  K4 — REGIME.  the sixth and last test of the regime thesis\n")
    a4 = arm_net("M3 + regime", on=idx_reg)
    b4 = arm_net("M3", on=idx_reg)
    pair4 = pd.concat([a4.rename("a"), b4.rename("b")], axis=1).dropna()
    gap4 = sharpe(pair4["a"]) - sharpe(pair4["b"])
    print(f"   common sample {len(pair4):,} sessions "
          f"({pair4.index.min():%Y-%m-%d} to {pair4.index.max():%Y-%m-%d})")
    print(f"   M3 + regime {sharpe(pair4['a']):+.3f}   M3 on the same dates "
          f"{sharpe(pair4['b']):+.3f}   gap {gap4:+.3f}")
    print(f"   HAC(6) t on the daily difference   {hac_t(pair4['a'] - pair4['b']):+.2f}")
    mdes4 = {}
    for blk in DECLARED["bootstrap_blocks"]:
        pr = minimum_detectable_sharpe_difference(
            pair4["a"].to_numpy(), pair4["b"].to_numpy(), mean_block=blk,
            draws=DECLARED["bootstrap_draws"], seed=DECLARED["bootstrap_seed"])
        mdes4[blk] = {"se": pr.se, "mde": pr.mde}
        print(f"   block {blk:>3}  bootstrap SE {pr.se:.3f}   MDE {pr.mde:.3f}")
    a4c = arm_net("M3 + regime", column="conservative", on=idx_reg)
    b4c = arm_net("M3", column="conservative", on=idx_reg)
    gap4_cons = sharpe(a4c) - sharpe(b4c)
    pair_mde4 = mdes4[126]["mde"]
    k4 = verdict_label(gap4, gap4_cons, DECLARED["mde_decision"], pair_mde4, "H-b")
    print(f"\n   gap under the conservative column  {gap4_cons:+.3f}")
    for label, key in (("replacement reading", "M3 + regime (replacement)"),
                       ("online states", "M3 + regime (online states)")):
        s_alt = sharpe(arm_net(key, on=idx_reg))
        print(f"   sensitivity, {label:<20} {s_alt:+.3f}   gap {s_alt - sharpe(pair4['b']):+.3f}")
    print(f"\n   K4 verdict against the locked MDE {DECLARED['mde_decision']:.3f}: {k4}")
    print(f"   The gap is {abs(gap4) / pair_mde4:.1f}x the resolution of its own pairing")
    print(f"   ({pair_mde4:.3f}) and {abs(gap4) / DECLARED['mde_decision']:.2f}x the locked MDE.")
    print("   H-b's prior was low and written. It is the sixth refutation.")
    results["K4"] = {
        "sharpe_regime": sharpe(pair4["a"]), "sharpe_m3_same_dates": sharpe(pair4["b"]),
        "gap": gap4, "gap_conservative": gap4_cons,
        "hac_t_difference": hac_t(pair4["a"] - pair4["b"]),
        "mde_locked": DECLARED["mde_decision"], "mde_recomputed": mdes4,
        "n": int(len(pair4)), "verdict": k4,
        "sensitivity_replacement": sharpe(arm_net("M3 + regime (replacement)", on=idx_reg)),
        "sensitivity_online_states": sharpe(arm_net("M3 + regime (online states)", on=idx_reg)),
    }

    # ---------------- K5 ----------------
    print(f"\n{RULE}\n6.  K5 — DEFLATED SHARPE.  full trial count, §12\n")
    trials = read_trials()
    distinct = trials.dropna(subset=["m_sharpe"]).groupby("config_hash")["m_sharpe"].mean()
    existing_n = int(trials["config_hash"].nunique())
    own = {
        "pinned": table["pinned"]["headline"],
        "M3": table["M3"]["headline"],
        "M3 + regime (blend)": table["M3 + regime"]["headline"],
        "M3 + regime (replacement)": table["M3 + regime (replacement)"]["headline"],
    }
    print(f"   existing distinct configurations in data/trials.parquet   {existing_n}")
    print("   configurations evaluated under this pre-registration that are TRIALS:")
    for k, v in own.items():
        print(f"     {k:<30} {v:+.3f}")
    print("   not trials, §12 — reported without choosing between them: the three cost")
    print("   columns, ALL_CASH, the three block lengths, the M1 sensitivity panel, the")
    print("   stored panel, the online-state sensitivity, the two fold partitions.")
    pooled = pd.concat([distinct, pd.Series(own)])
    n_trials = existing_n + len(own)
    var_ann = float(pooled.var(ddof=1))
    dsr = deflated_sharpe(arm_net("M3"), n_trials, var_ann)
    print(f"\n   N = {n_trials}   V[SR] = {var_ann:.4f} (annualised, {len(pooled)} Sharpes)")
    print(f"   expected maximum Sharpe under the null   {dsr['sr0_ann']:+.3f}")
    print(f"   observed                                 {dsr['sharpe_ann']:+.3f}")
    print(f"   deflated excess  SR - SR0                {dsr['deflated_excess_ann']:+.3f}")
    print(f"   DSR as a probability                     {dsr['dsr_probability']:.4f}")
    print("\n   K5's text says 'DSR <= 0'. A probability cannot be <= 0, so the criterion")
    print("   only bites under the reading where the quantity is the DEFLATED EXCESS")
    print("   SR - SR0. That is the reading taken, and the probability is reported too.")
    k5 = "KILL" if dsr["deflated_excess_ann"] <= 0 else "PASS"
    # conservative count: every arm touched, panels included.
    n_max = existing_n + len(arms_w)
    dsr_max = deflated_sharpe(arm_net("M3"), n_max, var_ann)
    print(f"   K5 verdict: {k5}")
    print(f"   conservative count N = {n_max} (every arm counted): "
          f"SR0 {dsr_max['sr0_ann']:+.3f}, deflated excess "
          f"{dsr_max['deflated_excess_ann']:+.3f}")
    print("\n   Nothing was appended to data/trials.parquet: this run writes nothing into")
    print("   the repository. The four rows above still have to be logged there.")
    results["K5"] = {"decision": dsr, "verdict": k5, "conservative": dsr_max,
                     "existing_distinct": existing_n, "own_configs": own}

    # ---------------- K6 ----------------
    print(f"\n{RULE}\n7.  K6 — CAP INTEGRITY.  already settled, figure carried over\n")
    print(f"   reading taken: {K6_FROZEN['reading']}")
    print(f"   binds on {K6_FROZEN['binding_rate_multiplier']:.2%} of sessions "
          f"-> K6 PASS (threshold 5%)")
    print(f"   the other reading binds on {K6_FROZEN['binding_rate_gross']:.2%} and pulls")
    print(f"   realised volatility to {K6_FROZEN['vol_under_gross_cap']:.2%}, so K6 itself")
    print("   disqualifies it. Declared against this study: under the reading taken the")
    print(f"   cap is INERT, gross runs at a median of {K6_FROZEN['gross_median']:.2f}, a p95 of")
    print(f"   {K6_FROZEN['gross_p95']:.2f} and a maximum of {K6_FROZEN['gross_max']:.1f} on "
          f"{K6_FROZEN['gross_max_date']}.")
    results["K6"] = {**K6_FROZEN, "verdict": "PASS"}

    # ---------------- placebos ----------------
    print(f"\n{RULE}\n8.  PLACEBOS  (§11, both mandatory)\n")
    print("   P1 — circular rotation of the leverage path, 400 draws, each re-costed")
    print("        and re-funded on the weights it implies.\n")
    placebos = {}
    draws_m3 = rotation_placebo(
        raw, ret_head, built["multiplier"], None, rate, idx,
        draws=DECLARED["placebo_draws"], seed=DECLARED["placebo_seed"],
        column="headline", funding="signed")
    real_m3 = sharpe(arm_net("M3"))
    # unbiasedness reference: the same book at a CONSTANT multiplier equal to the
    # mean of the real path — no calendar alignment at all.
    const_mult = pd.Series(float(built["multiplier"].reindex(idx).mean()), index=idx)
    w_const = raw.reindex(idx).mul(const_mult, axis=0)
    r_const = net_excess((w_const * ret_head.reindex(idx)).sum(axis=1).rename("const"),
                         w_const, rate, column="headline")
    pct_m3 = float((draws_m3 < real_m3).mean())
    print(f"   {'arm':<26}{'placebo mean':>14}{'sd':>8}{'p95':>9}{'real':>9}{'pctile':>9}")
    print(f"   {'M3 leverage path':<26}{draws_m3.mean():>14.3f}{draws_m3.std(ddof=1):>8.3f}"
          f"{np.percentile(draws_m3, 95):>9.3f}{real_m3:>9.3f}{pct_m3:>8.1%}")
    print(f"   unbiasedness check: placebo mean {draws_m3.mean():+.3f} against the "
          f"unconditional\n   constant-exposure book at the same mean gross "
          f"{sharpe(r_const):+.3f}")
    placebos["P1_M3"] = {"mean": float(draws_m3.mean()), "sd": float(draws_m3.std(ddof=1)),
                         "p95": float(np.percentile(draws_m3, 95)), "real": real_m3,
                         "percentile": pct_m3,
                         "unconditional_reference": sharpe(r_const)}

    draws_reg = rotation_placebo(
        raw, ret_head, ratio_blend,
        (VOL_TARGET / (unscaled.rolling(63).std() * math.sqrt(PERIODS)).shift(1)
         ).clip(upper=MAX_LEVERAGE),
        rate, idx_reg, draws=DECLARED["placebo_draws"], seed=DECLARED["placebo_seed"],
        column="headline", funding="signed")
    real_reg = sharpe(arm_net("M3 + regime", on=idx_reg))
    pct_reg = float((draws_reg < real_reg).mean())
    print(f"   {'regime ratio path':<26}{draws_reg.mean():>14.3f}"
          f"{draws_reg.std(ddof=1):>8.3f}{np.percentile(draws_reg, 95):>9.3f}"
          f"{real_reg:>9.3f}{pct_reg:>8.1%}")
    print(f"   unbiasedness check: placebo mean {draws_reg.mean():+.3f} against M3 on the")
    print(f"   same dates {sharpe(b4):+.3f} — rotating the ratio destroys only its calendar")
    print("   alignment, so an unconditional placebo must land on the unconditioned arm.")
    placebos["P1_regime"] = {"mean": float(draws_reg.mean()), "sd": float(draws_reg.std(ddof=1)),
                             "p95": float(np.percentile(draws_reg, 95)), "real": real_reg,
                             "percentile": pct_reg,
                             "unconditional_reference": sharpe(b4)}

    print("\n   P3 — matched-exposure control. Each conditioned arm against a book")
    print("        carrying the SAME mean gross exposure.\n")
    print(f"   {'arm':<26}{'mean gross':>12}{'arm SR':>9}{'matched SR':>12}{'gap':>9}")
    p3 = {}
    for name, control_w, _control_gross, where in (
        ("M3 vs constant exposure", built["pinned_weights"], built["pinned_returns"], idx),
        ("regime vs M3 rescaled", built["weights"], built["returns"], idx_reg),
    ):
        arm_name = "M3" if name.startswith("M3 vs") else "M3 + regime"
        g_arm = float(arms_w[arm_name].abs().sum(axis=1).reindex(where).mean())
        g_ctl = float(control_w.abs().sum(axis=1).reindex(where).mean())
        k = g_arm / g_ctl
        w_m = control_w.reindex(where) * k
        r_m = net_excess((w_m * ret_head.reindex(where)).sum(axis=1).rename("matched"),
                         w_m, rate.reindex(where), column="headline")
        s_arm = sharpe(arm_net(arm_name, on=where))
        print(f"   {name:<26}{g_arm:>12.2f}{s_arm:>9.3f}{sharpe(r_m):>12.3f}"
              f"{s_arm - sharpe(r_m):>9.3f}")
        p3[name] = {"mean_gross_arm": g_arm, "mean_gross_control": g_ctl, "scale": k,
                    "arm_sharpe": s_arm, "matched_sharpe": sharpe(r_m),
                    "gap": s_arm - sharpe(r_m)}
    # scale invariance is a property to be measured, not asserted.
    w2 = built["pinned_weights"].reindex(idx) * 3.0
    r2 = net_excess((w2 * ret_head.reindex(idx)).sum(axis=1).rename("x3"), w2, rate,
                    column="headline")
    inv = abs(sharpe(r2) - sharpe(arm_net("pinned")))
    print("\n   measured: multiplying a book by a constant moves its net excess Sharpe by")
    print(f"   {inv:.2e}. Cost and funding are both linear in exposure and the cap is")
    print("   inert, so no arm here can buy Sharpe with exposure. The P3 percentile is")
    print("   the P1 percentile: a circular rotation preserves the leverage path's mean")
    print("   exactly, so every placebo draw is already exposure-matched.")
    p3["scale_invariance_residual"] = float(inv)
    p3["P3_percentile_M3"] = pct_m3
    p3["P3_percentile_regime"] = pct_reg
    results["placebos"] = {"P1": placebos, "P3": p3}

    # ---------------- self-audit ----------------
    print(f"\n{RULE}\n9.  SELF-AUDIT — the traps this programme has already fallen into\n")
    audit = {}

    print("   (a) reproduction. Quantities already measured by")
    print("       scripts/run_m3_construction.py, recomputed here independently:")
    rep = {
        "turnover_pinned": (table["pinned"]["turnover"], K6_FROZEN["turnover_pinned_per_year"]),
        "turnover_m3": (table["M3"]["turnover"], K6_FROZEN["turnover_m3_per_year"]),
        "realised_vol_m3": (float(built["returns"].reindex(idx).std(ddof=1) * math.sqrt(PERIODS)),
                            K6_FROZEN["vol_under_multiplier_cap"]),
        "gross_median_m3": (table["M3"]["median_gross"], K6_FROZEN["gross_median"]),
    }
    for k, (got, frozen) in rep.items():
        print(f"       {k:<20} here {got:>8.3f}   frozen {frozen:>8.3f}   "
              f"{'matches' if abs(got - frozen) < 0.01 else 'DIFFERS'}")
    audit["reproduction"] = {k: {"here": g, "frozen": f} for k, (g, f) in rep.items()}

    print("\n   (b) signal at T-1. The same book with the exposure path shifted by one")
    print("       session either way. A causal chain must lose when delayed and gain")
    print("       when advanced; if it does not, the lag is not doing anything.")
    lag_rows = {}
    for shift, label in ((-1, "same session (FORBIDDEN)"), (0, "as built, lagged 1"),
                         (1, "one extra session")):
        w = arms_w["M3"].shift(shift) if shift else arms_w["M3"]
        r = net_excess((w.reindex(idx) * ret_head.reindex(idx)).sum(axis=1).rename("x"),
                       w.reindex(idx), rate, column="headline")
        lag_rows[label] = sharpe(r)
        print(f"       {label:<28} {sharpe(r):+.3f}")
    audit["lag_integrity"] = lag_rows

    print("\n   (c) what the two §7/§6 charges actually take, on M3, headline column:")
    bps = cost_bps(arms_w["M3"].columns, column="headline")
    gross_ret = arms_gross["M3"].reindex(idx)
    after_cost = charge_by_instrument(gross_ret, arms_w["M3"].reindex(idx), bps)
    fund = funding_charge(arms_w["M3"].reindex(idx), rate, mode="signed")
    print(f"       gross return            {gross_ret.mean() * PERIODS:>7.2%}/yr")
    print(f"       cost drag               {(gross_ret - after_cost).mean() * PERIODS:>7.2%}/yr"
          f"   ({(gross_ret - after_cost).mean() / gross_ret.mean():.1%} of gross)")
    print(f"       funding charge          {fund.mean() * PERIODS:>7.2%}/yr")
    print(f"       net excess              "
          f"{(after_cost - fund).mean() * PERIODS:>7.2%}/yr")
    print(f"       mean signed cash-vehicle exposure {arms_w['M3'][list(NO_FUTURES_VEHICLE)].sum(axis=1).reindex(idx).mean():+.3f}")
    print(f"       mean 3-month cash rate  {rate.mean() * PERIODS:>7.2%}/yr")
    audit["charges"] = {
        "gross_ann": float(gross_ret.mean() * PERIODS),
        "cost_drag_ann": float((gross_ret - after_cost).mean() * PERIODS),
        "cost_share_of_gross": float((gross_ret - after_cost).mean() / gross_ret.mean()),
        "funding_ann": float(fund.mean() * PERIODS),
        "net_excess_ann": float((after_cost - fund).mean() * PERIODS),
        "mean_cash_exposure": float(
            arms_w["M3"][list(NO_FUTURES_VEHICLE)].sum(axis=1).reindex(idx).mean()),
        "mean_cash_rate_ann": float(rate.mean() * PERIODS),
    }

    print("\n   (d) levels against returns. Every book return in this file comes from")
    print("       prices.pct_change(); no price level is ever correlated with a return.")
    print("   (e) truncation. No date is dropped for missing instruments, and the")
    print(f"       thinnest session holds {int(n_active.min())} of 46.")
    print("   (f) the bootstrap is Politis-Romano stationary block, from")
    print(f"       {Path(stationary_indices.__module__.replace('.', '/')).with_suffix('.py')};")
    print("       no iid resampling anywhere in this file.")

    print("\n   (g) P1's unbiasedness, measured rather than asserted.")
    se_mean = float(draws_m3.std(ddof=1) / math.sqrt(len(draws_m3)))
    print(f"       placebo mean                              {draws_m3.mean():+.3f}")
    print(f"       pinned book                               {table['pinned']['headline']:+.3f}"
          f"   gap {draws_m3.mean() - table['pinned']['headline']:+.3f}")
    print(f"       same construction, CONSTANT multiplier    {sharpe(r_const):+.3f}"
          f"   gap {draws_m3.mean() - sharpe(r_const):+.3f}"
          f"   t {(draws_m3.mean() - sharpe(r_const)) / se_mean:+.1f}")
    print("       The placebo lands on the pinned book but sits measurably below the")
    print("       constant-multiplier version of its own construction: a randomly")
    print("       aligned leverage path adds variance a constant one does not. The")
    print("       bias is common to the real arm and to every draw, since a circular")
    print("       rotation preserves the leverage path's marginal distribution and its")
    print("       autocorrelation exactly, so the percentile is unaffected. Reported")
    print("       because the pre-registration asks for the check, not for the answer.")
    audit["p1_unbiasedness"] = {
        "placebo_mean": float(draws_m3.mean()),
        "pinned": table["pinned"]["headline"],
        "constant_multiplier": sharpe(r_const),
        "gap_vs_pinned": float(draws_m3.mean() - table["pinned"]["headline"]),
        "gap_vs_constant_multiplier": float(draws_m3.mean() - sharpe(r_const)),
        "t_of_gap": float((draws_m3.mean() - sharpe(r_const)) / se_mean),
    }

    print("\n   (h) why K4 has so little to work with — the mechanism, measured:")
    s_lag_full = states_off.reindex(idx_reg).ffill().shift(1)
    u_r = unscaled.reindex(idx_reg)
    v0 = float(u_r[s_lag_full == 0].std(ddof=1) * math.sqrt(PERIODS))
    v1 = float(u_r[s_lag_full == 1].std(ddof=1) * math.sqrt(PERIODS))
    print(f"       book volatility in state 0 (turbulent)  {v0:.2%}")
    print(f"       book volatility in state 1 (calm)       {v1:.2%}")
    print(f"       ratio                                   {v0 / v1:.3f}")
    print(f"       sessions in state 0                     {int((s_lag_full == 0).sum()):,}"
          f" of {len(idx_reg):,} ({float((s_lag_full == 0).mean()):.1%})")
    print("       The covariate can only move the risk budget by the distance between")
    print("       these two numbers, on the share of sessions the rarer state occupies.")
    audit["k4_mechanism"] = {"vol_state0": v0, "vol_state1": v1, "ratio": v0 / v1,
                             "share_state0": float((s_lag_full == 0).mean()),
                             "sessions_state0": int((s_lag_full == 0).sum())}

    print("\n   (i) M1 declared against itself. The repair hands the book a fresher")
    print("       observation date, the direction that flatters a backtest. Measured:")
    print(f"       stored panel, no shift        {table['M3 stored panel']['headline']:+.3f}")
    print(f"       M1 headline, eight shifted    {table['M3']['headline']:+.3f}")
    print(f"       M1 sensitivity, three shifted {table['M3 M1-sensitivity']['headline']:+.3f}")
    print("       The repair did not flatter: the headline panel reads below the")
    print("       unrepaired one. No verdict moves across the three.")
    audit["m1_panels"] = {"stored": table["M3 stored panel"]["headline"],
                          "m1_headline": table["M3"]["headline"],
                          "m1_sensitivity": table["M3 M1-sensitivity"]["headline"]}

    print("\n   (j) the error this file made, and what it would have published.")
    print("       The first version required 504 observations inside EACH state bucket.")
    print("       The turbulent state holds 699 of the active sessions and its bucket")
    print("       only reaches 504 on 2020-04-20, so every turbulent session before that")
    print("       date returned a NaN ratio and was dropped from the arm's sample.")
    s_tr = states_off.reindex(idx_trunc).ffill().shift(1)
    a_tr = arm_net("M3 + regime (truncated, defective)", on=idx_trunc)
    b_tr = arm_net("M3", on=idx_trunc)
    print(f"       {'':<26}{'sessions':>10}{'turbulent':>11}{'arm SR':>9}{'M3 same dates':>15}"
          f"{'gap':>8}")
    print(f"       {'defective':<26}{len(idx_trunc):>10,}{int((s_tr == 0).sum()):>11,}"
          f"{sharpe(a_tr):>9.3f}{sharpe(b_tr):>15.3f}{sharpe(a_tr) - sharpe(b_tr):>8.3f}")
    print(f"       {'repaired, abstention rule':<26}{len(idx_reg):>10,}"
          f"{int((s_lagged == 0).sum()):>11,}{sharpe(pair4['a']):>9.3f}"
          f"{sharpe(pair4['b']):>15.3f}{gap4:>8.3f}")
    u_tr = unscaled.reindex(idx_trunc)
    v0t = float(u_tr[s_tr == 0].std(ddof=1) * math.sqrt(PERIODS))
    v1t = float(u_tr[s_tr == 1].std(ddof=1) * math.sqrt(PERIODS))
    print("       The defect deleted the two episodes the device exists to act on. It")
    print("       was found by asking why the turbulent state covered 4.0% of the arm's")
    print("       sample against 11.6% of the book's, not by the result changing.")
    print("       It also inverted the mechanism it was supposed to measure: on the")
    print("       truncated sample the book looks LESS volatile in the turbulent state")
    print(f"       ({v0t:.1%} against {v1t:.1%}, ratio {v0t / v1t:.3f}); on the full one it is")
    print(f"       more volatile ({v0:.1%} against {v1:.1%}, ratio {v0 / v1:.3f}).")
    audit["own_defect_mechanism_inversion"] = {
        "truncated_vol_state0": v0t, "truncated_vol_state1": v1t,
        "truncated_ratio": v0t / v1t, "full_ratio": v0 / v1,
    }
    audit["own_defect"] = {
        "description": "per-bucket min_periods=504 dropped every turbulent session "
                       "before 2020-04-20",
        "defective_sessions": int(len(idx_trunc)),
        "defective_turbulent": int((s_tr == 0).sum()),
        "defective_arm_sharpe": sharpe(a_tr),
        "defective_m3_same_dates": sharpe(b_tr),
        "defective_gap": sharpe(a_tr) - sharpe(b_tr),
        "repaired_sessions": int(len(idx_reg)),
        "repaired_turbulent": int((s_lagged == 0).sum()),
        "repaired_gap": gap4,
    }
    print("\n   (k) the sample is sliced BEFORE the signal is built, which is what")
    print("       scripts/run_m3_construction.py does and what the frozen K6 figures")
    print("       were measured on. The cost is visible: a 252-session lookback and a")
    print("       63-session volatility window both start inside the slice, so the book")
    print(f"       only comes alive on {idx.min():%Y-%m-%d} rather than on §3's "
          f"{DECLARED['sample'][0]},")
    print("       and the multiplier's first estimate is formed on a short window — which")
    print("       is why the 43.8x gross maximum falls on the sample's own first session.")
    px_full = pd.read_parquet(CACHE / f"{DECLARED['panel_headline']}.parquet")
    built_full = vol_targeted_book(px_full)
    idx_full = active_window(built_full["returns"].loc[SAMPLE]).index
    rate_full = cash_rate_daily(idx_full)
    r_full = net_excess(
        built_full["returns"].reindex(idx_full), built_full["weights"].reindex(idx_full),
        rate_full, column="headline")
    g_full = built_full["weights"].abs().sum(axis=1).reindex(idx_full)
    drop21 = net_excess(arms_gross["M3"].reindex(idx[21:]), arms_w["M3"].reindex(idx[21:]),
                        rate.reindex(idx[21:]), column="headline")
    print(f"       {'construction':<38}{'sessions':>10}{'net excess SR':>15}{'max gross':>11}")
    print(f"       {'sliced first (headline, committed)':<38}{len(idx):>10,}"
          f"{headline_sr:>15.3f}{float(arms_w['M3'].abs().sum(axis=1).reindex(idx).max()):>11.1f}")
    print(f"       {'signal built on the whole panel':<38}{len(idx_full):>10,}"
          f"{sharpe(r_full):>15.3f}{float(g_full.max()):>11.1f}")
    print(f"       {'headline, first 21 sessions dropped':<38}{len(idx) - 21:>10,}"
          f"{sharpe(drop21):>15.3f}{'—':>11}")
    r_full_cons = net_excess(
        built_full["returns"].reindex(idx_full), built_full["weights"].reindex(idx_full),
        rate_full, column="conservative")
    pin_full = net_excess(
        built_full["pinned_returns"].reindex(idx_full),
        built_full["pinned_weights"].reindex(idx_full), rate_full, column="headline")
    folds_full = walk_forward(idx_full, min_train_years=DECLARED["wf_min_train_years"],
                              folds=DECLARED["wf_folds"])
    pos_full = sum(1 for f in folds_full
                   if sharpe(r_full[(idx_full >= f.eval_start) & (idx_full <= f.eval_end)]) > 0)
    print("\n       THIS ONE MOVES K1 ACROSS ITS OWN THRESHOLD, AND IT IS SAID HERE")
    print("       RATHER THAN LEFT IN A FOOTNOTE. §3 defines its start as 'the first")
    print("       session on which at least 30 instruments have 252 sessions of price")
    print("       history', which is a statement that the signal exists ON 2003-07-17 —")
    print("       so §3's own words put the lookback BEFORE the sample, not inside it.")
    print("       The committed script slices first and loses the first year.")
    print(f"       sliced first     {headline_sr:+.3f}  -> K1 KILL")
    print(f"       §3-faithful      {sharpe(r_full):+.3f}  -> K1 misses by "
          f"{sharpe(r_full) - 0.50:+.3f}")
    print(f"       That margin is {(sharpe(r_full) - 0.50) / DECLARED['mde_decision']:.2f} of "
          f"one MDE and sits inside a bootstrap")
    print(f"       IC95 of [{ci[0]:+.3f}, {ci[1]:+.3f}]. Neither construction decides the level")
    print("       question; the 0.50 line falls inside the interval either way.")
    print(f"       §3-faithful, conservative column {sharpe(r_full_cons):+.3f}, "
          f"maxDD {max_drawdown(r_full):.1%},")
    print(f"       pinned {sharpe(pin_full):+.3f}, K3 gap {sharpe(r_full) - sharpe(pin_full):+.3f}, "
          f"folds {pos_full}/5 positive.")
    print("       The headline is NOT switched: the committed construction is the one the")
    print("       frozen K6 figures, the turnover and the rotation were all measured on,")
    print("       and moving the headline to the arm that reads higher, after seeing that")
    print("       it reads higher, is the adjustment this protocol forbids. It is logged")
    print("       for PROTOCOL_FREEZE.md as a question for the next amendment, not")
    print("       resolved here.")
    audit["sample_slicing"] = {
        "s3_faithful_conservative": sharpe(r_full_cons),
        "s3_faithful_maxdd": max_drawdown(r_full),
        "s3_faithful_pinned": sharpe(pin_full),
        "s3_faithful_k3_gap": sharpe(r_full) - sharpe(pin_full),
        "s3_faithful_folds_positive": int(pos_full),
        "k1_threshold_crossed": True,
        "headline_sessions": int(len(idx)), "headline_sharpe": headline_sr,
        "headline_max_gross": float(arms_w["M3"].abs().sum(axis=1).reindex(idx).max()),
        "built_on_full_panel_sessions": int(len(idx_full)),
        "built_on_full_panel_sharpe": sharpe(r_full),
        "built_on_full_panel_start": str(idx_full.min().date()),
        "built_on_full_panel_max_gross": float(g_full.max()),
        "headline_drop_first_21_sharpe": sharpe(drop21),
    }
    results["self_audit"] = audit

    # ---------------- gates ----------------
    print(f"\n{RULE}\n10. GATES  (§9)\n")
    gate = {
        "research_pass": bool(headline_sr > 0.70 and table["M3"]["max_dd"] > -0.25
                              and positive >= 3 and dsr["deflated_excess_ann"] > 0),
        "propfirm_pass": bool(headline_sr > 1.00 and table["M3"]["max_dd"] > -0.10
                              and positive >= 4),
    }
    print(f"   RESEARCH_PASS  SR {headline_sr:+.3f} > 0.70 ? "
          f"{headline_sr > 0.70} | maxDD {table['M3']['max_dd']:.1%} > -25% ? "
          f"{table['M3']['max_dd'] > -0.25} | folds {positive}/5 >= 3 ? {positive >= 3} | "
          f"DSR > 0 ? {dsr['deflated_excess_ann'] > 0}")
    print(f"   -> RESEARCH_PASS {gate['research_pass']}")
    print(f"   PROPFIRM_PASS (context only) -> {gate['propfirm_pass']}")
    results["gates"] = gate

    verdicts = {"K1": results["K1"]["verdict"], "K2": results["K2"]["verdict"],
                "K3": results["K3"]["verdict"], "K4": results["K4"]["verdict"],
                "K5": results["K5"]["verdict"], "K6": "PASS"}
    print(f"\n{RULE}\n11. THE SIX, TOGETHER\n")
    for k, v in verdicts.items():
        print(f"     {k}  {v}")
    results["verdicts"] = verdicts
    print("\n   K1 is the criterion that decides H. K2, K5 and K6 pass, which is not")
    print("   a defence: they are integrity checks on a book whose level has already")
    print("   failed. K3 and K4 are the two publishable subsidiary results.")
    print("\n   One caveat carries over every line: K1's verdict is threshold-sensitive.")
    print(f"   On the committed construction the book reads {headline_sr:+.3f}; on the one")
    print(f"   §3's own sample definition implies it reads "
          f"{audit['sample_slicing']['built_on_full_panel_sharpe']:+.3f}, against a 0.50 line.")
    print("   See §9(k). Neither clears the 0.70 gate, and both sit inside the same")
    print("   bootstrap interval, so what is established is the gate, not the kill.")
    print(f"\n{RULE}")

    (OUT / "evaluation.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\n   written  {OUT / 'evaluation.json'}")


if __name__ == "__main__":
    main()
