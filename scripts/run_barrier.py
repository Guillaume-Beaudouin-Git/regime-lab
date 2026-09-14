"""Regime sizing against a propfirm drawdown barrier.

The study's central result is that the regime classification separates forward
*variance* and not forward *mean*. Every device built on it inside a classical
book has been refuted, four times independently, because a classical book is
scored by Sharpe and a pure variance signal moves the denominator without
moving the numerator.

A propfirm account is a different objective. It is a survival problem against a
drawdown barrier plus a profit target, and the probability of touching a barrier
is governed by variance directly. So a signal that predicts only variance should
be worth something *here* even though it is worth nothing in a book. That is the
thesis, and the three nulls below are built to let it fail.

Reported quantities are dp(pass) and d(duration). Not dSharpe: a Sharpe here
would be answering a different question.

Nothing is written into either repository. The propfirm geometry comes from
Algo_claude and is used, not reimplemented; the states, the book, the frozen
sizing rule and the block bootstrap come from regime-lab and are used, not
reimplemented.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from regime_lab.config import CACHE, ROOT  # noqa: E402

# This is the only script here that needs a sibling repository. The propfirm
# barrier geometry is not reimplemented: it is imported from the audited
# simulator in Algo_claude, because a geometry written here would be the thing
# under test rather than the thing tested against. Algo_claude is NOT a declared
# dependency of this package and a fresh clone will not have it, so the failure
# is made loud rather than arriving as a bare ImportError.
#
# The path is resolved through ROOT rather than __file__ so that the module still
# loads under the exec harness in tests/test_importable.py, which supplies no
# __file__. The guard raises only when main() is reached, for the same reason:
# importing this file must stay free of side effects.
ALGO_CLAUDE = Path(os.environ.get("ALGO_CLAUDE_PATH", ROOT.parent / "Algo_claude"))
OUT = Path(os.environ.get("BARRIER_OUT", CACHE))
_SIMULATOR = ALGO_CLAUDE / "Portfolio" / "risk" / "trailing_dd_simulator.py"
if _SIMULATOR.exists():
    sys.path.insert(0, str(ALGO_CLAUDE))


def _require_simulator() -> None:
    """Fail with an explanation, not an ImportError, and only when actually run."""
    if not _SIMULATOR.exists():
        raise SystemExit(
            "scripts/run_barrier.py needs the audited propfirm simulator, which "
            f"lives in a separate repository and was not found at {ALGO_CLAUDE}.\n"
            "Set ALGO_CLAUDE_PATH to its checkout. Every other script in this "
            "repository runs without it, and the published barrier figures are "
            "in docs/RESULTS_BARRIER.md."
        )


if _SIMULATOR.exists():
    from Portfolio.risk.propfirm_canonical import (  # noqa: E402
        FTMO_DAILY_DD_PCT,
        FTMO_MAX_DD_PCT,
        FTMO_MIN_TRADING_DAYS,
        FTMO_PROFIT_TARGET_PCT,
    )
    from Portfolio.risk.trailing_dd_simulator import (  # noqa: E402
        PropfirmConfig,
        make_iter10_trade_generator,
        simulate_n_runs,
        simulate_run,
    )

from regime_lab.analysis.bootstrap import stationary_indices  # noqa: E402
from regime_lab.extensions.trend import VOL_TARGET, book  # noqa: E402
from regime_lab.models.mapping import state_to_size  # noqa: E402

RULE = "=" * 78
Z_MDE = 1.959964 + 0.841621  # two-sided 5%, 80% power -- the constants in analysis/power.py


# ---------------------------------------------------------------------------
# DECLARED BEFORE ANY RESULT WAS READ
# ---------------------------------------------------------------------------

DECLARED = {
    # A' sparse jump carries the primary test: it is the project's best-validated
    # classifier against the external NBER benchmark (93.2% balanced accuracy,
    # kappa 0.53) and the one family where the sample resolves the portfolio
    # question. All five families are reported as a panel with a family-wise
    # null, so the primary is fixed here and not chosen after the panel is read.
    "primary_family": "A' sparse jump",
    # Causal-estimator burn-ins, identical for every arm and every null so that
    # no arm is given a longer history than another.
    "min_history": 504,     # two years, for any expanding rank of volatility
    "min_normaliser": 126,  # six months, for the exposure normaliser
    # Frozen sizing rule. VOL_TARGET is extensions/trend.py's own constant;
    # cap 2.0 is state_to_size's own default. Neither is chosen here.
    "size_target": VOL_TARGET,
    "size_cap": 2.0,
    # Round-trip costs by asset class, from the project's declared table:
    # futures ~0.01%, equities 0.05-0.10%, FX 0.15%, crypto 0.17% (no crypto in
    # this universe). Charged as half a round trip per crossing, so an
    # out-and-back costs exactly the declared figure.
    "cost_bps_roundtrip": {"future": 1.0, "equity_etf": 7.5, "fx": 15.0},
    "cost_convention": "half the declared round trip per crossing",
    # Politis-Romano stationary block bootstrap. 63 sessions is one quarter, the
    # default in analysis/power.py, chosen there as the length of a stress
    # episode. Barrier crossings cluster heavily, so iid is not an option.
    "mean_block": 63,
    "block_sensitivity": [21, 63, 126],
    "n_paths": 4000,       # pseudo-accounts for a point estimate
    "n_paths_null": 2000,  # pseudo-accounts inside each null draw
    "n_outer": 300,        # pseudo-histories, for the history-level uncertainty
    "n_inner": 300,        # windows drawn inside each pseudo-history
    "n_placebo": 200,      # realised-vol-quantile sizers (null 1)
    "n_permute": 400,      # duration-preserving state permutations (null 2)
    "n_rotate": 400,       # circular rotations of the leverage profile (null 3)
    "n_family": 150,       # draws of the family-wise max statistic
    # Notional calibration, fixed on the CONSTANT-SIZE arm alone at the risk
    # level whose p(pass) is nearest one half. That arm carries no regime
    # information, so the choice cannot favour either side, and the full ladder
    # is printed so nothing hides behind it.
    # TWO AMENDMENTS, both logged and both named in the report.
    #
    # (1) The ladder was first declared [0.01 .. 0.12] and it did NOT bracket
    #     0.50: p(pass) on the constant arm plateaus near 0.32-0.37 and never
    #     reaches a half at any leverage, so "nearest 0.50" selected the grid
    #     edge. The ladder was extended upward.
    # (2) With the ladder extended, "nearest 0.50" became "the maximum" and
    #     selected 25% account volatility -- where the median account resolves
    #     in TEN sessions. A barrier that resolves in ten sessions cannot
    #     express a regime whose episodes run 43 to 2,698 sessions, so that rung
    #     destroys the mechanism under test. The outcome-based calibration rule
    #     is therefore ABANDONED and replaced by a purely geometric anchor:
    #     account annualised volatility equal to the barrier depth itself
    #     (trailing_dd_eval / capital). The barrier is then exactly one year of
    #     volatility away. No outcome enters that choice. And the comparison is
    #     reported at EVERY rung of the ladder for every arm, so no rung is
    #     privileged and nothing hides behind the anchor.
    "risk_ladder_ann_vol": [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12,
                            0.16, 0.20, 0.25, 0.32],
    "calibration_rule": "geometric: account annual vol = barrier depth "
                        "(trailing_dd_eval / capital); no outcome enters it",
    "exposure_elasticity_grid": [0.90, 0.95, 1.00, 1.025, 1.05, 1.08, 1.12],
    "placebo_windows": [21, 42, 63, 126, 252],
    "placebo_thresholds": "40 points on [0.50, 0.95]",
    "seed": 20260913,
}

# Run scale. Every count actually used is written into the JSON and must be
# quoted in the report: a silent truncation reads as full coverage, and this
# project has already been caught twice by exactly that.
_SCALE = os.environ.get("BARRIER_SCALE", "full")
_QUICK = {"n_paths": 1500, "n_paths_null": 600, "n_outer": 80, "n_inner": 150,
          "n_placebo": 60, "n_permute": 120, "n_rotate": 120, "n_family": 24}
if _SCALE == "quick":
    DECLARED.update(_QUICK)
DECLARED["run_scale"] = _SCALE

COST_CLASS = {
    **{t: "future" for t in ["CL=F", "CT=F", "GC=F", "HG=F", "HO=F", "KC=F", "NG=F",
                             "PL=F", "SB=F", "SI=F", "ZC=F", "ZS=F", "ZW=F"]},
    **{t: "future" for t in ["^AEX", "^AXJO", "^FCHI", "^FTSE", "^GDAXI", "^GSPC",
                             "^GSPTSE", "^HSI", "^IBEX", "^N225", "^NDX", "^RUT",
                             "^STOXX50E", "DX-Y.NYB"]},
    **{t: "equity_etf" for t in ["AGG", "BWX", "EMB", "HYG", "IEF", "LQD", "SHY",
                                 "TIP", "TLT"]},
    **{t: "fx" for t in ["AUDUSD=X", "CAD=X", "CHF=X", "EURUSD=X", "GBPUSD=X",
                         "JPY=X", "MXN=X", "NOK=X", "NZDUSD=X", "SEK=X"]},
}


# ---------------------------------------------------------------------------
# 1 -- the real geometry, read off the two Algo_claude modules
# ---------------------------------------------------------------------------

def report_geometry() -> PropfirmConfig:
    cfg = PropfirmConfig()
    lock = cfg.capital + cfg.trailing_dd_funded + cfg.lock_trigger_excess
    print(RULE)
    print("1. GEOMETRIE REELLE, citee depuis Algo_claude/Portfolio/risk/")
    print(RULE)
    print(f"""
  trailing_dd_simulator.py -- Tradeify Select Flex 25K, parametres cites :
    capital                {cfg.capital:>11,.0f} $
    profit_target          {cfg.profit_target:>11,.0f} $  = {cfg.profit_target/cfg.capital:.1%} du capital
    trailing_dd_eval       {cfg.trailing_dd_eval:>11,.0f} $  = {cfg.trailing_dd_eval/cfg.capital:.1%}
    trailing_dd_funded     {cfg.trailing_dd_funded:>11,.0f} $
    lock_trigger_excess    {cfg.lock_trigger_excess:>11,.0f} $  -> verrou a {lock:,.0f} $
    locked_floor_offset    {cfg.locked_floor_offset:>11,.0f} $  -> plancher verrouille {cfg.capital+cfg.locked_floor_offset:,.0f} $
    consistency_cap_pct    {cfg.consistency_cap_pct:>11.0%}    -> {cfg.consistency_cap_pct*cfg.profit_target:,.0f} $/jour comptabilises (eval seule)
    max_eval_days          {cfg.max_eval_days:>11d}
    funded_days            {cfg.funded_days:>11d}

  Nature du drawdown, lue dans simulate_run() ligne par ligne :
    GLISSANT et non statique. `floor_e = max(floor_e, equity - trailing_dd)` :
      le plancher monte avec le pic d'equity de fin de journee et ne redescend
      jamais. En phase eval il ne se verrouille JAMAIS ("no lock in eval").
    Sur EQUITY et non sur balance : `equity += pnl`, puis `if equity <= floor_e`.
    Le test de rupture arrive AVANT la mise a jour du plancher du jour, donc la
      perte nette du jour est traitee comme le pire creux intra-journalier.
    Objectif : DEUX conditions simultanees, `cum_capped >= 1500` (somme des
      jours positifs, chacun plafonne a 600) ET `equity >= 26500`.
    Contrainte de duree : aucune dans les regles du prop ; 180 jours dans le
      simulateur, puis 180 jours de phase financee.

  Ce que la geometrie EOD coute, cite depuis le docstring du simulateur :
    "It overestimates survival for strategies with large intraday swings that
     recover by EOD". Un livre quotidien n'a qu'un prix par jour : je ne peux
    pas faire mieux. La p(pass) rapportee est donc un MAJORANT. Le biais est
    identique sur les deux bras, il deplace les niveaux et non le delta.

  propfirm_canonical.py -- PROPFIRM_CANONIQUE = FTMO, geometrie DIFFERENTE :
    perte journaliere {FTMO_DAILY_DD_PCT:.0%} du solde de depart, drawdown maximum
    {FTMO_MAX_DD_PCT:.0%} STATIQUE, cible {FTMO_PROFIT_TARGET_PCT:.0%}, minimum {FTMO_MIN_TRADING_DAYS} jours de trading.
    trailing_dd_simulator.py ne peut pas exprimer un plancher statique : le sien
    monte toujours. Cette geometrie passe donc en SECONDAIRE, par un adaptateur
    de vingt lignes ecrit ici et signale comme tel a chaque mention.

  sizing.py : ERC (Maillard-Roncalli-Teiletche) et inverse-vol ENTRE strategies,
    fenetre 63 jours, plafond 0,60 par strategie, puis scale_to_target_vol.
    Un seul livre ici : aucune de ses fonctions n'entre dans ce calcul. Cite
    pour dire qu'il a ete lu et qu'il ne s'applique pas.
""")
    return cfg


def verify_simulator(cfg: PropfirmConfig) -> None:
    res = simulate_n_runs(20_000, make_iter10_trade_generator(360, 3, seed=42), cfg)
    ok = abs(res.p_eval_pass - 0.537) < 0.002 and abs(res.p_bust - 0.462) < 0.002
    print("  verification du simulateur AVANT usage. Son docstring annonce")
    print('  "P(pass) ~ 53.7%, P(bust) ~ 46.2% (N=20k, seed=42, max_eval_days=180)".')
    print(f"  obtenu ici : p(pass) {res.p_eval_pass:.3%}, p(bust) {res.p_bust:.3%}"
          f"  ->  {'REPRODUIT' if ok else 'ECART, ne pas utiliser'}\n")
    if not ok:
        raise SystemExit("le simulateur ne reproduit pas son propre chiffre documente")


# ---------------------------------------------------------------------------
# 2 -- book, costs, sizing arms
# ---------------------------------------------------------------------------

@dataclass
class Book:
    trend: pd.Series
    weights: pd.DataFrame
    W: np.ndarray          # weights as a float array, aligned on trend.index
    c: np.ndarray          # per-crossing cost, per instrument, in return units
    r: np.ndarray          # book return, aligned on trend.index
    sigma: pd.Series       # trailing 63-day annualised book vol, lagged


def build_book() -> Book:
    px = pd.read_parquet(CACHE / "trend_universe.parquet")
    trend, weights = book(px)
    trend = trend.dropna()
    weights = weights.reindex(trend.index).fillna(0.0)
    missing = [c for c in weights.columns if c not in COST_CLASS]
    if missing:
        raise KeyError(f"instruments sans classe de cout declaree : {missing}")
    bps = np.array([DECLARED["cost_bps_roundtrip"][COST_CLASS[c]] for c in weights.columns])
    sigma = (trend.rolling(63).std() * np.sqrt(252)).shift(1)
    return Book(trend, weights, weights.to_numpy(), bps / 2.0 / 1e4,
                trend.to_numpy(), sigma)


def net_returns(bk: Book, lev: np.ndarray) -> np.ndarray:
    """Book scaled by ``lev``, minus the cost of the turnover that creates."""
    scaled = bk.W * lev[:, None]
    traded = np.abs(np.diff(scaled, axis=0, prepend=scaled[:1]))
    return bk.r * lev - traded @ bk.c


def normalise(lev: np.ndarray, min_periods: int) -> np.ndarray:
    """Divide a leverage profile by its own causal running mean.

    Every arm and every null goes through this, so all of them carry the same
    average dollar exposure and the only thing that differs is *when* the risk
    is taken. T3 showed the apparent gain of these devices travels through
    exposure rather than through the mean, so matching exposure is not optional.
    """
    s = pd.Series(lev).shift(1)
    scale = s.expanding(min_periods=min_periods).mean().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return lev / scale


_SV_CACHE: dict[str, pd.DataFrame] = {}


def state_vol_table(family: str) -> pd.DataFrame:
    if family not in _SV_CACHE:
        rd = pd.read_parquet(CACHE / "refit_diagnostics.parquet")
        sv = rd[rd["family"] == family].set_index("refit")[["state_vol_0", "state_vol_1"]]
        sv.index = pd.to_datetime(sv.index)
        _SV_CACHE[family] = sv
    return _SV_CACHE[family]


def regime_leverage(states: pd.Series, family: str, index: pd.Index) -> np.ndarray:
    """The FROZEN rule, called as written: leverage inverse to the state's vol."""
    lev = state_to_size(states, state_vol_table(family),
                        target=DECLARED["size_target"], cap=DECLARED["size_cap"])
    return lev.replace(0.0, np.nan).reindex(index).to_numpy()


def vol_quantile(bk: Book, window: int) -> np.ndarray:
    sigma = (bk.trend.rolling(window).std() * np.sqrt(252))
    return sigma.expanding(min_periods=DECLARED["min_history"]).rank(pct=True).shift(1).to_numpy()


def bucket_leverage(bk: Book, quantile: np.ndarray, threshold: float) -> np.ndarray:
    """Two-bucket sizing on a realised-vol quantile -- the shape of the state rule.

    The bucket's volatility is a causal expanding mean of realised volatility
    over the past days that fell in that bucket, so the placebo is handed the
    same kind of information the frozen rule reads (a training-window volatility
    per state). Leverage is expressed as a ratio to the book's own causal
    average volatility and capped at the same 2.0, so the cap bites at the same
    relative point on every arm.
    """
    sig = bk.sigma.to_numpy()
    ok = np.isfinite(quantile) & np.isfinite(sig)
    hot = quantile >= threshold
    ref = bk.sigma.expanding(min_periods=DECLARED["min_history"]).mean().to_numpy()
    out = np.full(len(sig), np.nan)
    for flag in (False, True):
        mask = ok & (hot == flag)
        est = pd.Series(np.where(mask, sig, np.nan)).expanding(min_periods=1).mean().to_numpy()
        out[mask] = ref[mask] / est[mask]
    return np.clip(out, None, DECLARED["size_cap"])


def inverse_vol_leverage(bk: Book) -> np.ndarray:
    """A dynamic volatility target with no parameter at all.

    Scale to your own past average volatility. No percentile, no threshold, no
    fitted constant, and no knowledge of regimes. This is the control that beat
    the practitioner attenuator by a tenth of a Sharpe in docs/EXTENSIONS.md,
    and it is the control that has to be beaten here too.
    """
    ref = bk.sigma.expanding(min_periods=DECLARED["min_history"]).mean().to_numpy()
    return np.clip(ref / bk.sigma.to_numpy(), None, DECLARED["size_cap"])


def permute_runs(states: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Shuffle the ORDER of the state's runs, keeping every run length intact.

    Runs alternate by construction, so the lengths of the state-0 runs are
    permuted among the state-0 slots and likewise for state 1. Total time in
    each state and the whole distribution of episode durations survive exactly;
    only the alignment with the calendar is destroyed.
    """
    v = states.to_numpy()
    change = np.r_[True, v[1:] != v[:-1]]
    labels = v[change]
    lengths = np.diff(np.r_[np.flatnonzero(change), len(v)])
    out = lengths.copy()
    for lab in np.unique(labels):
        slots = np.flatnonzero(labels == lab)
        out[slots] = rng.permutation(lengths[slots])
    return pd.Series(np.repeat(labels, out), index=states.index, name="permuted")


# ---------------------------------------------------------------------------
# 3 -- the adapter: daily book return -> daily USD P&L on the account
# ---------------------------------------------------------------------------

def to_usd(net: np.ndarray, capital: float, ann_vol: float, book_vol: float) -> np.ndarray:
    """The whole adapter. The simulator already takes a daily USD P&L array
    including no-trade days, so no part of the barrier is rewritten here."""
    return net * (ann_vol / book_vol) * capital


@dataclass
class Metrics:
    p_pass: float
    p_bust: float
    p_timeout: float
    med_days_pass: float
    med_days_bust: float
    p90_days_pass: float
    med_underwater: float
    mean_exposure: float
    ann_vol: float


def longest_underwater(paths: np.ndarray) -> np.ndarray:
    """Longest run of days below the running equity peak, per path.

    AlphaSimplex measures that trend performance tracks the DURATION of a
    drawdown and not its depth: Covid was -34% in 23 days and trend made -2.4%,
    the internet bubble -47% over 545 days and trend made +62%, and below about
    12.5% of drawdown trend has never been positive. So duration is reported at
    the same rank as p(pass), not as a footnote.
    """
    equity = np.cumsum(paths, axis=1)
    under = equity < np.maximum.accumulate(equity, axis=1)
    run = np.zeros(len(paths), dtype=np.int32)
    best = np.zeros(len(paths), dtype=np.int32)
    for t in range(under.shape[1]):
        run = (run + 1) * under[:, t]
        np.maximum(best, run, out=best)
    return best


def evaluate(paths: np.ndarray, cfg: PropfirmConfig, exposure: float,
             ann_vol: float) -> tuple[Metrics, np.ndarray]:
    n = len(paths)
    passed = np.zeros(n, dtype=bool)
    busted = np.zeros(n, dtype=bool)
    days = np.zeros(n)
    for i in range(n):
        res = simulate_run(paths[i], cfg)
        passed[i] = res.eval_passed
        busted[i] = res.busted and res.bust_phase == "eval"
        days[i] = res.eval_days
    under = longest_underwater(paths)
    m = Metrics(
        p_pass=float(passed.mean()),
        p_bust=float(busted.mean()),
        p_timeout=float((~passed & ~busted).mean()),
        med_days_pass=float(np.median(days[passed])) if passed.any() else np.nan,
        med_days_bust=float(np.median(days[busted])) if busted.any() else np.nan,
        p90_days_pass=float(np.percentile(days[passed], 90)) if passed.any() else np.nan,
        med_underwater=float(np.median(under)),
        mean_exposure=exposure,
        ann_vol=ann_vol,
    )
    return m, passed


# ---------------------------------------------------------------------------
# FTMO secondary geometry -- adapter written here, NOT the repository simulator
# ---------------------------------------------------------------------------

def simulate_run_ftmo(daily_pnl: np.ndarray, capital: float) -> tuple[bool, bool, int]:
    """The static-floor geometry declared canonical in propfirm_canonical.py.

    trailing_dd_simulator.py implements a trailing floor only and cannot express
    a static one, so this function is mine. It implements exactly the four
    constants of that module and invents nothing: a daily loss limit of
    FTMO_DAILY_DD_PCT of the starting balance, a floor at (1 - FTMO_MAX_DD_PCT)
    of the starting balance that never moves, a target of
    FTMO_PROFIT_TARGET_PCT, and a minimum of FTMO_MIN_TRADING_DAYS traded days.
    """
    daily_limit = FTMO_DAILY_DD_PCT * capital
    floor = capital * (1.0 - FTMO_MAX_DD_PCT)
    target = capital * (1.0 + FTMO_PROFIT_TARGET_PCT)
    equity = capital
    traded = 0
    for i, pnl in enumerate(daily_pnl, start=1):
        if pnl != 0.0:
            traded += 1
        if -pnl >= daily_limit:
            return False, True, i
        equity += pnl
        if equity <= floor:
            return False, True, i
        if equity >= target and traded >= FTMO_MIN_TRADING_DAYS:
            return True, False, i
    return False, False, len(daily_pnl)


def evaluate_ftmo(paths: np.ndarray, capital: float, exposure: float,
                  ann_vol: float) -> Metrics:
    n = len(paths)
    passed = np.zeros(n, dtype=bool)
    busted = np.zeros(n, dtype=bool)
    days = np.zeros(n)
    for i in range(n):
        p, b, d = simulate_run_ftmo(paths[i], capital)
        passed[i], busted[i], days[i] = p, b, d
    under = longest_underwater(paths)
    return Metrics(
        p_pass=float(passed.mean()),
        p_bust=float(busted.mean()),
        p_timeout=float((~passed & ~busted).mean()),
        med_days_pass=float(np.median(days[passed])) if passed.any() else np.nan,
        med_days_bust=float(np.median(days[busted])) if busted.any() else np.nan,
        p90_days_pass=float(np.percentile(days[passed], 90)) if passed.any() else np.nan,
        med_underwater=float(np.median(under)),
        mean_exposure=exposure,
        ann_vol=ann_vol,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    _require_simulator()
    t_start = time.perf_counter()
    cfg = report_geometry()
    verify_simulator(cfg)

    rng = np.random.default_rng(DECLARED["seed"])
    horizon = cfg.max_eval_days + cfg.funded_days
    bk = build_book()
    states_all = pd.read_parquet(CACHE / "states_offline.parquet")
    families = list(states_all.columns)
    idx_all = bk.trend.index

    print(RULE)
    print("2. LE LIVRE, SES COUTS, SON ECHANTILLON")
    print(RULE)
    print(f"\n  livre TSMOM de reference (extensions/trend.py, non modifie) : "
          f"{bk.weights.shape[1]} instruments,")
    print(f"  {len(bk.trend):,} sessions, {idx_all.min():%Y-%m} a {idx_all.max():%Y-%m}, "
          f"vol annualisee {bk.trend.std(ddof=1)*np.sqrt(252):.2%}")
    print(f"  Sharpe brut {bk.trend.mean()/bk.trend.std(ddof=1)*np.sqrt(252):.2f} "
          "-- cite POUR IDENTIFIER LE LIVRE, pas comme sortie de ce travail.")
    active = bk.weights.ne(0.0).sum(axis=1)
    print("\n  je n'exige PAS que les 46 instruments aient de la donnee a chaque date :")
    print(f"  instruments actifs {active.iloc[1]:.0f} en {idx_all[1]:%Y-%m}, "
          f"{active.loc['2007-12-31':].iloc[0]:.0f} en 2008-01, "
          f"{active.iloc[-1]:.0f} a la fin, mediane {active.median():.0f}.")
    print("  (le piege maison, reproduit deux fois : cette contrainte avait fait")
    print("   demarrer un echantillon en 2007 au lieu de 2000, jete 8 ans, et")
    print("   fabrique un t de 2,18 qui n'etait que la troncature.)")

    turn = np.abs(np.diff(bk.W, axis=0, prepend=bk.W[:1]))
    blended = float((turn @ (bk.c * 2 * 1e4)).sum() / turn.sum())
    drag = float((turn @ bk.c).mean() * 252)
    print(f"\n  couts declares par classe, aller-retour : futures "
          f"{DECLARED['cost_bps_roundtrip']['future']:.0f} bp, ETF obligataires "
          f"{DECLARED['cost_bps_roundtrip']['equity_etf']:.1f} bp, FX "
          f"{DECLARED['cost_bps_roundtrip']['fx']:.0f} bp, crypto 17 bp (absent de l'univers)")
    print(f"  convention : {DECLARED['cost_convention']}")
    print(f"  aller-retour moyen pondere par la rotation REELLE : {blended:.2f} bp")
    print(f"  ponction annuelle sur le bras a taille constante  : {drag:.2%} de rendement")
    print("  pourquoi : 13 futures et 14 indices actions tradables en futures a 1 bp,")
    print("  9 ETF obligataires dans la bande actions 5-10 bp, 10 paires FX a 15 bp.")

    # ---- arms ---------------------------------------------------------------
    states = states_all[DECLARED["primary_family"]].dropna()
    v_states = states.to_numpy()
    share_hot = float((states == 0.0).mean())
    lev_regime_free = regime_leverage(states, DECLARED["primary_family"], idx_all)
    q63 = vol_quantile(bk, 63)
    mp = DECLARED["min_normaliser"]

    arms = {
        "taille constante": np.ones(len(idx_all)),
        "regime, gross libre": lev_regime_free,
        "regime, exposition appariee": normalise(lev_regime_free, mp),
        "placebo vol appariee (2 seaux)": normalise(bucket_leverage(bk, q63, 1.0 - share_hot), mp),
        "cible de vol, sans parametre": normalise(inverse_vol_leverage(bk), mp),
    }

    valid = np.ones(len(idx_all), dtype=bool)
    for v in arms.values():
        valid &= np.isfinite(v)
    first = int(np.argmax(valid))
    last = len(valid) - int(np.argmax(valid[::-1]))
    if not valid[first:last].all():
        raise SystemExit(f"echantillon commun non contigu : "
                         f"{int((~valid[first:last]).sum())} trous")
    sl = slice(first, last)
    common = idx_all[sl]
    n_common = len(common)
    print(f"\n  ECHANTILLON COMMUN a tous les bras et a tous les nuls : {n_common:,} "
          f"sessions,\n  {common[0]:%Y-%m-%d} a {common[-1]:%Y-%m-%d}, contigu.")
    print("  (la lecon de l'amendement du 10/09 : comparer des constructions")
    print("   mesurees sur des echantillons differents fabrique des ecarts qui")
    print("   sont de l'amorcage, pas du conditionnement.)")

    nets = {k: net_returns(bk, np.nan_to_num(v, nan=0.0))[sl] for k, v in arms.items()}
    book_vol = float(nets["taille constante"].std(ddof=1) * np.sqrt(252))
    cap_hits = float((lev_regime_free[sl] >= DECLARED["size_cap"] - 1e-9).mean())
    print(f"\n  plafond de levier {DECLARED['size_cap']} : mord sur {cap_hits:.2%} des sessions "
          "du bras de regime")
    print(f"\n  {'bras':<32}{'exposition moy.':>16}{'ecart-type expo':>17}{'vol ann.':>10}")
    for k, v in arms.items():
        print(f"  {k:<32}{np.nanmean(v[sl]):>16.3f}{np.nanstd(v[sl]):>17.3f}"
              f"{nets[k].std(ddof=1)*np.sqrt(252):>10.2%}")

    # ---- notional calibration ----------------------------------------------
    print("\n" + RULE)
    print("3. NIVEAU DE RISQUE DU COMPTE -- toute l'echelle, tous les bras")
    print(RULE)
    idx = np.stack([stationary_indices(n_common, DECLARED["mean_block"], rng)[:horizon]
                    for _ in range(DECLARED["n_paths"])])
    idxn = idx[:DECLARED["n_paths_null"]]
    print(f"\n  {'vol ann. du compte':<20}{'levier':>7}{'p(pass)':>9}{'p(bust)':>9}"
          f"{'p(timeout)':>11}{'j.cible':>9}{'j.echec':>9}")
    ladder = {}
    for av in DECLARED["risk_ladder_ann_vol"]:
        m, _ = evaluate(to_usd(nets["taille constante"][idx], cfg.capital, av, book_vol),
                        cfg, 1.0, av)
        ladder[av] = m
        print(f"  {av:<20.1%}{av/book_vol:>7.2f}{m.p_pass:>9.1%}{m.p_bust:>9.1%}"
              f"{m.p_timeout:>11.1%}{m.med_days_pass:>9.0f}{m.med_days_bust:>9.0f}")
    top = max(ladder, key=lambda a: ladder[a].p_pass)
    chosen = cfg.trailing_dd_eval / cfg.capital
    if chosen not in ladder:
        raise SystemExit("l'ancre geometrique n'est pas un barreau de l'echelle")
    print(f"""
  DEUX AMENDEMENTS, journalises et nommes dans le rapport.

  (1) L'echelle declaree s'arretait a 12% et n'encadrait PAS 0,50. La p(pass)
      du bras constant plafonne : maximum {ladder[top].p_pass:.1%} a {top:.0%} de vol, et
      elle ne franchit 0,50 a AUCUN levier. La regle "au plus proche de 0,50"
      selectionnait donc le bord de la grille. Echelle prolongee vers le haut.

  (2) Prolongee, la regle est devenue "le maximum" et a designe {top:.0%} de vol,
      ou le compte median se resout en {ladder[top].med_days_bust:.0f} sessions. Une barriere qui
      se resout en dix sessions ne peut pas exprimer un regime dont les
      episodes courent de 43 a 2 698 sessions : ce barreau DETRUIT le
      mecanisme teste. La regle de calibrage fondee sur un resultat est donc
      ABANDONNEE et remplacee par une ancre purement geometrique.

  Ancre retenue : vol annualisee du compte = profondeur de la barriere
    = trailing_dd_eval / capital = {chosen:.0%}. La barriere est alors a exactement
    une annee de volatilite. Aucun resultat n'entre dans ce choix.
    Levier implicite sur le livre : x{chosen/book_vol:.2f}.
  Et la comparaison est rapportee a CHAQUE barreau (section 4bis), donc aucun
    barreau n'est privilegie et rien ne se cache derriere l'ancre.""")

    print("\n  ELASTICITE A L'EXPOSITION, mesuree sur le bras CONSTANT seul, a")
    print("  l'ancre. T3 vient de montrer que le gain apparent de ces dispositifs")
    print("  passe par le denominateur et l'exposition. Pour savoir combien d'une")
    print("  dp(pass) est achetee par de l'exposition nue et rien d'autre, je")
    print("  mesure le prix de l'exposition sur un bras sans aucun signal.")
    print(f"\n  {'multiplicateur d exposition':<28}{'p(pass)':>10}{'dp':>9}")
    elast = {}
    for mult in DECLARED["exposure_elasticity_grid"]:
        m, _ = evaluate(to_usd(nets["taille constante"][idx] * mult, cfg.capital,
                               chosen, book_vol), cfg, mult, chosen)
        elast[mult] = m.p_pass
    p_ref = elast[1.0]
    for mult, p in elast.items():
        print(f"  {mult:<28.3f}{p:>10.1%}{p - p_ref:>+9.1%}")
    slope = float(np.polyfit(list(elast), [elast[k] for k in elast], 1)[0])
    print(f"\n  pente : {slope:+.4f} de p(pass) par unite de multiplicateur, soit")
    print(f"  {slope/100:+.4f} par point de pourcentage d'exposition. C'est le prix")
    print("  de l'exposition nue, et c'est le barometre contre lequel tout residu")
    print("  de dp(pass) doit etre lu.")

    def paths(name: str, ix: np.ndarray = idx) -> np.ndarray:
        return to_usd(nets[name][ix], cfg.capital, chosen, book_vol)

    def paths_of(net: np.ndarray, ix: np.ndarray = idxn) -> np.ndarray:
        return to_usd(net[ix], cfg.capital, chosen, book_vol)

    # ---- the arms -----------------------------------------------------------
    print("\n" + RULE)
    print("4. LES DEUX VERSIONS DU MEME LIVRE (etats AVEC LATENCE)")
    print(RULE)
    print(f"\n  famille primaire {DECLARED['primary_family']}, lue dans")
    print("  data/cache/states_offline.parquet -- la version avec recul, la seule")
    print("  honnete (latence reelle 0 a 13 jours).")
    print(f"  part du temps dans l'etat turbulent : {share_hot:.1%}")
    print(f"  {DECLARED['n_paths']:,} pseudo-comptes, bloc moyen "
          f"{DECLARED['mean_block']} sessions, {horizon} jours par compte\n")
    print(f"  {'bras':<32}{'p(pass)':>9}{'p(bust)':>9}{'j.cible':>9}{'j.echec':>9}"
          f"{'p90 cible':>11}{'sous l eau':>12}{'expo':>7}")
    results: dict[str, Metrics] = {}
    flags: dict[str, np.ndarray] = {}
    for name in arms:
        m, f = evaluate(paths(name), cfg, float(np.nanmean(arms[name][sl])), chosen)
        results[name], flags[name] = m, f
        print(f"  {name:<32}{m.p_pass:>9.1%}{m.p_bust:>9.1%}{m.med_days_pass:>9.0f}"
              f"{m.med_days_bust:>9.0f}{m.p90_days_pass:>11.0f}"
              f"{m.med_underwater:>12.0f}{m.mean_exposure:>7.2f}")

    base = results["taille constante"]
    print(f"\n  {'delta contre taille constante':<32}{'dp(pass)':>10}{'d j.cible':>11}"
          f"{'d j.echec':>11}{'d p90':>8}{'d sous l eau':>14}")
    deltas: dict[str, dict[str, float]] = {}
    for name, m in results.items():
        if name == "taille constante":
            continue
        deltas[name] = {"dp": m.p_pass - base.p_pass,
                        "d_days_pass": m.med_days_pass - base.med_days_pass,
                        "d_days_bust": m.med_days_bust - base.med_days_bust,
                        "d_p90_pass": m.p90_days_pass - base.p90_days_pass,
                        "d_under": m.med_underwater - base.med_underwater,
                        "exposure": m.mean_exposure}
        d = deltas[name]
        print(f"  {name:<32}{d['dp']:>+10.1%}{d['d_days_pass']:>+11.0f}"
              f"{d['d_days_bust']:>+11.0f}{d['d_p90_pass']:>+8.0f}{d['d_under']:>+14.0f}")

    print("\n  CE QUE L'EXPOSITION EXPLIQUE DEJA, avec la pente mesuree ci-dessus.")
    print("  Aucun de ces bras n'est exactement a 1,00 d'exposition : le")
    print("  normalisateur est une moyenne expansive causale, pas un scalaire")
    print("  plein echantillon, donc il reste un residu. Le voici chiffre.")
    print(f"\n  {'bras':<32}{'expo':>7}{'dp observe':>12}{'dp attendu par expo.':>22}"
          f"{'residu':>9}")
    for name, d in deltas.items():
        exp_dp = slope * (d["exposure"] - 1.0)
        d["dp_explained_by_exposure"] = exp_dp
        d["dp_residual"] = d["dp"] - exp_dp
        print(f"  {name:<32}{d['exposure']:>7.3f}{d['dp']:>+12.1%}{exp_dp:>+22.1%}"
              f"{d['dp'] - exp_dp:>+9.1%}")

    print("\n  CONSTANTES DE TEMPS -- la comparaison qui gouverne tout le reste.")
    print(f"    duree mediane de vie d'un compte qui echoue : {base.med_days_bust:.0f} sessions")
    print(f"    duree mediane jusqu'a la cible quand elle est atteinte : "
          f"{base.med_days_pass:.0f} sessions")
    _vs = pd.Series(v_states)
    rl = _vs.groupby((_vs != _vs.shift()).cumsum()).size()
    print(f"    duree mediane d'un episode de regime : {rl.median():.0f} sessions, "
          f"moyenne {rl.mean():.0f}, max {rl.max():.0f}")
    print("    un compte qui se resout en quelques dizaines de sessions ne traverse")
    print("    pas un episode de regime. Le rapport des deux constantes de temps est")
    print(f"    de {rl.median()/max(base.med_days_bust, 1):.1f} a "
          f"{rl.max()/max(base.med_days_bust, 1):.0f}.")

    PRIMARY = "regime, exposition appariee"

    print("\n" + RULE)
    print("4bis. LE MEME DELTA A CHAQUE BARREAU DE L'ECHELLE DE RISQUE")
    print(RULE)
    print("\n  l'ancre est geometrique, mais elle reste un choix. Voici dp(pass)")
    print("  a tous les niveaux de risque, pour que le lecteur voie si le verdict")
    print("  depend du barreau. L'ancre est marquee d'une etoile.")
    print(f"\n  {'vol ann.':<10}{'constant':>10}{'regime libre':>14}"
          f"{'regime apparie':>16}{'placebo vol':>13}{'cible de vol':>14}")
    rung: dict[float, dict[str, float]] = {}
    for av in DECLARED["risk_ladder_ann_vol"]:
        row = {}
        for name in arms:
            m, _ = evaluate(to_usd(nets[name][idx], cfg.capital, av, book_vol),
                            cfg, 1.0, av)
            row[name] = m.p_pass
        rung[av] = row
        star = " *" if abs(av - chosen) < 1e-12 else "  "
        print(f"  {f'{av:.0%}' + star:<10}{row['taille constante']:>10.1%}"
              + "".join(f"{row[n] - row['taille constante']:>+14.1%}"
                        if i == 0 else f"{row[n] - row['taille constante']:>+16.1%}"
                        if i == 1 else f"{row[n] - row['taille constante']:>+13.1%}"
                        if i == 2 else f"{row[n] - row['taille constante']:>+14.1%}"
                        for i, n in enumerate(list(arms)[1:])))
    wins = sum(1 for av in rung
               if rung[av][PRIMARY] > max(rung[av]["placebo vol appariee (2 seaux)"],
                                          rung[av]["cible de vol, sans parametre"]))
    print(f"\n  barreaux ou le regime apparie battait LES DEUX temoins : "
          f"{wins} sur {len(rung)}")

    obs = deltas[PRIMARY]["dp"]
    obs_d = deltas[PRIMARY]["d_under"]
    obs_bust = deltas[PRIMARY]["d_days_bust"]

    # ---- uncertainty --------------------------------------------------------
    print("\n" + RULE)
    print("5. INCERTITUDE ET EFFET MINIMUM DETECTABLE")
    print(RULE)
    print("\n  deux niveaux, nommes separement parce qu'ils ne mesurent pas la meme")
    print("  chose. (a) bruit de Monte-Carlo : a quel point la simulation fixe dp")
    print("  sur CETTE histoire. (b) incertitude d'histoire : de combien dp")
    print("  bougerait sur un autre echantillon de 24 ans. C'est (b) qui alimente")
    print("  le MDE, et c'est (b) qui est grande.")
    ref_flag = flags["taille constante"].astype(float)
    for name in list(arms)[1:]:
        d = flags[name].astype(float) - ref_flag
        print(f"    {name:<32} dp {d.mean():+.4f}  se Monte-Carlo "
              f"{d.std(ddof=1)/np.sqrt(len(d)):.4f}")

    print(f"\n  bootstrap stationnaire EMBOITE : {DECLARED['n_outer']} pseudo-histoires "
          f"de {n_common:,} sessions,\n  {DECLARED['n_inner']} fenetres de {horizon} "
          "jours tirees dans chacune. Blocs, jamais iid : les")
    print("  passages de barriere sont massivement groupes dans le temps.")
    abs_p = {k: [] for k in arms}
    outer = {k: [] for k in list(arms)[1:]}
    outer_u = {k: [] for k in list(arms)[1:]}
    outer_b = {k: [] for k in list(arms)[1:]}
    ar = np.arange(horizon)
    for _ in range(DECLARED["n_outer"]):
        hist = stationary_indices(n_common, DECLARED["mean_block"], rng)
        st = rng.integers(0, n_common, size=DECLARED["n_inner"])
        wi = hist[(st[:, None] + ar[None, :]) % n_common]
        r0, _ = evaluate(to_usd(nets["taille constante"][wi], cfg.capital, chosen, book_vol),
                         cfg, 1.0, chosen)
        abs_p["taille constante"].append(r0.p_pass)
        for name in outer:
            m, _ = evaluate(to_usd(nets[name][wi], cfg.capital, chosen, book_vol),
                            cfg, 1.0, chosen)
            abs_p[name].append(m.p_pass)
            outer[name].append(m.p_pass - r0.p_pass)
            outer_u[name].append(m.med_underwater - r0.med_underwater)
            outer_b[name].append(m.med_days_bust - r0.med_days_bust)

    print(f"\n  {'bras':<32}{'dp obs.':>9}{'se hist.':>9}{'IC 95%':>20}{'MDE':>8}"
          f"{'verdict':>15}")
    unc: dict[str, dict] = {}
    for name, vals in outer.items():
        a = np.array(vals)
        se = float(a.std(ddof=1))
        mde = Z_MDE * se
        lo, hi = np.percentile(a, [2.5, 97.5])
        o = deltas[name]["dp"]
        v = "PASS" if abs(o) > mde and lo * hi > 0 else (
            "UNDERPOWERED" if abs(o) <= mde else "signe instable")
        unc[name] = {"mean": float(a.mean()), "se": se, "mde": float(mde),
                     "ci95": [float(lo), float(hi)], "observed": o, "verdict": v}
        print(f"  {name:<32}{o:>+9.1%}{se:>9.1%}{f'[{lo:+.1%}, {hi:+.1%}]':>20}"
              f"{mde:>8.1%}{v:>15}")

    print(f"\n  {'bras':<32}{'d sous l eau':>13}{'se':>8}{'MDE (j)':>10}"
          f"{'d j.echec':>11}{'se':>8}{'MDE (j)':>10}")
    unc_dur: dict[str, dict] = {}
    for name in outer_u:
        au, ab = np.array(outer_u[name]), np.array(outer_b[name])
        unc_dur[name] = {
            "under_obs": deltas[name]["d_under"], "under_se": float(au.std(ddof=1)),
            "under_mde": float(Z_MDE * au.std(ddof=1)),
            "bust_obs": deltas[name]["d_days_bust"], "bust_se": float(ab.std(ddof=1)),
            "bust_mde": float(Z_MDE * ab.std(ddof=1))}
        u = unc_dur[name]
        print(f"  {name:<32}{u['under_obs']:>+13.1f}{u['under_se']:>8.1f}"
              f"{u['under_mde']:>10.1f}{u['bust_obs']:>+11.1f}{u['bust_se']:>8.1f}"
              f"{u['bust_mde']:>10.1f}")

    print("\n  TETE A TETE. Le bras de regime ne doit pas battre le livre a taille")
    print("  constante -- il doit battre les deux temoins qui ne savent rien des")
    print("  regimes. C'est le test qui a decide dans docs/EXTENSIONS.md, ou le")
    print("  dispositif passait son placebo et perdait contre une regle sans")
    print("  parametre. Paire sur les memes pseudo-histoires.")
    print(f"\n  {'comparaison':<44}{'dp(pass)':>10}{'se':>8}{'MDE':>8}{'verdict':>15}")
    head: dict[str, dict] = {}
    pa = {k: np.array(v) for k, v in abs_p.items()}
    for ctrl in ("placebo vol appariee (2 seaux)", "cible de vol, sans parametre"):
        d = pa[PRIMARY] - pa[ctrl]
        se = float(d.std(ddof=1))
        o = results[PRIMARY].p_pass - results[ctrl].p_pass
        v = "PASS" if abs(o) > Z_MDE * se else "UNDERPOWERED"
        if o < 0:
            v = "PERD" if abs(o) > Z_MDE * se else "UNDERPOWERED"
        head[ctrl] = {"observed": o, "se": se, "mde": float(Z_MDE * se), "verdict": v}
        print(f"  {'etat moins ' + ctrl:<44}{o:>+10.1%}{se:>8.1%}"
              f"{Z_MDE*se:>8.1%}{v:>15}")

    # ---- null 1 -------------------------------------------------------------
    print("\n" + RULE)
    print("6. NUL 1 -- DIMENSIONNER PAR UN QUANTILE DE VOLATILITE REALISEE")
    print(RULE)
    print("\n  c'est LE placebo qui a tue tous les effets de ce projet. L'etat doit")
    print("  apporter quelque chose AU-DELA de lui, pas seulement au-dela du livre")
    print("  a taille constante. T3 vient de montrer que le gain apparent de ces")
    print("  dispositifs passe par le BETA et le denominateur : une p(pass) qui")
    print("  monte parce que le livre est moins expose n'est pas un resultat, et")
    print("  ce placebo est exactement ce qui le detecte.")
    print(f"  {DECLARED['n_placebo']} dimensionneurs, fenetre dans "
          f"{DECLARED['placebo_windows']}, seuil : {DECLARED['placebo_thresholds']}")
    thr = np.linspace(0.50, 0.95, 40)
    quants = {w: vol_quantile(bk, w) for w in DECLARED["placebo_windows"]}
    n1, n1u, n1b, n1e = [], [], [], []
    r0, _ = evaluate(paths_of(nets["taille constante"]), cfg, 1.0, chosen)
    for _ in range(DECLARED["n_placebo"]):
        w = int(rng.choice(DECLARED["placebo_windows"]))
        th = float(rng.choice(thr))
        lev = normalise(bucket_leverage(bk, quants[w], th), mp)
        net = net_returns(bk, np.nan_to_num(lev, nan=0.0))[sl]
        m, _ = evaluate(paths_of(net), cfg, 1.0, chosen)
        n1.append(m.p_pass - r0.p_pass)
        n1u.append(m.med_underwater - r0.med_underwater)
        n1b.append(m.med_days_bust - r0.med_days_bust)
        n1e.append(float(np.nanmean(lev[sl])))
    a1, a1u, a1b = np.array(n1), np.array(n1u), np.array(n1b)
    p1 = float((a1 < obs).mean())
    p1u = float((a1u > obs_d).mean())
    p1b = float((a1b < obs_bust).mean())
    print(f"\n  nul 1, dp(pass)     : moyenne {a1.mean():+.1%}, ecart-type "
          f"{a1.std(ddof=1):.1%}, p95 {np.percentile(a1, 95):+.1%}")
    print(f"  reel (etat)         : {obs:+.1%}   ->  {p1:.0%}e percentile du nul 1")
    print(f"  nul 1, d sous l eau : moyenne {a1u.mean():+.1f} j, ecart-type "
          f"{a1u.std(ddof=1):.1f}")
    print(f"  reel (etat)         : {obs_d:+.1f} j  ->  {p1u:.0%}e percentile "
          "(cote raccourcissement)")
    print(f"  nul 1, d j.echec    : moyenne {a1b.mean():+.1f} j, ecart-type "
          f"{a1b.std(ddof=1):.1f}")
    print(f"  reel (etat)         : {obs_bust:+.1f} j  ->  {p1b:.0%}e percentile")
    print(f"  exposition moyenne des placebos : {np.mean(n1e):.3f} "
          f"(etat : {np.nanmean(arms[PRIMARY][sl]):.3f}) -- appariee par construction")

    # ---- null 2 -------------------------------------------------------------
    print("\n" + RULE)
    print("7. NUL 2 -- PERMUTER LES ETATS EN CONSERVANT LEUR DUREE")
    print(RULE)
    n_runs = int(np.r_[True, v_states[1:] != v_states[:-1]].sum())
    print(f"\n  l'etat ne change que {n_runs} fois sur {len(states):,} sessions. C'est la")
    print("  taille d'echantillon reelle de toute affirmation conditionnelle ici,")
    print("  pas 6 375 jours. analysis/power.py le dit deja : le nombre d'episodes")
    print("  de stress independants est proche de quatorze.")
    print("  Le nul permute l'ORDRE des episodes et conserve chaque duree a")
    print("  l'identique ; seul l'alignement au calendrier tombe.")
    n2, n2u, n2b = [], [], []
    for _ in range(DECLARED["n_permute"]):
        perm = permute_runs(states, rng)
        lev = normalise(regime_leverage(perm, DECLARED["primary_family"], idx_all), mp)
        net = net_returns(bk, np.nan_to_num(lev, nan=0.0))[sl]
        m, _ = evaluate(paths_of(net), cfg, 1.0, chosen)
        n2.append(m.p_pass - r0.p_pass)
        n2u.append(m.med_underwater - r0.med_underwater)
        n2b.append(m.med_days_bust - r0.med_days_bust)
    a2, a2u, a2b = np.array(n2), np.array(n2u), np.array(n2b)
    p2 = float((a2 < obs).mean())
    p2u = float((a2u > obs_d).mean())
    p2b = float((a2b < obs_bust).mean())
    print(f"\n  nul 2, dp(pass)     : moyenne {a2.mean():+.1%}, ecart-type "
          f"{a2.std(ddof=1):.1%}, p95 {np.percentile(a2, 95):+.1%}")
    print(f"  reel (etat)         : {obs:+.1%}   ->  {p2:.0%}e percentile du nul 2")
    print(f"  nul 2, d sous l eau : moyenne {a2u.mean():+.1f} j, ecart-type "
          f"{a2u.std(ddof=1):.1f}")
    print(f"  reel (etat)         : {obs_d:+.1f} j  ->  {p2u:.0%}e percentile")
    print(f"  nul 2, d j.echec    : moyenne {a2b.mean():+.1f} j, ecart-type "
          f"{a2b.std(ddof=1):.1f}")
    print(f"  reel (etat)         : {obs_bust:+.1f} j  ->  {p2b:.0%}e percentile")

    # ---- null 3 -------------------------------------------------------------
    print("\n" + RULE)
    print("8. NUL 3 -- ROTATION CIRCULAIRE DU PROFIL DE LEVIER")
    print(RULE)
    print("\n  la forme de placebo etablie dans ce depot (controle L58 de")
    print("  docs/EXTENSIONS.md) : l'autocorrelation du profil est preservee")
    print("  EXACTEMENT, seul son alignement aux dates est detruit.")
    lv = arms[PRIMARY][sl]
    n3 = []
    for k in rng.integers(1, n_common, size=DECLARED["n_rotate"]):
        lev = np.full(len(idx_all), np.nan)
        lev[sl] = np.roll(lv, int(k))
        net = net_returns(bk, np.nan_to_num(lev, nan=0.0))[sl]
        m, _ = evaluate(paths_of(net), cfg, 1.0, chosen)
        n3.append(m.p_pass - r0.p_pass)
    a3 = np.array(n3)
    p3 = float((a3 < obs).mean())
    print(f"\n  nul 3, dp(pass)     : moyenne {a3.mean():+.1%}, ecart-type "
          f"{a3.std(ddof=1):.1%}, p95 {np.percentile(a3, 95):+.1%}")
    print(f"  reel (etat)         : {obs:+.1%}   ->  {p3:.0%}e percentile du nul 3")

    # ---- five families ------------------------------------------------------
    print("\n" + RULE)
    print("9. LES CINQ FAMILLES ET LA CORRECTION POUR TESTS MULTIPLES")
    print(RULE)
    print(f"\n  {'famille':<20}{'p(pass)':>9}{'dp':>8}{'j.cible':>9}{'j.echec':>9}"
          f"{'sous l eau':>12}{'expo':>7}")
    panel: dict[str, Metrics] = {}
    for fam in families:
        s = states_all[fam].dropna()
        lev = normalise(regime_leverage(s, fam, idx_all), mp)
        net = net_returns(bk, np.nan_to_num(lev, nan=0.0))[sl]
        m, _ = evaluate(to_usd(net[idx], cfg.capital, chosen, book_vol), cfg,
                        float(np.nanmean(lev[sl])), chosen)
        panel[fam] = m
        print(f"  {fam:<20}{m.p_pass:>9.1%}{m.p_pass-base.p_pass:>+8.1%}"
              f"{m.med_days_pass:>9.0f}{m.med_days_bust:>9.0f}"
              f"{m.med_underwater:>12.0f}{m.mean_exposure:>7.2f}")
    best = max(panel, key=lambda f: panel[f].p_pass)
    obs_max = panel[best].p_pass - base.p_pass
    print(f"\n  meilleure famille : {best}, dp {obs_max:+.1%}")
    print(f"  nul familial : le nul 2 refait {DECLARED['n_family']} fois en prenant a")
    print("  chaque tirage le MAXIMUM de dp sur les cinq familles. Sans cela,")
    print("  cinq essais achetent un percentile.")
    fam_null = []
    for _ in range(DECLARED["n_family"]):
        b = -np.inf
        for fam in families:
            s = states_all[fam].dropna()
            lev = normalise(regime_leverage(permute_runs(s, rng), fam, idx_all), mp)
            net = net_returns(bk, np.nan_to_num(lev, nan=0.0))[sl]
            m, _ = evaluate(paths_of(net), cfg, 1.0, chosen)
            b = max(b, m.p_pass - r0.p_pass)
        fam_null.append(b)
    af = np.array(fam_null)
    pf = float((af < obs_max).mean())
    print(f"\n  nul familial (max sur 5) : moyenne {af.mean():+.1%}, ecart-type "
          f"{af.std(ddof=1):.1%}, p95 {np.percentile(af, 95):+.1%}")
    print(f"  maximum reel             : {obs_max:+.1%}   ->  {pf:.0%}e percentile")

    # ---- FTMO ---------------------------------------------------------------
    print("\n" + RULE)
    print("10. GEOMETRIE SECONDAIRE -- FTMO, PLANCHER STATIQUE (adaptateur local)")
    print(RULE)
    print(f"\n  perte journaliere {FTMO_DAILY_DD_PCT:.0%}, plancher {FTMO_MAX_DD_PCT:.0%} STATIQUE, "
          f"cible {FTMO_PROFIT_TARGET_PCT:.0%}, minimum {FTMO_MIN_TRADING_DAYS} jours.")
    print("  Ce n'est PAS le simulateur du depot : son plancher monte toujours et")
    print("  ne peut pas exprimer un plancher fixe. L'adaptateur fait vingt lignes")
    print("  et n'implemente que les quatre constantes de propfirm_canonical.py.")
    print(f"\n  {'bras':<32}{'p(pass)':>9}{'p(bust)':>9}{'j.cible':>9}{'j.echec':>9}"
          f"{'sous l eau':>12}{'expo':>7}")
    ftmo: dict[str, Metrics] = {}
    for name in arms:
        m = evaluate_ftmo(paths(name), cfg.capital, float(np.nanmean(arms[name][sl])), chosen)
        ftmo[name] = m
        print(f"  {name:<32}{m.p_pass:>9.1%}{m.p_bust:>9.1%}{m.med_days_pass:>9.0f}"
              f"{m.med_days_bust:>9.0f}{m.med_underwater:>12.0f}{m.mean_exposure:>7.2f}")
    fb = ftmo["taille constante"]
    print(f"\n  {'delta contre taille constante':<32}{'dp(pass)':>10}{'d j.cible':>11}"
          f"{'d j.echec':>11}")
    for name, m in ftmo.items():
        if name == "taille constante":
            continue
        print(f"  {name:<32}{m.p_pass-fb.p_pass:>+10.1%}"
              f"{m.med_days_pass-fb.med_days_pass:>+11.0f}"
              f"{m.med_days_bust-fb.med_days_bust:>+11.0f}")

    # ---- historical windows -------------------------------------------------
    print("\n" + RULE)
    print("11. CONTROLE HORS BOOTSTRAP -- LES FENETRES HISTORIQUES REELLES")
    print(RULE)
    print("\n  le bootstrap detruit la chronologie au-dela du bloc. Ici aucun")
    print("  reechantillonnage : toutes les fenetres consecutives de 360 sessions")
    print("  telles qu'elles se sont produites. Pas d'IC valide, elles se")
    print("  recouvrent, mais c'est la seule mesure qui contienne les vraies")
    print("  sequences de 2008 et de 2020.")
    starts = np.arange(0, n_common - horizon)
    hidx = starts[:, None] + ar[None, :]
    print(f"  {len(starts):,} fenetres, {common[0]:%Y-%m} a {common[-1]:%Y-%m}\n")
    print(f"  {'bras':<32}{'p(pass)':>9}{'p(bust)':>9}{'j.cible':>9}{'j.echec':>9}"
          f"{'sous l eau':>12}")
    hist_res: dict[str, Metrics] = {}
    for name in arms:
        m, _ = evaluate(to_usd(nets[name][hidx], cfg.capital, chosen, book_vol),
                        cfg, float(np.nanmean(arms[name][sl])), chosen)
        hist_res[name] = m
        print(f"  {name:<32}{m.p_pass:>9.1%}{m.p_bust:>9.1%}{m.med_days_pass:>9.0f}"
              f"{m.med_days_bust:>9.0f}{m.med_underwater:>12.0f}")
    hb = hist_res["taille constante"]
    print(f"\n  {'delta contre taille constante':<32}{'dp(pass)':>10}{'d j.cible':>11}"
          f"{'d j.echec':>11}{'d sous l eau':>14}")
    for name, m in hist_res.items():
        if name == "taille constante":
            continue
        print(f"  {name:<32}{m.p_pass-hb.p_pass:>+10.1%}"
              f"{m.med_days_pass-hb.med_days_pass:>+11.0f}"
              f"{m.med_days_bust-hb.med_days_bust:>+11.0f}"
              f"{m.med_underwater-hb.med_underwater:>+14.0f}")

    # ---- block sensitivity --------------------------------------------------
    print("\n" + RULE)
    print("12. SENSIBILITE A LA LONGUEUR DE BLOC")
    print(RULE)
    print(f"\n  {'bloc moyen':<12}{'dp etat':>10}{'dp placebo vol':>17}"
          f"{'dp cible de vol':>18}")
    sens = {}
    for mb in DECLARED["block_sensitivity"]:
        r2 = np.random.default_rng(DECLARED["seed"] + mb)
        ix = np.stack([stationary_indices(n_common, mb, r2)[:horizon]
                       for _ in range(DECLARED["n_paths_null"])])
        mc, _ = evaluate(to_usd(nets["taille constante"][ix], cfg.capital, chosen, book_vol),
                         cfg, 1.0, chosen)
        row = []
        for name in (PRIMARY, "placebo vol appariee (2 seaux)", "cible de vol, sans parametre"):
            m, _ = evaluate(to_usd(nets[name][ix], cfg.capital, chosen, book_vol),
                            cfg, 1.0, chosen)
            row.append(m.p_pass - mc.p_pass)
        sens[mb] = row
        print(f"  {mb:<12d}{row[0]:>+10.1%}{row[1]:>+17.1%}{row[2]:>+18.1%}")

    # ---- persist ------------------------------------------------------------
    payload = {
        "declared": DECLARED,
        "sample": {"n_sessions": n_common, "start": str(common[0].date()),
                   "end": str(common[-1].date()), "n_state_runs": n_runs,
                   "share_turbulent": share_hot, "book_vol": book_vol,
                   "cap_hit_rate": cap_hits},
        "costs": {"blended_roundtrip_bps": blended, "annual_drag": drag},
        "notional": {"ann_vol": chosen, "leverage_on_book": chosen / book_vol,
                     "p_pass_max": ladder[top].p_pass, "ann_vol_at_max": top},
        "exposure_elasticity": {"grid": {f"{k:.3f}": v for k, v in elast.items()},
                                "slope_per_unit_exposure": slope},
        "head_to_head": head,
        "ladder": {f"{k:.3f}": asdict(m) for k, m in ladder.items()},
        "ladder_all_arms": {f"{k:.3f}": m for k, m in rung.items()},
        "time_constants": {"med_days_bust": base.med_days_bust,
                           "med_days_pass": base.med_days_pass,
                           "med_regime_episode": float(rl.median()),
                           "mean_regime_episode": float(rl.mean()),
                           "max_regime_episode": float(rl.max())},
        "trailing": {k: asdict(v) for k, v in results.items()},
        "trailing_deltas": deltas,
        "uncertainty_ppass": unc,
        "uncertainty_duration": unc_dur,
        "null1_volquantile": {"n": len(a1), "mean_dp": float(a1.mean()),
                              "sd_dp": float(a1.std(ddof=1)),
                              "p95_dp": float(np.percentile(a1, 95)),
                              "observed_dp": obs, "pct_dp": p1,
                              "mean_d_under": float(a1u.mean()),
                              "sd_d_under": float(a1u.std(ddof=1)),
                              "observed_d_under": obs_d, "pct_d_under": p1u,
                              "mean_d_bust": float(a1b.mean()),
                              "sd_d_bust": float(a1b.std(ddof=1)),
                              "observed_d_bust": obs_bust, "pct_d_bust": p1b,
                              "mean_exposure": float(np.mean(n1e))},
        "null2_permutation": {"n": len(a2), "mean_dp": float(a2.mean()),
                              "sd_dp": float(a2.std(ddof=1)),
                              "p95_dp": float(np.percentile(a2, 95)),
                              "observed_dp": obs, "pct_dp": p2,
                              "mean_d_under": float(a2u.mean()),
                              "sd_d_under": float(a2u.std(ddof=1)),
                              "observed_d_under": obs_d, "pct_d_under": p2u,
                              "mean_d_bust": float(a2b.mean()),
                              "sd_d_bust": float(a2b.std(ddof=1)),
                              "observed_d_bust": obs_bust, "pct_d_bust": p2b},
        "null3_rotation": {"n": len(a3), "mean_dp": float(a3.mean()),
                           "sd_dp": float(a3.std(ddof=1)),
                           "observed_dp": obs, "pct_dp": p3},
        "families": {k: asdict(v) for k, v in panel.items()},
        "family_max_null": {"n": len(af), "mean": float(af.mean()),
                            "sd": float(af.std(ddof=1)), "observed_max": obs_max,
                            "pct": pf, "best_family": best},
        "ftmo": {k: asdict(v) for k, v in ftmo.items()},
        "historical_windows": {k: asdict(v) for k, v in hist_res.items()},
        "block_sensitivity": {str(k): v for k, v in sens.items()},
    }
    (OUT / "barrier_results.json").write_text(json.dumps(payload, indent=2, default=float))
    print("\n" + RULE)
    print(f"chiffres ecrits dans {OUT / 'barrier_results.json'}")
    print(f"duree totale {time.perf_counter() - t_start:.0f} s")
    print(RULE)


if __name__ == "__main__":
    main()
