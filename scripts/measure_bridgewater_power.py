"""Bridgewater study — the power of the declared tests, measured blind before the lock.

The draft pre-registration (`pilotage/plans_de_recherche/bridgewater/PRESPEC_BRIDGEWATER.md`,
to become `docs/PRESPEC_BRIDGEWATER.md`) states its power in §4 from formulas on the
whole panel. This script re-derives those figures, then measures the declared
statistics on the declared object:

- **level A**: the reduction in ``D = log var(worst cell) - log var(best cell)``, balanced
  leg against blind leg, on the pooled test folds of an expanding walk-forward
  (`evaluate.stamp_folds`), on the instrument-level book of §2
  (`evaluate.InstrumentLegBuilder`: 10% target, cap 3, headline costs, ETFs funded);
- its null under **P1**: content-free partitions with the real quarterly clock,
  occupancy and episodes (`analysis.placebo`, ``method="uniform"``). From it come the
  95th and 99th percentiles, the MDE, and the power against the draft's alternatives
  ``log 1.5 = 0.405`` and ``log 1.2 = 0.182``;
- the same null for the **B1** candidate axes and for the **volatility witness**;
- **B2** and **level C**: the null of B2's D reduction, P3's size on content-free
  partitions, and the MDE of their paired Sharpe differences on DEMEANED legs
  (`selection.protocol.blinded_mde`).

**Blind.** Nothing here is conditioned on the real quadrant labels or the real B1 labels:
- The real labels enter as label-only statistics (counts, clock, fold cuts, cell
  qualification counts) and as the template the placebo draws copy.
- `LevelA.run` sees placebo paths only. A guard refuses the real quadrant path and
  every real B1 path.
- The witness's own reduction is not computed: its value is the comparator of the
  declared test, so reading it before the lock would inform the lock.
- Level C runs on the transition dates of placebo draws, never on the real ones.
- No mean return is printed. Sharpe-type quantities go through `blinded_mde`, which
  demeans both legs and refuses any mean that reaches it.

Usage:
    .venv/bin/python scripts/measure_bridgewater_power.py
    .venv/bin/python scripts/measure_bridgewater_power.py --draws 2000 --sens-draws 1000 \
        --json docs/artifacts/bridgewater/power.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from regime_lab.config import CACHE, RAW, ROOT
from regime_lab.construction import evaluate as E
from regime_lab.construction import quadrant as Q
from regime_lab.construction import sleeves as S
from regime_lab.data.pit import validate
from regime_lab.selection.protocol import blinded_mde, mde_at, standalone_sharpe_threshold

SPF = RAW / "spf"
MACRO = RAW / "macro"
H3_RAW = ROOT / "chantiers" / "macro-momentum" / "data" / "raw" / "h3_macro"
PERIODS = 252
DAYS_PER_YEAR = 365.25
SEED = 20260924

SIDAK = E.sidak_alpha()
#: Holm over the five declared tests: the k-th smallest p is compared with α/(5-k+1).
HOLM = tuple(E.FAMILY_ALPHA / k for k in range(E.DECLARED_TESTS, 0, -1))
#: §4's alternatives on D: 3:1 -> 2:1 and 3:1 -> 2.5:1 in extreme-cell variance.
ALTERNATIVES = {"log 1.5": float(np.log(1.5)), "log 1.2": float(np.log(1.2))}
Z80 = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)

RULE = "=" * 100


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


# ------------------------------------------------------------------ reproduction log


@dataclass
class Check:
    section: str
    quantity: str
    disclosed: str
    measured: str
    match: bool
    declared: str = ""


CHECKS: list[Check] = []


def check(where: str, quantity: str, disclosed: float | str, measured: float | str, *,
          fmt: str = "{:.3f}", tol: float = 0.0, declared: str = "") -> None:
    """Record a disclosed figure, its re-measurement and, apart, the declared object's."""
    if isinstance(disclosed, str) or isinstance(measured, str):
        ok = str(disclosed) == str(measured)
        d, m = str(disclosed), str(measured)
    else:
        both_inf = np.isinf(disclosed) and np.isinf(measured)
        ok = both_inf or abs(float(disclosed) - float(measured)) <= tol + 1e-12
        d, m = fmt.format(disclosed), fmt.format(measured)
    CHECKS.append(Check(where, quantity, d, m, bool(ok), declared))


# ------------------------------------------------------------------------ the guard


class GuardedLevelA(E.LevelA):
    """`LevelA` that refuses to run on any path registered as a real label path."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forbidden: list[np.ndarray] = []
        self.runs = 0

    def forbid(self, path: pd.Series) -> None:
        self.forbidden.append(path.reindex(self.sessions).to_numpy(float))

    def run(self, labels, *, weights=None):
        paths = labels.values() if isinstance(labels, Mapping) else [labels]
        for path in paths:
            values = path.reindex(self.sessions).to_numpy(float)
            if any(np.array_equal(values, bad, equal_nan=True) for bad in self.forbidden):
                raise RuntimeError("blindness guard: a real label path reached LevelA.run")
        self.runs += 1
        return super().run(labels, weights=weights)


# ---------------------------------------------------------------------- null summaries


def pct_rank(null: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Percentile of each value in ``null``, ties counted half (`placebo_percentile`)."""
    ordered = np.sort(null)
    below = np.searchsorted(ordered, values, side="left")
    equal = np.searchsorted(ordered, values, side="right") - below
    return (below + 0.5 * equal) / ordered.size


def power_shift(null: np.ndarray, shift: float, alpha: float, floor: float | None = None
                ) -> float:
    """Share of null draws that, shifted by ``shift``, reach the ``1 - α`` percentile rule
    (and ``floor``, if given): the location-shift power of the Two Sigma convention."""
    x = null + shift
    ok = pct_rank(null, x) >= 1.0 - alpha
    if floor is not None:
        ok &= x >= floor
    return float(ok.mean())


def summarise(name: str, null: E.NullDistribution) -> dict:
    """Shape, thresholds, MDE and power of one null of a reduction in D."""
    v = null.values
    out = {"name": name, **null.describe()}
    out["q_sidak"] = float(null.quantile(1.0 - SIDAK)) if null.finite else float("nan")
    out["holm_quantiles"] = [float(null.quantile(1.0 - a)) if null.finite else float("nan")
                             for a in HOLM]
    out["share_negative"] = float((v < 0).mean())
    out["share_zero"] = float((v == 0).mean())
    out["power"] = {}
    if null.finite:
        for label, shift in {**ALTERNATIVES, "0.204": E.DRAFT_MDE}.items():
            out["power"][label] = {
                "alpha_05": power_shift(v, shift, 0.05),
                "sidak": power_shift(v, shift, SIDAK),
                "alpha_05_and_floor_0204": power_shift(v, shift, 0.05, E.DRAFT_MDE),
            }
    return out


def print_null(s: dict, *, header: bool = False) -> None:
    if header:
        print(f"  {'null':44} {'n':>5} {'mean':>7} {'sd':>6} {'skew':>6} {'atom':>5} "
              f"{'q50':>7} {'q95':>7} {'q99':>7} {'qSid':>7} {'MDEs05':>7} {'MDEsSid':>7} "
              f"{'MDEsd05':>7} {'P.405':>6} {'P.182':>6} {'P.405S':>6} {'P.182S':>6}")
    p = s["power"]

    def fmt(x: float) -> str:
        return f"{x:7.3f}"

    print(f"  {s['name']:44} {int(s['draws']):5d} {fmt(s['mean'])} {s['sd']:6.3f} "
          f"{s['skewness']:6.2f} {s['atom_share']:5.3f} {fmt(s['q50'])} {fmt(s['q95'])} "
          f"{fmt(s['q99'])} {fmt(s['q_sidak'])} {fmt(s['mde_shift_05'])} "
          f"{fmt(s['mde_shift_sidak'])} {fmt(s['mde_sd_05'])} "
          f"{p['log 1.5']['alpha_05']:6.3f} {p['log 1.2']['alpha_05']:6.3f} "
          f"{p['log 1.5']['sidak']:6.3f} {p['log 1.2']['sidak']:6.3f}")


NULL_LEGEND = (
    "  mean/q50/q95/q99: the reduction D(blind) - D(balanced) under content-free partitions "
    "(positive = the\n  balanced leg narrows D). qSid: the 1 - 0.0102 quantile (Sidak over 5). "
    "MDEs: location-shift MDE\n  q(1-a) - q(0.20); MDEsd: (z_(1-a) + z_0.80) x sd. P.405 / P.182: "
    "share of draws that, shifted by\n  log 1.5 / log 1.2, reach the 95th percentile (S: the "
    "Sidak percentile)."
)


# --------------------------------------------------------------------------- inputs


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs_section() -> dict:
    section("0. Inputs: file inventory of §1.1 / §1.4, and the SHA-256 of every input read")
    files = [SPF / name for name, _ in Q.SPF_LEVEL_FILES.values()] + [
        SPF / Q.SPF_GROWTH_FILE[0], SPF / Q.RELEASE_DATES_CSV, SPF / Q.T10YIE_FILE,
        CACHE / "trend_universe_m1.parquet", MACRO / "rate_cash_3m.parquet",
        MACRO / "fin_baa_spread.parquet",
    ]
    missing = [str(p.relative_to(ROOT)) for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"missing inputs {missing}; run scripts/fetch_spf.py and the data "
                         "fetches of AVANCEMENT.md first")
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    for path, digest in hashes.items():
        print(f"  {path:45} {digest[:16]}")

    print("\n  §1.1 SPF median files (rows x columns as pandas opens them, survey span):")
    shapes = {"rgdp": "232 x 12", "cpi": "232 x 11", "unemp": "232 x 12", "indprod": "232 x 10"}
    for key, (name, required) in Q.SPF_LEVEL_FILES.items():
        frame = Q.verify_workbook((SPF / name).read_bytes(), required=required, label=name)
        table = Q.spf_table(frame, label=name)
        shape = f"{frame.shape[0]} x {frame.shape[1]}"
        span = f"{table.index[0]}-{table.index[-1]}"
        print(f"    {name:28} {shape:>9}  {span}")
        check("§1.1", f"{name} shape", shapes[key], shape)
        check("§1.1", f"{name} coverage", "1968Q4-2026Q3", span)

    print("\n  §1.4 on-disk point-in-time series:")
    disclosed = {
        "rate_cash_3m": (9177, "1990-01-02", "2026-09-08"),
        "fin_baa_spread": (9171, "1990-01-02", "2026-09-04"),
        "fin_aaa_spread": (9171, "1990-01-02", "2026-09-04"),
        "macro_cpi": (438, "1990-01-01", "2026-07-01"),
        "macro_indpro": (439, "1990-01-01", "2026-07-01"),
    }
    for name, (rows, first, last) in disclosed.items():
        path = MACRO / f"{name}.parquet"
        if not path.exists():
            print(f"    {name:18} absent")
            continue
        frame = validate(pd.read_parquet(path))
        a, b = f"{frame['period'].min():%Y-%m-%d}", f"{frame['period'].max():%Y-%m-%d}"
        periods = frame["period"].nunique()
        print(f"    {name:18} {len(frame):>6,} rows {periods:>6,} periods  {a} -> {b}")
        check("§1.4", f"{name} rows", rows, len(frame), fmt="{:.0f}")
        check("§1.4", f"{name} span", f"{first} -> {last}", f"{a} -> {b}")
    ip = validate(pd.read_parquet(MACRO / "macro_indpro.parquet")).set_index("period")["value"]
    feb, mar = float(ip.loc["1990-02-01"]), float(ip.loc["1990-03-01"])
    print(f"    macro_indpro 1990-02 -> 1990-03: {feb:.1f} -> {mar:.1f}")
    check("§1.4", "macro_indpro Feb->Mar 1990", "141.8 -> 108.8", f"{feb:.1f} -> {mar:.1f}")
    payems = H3_RAW / "payems.parquet"
    if payems.exists():
        frame = pd.read_parquet(payems)
        print(f"    h3_macro/payems    {len(frame):>6,} rows {frame['period'].nunique():>6,} "
              "periods (vintages)")
        check("§1.4", "h3 payems rows / periods", "4560 / 344",
              f"{len(frame)} / {frame['period'].nunique()}")
    t10 = validate(pd.read_parquet(SPF / Q.T10YIE_FILE))
    upto = t10.loc[t10["period"] <= "2026-09-21"]
    print(f"    fred_t10yie        {len(t10):>6,} valued rows {t10['period'].min():%Y-%m-%d} -> "
          f"{t10['period'].max():%Y-%m-%d}; {len(upto):,} up to 2026-09-21")
    check("§1.4", "T10YIE first date", "2003-01-02", f"{t10['period'].min():%Y-%m-%d}")
    return hashes


# ---------------------------------------------------------------- §3-§5 arithmetic


def draft_arithmetic() -> dict:
    section("1. The draft's §3-§5 arithmetic, re-derived (formulas only, no data)")
    out: dict = {}
    print("  §3 Sidak table (family 0.05):")
    for k, (a_d, z_d) in {1: (0.05000, 1.960), 3: (0.01695, 2.388), 5: (0.01021, 2.569),
                          6: (0.00851, 2.631)}.items():
        a = E.sidak_alpha(k)
        z = stats.norm.ppf(1.0 - a / 2.0)
        print(f"    {k} tests: alpha {a:.5f}  two-sided z {z:.3f}")
        check("§3", f"Sidak alpha, {k} tests", a_d, round(a, 5), fmt="{:.5f}")
        check("§3", f"Sidak z, {k} tests", z_d, round(z, 3), fmt="{:.3f}")
    print(f"  Holm over 5 (not in the draft): {', '.join(f'{a:.4f}' for a in HOLM)}")
    out["holm"] = list(HOLM)

    print("\n  §4 standalone resolvable Sharpe (Lo, 80% power, two-sided 5%):")
    rows = {"5-sleeve panel 23.60 yr": (23.60, 0.632), "cell (g-,i+) 1792 s": (1792 / 252, 1.569),
            "cell (g+,i+) 1675 s": (1675 / 252, 1.698), "cell (g+,i-) 1630 s": (1630 / 252, 1.757),
            "cell (g-,i-) 841 s": (841 / 252, float("inf"))}
    for name, (years, disclosed) in rows.items():
        sr = standalone_sharpe_threshold(years)
        print(f"    {name:26} {years:6.2f} yr -> {sr:6.3f}")
        check("§4", f"resolvable SR, {name}", disclosed, round(sr, 3) if np.isfinite(sr) else sr)
    zero_room = Z80 ** 2 / 2
    print(f"    denominator positive above z^2/2 = {zero_room:.3f} years")
    check("§4", "years for a positive denominator", 3.92, round(zero_room, 2), fmt="{:.2f}")

    print("\n  §4 paired table, MDE = 2.8016 sqrt(2(1-rho)) SE, SE = sqrt((1 + 0.5^2/2) / years):")
    paired = {"full panel 23.6 yr": (23.6, 0.2183, (0.274, 0.193, 0.122, 0.087)),
              "one fold of five 4.7 yr": (23.6 / 5, 0.4882, (0.612, 0.433, 0.274, 0.193)),
              "largest cell 7.1 yr": (1792 / 252, 0.3977, (0.498, 0.352, 0.223, 0.158)),
              "smallest cell 3.3 yr": (841 / 252, 0.5806, (0.727, 0.514, 0.325, 0.230))}
    for name, (years, se_d, mdes_d) in paired.items():
        se = float(np.sqrt((1.0 + 0.25 / 2.0) / years))
        mdes = [Z80 * np.sqrt(2 * (1 - rho)) * se for rho in (0.90, 0.95, 0.98, 0.99)]
        print(f"    {name:26} SE {se:.4f}  " + "  ".join(f"{m:.3f}" for m in mdes))
        check("§4", f"paired SE, {name}", se_d, round(se, 4), fmt="{:.4f}")
        for rho, m, d in zip((0.90, 0.95, 0.98, 0.99), mdes, mdes_d, strict=True):
            check("§4", f"paired MDE rho={rho}, {name}", d, round(m, 3))

    print("\n  §4 variance statistic, SE(log sigma) = 1/sqrt(2 n_eff), n_eff = n/63:")
    for n, (neff_d, mde_d) in {5938: (94.3, 0.204), 1792: (28.4, 0.371),
                               841: (13.3, 0.542)}.items():
        neff = n / 63.0
        mde = E.draft_mde_log_sigma(n)
        print(f"    {n:5d} sessions: n_eff {neff:5.1f}  MDE {mde:.3f}   (the unpaired two-cell "
              f"log-VARIANCE difference against a cell of equal size: "
              f"{E.analytic_mde_log_variance_difference(n, n):.3f})")
        check("§4", f"variance n_eff, {n} sessions", neff_d, round(neff, 1), fmt="{:.1f}")
        check("§4", f"variance MDE, {n} sessions", mde_d, round(mde, 3))
    grow = np.exp(E.DRAFT_MDE) - 1
    print(f"    exp(0.204) - 1 = {grow:.1%}; log 1.5 = {np.log(1.5):.3f} "
          f"({np.log(1.5) / E.DRAFT_MDE:.2f} x 0.204); log 1.2 = {np.log(1.2):.3f}")
    check("§4", "reduction threshold in %", 22.6, round(100 * grow, 1), fmt="{:.1f}")
    check("§4", "log 1.5", 0.405, round(np.log(1.5), 3))
    check("§4", "log 1.2", 0.182, round(np.log(1.2), 3))
    two = E.analytic_mde_log_variance_difference(1792, 841)
    print(f"    the unpaired two-cell MDE, largest vs smallest cell (1792 vs 841): {two:.3f}")
    out["two_cell_mde"] = two

    print("\n  §5 cost table at the draft's 1.217 bp, Sharpe at a 10% target:")
    blend = (1 * 1.0 + 13 * 1.5 + 12 * 1.0 + 4 * 1.0) / 30
    check("§5", "blended bp (draft formula)", 1.217, round(blend, 3))
    for extra, (drag_d, sr_d) in {2: (0.024, 0.0024), 4: (0.049, 0.0049), 8: (0.097, 0.0097),
                                  16: (0.195, 0.0195)}.items():
        drag = extra * blend / 1e4
        print(f"    +{extra:2d} round trips -> {drag:.3%}  {drag / 0.10:.4f} Sharpe")
        check("§5", f"+{extra} trips drag %", drag_d, round(100 * drag, 3))
        check("§5", f"+{extra} trips Sharpe", sr_d, round(drag / 0.10, 4), fmt="{:.4f}")
    kill, kill_cash = 0.10 * 0.10 * 1e4 / blend, 0.10 * 0.10 * 1e4 / 7.5
    print(f"    kill at 0.10 Sharpe: {kill:.1f} round trips at 1.217 bp; {kill_cash:.1f} at 7.5 bp")
    check("§5", "kill round trips, headline", 82, round(kill), fmt="{:.0f}")
    check("§5", "kill round trips, all_cash", 13.3, round(kill_cash, 1), fmt="{:.1f}")
    return out


# ---------------------------------------------------------------- the declared object


@dataclass
class Declared:
    sessions: pd.DatetimeIndex
    excess: pd.DataFrame
    within: pd.DataFrame
    sleeve_r: pd.DataFrame
    usable: pd.DatetimeIndex
    stamped: pd.Series
    stamped_all: pd.DataFrame
    panel: pd.DataFrame
    release: pd.DataFrame
    folds: tuple
    engine: GuardedLevelA
    real_path: pd.Series
    builder: E.InstrumentLegBuilder

    @property
    def test(self) -> pd.DatetimeIndex:
        return self.engine.test_sessions


def stamped_series(quarterly: pd.DataFrame) -> pd.Series:
    table = quarterly.sort_values("available_at").drop_duplicates("available_at", keep="last")
    return pd.Series(table["label"].to_numpy(float),
                     index=pd.DatetimeIndex(table["available_at"]).astype("datetime64[ns]"),
                     name="quadrant")


def build_declared() -> Declared:
    prices = S.load_prices(S.HEADLINE_SLEEVES)
    start = S.configuration_start(prices, S.HEADLINE_SLEEVES)
    cash = S.cash_rate(prices.index)
    excess = S.excess_returns(S.price_returns(prices), cash).loc[start:]
    sessions = pd.DatetimeIndex(excess.index)
    rgdp = Q.read_spf(SPF / "median_rgdp_level.xlsx", ("RGDP1", "RGDP2"))
    cpi = Q.read_spf(SPF / "median_cpi_level.xlsx", ("CPI1", "CPI2"))
    release = Q.read_release_dates(SPF / Q.RELEASE_DATES_CSV)
    panel = Q.surprise_panel(rgdp, cpi)
    quarterly = Q.quarterly_labels(
        panel, Q.availability(panel.index, rule="release", release_dates=release))
    stamped = stamped_series(quarterly)
    dates = [d for d in stamped.index if sessions[0] <= d <= sessions[-1]]
    within = S.hold(S.within_sleeve_weights(excess, S.HEADLINE_SLEEVES, dates), sessions)
    sleeve_r = S.sleeve_returns(excess, within, S.HEADLINE_SLEEVES)
    usable = sleeve_r.index[sleeve_r.notna().all(axis=1)]
    folds = E.stamp_folds(stamped, sessions, usable=usable, min_quarters=4)
    builder = E.InstrumentLegBuilder(excess, within, S.HEADLINE_SLEEVES)
    engine = GuardedLevelA(sleeve_r, folds, leg_builder=builder)
    real_path = E.expand_to_sessions(stamped, sessions, lag=1)
    daily = Q.daily_labels(quarterly, sessions, lag=1).astype(float)
    if not np.array_equal(daily.to_numpy(), real_path.to_numpy(), equal_nan=True):
        raise RuntimeError("the evaluation expansion and quadrant.daily_labels disagree")
    engine.forbid(real_path)
    return Declared(sessions, excess, within, sleeve_r, usable, stamped, quarterly, panel,
                    release, folds, engine, real_path, builder)


def codes_of(path: pd.Series | np.ndarray) -> np.ndarray:
    """Integer cell codes of a label path, -1 where there is no label (label-only)."""
    values = np.asarray(path, dtype=float)
    return np.where(np.isfinite(values), np.nan_to_num(values, nan=-1.0), -1.0).astype(np.int64)


def label_counts(codes: np.ndarray) -> tuple[list[int], list[int]]:
    sessions, episodes = E.cell_counts(codes)
    return sessions.tolist(), episodes.tolist()


def quarters_by_cell(stamped: pd.Series, sessions: pd.DatetimeIndex,
                     window: pd.DatetimeIndex) -> list[int]:
    """Quarters of each cell whose label is in force on at least one session of ``window``."""
    in_force = E.stamps_in_force(stamped, sessions)
    position = E.expand_to_sessions(pd.Series(np.arange(len(in_force)), index=in_force.index),
                                    sessions)
    held = np.unique(position.reindex(window).dropna().to_numpy()).astype(int)
    counts = in_force.iloc[held].value_counts()
    return [int(counts.get(float(k), 0)) for k in range(E.N_CELLS)]


def declared_section(d: Declared) -> dict:
    section("2. The declared object of level A (label COUNTS and state-blind books only)")
    s = d.sessions
    print(f"  headline sample {s[0]:%Y-%m-%d} -> {s[-1]:%Y-%m-%d}: {len(s):,} sessions")
    print(f"  within-sleeve inverse volatility on trailing 252 sessions at each of the "
          f"{d.within.dropna(how='all').drop_duplicates().shape[0]} stamps with a full window; "
          f"sleeve returns from {d.usable[0]:%Y-%m-%d}")
    print("  blind leg: ERC on every finite training session of the fold (expanding); balanced "
          "leg: R1 equal\n  cell variance on training cells (63 sessions, 3 episodes); both "
          "constant within a fold;\n  book: instrument weights, 10% target on 63 sessions "
          "lagged one, cap 3, headline costs on\n  held weights, ETFs funded; D read on the "
          "pooled test sessions, each leg's own extremes, centred.")
    print("\n  walk-forward (evaluate.stamp_folds, min 4 quarters per cell before the first test):")
    print(f"  {'fold':>4} {'train end':>10} {'test':>23} {'sess':>5} {'q':>3} "
          f"{'train q per cell':>18} {'test q per cell':>16} {'test sessions per cell':>24}")
    rows = []
    codes = codes_of(d.real_path)  # label-only
    for f in d.folds:
        test, train = f.test(s), f.train(s)
        tq = quarters_by_cell(d.stamped, s, test)
        trq = quarters_by_cell(d.stamped, s, train[train >= d.usable[0]])
        ts, _ = label_counts(codes[s.isin(test)])
        rows.append({"fold": f.number, "train_end": str(f.train_end.date()),
                     "test_start": str(f.test_start.date()), "test_end": str(f.test_end.date()),
                     "test_sessions": len(test), "test_quarters": sum(tq),
                     "train_quarters_by_cell": trq, "test_quarters_by_cell": tq,
                     "test_sessions_by_cell": ts})
        print(f"  {f.number:4d} {f.train_end:%Y-%m-%d} {f.test_start:%Y-%m-%d} -> "
              f"{f.test_end:%Y-%m-%d} {len(test):5d} {sum(tq):3d} {str(trq):>18} {str(tq):>16} "
              f"{str(ts):>24}")
    test = d.test
    years_s, years_c = len(test) / PERIODS, (test[-1] - test[0]).days / DAYS_PER_YEAR
    print(f"  pooled test: {len(test):,} sessions, {years_s:.2f} session-years, "
          f"{years_c:.2f} calendar years ({test[0]:%Y-%m-%d} -> {test[-1]:%Y-%m-%d})")
    check("§1.3/§7", "sessions per fold", 1187, len(test) / 5, fmt="{:.0f}", tol=0.5,
          declared=f"{len(test)} test sessions, folds {[r['test_sessions'] for r in rows]}")

    print("\n  real label on the pooled test sessions (label-only), sessions / episodes per cell:")
    ses, epi = label_counts(codes[s.isin(test)])
    print(f"    sessions {ses}  episodes {epi}")
    for k, n in enumerate(ses):
        sr = standalone_sharpe_threshold(n / PERIODS)
        print(f"    {Q.CELLS[k]:24} {n:5d} sessions = {n / PERIODS:5.2f} yr -> resolvable "
              f"SR {sr:6.3f} (the §3 A secondary statistic, Lo, one leg)")
    print("\n  real label on each fold's TRAINING path (label-only): the cell-qualification counts")
    print("  of OC-A8 (>= 63 sessions and >= 3 episodes), sessions / episodes per cell:")
    qualify = []
    for f in d.folds:
        train = s.isin(f.train(s)) & s.isin(d.usable)
        sn, ep = label_counts(codes[train])
        ok = sum(a >= E.MIN_CELL_SESSIONS and b >= E.MIN_CELL_EPISODES
                 for a, b in zip(sn, ep, strict=True))
        qualify.append(ok)
        print(f"    fold {f.number}: sessions {sn}  episodes {ep}  cells qualifying {ok} of 4")

    blind = d.engine.blind_leg
    held = blind.held.reindex(test)
    sd = float(blind.returns.reindex(test).std(ddof=1) * np.sqrt(PERIODS))
    print("\n  the engine's blind leg (state-blind), on the pooled test sessions:")
    print("    sleeve weights per fold:")
    print("    " + d.engine.blind_weights.round(3).to_string().replace("\n", "\n    "))
    turnover_blind = S.annual_turnover(held)
    cap = float(blind.cap_binds.reindex(test).mean())
    gross = held.abs().sum(axis=1)
    print(f"    realised sd {sd:.2%}; cap binds {cap:.1%}; gross median {gross.median():.2f} "
          f"p95 {gross.quantile(0.95):.2f}; held turnover {turnover_blind:.2f}x/yr; finite on "
          f"{int(blind.returns.reindex(test).notna().sum())} of {len(test)} test sessions")
    # The sibling module's declared blind book re-estimates every stamp (OC-A7).
    dates = [t for t in d.stamped.index if s[0] <= t <= s[-1]]
    book, _, _ = S.blind_book(d.excess, S.HEADLINE_SLEEVES, dates)
    rolling_sd = float(book.returns.reindex(test).std(ddof=1) * np.sqrt(PERIODS))
    rolling_to = S.annual_turnover(book.weights.reindex(test))
    both = pd.concat([blind.returns.reindex(test), book.returns.reindex(test)], axis=1).dropna()
    corr = float(both.corr().iloc[0, 1])
    print(f"    OC-A7: the sleeves module's blind book (ERC re-estimated at every stamp on 252 "
          f"sessions)\n    realises {rolling_sd:.2%} with {rolling_to:.2f}x/yr on the same "
          f"sessions; correlation with the engine's blind leg {corr:.3f}")
    return {"folds": rows, "pooled_test_sessions": len(test), "years_sessions": years_s,
            "years_calendar": years_c, "test_cell_sessions": ses, "test_cell_episodes": epi,
            "train_cells_qualifying": qualify,
            "blind_weights": d.engine.blind_weights.round(6).to_dict(orient="index"),
            "blind_sd": sd, "blind_cap_share": cap, "blind_turnover": turnover_blind,
            "rolling_blind_sd": rolling_sd, "rolling_blind_turnover": rolling_to,
            "rolling_blind_corr": corr}


# ------------------------------------------------------------------------------- nulls


def statistics_for(d: Declared) -> dict[str, Callable[[E.LevelAResult], float]]:
    """What each placebo draw records: D, its legs, fallbacks, exposure, turnover."""

    def test_mean(series: pd.Series, idx: pd.DatetimeIndex) -> float:
        return float(series.reindex(idx).mean())

    return {
        "reduction": lambda r: r.reduction,
        "d_blind": lambda r: r.d_blind,
        "d_balanced": lambda r: r.d_balanced,
        "fallback_folds": lambda r: float(len(r.fallback_folds)),
        "unconverged_folds": lambda r: float(sum(not f.converged for f in (r.fits or ()))),
        "balanced_sd": lambda r: float(r.balanced.std(ddof=1) * np.sqrt(PERIODS)),
        "balanced_cap_share": lambda r: test_mean(r.balanced_leg.cap_binds, r.test_labels.index),
        "balanced_gross": lambda r: test_mean(r.balanced_leg.held.abs().sum(axis=1),
                                              r.test_labels.index),
        "extra_turnover": lambda r: (
            S.annual_turnover(r.balanced_leg.held.reindex(r.test_labels.index))
            - S.annual_turnover(r.blind_leg.held.reindex(r.test_labels.index))),
        "p2_gross": lambda r: E.exposure_matched(r, d.excess, on="gross").reduction,
        "weight_l1": lambda r: float(
            (r.balanced_weights - r.blind_weights).abs().sum(axis=1).mean()),
        "zero_sleeves": lambda r: float((r.balanced_weights < 1e-6).sum(axis=1).mean()),
        **{f"w_{sleeve}": (lambda r, sleeve=sleeve: float(r.balanced_weights[sleeve].mean()))
           for sleeve in d.sleeve_r.columns},
        **{f"zero_{sleeve}": (
            lambda r, sleeve=sleeve: float((r.balanced_weights[sleeve] < 1e-6).mean()))
           for sleeve in d.sleeve_r.columns},
    }


def run_null(engine: GuardedLevelA, labels: pd.Series, n: int, *, seed: int,
             blocks: pd.Series | None = None, stamped: bool = True,
             statistics: dict | None = None) -> dict[str, E.NullDistribution]:
    stats_ = statistics or {"reduction": lambda r: r.reduction}
    return E.null_statistics(engine, labels, n, statistics=stats_, seed=seed, blocks=blocks,
                             stamped=stamped)


#: Recorded on every B1 and witness draw: a draw whose balanced leg falls back to blind in
#: a fold reads a reduction of zero on that fold's sessions, which is a different object.
WITH_FALLBACK = {"reduction": lambda r: r.reduction,
                 "fallback_folds": lambda r: float(len(r.fallback_folds))}


def qualifying_cells(d: Declared, paths: pd.Series | Mapping[int, pd.Series]) -> list[int]:
    """Label-only: cells of each fold's TRAINING path meeting OC-A8 (63 sessions, 3 episodes)."""
    s = d.sessions
    out = []
    for f in d.folds:
        path = paths[f.number] if isinstance(paths, Mapping) else paths
        codes = codes_of(path.reindex(s).to_numpy(float))
        sn, ep = label_counts(codes[s.isin(f.train(s)) & s.isin(d.usable)])
        out.append(int(sum(a >= E.MIN_CELL_SESSIONS and b >= E.MIN_CELL_EPISODES
                           for a, b in zip(sn, ep, strict=True))))
    return out


def fallback_summary(s: dict, null: dict[str, E.NullDistribution]) -> None:
    fb = null["fallback_folds"].values
    s["fallback_mean"] = float(fb.mean())
    s["fallback_any"] = float(np.mean(fb > 0))


def draws_against_real(stamps: pd.Series, n: int, seed: int,
                       blocks: pd.Series | None = None) -> dict:
    """Label-only hygiene: no draw is the real sequence, and how far each is from it."""
    draws = E.p1_draws(stamps, n, seed=seed, blocks=blocks).draws
    agree = (draws.to_numpy() == stamps.to_numpy()[:, None]).mean(axis=0)
    return {"identical": int((agree == 1.0).sum()), "agreement_max": float(agree.max()),
            "agreement_mean": float(agree.mean())}


def primary_section(d: Declared, n: int) -> tuple[dict, dict]:
    section(f"3. The null of the level-A reduction under P1 (primary, {n} draws)")
    stamps = E.stamps_in_force(d.stamped, d.sessions)
    feas = E.p1_feasibility(stamps)
    blocks = E.calendar_blocks(stamps.index, d.sessions, d.folds)
    feas_b = E.p1_feasibility(stamps, blocks)
    print(f"  P1 on the {len(stamps)} stamps in force: {int(feas['transitions'].iloc[0])} "
          f"transitions, log10 of join-free orders {feas['log10_orders'].iloc[0]:.1f}; per "
          f"calendar block (OC-P1a) the freedom runs {feas_b['log10_orders'].min():.1f} to "
          f"{feas_b['log10_orders'].max():.1f}")
    print("    " + feas_b.to_string().replace("\n", "\n    "))
    hygiene = draws_against_real(stamps, n, SEED)
    print(f"  draws identical to the real sequence: {hygiene['identical']}; share of stamps "
          f"agreeing with it: mean {hygiene['agreement_mean']:.3f}, max "
          f"{hygiene['agreement_max']:.3f} (independent draws at this occupancy: 0.266)")
    paths = E.placebo_session_paths(d.engine, d.stamped, min(n, 200), seed=SEED)
    dev = E.session_occupancy_deviation(d.real_path, paths)
    print(f"  session-level occupancy: largest deviation from the real one over 200 draws "
          f"{dev:.3f} (quarterly occupancy, clock and episodes are exact)")

    t0 = time.time()
    null = run_null(d.engine, d.stamped, n, seed=SEED, statistics=statistics_for(d))
    elapsed = time.time() - t0
    red = null["reduction"]
    summary = summarise("level A, declared object", red)
    print(f"\n  {n} draws in {elapsed:.0f} s; every draw finite: {red.finite}")
    print(NULL_LEGEND)
    print_null(summary, header=True)
    print("\n  the legs under the null (content-free):")
    for key in ("d_blind", "d_balanced"):
        v = null[key].values
        print(f"    {key:12} mean {v.mean():.3f}  sd {v.std(ddof=1):.3f}  q05 "
              f"{np.quantile(v, 0.05):.3f}  q95 {np.quantile(v, 0.95):.3f}")
    fb = null["fallback_folds"].values
    print(f"    fallback folds per draw: mean {fb.mean():.3f}, draws with any "
          f"{np.mean(fb > 0):.3f}, with all five {np.mean(fb == 5):.4f}; unconverged folds "
          f"per draw {null['unconverged_folds'].values.mean():.3f}")
    l1, zeros = null["weight_l1"].values, null["zero_sleeves"].values
    print(f"    balanced weights (OC-A5/A6): L1 distance to the blind weights, mean over folds, "
          f"median {np.median(l1):.3f} (q05 {np.quantile(l1, 0.05):.3f}, q95 "
          f"{np.quantile(l1, 0.95):.3f}; 2 is the maximum); sleeves at zero per fold, mean "
          f"{zeros.mean():.2f} of 5")
    sleeves = list(d.sleeve_r.columns)
    blind_mean = d.engine.blind_weights.mean()
    print("    " + "; ".join(
        f"{c}: blind {blind_mean[c]:.3f}, balanced median {np.median(null[f'w_{c}'].values):.3f}"
        f", at zero {null[f'zero_{c}'].values.mean():.0%}" for c in sleeves))
    bsd, cap = null["balanced_sd"].values, null["balanced_cap_share"].values
    print(f"    balanced leg on test: realised sd median {np.median(bsd):.2%} "
          f"({np.quantile(bsd, 0.05):.2%} to {np.quantile(bsd, 0.95):.2%}); cap binds median "
          f"{np.median(cap):.1%}; gross median {np.median(null['balanced_gross'].values):.2f}")
    extra = null["extra_turnover"].values
    print(f"    extra held turnover of the balanced leg over the blind leg: median "
          f"{np.median(extra):+.2f}x/yr, q95 {np.quantile(extra, 0.95):+.2f} (the §5 kill, "
          "re-derived on held weights: 32.2 units a year at headline)")
    p2 = summarise("P2 gross-matched reduction", null["p2_gross"])
    print_null(p2, header=False)
    print(f"\n  thresholds of the primary: q95 {summary['q95']:.3f}, q99 {summary['q99']:.3f}, "
          f"Sidak (1 - {SIDAK:.4f}) {summary['q_sidak']:.3f}; Holm ladder "
          + ", ".join(f"{a:.4f}: {q:.3f}" for a, q in zip(HOLM, summary["holm_quantiles"],
                                                          strict=True)))
    print(f"  MDE of the declared statistic: {summary['mde_shift_05']:.3f} (shift, alpha 0.05), "
          f"{summary['mde_shift_sidak']:.3f} (shift, Sidak), {summary['mde_sd_05']:.3f} "
          f"(sd, 0.05), {summary['mde_sd_sidak']:.3f} (sd, Sidak) — against the draft's 0.204")
    check("§3 A/§4", "MDE of D (the PASS floor)", 0.204, round(summary["mde_shift_05"], 3),
          declared=f"shift {summary['mde_shift_05']:.3f} / Sidak {summary['mde_shift_sidak']:.3f}"
                   f"; sd {summary['mde_sd_05']:.3f} / {summary['mde_sd_sidak']:.3f}")
    # §4's verdict: log 1.5 decidable (at least 80% power), log 1.2 not.
    for label, verdict in (("log 1.5", "decidable"), ("log 1.2", "underpowered")):
        for rule, key in (("P1 95th pct", "alpha_05"), ("Sidak", "sidak")):
            power = summary["power"][label][key]
            check("§4", f"{label} at {rule}", verdict,
                  "decidable" if power >= 0.80 else "underpowered",
                  declared=f"location-shift power {power:.3f}")
    extras = {
        "feasibility_one_block": feas.reset_index().to_dict(orient="records"),
        "feasibility_blocks": feas_b.reset_index().to_dict(orient="records"),
        "hygiene": hygiene, "session_occupancy_deviation": dev, "seconds": elapsed,
        "d_blind": E.NullDistribution(null["d_blind"].values).describe(),
        "d_balanced": E.NullDistribution(null["d_balanced"].values).describe(),
        "fallback_mean": float(fb.mean()), "fallback_any": float(np.mean(fb > 0)),
        "fallback_all": float(np.mean(fb == 5)),
        "balanced_sd_median": float(np.median(bsd)),
        "balanced_cap_median": float(np.median(cap)),
        "extra_turnover_median": float(np.median(extra)),
        "extra_turnover_q95": float(np.quantile(extra, 0.95)),
        "p2_gross": p2,
        "weight_l1_median": float(np.median(l1)), "zero_sleeves_mean": float(zeros.mean()),
        "balanced_weight_median": {c: float(np.median(null[f"w_{c}"].values)) for c in sleeves},
        "sleeve_zero_share": {c: float(null[f"zero_{c}"].values.mean()) for c in sleeves},
    }
    return summary, {"null": null, **extras}


def p2_vol_section(d: Declared, n: int) -> dict:
    """P2's ex-ante-volatility reading under the null, on the first ``n`` primary draws."""
    paths = E.placebo_session_paths(d.engine, d.stamped, n, seed=SEED)
    values = np.array([E.exposure_matched(d.engine.run(paths[c]), d.excess, on="vol").reduction
                       for c in paths.columns])
    s = summarise(f"P2 vol-matched reduction ({n} draws)", E.NullDistribution(values))
    print_null(s)
    return s


def sensitivities(d: Declared, n: int) -> list[dict]:
    section(f"4. Sensitivities of the level-A null ({n} draws each; the choices the lock owns)")
    print(NULL_LEGEND)
    stamps = E.stamps_in_force(d.stamped, d.sessions)
    blocks = E.calendar_blocks(stamps.index, d.sessions, d.folds)
    out = []

    def engine(**kw) -> GuardedLevelA:
        folds = kw.pop("folds", d.folds)
        builder = kw.pop("leg_builder", d.builder)
        g = GuardedLevelA(d.sleeve_r, folds, leg_builder=builder, **kw)
        g.forbid(d.real_path)
        return g

    def one(name: str, g: GuardedLevelA, labels: pd.Series, **kw) -> None:
        s = summarise(name, run_null(g, labels, n, seed=SEED, **kw)["reduction"])
        print_null(s, header=not out)
        out.append(s)

    one("primary, repeated at this draw count", d.engine, d.stamped)
    one("P1 within calendar blocks (OC-P1a)", d.engine, d.stamped,
        blocks=blocks.reindex(stamps.index))
    one("no volatility target, no cost (OC-A1)", engine(
        leg_builder=E.SleeveLegBuilder(d.sleeve_r, scaling="none")), d.stamped)
    one("sleeve-level book at the target, no cost", engine(
        leg_builder=E.SleeveLegBuilder(d.sleeve_r)), d.stamped)
    one("no leverage cap", engine(leg_builder=E.InstrumentLegBuilder(
        d.excess, d.within, S.HEADLINE_SLEEVES, cap=np.inf)), d.stamped)
    one("R2 equal share, occupancy-weighted D (OC-A4)", engine(
        reading="equal_share", occupancy_weighted=True), d.stamped)
    one("R2 equal share, plain D", engine(reading="equal_share"), d.stamped)
    one("extremes picked on the blind leg (OC-A2)", engine(extremes="blind"), d.stamped)
    one("variance about zero (OC-A3)", engine(about_zero=True), d.stamped)
    one("blind leg inverse volatility (OC-A7)", engine(blind_rule="inverse_vol"), d.stamped)
    for weight in (1e-2, 1.0):
        one(f"tie-break pull to blind x {weight:g} (OC-A6)", engine(tie_break=weight),
            d.stamped)
    for m in (3, 5):
        folds = E.stamp_folds(d.stamped, d.sessions, usable=d.usable, min_quarters=m)
        g = engine(folds=folds)
        one(f"folds: min {m} quarters per cell ({len(g.test_sessions)} test s)", g, d.stamped)
    plus = Q.quarterly_labels(d.panel, Q.availability(d.panel.index, rule="release",
                                                      release_dates=d.release, extra_months=1))
    d.engine.forbid(E.expand_to_sessions(stamped_series(plus), d.sessions))
    one("stamps + 1 month (71 transitions)", d.engine, stamped_series(plus))
    rgdp_growth = Q.read_spf(SPF / Q.SPF_GROWTH_FILE[0], Q.SPF_GROWTH_FILE[1])["DRGDP2"]
    first = Q.read_rtdsm_first_release((SPF / Q.RTDSM_ROUTPUT_FILE).read_bytes())
    panel = Q.surprise_panel(
        Q.read_spf(SPF / "median_rgdp_level.xlsx", ("RGDP1", "RGDP2")),
        Q.read_spf(SPF / "median_cpi_level.xlsx", ("CPI1", "CPI2")),
        first_release=first, drgdp2=rgdp_growth).dropna(subset=["growth_within_vintage"])
    wv = Q.quarterly_labels(panel, Q.availability(panel.index, rule="release",
                                                  release_dates=d.release),
                            growth="growth_within_vintage")
    wv_path = E.expand_to_sessions(stamped_series(wv), d.sessions)
    d.engine.forbid(wv_path)
    one("within-vintage growth axis (68 transitions)", d.engine, stamped_series(wv))
    return out


# ------------------------------------------------------------------------------- B1


def b1_section(d: Declared, n: int) -> list[dict]:
    section(f"5. B1 candidate axes: clocks (label-only) and nulls ({n} draws each)")
    t10 = validate(pd.read_parquet(SPF / Q.T10YIE_FILE))
    baa = validate(pd.read_parquet(MACRO / "fin_baa_spread.parquet"))
    panel = Q.market_panel(t10, baa, d.sessions)
    stamps = pd.DatetimeIndex(E.stamps_in_force(d.stamped, d.sessions).index)
    candidates: dict[str, tuple[pd.Series, bool]] = {}
    for w in (63, 252):
        candidates[f"change {w}, daily"] = (Q.b1_change_labels(panel, w), False)
    candidates["level vs expanding median, daily"] = (Q.b1_level_labels(panel), False)
    for name, lab0 in (("change 252", Q.b1_change_labels(panel, 252, lag=0)),
                       ("level vs median", Q.b1_level_labels(panel, lag=0))):
        at = E.expand_to_sessions(lab0.dropna().astype(float), stamps, lag=0).dropna()
        candidates[f"{name}, sampled at the SPF stamps"] = (at, True)
    out = []
    print(f"  {'candidate':40} {'/yr':>6} {'occupancy (test sessions)':>28} "
          f"{'test ep per cell':>18} {'train cells ok':>16}")
    test_mask = d.sessions.isin(d.test)
    for name, (labels, stamped) in candidates.items():
        path = E.expand_to_sessions(labels, d.sessions) if stamped else labels.reindex(
            d.sessions).astype(float)
        d.engine.forbid(path)
        valid = path.dropna()
        clock = Q.clock(valid.astype("Int64"))
        ses, epi = label_counts(codes_of(path.to_numpy(float)[test_mask]))
        occ = np.array(ses) / max(sum(ses), 1)
        qualify = qualifying_cells(d, path)
        print(f"  {name:40} {clock.per_year:6.2f} {' / '.join(f'{x:.3f}' for x in occ):>28} "
              f"{str(epi):>18} {str(qualify):>16}")
        null = run_null(d.engine, labels.astype(float), n, seed=SEED, stamped=stamped,
                        statistics=WITH_FALLBACK)
        s = summarise(f"B1 {name}", null["reduction"])
        fallback_summary(s, null)
        s["clock_per_year"] = clock.per_year
        s["test_cell_sessions"], s["test_cell_episodes"] = ses, epi
        s["train_cells_qualifying"] = qualify
        out.append(s)
    print()
    print(NULL_LEGEND)
    for i, s in enumerate(out):
        print_null(s, header=i == 0)
    print("  fallback folds per draw (balanced leg = blind): "
          + "; ".join(f"{s['name'][3:]} {s['fallback_mean']:.2f} (any {s['fallback_any']:.2f})"
                      for s in out))
    return out


# --------------------------------------------------------------------------- witness


def stitch(paths: dict[int, pd.Series], folds, sessions: pd.DatetimeIndex) -> pd.Series:
    """One session path: fold 1's before the first test, each fold's own on its test."""
    out = paths[folds[0].number].copy()
    for f in folds:
        idx = f.test(sessions)
        out.loc[idx] = paths[f.number].loc[idx]
    return out


def witness_section(d: Declared, n: int) -> list[dict]:
    section(f"6. The volatility witness: clocks and nulls ({n} draws each); its own reduction "
            "is NOT computed")
    blind_path = pd.DataFrame(np.nan, index=d.sessions, columns=d.sleeve_r.columns)
    w = d.engine.blind_weights
    blind_path.loc[d.sessions < d.folds[0].test_start] = w.iloc[0].to_numpy()
    for f in d.folds:
        blind_path.loc[f.test(d.sessions)] = w.loc[f.number].to_numpy()
    drivers = {
        "W1 ^GSPC": d.excess["^GSPC"],
        "W2 blind leg, unscaled": (d.sleeve_r * blind_path).sum(axis=1, min_count=5),
    }
    stamps = pd.DatetimeIndex(E.stamps_in_force(d.stamped, d.sessions).index)
    out = []
    print(f"  {'witness (window 21, 4 training-quantile bins)':52} {'/yr':>6} "
          f"{'test occupancy':>28} {'train cells ok':>16}")
    test_mask = d.sessions.isin(d.test)
    for dname, driver in drivers.items():
        for grid_name, grid in (("daily", None), ("stamp grid", stamps)):
            paths = E.volatility_witness_labels(driver, d.folds, sessions=d.sessions, grid=grid)
            path = stitch(paths, d.folds, d.sessions)
            valid = path.dropna()
            clock = Q.clock(valid.astype("Int64"))
            ses, _ = label_counts(codes_of(path.to_numpy(float)[test_mask]))
            occ = np.array(ses) / max(sum(ses), 1)
            qualify = qualifying_cells(d, paths)
            name = f"{dname}, {grid_name}"
            print(f"  {name:52} {clock.per_year:6.2f} "
                  f"{' / '.join(f'{x:.3f}' for x in occ):>28} {str(qualify):>16}")
            if grid is None:
                null = run_null(d.engine, path, n, seed=SEED, stamped=False,
                                statistics=WITH_FALLBACK)
            else:
                # The bin in force from the session after each stamp, stamped at the stamp.
                pos = d.sessions.searchsorted(stamps, side="left") + 1
                keep = pos < len(d.sessions)
                at = pd.Series(path.to_numpy(float)[pos[keep]], index=stamps[keep]).dropna()
                null = run_null(d.engine, at, n, seed=SEED, statistics=WITH_FALLBACK)
            s = summarise(f"witness {name}", null["reduction"])
            fallback_summary(s, null)
            s["clock_per_year"] = clock.per_year
            s["train_cells_qualifying"] = qualify
            out.append(s)
    print()
    print(NULL_LEGEND)
    for i, s in enumerate(out):
        print_null(s, header=i == 0)
    print("  fallback folds per draw (balanced leg = blind): "
          + "; ".join(f"{s['name'][8:]} {s['fallback_mean']:.2f} (any {s['fallback_any']:.2f})"
                      for s in out))
    print("  The witness's own reduction (its real partition) is the comparator of the declared "
          "test: not\n  computed before the lock. The daily-grid nulls draw on session paths "
          "(stamped=False).")
    return out


# ------------------------------------------------------------------------------- B2


def b2_section(d: Declared, n: int, n_p3: int) -> dict:
    section(f"7. B2: map spaces, the null of its D reduction ({n} draws), P3 on content-free "
            f"partitions ({n_p3}), paired-Sharpe MDE")
    out: dict = {"spaces": {}}
    for space in ("permutation", "profile", "axis_consistent"):
        summary = E.map_space_summary(space)
        out["spaces"][space] = {"maps": summary.maps, "books": summary.budget_vectors,
                                "ties": summary.reference_ties,
                                "min_p": summary.min_p_value,
                                "max_percentile": summary.max_percentile}
        print(f"  {space:16} maps {summary.maps:6d}  distinct books {summary.budget_vectors:5d}  "
              f"draft-map ties {summary.reference_ties:3d}  min p {summary.min_p_value:.4f}  "
              f"max percentile {summary.max_percentile:.4f}")
    size = E.map_space_size("any_subset")
    print(f"  any_subset       maps {size:6d} (not enumerated)")
    budgets = E.map_risk_budgets(E.DRAFT_MAP)
    print("  draft-map budgets: " + ", ".join(f"{k} {v:.4f}" for k, v in budgets.items()))
    weights = {rule: E.b2_fold_weights(d.engine, E.DRAFT_MAP, rule=rule)
               for rule in ("inverse_vol", "erc")}
    for rule, w in weights.items():
        frame = pd.DataFrame(w, index=d.engine.fold_numbers, columns=d.engine.sleeves)
        print(f"  B2 weights per fold ({rule}, training moments only):")
        print("    " + frame.round(3).to_string().replace("\n", "\n    "))
    paths = E.placebo_session_paths(d.engine, d.stamped, n, seed=SEED)
    values = np.array([d.engine.run(paths[c], weights=weights["inverse_vol"]).reduction
                       for c in paths.columns])
    s = summarise("B2 draft map (inverse vol) vs blind", E.NullDistribution(values))
    print(NULL_LEGEND)
    print_null(s, header=True)
    out["null"] = s

    # P3's size when the partition is content-free: where does the draft map rank among
    # the permutation (and profile) maps, partition by partition?
    for space in ("permutation", "profile"):
        maps = list(E.enumerate_maps(space))
        books: dict[tuple, np.ndarray] = {}
        keys = []
        for m in maps:
            b = E.map_risk_budgets(m, empty="renormalise")
            key = tuple(np.round(b.to_numpy(float), 12))
            keys.append(key)
            if key not in books:
                books[key] = E.b2_fold_weights(d.engine, m, empty="renormalise")
        ref = tuple(np.round(budgets.to_numpy(float), 12))
        pct = []
        for c in paths.columns[:n_p3]:
            red = {k: d.engine.run(paths[c], weights=w).reduction for k, w in books.items()}
            null = np.array([red[k] for k in keys])
            pct.append(float(pct_rank(null, np.array([red[ref]]))[0]))
        pct_arr = np.array(pct)
        out[f"p3_{space}"] = {"draws": len(pct), "pass_95": float((pct_arr >= 0.95).mean()),
                              "pct_mean": float(pct_arr.mean()),
                              "pct_deciles": np.quantile(pct_arr, np.linspace(0.1, 0.9, 9))
                              .round(3).tolist()}
        print(f"  P3 {space:12} on {len(pct)} content-free partitions: draft map at or above "
              f"the 95th percentile of its space in {(pct_arr >= 0.95).mean():.3f} of them; "
              f"mean percentile {pct_arr.mean():.3f}")

    result = d.engine.run(paths[paths.columns[0]], weights=weights["inverse_vol"])
    a, b = result.balanced.to_numpy(float), result.blind.to_numpy(float)
    out["mde"] = {}
    print("\n  paired Sharpe MDE, B2 map leg vs blind leg, pooled test, DEMEANED legs "
          "(blinded_mde):")
    for block in E.POWER_BLOCKS:
        r = blinded_mde(a, b, mean_block=block, alpha=0.05)
        out["mde"][block] = {"se": r.se, "mde_05": r.mde, "mde_sidak": mde_at(r, SIDAK)}
        print(f"    block {block:3d}: SE {r.se:.4f}  MDE {r.mde:.3f} (0.05)  "
              f"{mde_at(r, SIDAK):.3f} (Sidak)")
    both = pd.DataFrame({"a": a, "b": b}).dropna()
    out["corr"] = float(both.corr().iloc[0, 1])
    out["sd_b2"] = float(both["a"].std(ddof=1) * np.sqrt(PERIODS))
    print(f"    correlation of the legs {out['corr']:.3f}; B2 leg realised sd {out['sd_b2']:.2%}")
    return out


# ---------------------------------------------------------------------------- level C


def level_c_section(d: Declared, n_mde: int, n_turn: int) -> dict:
    section(f"8. Level C on PLACEBO transition dates: paired-Sharpe MDE ({n_mde} draws), "
            f"turnover-saving null ({n_turn})")
    s = d.sessions
    stamps = pd.DatetimeIndex([t for t in d.stamped.index if s[0] <= t <= s[-1]])
    quarter_ends = pd.DatetimeIndex(
        pd.Series(s, index=s).groupby(s.to_period("Q")).max().to_numpy())
    test = d.test
    years = len(test) / PERIODS
    calendars = {"every SPF stamp": stamps, "calendar quarter-ends": quarter_ends}
    books = {}
    for name, dates in calendars.items():
        book, _, _ = S.blind_book(d.excess, S.HEADLINE_SLEEVES, list(dates))
        books[name] = book
        per_year = np.asarray(dates.isin(test)).sum() / years
        print(f"  calendar '{name}': {per_year:.2f} rebalances a year on the test sessions; "
              f"held turnover {S.annual_turnover(book.weights.reindex(test)):.2f}x/yr")
    in_force = E.stamps_in_force(d.stamped, s)
    draws = E.p1_draws(in_force, max(n_mde, n_turn), seed=SEED).draws
    out: dict = {"mde": {name: {b: [] for b in E.POWER_BLOCKS} for name in calendars},
                 "saving": {name: [] for name in calendars}, "per_year": []}
    t0 = time.time()
    for j, column in enumerate(draws.columns):
        moves = E.transition_stamps(draws[column])
        moves = pd.DatetimeIndex([t for t in moves if s[0] <= t <= s[-1]])
        book, _, _ = S.blind_book(d.excess, S.HEADLINE_SLEEVES, list(moves))
        out["per_year"].append(float(np.asarray(moves.isin(test)).sum() / years))
        net = S.net_returns(book).reindex(test)
        turnover = S.annual_turnover(book.weights.reindex(test))
        for name, cal in books.items():
            out["saving"][name].append(S.annual_turnover(cal.weights.reindex(test)) - turnover)
            if j < n_mde:
                ref = S.net_returns(cal).reindex(test)
                for block in E.POWER_BLOCKS:
                    r = blinded_mde(net.to_numpy(float), ref.to_numpy(float), mean_block=block,
                                    alpha=0.05)
                    out["mde"][name][block].append((r.se, r.mde, mde_at(r, SIDAK)))
    print(f"  placebo transition dates: {np.mean(out['per_year']):.2f} a year on the test "
          f"sessions (sd {np.std(out['per_year']):.2f}); {time.time() - t0:.0f} s")
    summary: dict = {"per_year_mean": float(np.mean(out["per_year"])), "calendars": {}}
    for name in calendars:
        print(f"\n  against '{name}':")
        rows = {}
        for block in E.POWER_BLOCKS:
            arr = np.array(out["mde"][name][block])
            rows[block] = {"se_median": float(np.median(arr[:, 0])),
                           "mde_05_median": float(np.median(arr[:, 1])),
                           "mde_05_range": [float(arr[:, 1].min()), float(arr[:, 1].max())],
                           "mde_sidak_median": float(np.median(arr[:, 2]))}
            print(f"    block {block:3d}: SE median {rows[block]['se_median']:.4f}; paired Sharpe "
                  f"MDE median {rows[block]['mde_05_median']:.3f} "
                  f"({rows[block]['mde_05_range'][0]:.3f} to {rows[block]['mde_05_range'][1]:.3f})"
                  f" at 0.05, {rows[block]['mde_sidak_median']:.3f} at Sidak")
        saving = E.NullDistribution(np.array(out["saving"][name]))
        q = saving.quantile([0.2, 0.5, 0.95, 0.99])
        print(f"    turnover saving (calendar - conditional, held, x/yr) under the null: median "
              f"{q[1]:+.3f}, sd {saving.sd:.3f}, skew {saving.skewness:+.2f}, atom share "
              f"{saving.atom_share:.3f}, q99 - q20 {q[3] - q[0]:.3f}, distinct values "
              f"{saving.n_distinct}")
        summary["calendars"][name] = {"mde": rows, "saving": saving.describe()}
    print("\n  The target moves between rebalances (ERC re-estimated on 252 sessions at every "
          "stamp),\n  so the level-C null is not the Two Sigma C-1 atom. Real transition dates "
          "were not used.")
    return summary


# ------------------------------------------------------------------------------ main


def reproduction_table() -> None:
    section("9. Disclosed figures of §1.4 and §3-§5 against the re-measurement")
    width = max(len(c.quantity) for c in CHECKS)
    for c in CHECKS:
        tail = f"   declared object: {c.declared}" if c.declared else ""
        print(f"  {c.section:8} {c.quantity:{width}}  disclosed {c.disclosed:>24}  measured "
              f"{c.measured:>24}  {'ok' if c.match else 'DIFFERS'}{tail}")
    print(f"\n  {sum(c.match for c in CHECKS)} of {len(CHECKS)} reproduce")


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items() if not isinstance(
            v, E.NullDistribution | pd.DataFrame)}
    if isinstance(obj, list | tuple):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return str(obj)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--draws", type=int, default=2000, help="primary null draws")
    parser.add_argument("--sens-draws", type=int, default=1000,
                        help="draws for each sensitivity, B1, witness and B2 null")
    parser.add_argument("--p3-draws", type=int, default=200)
    parser.add_argument("--p2-draws", type=int, default=300)
    parser.add_argument("--c-mde-draws", type=int, default=10)
    parser.add_argument("--c-turnover-draws", type=int, default=200)
    parser.add_argument("--json", default=None, help="write every figure to this path")
    args = parser.parse_args()
    t0 = time.time()
    results: dict = {"seed": SEED, "draws": args.draws, "sens_draws": args.sens_draws}
    results["inputs"] = inputs_section()
    results["draft"] = draft_arithmetic()
    declared = build_declared()
    results["declared"] = declared_section(declared)
    primary, extras = primary_section(declared, args.draws)
    results["primary"] = primary
    results["primary_extras"] = {k: v for k, v in extras.items() if k != "null"}
    print()
    results["p2_vol"] = p2_vol_section(declared, args.p2_draws)
    results["sensitivities"] = sensitivities(declared, args.sens_draws)
    results["b1"] = b1_section(declared, args.sens_draws)
    results["witness"] = witness_section(declared, args.sens_draws)
    results["b2"] = b2_section(declared, args.sens_draws, args.p3_draws)
    results["level_c"] = level_c_section(declared, args.c_mde_draws, args.c_turnover_draws)
    reproduction_table()
    results["checks"] = [c.__dict__ for c in CHECKS]
    runs = declared.engine.runs
    print(f"\n  LevelA.run calls on the primary engine: {runs:,}, every one on a placebo path "
          f"(the guard refused none); total {time.time() - t0:.0f} s")
    if args.json:
        path = ROOT / args.json if not args.json.startswith("/") else None
        if path is None:
            raise SystemExit("--json takes a path relative to the repository root")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(jsonable(results), indent=1, default=str) + "\n")
        print(f"  written: {args.json}")


if __name__ == "__main__":
    main()
