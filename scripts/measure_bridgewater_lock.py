"""Bridgewater study — the declared objects of the lock, measured blind after the audits.

The audits of 2026-09-25 found that the draft lock's power, bar and witness figures did
not describe its declared verdict rule, and that its evaluation partition (the quadrant
held from its SPF release, 1.5 to 2.5 quarters after the environment it names) was
close to independent of the environment in force. This script measures the object the
corrected lock declares, from committed package code only
(`construction.declared`, `construction.inference`, `construction.evaluate`):

1. **the objects**: the stamped object of level C, checked identical to
   ``build_declared`` of `scripts/measure_bridgewater_power.py` at ``39ef608``; the
   contemporaneous object of levels A, B1 and B2 (label of the calendar quarter, folds
   cut at quarter ends, training cells restricted to labels public at the fold's cut);
2. **label-only statistics** of every partition, fold layouts, training masks, the
   +1-month masks, cell qualification as `evaluate.fit_balanced` counts it;
3. **state-blind books**: the blind leg, the B2 map leg, the time-zone diagnostic, the
   participation ratio and entropy rank of the sleeves, P2's undefined sessions;
4. **nulls on content-free partitions** (P1 per calendar block, uniform, 2,000 draws,
   seed 20260924) for A, B2 (map leg fixed), B1, P2, the witnesses, and level C's
   placebo conditioned on the real test-span transition count; their bars, and the
   power of the **verdict rule** (Δ ≥ T and p ≤ α_S together) under a location shift;
5. **the witnesses' own reductions** (their partitions carry no quadrant or B1 label),
   standardised in their own nulls, and the effective PASS bar they imply;
6. **planted alternatives**: sleeve-specific and volatility-proxy variance effects
   planted on content-free partitions, run through the whole declared pipeline on the
   targeted and on the unscaled book;
7. **reproduction** of the stamped per-block null of `power.json` under one BLAS
   thread, to 1e-9.

**Blind.** No statistic is conditioned on the real quadrant labels or the real B1
labels. Every engine is a `inference.GuardedLevelA` that refuses the real quadrant,
within-vintage, +1-month, stamped and B1 paths, and any path agreeing with one of them
on 80% of sessions or more under the best relabelling. Level C runs on placebo
transition dates only. The witnesses' reductions are computed on their own (volatility)
partitions, which the blindness rule allows.

Usage:
    .venv/bin/python scripts/measure_bridgewater_lock.py \
        --json docs/artifacts/bridgewater/lock.json > docs/artifacts/bridgewater/lock.txt
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections.abc import Callable, Mapping
from importlib import metadata

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from regime_lab.config import RAW, ROOT
from regime_lab.construction import declared as D
from regime_lab.construction import evaluate as E
from regime_lab.construction import inference as I
from regime_lab.construction import quadrant as Q
from regime_lab.construction import sleeves as S
from regime_lab.data.pit import validate
from regime_lab.extensions.couverture import participation_ratio
from regime_lab.selection.protocol import blinded_mde, mde_at
from regime_lab.selection.tree import ReadingRefused, draw_map, git_head, require_committed

SEED = 20260924
SIDAK = E.sidak_alpha()
PERIODS = 252
DAYS_PER_YEAR = 365.25
#: §4's alternatives on D (3:1 -> 2.5:1 and 3:1 -> 2:1), plus a larger one.
SHIFTS = {"log 1.2": float(np.log(1.2)), "log 1.5": float(np.log(1.5)), "0.6": 0.6}
PACKAGES = ("numpy", "pandas", "scipy", "statsmodels", "openpyxl", "threadpoolctl")
RULE = "=" * 100
POWER_JSON = ROOT / "docs" / "artifacts" / "bridgewater" / "power.json"
POWER_SCRIPT = ROOT / "scripts" / "measure_bridgewater_power.py"


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def fingerprint(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def codes_of(path: pd.Series) -> np.ndarray:
    v = path.to_numpy(float)
    return np.where(np.isfinite(v), np.nan_to_num(v, nan=-1.0), -1.0).astype(np.int64)


def years(index: pd.DatetimeIndex) -> tuple[float, float]:
    return len(index) / PERIODS, (index[-1] - index[0]).days / DAYS_PER_YEAR


# ----------------------------------------------------------------------- summaries


def summarise(name: str, null: E.NullDistribution, *, floor: float = E.DRAFT_MDE,
              se: float | None = None) -> dict:
    """Shape, bar and the power of the verdict rule of one null (content-free)."""
    v = null.values
    out: dict = {"name": name, **null.describe(), "fingerprint": fingerprint(v)}
    out["q_sidak"] = float(null.quantile(1.0 - SIDAK)) if null.finite else float("nan")
    out["bar"] = I.bar(null, SIDAK, floor)
    out["floor"] = floor
    out["power_percentile_only"] = {k: I.percentile_power(v, s, SIDAK) for k, s in SHIFTS.items()}
    shifts = {**SHIFTS, "T": out["bar"]}
    out["power_verdict"] = {k: I.verdict_power(v, s, out["bar"], SIDAK, se=se)
                            for k, s in shifts.items()}
    out["shift_80"] = I.shift_for_power(v, out["bar"], SIDAK, se=se)
    out["size_at_bar"] = float(np.mean(v >= out["bar"]))
    return out


def print_nulls(rows: list[dict]) -> None:
    print(f"  {'null':40} {'n':>5} {'mean':>7} {'sd':>6} {'skew':>6} {'atom':>5} {'q20':>7} "
          f"{'q50':>7} {'q95':>7} {'qSid':>7} {'MDEsh':>6} {'MDEsd':>6} {'bar T':>6} "
          f"{'P(T)':>5} {'Plog1.5':>7} {'d80':>6}")
    for s in rows:
        print(f"  {s['name']:40} {int(s['draws']):5d} {s['mean']:7.3f} {s['sd']:6.3f} "
              f"{s['skewness']:6.2f} {s['atom_share']:5.3f} {s['q20']:7.3f} {s['q50']:7.3f} "
              f"{s['q95']:7.3f} {s['q_sidak']:7.3f} {s['mde_shift_sidak']:6.3f} "
              f"{s['mde_sd_sidak']:6.3f} {s['bar']:6.3f} {s['power_verdict']['T']:5.3f} "
              f"{s['power_verdict']['log 1.5']:7.3f} {s['shift_80']:6.3f}")


LEGEND = (
    "  Each null: the statistic on content-free P1 draws (uniform, per calendar block). qSid: "
    "the 1 - alpha_S\n  quantile (alpha_S = 0.0102). MDEsh / MDEsd: shift and Gaussian MDE at "
    "alpha_S. bar T = max(floor,\n  MDEsh, MDEsd). P(T), Plog1.5: power of the VERDICT RULE "
    "(Delta >= T and p <= alpha_S together)\n  under a location shift of T and of log 1.5. d80: "
    "the shift that gives 80% power under that rule."
)


# ------------------------------------------------------------------------- objects


def load_power_script():
    spec = importlib.util.spec_from_file_location("measure_bridgewater_power", POWER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def same_object_as_39ef608(stamped: D.StampedObject) -> dict:
    """The package's stamped object against the power script's ``build_declared``."""
    old = load_power_script().build_declared()
    checks = {
        "sessions": old.sessions.equals(stamped.sessions),
        "excess": old.excess.equals(stamped.panel.excess),
        "within": old.within.equals(stamped.panel.within),
        "sleeve_returns": old.sleeve_r.equals(stamped.panel.sleeve_returns),
        "usable": old.usable.equals(stamped.panel.usable),
        "stamped": old.stamped.equals(stamped.stamped),
        "real_path": old.real_path.equals(stamped.real_path),
        "folds": tuple(old.folds) == tuple(stamped.folds),
    }
    return checks


def fit_counts(path: pd.Series, sessions: pd.DatetimeIndex, fold) -> tuple[list, list]:
    """Sessions and episodes per cell on a fold's training path, as fit_balanced counts."""
    train = np.asarray(sessions <= fold.train_end)
    n, ep = E.cell_counts(codes_of(path)[train])
    return n.tolist(), ep.tolist()


def test_counts(paths: Mapping[int, pd.Series], sessions: pd.DatetimeIndex, folds
                ) -> tuple[list[int], list[int], list[list[int]]]:
    codes = []
    per_fold = []
    for f in folds:
        mine = np.asarray((sessions >= f.test_start) & (sessions <= f.test_end))
        c = codes_of(paths[f.number])[mine]
        per_fold.append(E.cell_counts(c)[0].tolist())
        codes.append(c)
    n, ep = E.cell_counts(np.concatenate(codes))
    return n.tolist(), ep.tolist(), per_fold


def persistence(labels: pd.Series, n_perm: int = 20_000, seed: int = 0) -> dict:
    """P(same as last quarter) and an order-permutation p (label-only)."""
    v = labels.to_numpy(int)
    same = float((v[1:] == v[:-1]).mean())
    rng = np.random.default_rng(seed)
    perm = np.array([(p[1:] == p[:-1]).mean() for p in (rng.permutation(v) for _ in range(n_perm))])
    occ = np.bincount(v, minlength=4) / v.size
    return {"same": same, "independent": float((occ ** 2).sum()),
            "p_le_observed": float((perm <= same).mean()), "pairs": int(v.size - 1)}


def layout_section(obj: D.ContemporaneousObject, name: str, *, show: bool = True) -> dict:
    s = obj.sessions
    paths = obj.paths()
    rows = []
    for f in obj.folds:
        n, ep = fit_counts(paths[f.number], s, f)
        masked = np.asarray((s <= f.train_end) & paths[f.number].isna()
                            & obj.partition.path(s).notna())
        with_returns = np.asarray(s.isin(obj.panel.usable) & (s <= f.train_end))
        nr, epr = E.cell_counts(codes_of(paths[f.number])[with_returns])
        qualify = int(sum(a >= E.MIN_CELL_SESSIONS and b >= E.MIN_CELL_EPISODES
                          for a, b in zip(n, ep, strict=True)))
        rows.append({"fold": f.number, "train_end": str(f.train_end.date()),
                     "test_start": str(f.test_start.date()), "test_end": str(f.test_end.date()),
                     "test_sessions": int(len(f.test(s))),
                     "test_quarters": int(len(pd.period_range(f.test_start, f.test_end,
                                                              freq="Q"))),
                     "masked_quarters": sorted(set(D.session_quarters(s[masked]).astype(str))),
                     "fit_sessions": n, "fit_episodes": ep,
                     "sessions_with_returns": nr.tolist(), "episodes_with_returns": epr.tolist(),
                     "cells_qualifying": qualify})
    ses, epi, per_fold = test_counts(paths, s, obj.folds)
    test = pd.DatetimeIndex(np.concatenate([f.test(s).to_numpy() for f in obj.folds]))
    ys, yc = years(test)
    if show:
        print(f"  {name}: expanding walk-forward cut at quarter ends (4 public counted quarters "
              "per cell before the first test)")
        print(f"  {'fold':>4} {'train end':>10} {'test':>24} {'sess':>5} {'q':>3} {'masked':>8} "
              f"{'fit sessions (fit_balanced)':>28} {'episodes':>14} {'ok':>3} "
              f"{'test sessions per cell':>24}")
        for r, tp in zip(rows, per_fold, strict=True):
            print(f"  {r['fold']:4d} {r['train_end']:>10} {r['test_start']} -> {r['test_end']} "
                  f"{r['test_sessions']:5d} {r['test_quarters']:3d} "
                  f"{','.join(r['masked_quarters']) or '-':>8} {str(r['fit_sessions']):>28} "
                  f"{str(r['fit_episodes']):>14} {r['cells_qualifying']:3d} {str(tp):>24}")
        print("  the same training counts on sessions with all five sleeve returns only: "
              + "; ".join(f"fold {r['fold']} {r['sessions_with_returns']} / "
                          f"{r['episodes_with_returns']}" for r in rows))
        print(f"  pooled test: {len(test):,} sessions ({test[0]:%Y-%m-%d} -> {test[-1]:%Y-%m-%d}), "
              f"{ys:.2f} session-years, {yc:.2f} calendar years; per cell {ses} sessions, "
              f"{epi} episodes")
    return {"folds": rows, "test_sessions": int(len(test)), "test_first": str(test[0].date()),
            "test_last": str(test[-1].date()), "years_sessions": ys, "years_calendar": yc,
            "test_cell_sessions": ses, "test_cell_episodes": epi,
            "test_sessions_by_fold_cell": per_fold}


def objects_section(stamped: D.StampedObject, obj: D.ContemporaneousObject,
                    inputs: D.QuadrantInputs) -> dict:
    section("1. The declared objects (label COUNTS, dates and fold cuts only)")
    out: dict = {}
    checks = same_object_as_39ef608(stamped)
    print("  stamped object (level C) against build_declared of the power script at 39ef608: "
          + ", ".join(f"{k} {'identical' if v else 'DIFFERS'}" for k, v in checks.items()))
    if not all(checks.values()):
        raise SystemExit("the package's stamped object is not the object of 39ef608")
    out["stamped_identical_to_39ef608"] = checks
    s = obj.sessions
    p = obj.partition
    print(f"\n  contemporaneous quadrant: {len(p.labels)} quarters {p.labels.index[0]} -> "
          f"{p.labels.index[-1]}; sessions {s[0]:%Y-%m-%d} -> {s[-1]:%Y-%m-%d} ({len(s):,})")
    path = p.path(s)
    counts = path.value_counts().sort_index()
    quarters = p.labels.value_counts().sort_index()
    for k in range(4):
        print(f"    {Q.CELLS[k]:24} {int(counts.get(float(k), 0)):5d} sessions "
              f"{counts.get(float(k), 0) / path.notna().sum():6.1%}  "
              f"{int(quarters.get(float(k), 0)):3d} quarters")
    unlabelled = s[path.isna().to_numpy()]
    print(f"    unlabelled sessions: {len(unlabelled)} ({unlabelled[0]:%Y-%m-%d} -> "
          f"{unlabelled[-1]:%Y-%m-%d}: 2026Q3, whose surprise needs survey 2026Q4)")
    per = persistence(p.labels)
    clock_q = Q.clock(pd.Series(p.labels.to_numpy(int), index=p.labels.index.start_time))
    clock_s = Q.clock(path.dropna().astype("Int64"))
    print(f"    clock {clock_q.transitions} transitions over {len(p.labels)} quarters, "
          f"{clock_q.per_year:.3f}/yr on quarter starts; {clock_s.transitions} session changes, "
          f"{clock_s.per_year:.3f}/yr; P(same) {per['same']:.3f} over {per['pairs']} pairs vs "
          f"{per['independent']:.3f} independent, permutation p {per['p_le_observed']:.3f}")
    both = pd.concat([stamped.real_path, path], axis=1).dropna()
    agree = float((both.iloc[:, 0] == both.iloc[:, 1]).mean())
    print(f"    the stamped cell (39ef608) equals the contemporaneous cell on {agree:.3f} of "
          f"{len(both):,} sessions (independent draws at this occupancy: {per['independent']:.3f})")
    out["contemporaneous"] = {
        "quarters": int(len(p.labels)), "first": str(p.labels.index[0]),
        "last": str(p.labels.index[-1]),
        "cell_sessions": [int(counts.get(float(k), 0)) for k in range(4)],
        "cell_quarters": [int(quarters.get(float(k), 0)) for k in range(4)],
        "unlabelled_sessions": int(len(unlabelled)), "persistence": per,
        "transitions": clock_q.transitions, "per_year": clock_q.per_year,
        "per_year_sessions": clock_s.per_year, "stamped_agreement": agree,
        "quarters_by_cell": {Q.CELLS[k]: [str(q) for q in p.labels.index[p.labels == k]]
                             for k in range(4)},
    }
    print()
    out["layout"] = layout_section(obj, "headline, quadrant")
    alt = D.quarter_folds(p.labels, p.available, s, min_quarters=3, usable=obj.panel.usable)
    alt_test = sum(len(f.test(s)) for f in alt)
    print(f"  not taken: 3 quarters per cell would test from {alt[0].test_start:%Y-%m-%d} "
          f"({alt_test:,} sessions); counting the quarters before the first sleeve return "
          f"would test from {D.quarter_folds(p.labels, p.available, s)[0].test_start:%Y-%m-%d}")
    out["layout_min3_first_test"] = str(alt[0].test_start.date())
    out["layout_min3_sessions"] = int(alt_test)
    decades = Q.composition(path, Q.decade(s))
    print("  sessions per cell by decade: " + "; ".join(
        f"{d} {row.tolist()}" for d, row in decades.iterrows()))
    out["decades"] = {d: row.tolist() for d, row in decades.iterrows()}

    # +1 month: only the availability dates move; count the masks that change.
    plus = D.contemporaneous_object(inputs=inputs, extra_months=1, folds=obj.folds)
    m0 = D.fold_masks(p.available, s, obj.folds)
    m1 = D.fold_masks(plus.partition.available, s, obj.folds)
    changed = {f.number: sorted(set(D.session_quarters(s[m0[f.number] != m1[f.number]])
                                    .astype(str))) for f in obj.folds}
    print(f"  +1 month (every SPF stamp one month later): training quarters whose public status "
          f"changes at a fold's cut: {changed}")
    stamps_plus = D.stamped_series(plus.quarterly)
    inside = stamps_plus.index[(stamps_plus.index >= s[0]) & (stamps_plus.index <= s[-1])]
    print(f"  +1-month stamps in the sample on a non-session: {int((~inside.isin(s)).sum())} of "
          f"{len(inside)} (level C's timing rerun)")
    out["plus_one_month_mask_changes"] = changed
    out["plus_one_month_nonsession_stamps"] = [int((~inside.isin(s)).sum()), int(len(inside))]
    return out


# ------------------------------------------------------------------- state-blind


def blind_books_section(obj: D.ContemporaneousObject, engine: I.GuardedLevelA,
                        w_b2: np.ndarray) -> dict:
    section("2. State-blind books on the contemporaneous test sessions (no label is read)")
    test = engine.test_sessions
    out: dict = {}
    print("  blind leg (ERC on each fold's expanding training covariance), sleeve weights:")
    print("    " + engine.blind_weights.round(3).to_string().replace("\n", "\n    "))
    legs = {"blind": engine.blind_leg,
            "B2 map leg": engine.builder(engine._path(w_b2))}
    for name, leg in legs.items():
        r = leg.returns.reindex(test)
        held = leg.held.reindex(test)
        gross = held.abs().sum(axis=1)
        row = {"sd": float(r.std(ddof=1) * np.sqrt(PERIODS)),
               "cap_share": float(leg.cap_binds.reindex(test).mean()),
               "gross_median": float(gross.median()), "gross_p95": float(gross.quantile(0.95)),
               "turnover": S.annual_turnover(held), "finite": int(r.notna().sum())}
        out[name] = row
        print(f"  {name:10}: realised sd {row['sd']:.2%}; cap binds {row['cap_share']:.1%}; gross "
              f"median {row['gross_median']:.2f} p95 {row['gross_p95']:.2f}; held turnover "
              f"{row['turnover']:.2f}x/yr; finite on {row['finite']} of {len(test)} test sessions")
    frame = pd.DataFrame(w_b2, index=engine.fold_numbers, columns=engine.sleeves)
    print("  B2 map-leg sleeve weights (training volatilities, no cell):")
    print("    " + frame.round(3).to_string().replace("\n", "\n    "))

    # Time zones (§12.4): the eight foreign columns held one session later.
    late = D.leg_builder(obj.panel, foreign_lag=1, foreign=D.FOREIGN_CLOSES)
    base = engine.blind_leg.returns.reindex(test)
    shifted = late(engine._path(engine._blind)).returns.reindex(test)
    asia = D.leg_builder(obj.panel, foreign_lag=1, foreign=("^N225", "^HSI", "^AXJO"))
    shifted_asia = asia(engine._path(engine._blind)).returns.reindex(test)
    tz = {"dlogvar_all": float(np.log(shifted.var(ddof=1)) - np.log(base.var(ddof=1))),
          "dlogvar_asia": float(np.log(shifted_asia.var(ddof=1)) - np.log(base.var(ddof=1))),
          "corr": float(pd.concat([base, shifted], axis=1).dropna().corr().iloc[0, 1])}
    print(f"  time zones: holding the 8 foreign columns' scaled weights one session later moves "
          f"the blind leg's test log variance by {tz['dlogvar_all']:+.4f} "
          f"({tz['dlogvar_asia']:+.4f} for Asia alone); correlation {tz['corr']:.4f}")
    out["time_zones"] = tz

    # The three drafting figures of §I.4.
    r5 = obj.panel.sleeve_returns.dropna()
    corr = r5.corr().to_numpy()
    corr_test = obj.panel.sleeve_returns.reindex(test).dropna().corr().to_numpy()
    figs = {"pr_all": participation_ratio(corr), "pr_test": participation_ratio(corr_test),
            "entropy_all": I.entropy_rank(corr), "entropy_test": I.entropy_rank(corr_test),
            "sessions_all": int(len(r5)),
            "sessions_test": int(obj.panel.sleeve_returns.reindex(test).dropna().shape[0])}
    held = engine.blind_leg.held
    for flag in (False, True):
        vol = E.ex_ante_volatility(held, obj.panel.excess, missing_as_zero=flag).reindex(test)
        figs[f"p2_undefined_{'zero' if flag else 'strict'}"] = int(vol.isna().sum())
    print(f"  sleeves' correlation (Pearson): participation ratio {figs['pr_all']:.3f} over "
          f"{figs['sessions_all']:,} sessions, {figs['pr_test']:.3f} over the "
          f"{figs['sessions_test']:,} test sessions; entropy rank {figs['entropy_all']:.2f} / "
          f"{figs['entropy_test']:.2f}")
    print(f"  P2's ex-ante volatility undefined on {figs['p2_undefined_strict']} test sessions "
          f"when a missing return voids the window, {figs['p2_undefined_zero']} when it counts "
          "as zero (the declared reading)")
    out["figures"] = figs
    return out


# ------------------------------------------------------------------------- nulls


def a_statistics(obj: D.ContemporaneousObject) -> dict[str, Callable[[E.LevelAResult], float]]:
    def mean_on(series: pd.Series, idx: pd.DatetimeIndex) -> float:
        return float(series.reindex(idx).mean())

    return {
        "reduction": lambda r: r.reduction,
        "d_blind": lambda r: r.d_blind,
        "d_balanced": lambda r: r.d_balanced,
        "fallback_folds": lambda r: float(len(r.fallback_folds)),
        "unconverged_folds": lambda r: float(sum(not f.converged for f in (r.fits or ()))),
        "balanced_sd": lambda r: float(r.balanced.std(ddof=1) * np.sqrt(PERIODS)),
        "balanced_cap_share": lambda r: mean_on(r.balanced_leg.cap_binds, r.test_labels.index),
        "balanced_gross": lambda r: mean_on(r.balanced_leg.held.abs().sum(axis=1),
                                            r.test_labels.index),
        "extra_turnover": lambda r: (
            S.annual_turnover(r.balanced_leg.held.reindex(r.test_labels.index))
            - S.annual_turnover(r.blind_leg.held.reindex(r.test_labels.index))),
        "p2_vol": lambda r: E.exposure_matched(r, obj.panel.excess, on="vol",
                                               missing_as_zero=True).reduction,
        "p2_gross": lambda r: E.exposure_matched(r, obj.panel.excess, on="gross").reduction,
        "zero_sleeves": lambda r: float((r.balanced_weights < 1e-6).sum(axis=1).mean()),
    }


B2_STATISTICS = {
    "reduction": lambda r: r.reduction,
}


def hygiene(partition: D.Partition, blocks: pd.Series, n: int) -> dict:
    draws = I.quarter_draws(partition.labels, n, seed=SEED, blocks=blocks)
    out = I.draws_against_real(partition.labels, draws)
    out["identical_in_block"] = I.blockwise_identical(partition.labels, draws, blocks)
    feas = E.p1_feasibility(partition.labels, blocks.reindex(partition.labels.index))
    out["log10_orders"] = feas["log10_orders"].round(3).to_dict()
    return out


def a_section(obj: D.ContemporaneousObject, engine: I.GuardedLevelA, n: int) -> dict:
    section(f"3. Level A: the null of the reduction under P1 per calendar block ({n} draws)")
    blocks = obj.blocks()
    hyg = hygiene(obj.partition, blocks, n)
    print(f"  P1 freedom per block (log10 join-free orders): {hyg['log10_orders']}")
    print(f"  draws identical to the real sequence: {hyg['identical']}; agreement mean "
          f"{hyg['agreement_mean']:.3f}, max {hyg['agreement_max']:.3f}; draws reproducing the "
          f"real labels inside one block: {hyg['identical_in_block']}")
    t0 = time.time()
    null = I.null_statistics(engine, obj.partition, n, statistics=a_statistics(obj), seed=SEED,
                             blocks=blocks)
    elapsed = time.time() - t0
    red = null["reduction"]
    summary = summarise("A: D(blind) - D(balanced)", red)
    p2 = summarise("A: P2 vol-matched, no cap", null["p2_vol"])
    gross = summarise("A: P2 gross-matched (diagnostic)", null["p2_gross"])
    print(f"  {n} draws in {elapsed:.0f} s; every draw finite: {red.finite}")
    print(LEGEND)
    print_nulls([summary, p2, gross])
    fb = null["fallback_folds"].values
    print(f"  legs under the null: D_blind mean {null['d_blind'].values.mean():.3f}, D_balanced "
          f"mean {null['d_balanced'].values.mean():.3f}; fallback folds per draw {fb.mean():.3f} "
          f"(any {np.mean(fb > 0):.3f}); unconverged {null['unconverged_folds'].values.mean():.3f}")
    bsd, cap = null["balanced_sd"].values, null["balanced_cap_share"].values
    print(f"  balanced leg: realised sd median {np.median(bsd):.2%}; cap binds median "
          f"{np.median(cap):.1%}; gross median {np.median(null['balanced_gross'].values):.2f}; "
          f"sleeves at zero per fold {null['zero_sleeves'].values.mean():.2f}; extra held "
          f"turnover median {np.median(null['extra_turnover'].values):+.2f}x/yr")
    v = red.values
    t = summary["bar"]
    print(f"  bar T_A = {t:.3f}; q(1 - alpha_S) {summary['q_sidak']:.3f}. Under the shift model "
          f"the binding line is Delta >= T_A: power at T_A = P(null >= 0) = "
          f"{summary['power_verdict']['T']:.3f}; at log 1.5 "
          f"{summary['power_verdict']['log 1.5']:.3f} (percentile line alone: "
          f"{summary['power_percentile_only']['log 1.5']:.3f}); 80% "
          f"needs a shift of {summary['shift_80']:.3f}; the operative size P(null >= T_A) = "
          f"{summary['size_at_bar']:.4f}")
    print(f"  P2 survival quantities: q95 {p2['q95']:.3f}, median {p2['q50']:.3f} (the symmetric "
          f"FAIL line); share of draws with Delta <= 0: {np.mean(v <= 0):.3f}")
    return {"summary": summary, "p2_vol": p2, "p2_gross": gross, "hygiene": hyg,
            "seconds": elapsed, "fallback_mean": float(fb.mean()),
            "fallback_any": float(np.mean(fb > 0)),
            "unconverged_mean": float(null["unconverged_folds"].values.mean()),
            "d_blind_mean": float(null["d_blind"].values.mean()),
            "d_balanced_mean": float(null["d_balanced"].values.mean()),
            "balanced_sd_median": float(np.median(bsd)),
            "balanced_cap_median": float(np.median(cap)),
            "balanced_gross_median": float(np.median(null["balanced_gross"].values)),
            "zero_sleeves_mean": float(null["zero_sleeves"].values.mean()),
            "extra_turnover_median": float(np.median(null["extra_turnover"].values)),
            "extra_turnover_q95": float(np.quantile(null["extra_turnover"].values, 0.95)),
            "share_nonpositive": float(np.mean(v <= 0)),
            "_values": v, "_d_blind": null["d_blind"].values, "_p2": null["p2_vol"].values}


def b2_section(obj: D.ContemporaneousObject, engine: I.GuardedLevelA, w_b2: np.ndarray,
               n: int) -> dict:
    section(f"4. B2: the map leg held fixed, the partition drawn ({n} draws, the A draws)")
    stats_ = {"reduction": lambda r: r.reduction,
              "p2_vol": lambda r: E.exposure_matched(r, obj.panel.excess, on="vol",
                                                     missing_as_zero=True).reduction}
    null = I.null_statistics(engine, obj.partition, n, statistics=stats_, seed=SEED,
                             blocks=obj.blocks(), weights=w_b2)
    s = summarise("B2: D(blind) - D(map leg)", null["reduction"])
    p2 = summarise("B2: P2 vol-matched, no cap", null["p2_vol"])
    print(LEGEND)
    print_nulls([s, p2])
    space = E.map_space_summary("permutation")
    print(f"  P3 space (permutation): {space.maps} maps, {space.budget_vectors} books, draft ties "
          f"{space.reference_ties}; min p {space.min_p_value:.4f}; max percentile "
          f"{space.max_percentile:.4f}; at most 4 maps (2 books) strictly above the draft map "
          "for a percentile >= 0.95")
    return {"summary": s, "p2_vol": p2, "_values": null["reduction"].values,
            "_p2": null["p2_vol"].values,
            "p3": {"maps": space.maps, "books": space.budget_vectors,
                   "ties": space.reference_ties}}


def b1_section(obj: D.ContemporaneousObject, engine: I.GuardedLevelA, b1: D.Partition,
               n: int) -> dict:
    section(f"5. B1: the market-implied axis on the quarterly grid ({n} draws)")
    s = obj.sessions
    paths = D.fold_paths(b1.labels, b1.available, s, obj.folds)
    clock = Q.clock(pd.Series(b1.labels.to_numpy(int), index=b1.labels.index.start_time))
    per = persistence(b1.labels)
    ses, epi, _ = test_counts(paths, s, obj.folds)
    qualify = []
    for f in obj.folds:
        a, b = fit_counts(paths[f.number], s, f)
        qualify.append(int(sum(x >= E.MIN_CELL_SESSIONS and y >= E.MIN_CELL_EPISODES
                               for x, y in zip(a, b, strict=True))))
    print(f"  {len(b1.labels)} quarters {b1.labels.index[0]} -> {b1.labels.index[-1]} (the 252-"
          f"session change at each quarter's last session); {clock.per_year:.2f} transitions/yr;"
          f" P(same) {per['same']:.3f} vs {per['independent']:.3f} independent (p "
          f"{per['p_le_observed']:.3f})")
    print(f"  pooled test sessions per cell {ses}, episodes {epi}; training cells qualifying per "
          f"fold {qualify}")
    blocks = D.quarter_blocks(b1.labels.index, s, obj.folds)
    hyg = hygiene(b1, blocks, n)
    print(f"  P1 freedom per block: {hyg['log10_orders']}; identical draws {hyg['identical']}; "
          f"agreement max {hyg['agreement_max']:.3f}; draws reproducing one block: "
          f"{hyg['identical_in_block']}")
    stats_ = {"reduction": lambda r: r.reduction,
              "fallback_folds": lambda r: float(len(r.fallback_folds)),
              "p2_vol": lambda r: E.exposure_matched(r, obj.panel.excess, on="vol",
                                                     missing_as_zero=True).reduction}
    null = I.null_statistics(engine, b1, n, statistics=stats_, seed=SEED, blocks=blocks)
    summary = summarise("B1: D(blind) - D(balanced)", null["reduction"])
    p2 = summarise("B1: P2 vol-matched, no cap", null["p2_vol"])
    print(LEGEND)
    print_nulls([summary, p2])
    fb = null["fallback_folds"].values
    print(f"  fallback folds per draw {fb.mean():.3f} (draws with any {np.mean(fb > 0):.3f})")
    return {"summary": summary, "p2_vol": p2, "clock_per_year": clock.per_year,
            "persistence": per, "test_cell_sessions": ses, "test_cell_episodes": epi,
            "train_cells_qualifying": qualify, "hygiene": hyg,
            "fallback_mean": float(fb.mean()), "fallback_any": float(np.mean(fb > 0)),
            "first": str(b1.labels.index[0]), "last": str(b1.labels.index[-1]),
            "_values": null["reduction"].values}


def witness_section(obj: D.ContemporaneousObject, engine: I.GuardedLevelA, w_b2: np.ndarray,
                    n: int, tests: Mapping[str, np.ndarray]) -> dict:
    section(f"6. The volatility witnesses: own reductions, own nulls ({n} draws), common scale")
    s = obj.sessions
    out: dict = {}
    w1 = E.volatility_witness_labels(obj.panel.excess["^GSPC"], obj.folds, sessions=s)
    w1_stitched = pd.Series(np.nan, index=s)
    first_test = obj.folds[0].test_start
    w1_stitched[s < first_test] = w1[obj.folds[0].number][s < first_test]
    for f in obj.folds:
        idx = f.test(s)
        w1_stitched.loc[idx] = w1[f.number].loc[idx]
    last = D.quarter_last_sessions(obj.partition.labels.index, s)
    per_fold, stitched = D.quarterly_volatility_labels(D.traded_blind_driver(engine), s,
                                                       obj.folds, last)
    stitched = stitched[stitched.index <= D.last_test_quarter(obj.folds)]
    w2 = D.Partition("w2", stitched, last.reindex(stitched.index))
    w2_paths = D.fold_paths(per_fold, last, s, obj.folds)
    blocks_q = D.quarter_blocks(w2.labels.index, s, obj.folds)
    blocks_s = I.session_blocks(s, obj.folds)
    span = w1_stitched.dropna()
    clocks = {"W1": Q.clock(span.astype("Int64")).per_year,
              "W2": Q.clock(pd.Series(w2.labels.to_numpy(int),
                                      index=w2.labels.index.start_time)).per_year}
    print(f"  W1: ^GSPC excess, 21-session realised vol, 4 training-quantile bins, lag 1, daily "
          f"({clocks['W1']:.2f}/yr). W2: the blind leg's traded unscaled return, realised vol of "
          f"each calendar quarter, 4 training-quantile bins, contemporaneous "
          f"({clocks['W2']:.2f}/yr)")
    rows = []
    for form, weights in (("A", None), ("B2", w_b2)):
        real = {"W1": engine.run(w1, weights=weights).reduction,
                "W2": engine.run(w2_paths, weights=weights).reduction}
        nulls = {
            "W1": _w1_null(engine, w1_stitched.loc[span.index[0]:], n, blocks_s, weights),
            "W2": I.null_statistics(engine, w2, n, statistics={"r": lambda r: r.reduction},
                                    seed=SEED, blocks=blocks_q, weights=weights)["r"],
        }
        for name in ("W1", "W2"):
            z = I.z_score(real[name], nulls[name].values)
            row = {"form": form, "witness": name, "reduction": real[name], "z": z,
                   **{k: nulls[name].describe()[k] for k in ("mean", "sd", "q95", "atom_share")},
                   "fingerprint": fingerprint(nulls[name].values),
                   "percentile": float(I.pct_rank(nulls[name].values, real[name])[0]),
                   "_values": nulls[name].values}
            rows.append(row)
            print(f"  {form:2} {name}: own reduction {real[name]:+.3f}; its null mean "
                  f"{row['mean']:+.3f} sd {row['sd']:.3f}; percentile {row['percentile']:.4f}; "
                  f"z {z:+.2f}")
    out["rows"] = rows
    out["clocks"] = clocks
    # The effective bars (§12.15): Delta must clear T and exceed max z_W on the common scale.
    bars = {}
    for test, values in tests.items():
        form = "B2" if test == "B2" else "A"
        z = [r["z"] for r in rows if r["form"] == form]
        null = E.NullDistribution(values)
        t = I.bar(null, SIDAK, E.DRAFT_MDE)
        eff = I.effective_bar(values, t, z)
        bars[test] = {"T": t, "max_witness_z": float(max(z)), "effective": eff,
                      "power_at_effective": I.verdict_power(values, eff, eff, SIDAK),
                      "shift_80_effective": I.shift_for_power(values, eff, SIDAK)}
        print(f"  {test}: T {t:.3f}; largest witness z {max(z):+.2f} -> effective bar "
              f"{eff:.3f}; 80% power under the shift model needs "
              f"{bars[test]['shift_80_effective']:.3f}")
    out["effective_bars"] = bars
    w2_counts = w2.labels.value_counts().sort_index().to_dict()
    out["w2_quarters_per_bin"] = {str(k): int(v) for k, v in w2_counts.items()}
    return out


def _w1_null(engine: I.GuardedLevelA, path: pd.Series, n: int, blocks: pd.Series,
             weights: np.ndarray | None) -> E.NullDistribution:
    """W1's null: its daily P1 draws per session block, the balanced leg refitted on each
    draw (A form) or the map leg held fixed (B2 form)."""
    paths = E.placebo_session_paths(engine, path, n, seed=SEED, blocks=blocks, stamped=False)
    return E.NullDistribution(np.array([engine.run(paths[c], weights=weights).reduction
                                        for c in paths.columns]))


def reproduction_section(stamped: D.StampedObject, n: int) -> dict:
    section(f"7. Reproduction of power.json's per-block stamped null ({n} draws, one BLAS thread)")
    power = json.loads(POWER_JSON.read_text())
    entry = next(e for e in power["sensitivities"] if e["name"].startswith("P1 within"))
    engine = I.GuardedLevelA(stamped.panel.sleeve_returns, stamped.folds,
                             leg_builder=D.leg_builder(stamped.panel))
    engine.forbid(stamped.real_path)
    stamps = E.stamps_in_force(stamped.stamped, stamped.sessions)
    blocks = E.calendar_blocks(stamps.index, stamped.sessions, stamped.folds)
    null = E.null_statistics(engine, stamped.stamped, n, statistics={"r": lambda r: r.reduction},
                             seed=SEED, blocks=blocks)["r"]
    now = null.describe()
    keys = ("mean", "sd", "q20", "q95", "q99", "mde_shift_05", "mde_shift_sidak", "mde_sd_05",
            "mde_sd_sidak")
    gaps = {k: abs(float(now[k]) - float(entry[k])) for k in keys}
    worst = max(gaps.values())
    print("  " + ", ".join(f"{k} {now[k]:.6f} (|gap| {gaps[k]:.1e})" for k in keys))
    print(f"  largest gap {worst:.2e}: {'REPRODUCES to 1e-9' if worst <= 1e-9 else 'DOES NOT'}")
    return {"gaps": gaps, "largest_gap": worst, "reproduces": bool(worst <= 1e-9),
            "summary": summarise("stamped label, per block (sensitivity)", null),
            "_values": null.values}


# ------------------------------------------------------------------------- level C


def witness_stamp_dates(driver: pd.Series, stamped: D.StampedObject) -> tuple[
        pd.DatetimeIndex, pd.Series]:
    """§12.15 at C: witness rebalances (fold-consistent bins) and the stitched stamp sequence."""
    s = stamped.sessions
    stamps = E.stamps_in_force(stamped.stamped, s)
    paths = E.volatility_witness_labels(driver, stamped.folds, sessions=s)
    blocks = E.calendar_blocks(stamps.index, s, stamped.folds)
    pos = s.searchsorted(stamps.index, side="left") + 1
    ok = pos < len(s)
    fold_of = {"0:train": stamped.folds[0].number,
               **{f"{f.number}:test": f.number for f in stamped.folds}}
    bins = {}
    stitched = []
    for tau, p, b in zip(stamps.index[ok], pos[ok], blocks[ok], strict=True):
        f = fold_of[b]
        bins[tau] = {g.number: paths[g.number].iloc[p] for g in stamped.folds}
        stitched.append(bins[tau][f])
    stitched = pd.Series(stitched, index=stamps.index[ok], dtype=float)
    moves = []
    taus = list(bins)
    for prev, tau in zip(taus[:-1], taus[1:], strict=True):
        f = fold_of[blocks[tau]]
        a, b = bins[prev][f], bins[tau][f]
        if np.isfinite(a) and np.isfinite(b) and a != b and s[0] <= tau <= s[-1]:
            moves.append(tau)
    return pd.DatetimeIndex(moves), stitched.dropna()


def c_section(stamped: D.StampedObject, n: int, workers: int) -> dict:
    section(f"8. Level C on the stamped object: placebo conditioned on the real transition "
            f"count ({n} draws)")
    s = stamped.sessions
    test = stamped.test_sessions()
    excess = stamped.panel.excess
    sleeves = stamped.panel.sleeves
    stamps = E.stamps_in_force(stamped.stamped, s)
    blocks = E.calendar_blocks(stamps.index, s, stamped.folds)
    calendar = pd.DatetimeIndex([t for t in stamped.stamped.index if s[0] <= t <= s[-1]])
    real_moves = I.transitions_between(stamped.stamped, s[0], s[-1])
    target = I.count_in_window(real_moves, s, test)
    n_cal = I.count_in_window(calendar, s, test)
    print(f"  calendar: every SPF stamp in the sample ({len(calendar)}; {n_cal} held first on a "
          f"test session); transitions of the real label: {len(real_moves)} in the sample, "
          f"{target} held first on a test session (label-only)")
    accepted, ids = I.conditioned_draws(stamps, n, seed=SEED, blocks=blocks, sessions=s,
                                        window=test, target=target)
    print(f"  per-block P1 draws with exactly {target} test-span transitions: the first {n} "
          f"are draws 0..{ids[-1]} ({n / (ids[-1] + 1):.1%} accepted)")
    all_counts = []
    probe = E.p1_draws(stamps, 2000, seed=SEED, blocks=blocks.reindex(stamps.index)).draws
    for d in probe.columns:
        all_counts.append(I.count_in_window(I.transitions_between(probe[d], s[0], s[-1]), s, test))
    all_counts = np.array(all_counts)
    print(f"  unconditioned per-block draws: test-span count mean {all_counts.mean():.2f}, sd "
          f"{all_counts.std(ddof=1):.2f}, equal to {target} in "
          f"{np.mean(all_counts == target):.1%}; "
          f"the real count's percentile {I.pct_rank(all_counts, target)[0]:.3f}")
    book, _, _ = S.blind_book(excess, sleeves, list(calendar))
    cal_net = S.net_returns(book).reindex(test)
    cal_sr = I.sharpe(cal_net)
    dates = tuple(I.transitions_between(accepted[d], s[0], s[-1]) for d in accepted.columns)
    setup = I.CSetup(excess, sleeves, dates, test, cal_sr)
    t0 = time.time()
    values = np.array(draw_map(I.c_placebo_task, n, setup, workers=workers), dtype=float)
    print(f"  {n} placebo books in {time.time() - t0:.0f} s")
    null = E.NullDistribution(values)
    # The bootstrap SE of the paired Sharpe difference, from placebo pairs (demeaned).
    se_rows = {b: [] for b in E.POWER_BLOCKS}
    for d in range(min(20, n)):
        pb, _, _ = S.blind_book(excess, sleeves, list(dates[d]))
        a = S.net_returns(pb).reindex(test).to_numpy(float)
        for b in E.POWER_BLOCKS:
            r = blinded_mde(a, cal_net.to_numpy(float), mean_block=b, alpha=SIDAK)
            se_rows[b].append((r.se, r.mde, mde_at(r, SIDAK)))
    se = {b: float(np.median([x[0] for x in rows])) for b, rows in se_rows.items()}
    boot_mde = {b: float(np.median([x[2] for x in rows])) for b, rows in se_rows.items()}
    se_star = max(se.values())
    t_c = float(max(max(boot_mde.values()), null.mde_shift(SIDAK), null.mde_sd(SIDAK)))
    summary = summarise("C: SR(placebo dates) - SR(every stamp)", null, floor=0.0, se=se_star)
    summary["bar"] = t_c
    summary["power_verdict"] = {k: I.verdict_power(values, v, t_c, SIDAK, se=se_star)
                                for k, v in {"0.05": 0.05, "0.10": 0.10, "T": t_c}.items()}
    summary["shift_80"] = I.shift_for_power(values, t_c, SIDAK, se=se_star)
    q = null.quantile([0.2, 0.5, 0.99])
    print(f"  placebo null: mean {values.mean():+.4f} sd {null.sd:.4f} skew {null.skewness:+.2f} "
          f"atom {null.atom_share:.4f}; q20 {q[0]:+.4f} q50 {q[1]:+.4f} q99 {q[2]:+.4f}")
    print("  bootstrap SE of placebo pairs (median of 20, demeaned): "
          + ", ".join(f"block {b} {se[b]:.4f}" for b in se)
          + f"; SE* {se_star:.4f}; placebo sd / SE* {null.sd / se_star:.2f}")
    print(f"  T_C = max(bootstrap MDE {max(boot_mde.values()):.4f}, placebo MDE_shift "
          f"{null.mde_shift(SIDAK):.4f}, placebo MDE_sd {null.mde_sd(SIDAK):.4f}) = {t_c:.4f}; "
          f"verdict power at T_C {summary['power_verdict']['T']:.3f}; 80% at a shift of "
          f"{summary['shift_80']:.4f}")
    # Witness books at C: how many stamps each skips (label-free).
    wit = {}
    blind_driver = _stamped_blind_driver(stamped)
    for name, driver in (("W1", excess["^GSPC"]), ("W2", blind_driver)):
        moves, _ = witness_stamp_dates(driver, stamped)
        wit[name] = {"rebalances_test": I.count_in_window(moves, s, test),
                     "skipped_share": 1.0 - I.count_in_window(moves, s, test) / n_cal}
    print(f"  stamps skipped on the test span: quadrant {1 - target / n_cal:.1%}; witness books "
          + ", ".join(f"{k} {v['skipped_share']:.1%} ({v['rebalances_test']} rebalances)"
                      for k, v in wit.items()))
    return {"summary": summary, "se": se, "boot_mde_sidak": boot_mde, "se_star": se_star,
            "T_C": t_c, "target_count": target, "calendar_on_test": n_cal,
            "accepted_last_draw": int(ids[-1]),
            "unconditioned_count_mean": float(all_counts.mean()),
            "unconditioned_count_sd": float(all_counts.std(ddof=1)),
            "unconditioned_equal_share": float(np.mean(all_counts == target)),
            "witness_books": wit, "_values": values}


def _stamped_blind_driver(stamped: D.StampedObject) -> pd.Series:
    engine = E.LevelA(stamped.panel.sleeve_returns, stamped.folds,
                      leg_builder=D.leg_builder(stamped.panel))
    return D.traded_blind_driver(engine)


# --------------------------------------------------------------------- planted


SCENARIOS = {
    # Equity variance up in (growth-, inflation+), duration variance up in (growth+, inflation+).
    "sleeve-specific": {1: S.HEADLINE_SLEEVES["equity"], 3: S.HEADLINE_SLEEVES["duration"]},
    # Every instrument's variance up in (growth-, inflation+): a volatility proxy.
    "volatility proxy": {1: tuple(S.instruments(S.HEADLINE_SLEEVES))},
}
FACTORS = (1.25, 1.5, 2.0, 3.0)


def planted_section(obj: D.ContemporaneousObject, a: dict, witness: dict, unscaled: dict,
                    k: int, workers: int) -> dict:
    section(f"9. Planted alternatives on {k} content-free partitions (the first {k} A draws)")
    draws = I.quarter_draws(obj.partition.labels, k, seed=SEED, blocks=obj.blocks())
    tasks = tuple((d, sc, r, book) for sc in SCENARIOS for r in FACTORS for d in range(k)
                  for book in ("targeted", "unscaled"))
    setup = I.PlantedSetup(obj.panel.excess, "headline", obj.stamped, obj.folds,
                           obj.partition.labels.index, obj.partition.available,
                           draws.to_numpy(float), tasks, SCENARIOS)
    t0 = time.time()
    results = draw_map(I.planted_task, len(tasks), setup, workers=workers)
    print(f"  {len(tasks)} planted worlds in {time.time() - t0:.0f} s")
    null_t = a["_values"]
    t_a = a["summary"]["bar"]
    p2_null = a["_p2"]
    z_rows = {r["witness"]: r for r in witness["rows"] if r["form"] == "A"}
    null_u = unscaled["_values"]
    t_u = unscaled["summary"]["bar"]
    base_d = a["_d_blind"][:k]
    table = []
    print(f"  {'scenario':18} {'r':>5} {'book':9} {'dD_blind':>8} {'mean D':>7} {'line5':>6} "
          f"{'P2 ok':>6} {'dom W':>6} {'PASS-ok':>7} {'FAIL':>6}")
    for sc in SCENARIOS:
        for r in FACTORS:
            for book in ("targeted", "unscaled"):
                idx = [j for j, t in enumerate(tasks) if t[1] == sc and t[2] == r and t[3] == book]
                red = np.array([results[j]["reduction"] for j in idx])
                db = np.array([results[j]["d_blind"] for j in idx])
                if book == "targeted":
                    line5 = (red >= t_a) & (I.p_values(null_t, red) <= SIDAK)
                    p2v = np.array([results[j]["p2_vol"] for j in idx])
                    p2ok = (p2v > 0) & (I.pct_rank(p2_null, p2v) >= 0.95)
                    z_a = np.array([I.z_score(x, null_t) for x in red])
                    z1 = np.array([I.z_score(results[j]["w1"], _null_of(z_rows["W1"]))
                                   for j in idx])
                    z2 = np.array([I.z_score(results[j]["w2"], _null_of(z_rows["W2"]))
                                   for j in idx])
                    dom = (z1 >= z_a) | (z2 >= z_a)
                    ok = line5 & p2ok & ~dom
                    fail = (red <= 0) & ((p2v <= 0) | (I.pct_rank(p2_null, p2v) < 0.5))
                    row = {"scenario": sc, "r": r, "book": book,
                           "d_blind_shift": float(np.mean(db - base_d)),
                           "mean_reduction": float(red.mean()), "line5": float(line5.mean()),
                           "p2_survives": float(p2ok.mean()), "dominated": float(dom.mean()),
                           "pass_eligible": float(ok.mean()), "fail": float(fail.mean())}
                else:
                    line5 = (red >= t_u) & (I.p_values(null_u, red) <= SIDAK)
                    row = {"scenario": sc, "r": r, "book": book,
                           "d_blind_shift": float("nan"), "mean_reduction": float(red.mean()),
                           "line5": float(line5.mean()), "p2_survives": float("nan"),
                           "dominated": float("nan"), "pass_eligible": float("nan"),
                           "fail": float(np.mean(red <= 0))}
                table.append(row)
                print(f"  {sc:18} {r:5.2f} {book:9} {row['d_blind_shift']:8.3f} "
                      f"{row['mean_reduction']:7.3f} {row['line5']:6.3f} {row['p2_survives']:6.3f} "
                      f"{row['dominated']:6.3f} {row['pass_eligible']:7.3f} {row['fail']:6.3f}")
    # The approximation above reads each planted Delta against the UNPLANTED null. Check it
    # on two planted worlds, with their own P1 null (another seed, 300 draws).
    checks = []
    for sc in SCENARIOS:
        labels = pd.Series(draws[0].to_numpy(float), index=obj.partition.labels.index)
        path = D.contemporaneous_path(labels, obj.sessions)
        factors = {c: dict.fromkeys(m, 2.0) for c, m in SCENARIOS[sc].items()}
        panel = D.sleeve_panel("headline", obj.stamped,
                               excess=I.plant_variance(obj.panel.excess, path, factors))
        planted_engine = E.LevelA(panel.sleeve_returns, obj.folds,
                                  leg_builder=D.leg_builder(panel))
        world = D.Partition("planted", labels, obj.partition.available)
        own = I.null_statistics(planted_engine, world, 300, statistics={"r": lambda r: r.reduction},
                                seed=SEED + 1, blocks=obj.blocks())["r"]
        checks.append({"scenario": sc, "r": 2.0, "sd": own.sd, "mean": float(own.values.mean()),
                       "q_sidak": float(own.quantile(1 - SIDAK)), "bar": I.bar(own, SIDAK)})
        print(f"  own null of a planted world ({sc}, r 2, 300 draws): mean "
              f"{checks[-1]['mean']:+.3f} sd {own.sd:.3f} q(1-aS) {checks[-1]['q_sidak']:.3f} bar "
              f"{checks[-1]['bar']:.3f}; the unplanted null: mean {null_t.mean():+.3f} sd "
              f"{np.std(null_t, ddof=1):.3f} bar {t_a:.3f}")
    print("  line5: Delta >= T and p_P1 <= alpha_S against the unplanted null (the bootstrap line "
          "is not simulated);\n  P2 ok: vol-matched reduction > 0 and >= the 95th percentile of "
          "its null; dom W: a witness's z\n  (own reduction on the planted returns) >= the "
          "test's z; PASS-ok: line5, P2 and not dominated; FAIL:\n  Delta <= 0 with the "
          "vol-matched reading not positive or below its null median. The unscaled book\n  "
          "(sleeve level, no target, no cost) is read against its own null and bar.")
    return {"rows": table, "k": k, "factors": list(FACTORS), "own_null_checks": checks,
            "scenarios": {k_: {str(c): list(v) for c, v in sc.items()}
                          for k_, sc in SCENARIOS.items()}}


def _null_of(row: dict) -> np.ndarray:
    """The witness's own null draws (its z needs only their mean and sd)."""
    return row["_values"]


# -------------------------------------------------------------------------- main


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list | tuple):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return str(obj)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--repro-draws", type=int, default=1000)
    parser.add_argument("--c-draws", type=int, default=2000)
    parser.add_argument("--planted", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    t0 = time.time()
    with threadpool_limits(limits=1):
        results = run(args)
    results["seconds"] = time.time() - t0
    print(f"\n  total {results['seconds']:.0f} s")
    if args.json:
        if args.json.startswith("/"):
            raise SystemExit("--json takes a path relative to the repository root")
        path = ROOT / args.json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(jsonable(results), indent=1, default=str) + "\n")
        print(f"  written: {args.json}")


def run(args: argparse.Namespace) -> dict:
    results: dict = {"seed": SEED, "draws": args.draws, "alpha_sidak": SIDAK}
    section("0. Inputs, content hashes and versions")
    bytes_ = D.byte_hashes()
    content = D.content_hashes()
    for r in D.INPUT_FILES:
        print(f"  {r:48} bytes {bytes_[r][:16]}  content {content[r][:16]}")
    code = ["scripts/measure_bridgewater_lock.py", "scripts/measure_bridgewater_power.py",
            *(f"regime_lab/construction/{m}.py" for m in ("declared", "inference", "evaluate",
                                                            "quadrant", "sleeves"))]
    dirty = []
    for relative in code:
        try:
            require_committed(ROOT / relative, ROOT)
        except ReadingRefused:
            dirty.append(relative)
    head = git_head(ROOT)
    print(f"  code at HEAD {head[:12]}; " + ("every code file committed and unchanged" if not dirty
                                              else f"NOT COMMITTED OR CHANGED: {dirty}"))
    results.update({"git_head": head, "uncommitted_code": dirty})
    versions = {p: metadata.version(p) for p in PACKAGES}
    print("  " + ", ".join(f"{k} {v}" for k, v in versions.items()) + "; BLAS threads 1")
    results.update({"byte_hashes": bytes_, "content_hashes": content, "versions": versions})

    inputs = D.load_quadrant_inputs()
    stamped = D.declared_object(inputs=inputs)
    obj = D.contemporaneous_object(inputs=inputs)
    results["objects"] = objects_section(stamped, obj, inputs)

    engine = obj.engine(I.GuardedLevelA)
    engine.forbid(obj.paths())
    engine.forbid(stamped.real_path)
    wv = D.contemporaneous_object(inputs=inputs, growth="growth_within_vintage", folds=obj.folds)
    engine.forbid(wv.paths())
    t10 = validate(pd.read_parquet(RAW / "spf" / Q.T10YIE_FILE))
    baa = validate(pd.read_parquet(RAW / "macro" / "fin_baa_spread.parquet"))
    b1 = D.b1_partition(t10, baa, obj.sessions, through=D.last_test_quarter(obj.folds))
    engine.forbid(D.fold_paths(b1.labels, b1.available, obj.sessions, obj.folds))
    changed = (wv.partition.path(obj.sessions) != obj.partition.path(obj.sessions)) & \
        wv.partition.path(obj.sessions).notna()
    print(f"  within-vintage axis: {int(changed.sum())} sessions relabelled against the declared "
          f"axis; {int(wv.partition.path(obj.sessions).isna().sum())} unlabelled")
    results["within_vintage_relabelled"] = int(changed.sum())

    w_b2 = E.b2_fold_weights(engine, E.DRAFT_MAP, rule="inverse_vol")
    results["books"] = blind_books_section(obj, engine, w_b2)
    a = a_section(obj, engine, args.draws)
    results["A"] = a
    b2 = b2_section(obj, engine, w_b2, args.draws)
    results["B2"] = b2
    b1r = b1_section(obj, engine, b1, args.draws)
    results["B1"] = b1r

    unscaled_engine = obj.engine(I.GuardedLevelA, leg_builder=None, scaling="none")
    unscaled_engine.forbid(obj.paths())
    un = I.null_statistics(unscaled_engine, obj.partition, args.draws,
                           statistics={"r": lambda r: r.reduction}, seed=SEED,
                           blocks=obj.blocks())["r"]
    unscaled = {"summary": summarise("A on the unscaled book (no target, no cost)", un),
                "_values": un.values}
    print_nulls([unscaled["summary"]])
    results["A_unscaled"] = unscaled

    witness = witness_section(obj, engine, w_b2, args.draws,
                              {"A": a["_values"], "B1": b1r["_values"], "B2": b2["_values"]})
    results["witness"] = witness
    results["reproduction"] = reproduction_section(stamped, args.repro_draws)
    results["C"] = c_section(stamped, args.c_draws, args.workers)
    results["planted"] = planted_section(obj, a, witness, unscaled, args.planted, args.workers)

    section("10. The guard")
    print(f"  LevelA.run on the guarded engines: {engine.runs:,} (A, B1, B2, witnesses) and "
          f"{unscaled_engine.runs:,} (unscaled); refused none; the closest path to a real one "
          f"agreed on {max(engine.closest_seen, unscaled_engine.closest_seen):.3f} of sessions "
          f"(the guard refuses at {I.GUARD_AGREEMENT})")
    results["guard"] = {"runs": engine.runs + unscaled_engine.runs,
                        "closest": max(engine.closest_seen, unscaled_engine.closest_seen),
                        "threshold": I.GUARD_AGREEMENT}
    return results


if __name__ == "__main__":
    main()
