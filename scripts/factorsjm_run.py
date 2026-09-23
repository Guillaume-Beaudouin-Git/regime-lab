"""One sparse jump model per factor (Shu and Mulvey, arXiv 2410.14841), then one reading.

The protocol is `docs/PRESPEC_FACTORSJM.md`. EVERYTHING BELOW IS FIXED, AND COMMITTED,
BEFORE ANY RETURN CONDITIONED ON A STATE IS READ.

Three phases, run in this order, each by hand:

    .venv/bin/python scripts/factorsjm_run.py --fit     # hours: every candidate path,
                                                        # then the paper's tuning
                                                        # (--select redoes the tuning only)
    .venv/bin/python scripts/factorsjm_run.py           # instrument: axes, MDE, guards;
                                                        # no conditioned return printed
    .venv/bin/python scripts/factorsjm_run.py --read    # THE reading, once: 18 trials

`--synthetic` runs the whole chain on simulated factors, writes nothing under `data/` or
`docs/`, and logs to a scratch register: it exists to find bugs before the one reading.

The six factors: Mkt-RF, SMB, HML, RMW, CMA (Ken French five-factor file, daily, from
1963-07-01) and UMD (Ken French momentum file). Each has its own sparse jump model on the
paper's features (`regime_lab/extensions/factorsjm.py`); the penalty is re-chosen every
six months among `GRID` by the paper's rule.

The family (8 tests, alpha 0.05 / 8 in every MDE)
    B1  book on/off: six volatility-targeted sleeves, each held in its bull state and
        flat in its bear state, against the same six sleeves always held       PRIMARY
    B2  book, the paper's exposure clip(mu_state / 5%, -1, 1), against the same
    F_k the paper's exposure on sleeve k alone against sleeve k always held, k = 6
Sensitivities, logged, never deciding: B1 and B2 at a fixed penalty of 50 (the paper's
example); B1 and B2 with the market sleeve held (the paper times no market).
Replication block, logged in its own family: the paper's raw single-factor long-short
strategy, 2007-01-03 -> 2024-06-28, against the published Sharpe ratios.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from regime_lab.analysis import trials
from regime_lab.config import CACHE, RAW, ROOT
from regime_lab.evaluation.predictive import volatility_quantile_placebo
from regime_lab.extensions import crisis
from regime_lab.extensions import factorsjm as fj
from regime_lab.selection.protocol import blinded_mde, mde_at, paired_hac_t

RULE = "=" * 84
FAMILY = "factorsjm"
FAMILY_REP = "factorsjm_replication"
FACTORS = ("mkt", "smb", "hml", "rmw", "cma", "umd")
GRID = (10.0, 20.0, 50.0, 100.0, 200.0, 500.0)
FIXED_PENALTY = 50.0
MAX_FEATS = 9.5
REFIT_MONTHS = 6
FIRST_REFIT = pd.Timestamp("1972-01-01")
FIRST_TUNE = pd.Timestamp("1978-01-01")
PAPER_WINDOW = (pd.Timestamp("2007-01-03"), pd.Timestamp("2024-06-28"))
N_TESTS = 8
ALPHA = 0.05 / N_TESTS
BLOCKS = (21, 63, 126)
DRAWS = 2000
ROTATIONS = 400
MIN_SHIFT = 252
FOLDS = 5
WORKERS = 4
TREND_WINDOW = 252

#: Shu and Mulvey, single-factor long-short strategy, 2007-2024: Sharpe, shifts a year.
PAPER = {"hml": ("Value", 0.39, 3.16), "smb": ("Size", 0.20, 2.57),
         "umd": ("Momentum", 0.16, 3.66), "rmw": ("Quality", 0.21, 0.64)}

PATHS = CACHE / "factorsjm_paths.parquet"
STATES = CACHE / "factorsjm_states.parquet"
TUNING = CACHE / "factorsjm_tuning.parquet"
ARTIFACTS = ROOT / "docs" / "artifacts" / "factorsjm"
INPUTS = (
    RAW / "factorsjm" / "factors_5_full.parquet",
    RAW / "crisis" / "french_umd.parquet",
    RAW / "factorsjm" / "dgs1.parquet",
    RAW / "factorsjm" / "dgs10.parquet",
)


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
class Tee:
    """Print to the terminal and to a file at once."""

    def __init__(self, path: Path | None) -> None:
        self.stdout = sys.stdout
        self.file = path.open("w") if path is not None else None

    def write(self, text: str) -> None:
        self.stdout.write(text)
        if self.file is not None:
            self.file.write(text)

    def flush(self) -> None:
        self.stdout.flush()
        if self.file is not None:
            self.file.flush()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_real() -> dict:
    ff = pd.read_parquet(INPUTS[0]).pivot(index="period", columns="series_id", values="value")
    ff.index = pd.DatetimeIndex(ff.index)
    ff = ff / 100.0
    umd = pd.read_parquet(INPUTS[1]).set_index("period")["value"]
    umd.index = pd.DatetimeIndex(umd.index)
    factors = pd.DataFrame({
        "mkt": ff["ff_mkt-rf"], "smb": ff["ff_smb"], "hml": ff["ff_hml"],
        "rmw": ff["ff_rmw"], "cma": ff["ff_cma"], "umd": umd.reindex(ff.index),
    }).dropna()
    rates = {}
    for name, path in (("y1", INPUTS[2]), ("y10", INPUTS[3])):
        frame = pd.read_parquet(path)
        one = frame.set_index("available_at")["value"].astype(float)
        one.index = pd.DatetimeIndex(one.index)
        rates[name] = one.dropna().sort_index()
    return {"factors": factors, "y1": rates["y1"], "y10": rates["y10"]}


def load_synthetic(seed: int = 11) -> dict:
    """Simulated factors with two-state means, same calendar as the real ones."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("1990-01-01", "2026-07-31")
    n = len(index)
    out = {}
    for k, name in enumerate(FACTORS):
        state = np.zeros(n)
        pos, s = 0, 1
        while pos < n:
            length = int(rng.integers(100, 500))
            state[pos:pos + length] = s
            pos, s = pos + length, 1 - s
        mu = np.where(state == 1, 0.0006, -0.0003)
        out[name] = mu + rng.normal(0, 0.006 + 0.001 * k, n)
    factors = pd.DataFrame(out, index=index)
    days = pd.date_range(index.min(), index.max(), freq="D")
    y1 = pd.Series(4 + np.cumsum(rng.normal(0, 0.03, len(days))), index=days)
    y10 = y1 + 1 + np.cumsum(rng.normal(0, 0.02, len(days)))
    return {"factors": factors, "y1": y1, "y10": y10}


def features(data: dict) -> dict[str, pd.DataFrame]:
    f = data["factors"]
    market = fj.market_features(f["mkt"], data["y1"], data["y10"])
    out = {}
    for name in FACTORS:
        if name == "mkt":
            own = fj.active_features(f["mkt"])
            out[name] = pd.concat([own, market.drop(columns=["mkt_ret_ewm_21"])], axis=1)
        else:
            own = fj.active_features(f[name], f["mkt"])
            out[name] = pd.concat([own, market], axis=1)
    return out


def sleeves(factors: pd.DataFrame) -> pd.DataFrame:
    """Each factor held at min(10% / sigma_63, 3), sigma lagged one session."""
    return pd.DataFrame({k: crisis.vol_target_weight(factors[k]) * factors[k]
                         for k in factors.columns})


# ---------------------------------------------------------------------------
# phase 1: fit
# ---------------------------------------------------------------------------
def _one_path(args: tuple) -> pd.DataFrame:
    name, penalty, feats, active, sleeve, refits = args
    path = fj.walk_forward(feats, active, refits, jump_penalty=penalty, max_feats=MAX_FEATS,
                           targets={"active": active, "sleeve": sleeve})
    path["factor"] = name
    path["penalty"] = penalty
    return path


def fit_paths(data: dict, grid: tuple[float, ...], first_refit: pd.Timestamp) -> pd.DataFrame:
    """Every candidate path: one walk-forward per factor and per penalty. No return is read."""
    f = data["factors"]
    feats = features(data)
    sl = sleeves(f)
    refits = fj.refit_schedule(f.index, first_refit, months=REFIT_MONTHS)
    jobs = [(k, lam, feats[k], f[k], sl[k], refits) for k in FACTORS for lam in grid]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        paths = list(pool.map(_one_path, jobs))
    return pd.concat(paths).rename_axis("date").reset_index()


def select(data: dict, allpaths: pd.DataFrame, grid: tuple[float, ...],
           first_refit: pd.Timestamp, first_tune: pd.Timestamp
           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The paper's tuning rule, factor by factor, on the cached candidate paths."""
    f = data["factors"]
    refits = fj.refit_schedule(f.index, first_refit, months=REFIT_MONTHS)
    tunes = [r for r in refits if r >= first_tune]
    stitched, scores = [], []
    for k in FACTORS:
        cand = {lam: allpaths[(allpaths.factor == k) & (allpaths.penalty == lam)]
                .set_index("date").drop(columns=["factor", "penalty"]) for lam in grid}
        chosen, table = fj.select_penalty(cand, f[k], tunes)
        chosen["factor"] = k
        table["factor"] = k
        stitched.append(chosen.rename_axis("date").reset_index())
        scores.append(table.reset_index())
    tuning = pd.concat(scores)
    tuning["chosen"] = tuning["chosen"].astype(float)
    return pd.concat(stitched), tuning


# ---------------------------------------------------------------------------
# the arms
# ---------------------------------------------------------------------------
def exposures_from_path(long: pd.DataFrame, index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    onoff, paper, raw, state = {}, {}, {}, {}
    for k in FACTORS:
        p = long[long.factor == k].set_index("date").reindex(index)
        state[k] = p["state"]
        onoff[k] = fj.onoff_exposure(p["state"])
        paper[k] = fj.paper_exposure(fj.current_mu(p, "sleeve"))
        raw[k] = fj.paper_exposure(fj.current_mu(p, "active"))
    return {"state": pd.DataFrame(state), "onoff": pd.DataFrame(onoff),
            "paper": pd.DataFrame(paper), "raw": pd.DataFrame(raw)}


def control_labels(factors: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The one-line partitions, 1 = the good side, stamped on the session they are known."""
    return {
        "volmed": pd.DataFrame({k: volatility_quantile_placebo(factors[k]) for k in FACTORS}),
        "vol80": pd.DataFrame({k: crisis.volatility_tail_rule(factors[k]) for k in FACTORS}),
        "trend": pd.DataFrame({k: fj.trailing_return_rule(factors[k], TREND_WINDOW)
                               for k in FACTORS}),
        "static": pd.DataFrame(1.0, index=factors.index, columns=list(FACTORS)),
    }


def control_paper(labels: pd.DataFrame, sl: pd.DataFrame, refits, index) -> pd.DataFrame:
    out = {}
    for k in FACTORS:
        pm = fj.partition_means(labels[k], sl[k], refits, index)
        out[k] = fj.paper_exposure(fj.current_mu(pm, "target")).reindex(index)
    return pd.DataFrame(out)


def build(data: dict, paths: pd.DataFrame, states: pd.DataFrame, first_refit) -> dict:
    f = data["factors"]
    index = f.index
    sl = sleeves(f)
    refits = fj.refit_schedule(index, first_refit, months=REFIT_MONTHS)
    sjm = exposures_from_path(states, index)
    fixed = exposures_from_path(paths[paths.penalty == FIXED_PENALTY], index)
    labels = control_labels(f)
    exp = {
        "sjm_onoff": sjm["onoff"], "sjm_paper": sjm["paper"], "sjm_raw": sjm["raw"],
        "fixed_onoff": fixed["onoff"], "fixed_paper": fixed["paper"],
        "uncond": pd.DataFrame(1.0, index=index, columns=list(FACTORS)),
    }
    for name in ("volmed", "vol80", "trend"):
        exp[f"{name}_onoff"] = fj.onoff_exposure(labels[name])
    for name in ("volmed", "vol80", "trend", "static"):
        exp[f"{name}_paper"] = control_paper(labels[name], sl, refits, index)
    for base in ("sjm_onoff", "sjm_paper", "fixed_onoff", "fixed_paper"):
        five = exp[base].copy()
        five["mkt"] = five["mkt"].where(five["mkt"].isna(), 1.0)
        exp[f"{base}_mkt_held"] = five

    held = {name: e.shift(fj.LAG) for name, e in exp.items()}
    ready = pd.concat([h.notna().all(axis=1) for h in held.values()], axis=1).all(axis=1)
    ready &= sl.notna().all(axis=1)
    start = ready[ready & (index >= states["date"].min())].index.min()
    oos = index[index >= start]
    return {"factors": f, "sleeves": sl.loc[oos], "held": {n: h.loc[oos] for n, h in held.items()},
            "exposure": exp, "state": sjm["state"], "fixed_state": fixed["state"],
            "labels": labels, "index": oos}


# ---------------------------------------------------------------------------
# measures
# ---------------------------------------------------------------------------
def sharpe(x: pd.Series) -> float:
    v = x.to_numpy(float)
    if not np.isfinite(v).all() or len(v) < 2 or not v.std(ddof=1) > 0:
        return float("nan")
    return float(v.mean() / v.std(ddof=1) * np.sqrt(252))


def max_drawdown(x: pd.Series) -> float:
    curve = (1.0 + x.fillna(0.0)).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def book(held: pd.DataFrame, sl: pd.DataFrame, columns=FACTORS) -> pd.Series:
    cols = list(columns)
    return (held[cols] * sl[cols]).mean(axis=1)


def describe(x: pd.Series, held: pd.DataFrame | None = None) -> dict:
    row = {"sharpe": sharpe(x), "ann_return": float(x.mean() * 252),
           "ann_vol": float(x.std(ddof=1) * np.sqrt(252)), "max_dd": max_drawdown(x)}
    if held is not None:
        row["turnover"] = float(held.diff().abs().mean().mean() * 252)
        row["gross"] = float(held.abs().mean().mean())
    return row


def scaled_t(a: pd.Series, b: pd.Series) -> float:
    """HAC-6 t of the daily difference, each leg divided by its own standard deviation."""
    return paired_hac_t(a / a.std(ddof=1), b / b.std(ddof=1))


def mde(a: pd.Series, b: pd.Series) -> tuple[float, list[float]]:
    values = []
    for block in BLOCKS:
        res = blinded_mde(a.to_numpy(), b.to_numpy(), mean_block=block, draws=DRAWS, seed=0)
        values.append(mde_at(res, ALPHA))
    return max(values), values


def participation(frame: pd.DataFrame) -> float:
    """Participation ratio of the columns that move; a constant column counts for nothing."""
    clean = frame.dropna()
    moving = clean.loc[:, clean.std() > 0]
    if moving.shape[1] == 0:
        return 0.0
    return fj.participation_ratio(moving.corr().to_numpy())


def tests(p: dict) -> list[dict]:
    """The family and the sensitivities: legs, the exposure behind each, controls."""
    sl, h = p["sleeves"], p["held"]
    u_book = book(h["uncond"], sl)
    out = []

    def add(name, role, arm, controls, columns=FACTORS, rot_cols=None):
        cols = list(columns)
        leg = book(h[arm], sl, cols)
        base = book(h["uncond"], sl, cols) if cols != list(FACTORS) else u_book
        out.append({"name": name, "role": role, "arm": arm, "columns": cols, "leg": leg,
                    "base": base, "controls": controls, "rot_cols": rot_cols or cols})

    add("B1_book_onoff", "primary", "sjm_onoff", ("volmed_onoff", "vol80_onoff", "trend_onoff"))
    add("B2_book_paper", "test", "sjm_paper",
        ("volmed_paper", "vol80_paper", "trend_paper", "static_paper"))
    for k in FACTORS:
        add(f"F_{k}_paper", "test", "sjm_paper",
            ("volmed_paper", "vol80_paper", "trend_paper", "static_paper"), columns=(k,))
    add("S1_book_onoff_fixed50", "sensitivity", "fixed_onoff",
        ("volmed_onoff", "vol80_onoff", "trend_onoff"))
    add("S2_book_paper_fixed50", "sensitivity", "fixed_paper",
        ("volmed_paper", "vol80_paper", "trend_paper", "static_paper"))
    add("S3_book_onoff_mkt_held", "sensitivity", "sjm_onoff_mkt_held",
        ("volmed_onoff", "vol80_onoff", "trend_onoff"))
    add("S4_book_paper_mkt_held", "sensitivity", "sjm_paper_mkt_held",
        ("volmed_paper", "vol80_paper", "trend_paper", "static_paper"))
    return out


def control_held(p: dict, test: dict, control: str) -> pd.DataFrame:
    """The control's held exposures, with the market held whenever the tested arm holds it."""
    held = p["held"][control].copy()
    if test["arm"].endswith("_mkt_held"):
        held["mkt"] = p["held"]["uncond"]["mkt"]
    return held[test["columns"]]


def control_book(p: dict, test: dict, control: str) -> pd.Series:
    return book(control_held(p, test, control), p["sleeves"], test["columns"])


# ---------------------------------------------------------------------------
# phase 2: instrument
# ---------------------------------------------------------------------------
def instrument(p: dict, paths: pd.DataFrame, tuning: pd.DataFrame, real: bool) -> dict:
    idx = p["index"]
    f = p["factors"]
    print(RULE)
    print("ONE SPARSE JUMP MODEL PER FACTOR — INSTRUMENT (no conditioned return is printed)")
    print(RULE)
    if real:
        print("\n0.  INPUTS (SHA-256)\n")
        for path in INPUTS:
            print(f"   {sha256(path)[:16]}  {path.relative_to(ROOT)}")
    if real:
        print("\n0b. DATA CHECKS\n")
        short = pd.read_parquet(RAW / "panels" / "factors_5.parquet").pivot(
            index="period", columns="series_id", values="value")
        short.index = pd.DatetimeIndex(short.index)
        full = pd.read_parquet(INPUTS[0]).pivot(index="period", columns="series_id",
                                                 values="value")
        full.index = pd.DatetimeIndex(full.index)
        gap = float((full.reindex(short.index) - short).abs().max().max())
        umd = pd.read_parquet(INPUTS[1])["period"]
        missing = len(full.index.difference(pd.DatetimeIndex(umd)))
        print(f"   max |full file - 1990 file| on {len(short):,} common sessions: {gap:.4f} "
              f"(percent)")
        print(f"   five-factor sessions without a UMD value: {missing}")
        if gap > 0 or missing > 0:
            print("   DATA CHECK FAILED. No verdict.")
            return {}
    print(f"\n1.  SAMPLES\n\n   factors {f.index.min():%Y-%m-%d} -> {f.index.max():%Y-%m-%d}, "
          f"{len(f):,} sessions")
    print(f"   out of sample (every leg finite) {idx.min():%Y-%m-%d} -> {idx.max():%Y-%m-%d}, "
          f"{len(idx):,} sessions, {len(idx) / 252:.1f} years")
    pw = idx[(idx >= PAPER_WINDOW[0]) & (idx <= PAPER_WINDOW[1])]
    print(f"   paper window {pw.min():%Y-%m-%d} -> {pw.max():%Y-%m-%d}, {len(pw):,} sessions")

    print(f"\n2.  TUNING — the penalty chosen every six months (count of choices), "
          f"features kept\n\n   {'factor':<7}" + "".join(f"{lam:>7g}" for lam in GRID)
          + f"{'none':>7}{'features':>10}")
    states_long = p["states_long"]
    for k in FACTORS:
        t = tuning[tuning.factor == k]
        counts = t["chosen"].value_counts()
        nsel = states_long[states_long.factor == k]["n_selected"].mean()
        print(f"   {k:<7}" + "".join(f"{int(counts.get(lam, 0)):>7}" for lam in GRID)
              + f"{int(t['chosen'].isna().sum()):>7}{nsel:>10.1f}")

    print("\n3.  THE FOUR AXES OF THE RULE (docs, CLAUDE.md), measured before any reading\n")
    state = p["state"].loc[idx]
    print("   axis 2 — transitions a year, out of sample; share of sessions in bull")
    total = 0.0
    for k in FACTORS:
        tr = fj.transitions_per_year(state[k])
        total += tr
        trp = fj.transitions_per_year(state[k].loc[pw])
        fx = fj.transitions_per_year(p["fixed_state"].loc[idx, k])
        ref = f"   paper {PAPER[k][0]} {PAPER[k][2]:.2f}" if k in PAPER else ""
        print(f"   {k:<5} {tr:>6.2f}/yr   2007-2024 {trp:>5.2f}/yr   fixed-50 {fx:>5.2f}/yr   "
              f"bull {state[k].mean():>6.1%}{ref}")
    print(f"   book: {total:.2f} factor-state changes a year in all")

    print("\n   axis 1 — is each latent a volatility rule, or a trailing-return rule?")
    print("   kappa of the bear state with: own 21-session vol above its expanding median;")
    print("   above its 80th percentile; the market's median rule; the factor's 252-session")
    print("   return negative. Last column: correlation of bear with log market RV21.")
    mkt_rule = volatility_quantile_placebo(f["mkt"]).reindex(idx)
    log_rv = np.log(f["mkt"].rolling(21).std()).reindex(idx)
    axis1 = {}
    for k in FACTORS:
        bear = state[k].eq(fj.BEAR).to_numpy()
        vals = []
        for lab in (p["labels"]["volmed"][k], p["labels"]["vol80"][k], mkt_rule,
                    p["labels"]["trend"][k]):
            lab = lab.reindex(idx)
            vals.append(fj.cohen_kappa(bear, lab.eq(0).to_numpy()))
        corr = float(np.corrcoef(bear.astype(float), log_rv.to_numpy())[0, 1])
        axis1[k] = vals + [corr]
        print(f"   {k:<5} own med {vals[0]:+.2f}   own 80th {vals[1]:+.2f}   market med "
              f"{vals[2]:+.2f}   trend {vals[3]:+.2f}   corr(RV) {corr:+.2f}")

    print("\n   axis 4 — participation ratio (effective dimension), of 6")
    raw90 = f.loc["1990-01-02":]
    pr = {
        "raw factors 1990-2026 (advisor: 4.71)": participation(raw90),
        "raw factors, out of sample": participation(f.loc[idx]),
        "volatility-targeted sleeves, out of sample": participation(p["sleeves"]),
        "on/off exposures (held), out of sample": participation(p["held"]["sjm_onoff"]),
        "paper exposures (held), out of sample": participation(p["held"]["sjm_paper"]),
    }
    for label, v in pr.items():
        print(f"   {label:<46} {v:.2f}")
    corr_e = p["held"]["sjm_onoff"].corr().to_numpy()
    off = corr_e[~np.eye(6, dtype=bool)]
    print(f"   mean pairwise correlation of the on/off exposures {off.mean():+.2f} "
          f"(min {off.min():+.2f}, max {off.max():+.2f})")
    print("   axis 3 — usage: an allocation across six factors, not a switch on one book")

    print(f"\n4.  MINIMUM DETECTABLE EFFECT — demeaned legs, alpha 0.05/{N_TESTS} = {ALPHA:.4f}, "
          "blocks 21 / 63 / 126, the largest\n")
    thresholds = {}
    bad = False
    for t in tests(p):
        nan = int(t["leg"].isna().sum() + t["base"].isna().sum())
        for c in t["controls"]:
            nan += int(control_book(p, t, c).isna().sum())
        bad |= nan > 0
        value, values = mde(t["leg"], t["base"])
        thresholds[t["name"]] = value
        joined = " / ".join(f"{v:.3f}" for v in values)
        print(f"   {t['name']:<26} {t['role']:<12} MDE {joined} -> {value:.3f}   NaN {nan}")
    if bad or not all(np.isfinite(v) for v in thresholds.values()):
        print("\n   A READING IS NOT FINITE. No verdict.")
        return {}
    return {"thresholds": thresholds, "axis1": axis1, "participation": pr}


# ---------------------------------------------------------------------------
# phase 3: the reading
# ---------------------------------------------------------------------------
def verdict(delta, mde_, t, pct, control_deltas: dict) -> str:
    values = [delta, mde_, t, pct, *control_deltas.values()]
    if not all(np.isfinite(values)):
        return "NOT FINITE — no verdict"
    if delta >= mde_:
        beats = all(delta > d for d in control_deltas.values())
        if np.sign(t) == np.sign(delta) and pct >= 0.95 and beats:
            return "USEFUL — all conditions hold"
        return "NOT SHOWN — above the MDE, fails the t, the placebo or a one-line rule"
    if delta > 0:
        return "UNDERPOWERED — positive, below the MDE, never useful"
    if delta > -mde_:
        return "NOT USEFUL — the regime does not raise the Sharpe"
    if np.sign(t) == np.sign(delta) and pct <= 0.05:
        return "HARMFUL — the regime measurably lowers the Sharpe"
    return "NOT USEFUL — negative beyond the MDE, not confirmed by the t or the placebo"


def rotation_null(p: dict, t: dict, shifts: np.ndarray, base_sharpe: float) -> np.ndarray:
    held = p["held"][t["arm"]][t["columns"]].to_numpy()
    sl = p["sleeves"][t["columns"]].to_numpy()
    out = []
    for k in shifts:
        leg = (np.roll(held, k, axis=0) * sl).mean(axis=1)
        out.append(sharpe(pd.Series(leg)) - base_sharpe)
    return np.array(out)


def lo_se(sr: float, years: float) -> float:
    return float(np.sqrt((1.0 + sr * sr / 2.0) / years))


def read(p: dict, inst: dict, log_path: Path | None) -> dict:
    idx = p["index"]
    rng = np.random.default_rng(20260923)
    shifts = rng.integers(MIN_SHIFT, len(idx) - MIN_SHIFT, size=ROTATIONS)
    folds = np.array_split(np.arange(len(idx)), FOLDS)
    pw = (idx >= PAPER_WINDOW[0]) & (idx <= PAPER_WINDOW[1])
    record: dict = {"tests": {}, "replication": {}}
    kwargs = {"path": log_path} if log_path is not None else {}

    for t in tests(p):
        leg, base = t["leg"], t["base"]
        held = p["held"][t["arm"]][t["columns"]]
        rows = {"alone (always held)": describe(base, p["held"]["uncond"][t["columns"]]),
                f"regime ({t['arm']})": describe(leg, held)}
        cdelta = {}
        for c in t["controls"]:
            cb = control_book(p, t, c)
            rows[c] = describe(cb, control_held(p, t, c))
            cdelta[c] = sharpe(cb) - sharpe(base)
        base_sr = sharpe(base)
        delta = sharpe(leg) - base_sr
        tt = scaled_t(leg, base)
        null = rotation_null(p, t, shifts, base_sr)
        pct = float((null < delta).mean()) if np.isfinite(null).all() else float("nan")
        thr = inst["thresholds"][t["name"]]
        v = verdict(delta, thr, tt, pct, cdelta)
        if t["role"] == "sensitivity":
            v = "SENSITIVITY, never deciding — " + v
        fold_d = [sharpe(leg.iloc[i]) - sharpe(base.iloc[i]) for i in folds]
        paper_d = sharpe(leg[pw]) - sharpe(base[pw])
        pre = idx < pd.Timestamp("1990-01-01")
        pre_d = sharpe(leg[pre]) - sharpe(base[pre])
        post_d = sharpe(leg[~pre]) - sharpe(base[~pre])
        diff = leg - base
        ir = float(diff.mean() / diff.std(ddof=1) * np.sqrt(252))
        ir_pw = float(diff[pw].mean() / diff[pw].std(ddof=1) * np.sqrt(252))

        print(f"\n{RULE}\n{t['name']}  [{t['role']}]  factors {', '.join(t['columns'])}   "
              f"{idx.min():%Y-%m-%d} -> {idx.max():%Y-%m-%d}, zero cost, excess of cash\n")
        print(f"   {'arm':<26}{'Sharpe':>8}{'return':>9}{'vol':>8}{'max DD':>9}"
              f"{'turnover':>10}{'gross':>7}")
        for arm, r in rows.items():
            print(f"   {arm:<26}{r['sharpe']:>+8.2f}{r['ann_return']:>+9.1%}{r['ann_vol']:>8.1%}"
                  f"{r['max_dd']:>9.1%}{r.get('turnover', np.nan):>10.1f}"
                  f"{r.get('gross', np.nan):>7.2f}")
        print(f"\n   delta {delta:+.3f}   MDE {thr:.3f}   t (scaled legs, HAC 6) {tt:+.2f}   "
              f"placebo pct {pct:.1%} (median {np.median(null):+.3f}, p95 "
              f"{np.quantile(null, 0.95):+.3f})")
        print("   one-line rules, delta: " + ", ".join(f"{c} {d:+.3f}" for c, d in cdelta.items()))
        print("   folds (5, descriptive): " + " ".join(f"{d:+.2f}" for d in fold_d)
              + f"   positive {np.mean(np.array(fold_d) > 0):.0%}")
        print(f"   descriptive: 2007-2024 delta {paper_d:+.3f}; before 1990 {pre_d:+.3f}, "
              f"after {post_d:+.3f}; IR of (regime - alone) {ir:+.2f}, 2007-2024 {ir_pw:+.2f}")
        print(f"   => {v}")
        entry = {"sharpe": sharpe(leg), "sharpe_alone": base_sr, "delta": delta,
                 "threshold": thr, "t_hac": tt, "placebo_pct": pct,
                 **{f"delta_{c}": d for c, d in cdelta.items()},
                 "delta_2007_2024": paper_d, "delta_pre1990": pre_d, "delta_post1990": post_d,
                 "folds_positive": float(np.mean(np.array(fold_d) > 0)),
                 "ir_vs_alone": ir, "ir_vs_alone_2007_2024": ir_pw,
                 "ann_vol": rows[f"regime ({t['arm']})"]["ann_vol"],
                 "max_dd": rows[f"regime ({t['arm']})"]["max_dd"],
                 "max_dd_alone": rows["alone (always held)"]["max_dd"],
                 "turnover": rows[f"regime ({t['arm']})"]["turnover"],
                 "gross": rows[f"regime ({t['arm']})"]["gross"],
                 "sessions": int(len(idx)), "verdict": v}
        record["tests"][t["name"]] = entry
        trials.log(
            FAMILY,
            {"test": t["name"], "role": t["role"], "arm": t["arm"], "factors": t["columns"],
             "controls": list(t["controls"]), "grid": list(GRID), "max_feats": MAX_FEATS,
             "lag": fj.LAG, "cost": "zero", "sizing": "min(0.10/sigma63, 3) per sleeve",
             "alpha": f"0.05/{N_TESTS}",
             "sample": [str(idx.min().date()), str(idx.max().date())]},
            entry, **kwargs,
        )

    record["replication"] = replicate(p, pw, kwargs)
    n = trials.summary(**kwargs).get("n_distinct")
    print(f"\n   logged ({n} distinct configurations in the register)")
    return record


def replicate(p: dict, pw: np.ndarray, kwargs: dict) -> dict:
    idx = p["index"]
    f = p["factors"].loc[idx]
    held = p["held"]["sjm_raw"]
    ls = held * f
    win = idx[pw]
    years = len(win) / 252
    print(f"\n{RULE}\nREPLICATION — the paper's single-factor long-short strategy, raw factor, "
          f"no volatility target,\nzero cost, {win.min():%Y-%m-%d} -> {win.max():%Y-%m-%d} "
          f"({years:.1f} years)\n")
    print(f"   {'factor':<7}{'ours':>7}{'t':>7}{'paper':>8}{'shifts ours':>13}{'paper':>7}"
          f"{'long always':>13}{'full OOS':>10}   verdict")
    out = {}
    signs = []
    for k in FACTORS:
        x = ls.loc[win, k]
        sr = sharpe(x)
        se = lo_se(sr, years)
        tt = sr / se if np.isfinite(sr) else float("nan")
        shifts = fj.transitions_per_year(p["state"].loc[win, k])
        long_sr = sharpe(f.loc[win, k])
        full = sharpe(ls[k])
        if k in PAPER:
            name, ref, ref_shift = PAPER[k]
            sign = "SIGN REPRODUCED" if sr > 0 else "SIGN NOT REPRODUCED"
            compat = ("compatible" if abs(sr - ref) <= 1.96 * se
                      else "NOT compatible")
            v = f"{sign}, {compat} with {ref:.2f} ({name})"
            signs.append(sr > 0)
            ref_txt, ref_s = f"{ref:>8.2f}", f"{ref_shift:>7.2f}"
        else:
            v = "no counterpart in the paper (descriptive)"
            ref_txt, ref_s = f"{'—':>8}", f"{'—':>7}"
        print(f"   {k:<7}{sr:>+7.2f}{tt:>+7.2f}{ref_txt}{shifts:>13.2f}{ref_s}{long_sr:>+13.2f}"
              f"{full:>+10.2f}   {v}")
        entry = {"sharpe": sr, "t_lo": tt, "shifts_per_year": shifts,
                 "sharpe_long_always": long_sr, "sharpe_full_oos": full, "verdict": v}
        if k in PAPER:
            entry["sharpe_paper"] = PAPER[k][1]
        out[k] = entry
        trials.log(
            FAMILY_REP,
            {"test": f"R_{k}_long_short_raw", "factor": k, "window": [str(win.min().date()),
             str(win.max().date())], "mapping": "clip(mu_state/5%, -1, 1)", "lag": fj.LAG,
             "cost": "zero", "grid": list(GRID), "max_feats": MAX_FEATS},
            entry, **kwargs,
        )
    n_pos = int(sum(signs))
    overall = ("REPRODUCED in sign (4 of 4)" if n_pos == 4 else
               f"PARTIALLY REPRODUCED in sign ({n_pos} of 4)" if n_pos > 0 else
               "NOT REPRODUCED (0 of 4 positive)")
    corr = ls.loc[win].corr().to_numpy()
    off = corr[~np.eye(6, dtype=bool)]
    print(f"\n   overall, the four factors with a counterpart: {overall}")
    print(f"   pairwise correlation of the six strategies, 2007-2024: mean {off.mean():+.2f}, "
          f"min {off.min():+.2f}, max {off.max():+.2f} (paper: 0.05 to 0.48)")
    out["overall"] = overall
    out["corr_mean"] = float(off.mean())
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    global GRID, FIXED_PENALTY
    synthetic = "--synthetic" in sys.argv
    if synthetic:
        GRID, FIXED_PENALTY = (50.0, 200.0), 50.0
        if "FACTORSJM_SCRATCH" not in os.environ:
            raise SystemExit("--synthetic needs FACTORSJM_SCRATCH, a scratch directory")
    scratch = Path(os.environ["FACTORSJM_SCRATCH"]) if synthetic else None
    paths_file = scratch / "fjm_paths.parquet" if synthetic else PATHS
    states_file = scratch / "fjm_states.parquet" if synthetic else STATES
    tuning_file = scratch / "fjm_tuning.parquet" if synthetic else TUNING
    first_refit = pd.Timestamp("1998-01-01") if synthetic else FIRST_REFIT
    first_tune = pd.Timestamp("2004-01-01") if synthetic else FIRST_TUNE
    data = load_synthetic() if synthetic else load_real()

    if "--fit" in sys.argv or "--select" in sys.argv:
        if "--fit" in sys.argv:
            allpaths = fit_paths(data, GRID, first_refit)
            allpaths.to_parquet(paths_file, index=False)
            print(f"wrote {len(allpaths):,} candidate rows")
        allpaths = pd.read_parquet(paths_file)
        stitched, scores = select(data, allpaths, GRID, first_refit, first_tune)
        stitched.to_parquet(states_file, index=False)
        scores.to_parquet(tuning_file, index=False)
        print(f"wrote {len(stitched):,} selected rows, {len(scores):,} tuning rows")
        return

    paths = pd.read_parquet(paths_file)
    states = pd.read_parquet(states_file)
    tuning = pd.read_parquet(tuning_file)
    reading = "--read" in sys.argv
    if reading and not synthetic:
        done = trials.read()
        if not done.empty and done["family"].isin([FAMILY, FAMILY_REP]).any():
            print("THIS STUDY HAS ALREADY BEEN READ (rows in data/trials.parquet). Refused.")
            return
    p = build(data, paths, states, first_refit)
    p["states_long"] = states
    if not synthetic:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out_file = (scratch / ("fjm_reading.txt" if reading else "fjm_instrument.txt") if synthetic
                else ARTIFACTS / ("reading.txt" if reading else "instrument.txt"))
    sys.stdout = Tee(out_file)
    inst = instrument(p, paths, tuning, real=not synthetic)
    if not inst:
        return
    if not reading:
        print(f"\n{RULE}\nNOT READ — re-run with --read, once.\n{RULE}")
        return
    log_path = scratch / "fjm_trials.parquet" if synthetic else None
    record = read(p, inst, log_path)
    record["instrument"] = {"thresholds": inst["thresholds"], "axis1": inst["axis1"],
                            "participation": inst["participation"]}
    target = scratch / "fjm_reading.json" if synthetic else ARTIFACTS / "reading.json"
    target.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(RULE)


if __name__ == "__main__":
    main()
