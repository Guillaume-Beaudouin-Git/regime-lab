"""The shared core of the Two Sigma instruments and readings — `docs/PRESPEC_TWOSIGMA.md`.

The specification is `docs/PRESPEC_TWOSIGMA.md`, **LOCKED at commit 30f7d69**. Its §12
fixes every computation and its §13 the decision procedure; where the two differ, §12.6
to §12.10 decide a lock's own statistic and §13 the order, the precedence, Holm and the
logging. Nothing in this module settles a choice the lock left open without saying so in
the docstring of the function that makes it. A deviation from the lock is an amendment
in `docs/PROTOCOL_FREEZE.md`, never an edit here.

**What this module is.** The code the three level instruments (A, B, C) and their
readings share, so that the three build on one reading of §12:

- the data of the tree, loaded once (:func:`load_tree_data`, §3, §12.1, §12.3);
- the declared variants (:data:`PRIMARY`, the §12.13 sensitivities, and :data:`PIT18`,
  the §8 control-5 rebuild);
- the partition and the two volatility witnesses as stamped per-fold paths in the format
  of ``context.session_paths`` (§12.2, §12.8), the lag (§12.2) and the cells (§12.4);
- the uniform matched placebo, with any draw turned back into a path of the same format
  (§12.11), so that every downstream function takes ``paths`` and runs identically on
  real and on placebo labels;
- the traded mix and the books (§12.3), the paired thresholds, readings and cost kill
  (§12.6, §12.9), the null percentiles (§12.7, §12.8, §12.10);
- the verdict lines, their precedence and Holm (§12.6-§12.10, §13.2, §13.3);
- the trial row schema (§13.5), the threshold file and the refusal to read (§13.1);
- a deterministic process-pool map for the placebo loops.

**What it is not.** It estimates nothing from returns: no ``m_ik``, no second-moment
matrix, no cadence rule. Those belong to the level modules, which call this one.

**Blindness (§11, §13.1 step 2).** This module prints nothing. Functions that return
return-derived quantities (a book's net returns, a Sharpe, a beta, a paired reading) are
used by the readings and by the instruments' allowed prints only; :class:`BookResult`'s
``repr`` shows the allowed quantities alone (realised sd, turnover, cap share, missing
sessions), so an accidental print of a book reads no mean. The instrument mode may print
counts, the turnover of the arms it builds, the kill line, realised sd and cap share,
demeaned-leg MDEs and T_A1/T_B2/MDE_C/S*, null percentiles of A-2/B-1/C-1 and the NFCI
diagnostic, and nothing else. The reading mode must first pass
:func:`verify_for_reading`.

**Non-finite readings (§13.4).** A statistic whose input is missing or non-finite comes
out NaN (never a number computed on the sessions that happen to be present), a verdict
function returns UNDECIDABLE on any NaN, and a non-finite threshold is refused by the
threshold file. No verdict is ever read off a NaN. Structural errors (a session outside
every window, a table missing a signal) raise instead.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import multiprocessing
import os
import subprocess
import tomllib
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits

from regime_lab.analysis import trials
from regime_lab.analysis.placebo import PlaceboCheck, check_placebos, matched_placebos
from regime_lab.config import ROOT
from regime_lab.data.pit import validate
from regime_lab.extensions.trend import MAX_LEVERAGE, VOL_TARGET
from regime_lab.selection import protocol
from regime_lab.selection.context import (
    CONTEXT_FEATURES,
    LOG_RV,
    N_INIT,
    N_STATES,
    RANDOM_STATE,
    SEGMENTS,
    SMOOTHING_WINDOW,
    WalkForwardContext,
    as_of,
    context_panel,
    load_context_features,
    load_log_realised_vol,
    log_realised_vol,
    placebo_groups,
    qualifying_cells,
    session_paths,
    trailing_mode,
    walk_forward_context,
)
from regime_lab.selection.folds import (
    N_FOLDS,
    TEST_YEARS,
    Fold,
    fold_of,
    inferential_sessions,
    walk_forward_folds,
)
from regime_lab.selection.library import LIBRARY, complete_sessions, industry_library, load_panel

NAN = float("nan")
PERIODS = protocol.PERIODS

# ---------------------------------------------------------------------------------------
# constants fixed by the lock
# ---------------------------------------------------------------------------------------

#: §5, §12.3: the three cost columns; 5 bp decides (§13.2), 10 bp only through verdict
#: line 4, 20 bp only through the kill.
COST_COLUMNS: tuple[float, ...] = (
    protocol.COST_BPS["realistic"], protocol.COST_BPS["conservative"],
    protocol.COST_BPS["stress"],
)
DECIDING_BPS = protocol.COST_BPS["realistic"]
KILL_BPS = protocol.COST_BPS["stress"]
#: §5, §12.6 step 3: the draft's 12x/yr, kept as the kill's cap.
KILL_CAP = 12.0
#: §6 A-1, §12.6 step 2: the draft's bar, kept as a floor of T_A1 and T_B2.
THRESHOLD_FLOOR = 0.338
#: §12.6 step 1, §12.12: blinded bootstrap draws, seed, and the three mean blocks.
BOOT_DRAWS = 2_000
BOOT_SEED = 0
POWER_BLOCKS: tuple[int, ...] = protocol.POWER_BLOCKS
HAC_LAGS = protocol.HAC_LAGS
#: §12.11: 1,000 uniform draws, seed 0, on (fold, segment) blocks.
N_DRAWS = 1_000
PLACEBO_SEED = 0
PLACEBO_METHOD = "uniform"
#: §12.7, §12.8, §12.10: a placebo lock holds at p <= 0.01; §12.6 lines 6-7 and §8
#: control 3: the 95th percentile of the placebo arms.
PLACEBO_P_BAR = 0.01
PLACEBO_PCT_BAR = 0.95
#: §12.8: each witness must be beaten at one-sided p <= 0.05.
WITNESS_P_BAR = 0.05
#: §12.6 line 1, §12.9: more than half the test sessions at the fallback is UNDECIDABLE.
MAX_FALLBACK_SHARE = 0.5
#: §7, §13.3: Holm over the six primaries at family-wise 0.05, two-sided.
FAMILY_ALPHA = protocol.FAMILY_ALPHA
PRIMARIES: tuple[str, ...] = ("A-1", "A-2", "B-1", "B-2", "C-1", "C-2")
BONFERRONI = FAMILY_ALPHA / len(PRIMARIES)
#: §13.2: the locks that decide each level. C-2 is a bound, never a PASS (§12.10).
LEVEL_LOCKS: dict[str, tuple[str, ...]] = {"A": ("A-1", "A-2"), "B": ("B-1", "B-2"),
                                           "C": ("C-1",)}
#: §12.8: the W2 window, the trailing realised variance of the equal-weight leg.
W2_WINDOW = 63
#: §3, §8 control 5: the two columns built from the NFCI's current, revised vintage.
NFCI_FEATURES: tuple[str, ...] = ("fin_nfci", "fin_nfci_chg13w")
PIT_FEATURES: tuple[str, ...] = tuple(f for f in CONTEXT_FEATURES if f not in NFCI_FEATURES)

#: §13.1 step 3: every data file the instruments read, relative to the repository root.
#: `size_bm_25.parquet` (§3) is not read: it enters only through the cached context
#: feature ``xs_dispersion_szbm`` of ``features.parquet``, which is hashed.
INPUT_FILES: tuple[str, ...] = (
    "data/raw/panels/industry_49.parquet",
    "data/raw/panels/factors_5.parquet",
    "data/cache/features.parquet",
    "data/raw/prices/cross_asset.parquet",
)
#: §0 header, §13.1 step 4: the packages whose versions bit-reproducibility rests on.
PINNED_PACKAGES: tuple[str, ...] = (
    "numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "threadpoolctl",
)
THRESHOLDS_FILE = "docs/artifacts/twosigma/thresholds.json"
THRESHOLDS_PATH = ROOT / THRESHOLDS_FILE
#: How an instrument printout, and so the threshold file, writes a value that is not
#: finite or not determined (§13.4): as this string, never as a number. The three levels
#: share it, so that a lock UNDECIDABLE before its reading is recorded the same way.
NON_FINITE = "non-finite"

#: §13.5: the trial family and the fixed configuration fields.
TRIAL_FAMILY = "twosigma"
PRESPEC_TAG = "docs/PRESPEC_TWOSIGMA.md, LOCKED 2026-09-23"
LOCK_COMMIT = "30f7d69"
BUILD_COMMIT = "3b64ab2"
CONTEXT_TRAIN_START = "1995-01-04"


# ---------------------------------------------------------------------------------------
# the data of the tree — §3, §4, §12.1, §12.3
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class TreeData:
    """Everything the tree is computed from, built once by :func:`build_tree_data`.

    Every frame is on ``sessions`` (§12.1: the 7,946 industry sessions 1995-01-04 to
    2026-07-31 on the real store) unless said otherwise. Treat the frames as read-only.

    ``returns``    decimal industry returns;
    ``factors``    the long point-in-time Ken French factor panel (percent), validated;
    ``rf``         ``ff_rf`` stamped at ``available_at`` and carried forward (§12.3);
    ``excess``     instrument-level excess returns ``r_n - rf`` (§12.3);
    ``market``     ``ff_mkt-rf`` at ``period``, for realised beta (§8 control 3, §12.6);
    ``signals``    the ten LIBRARY held weight frames, lagged one session (§4);
    ``legs``       the signal legs ``x`` of §12.3, sessions x the ten signals;
    ``log_rv``     log 21-session realised vol of ``eq_us_large``, as of each session;
    ``features``   the twenty §4 context features on the feature calendar (raw);
    ``log_rv_raw`` ``log_rv`` on its own price calendar;
    ``context``    ``context.context_panel`` of the twenty features, feature calendar.
    """

    sessions: pd.DatetimeIndex
    folds: tuple[Fold, ...]
    fold: pd.Series
    test_sessions: pd.DatetimeIndex
    returns: pd.DataFrame
    factors: pd.DataFrame
    rf: pd.Series
    excess: pd.DataFrame
    market: pd.Series
    signals: dict[str, pd.DataFrame]
    legs: pd.DataFrame
    log_rv: pd.Series
    features: pd.DataFrame
    log_rv_raw: pd.Series
    context: pd.DataFrame

    def __repr__(self) -> str:
        s = self.sessions
        return (f"TreeData({len(s):,} sessions {s[0]:%Y-%m-%d}..{s[-1]:%Y-%m-%d}, "
                f"{len(self.folds)} folds, {len(self.test_sessions):,} test sessions, "
                f"{self.excess.shape[1]} instruments, {len(self.signals)} signals)")


def signal_legs(
    signals: Mapping[str, pd.DataFrame], excess: pd.DataFrame, *, bps: float = DECIDING_BPS
) -> pd.DataFrame:
    """The signal legs of §12.3: each signal traded alone at unit gross, net of 5 bp.

    ``x_i,t = Σ_n w_i,n,t (r_n,t − rf_t) − 0.0005 Σ_n |w_i,n,t − w_i,n,t−1|``, that is
    ``protocol.net_of_costs`` applied to ``(W_i × excess).sum(axis=1)`` with the signal's
    own weights. Not volatility-targeted. A session where any weight or return is missing
    gives NaN. The legs are built on the frames as given: on :class:`TreeData` that is the
    inferential sessions, so the first session's ``|Δw|`` is charged zero, as
    ``protocol.net_of_costs`` does for every book.
    """
    columns = {}
    for name, weights in signals.items():
        product = weights * excess.reindex(index=weights.index, columns=weights.columns)
        gross = product.sum(axis=1).where(product.notna().all(axis=1))
        columns[name] = protocol.net_of_costs(gross, weights, bps=bps)
    return pd.DataFrame(columns)


def build_tree_data(
    industries: pd.DataFrame,
    factors: pd.DataFrame,
    library_market: pd.Series,
    features: pd.DataFrame,
    log_rv: pd.Series,
    *,
    sessions: pd.DatetimeIndex | None = None,
    folds: tuple[Fold, ...] | None = None,
    n_folds: int = N_FOLDS,
    test_years: float = TEST_YEARS,
) -> TreeData:
    """Assemble :class:`TreeData` from raw inputs; pure, so tests can feed it synthetic data.

    ``industries``      wide decimal industry returns over their whole history (the library
                        needs its warm-up), as ``library.load_panel("industry_49")``;
    ``factors``         the long PIT factor panel with ``ff_rf`` and ``ff_mkt-rf`` (percent);
    ``library_market``  ``ff_mkt-rf`` in decimal at ``period``, the regressor of
                        ``IND_LOWBETA`` (§4 "The loaders");
    ``features``        a frame holding the twenty ``CONTEXT_FEATURES``;
    ``log_rv``          ``context.log_realised_vol`` of ``eq_us_large`` on its own calendar;
    ``sessions``        the inferential sessions (default: every session on which the ten
                        signals are complete);
    ``folds``           default ``walk_forward_folds(sessions, n_folds=5, test_years=4.0,
                        anchor="end")`` (§12.1); ``n_folds`` and ``test_years`` exist for
                        synthetic samples shorter than the real one.

    Refuses (``ValueError``) a session on which a signal, the cash rate or an excess
    return is missing: §3 verifies the panel is fully populated, and a gap must surface.
    """
    factors = validate(factors)
    weights = industry_library(industries, library_market, LIBRARY)
    complete = complete_sessions(weights)
    sessions = complete if sessions is None else pd.DatetimeIndex(sessions)
    if not sessions.is_monotonic_increasing or sessions.has_duplicates:
        raise ValueError("sessions must be sorted and unique")
    if not sessions.isin(complete).all():
        raise ValueError("the ten-signal library is not complete on every session")
    signals = {name: frame.loc[sessions] for name, frame in weights.items()}
    rf = protocol.risk_free(factors, sessions)
    if rf.isna().any():
        raise ValueError("ff_rf is missing on a session")
    returns = industries.loc[sessions]
    excess = protocol.excess_returns(returns, rf)
    if not np.isfinite(excess.to_numpy(float)).all():
        raise ValueError("an excess return is missing on a session")
    market = protocol.market_excess(factors, sessions)
    legs = signal_legs(signals, excess)
    if folds is None:
        folds = walk_forward_folds(sessions, n_folds=n_folds, test_years=test_years,
                                   anchor="end")
    folds = tuple(folds)
    fold = fold_of(sessions, folds)
    missing = [c for c in CONTEXT_FEATURES if c not in features.columns]
    if missing:
        raise KeyError(f"context features missing: {missing}")
    raw_features = features[list(CONTEXT_FEATURES)].sort_index()
    return TreeData(
        sessions=sessions,
        folds=folds,
        fold=fold,
        test_sessions=sessions[fold.to_numpy() > 0],
        returns=returns,
        factors=factors,
        rf=rf,
        excess=excess,
        market=market,
        signals=signals,
        legs=legs,
        log_rv=as_of(log_rv, sessions).rename(LOG_RV),
        features=raw_features,
        log_rv_raw=log_rv.sort_index().rename(LOG_RV),
        context=context_panel(raw_features, log_rv),
    )


def _assert_period_stamps(root: Path) -> None:
    """§14: the loaders pivot on ``period``; refuse if the store ever encodes a delay.

    ``library.load_panel``, ``context.load_log_realised_vol`` and
    ``folds.inferential_sessions`` read ``period`` and ignore ``available_at``. That is
    right only while the two coincide on every row they read, which §14 asks to assert
    before the instruments.
    """
    checks = (
        ("data/raw/panels/industry_49.parquet", None),
        ("data/raw/panels/factors_5.parquet", "ff_mkt-rf"),
        ("data/raw/prices/cross_asset.parquet", "eq_us_large"),
    )
    for relative, series in checks:
        frame = pd.read_parquet(root / relative, columns=["series_id", "period", "available_at"])
        if series is not None:
            frame = frame.loc[frame["series_id"] == series]
        same = pd.to_datetime(frame["available_at"]) == pd.to_datetime(frame["period"])
        if frame.empty or not bool(same.all()):
            raise ValueError(f"{relative}: available_at differs from period "
                             f"({'series ' + series if series else 'all series'})")


def load_tree_data(root: Path = ROOT) -> TreeData:
    """The tree's data from the store, through the committed loaders (§3, §4, §12.1).

    Reads exactly :data:`INPUT_FILES` under ``root``: industry returns by
    ``library.load_panel("industry_49")``, the factor panel validated for ``ff_rf`` and
    ``ff_mkt-rf``, the library's market regressor ``load_panel("factors_5")["ff_mkt-rf"]``
    at ``period``, the twenty context features, the log realised volatility of
    ``eq_us_large``, and ``folds.inferential_sessions()``. Asserts first that
    ``available_at == period`` wherever a loader pivots on ``period`` (§14).
    """
    root = Path(root)
    _assert_period_stamps(root)
    raw = root / "data" / "raw"
    industries = load_panel("industry_49", root=raw)
    factors = pd.read_parquet(raw / "panels" / "factors_5.parquet")
    library_market = load_panel("factors_5", root=raw)["ff_mkt-rf"]
    features = load_context_features(root / "data" / "cache" / "features.parquet")
    log_rv = load_log_realised_vol(raw / "prices" / "cross_asset.parquet")
    sessions = inferential_sessions(raw / "panels" / "industry_49.parquet")
    return build_tree_data(industries, factors, library_market, features, log_rv,
                           sessions=sessions)


# ---------------------------------------------------------------------------------------
# variants — §7, §8 control 5, §12.13
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """One declared configuration of the tree.

    ``name`` is the §13.5 ``variant`` field (``primary``, ``K=3``, ``K=5``, ``K=6``,
    ``d=0.25``, ``d=1.00``, ``smooth21``), or ``pit18`` for the §8 control-5 rebuild,
    which spends no trial row. ``smoothing`` is the trailing-mode window of §12.13
    (``None``: no smoothing). ``levels`` lists what the variant is evaluated at: the
    three levels, or ``("A-1",)`` for the d rows (§7).
    """

    name: str
    K: int = N_STATES
    d: float = protocol.TILT_D
    smoothing: int | None = None
    features: tuple[str, ...] = CONTEXT_FEATURES
    levels: tuple[str, ...] = ("A", "B", "C")
    role: Literal["primary", "sensitivity", "control"] = "sensitivity"

    @property
    def partition_key(self) -> tuple[int, int | None, tuple[str, ...]]:
        """Variants with equal keys share one partition (the d rows share the primary's)."""
        return (self.K, self.smoothing, self.features)

    @property
    def cell_floor(self) -> float:
        """§13.4: UNDECIDABLE below half of the 5K cells (10 at K = 4)."""
        return 5 * self.K / 2


PRIMARY = Variant("primary", role="primary")
K3 = Variant("K=3", K=3)
K5 = Variant("K=5", K=5)
K6 = Variant("K=6", K=6)
D025 = Variant("d=0.25", d=protocol.TILT_SENSITIVITIES[0], levels=("A-1",))
D100 = Variant("d=1.00", d=protocol.TILT_SENSITIVITIES[1], levels=("A-1",))
SMOOTH21 = Variant("smooth21", smoothing=SMOOTHING_WINDOW)
PIT18 = Variant("pit18", features=PIT_FEATURES, role="control")
SENSITIVITY_VARIANTS: tuple[Variant, ...] = (K3, K5, K6, D025, D100, SMOOTH21)
VARIANTS: dict[str, Variant] = {v.name: v for v in (PRIMARY, *SENSITIVITY_VARIANTS, PIT18)}

#: §7: the 20 declared evaluations as (test, variant) rows, in the §13.5 schema.
DECLARED_ROWS: tuple[tuple[str, str], ...] = (
    *((test, PRIMARY.name) for test in PRIMARIES),
    *((level, v.name) for v in (K3, K5, K6) for level in ("A", "B", "C")),
    *(("A-1", v.name) for v in (D025, D100)),
    *((level, SMOOTH21.name) for level in ("A", "B", "C")),
)
if len(DECLARED_ROWS) != protocol.DECLARED_EVALUATIONS:  # pragma: no cover - import guard
    raise ImportError("the declared rows do not number the §7 count")


def section_name(level: str, variant: Variant) -> str:
    """The threshold file's section of one level's instrument at one variant (§13.1 step 3).

    ``A``, ``B`` or ``C`` for the primary; ``<level>:<variant>`` for a §12.13 sensitivity
    (``A:K=3``, ``B:smooth21``, ``A:d=0.25``). The PIT rebuild has none: its thresholds are
    computed at the reading and never committed (§8 control 5).
    """
    if level not in LEVEL_LOCKS:
        raise ValueError(f"not a level of the tree: {level!r}")
    if variant.role == "control":
        raise ValueError("the PIT rebuild's thresholds are not committed (§8 control 5)")
    return level if variant.role == "primary" else f"{level}:{variant.name}"


def pit_variant(variant: Variant) -> Variant:
    """The §8 control-5 rebuild of ``variant``: the 18 features without ``fin_nfci`` and
    ``fin_nfci_chg13w``, the same K, smoothing and d (``n_init``, seed, folds and training
    start are fixed in :func:`partition_paths`). :data:`PIT18` for the primary. A
    sensitivity's would-be PASS is rebuilt at its own K, smoothing and d, since §12.13
    reruns "each level's full evaluation" there; the rebuild only downgrades."""
    if variant.role == "control":
        raise ValueError(f"{variant.name} is already the PIT rebuild (§8 control 5)")
    if (variant.partition_key, variant.d) == (PRIMARY.partition_key, PRIMARY.d):
        return PIT18
    return dataclasses.replace(variant, name=f"{PIT18.name}[{variant.name}]",
                               features=PIT_FEATURES, role="control")


# ---------------------------------------------------------------------------------------
# paths — §12.2, §12.4, §12.8, §12.13
# ---------------------------------------------------------------------------------------


class Partition(NamedTuple):
    """A variant's stamped per-fold paths (``session_paths`` format) and its refit."""

    paths: pd.Series
    wf: WalkForwardContext


def variant_panel(data: TreeData, variant: Variant) -> pd.DataFrame:
    """The context panel a variant is fitted on: its features plus the as-of log rv.

    The twenty-feature panel is ``context.context_panel`` itself. Another feature set
    (the 18 of :data:`PIT18`) is built the same way — the features, the as-of log rv,
    complete rows only — on its own columns. On the real store the two panels hold the
    same rows over the sample (measured 2026-09-23, shapes only).
    """
    if tuple(variant.features) == CONTEXT_FEATURES:
        return data.context
    unknown = [f for f in variant.features if f not in data.features.columns]
    if unknown:
        raise KeyError(f"not a context feature: {unknown}")
    panel = data.features[list(variant.features)].copy()
    panel[LOG_RV] = as_of(data.log_rv_raw, panel.index)
    return panel.dropna()


def partition_paths(data: TreeData, variant: Variant = PRIMARY) -> Partition:
    """A variant's walk-forward partition as stamped per-fold paths (§12.2, §12.13).

    ``walk_forward_context(panel, folds, orthogonalise=True)`` at the variant's K,
    ``n_init`` 20, ``random_state`` 0, each fold trained from its own first training
    session (the 1995-01-04 start of §12.2), then ``session_paths`` on the industry
    sessions: index ``(fold, segment, session)``, fold *f*'s own numbering, **stamped,
    not lagged**. With ``variant.smoothing`` the paths go through
    ``context.trailing_mode`` before any lag, so the first ``window - 1`` sessions of
    each fold's path read NaN (§12.13). The d variants share the primary's partition.
    """
    wf = walk_forward_context(
        variant_panel(data, variant), data.folds, orthogonalise=True, n_states=variant.K,
        n_init=N_INIT, random_state=RANDOM_STATE,
    )
    paths = session_paths(wf, data.folds, data.sessions)
    if variant.smoothing is not None:
        paths = trailing_mode(paths, variant.smoothing)
    return Partition(paths, wf)


def quantile_bins(values: pd.Series, reference: pd.Series, n_bins: int) -> pd.Series:
    """Bin ``values`` at the ``1/K .. (K-1)/K`` quantiles of ``reference`` (§12.8).

    Cut-offs are ``numpy.quantile`` with its default linear interpolation, over the
    finite values of ``reference``. Bins are right-closed, as ``pandas.qcut``'s: a value
    equal to a cut-off falls in the lower bin. Labels run 0 (lowest) to K-1; a
    non-finite value reads NaN.
    """
    if n_bins < 1:
        raise ValueError("at least one bin")
    ref = reference.to_numpy(float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        raise ValueError("no finite reference value to cut the bins on")
    edges = np.quantile(ref, np.arange(1, n_bins) / n_bins)
    v = values.to_numpy(float)
    labels = np.searchsorted(edges, v, side="left").astype(float)
    labels[~np.isfinite(v)] = np.nan
    return pd.Series(labels, index=values.index, name="state")


def witness_values(data: TreeData, which: Literal["W1", "W2"]) -> pd.Series:
    """The variable a witness bins, on the sessions, known at the close of each (§12.8).

    W1: ``log_rv``, the 21-session log realised volatility of ``eq_us_large`` as of each
    session. W2: ``log v_t``, where ``v_t`` is the mean of ``x̄²`` over the 63 sessions
    ending at *t* and ``x̄`` the mean of the ten signal legs; the first 62 sessions have
    no ``v_t``.
    """
    if which == "W1":
        return data.log_rv.rename("W1")
    if which == "W2":
        blend_leg = data.legs.mean(axis=1).where(data.legs.notna().all(axis=1))
        v = (blend_leg**2).rolling(W2_WINDOW, min_periods=W2_WINDOW).mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(v).rename("W2")
    raise ValueError(f"witness must be 'W1' or 'W2', not {which!r}")


def witness_paths(data: TreeData, which: Literal["W1", "W2"], K: int) -> pd.Series:
    """A volatility witness as stamped per-fold paths, in ``session_paths`` format (§12.8).

    For fold *f*, K quantile bins of :func:`witness_values` with cut-offs over fold *f*'s
    training sessions (:func:`quantile_bins`), labelled on its training then test
    sessions, **stamped at t, not lagged** — lag it exactly as the partition. The paths
    are qualified, estimated on and traded exactly like the partition's (§12.6, §12.7,
    §12.8, §12.10). NaN where the witness has no value (W2's first 62 sessions).
    """
    values = witness_values(data, which)
    pieces = []
    for fold in data.folds:
        train = fold.train(data.sessions)
        for segment, dates in zip(SEGMENTS, (train, fold.test(data.sessions)), strict=True):
            labels = quantile_bins(values.loc[dates], values.loc[train], K)
            index = pd.MultiIndex.from_arrays(
                [np.full(len(dates), fold.number), np.full(len(dates), segment), dates],
                names=["fold", "segment", "session"],
            )
            pieces.append(pd.Series(labels.to_numpy(), index=index))
    out = pd.concat(pieces).rename("state")
    return out if out.isna().any() else out.astype(np.int64)


def _check_paths(paths: pd.Series) -> None:
    if list(paths.index.names) != ["fold", "segment", "session"]:
        raise ValueError("paths must be indexed by (fold, segment, session)")
    fold = paths.index.get_level_values("fold").to_numpy()
    session = paths.index.get_level_values("session").to_numpy("datetime64[ns]").view(np.int64)
    same = fold[1:] == fold[:-1]
    if (np.diff(session)[same] <= 0).any():
        raise ValueError("each fold's path must run forward in time, training then test")
    if int((~same).sum()) + 1 != len(np.unique(fold)):
        raise ValueError("each fold's rows must be contiguous")


def lagged_paths(paths: pd.Series) -> pd.Series:
    """Every fold's path lagged one session inside the fold (§12.2), same index.

    Session *t* holds the label its fold's model stamped at the close of *t−1*; the first
    training session of each fold is unlabelled, and the first test session holds the
    label stamped on the last training session. The lag never crosses folds. Float,
    NaN where no label is held.
    """
    _check_paths(paths)
    return paths.astype(float).groupby(level="fold", sort=False).shift(1).rename("state")


def lagged_path(paths: pd.Series, fold: int) -> pd.Series:
    """One fold's training-then-test path lagged one session, indexed by session (§12.2)."""
    _check_paths(paths)
    return paths.xs(fold, level="fold").droplevel("segment").astype(float).shift(1)


@dataclass(frozen=True, eq=False)
class Cells:
    """The cells of §12.4 on one stamped path.

    ``table`` is ``context.qualifying_cells(paths)``. ``train[f]`` and ``test[f]`` are the
    states qualifying in fold *f*'s training and test blocks; ``pairs`` the (fold, state)
    pairs qualifying on both, which enter A-2. ``abstaining`` counts the test sessions
    whose lagged state has no qualifying training cell (missing states included) — the
    sessions that hold the abstaining row.
    """

    table: pd.DataFrame
    train: dict[int, frozenset[int]]
    test: dict[int, frozenset[int]]
    pairs: tuple[tuple[int, int], ...]
    abstaining: int
    test_sessions: int

    @property
    def n_train(self) -> int:
        return sum(len(s) for s in self.train.values())

    @property
    def n_test(self) -> int:
        return sum(len(s) for s in self.test.values())

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def per_fold(self) -> str:
        """A-2 pairs per fold, as §12.4 prints them (``2/3/4/3/2`` at K = 4)."""
        return "/".join(str(sum(1 for f, _ in self.pairs if f == fold)) for fold in self.train)


def cells(paths: pd.Series) -> Cells:
    """The §12.4 cells of a stamped path: ``context.qualifying_cells`` (63 sessions, 3
    episodes per (fold, segment) block, counted on the stamped path) and what follows."""
    _check_paths(paths)
    table = qualifying_cells(paths)
    q = table["qualifies"]
    folds = list(dict.fromkeys(paths.index.get_level_values("fold").tolist()))

    def qualifying(fold: int, segment: str) -> frozenset[int]:
        block = q.xs((fold, segment), level=("fold", "segment"))
        return frozenset(int(s) for s in block.index[block.to_numpy()])

    train = {int(f): qualifying(f, "train") for f in folds}
    test = {int(f): qualifying(f, "test") for f in folds}
    pairs = tuple((f, k) for f in train for k in sorted(train[f] & test[f]))
    lagged = lagged_paths(paths)
    abstaining, n_test = 0, 0
    for f in folds:
        held = lagged.xs((f, "test"), level=("fold", "segment")).to_numpy(float)
        allowed = np.array(sorted(train[int(f)]), dtype=float)
        abstaining += int((~np.isin(held, allowed)).sum())
        n_test += len(held)
    return Cells(table, train, test, pairs, abstaining, n_test)


# ---------------------------------------------------------------------------------------
# the placebo — §8 construction 4, §12.11
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PlaceboDraws:
    """The uniform matched placebo on a stamped path, stored compactly.

    ``codes`` is rows x draws, int8, the drawn state of every row of ``index`` (the real
    paths' index) and -1 where the real path is unlabelled (the smoothed paths' first
    20 sessions per fold, left out of every block, §12.11, §12.13). ``check`` is
    ``analysis.placebo.check_placebos`` on the labelled rows; ``exact`` must hold or every
    lock using the placebo is UNDECIDABLE (§12.11). ``source`` is
    :func:`paths_fingerprint` of the stamped path the draws were made on: two partitions
    over the same sessions share one index, so the index alone cannot tell which path a
    set of draws belongs to (:meth:`drawn_on`, :func:`check_draws`).
    """

    index: pd.MultiIndex
    codes: np.ndarray
    check: PlaceboCheck
    seed: int
    method: str = PLACEBO_METHOD
    source: str = ""

    @property
    def n(self) -> int:
        return int(self.codes.shape[1])

    @property
    def exact(self) -> bool:
        return self.check.exact

    def drawn_on(self, paths: pd.Series) -> bool:
        """Were these draws made on ``paths`` (same index, same stamped labels)?"""
        return bool(self.index.equals(paths.index) and self.source == paths_fingerprint(paths))

    def draw(self, j: int) -> pd.Series:
        """Draw ``j`` as a paths Series in the real paths' format (same index, NaN where
        the real path is unlabelled), for any function that takes ``paths``."""
        values = self.codes[:, j]
        if (values < 0).any():
            out = np.where(values < 0, np.nan, values.astype(float))
            return pd.Series(out, index=self.index, name="state")
        return pd.Series(values.astype(np.int64), index=self.index, name="state")

    def __repr__(self) -> str:
        return (f"PlaceboDraws({self.n} draws, method={self.method!r}, seed={self.seed}, "
                f"exact={self.exact})")


def paths_fingerprint(paths: pd.Series) -> str:
    """SHA-256 of a stamped path's labels in row order (unlabelled rows as -1); with the
    index, it fixes the path a set of placebo draws was made on."""
    values = paths.to_numpy(float)
    codes = np.where(np.isfinite(values), values, -1.0).astype(np.int64)
    return hashlib.sha256(np.ascontiguousarray(codes).tobytes()).hexdigest()


def check_draws(draws: PlaceboDraws, paths: pd.Series) -> PlaceboDraws:
    """Refuse draws that are not §12.11's on ``paths``: made on another path, not
    ``method="uniform"``, or not at seed 0. Returns ``draws``."""
    if not isinstance(draws, PlaceboDraws):
        raise TypeError("draws must be a PlaceboDraws")
    if not draws.drawn_on(paths):
        raise ValueError("the placebo draws were not drawn on these paths")
    if draws.method != PLACEBO_METHOD or draws.seed != PLACEBO_SEED:
        raise ValueError(f"the placebo must be {PLACEBO_METHOD!r} at seed {PLACEBO_SEED} "
                         "(§12.11)")
    return draws


def placebo_draws(paths: pd.Series, n: int = N_DRAWS, seed: int = PLACEBO_SEED) -> PlaceboDraws:
    """§12.11's placebo: ``matched_placebos(..., groups=placebo_groups, method="uniform")``.

    Blocks are the (fold, segment) blocks of ``paths``; each is redrawn separately,
    keeping its clock, occupancy and episode multiset exactly. ``method="uniform"`` is
    passed explicitly (the module's default is ``swap``, §14). Unlabelled rows — the
    first ``window - 1`` sessions of each fold's smoothed path — are left out of their
    block before drawing and read unlabelled in every draw (§12.11, §12.13); they must
    lead their block, since leaving out a row inside a block would merge the episodes on
    either side of it. Draw *j* depends only on ``seed`` and *j*. Every draw keeps each
    block's per-state sessions and episodes, so it qualifies exactly the cells of the
    real path (§12.4): ``cells(paths).train`` may be passed to :func:`traded_mix` for
    every draw.
    """
    _check_paths(paths)
    labelled = paths.notna()
    for block, mask in labelled.groupby(placebo_groups(paths), sort=False):
        present = mask.to_numpy()
        first = int(np.argmax(present)) if present.any() else len(present)
        if not present[first:].all():
            raise ValueError(f"block {block}: an unlabelled session lies inside the block")
    real = paths[labelled].astype(np.int64)
    if real.max() > np.iinfo(np.int8).max or real.min() < 0:
        raise ValueError("states must be small non-negative integers")
    groups = placebo_groups(real)
    drawn = matched_placebos(real, n, seed=seed, groups=groups, method=PLACEBO_METHOD)
    check = check_placebos(real, drawn.draws, groups=groups)
    codes = np.full((len(paths), n), -1, dtype=np.int8)
    codes[labelled.to_numpy()] = drawn.draws.to_numpy().astype(np.int8)
    return PlaceboDraws(index=paths.index, codes=codes, check=check, seed=seed,
                        source=paths_fingerprint(paths))


# ---------------------------------------------------------------------------------------
# the traded path and the books — §12.3, §12.6 "Inputs", §12.9
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class TradedMix:
    """A traded signal mix on every session, and which test sessions hold the fallback."""

    mix: pd.DataFrame
    fallback: pd.Series

    @property
    def fallback_share(self) -> float:
        """Share of the test sessions at the fallback (§12.6 line 1: UNDECIDABLE > 0.5)."""
        return float(self.fallback.mean()) if len(self.fallback) else NAN


def _as_mix(weights: pd.Series, names: list[str], what: str) -> pd.Series:
    weights = weights.reindex(names).astype(float)
    if weights.isna().any() or not np.isclose(weights.sum(), 1.0, atol=1e-9):
        raise ValueError(f"the {what} must weight every signal and sum to one")
    return weights


def traded_mix(
    sessions: pd.DatetimeIndex,
    folds: Sequence[Fold],
    tables: Mapping[int, pd.DataFrame | None],
    paths: pd.Series,
    fallback: pd.Series | Mapping[int, pd.Series] | None = None,
    *,
    pre_test: pd.Series | None = None,
    names: Sequence[str] = LIBRARY,
    train_cells: Mapping[int, frozenset[int]] | None = None,
) -> TradedMix:
    """The traded path of §12.3: one mix per session over the whole sample.

    - Every session before the first test session holds ``pre_test`` (default the
      control's equal weight: no map exists yet).
    - Fold *f*'s test sessions hold ``protocol.map_states(path_f, tables[f],
      fallback=fallback_f, lag=1)`` on fold *f*'s own training-then-test path (§12.2).
    - ``tables[f]`` is states x signals, rows summing to one, or ``None`` when fold *f*
      holds the control (§12.5 step 7). Rows of states whose **training cell does not
      qualify** on ``paths`` are ignored: such a state abstains (§12.4). A state whose
      estimate failed (B's non-positive-definite cell, §12.8) is simply left out.
    - ``fallback`` is the mix of an abstaining session: equal weight by default (A,
      §12.6), or one Series per fold (B: ``risk_parity_mix(Σ_f)``, §12.9).

    ``TradedMix.fallback`` flags each test session whose lagged state is missing, has no
    qualifying training cell, has no row in the table, or whose fold holds the control
    (§12.6 "fallback session", §12.9 "Fallback at B-2"). ``train_cells`` may pass
    ``cells(paths).train`` to skip recomputing it.
    """
    sessions = pd.DatetimeIndex(sessions)
    names = list(names)
    equal = pd.Series(1.0 / len(names), index=names)
    pre = equal if pre_test is None else _as_mix(pre_test, names, "pre-test mix")
    qualified = cells(paths).train if train_cells is None else train_cells
    mix = pd.DataFrame(np.tile(pre.to_numpy(), (len(sessions), 1)), index=sessions,
                       columns=names)
    covered = pd.Series(0, index=sessions)
    first_test = min(f.test_start for f in folds)
    flags = []
    for fold in folds:
        test = fold.test(sessions)
        covered.loc[test] += 1
        if isinstance(fallback, Mapping):
            fb = _as_mix(fallback[fold.number], names, f"fold {fold.number} fallback")
        else:
            fb = equal if fallback is None else _as_mix(fallback, names, "fallback")
        if fold.number not in tables:
            raise KeyError(f"no table for fold {fold.number} (pass None for the control)")
        table = tables[fold.number]
        if table is None:
            mix.loc[test] = np.tile(fb.to_numpy(), (len(test), 1))
            flags.append(pd.Series(True, index=test))
            continue
        missing = [n for n in names if n not in table.columns]
        if missing:
            raise ValueError(f"fold {fold.number}: the table has no weight for {missing}")
        allowed = qualified.get(fold.number, frozenset())
        keep = [s for s in table.index if np.isfinite(s) and int(s) in allowed]
        table = table.loc[keep, names]
        path = paths.xs(fold.number, level="fold").droplevel("segment")
        if not test.isin(path.index).all():
            raise ValueError(f"fold {fold.number}: its path does not cover its test sessions")
        held = protocol.map_states(path, table, fallback=fb, lag=1).loc[test]
        mix.loc[test] = held.to_numpy()
        lagged = path.astype(float).shift(1).loc[test].to_numpy()
        flags.append(pd.Series(~np.isin(lagged, table.index.to_numpy(float)), index=test))
    before = sessions < first_test
    if not ((covered.to_numpy() == 1) | before).all() or (covered.to_numpy()[before] > 0).any():
        raise ValueError("every session must be pre-test or in exactly one test window")
    return TradedMix(mix=mix, fallback=pd.concat(flags).astype(bool).rename("fallback"))


def per_fold_mix(
    sessions: pd.DatetimeIndex,
    folds: Sequence[Fold],
    mixes: Mapping[int, pd.Series],
    *,
    pre_test: pd.Series | None = None,
    names: Sequence[str] = LIBRARY,
) -> pd.DataFrame:
    """A traded mix that holds one mix per fold on its test sessions (§12.3, §12.9).

    B-2's pooled arm holds ``risk_parity_mix(Σ_f)`` on every test session of fold *f*;
    every session before the first test session holds ``pre_test`` (default equal
    weight), as on every traded path. No state is read, so no session is a fallback.
    """
    names = list(names)
    equal = pd.Series(1.0 / len(names), index=names)
    pre = equal if pre_test is None else _as_mix(pre_test, names, "pre-test mix")
    mix = pd.DataFrame(np.tile(pre.to_numpy(), (len(sessions), 1)), index=sessions,
                       columns=names)
    for fold in folds:
        test = fold.test(pd.DatetimeIndex(sessions))
        held = _as_mix(mixes[fold.number], names, f"fold {fold.number} mix")
        mix.loc[test] = np.tile(held.to_numpy(), (len(test), 1))
    return mix


def realised_sd(returns: pd.Series | np.ndarray) -> float:
    """``√252 × sd (ddof 1)`` of daily returns; NaN if any value is missing (§12.6 σ)."""
    v = np.asarray(returns, dtype=float)
    if v.size < 2 or not np.isfinite(v).all():
        return NAN
    return float(np.sqrt(PERIODS) * v.std(ddof=1))


def sharpe(returns: pd.Series | np.ndarray) -> float:
    """``√252 × mean / sd (ddof 1)`` over the sessions given (§12.3); NaN on any gap.

    For the readings only: the instruments never print a Sharpe (§13.1 step 2).
    """
    v = np.asarray(returns, dtype=float)
    if v.size < 2 or not np.isfinite(v).all():
        return NAN
    sd = v.std(ddof=1)
    return float(np.sqrt(PERIODS) * v.mean() / sd) if sd > 0 else NAN


@dataclass(frozen=True, eq=False, repr=False)
class BookResult:
    """A book on the traded path, read on the test sessions (§12.3, §12.6 "Inputs").

    The full 7,946-session path is volatility-targeted and costed, then every statistic
    is read on the test sessions only. ``sigma`` = √252 × sd (ddof 1) of the 5 bp net
    daily excess returns there; ``turnover`` = ``protocol.annual_turnover`` of the held
    weights **restricted first** to the test sessions; ``cap_share`` = share of test
    sessions on which the cap of 3 binds; ``missing`` = test sessions whose 5 bp net
    return is not finite (a missing paired leg is UNDECIDABLE, §13.4). The ``repr``
    shows these four only, never a mean.
    """

    book: protocol.TargetedBook
    net_full: dict[float, pd.Series]
    test: pd.DatetimeIndex
    market: pd.Series
    sigma: float
    turnover: float
    cap_share: float
    missing: int

    def net(self, bps: float = DECIDING_BPS) -> pd.Series:
        """Daily net excess returns on the test sessions, at ``bps`` (5, 10 or 20)."""
        return self.net_full[float(bps)].loc[self.test]

    @property
    def net5(self) -> pd.Series:
        return self.net(DECIDING_BPS)

    @property
    def held(self) -> pd.DataFrame:
        """Instrument weights actually held (after the multiplier), on the test sessions."""
        return self.book.weights.loc[self.test]

    def beta(self) -> float:
        """``protocol.realised_beta`` of the 5 bp net returns on ``market_excess``, test
        sessions (§8 control 3, §12.6 "Inputs"); NaN if a leg is missing."""
        if self.missing:
            return NAN
        return protocol.realised_beta(self.net5, self.market)

    def __repr__(self) -> str:
        return (f"BookResult({len(self.test):,} test sessions, sigma={self.sigma:.4f}, "
                f"turnover={self.turnover:.2f}x/yr, cap_share={self.cap_share:.3f}, "
                f"missing={self.missing})")


def build_book(
    data: TreeData,
    mix: pd.DataFrame | None = None,
    *,
    weights: pd.DataFrame | None = None,
    targeted: bool = True,
    costs: Sequence[float] = COST_COLUMNS,
) -> BookResult:
    """A book of §12.3: ``blend`` → ``target_volatility`` → ``net_of_costs``.

    ``mix`` is a traded signal mix (sessions x signals); ``None`` with no ``weights`` is
    the equal-weight control. ``weights`` passes instrument weights directly instead (B-2's
    state arm after ``protocol.match_gross``, §12.9). The book is held at 10% with the
    63-session window and the cap of 3 on the multiplier, then netted at each of
    ``costs`` bp on ``|Δw|`` of the weights held. ``targeted=False`` takes ``weights`` as
    already held (C's cadence books of §12.10, which hold the control's held weights):
    return ``w_t · (r_t − rf_t)``, no multiplier, ``cap_share`` NaN.
    """
    if mix is not None and weights is not None:
        raise ValueError("pass a signal mix or instrument weights, not both")
    if weights is None:
        weights = protocol.blend(data.signals, mix)
    weights = weights.reindex(index=data.sessions, columns=data.excess.columns)
    if targeted:
        book = protocol.target_volatility(weights, data.excess, target=VOL_TARGET,
                                          window=protocol.VOL_WINDOW, cap=MAX_LEVERAGE)
    else:
        product = weights * data.excess
        returns = product.sum(axis=1).where(product.notna().all(axis=1)).rename("book")
        empty = pd.Series(np.nan, index=data.sessions)
        book = protocol.TargetedBook(returns=returns, weights=weights, multiplier=empty,
                                     cap_binds=empty)
    net_full = {float(b): protocol.net_of_costs(book.returns, book.weights, bps=float(b))
                for b in costs}
    if DECIDING_BPS not in net_full:
        raise ValueError("the 5 bp column decides and must be computed")
    test = data.test_sessions
    net5 = net_full[DECIDING_BPS].loc[test]
    cap = book.cap_binds.loc[test].astype(float).to_numpy()
    return BookResult(
        book=book,
        net_full=net_full,
        test=test,
        market=data.market.loc[test],
        sigma=realised_sd(net5),
        turnover=protocol.annual_turnover(book.weights.loc[test]),
        cap_share=float(cap.mean()) if np.isfinite(cap).all() else NAN,
        missing=int((~np.isfinite(net5.to_numpy(float))).sum()),
    )


def control_book(data: TreeData, **kwargs: Any) -> BookResult:
    """The control of §4: the equal-weight blend of the ten signals on every session."""
    return build_book(data, None, **kwargs)


# ---------------------------------------------------------------------------------------
# paired thresholds, readings and the cost kill — §12.6, §12.9, §12.12
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PairThreshold:
    """The blinded resolution of a paired Sharpe comparison (§12.6 steps 1-2, §12.9).

    ``se[b]`` is the blinded bootstrap SE at mean block *b*; ``mde[b]`` its MDE at
    α 0.05 and ``mde_corrected[b]`` at α 0.05/6, both two-sided at power 0.80.
    ``threshold`` = max(0.338, max_b MDE(0.05/6, b)); ``se_star`` the SE at
    ``block_star``, the block with the largest MDE(0.05/6) (the first on a tie, in the
    order 21, 63, 126). ``observed`` is the largest ``|observed|`` over the three
    bootstraps: zero to rounding, or the legs were not demeaned. All NaN when a leg
    is missing on a test session (``block_star`` 0).
    """

    se: dict[int, float]
    mde: dict[int, float]
    mde_corrected: dict[int, float]
    threshold: float
    se_star: float
    block_star: int
    observed: float

    def as_dict(self) -> dict[str, Any]:
        """For the threshold file (§13.1 step 3)."""
        return {
            "se": {str(b): v for b, v in self.se.items()},
            "mde_0.05": {str(b): v for b, v in self.mde.items()},
            "mde_0.05/6": {str(b): v for b, v in self.mde_corrected.items()},
            "threshold": self.threshold,
            "se_star": self.se_star,
            "block_star": self.block_star,
        }


def _paired_arrays(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    if not a.index.equals(b.index):
        raise ValueError("the two legs must share one index, in date order")
    return a.to_numpy(float), b.to_numpy(float)


def _mde_job(shared: tuple[np.ndarray, np.ndarray, int, int], j: int) -> tuple[float, float]:
    a, b, draws, seed = shared
    result = protocol.blinded_mde(a, b, mean_block=POWER_BLOCKS[j], draws=draws, seed=seed)
    return result.se, result.observed


def pair_threshold(
    a5: pd.Series,
    b5: pd.Series,
    *,
    draws: int = BOOT_DRAWS,
    seed: int = BOOT_SEED,
    workers: int | None = 1,
) -> PairThreshold:
    """T_A1 or T_B2 on two arms' 5 bp net daily excess returns, test sessions (§12.6).

    ``protocol.blinded_mde(a5, b5, mean_block=b, draws=2000, seed=0)`` for b in 21 / 63 /
    126 — both legs demeaned before the bootstrap, so only a standard error leaves — then
    ``T = max(0.338, max_b SE_b × mde_z(0.05/6))`` and SE* at the block with the largest
    MDE(0.05/6). ``workers`` > 1 runs the three blocks in :func:`draw_map`; the result is
    the same. Returns NaNs, without bootstrapping, when a leg is missing on a session.
    """
    a, b = _paired_arrays(a5, b5)
    if len(a) < 2 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        nan = {blk: NAN for blk in POWER_BLOCKS}
        return PairThreshold(nan, dict(nan), dict(nan), NAN, NAN, 0, NAN)
    results = draw_map(_mde_job, len(POWER_BLOCKS), (a, b, draws, seed),
                       workers=min(workers or 1, len(POWER_BLOCKS)))
    z_family, z_corrected = protocol.mde_z(FAMILY_ALPHA), protocol.mde_z(BONFERRONI)
    se = {blk: float(s) for blk, (s, _) in zip(POWER_BLOCKS, results, strict=True)}
    mde = {blk: s * z_family for blk, s in se.items()}
    corrected = {blk: s * z_corrected for blk, s in se.items()}
    block_star = max(POWER_BLOCKS, key=lambda blk: (corrected[blk], -POWER_BLOCKS.index(blk)))
    return PairThreshold(
        se=se,
        mde=mde,
        mde_corrected=corrected,
        threshold=max(THRESHOLD_FLOOR, max(corrected.values())),
        se_star=se[block_star],
        block_star=block_star,
        observed=max(abs(float(o)) for _, o in results),
    )


@dataclass(frozen=True)
class PairReading:
    """The p-value of a paired Sharpe difference, as §12.6 "Reading" defines it.

    ``p_boot`` = 2(1 − Φ(|Δ|/SE*)); ``p_hac`` = 2(1 − Φ(|t|)) with ``t`` the HAC-6 t of
    the daily net differences, set to 1 when the sign of *t* differs from Δ's; ``p_raw``
    = max(p_boot, p_hac); ``p`` = ``p_raw``, or 1 when Δ ≤ 0 (§13.3). ``p`` is NaN when Δ,
    SE* or *t* is not finite, or a leg is missing.
    """

    delta: float
    se_star: float
    t_hac: float
    p_boot: float
    p_hac: float
    p_raw: float
    p: float


def pair_reading(
    delta: float, se_star: float, a5: pd.Series, b5: pd.Series, *, lags: int = HAC_LAGS
) -> PairReading:
    """p_A1 or p_B2 (§12.6 "Reading", §12.9): the larger of the bootstrap and HAC readings."""
    a, b = _paired_arrays(a5, b5)
    complete = np.isfinite(a).all() and np.isfinite(b).all()
    t = protocol.paired_hac_t(a5, b5, lags=lags) if complete else NAN
    finite = bool(np.isfinite([delta, se_star, t]).all()) and se_star > 0
    if not finite:
        return PairReading(delta, se_star, t, NAN, NAN, NAN, NAN)
    p_boot = float(2.0 * stats.norm.sf(abs(delta) / se_star))
    p_hac = float(2.0 * stats.norm.sf(abs(t)))
    if np.sign(t) != np.sign(delta):
        p_hac = 1.0
    p_raw = max(p_boot, p_hac)
    return PairReading(delta, se_star, t, p_boot, p_hac, p_raw, 1.0 if delta <= 0 else p_raw)


@dataclass(frozen=True)
class Kill:
    """The cost kill of §12.6 step 3 and §12.9: fires if ΔT > K_kill.

    ``fires`` is ``None`` when an input is not finite: a NaN is never a kill verdict.
    """

    delta_turnover: float
    k_kill: float
    fires: bool | None


def kill(
    delta_turnover: float, threshold: float, sigma: float, *, bps: float = KILL_BPS,
    cap: float = KILL_CAP,
) -> Kill:
    """``K_kill = min(12, T × σ × 10,000 / 20)``, T the decision bar and σ the arm's
    realised annualised sd (decimal) at 5 bp on the test sessions; fires if ΔT > K_kill."""
    if not np.isfinite([delta_turnover, threshold, sigma]).all():
        return Kill(delta_turnover, NAN, None)
    k = min(cap, threshold * sigma * 10_000.0 / bps)
    return Kill(delta_turnover, float(k), bool(delta_turnover > k))


@dataclass(frozen=True)
class NullSummary:
    """Percentiles of a placebo null (§12.7, §12.8, §12.10 instruments).

    ``resolution`` = q99 − q50, quoted in a NOT SHOWN (§13.6); ``mde_c`` = q99 − q20, the
    location shift that puts 80% of the null above its 99th percentile (§12.10).
    """

    q20: float
    q50: float
    q95: float
    q99: float

    @property
    def resolution(self) -> float:
        return self.q99 - self.q50

    @property
    def mde_c(self) -> float:
        return self.q99 - self.q20

    def as_dict(self) -> dict[str, float]:
        return {"q20": self.q20, "q50": self.q50, "q95": self.q95, "q99": self.q99,
                "q99-q50": self.resolution, "q99-q20": self.mde_c}


def null_summary(null: Sequence[float] | np.ndarray) -> NullSummary:
    """``numpy.percentile`` (default linear interpolation) at 20/50/95/99; all NaN if any
    draw is not finite, since such a lock is UNDECIDABLE (§13.4)."""
    draws = np.asarray(null, dtype=float)
    if draws.size == 0 or not np.isfinite(draws).all():
        return NullSummary(NAN, NAN, NAN, NAN)
    q = np.percentile(draws, [20, 50, 95, 99])
    return NullSummary(*(float(v) for v in q))


# ---------------------------------------------------------------------------------------
# verdicts — §12.6-§12.10, §13.2, §12.13
# ---------------------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
FAIL_COST = "FAIL (cost)"
UNDECIDABLE = "UNDECIDABLE"
UNDECIDED = "UNDECIDED"
UNDERPOWERED = "UNDERPOWERED"
NOT_SHOWN = "NOT SHOWN"
DOWNGRADED_BETA = "DOWNGRADED (beta)"
DOWNGRADED_NOT_CONDITIONAL = "DOWNGRADED (not conditional)"
DOWNGRADED_PIT = "DOWNGRADED (PIT)"
DOMINATED = "DOMINATED"
DOMINATED_VOLATILITY = "DOMINATED (volatility)"
PASS_NOT_ROBUST = "PASS (not robust)"
#: Every lock verdict the lock writes (§12.6 lines 1-10 use "DOMINATED (volatility)",
#: §12.7, §12.8 and §12.10 "DOMINATED").
LOCK_VERDICTS: frozenset[str] = frozenset({
    PASS, FAIL, FAIL_COST, UNDECIDABLE, UNDECIDED, UNDERPOWERED, NOT_SHOWN,
    DOWNGRADED_BETA, DOWNGRADED_NOT_CONDITIONAL, DOWNGRADED_PIT, DOMINATED,
    DOMINATED_VOLATILITY,
})
#: §13.2: a level that does not pass takes the first of these among its locks.
LEVEL_PRECEDENCE: tuple[str, ...] = (
    FAIL, UNDECIDABLE, UNDECIDED, UNDERPOWERED, NOT_SHOWN, "DOWNGRADED", DOMINATED,
)


def status_class(verdict: str) -> str:
    """The §13.2 class of a lock verdict: FAIL (cost) is a FAIL, every DOWNGRADED (…) a
    DOWNGRADED, DOMINATED (volatility) a DOMINATED."""
    if verdict not in LOCK_VERDICTS:
        raise ValueError(f"not a lock verdict: {verdict!r}")
    for prefix in (FAIL, "DOWNGRADED", DOMINATED):
        if verdict.startswith(prefix):
            return prefix
    return verdict


def level_verdict(verdicts: Sequence[str]) -> str:
    """A level's status from its deciding locks' verdicts (§13.2), a pure function.

    PASS only if every lock passes (A-1 and A-2; B-1 and B-2; C-1 alone, see
    :data:`LEVEL_LOCKS`). Otherwise the first status class present, in the order FAIL
    (FAIL (cost) included), UNDECIDABLE, UNDECIDED, UNDERPOWERED, NOT SHOWN, DOWNGRADED,
    DOMINATED; the lock's own verdict string is returned (e.g. ``FAIL (cost)``), since
    §13.6 words each differently. When locks read UNDERPOWERED and NOT SHOWN, the level
    is UNDERPOWERED (§12.0 row 36).
    """
    if not verdicts:
        raise ValueError("a level needs at least one lock verdict")
    classes = [status_class(v) for v in verdicts]
    if all(c == PASS for c in classes):
        return PASS
    for wanted in LEVEL_PRECEDENCE:
        for verdict, cls in zip(verdicts, classes, strict=True):
            if cls == wanted:
                return verdict
    raise AssertionError("unreachable: every non-PASS class is ranked")  # pragma: no cover


def _finite(*values: float | None) -> bool:
    return all(v is not None and np.isfinite(v) for v in values)


def paired_verdict(
    *,
    delta5: float,
    delta10: float,
    threshold: float,
    p: float,
    kill_fires: bool | None,
    beta_pct: float,
    diff_pct: float,
    delta_w: float,
    fallback_share: float,
    leg_missing: bool,
    holm_rejected: bool | None = None,
    other_finite: bool = True,
) -> str:
    """The first line of §12.6 (A-1) or §12.9 (B-2) that applies, lines 1-8, or PASS.

    1 UNDECIDABLE — any input non-finite (including a percentile that is NaN because a
      placebo draw was), ``kill_fires`` unknown, a leg missing, or the fallback on more
      than half of the test sessions; ``other_finite=False`` passes any other instrument
      or reading value that was not finite;
    2 FAIL (cost) — the kill fired;  3 FAIL — Δ ≤ 0 at 5 bp;
    4 UNDECIDED — Δ > 0 at 5 bp and ≤ 0 at 10 bp;
    5 UNDERPOWERED — 0 < Δ < T, or Δ ≥ T and Holm does not reject;
    6 DOWNGRADED (beta) — beta percentile ≥ 0.95;
    7 DOWNGRADED (not conditional) — difference percentile < 0.95;
    8 DOMINATED (volatility) — Δ_W ≥ Δ.
    Line 9, the PIT rebuild, is :func:`apply_pit`. ``holm_rejected`` defaults to the
    provisional Bonferroni reading ``p ≤ 0.05/6`` (§13.3).
    """
    if (not other_finite or leg_missing or kill_fires is None
            or not _finite(delta5, delta10, threshold, p, beta_pct, diff_pct, delta_w,
                           fallback_share)
            or fallback_share > MAX_FALLBACK_SHARE):
        return UNDECIDABLE
    if kill_fires:
        return FAIL_COST
    if delta5 <= 0:
        return FAIL
    if delta10 <= 0:
        return UNDECIDED
    rejected = p <= BONFERRONI if holm_rejected is None else holm_rejected
    if delta5 < threshold or not rejected:
        return UNDERPOWERED
    if beta_pct >= PLACEBO_PCT_BAR:
        return DOWNGRADED_BETA
    if diff_pct < PLACEBO_PCT_BAR:
        return DOWNGRADED_NOT_CONDITIONAL
    if delta_w >= delta5:
        return DOMINATED_VOLATILITY
    return PASS


def placebo_verdict(
    *,
    statistic: float,
    p: float,
    exact: bool,
    dominated: bool | None,
    holm_rejected: bool | None = None,
    gate_failed: bool | None = False,
    powered: bool = False,
    other_undecidable: bool = False,
) -> str:
    """The verdict of a placebo lock — A-2 (§12.7), B-1 (§12.8), C-1 (§12.10) — before PIT.

    UNDECIDABLE if the statistic or p is not finite (p is NaN when any draw is), the
    placebo check is not exact, ``dominated`` or ``gate_failed`` is unknown, or
    ``other_undecidable`` (cells below the floor, a pooled matrix not positive definite, a
    leg missing). Then FAIL if C-1's gate failed. The lock holds if p ≤ 0.01 and Holm
    rejects (default: the provisional p ≤ 0.05/6); if it does not hold, FAIL when the
    instrument measured power before the reading (C-1 with MDE_C ≤ S*: ``powered``),
    NOT SHOWN otherwise. A lock that holds is DOMINATED when ``dominated`` (A-2:
    R_W ≥ R; B-1: not both witnesses beaten at 0.05; C-1: S_W ≥ S_C), else PASS, to be
    passed through :func:`apply_pit`.
    """
    if (other_undecidable or not exact or dominated is None or gate_failed is None
            or not _finite(statistic, p)):
        return UNDECIDABLE
    if gate_failed:
        return FAIL
    rejected = p <= BONFERRONI if holm_rejected is None else holm_rejected
    if not (p <= PLACEBO_P_BAR and rejected):
        return FAIL if powered else NOT_SHOWN
    if dominated:
        return DOMINATED
    return PASS


def apply_pit(verdict: str, rebuild_verdict: str | None) -> str:
    """§8 control 5: a lock that would PASS is DOWNGRADED (PIT) unless its rebuild on the
    18-feature partition also reads PASS by the same lines (§12.6 line 9, §12.7, §12.8,
    §12.10). Downgrade only; required, and only computed, for a would-be PASS."""
    if verdict != PASS:
        return verdict
    if rebuild_verdict is None:
        raise ValueError("a lock that would PASS needs its PIT rebuild (§8 control 5)")
    return PASS if rebuild_verdict == PASS else DOWNGRADED_PIT


def robust_label(level: str, sensitivity_deltas: Sequence[float]) -> str:
    """§12.13: a level PASS reads PASS (not robust) when any K or smoothing sensitivity
    row at that level has ``delta ≤ 0``. Changes no verdict; a NaN delta is not ≤ 0."""
    if level == PASS and any(np.isfinite(d) and d <= 0 for d in sensitivity_deltas):
        return PASS_NOT_ROBUST
    return level


# ---------------------------------------------------------------------------------------
# Holm — §13.3
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class HolmResult:
    """The Holm step over the six primaries: who is rejected, and the step table."""

    rejected: frozenset[str]
    table: pd.DataFrame
    p: dict[str, float]


def holm(
    p: Mapping[str, float | None],
    *,
    verdicts: Mapping[str, str] | None = None,
    alpha: float = FAMILY_ALPHA,
) -> HolmResult:
    """Holm–Bonferroni over the six primaries, family-wise ``alpha`` (§13.3).

    The family is always the six of :data:`PRIMARIES`. A p that is missing, ``None``,
    not finite, or belongs to a lock whose verdict is UNDECIDABLE enters as 1; C-2 always
    enters as 1. (A-1's and B-2's p are already 1 when Δ ≤ 0, :func:`pair_reading`.)
    Ordering ``p_(1) ≤ … ≤ p_(6)`` (ties in primary order), the primaries ranked 1..j are
    rejected for the largest j with ``p_(i) ≤ alpha / (6 − i + 1)`` for every i ≤ j.
    Rejection is necessary, never sufficient.
    """
    unknown = sorted(set(p) - set(PRIMARIES))
    if unknown:
        raise ValueError(f"not a primary: {unknown}")
    entered: dict[str, float] = {}
    for test in PRIMARIES:
        value = p.get(test)
        undecidable = verdicts is not None and verdicts.get(test) == UNDECIDABLE
        if test == "C-2" or undecidable or value is None or not np.isfinite(value):
            entered[test] = 1.0
        else:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{test}: p = {value} is not a probability")
            entered[test] = float(value)
    order = sorted(PRIMARIES, key=lambda t: (entered[t], PRIMARIES.index(t)))
    m = len(PRIMARIES)
    rows, rejected, going = [], set(), True
    for i, test in enumerate(order, start=1):
        bar = alpha / (m - i + 1)
        going = going and entered[test] <= bar
        if going:
            rejected.add(test)
        rows.append({"rank": i, "test": test, "p": entered[test], "bar": bar,
                     "rejected": going})
    return HolmResult(frozenset(rejected), pd.DataFrame(rows).set_index("rank"), entered)


# ---------------------------------------------------------------------------------------
# trial logging — §13.5
# ---------------------------------------------------------------------------------------


def _row_role(test: str, variant: Variant) -> str:
    if variant.role == "control":
        raise ValueError(f"{variant.name}: the PIT rebuild spends no trial row (§8 control 5)")
    if (test, variant.name) not in DECLARED_ROWS:
        raise ValueError(f"({test!r}, {variant.name!r}) is not one of the 20 declared rows")
    return variant.role


def trial_config(test: str, variant: Variant) -> dict[str, Any]:
    """The §13.5 configuration of one declared row.

    ``d`` is the variant's tilt at A-1 and on the level-A sensitivity rows (whose delta is
    A-1's), null elsewhere: d has no meaning at A-2, B or C (§7).
    """
    role = _row_role(test, variant)
    return {
        "prespec": PRESPEC_TAG,
        "test": test,
        "role": role,
        "variant": variant.name,
        "K": variant.K,
        "d": variant.d if test in ("A-1", "A") else None,
        "library": list(LIBRARY),
        "context": {
            "features": list(CONTEXT_FEATURES),
            "orthogonalised_on": "log 21-session realised volatility of eq_us_large",
            "model": "K-means",
            "n_init": N_INIT,
            "random_state": RANDOM_STATE,
            "refit": "walk-forward, per fold",
            "labels": "out of sample, nearest centroid",
            "lag": 1,
        },
        "context_train_start": CONTEXT_TRAIN_START,
        "anchor": "end",
        "folds": "5 x 4.0 y",
        "cost_bps": DECIDING_BPS,
        "vol_target": VOL_TARGET,
        "cap": MAX_LEVERAGE,
        "placebo": {"method": PLACEBO_METHOD, "draws": N_DRAWS, "seed": PLACEBO_SEED,
                    "blocks": "(fold, segment)"},
        "bootstrap": {"method": "stationary", "blocks": list(POWER_BLOCKS),
                      "draws": BOOT_DRAWS, "seed": BOOT_SEED},
        "estimator": "§12.5, mean contrast",
        "min_cell": {"sessions": 63, "episodes": 3},
        "build": BUILD_COMMIT,
    }


def log_trial(
    test: str,
    variant: Variant,
    *,
    sharpe: float,
    delta: float,
    threshold: float,
    p: float,
    verdict: str,
    sessions: int,
    p2: float = NAN,
    placebo_pct: float = NAN,
    note: str = "",
    path: Path = trials.TRIALS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Append one §13.5 row through ``analysis.trials.log("twosigma", config, metrics)``.

    Primary rows: ``test`` in A-1 … C-2, ``variant`` :data:`PRIMARY`, ``verdict`` the
    provisional Bonferroni-level lock verdict. K and smoothing rows: ``test`` A, B or C,
    ``delta``/``threshold``/``p`` those of A-1, B-2, C-1, ``p2`` that of A-2 or B-1 (NaN
    at C), ``verdict`` the level verdict. d rows: ``test`` A-1, A-1's metrics. ``sharpe``
    is the arm the row evaluates (NaN where none). Refuses: a row outside the 20
    declared, the PIT rebuild, a row already logged for the same (test, variant) at
    ``path``, a verdict the lock does not write, a PASS at C-2, and any verdict other
    than UNDECIDABLE on a non-finite ``delta`` or ``p``.
    """
    config = trial_config(test, variant)
    if verdict not in LOCK_VERDICTS:
        raise ValueError(f"not a verdict of the lock: {verdict!r}")
    if test == "C-2" and verdict == PASS:
        raise ValueError("C-2 is a bound, never a PASS (§12.10)")
    if verdict != UNDECIDABLE and not _finite(delta, p):
        raise ValueError("a verdict other than UNDECIDABLE on a non-finite reading")
    if not (isinstance(sessions, int | np.integer) and sessions > 0):
        raise ValueError("sessions must be a positive count")
    existing = trials.read(path)
    if not existing.empty and "family" in existing:
        for raw in existing.loc[existing["family"] == TRIAL_FAMILY, "config"]:
            row = json.loads(raw)
            if row.get("test") == test and row.get("variant") == variant.name:
                raise ValueError(f"({test}, {variant.name}) is already logged: trials only "
                                 "append, and a second reading is a protocol breach")
    metrics = {
        "sharpe": float(sharpe),
        "delta": float(delta),
        "threshold": float(threshold),
        "p": float(p),
        "p2": float(p2),
        "placebo_pct": float(placebo_pct),
        "sessions": int(sessions),
        "verdict": verdict,
        "note": note,
    }
    trials.log(TRIAL_FAMILY, config, metrics, path=path)
    return config, metrics


# ---------------------------------------------------------------------------------------
# the threshold file and the refusal to read — §13.1 steps 3-4
# ---------------------------------------------------------------------------------------


class ReadingRefused(RuntimeError):
    """The reading may not run: §13.1 step 4's conditions are not all met."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_hashes(root: Path = ROOT, inputs: Sequence[str] = INPUT_FILES) -> dict[str, str]:
    """SHA-256 of every input file the instruments read, keyed by path relative to root."""
    return {relative: _sha256(Path(root) / relative) for relative in inputs}


def locked_versions(root: Path = ROOT, packages: Sequence[str] = PINNED_PACKAGES) -> dict[str, str]:
    """The versions ``uv.lock`` pins for ``packages``; refuses a missing or forked pin."""
    with open(Path(root) / "uv.lock", "rb") as handle:
        lock = tomllib.load(handle)
    found: dict[str, set[str]] = {}
    for package in lock.get("package", []):
        if package.get("name") in packages:
            found.setdefault(package["name"], set()).add(str(package["version"]))
    out = {}
    for name in packages:
        versions = found.get(name, set())
        if len(versions) != 1:
            raise ValueError(f"uv.lock pins {name} {sorted(versions) or 'nowhere'}")
        out[name] = versions.pop()
    return out


def installed_versions(packages: Sequence[str] = PINNED_PACKAGES) -> dict[str, str]:
    """The versions of ``packages`` installed in the running environment."""
    return {name: metadata.version(name) for name in packages}


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                          check=False)


def git_head(root: Path = ROOT) -> str:
    """The commit checked out at ``root``."""
    result = _git(root, "rev-parse", "HEAD")
    if result.returncode != 0:
        raise RuntimeError(f"git rev-parse HEAD failed: {result.stderr.strip()}")
    return result.stdout.strip()


def is_tracked(path: Path, root: Path = ROOT) -> bool:
    """Is ``path`` tracked by git at ``root`` (``git ls-files --error-unmatch``)? The
    instrument refuses to rewrite a threshold file that is (§13.1 step 1)."""
    relative = Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    return _git(Path(root), "ls-files", "--error-unmatch", "--", relative).returncode == 0


def require_committed(path: Path, root: Path = ROOT, *, what: str = "the file") -> str:
    """Refuse (:class:`ReadingRefused`) unless ``path`` exists, is tracked by git
    (``git ls-files --error-unmatch``) and is identical to its HEAD version
    (``git diff --quiet HEAD``, staged or not). Returns its path relative to ``root``."""
    root, path = Path(root), Path(path)
    if not path.exists():
        raise ReadingRefused(f"no file at {path.name}: {what} come first")
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    if not is_tracked(path, root):
        raise ReadingRefused(f"{relative} is not tracked by git: commit {what} first")
    if _git(root, "diff", "--quiet", "HEAD", "--", relative).returncode != 0:
        raise ReadingRefused(f"{relative} differs from its committed version at HEAD")
    return relative


_FLOAT_KEY = "float.hex"


def printable(value: Any) -> Any:
    """An instrument printout as plain JSON types that :func:`encode_thresholds` accepts.

    A float that is not finite, and an undetermined value (``None``, e.g. a kill whose
    inputs are not finite), become :data:`NON_FINITE`: the file records that the lock is
    UNDECIDABLE before its reading, never a number (§13.4). numpy scalars become Python
    ones and mapping keys strings. Shared by the three levels.
    """
    if value is None:
        return NON_FINITE
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        x = float(value)
        return x if math.isfinite(x) else NON_FINITE
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {str(k): printable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [printable(v) for v in value]
    raise TypeError(f"cannot print {type(value).__name__}")


def plain(value: Any) -> Any:
    """A reading as plain JSON types, for the reading artifact the Holm step reads back.

    numpy scalars become Python ones (a numpy bool never becomes a string), arrays and
    tuples lists, mapping keys strings, dataclass instances dicts. A float that is not
    finite **stays** a float: ``json`` writes it ``NaN`` and reads it back as NaN, so a
    verdict recomputed from the artifact is UNDECIDABLE exactly where the reading's was.
    ``repr``-exact floats make the round trip bitwise.
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return float(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return [plain(v) for v in value.tolist()]
    if isinstance(value, list | tuple | frozenset | set):
        items = sorted(value) if isinstance(value, frozenset | set) else value
        return [plain(v) for v in items]
    raise TypeError(f"cannot store {type(value).__name__} in a reading artifact")


def encode_thresholds(value: Any, where: str = "") -> Any:
    """Thresholds as JSON that round-trips bitwise: each float becomes
    ``{"float.hex": x.hex(), "repr": repr(x)}``. Refuses a non-finite float: a threshold
    that is not finite makes its lock UNDECIDABLE and is never committed as a number."""
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        x = float(value)
        if not math.isfinite(x):
            raise ValueError(f"non-finite threshold at {where or '<root>'}")
        return {_FLOAT_KEY: x.hex(), "repr": repr(x)}
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name in out or name == _FLOAT_KEY:
                raise ValueError(f"key {name!r} at {where or '<root>'} collides")
            out[name] = encode_thresholds(item, f"{where}/{name}")
        return out
    if isinstance(value, list | tuple):
        return [encode_thresholds(item, f"{where}[{i}]") for i, item in enumerate(value)]
    raise TypeError(f"cannot store {type(value).__name__} at {where or '<root>'}")


def decode_thresholds(value: Any) -> Any:
    """The inverse of :func:`encode_thresholds`: floats come back bit for bit."""
    if isinstance(value, dict):
        if set(value) == {_FLOAT_KEY, "repr"}:
            return float.fromhex(value[_FLOAT_KEY])
        return {key: decode_thresholds(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_thresholds(item) for item in value]
    return value


def _differences(a: Any, b: Any, where: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict) and _FLOAT_KEY not in a:
        keys = sorted(set(a) | set(b))
        return [d for k in keys for d in _differences(a.get(k), b.get(k), f"{where}/{k}")]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b, strict=True))
                for d in _differences(x, y, f"{where}[{i}]")]
    return [] if a == b else [where or "<root>"]


def write_thresholds(
    thresholds: Mapping[str, Any],
    *,
    section: str | None = None,
    path: Path = THRESHOLDS_PATH,
    root: Path = ROOT,
    inputs: Sequence[str] = INPUT_FILES,
) -> dict[str, Any]:
    """Write the threshold file of §13.1 step 3 and return what was written.

    The file records the lock, the SHA-256 of every input file, the versions ``uv.lock``
    pins for :data:`PINNED_PACKAGES`, the git HEAD each section was written at, and the
    thresholds, floats stored bitwise (:func:`encode_thresholds`). With ``section`` (for
    instance ``"A"``), ``thresholds`` replaces that section only, and the file's inputs
    and versions must equal the current ones; without it, ``thresholds`` is the whole
    mapping of sections. Refuses to write when the installed packages differ from
    ``uv.lock``: such a file could never pass :func:`verify_for_reading`.
    """
    root, path = Path(root), Path(path)
    hashes = input_hashes(root, inputs)
    locked = locked_versions(root)
    installed = installed_versions()
    if installed != locked:
        raise RuntimeError(f"installed packages {installed} differ from uv.lock {locked}")
    head = git_head(root)
    if section is None:
        sections = {str(k): encode_thresholds(v, f"/{k}") for k, v in thresholds.items()}
        heads = dict.fromkeys(sections, head)
    else:
        previous = json.loads(path.read_text()) if path.exists() else None
        sections, heads = {}, {}
        if previous is not None:
            if previous.get("inputs") != hashes or previous.get("packages") != locked:
                raise RuntimeError("the existing threshold file was written on other inputs or "
                                   "other packages; refusing to mix them")
            sections, heads = dict(previous["thresholds"]), dict(previous["git_head"])
        sections[section] = encode_thresholds(thresholds, f"/{section}")
        heads[section] = head
    payload = {
        "prespec": f"{PRESPEC_TAG} ({LOCK_COMMIT})",
        "inputs": hashes,
        "packages": locked,
        "git_head": heads,
        "thresholds": sections,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return payload


def read_thresholds(path: Path = THRESHOLDS_PATH) -> dict[str, Any]:
    """The threshold file with its floats decoded bit for bit."""
    payload = json.loads(Path(path).read_text())
    return {**payload, "thresholds": decode_thresholds(payload["thresholds"])}


def verify_for_reading(
    recomputed: Mapping[str, Any],
    *,
    section: str | None = None,
    path: Path = THRESHOLDS_PATH,
    root: Path = ROOT,
    inputs: Sequence[str] = INPUT_FILES,
) -> dict[str, Any]:
    """Refuse the reading unless §13.1 step 4 holds; return the stored thresholds.

    Raises :class:`ReadingRefused` unless: the threshold file is tracked by git
    (``git ls-files --error-unmatch``) and identical to its HEAD version
    (``git diff --quiet HEAD``, staged or not); every input file's SHA-256 equals the
    stored one; the versions stored, pinned by ``uv.lock`` and installed are the same;
    and ``recomputed`` — the section's thresholds when ``section`` is given, all of them
    otherwise — equals the stored thresholds **bitwise**, key for key.
    """
    root, path = Path(root), Path(path)
    if not path.exists():
        raise ReadingRefused(f"no threshold file at {path.name}: the instruments come first")
    require_committed(path, root, what="the thresholds")
    stored = json.loads(path.read_text())
    hashes = input_hashes(root, inputs)
    if stored.get("inputs") != hashes:
        changed = sorted(k for k in set(hashes) | set(stored.get("inputs", {}))
                         if hashes.get(k) != stored.get("inputs", {}).get(k))
        raise ReadingRefused(f"input files differ from the committed SHA-256: {changed}")
    locked, installed = locked_versions(root), installed_versions()
    if not stored.get("packages") == locked == installed:
        raise ReadingRefused(f"package versions differ: stored {stored.get('packages')}, "
                             f"uv.lock {locked}, installed {installed}")
    part = stored["thresholds"] if section is None else stored["thresholds"].get(section)
    if part is None:
        raise ReadingRefused(f"no committed thresholds for section {section!r}")
    encoded = json.loads(json.dumps(encode_thresholds(recomputed)))
    differences = _differences(encoded, part)
    if differences:
        raise ReadingRefused(f"recomputed thresholds differ bitwise from the committed ones "
                             f"at {differences}")
    return decode_thresholds(part)


# ---------------------------------------------------------------------------------------
# the deterministic process-pool map
# ---------------------------------------------------------------------------------------

_SHARED: Any = None
_LIMITS: Any = None


def _install(shared: Any) -> None:
    global _SHARED, _LIMITS
    _SHARED = shared
    _LIMITS = threadpool_limits(limits=1)


def _call(fn: Callable[[Any, int], Any], j: int) -> Any:
    return fn(_SHARED, j)


def draw_map[T](
    fn: Callable[[Any, int], T],
    n: int,
    shared: Any = None,
    *,
    workers: int | None = None,
    chunksize: int | None = None,
) -> list[T]:
    """``[fn(shared, j) for j in range(n)]``, in a process pool, ordered by ``j``.

    For the placebo loops (1,000 draws per variant). ``fn`` must be a module-level
    function and must depend on ``(shared, j)`` alone — derive any randomness from *j* —
    so the result is the same whatever ``workers`` is. ``shared`` (the tree's data, the
    placebo draws, …) is pickled once per worker, not once per task. Workers start with
    the ``spawn`` method on every platform, and every call, in a worker or in-process,
    runs with native thread pools limited to one thread, so that a result does not
    depend on how many processes computed it. ``workers`` defaults to the CPU count;
    ``workers <= 1`` runs in-process.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    workers = (os.cpu_count() or 1) if workers is None else workers
    if workers <= 1 or n <= 1:
        with threadpool_limits(limits=1):
            return [fn(shared, j) for j in range(n)]
    workers = min(workers, n)
    size = chunksize or max(1, n // (4 * workers))
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_install,
                             initargs=(shared,)) as pool:
        return list(pool.map(_call, itertools.repeat(fn, n), range(n), chunksize=size))


# ---------------------------------------------------------------------------------------
# synthetic data, for tests only
# ---------------------------------------------------------------------------------------


def synthetic_inputs(
    seed: int = 0,
    *,
    n_industries: int = 12,
    start: str = "2000-01-03",
    end: str = "2013-12-31",
    n_regimes: int = 4,
    persistence: float = 0.95,
) -> dict[str, Any]:
    """Raw synthetic inputs for :func:`build_tree_data`, keyed by its argument names.

    **For tests only; nothing here reads ``data/``.** A latent Markov regime (stay
    probability ``persistence``) moves the twenty context features, so K-means finds
    states with multi-session episodes; a persistent log-volatility process drives a
    market factor, the industries (beta times the market plus noise), ``eq_us_large`` and
    its log realised volatility, and leaks into the features so that orthogonalisation
    matters. The industry calendar drops about 2% of weekdays as holidays, which the
    feature calendar keeps, as on the real store. Returns carry no planted state content.
    """
    rng = np.random.default_rng(seed)
    calendar = pd.bdate_range(start, end)
    n = len(calendar)
    holidays = rng.random(n) < 0.02
    holidays[:2] = False
    trading = calendar[~holidays]

    regime = np.zeros(n, dtype=np.int64)
    switch = rng.random(n) > persistence
    step = rng.integers(1, n_regimes, n)
    for t in range(1, n):
        regime[t] = (regime[t - 1] + step[t]) % n_regimes if switch[t] else regime[t - 1]
    z = np.zeros(n)
    shocks = rng.normal(0.0, 0.08, n)
    for t in range(1, n):
        z[t] = 0.98 * z[t - 1] + shocks[t]
    vol = 0.01 * np.exp(z + 0.15 * regime)
    market = rng.normal(0.0002, 1.0, n) * vol
    betas = rng.uniform(0.6, 1.4, n_industries)
    returns = market[:, None] * betas[None, :] + rng.normal(0.0, 0.008, (n, n_industries))
    names = [f"ind_{i:02d}" for i in range(n_industries)]
    industries = pd.DataFrame(returns, index=calendar, columns=names).loc[trading]

    rf = np.full(len(trading), 0.008)  # percent a day
    mkt_rf = market[~holidays] * 100.0 - rf
    factors = pd.concat([
        pd.DataFrame({"series_id": sid, "period": trading, "available_at": trading, "value": v})
        for sid, v in (("ff_rf", rf), ("ff_mkt-rf", mkt_rf))
    ], ignore_index=True)
    prices = pd.Series(100.0 * np.exp(np.cumsum(np.log1p(market))), index=calendar)
    centres = rng.normal(0.0, 2.0, (n_regimes, len(CONTEXT_FEATURES)))
    level = (z - z.mean()) / (z.std() or 1.0)
    features = pd.DataFrame(
        centres[regime] + rng.normal(0.0, 1.0, (n, len(CONTEXT_FEATURES)))
        + 0.5 * level[:, None],
        index=calendar, columns=list(CONTEXT_FEATURES),
    )
    return {
        "industries": industries,
        "factors": factors,
        "library_market": pd.Series(mkt_rf / 100.0, index=trading, name="ff_mkt-rf"),
        "features": features,
        "log_rv": log_realised_vol(prices.loc[trading]),
    }


def synthetic_tree_data(
    seed: int = 0, *, n_folds: int = 5, test_years: float = 1.0, **kwargs: Any
) -> TreeData:
    """:func:`build_tree_data` on :func:`synthetic_inputs` — **for tests only**.

    The ten signals are complete about five years after the start (IND_SEASON's
    warm-up); the sessions are those complete sessions, and the folds ``n_folds`` test
    windows of ``test_years`` anchored at the end (5 x 1.0 y by default, about 1,280
    test sessions out of 2,300). ``kwargs`` go to :func:`synthetic_inputs`. A test that
    needs planted content replaces fields with ``dataclasses.replace``.
    """
    return build_tree_data(**synthetic_inputs(seed, **kwargs), n_folds=n_folds,
                           test_years=test_years)
