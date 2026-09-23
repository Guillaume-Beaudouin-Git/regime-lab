"""The context partition of the Two Sigma plan — `PRESPEC_TWOSIGMA.md` §4.

What the draft fixes. **Twenty context features, no ``vol_*`` column**, orthogonalised
session by session against the **log 21-day realised volatility of ``eq_us_large``**
on training data only; **K-means, K = 4, n_init = 20, random_state = 0**, refitted on
each training window; labels assigned **out of sample by nearest centroid**.

What the draft leaves open, and the reading taken here — each inherited from the
pre-lock scripts that produced the disclosed clock, so that the declared object and
the disclosed one differ only where §4 says they must:

* **standardisation** — z-scores on the training window, population standard
  deviation, before the regression; the residuals are not re-standardised;
* **the regression** — one slope per feature on the standardised log volatility,
  ``<v, z> / <v, v>``. Both sides are centred on the training window, so this is OLS
  with an intercept, fitted on training rows and applied unchanged to test rows;
* **alignment of the volatility** — as of the close: a feature date that is not a
  trading day of ``eq_us_large`` carries the last value known;
* **windows on the feature calendar** — the context panel lives on the feature
  calendar (every weekday), not on the industry sessions the folds are cut from. A
  fold's windows are therefore taken by its calendar cut-offs: training rows up to
  and including ``train_cutoff``, test rows after it up to ``test_cutoff``. The
  feature calendar is tiled with no row left out, and no row after the cut-off can
  enter a fit;
* **one thread** — scikit-learn's K-means reduces in parallel, and the order of a
  parallel sum changes its last bit from run to run: the same training rows gave
  centroids differing by 2e-16 between two calls. A pre-registered partition must
  come out bit-identical from the same code, so every fit and every prediction here
  runs on one thread. The disclosed counts are unchanged, and it is faster on panels
  this size;
* **label numbering across refits** — K-means numbers its clusters arbitrarily, so
  state 2 of one fold need not be state 2 of the next. Nothing inside a fold needs an
  alignment. A statistic pooled across folds does, and :func:`walk_forward_context`
  supplies one by matching each refit to the previous on the rows both were trained
  on. Transitions at fold boundaries are only meaningful after it.

**Timing.** A label is stamped on the session whose close produced its features. A
consumer that trades on it must lag it one session; this module never lags.
:func:`session_paths` gives each fold's labels on the industry sessions in that
fold's own numbering, training segment first, so that a one-session lag inside a
fold hands the first test session the label its own model stamped on the last
training session — never a label of the previous fold's model, whose numbering is
different.

**η², a unit trap in the disclosure.** §1 P3 sets η² = 0.048 (orthogonalised)
against 0.421 (not orthogonalised). The first was computed by the pre-lock script
against *log* realised volatility, the second against its *level*. On one scale the
pair reads 0.048 / 0.441 (log) or 0.085 / 0.421 (level); the contrast survives, the
quoted pair is not like for like.

**Blindness.** Nothing here reads a return of any signal or of any industry. The
only market series used is the volatility the partition is orthogonalised against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from regime_lab.analysis.placebo import run_lengths, transitions
from regime_lab.config import CACHE, RAW
from regime_lab.selection.folds import Fold

#: §4, grouped as the pre-lock script grouped them: liquidity and credit, rates,
#: cross-sectional dispersion, choppiness.
CONTEXT_FEATURES: tuple[str, ...] = (
    "fin_nfci", "fin_nfci_chg13w", "cre_baa", "cre_baa_chg63", "cre_quality",
    "cre_quality_chg63",
    "rat_slope_10y2y", "rat_slope_10y2y_chg63", "rat_slope_10y3m_chg63", "rat_cash_chg12m",
    "xs_dispersion_ind", "xs_dispersion_szbm", "xs_avg_corr", "xs_absorption",
    "xs_absorption_chg", "xs_breadth_63",
    "asy_vr_5", "asy_vr_20", "asy_hurst", "mom_breadth_200d",
)
VOLATILITY_SERIES = "eq_us_large"
RV_WINDOW = 21
N_STATES = 4
N_INIT = 20
RANDOM_STATE = 0
#: §4 sensitivities: reported, never able to found a PASS.
SENSITIVITY_STATES = (3, 5, 6)
#: Column holding the log realised volatility in a context panel.
LOG_RV = "log_rv"
#: The two segments of a fold's label path, in calendar order.
SEGMENTS = ("train", "test")
#: The lock's minimum-cell rule (§12.4): a state qualifies in a (fold, segment) block
#: when it holds at least this many sessions and this many episodes there.
MIN_CELL_SESSIONS = 63
MIN_CELL_EPISODES = 3
#: The lock's smoothing sensitivity (§12.13, from PLAN.md §f.4): a trailing mode over
#: this many sessions of a fold's stamped path.
SMOOTHING_WINDOW = 21


def load_context_features(path: Path | None = None) -> pd.DataFrame:
    """The twenty §4 features from ``data/cache/features.parquet``, on its own calendar."""
    frame = pd.read_parquet(path or CACHE / "features.parquet", columns=list(CONTEXT_FEATURES))
    return frame.sort_index()


def log_realised_vol(prices: pd.Series, window: int = RV_WINDOW) -> pd.Series:
    """Log of annualised realised volatility: sd of daily log returns over ``window``.

    Includes the return of the session it is stamped on, so it is known at that close.
    """
    returns = np.log(prices.sort_index()).diff()
    return np.log(returns.rolling(window).std() * np.sqrt(252)).rename(LOG_RV)


def load_log_realised_vol(
    path: Path | None = None, *, series: str = VOLATILITY_SERIES, window: int = RV_WINDOW
) -> pd.Series:
    """:func:`log_realised_vol` of one series of ``data/raw/prices/cross_asset.parquet``."""
    frame = pd.read_parquet(path or RAW / "prices" / "cross_asset.parquet")
    prices = frame.loc[frame["series_id"] == series].set_index("period")["value"]
    if prices.empty:
        raise ValueError(f"{series!r} is not in the cross-asset file")
    if prices.index.duplicated().any():
        raise ValueError(f"{series!r} has more than one value on some dates")
    return log_realised_vol(prices.astype(float), window)


def as_of(values: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """``values`` read at each date of ``index``: the last observation at or before it.

    Never looks forward. A date before the first observation reads NaN.
    """
    union = values.index.union(index)
    return values.reindex(union).ffill().reindex(index)


def context_panel(features: pd.DataFrame, log_rv: pd.Series) -> pd.DataFrame:
    """The context features plus the as-of log volatility, complete rows only."""
    missing = [c for c in CONTEXT_FEATURES if c not in features.columns]
    if missing:
        raise KeyError(f"context features missing: {missing}")
    panel = features[list(CONTEXT_FEATURES)].copy()
    panel[LOG_RV] = as_of(log_rv, panel.index)
    return panel.dropna()


@dataclass(frozen=True)
class ContextModel:
    """A partition fitted on one training window, applicable to any later session."""

    columns: tuple[str, ...]
    mean: pd.Series
    scale: pd.Series
    vol_mean: float
    vol_scale: float
    slopes: pd.Series | None
    kmeans: KMeans
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    train_labels: pd.Series

    @property
    def orthogonalised(self) -> bool:
        """Whether the features are residualised on log realised volatility."""
        return self.slopes is not None

    @property
    def centroids(self) -> np.ndarray:
        """Cluster centres in the (residualised) standardised space."""
        return self.kmeans.cluster_centers_

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Standardise with training moments and, if orthogonalised, residualise."""
        z = (panel[list(self.columns)] - self.mean) / self.scale
        if self.slopes is None:
            return z
        v = ((panel[LOG_RV] - self.vol_mean) / self.vol_scale).to_numpy()
        return z - np.outer(v, self.slopes.to_numpy())

    def predict(self, panel: pd.DataFrame) -> pd.Series:
        """Nearest-centroid labels, stamped on the panel's dates.

        Each row's label depends on that row and on the training fit only.
        """
        x = self.transform(panel).to_numpy()
        with threadpool_limits(limits=1):
            labels = self.kmeans.predict(x)
        return pd.Series(labels, index=panel.index, name="state")


def fit_context(
    panel: pd.DataFrame,
    *,
    orthogonalise: bool = True,
    n_states: int = N_STATES,
    n_init: int = N_INIT,
    random_state: int = RANDOM_STATE,
) -> ContextModel:
    """Fit the §4 partition on ``panel``, which must hold training rows only.

    Every column but ``log_rv`` is a feature; ``log_rv`` is needed only to
    orthogonalise. Moments, slopes and centroids all come from ``panel`` alone, and
    come out bit-identical from one call to the next (one thread).
    """
    if panel.isna().any().any():
        raise ValueError("the training panel has missing values; build it with context_panel")
    columns = tuple(c for c in panel.columns if c != LOG_RV)
    x = panel[list(columns)]
    mean, scale = x.mean(), x.std(ddof=0)
    flat = scale.index[~(scale > 0)].tolist()
    if flat:
        raise ValueError(f"features constant on the training window: {flat}")
    z = (x - mean) / scale
    vol_mean, vol_scale = float("nan"), float("nan")
    slopes = None
    if orthogonalise:
        vol = panel[LOG_RV]
        vol_mean, vol_scale = float(vol.mean()), float(vol.std(ddof=0))
        if not vol_scale > 0:
            raise ValueError("log realised volatility is constant on the training window")
        v = ((vol - vol_mean) / vol_scale).to_numpy()
        slopes = pd.Series(
            [np.dot(v, z[c].to_numpy()) / np.dot(v, v) for c in columns], index=list(columns)
        )
        z = z - np.outer(v, slopes.to_numpy())
    kmeans = KMeans(n_clusters=n_states, n_init=n_init, random_state=random_state)
    with threadpool_limits(limits=1):
        labels = kmeans.fit_predict(z.to_numpy())
    return ContextModel(
        columns=columns,
        mean=mean,
        scale=scale,
        vol_mean=vol_mean,
        vol_scale=vol_scale,
        slopes=slopes,
        kmeans=kmeans,
        train_start=panel.index.min(),
        train_end=panel.index.max(),
        train_labels=pd.Series(labels, index=panel.index, name="state"),
    )


def align_labels(reference: pd.Series, candidate: pd.Series, n_states: int) -> np.ndarray:
    """The relabelling of ``candidate`` that agrees most with ``reference``.

    Both series label the same dates. Returns ``mapping`` such that
    ``mapping[candidate]`` is in the reference's numbering, chosen by maximising the
    number of dates on which the two agree (Hungarian assignment).
    """
    both = pd.concat([reference, candidate], axis=1, join="inner").to_numpy(dtype=np.int64)
    table = np.zeros((n_states, n_states), dtype=np.int64)
    np.add.at(table, (both[:, 1], both[:, 0]), 1)
    rows, cols = linear_sum_assignment(-table)
    mapping = np.empty(n_states, dtype=np.int64)
    mapping[rows] = cols
    return mapping


@dataclass(frozen=True)
class WalkForwardContext:
    """Out-of-sample labels of a walk-forward refit.

    ``labels`` are in each fold's own numbering; ``aligned`` renumbers every fold to
    fold 1's through a chain of :func:`align_labels`. ``agreement[k]`` is the share
    of fold *k*'s training rows on which it agrees with fold *k-1* after alignment
    (1.0 for fold 1). ``numbers`` are the fold numbers, in the order of ``models``;
    ``mappings[k]`` renumbers fold *k*'s states into fold 1's
    (``aligned = mappings[k][labels]``).
    """

    labels: pd.Series
    aligned: pd.Series
    fold: pd.Series
    models: tuple[ContextModel, ...]
    agreement: tuple[float, ...]
    numbers: tuple[int, ...]
    mappings: tuple[np.ndarray, ...]

    def model(self, number: int) -> ContextModel:
        """The partition fitted for fold ``number``."""
        return self.models[self.numbers.index(number)]

    def test_labels(self, number: int) -> pd.Series:
        """Fold ``number``'s out-of-sample labels, in its own numbering."""
        return self.labels.loc[self.fold == number]

    def path(self, number: int) -> pd.Series:
        """Every label fold ``number``'s model stamps, in its own numbering.

        Its in-sample training labels followed by its out-of-sample test labels, on
        the feature calendar. The two never overlap: training rows end at the cut-off
        the test rows start after.
        """
        return pd.concat([self.model(number).train_labels, self.test_labels(number)])

    def within_fold_transitions(self) -> int:
        """Transitions inside test folds, which need no alignment."""
        return sum(transitions(part) for _, part in self.labels.groupby(self.fold))

    def boundary_transitions(self) -> int:
        """Transitions across fold boundaries, in the aligned numbering."""
        return transitions(self.aligned) - transitions_within(self.aligned, self.fold)


def transitions_within(labels: pd.Series, groups: pd.Series) -> int:
    """Transitions counted inside each group only."""
    return sum(transitions(part) for _, part in labels.groupby(groups.reindex(labels.index)))


def fold_windows(
    index: pd.DatetimeIndex, fold: Fold, train_from: pd.Timestamp | None = None
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """The dates of ``index`` in a fold's training and test windows, by its cut-offs.

    Training runs from ``train_from`` (default: the fold's first training session)
    through ``train_cutoff``; the test window is ``(train_cutoff, test_cutoff]``.
    """
    start = fold.train_start if train_from is None else train_from
    train = index[(index >= start) & (index <= fold.train_cutoff)]
    test = index[(index > fold.train_cutoff) & (index <= fold.test_cutoff)]
    return train, test


def walk_forward_context(
    panel: pd.DataFrame,
    folds: tuple[Fold, ...],
    *,
    orthogonalise: bool = True,
    train_from: pd.Timestamp | None = None,
    n_states: int = N_STATES,
    n_init: int = N_INIT,
    random_state: int = RANDOM_STATE,
) -> WalkForwardContext:
    """Refit the partition on each fold's training window, label its test window.

    ``train_from`` moves the start of every training window, for instance to the
    first date the context features exist rather than the first date the signals
    do; by default each fold trains from its own ``train_start``. Windows are taken
    by the folds' calendar cut-offs (:func:`fold_windows`).
    """
    models, pieces, aligned, fold_ids, agreement, mappings = [], [], [], [], [], []
    previous: ContextModel | None = None
    previous_map = np.arange(n_states)
    for fold in folds:
        train_dates, test_dates = fold_windows(panel.index, fold, train_from)
        train, test = panel.loc[train_dates], panel.loc[test_dates]
        if train.empty or test.empty:
            raise ValueError(f"fold {fold.number}: empty training or test panel")
        model = fit_context(
            train, orthogonalise=orthogonalise, n_states=n_states, n_init=n_init,
            random_state=random_state,
        )
        mapping = np.arange(n_states)
        share = 1.0
        if previous is not None:
            reference = pd.Series(previous_map[previous.predict(train)], index=train.index)
            mapping = align_labels(reference, model.train_labels, n_states)
            share = float((mapping[model.train_labels.to_numpy()] == reference.to_numpy()).mean())
        labels = model.predict(test)
        models.append(model)
        pieces.append(labels)
        aligned.append(pd.Series(mapping[labels.to_numpy()], index=labels.index))
        fold_ids.append(pd.Series(fold.number, index=labels.index))
        agreement.append(share)
        mappings.append(mapping)
        previous, previous_map = model, mapping
    return WalkForwardContext(
        labels=pd.concat(pieces).rename("state"),
        aligned=pd.concat(aligned).rename("state"),
        fold=pd.concat(fold_ids).rename("fold"),
        models=tuple(models),
        agreement=tuple(agreement),
        numbers=tuple(f.number for f in folds),
        mappings=tuple(mappings),
    )


def session_paths(
    wf: WalkForwardContext, folds: tuple[Fold, ...], sessions: pd.DatetimeIndex
) -> pd.Series:
    """Each fold's label path read as of the industry sessions it covers.

    Indexed by ``(fold, segment, session)``. Segment ``"train"`` holds the fold's
    in-sample training labels on the sessions of its training window
    (``fold.train(sessions)``), segment ``"test"`` its out-of-sample labels on its
    test sessions — both from :meth:`WalkForwardContext.path`, in the fold's own
    numbering, read by :func:`as_of`, and **stamped, not lagged**.

    To trade fold *k*, take ``paths.xs(k, level="fold").droplevel("segment")`` —
    training then test sessions of one model — and lag *that* one session (for
    instance ``protocol.map_states``): the first test session then holds the label
    fold *k*'s model stamped on the last training session. The training segment is
    also what a per-state estimate on training folds would read, and the pair of
    segments is what the level-A and level-B placebos must redraw
    (:func:`placebo_groups`).

    A session with no label at or before it on its fold's path reads NaN; the
    values are integers when there is none.
    """
    if tuple(f.number for f in folds) != wf.numbers:
        raise ValueError("the folds are not the ones the walk-forward was fitted on")
    pieces = []
    for fold in folds:
        path = wf.path(fold.number)
        for segment, dates in zip(SEGMENTS, (fold.train(sessions), fold.test(sessions)),
                                  strict=True):
            index = pd.MultiIndex.from_arrays(
                [np.full(len(dates), fold.number), np.full(len(dates), segment), dates],
                names=["fold", "segment", "session"],
            )
            pieces.append(pd.Series(as_of(path, dates).to_numpy(), index=index))
    out = pd.concat(pieces).rename("state")
    return out if out.isna().any() else out.astype(np.int64)


def placebo_groups(paths: pd.Series) -> pd.Series:
    """The ``(fold, segment)`` block of every row of :func:`session_paths`, as text.

    Passed as ``groups`` to ``analysis.placebo.matched_placebos``: each fold's
    training labels and test labels are then redrawn separately, each block keeping
    its own clock, occupancy and episodes exactly.
    """
    fold = paths.index.get_level_values("fold").astype(str)
    segment = paths.index.get_level_values("segment").astype(str)
    return pd.Series(fold + ":" + segment, index=paths.index, name="block")


def qualifying_cells(
    paths: pd.Series,
    *,
    min_sessions: int = MIN_CELL_SESSIONS,
    min_episodes: int = MIN_CELL_EPISODES,
) -> pd.DataFrame:
    """Sessions and episodes of each state in each (fold, segment) block; does it qualify.

    ``paths`` is :func:`session_paths` output, or a copy of it such as
    :func:`trailing_mode`'s. Counts are taken on the **stamped** path of each block, so
    a placebo that keeps every block's per-state sessions and episodes qualifies
    exactly the same cells. A session with no label belongs to no state, and it still
    separates the episodes on either side of it. Indexed by ``(fold, segment, state)``
    for every state seen anywhere in ``paths``; a state absent from a block has zero
    sessions and does not qualify.
    """
    labels = paths.dropna()
    states = np.sort(labels.unique()).astype(np.int64)
    rows = []
    for (fold, segment), block in paths.groupby(level=["fold", "segment"], sort=False):
        values = block.to_numpy(dtype=float)
        coded = np.where(np.isfinite(values), values, -1.0).astype(np.int64)
        _, run_states = run_lengths(coded)
        for state in states:
            sessions = int((coded == state).sum())
            episodes_ = int((run_states == state).sum())
            rows.append((fold, segment, int(state), sessions, episodes_))
    table = pd.DataFrame(rows, columns=["fold", "segment", "state", "sessions", "episodes"])
    table["qualifies"] = (table["sessions"] >= min_sessions) & (table["episodes"] >= min_episodes)
    return table.set_index(["fold", "segment", "state"])


def trailing_mode(paths: pd.Series, window: int = SMOOTHING_WINDOW) -> pd.Series:
    """The lock's smoothing sensitivity: a trailing mode of each fold's stamped path.

    Per fold, over its whole path (training sessions then test sessions, in the fold's
    own numbering), session *t* takes the label seen most often on the ``window``
    sessions ending at *t*, itself included; a tie goes to the smallest label. It is
    applied to stamped labels, before any lag, so it reads nothing after *t*. The first
    ``window - 1`` sessions of each path have no full window and read NaN, which a
    consumer holds as its fallback; nothing is back-filled, and no window reaches into
    another fold's path.
    """
    if window < 1:
        raise ValueError("window must be at least one session")
    if paths.isna().any():
        raise ValueError("the stamped path must be fully labelled")
    out = pd.Series(np.nan, index=paths.index, name=paths.name)
    n_states = int(paths.max()) + 1
    for fold in paths.index.get_level_values("fold").unique():
        mask = paths.index.get_level_values("fold") == fold
        values = paths.to_numpy()[mask].astype(np.int64)
        one_hot = np.zeros((len(values) + 1, n_states), dtype=np.int64)
        one_hot[np.arange(1, len(values) + 1), values] = 1
        cumulative = one_hot.cumsum(axis=0)
        if len(values) < window:
            continue
        counts = cumulative[window:] - cumulative[:-window]
        mode = np.full(len(values), np.nan)
        mode[window - 1:] = counts.argmax(axis=1)
        out.iloc[np.flatnonzero(mask)] = mode
    return out


def eta_squared(labels: pd.Series, values: pd.Series) -> float:
    """Share of the variance of ``values`` explained by the partition (one-way R²).

    Computed on the dates where both are present.
    """
    both = pd.concat([labels.rename("g"), values.rename("y")], axis=1, join="inner").dropna()
    if both.empty:
        return float("nan")
    y = both["y"].to_numpy(dtype=float)
    grand = y.mean()
    total = float(((y - grand) ** 2).sum())
    groups = both.groupby("g")["y"]
    between = float((groups.size() * (groups.mean() - grand) ** 2).sum())
    return between / total if total > 0 else float("nan")
