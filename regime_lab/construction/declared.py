"""Bridgewater study — the declared objects, assembled in one place from the stored inputs.

`docs/PRESPEC_BRIDGEWATER.md` §12.4 names this module. Every instrument, reading and
sensitivity builds its object here, so that the object the thresholds were measured on
is the object a reading reads. Before this module existed, the chain that turns prices
and SPF files into returns, labels and folds lived only in
``scripts/measure_bridgewater_power.py::build_declared`` (commit ``39ef608``).

**Inputs, and the content they are read to** (§12.1). The instruments and readings read
every file of :data:`INPUT_FILES` through this module only. A re-fetch appends rows (a
survey every quarter, a FRED value every day), so the object reads only the content up
to fixed cut-offs:
surveys up to :data:`SURVEY_CUTOFF`, real-time quarters up to :data:`RTDSM_CUTOFF`,
``T10YIE`` periods up to :data:`T10YIE_CUTOFF`. :func:`content_hashes` fingerprints that
content, so a re-fetch that only appended rows reproduces it, and one that revised a
used value does not.

**Two objects.**

- The *stamped* object (:func:`declared_object`): the quadrant of reference quarter
  ``q`` held from the release date of survey ``q+1``, lagged one session; folds cut at
  the stamps by `evaluate.stamp_folds`. It is the object of level C, an execution use,
  and of the stamped-label sensitivity. It is identical to ``build_declared`` at
  ``39ef608`` (the measurement script checks it).
- The *contemporaneous* object (:func:`contemporaneous_object`): every session carries
  the label of the calendar quarter it lies in, which is the environment in force on
  that session. Levels A, B1 and B2 read it. The label enters only the objective and
  the evaluation, never a traded decision, so it need not be known on the session it
  describes (§0.1). What must be known is what a fold fits on: a training quarter
  enters the cell covariances of a fold only if its label was public by the close of
  the fold's last training session (:func:`fold_paths`). Folds are cut at quarter ends
  (:func:`quarter_folds`).

**Shared assembly** (§12.4): returns are ``sleeves.price_returns`` on the whole stored
panel, funded by ``sleeves.cash_rate`` on the panel's index, then kept from the
configuration's start. The within-sleeve inverse-volatility weights are re-estimated at
every SPF release stamp between the first and the last session (a label-free calendar),
at the defaults of ``sleeves.within_sleeve_weights`` (252 sessions, ``min_valid`` 0.5,
daily), and held by ``sleeves.hold``. ``usable`` is the sessions on which every sleeve
return is finite.

Nothing here computes a statistic conditioned on a label. :func:`traded_blind_driver`
and :func:`quarterly_volatility_labels` read returns, but only to build the state-blind
driver and the volatility witness.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from regime_lab.config import ROOT
from regime_lab.construction import evaluate as E
from regime_lab.construction import quadrant as Q
from regime_lab.construction import sleeves as S
from regime_lab.data.pit import validate
from regime_lab.selection.folds import Fold

#: §12.1: every input of the study, and nothing else, relative to the repository root.
SPF_FILES: tuple[str, ...] = (
    "data/raw/spf/median_rgdp_level.xlsx",
    "data/raw/spf/median_cpi_level.xlsx",
    "data/raw/spf/median_unemp_level.xlsx",
    "data/raw/spf/median_indprod_level.xlsx",
    "data/raw/spf/median_rgdp_growth.xlsx",
    "data/raw/spf/routput_first_second_third.xlsx",
    "data/raw/spf/spf_release_dates.txt",
    "data/raw/spf/spf_release_dates.csv",
    "data/raw/spf/fred_t10yie.parquet",
)
CACHED_FILES: tuple[str, ...] = (
    "data/cache/trend_universe_m1.parquet",
    "data/raw/macro/rate_cash_3m.parquet",
    "data/raw/macro/fin_baa_spread.parquet",
)
INPUT_FILES: tuple[str, ...] = SPF_FILES + CACHED_FILES

#: The last survey, real-time quarter and FRED period the declared object reads.
SURVEY_CUTOFF = pd.Period("2026Q3", freq="Q")
RTDSM_CUTOFF = pd.Period("2026Q2", freq="Q")
T10YIE_CUTOFF = pd.Timestamp("2026-09-23")

#: B1 (§12.12): the change window, in sessions.
B1_WINDOW = 252
#: The witness W2 (§12.15): quantile bins of the quarter's realised volatility.
WITNESS_BINS = 4
#: A quarter enters the witness only with at least this many finite driver returns.
WITNESS_MIN_SESSIONS = 21

#: §12.4 and §12.18: the eight columns whose close precedes the US close of the same
#: date (Asia by about 15 h, Europe by about 4.5 h).
FOREIGN_CLOSES: tuple[str, ...] = (
    "^N225", "^HSI", "^AXJO", "^GDAXI", "^FTSE", "^AEX", "^FCHI", "^IBEX",
)


# ------------------------------------------------------------------------ the inputs


def _path(relative: str, root: Path) -> Path:
    return Path(root) / relative


def _canonical(frame: pd.DataFrame) -> bytes:
    """Bytes that depend on the values only: floats by `float.hex`, dates in ISO."""
    lines = [",".join(["index", *map(str, frame.columns)])]
    for key, row in zip(frame.index, frame.itertuples(index=False), strict=True):
        cells = [str(key)]
        for value in row:
            if isinstance(value, float | np.floating):
                cells.append("nan" if np.isnan(value) else float(value).hex())
            elif isinstance(value, pd.Timestamp):
                cells.append(value.isoformat())
            else:
                cells.append(str(value))
        lines.append(",".join(cells))
    return ("\n".join(lines) + "\n").encode()


def _spf_content(content: bytes, name: str) -> pd.DataFrame:
    frame = Q.verify_workbook(content, required=(), label=name)
    table = Q.spf_table(frame, label=name)
    return table.loc[table.index <= SURVEY_CUTOFF]


def _release_content(table: pd.DataFrame) -> pd.DataFrame:
    kept = table.loc[table.index <= SURVEY_CUTOFF, ["deadline", "release", "note"]]
    return kept.astype({"note": str})


def _t10yie_content(frame: pd.DataFrame) -> pd.DataFrame:
    kept = frame.loc[frame["period"] <= T10YIE_CUTOFF,
                     ["series_id", "period", "available_at", "value"]]
    kept = kept.sort_values("period", kind="stable").reset_index(drop=True)
    return kept.astype({"value": float})


def content_of(relative: str, content: bytes) -> bytes:
    """The canonical content the declared object reads from one input file (§12.1).

    - SPF median workbooks: the parsed table, surveys up to :data:`SURVEY_CUTOFF`.
    - The real-time file: the first-release column, quarters up to :data:`RTDSM_CUTOFF`.
    - The release-date table (text or CSV): surveys up to :data:`SURVEY_CUTOFF`.
    - ``fred_t10yie.parquet``: rows with ``period`` up to :data:`T10YIE_CUTOFF`.
    - Anything else (the three cached files): the bytes themselves.
    """
    name = Path(relative).name
    if name.startswith("median_") and name.endswith(".xlsx"):
        return _canonical(_spf_content(content, name))
    if name == Q.RTDSM_ROUTPUT_FILE:
        first = Q.read_rtdsm_first_release(content)
        return _canonical(first.loc[first.index <= RTDSM_CUTOFF].to_frame())
    if name == Q.RELEASE_DATES_TEXT:
        return _canonical(_release_content(Q.parse_release_dates(content.decode("ascii"))))
    if name == Q.RELEASE_DATES_CSV:
        raw = pd.read_csv(io.BytesIO(content), dtype={"note": str}, keep_default_na=False)
        table = pd.DataFrame({
            "deadline": pd.to_datetime(raw["deadline"]).to_numpy(),
            "release": pd.to_datetime(raw["release"]).to_numpy(),
            "note": raw["note"].to_numpy(),
        }, index=pd.PeriodIndex(raw["survey"], freq="Q", name="survey"))
        return _canonical(_release_content(table))
    if name == Q.T10YIE_FILE:
        return _canonical(_t10yie_content(validate(pd.read_parquet(io.BytesIO(content)))))
    return content


def byte_hashes(root: Path = ROOT, inputs: Sequence[str] = INPUT_FILES) -> dict[str, str]:
    """SHA-256 of the bytes of every input (provenance; §12.1)."""
    return {r: hashlib.sha256(_path(r, root).read_bytes()).hexdigest() for r in inputs}


def content_hashes(root: Path = ROOT, inputs: Sequence[str] = INPUT_FILES) -> dict[str, str]:
    """SHA-256 of the content the object reads from every input (:func:`content_of`).

    A reading requires these to equal the ones committed with the thresholds (§12.1,
    §13.1). A re-fetch that only appends rows beyond the cut-offs reproduces them.
    """
    return {r: hashlib.sha256(content_of(r, _path(r, root).read_bytes())).hexdigest()
            for r in inputs}


@dataclass(frozen=True)
class QuadrantInputs:
    """The SPF tables the quadrant reads, cut at :data:`SURVEY_CUTOFF`."""

    rgdp: pd.DataFrame
    cpi: pd.DataFrame
    release: pd.DataFrame
    drgdp2: pd.Series
    first_release: pd.Series


def load_quadrant_inputs(root: Path = ROOT) -> QuadrantInputs:
    """Read and re-verify the SPF inputs, keeping only the content :func:`content_of` hashes."""
    spf = Path(root) / "data" / "raw" / "spf"

    def median(name: str, columns: tuple[str, ...]) -> pd.DataFrame:
        table = Q.read_spf(spf / name, columns)
        return table.loc[table.index <= SURVEY_CUTOFF]

    release = Q.read_release_dates(spf / Q.RELEASE_DATES_CSV)
    first = Q.read_rtdsm_first_release((spf / Q.RTDSM_ROUTPUT_FILE).read_bytes())
    return QuadrantInputs(
        rgdp=median("median_rgdp_level.xlsx", ("RGDP1", "RGDP2")),
        cpi=median("median_cpi_level.xlsx", ("CPI1", "CPI2")),
        release=release.loc[release.index <= SURVEY_CUTOFF],
        drgdp2=median(Q.SPF_GROWTH_FILE[0], Q.SPF_GROWTH_FILE[1])["DRGDP2"],
        first_release=first.loc[first.index <= RTDSM_CUTOFF],
    )


def quarterly_quadrant(
    inputs: QuadrantInputs, *, growth: str = "growth", extra_months: int = 0
) -> pd.DataFrame:
    """Label, stamp and stamp source of every reference quarter (§1.1, §12.2, §12.3).

    ``growth="growth"`` is the declared axis; ``"growth_within_vintage"`` the §12.3
    sensitivity, whose undefined quarters are dropped. ``extra_months`` moves every
    stamp later by whole months (the §12.2 timing rerun).
    """
    if growth == "growth":
        panel = Q.surprise_panel(inputs.rgdp, inputs.cpi)
    else:
        panel = Q.surprise_panel(inputs.rgdp, inputs.cpi, first_release=inputs.first_release,
                                 drgdp2=inputs.drgdp2).dropna(subset=[growth])
    stamps = Q.availability(panel.index, rule="release", release_dates=inputs.release,
                            extra_months=extra_months)
    return Q.quarterly_labels(panel, stamps, growth=growth)


def stamped_series(quarterly: pd.DataFrame) -> pd.Series:
    """The quarterly labels indexed by ``available_at`` (§12.4 (iii)).

    Sorted stably by ``available_at``; where two quarters share a stamp (1990Q1 and
    1990Q2) the later quarter is kept. Identical to the power script's copy at
    ``39ef608`` on this data, where no stamp in the asset window is shared.
    """
    table = quarterly.sort_values("available_at", kind="stable")
    table = table.drop_duplicates("available_at", keep="last")
    return pd.Series(table["label"].to_numpy(float),
                     index=pd.DatetimeIndex(table["available_at"]).astype("datetime64[ns]"),
                     name="quadrant")


# ------------------------------------------------------------------- the returns


@dataclass(frozen=True)
class SleevePanel:
    """Instrument excess returns, the within-sleeve schedule and the sleeve returns."""

    configuration: str
    sleeves: dict[str, tuple[str, ...]]
    sessions: pd.DatetimeIndex
    excess: pd.DataFrame
    #: The within-sleeve re-estimation dates (SPF stamps in the sample).
    dates: pd.DatetimeIndex
    within: pd.DataFrame
    sleeve_returns: pd.DataFrame
    usable: pd.DatetimeIndex


def sleeve_panel(configuration: str, stamped: pd.Series, *,
                 excess: pd.DataFrame | None = None) -> SleevePanel:
    """§12.4: returns, funding, the within-sleeve schedule and the sleeve returns.

    ``stamped`` supplies the re-estimation calendar only (its dates, not its labels).
    ``excess`` replaces the stored excess returns, for the planted simulations of the
    lock measurement; by default they are built from the stored panel.
    """
    sleeves = S.sleeve_map(configuration)
    if excess is None:
        prices = S.load_prices(sleeves)
        start = S.configuration_start(prices, sleeves)
        excess = S.excess_returns(S.price_returns(prices), S.cash_rate(prices.index)).loc[start:]
    sessions = pd.DatetimeIndex(excess.index)
    dates = pd.DatetimeIndex([d for d in stamped.index if sessions[0] <= d <= sessions[-1]])
    within = S.hold(S.within_sleeve_weights(excess, sleeves, list(dates)), sessions)
    sleeve_r = S.sleeve_returns(excess, within, sleeves)
    usable = pd.DatetimeIndex(sleeve_r.index[sleeve_r.notna().all(axis=1)])
    return SleevePanel(configuration, sleeves, sessions, excess, dates, within, sleeve_r,
                       usable)


def leg_builder(panel: SleevePanel, **kwargs: object) -> E.InstrumentLegBuilder:
    """The declared book of §12.8 on this panel (`evaluate.InstrumentLegBuilder`)."""
    return E.InstrumentLegBuilder(panel.excess, panel.within, panel.sleeves, **kwargs)


# ----------------------------------------------------------- the stamped object (C)


@dataclass(frozen=True)
class StampedObject:
    """The stamped quadrant object: level C and the stamped-label sensitivity."""

    panel: SleevePanel
    quarterly: pd.DataFrame
    stamped: pd.Series
    folds: tuple[Fold, ...]
    #: ``evaluate.expand_to_sessions(stamped, sessions, lag=1)``: the real path.
    real_path: pd.Series

    @property
    def sessions(self) -> pd.DatetimeIndex:
        return self.panel.sessions

    def test_sessions(self) -> pd.DatetimeIndex:
        """The pooled test sessions of the stamped folds (4,126 at the headline)."""
        s = self.sessions
        return pd.DatetimeIndex(np.concatenate([f.test(s).to_numpy() for f in self.folds]))


def declared_object(
    configuration: str = "headline",
    *,
    root: Path = ROOT,
    growth: str = "growth",
    extra_months: int = 0,
    inputs: QuadrantInputs | None = None,
    min_quarters: int = 4,
) -> StampedObject:
    """The stamped object, identical to ``build_declared`` at ``39ef608`` (§12.4, §12.14).

    Folds are `evaluate.stamp_folds` at ``min_quarters``, counted from the first
    session with every sleeve return. The real path is checked equal to
    `quadrant.daily_labels` at lag 1, and no in-sample stamp may fall on a non-session
    unless ``extra_months`` is set (§12.2: 8 of 91 +1-month stamps do).
    """
    inputs = inputs or load_quadrant_inputs(root)
    quarterly = quarterly_quadrant(inputs, growth=growth, extra_months=extra_months)
    stamped = stamped_series(quarterly)
    panel = sleeve_panel(configuration, stamped)
    s = panel.sessions
    real = E.expand_to_sessions(stamped, s, lag=1)
    daily = Q.daily_labels(quarterly, s, lag=1).astype(float)
    if not np.array_equal(daily.to_numpy(), real.to_numpy(), equal_nan=True):
        raise RuntimeError("expand_to_sessions and quadrant.daily_labels disagree")
    if extra_months == 0:
        inside = stamped.index[(stamped.index >= s[0]) & (stamped.index <= s[-1])]
        if not inside.isin(s).all():
            raise RuntimeError("an in-sample stamp falls on a non-session")
    folds = E.stamp_folds(stamped, s, usable=panel.usable, min_quarters=min_quarters)
    return StampedObject(panel, quarterly, stamped, folds, real)


# ------------------------------------------------------ contemporaneous partitions


def session_quarters(sessions: pd.DatetimeIndex) -> pd.PeriodIndex:
    """The calendar quarter of every session."""
    return pd.DatetimeIndex(sessions).to_period("Q")


def contemporaneous_path(labels: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """The label of the calendar quarter each session lies in; NaN where none (§12.2).

    ``labels`` is indexed by quarter (`pandas.PeriodIndex`, frequency Q).
    """
    quarters = session_quarters(sessions)
    values = labels.astype(float).reindex(quarters).to_numpy()
    return pd.Series(values, index=pd.DatetimeIndex(sessions), name=labels.name)


@dataclass(frozen=True)
class QuarterGrid:
    """Positions of each quarter's first and last session on a session index."""

    quarters: pd.PeriodIndex
    first: np.ndarray
    last: np.ndarray

    @classmethod
    def of(cls, quarters: pd.PeriodIndex, sessions: pd.DatetimeIndex) -> QuarterGrid:
        sessions = pd.DatetimeIndex(sessions)
        q = pd.PeriodIndex(quarters, freq="Q")
        first = sessions.searchsorted(q.start_time, side="left")
        last = sessions.searchsorted(q.end_time, side="right") - 1
        held = first <= last
        return cls(q[held], first[held], last[held])


def quarter_folds(
    labels: pd.Series,
    available: pd.Series,
    sessions: pd.DatetimeIndex,
    *,
    n_folds: int = 5,
    min_quarters: int = 4,
    usable: pd.DatetimeIndex | None = None,
    n_cells: int = E.N_CELLS,
) -> tuple[Fold, ...]:
    """§12.5: an expanding walk-forward cut at quarter ends, from label COUNTS only.

    - A quarter is *counted* if its first session is on or after the first ``usable``
      session (default: the first session): a quarter with no sleeve return cannot
      inform a covariance.
    - The first test quarter is the first quarter ``Q`` at which every cell holds at
      least ``min_quarters`` counted quarters whose label was public
      (``available <= `` the last session before ``Q``). The quarter just before ``Q``
      is usually not yet public, so it does not count.
    - The labelled quarters from ``Q`` on form ``n_folds`` contiguous test folds of
      equal numbers of quarters (`numpy.array_split`: the first folds take the
      remainder). The last fold ends on the last session of the last labelled quarter.
    - Fold ``k`` trains on every session before its test window (expanding).

    Raises:
        ValueError: if the rule is never met, or leaves fewer quarters than folds.
    """
    sessions = pd.DatetimeIndex(sessions)
    grid = QuarterGrid.of(labels.index, sessions)
    lab = labels.reindex(grid.quarters).to_numpy(float)
    avail = pd.DatetimeIndex(available.reindex(grid.quarters))
    if np.isnan(lab).any() or avail.isna().any():
        raise ValueError("every quarter needs a label and an availability date")
    codes = E._codes(lab, n_cells)
    start = 0 if usable is None or len(usable) == 0 else int(
        sessions.searchsorted(pd.DatetimeIndex(usable)[0], side="left"))
    counted = grid.first >= start
    k0 = None
    for i in range(1, len(grid.quarters)):
        cut = sessions[grid.first[i] - 1]
        ok = counted[:i] & (avail[:i] <= cut)
        tally = np.bincount(codes[:i][ok], minlength=n_cells)[:n_cells]
        if (tally >= min_quarters).all():
            k0 = i
            break
    if k0 is None:
        raise ValueError(f"no cell count reaches {min_quarters} public quarters in every cell")
    rest = np.arange(k0, len(grid.quarters))
    if rest.size < n_folds:
        raise ValueError("fewer test quarters than folds")
    folds = []
    for number, group in enumerate(np.array_split(rest, n_folds), start=1):
        a, b = int(grid.first[group[0]]), int(grid.last[group[-1]])
        folds.append(Fold(number, sessions[0], sessions[a - 1], sessions[a], sessions[b],
                          sessions[a - 1], sessions[b]))
    return tuple(folds)


def known_by(available: pd.Series, sessions: pd.DatetimeIndex, cut: pd.Timestamp) -> np.ndarray:
    """Per session: is the label of its calendar quarter public by ``cut`` (its close)?"""
    quarters = session_quarters(sessions)
    dates = pd.DatetimeIndex(available.reindex(quarters))
    return np.asarray(dates <= cut) & np.asarray(dates.notna())


def fold_masks(available: pd.Series, sessions: pd.DatetimeIndex,
               folds: Sequence[Fold]) -> dict[int, np.ndarray]:
    """Per fold: the sessions whose label its fit and its evaluation may read (§12.5).

    A training session is kept if its quarter's label was public by the close of the
    fold's last training session; every test session is kept (the evaluation groups
    test sessions after the fact). Label-free: it reads availability dates only.
    """
    sessions = pd.DatetimeIndex(sessions)
    out = {}
    for f in folds:
        train = np.asarray(sessions <= f.train_end)
        test = np.asarray((sessions >= f.test_start) & (sessions <= f.test_end))
        out[f.number] = (train & known_by(available, sessions, f.train_end)) | test
    return out


def fold_paths(
    labels: pd.Series | Mapping[int, pd.Series],
    available: pd.Series,
    sessions: pd.DatetimeIndex,
    folds: Sequence[Fold],
) -> dict[int, pd.Series]:
    """One session path per fold for `evaluate.LevelA.run` (§12.2, §12.5).

    ``labels`` is one quarterly series, or one per fold number (a witness whose bins
    use each fold's own cut-offs). Each fold's path carries the contemporaneous label
    on its test sessions and on the training sessions of :func:`fold_masks`, NaN
    elsewhere.
    """
    masks = fold_masks(available, sessions, folds)
    out = {}
    for f in folds:
        series = labels[f.number] if isinstance(labels, Mapping) else labels
        path = contemporaneous_path(series, sessions)
        out[f.number] = path.where(masks[f.number]).rename("label")
    return out


def quarter_blocks(quarters: pd.PeriodIndex, sessions: pd.DatetimeIndex,
                   folds: Sequence[Fold]) -> pd.Series:
    """The calendar block of each quarter, for P1 per block (§12.10).

    ``"0:train"`` for the quarters whose first session precedes the first test session,
    then ``"<fold>:test"``. Quarters beyond the last test session join the last fold.
    """
    sessions = pd.DatetimeIndex(sessions)
    q = pd.PeriodIndex(quarters, freq="Q")
    first = sessions.searchsorted(q.start_time, side="left")
    starts = np.array([sessions.get_loc(f.test_start) for f in folds])
    which = np.searchsorted(starts, first, side="right")
    names = ["0:train"] + [f"{f.number}:test" for f in folds]
    return pd.Series([names[i] for i in which], index=q, name="block")


def quarters_in_sample(labels: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """The labelled quarters holding at least one session, in order."""
    grid = QuarterGrid.of(labels.dropna().index, sessions)
    return labels.reindex(grid.quarters).astype(float)


# ------------------------------------------------------- the contemporaneous object


@dataclass(frozen=True)
class Partition:
    """A quarterly partition: its labels and the date from which each is public."""

    name: str
    labels: pd.Series
    available: pd.Series

    def path(self, sessions: pd.DatetimeIndex) -> pd.Series:
        return contemporaneous_path(self.labels, sessions)

    def through(self, last: pd.Period) -> Partition:
        """The partition cut after quarter ``last`` (the last test quarter of the folds).

        A quarter beyond the evaluation must not enter a P1 block, where a draw could
        move its label into the test window.
        """
        keep = self.labels.index <= last
        return Partition(self.name, self.labels[keep], self.available[keep])


def quarter_last_sessions(quarters: pd.PeriodIndex, sessions: pd.DatetimeIndex) -> pd.Series:
    """The last session of each quarter, indexed by quarter (quarters with no session dropped)."""
    grid = QuarterGrid.of(quarters, sessions)
    return pd.Series(pd.DatetimeIndex(sessions)[grid.last], index=grid.quarters,
                     name="available")


def last_test_quarter(folds: Sequence[Fold]) -> pd.Period:
    """The calendar quarter of the last test session."""
    return pd.Timestamp(folds[-1].test_end).to_period("Q")


def quadrant_partition(quarterly: pd.DataFrame, sessions: pd.DatetimeIndex, *,
                       name: str = "quadrant") -> Partition:
    """The SPF quadrant as a contemporaneous partition: public at its release stamp."""
    labels = quarters_in_sample(quarterly["label"].astype(float), sessions).rename(name)
    available = pd.Series(pd.DatetimeIndex(quarterly["available_at"]),
                          index=quarterly.index).reindex(labels.index)
    return Partition(name, labels, available)


def b1_partition(t10yie: pd.DataFrame, baa: pd.DataFrame, sessions: pd.DatetimeIndex,
                 *, window: int = B1_WINDOW, through: pd.Period | None = None) -> Partition:
    """B1 on the quarterly grid (§12.12): the sign pair at each quarter's last session.

    `quadrant.market_panel` on ``sessions`` (a value known on a session is the previous
    session's close), `quadrant.b1_change_labels` at lag 0 over ``window`` sessions,
    sampled on the last session of each calendar quarter and applied to that quarter.
    The label of a quarter is public at the close of its last session. Quarters with no
    defined value (before the first full window) are dropped, and so are quarters after
    ``through`` (the last test quarter: the sample's last quarter is incomplete).
    """
    t10 = t10yie.loc[t10yie["period"] <= T10YIE_CUTOFF]
    panel = Q.market_panel(t10, baa, sessions)
    daily = Q.b1_change_labels(panel, window, lag=0).astype(float)
    quarters = session_quarters(sessions)
    last = quarter_last_sessions(pd.period_range(quarters[0], quarters[-1], freq="Q"),
                                 sessions)
    labels = pd.Series(daily.reindex(pd.DatetimeIndex(last.to_numpy())).to_numpy(float),
                       index=last.index, name="b1").dropna()
    partition = Partition("b1", labels, last.reindex(labels.index))
    return partition if through is None else partition.through(through)


def traded_blind_driver(engine: E.LevelA) -> pd.Series:
    """W2's driver (§12.15): the blind weights held on each session times the sleeves.

    ``Σ_s w_s(t) r_s(t)`` with ``w(t)`` the blind leg's unscaled sleeve weights on the
    traded path (fold 1's before the first test session, fold f's on its test
    sessions), NaN unless all five sleeve returns are finite. State-blind.
    """
    path = engine._path(engine._blind)
    x = engine.returns.reindex(columns=path.columns)
    out = (path * x).sum(axis=1, min_count=path.shape[1])
    return out.where(x.notna().all(axis=1) & path.notna().all(axis=1)).rename("driver")


def quarterly_volatility_labels(
    driver: pd.Series,
    sessions: pd.DatetimeIndex,
    folds: Sequence[Fold],
    available: pd.Series,
    *,
    n_bins: int = WITNESS_BINS,
    min_sessions: int = WITNESS_MIN_SESSIONS,
) -> tuple[dict[int, pd.Series], pd.Series]:
    """The timing-matched witness W2 (§12.15): quartile of each quarter's realised vol.

    - Each quarter's value is ``log(√252 · sd)`` of ``driver`` over its own sessions
      (ddof 1), defined with at least ``min_sessions`` finite returns.
    - Fold f's cut-offs are the quantiles of that value over the quarters public by the
      close of its last training session (``available <= train_end``).
    - Returns one quarterly label series per fold, and the stitched series (each quarter
      under the cut-offs of the fold it is tested in; earlier quarters under fold 1's),
      which the witness's own P1 null draws.
    """
    sessions = pd.DatetimeIndex(sessions)
    quarters = session_quarters(sessions)
    x = driver.reindex(sessions)
    grouped = x.groupby(quarters)
    count = grouped.count()
    sd = grouped.std(ddof=1) * np.sqrt(E.PERIODS)
    value = np.log(sd.where((count >= min_sessions) & (sd > 0))).dropna()
    value.index = pd.PeriodIndex(value.index, freq="Q")
    avail = available.reindex(value.index)
    per_fold: dict[int, pd.Series] = {}
    for f in folds:
        reference = value[np.asarray(pd.DatetimeIndex(avail) <= f.train_end)]
        if reference.size < n_bins:
            raise ValueError(f"fold {f.number}: too few public quarters for {n_bins} bins")
        cuts = np.quantile(reference.to_numpy(float), np.arange(1, n_bins) / n_bins)
        bins = np.searchsorted(cuts, value.to_numpy(float), side="right").astype(float)
        per_fold[f.number] = pd.Series(bins, index=value.index, name="witness")
    blocks = quarter_blocks(value.index, sessions, folds)
    stitched = per_fold[folds[0].number].copy()
    for f in folds:
        mine = np.asarray(blocks == f"{f.number}:test")
        stitched[mine] = per_fold[f.number][mine]
    return per_fold, stitched


@dataclass(frozen=True)
class ContemporaneousObject:
    """The object of levels A, B1 and B2 (§12.2-§12.9)."""

    panel: SleevePanel
    quarterly: pd.DataFrame
    stamped: pd.Series
    partition: Partition
    folds: tuple[Fold, ...]

    @property
    def sessions(self) -> pd.DatetimeIndex:
        return self.panel.sessions

    def paths(self, partition: Partition | None = None) -> dict[int, pd.Series]:
        """The per-fold real paths of ``partition`` (default: the quadrant)."""
        p = partition or self.partition
        return fold_paths(p.labels, p.available, self.sessions, self.folds)

    def blocks(self, partition: Partition | None = None) -> pd.Series:
        p = partition or self.partition
        return quarter_blocks(p.labels.index, self.sessions, self.folds)

    def engine(self, cls: type[E.LevelA] = E.LevelA, *, builder_kwargs: Mapping | None = None,
               **kwargs: object) -> E.LevelA:
        """The level-A engine of §12.9 on this object (declared settings as defaults)."""
        settings = {"reading": "equal_variance", "about_zero": False, "blind_rule": "erc",
                    "extremes": "own", "occupancy_weighted": False,
                    "min_cell_sessions": E.MIN_CELL_SESSIONS,
                    "min_cell_episodes": E.MIN_CELL_EPISODES, "tie_break": E.TIE_BREAK}
        settings.update(kwargs)
        if "leg_builder" not in settings:
            settings["leg_builder"] = leg_builder(self.panel, **dict(builder_kwargs or {}))
        return cls(self.panel.sleeve_returns, self.folds, **settings)


def contemporaneous_object(
    configuration: str = "headline",
    *,
    root: Path = ROOT,
    growth: str = "growth",
    extra_months: int = 0,
    inputs: QuadrantInputs | None = None,
    min_quarters: int = 4,
    excess: pd.DataFrame | None = None,
    folds: Sequence[Fold] | None = None,
) -> ContemporaneousObject:
    """The contemporaneous object (§12.2, §12.5): quadrant by calendar quarter.

    ``growth`` and ``extra_months`` give the within-vintage and +1-month variants: the
    labels move with ``growth``, only the availability dates move with
    ``extra_months``. ``folds`` imposes a layout (the variants keep the primary's);
    otherwise :func:`quarter_folds` builds it at ``min_quarters``. ``excess`` replaces
    the stored returns (planted simulations only).
    """
    inputs = inputs or load_quadrant_inputs(root)
    quarterly = quarterly_quadrant(inputs, growth=growth, extra_months=extra_months)
    # The within-sleeve calendar is always the primary SPF stamps: label-free dates.
    stamped = stamped_series(quarterly_quadrant(inputs))
    panel = sleeve_panel(configuration, stamped, excess=excess)
    partition = quadrant_partition(quarterly, panel.sessions)
    if folds is None:
        folds = quarter_folds(partition.labels, partition.available, panel.sessions,
                              min_quarters=min_quarters, usable=panel.usable)
    return ContemporaneousObject(panel, quarterly, stamped, partition, tuple(folds))
