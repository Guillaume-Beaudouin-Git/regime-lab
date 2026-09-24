"""Growth x inflation quadrant from the Philadelphia Fed SPF consensus, point in time.

The partition of `docs/PRESPEC_BRIDGEWATER.md` §1.1. For survey quarter ``q`` the
Survey of Professional Forecasters publishes a nowcast of quarter ``q`` (column 2 of
each median file) and, one survey later, the value of quarter ``q`` as it was then
known (column 1 of survey ``q+1``). The gap between the two is a consensus surprise.

Declared construction (§1.1), reproduced exactly::

    growth surprise_q    = (RGDP1_{q+1} - RGDP2_q) / RGDP1_q * 400
    inflation surprise_q =  CPI1_{q+1}  - CPI2_q
    label_q              =  2 * 1[growth > 0] + 1[inflation > 0]

``RGDP1_q`` is the level of quarter ``q-1`` as known at survey ``q``: the draft
script divides by that column, which is what the formula says. The denominator is a
positive level, so it never changes the sign and therefore never changes a label; it
only scales the growth axis.

**What the declared growth axis also carries.** ``RGDP1_{q+1} - RGDP2_q`` compares two
*levels* from two data vintages. Between survey ``q`` and survey ``q+1`` the level of
``q-1`` itself is revised (second and third estimates, the July annual revision) and,
at every comprehensive revision, rebased to a new chain-dollar year. The level gap is
therefore growth news plus the revision of the base, and at a rebasing it is the
rebasing. Measured: ten quarters exceed 10 annualised points (1985Q4 alone is +452.7);
nine of them have an ordinary within-vintage growth surprise (-1.2 to +3.0; one is
undefined), the tenth is 2020Q3 (+14.2 against +14.0, a genuine surprise). Without the
ten, the standard deviation falls from the draft's 35.481 to 2.351. Their dates (1985Q4,
1991Q4, 1995Q4, 1999Q3, 2003Q4, 2009Q2, 2013Q2, 2018Q2, 2023Q3) match the BEA
comprehensive revisions as recalled, which was not checked against a BEA document; the
level jump itself is measured. :func:`growth_surprise_within_vintage`
compares growth *rates*, each within its own vintage: the first-release growth of ``q``
(Philadelphia Fed real-time data set) minus the SPF nowcast growth ``DRGDP2_q``. It is
offered as a candidate for the lock, not adopted here.

**Availability.** A surprise for ``q`` is public when survey ``q+1`` is released. The
Philadelphia Fed publishes the true deadline and news release date of every survey
from 1990Q2 (`spf-release-dates.txt`); earlier dates are not known. Three stamp rules:

``release``
    The news release date of survey ``q+1``. Where it is unknown (surveys before
    1990Q2) the stamp falls back to the last calendar day of the survey quarter, later
    than every real-time release on record (the latest, 2019Q1, came 80 days into its
    quarter). The source of every stamp is kept.
``draft``
    The draft's approximation, ``start(q) + 4 months + 14 days``: the **15th** of the
    second month of ``q+1`` (the draft's text says the 14th; its code gives the 15th).
    It precedes the real release on 71 of the 146 dated surveys, by up to 35 days in
    the real-time era, which is a look-ahead.
``extra_months``
    Either rule plus whole months, the draft's declared one-month sensitivity.

A label becomes known on its stamp date and is traded from the next session: the
release is taken as known at the close of its release day (the Philadelphia Fed
releases in the morning; not verified survey by survey), and the signal at T-1 is
traded at T.

Nothing in this module reads a return.
"""

from __future__ import annotations

import io
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from regime_lab.data.pit import build_panel

SPF_DATA_URL = (
    "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/"
    "survey-of-professional-forecasters/data-files/files"
)
SPF_RELEASE_DATES_URL = (
    "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/"
    "survey-of-professional-forecasters/spf-release-dates.txt"
)
RTDSM_ROUTPUT_URL = (
    "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/"
    "real-time-data/data-files/xlsx/routput_first_second_third.xlsx"
)

#: The four median files the draft opened, and the columns each must carry.
SPF_LEVEL_FILES: dict[str, tuple[str, tuple[str, ...]]] = {
    "rgdp": ("median_rgdp_level.xlsx", ("RGDP1", "RGDP2")),
    "cpi": ("median_cpi_level.xlsx", ("CPI1", "CPI2")),
    "unemp": ("median_unemp_level.xlsx", ("UNEMP1", "UNEMP2")),
    "indprod": ("median_indprod_level.xlsx", ("INDPROD1", "INDPROD2")),
}
#: Auxiliary SPF file for the within-vintage growth candidate: DRGDP2 is the nowcast
#: growth of the survey quarter, computed by the Philadelphia Fed within the survey.
SPF_GROWTH_FILE: tuple[str, tuple[str, ...]] = ("median_rgdp_growth.xlsx", ("DRGDP2",))
RTDSM_ROUTPUT_FILE = "routput_first_second_third.xlsx"
RELEASE_DATES_TEXT = "spf_release_dates.txt"
RELEASE_DATES_CSV = "spf_release_dates.csv"
T10YIE_FILE = "fred_t10yie.parquet"

#: First line of the genuine release-date table.
RELEASE_DATES_HEADER = "Deadline and Release Dates for the Survey of Professional Forecasters"

#: Cell codes, ``2 * 1[growth > 0] + 1[inflation > 0]``.
CELLS: dict[int, str] = {
    0: "(growth-, inflation-)",
    1: "(growth-, inflation+)",
    2: "(growth+, inflation-)",
    3: "(growth+, inflation+)",
}

ZIP_MAGIC = b"PK\x03\x04"

StampRule = Literal["release", "draft"]

#: A B1 market value older than this is dropped rather than carried forward.
MAX_STALENESS = pd.Timedelta(days=10)


class SpfFileError(ValueError):
    """A downloaded or stored file is not the SPF / real-time file it claims to be."""


# ------------------------------------------------------------------ file checks


def _html_title(content: bytes) -> str:
    match = re.search(rb"<title>(.*?)</title>", content[:65536], flags=re.I | re.S)
    return match.group(1).decode("latin-1").strip() if match else "no title"


def require_zip(content: bytes, label: str) -> None:
    """Refuse anything that is not a ZIP container (an xlsx is one).

    The SPF server answers a wrong file name with HTTP 200 and an HTML page titled
    ``Error - 404`` (18,401 bytes on 2026-09-22), so the status code proves nothing.
    """
    head = content[:1024].lstrip().lower()
    if head.startswith((b"<!doctype", b"<html")) or b"<html" in head:
        raise SpfFileError(
            f"{label}: an HTML page ({len(content):,} bytes, title {_html_title(content)!r}), "
            "not a workbook"
        )
    if not content.startswith(ZIP_MAGIC):
        raise SpfFileError(f"{label}: no ZIP signature, not an xlsx ({len(content):,} bytes)")


def verify_workbook(content: bytes, *, required: Sequence[str], label: str) -> pd.DataFrame:
    """Open an SPF median workbook from its bytes and check it is what it claims.

    Returns:
        The first sheet as read by pandas.

    Raises:
        SpfFileError: on HTML, a missing ZIP signature, a file pandas cannot open, a
            missing ``YEAR``/``QUARTER`` or required column, or a non-contiguous survey
            calendar.
    """
    require_zip(content, label)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # unparsable print header
            frame = pd.read_excel(io.BytesIO(content), sheet_name=0)
    except Exception as exc:  # noqa: BLE001 - any reader failure means "not the file"
        raise SpfFileError(f"{label}: pandas cannot open it ({exc})") from exc
    missing = [c for c in ("YEAR", "QUARTER", *required) if c not in frame.columns]
    if missing:
        raise SpfFileError(f"{label}: missing columns {missing}; has {list(frame.columns)}")
    spf_table(frame, label=label)
    return frame


def spf_table(frame: pd.DataFrame, *, label: str = "spf") -> pd.DataFrame:
    """Index an SPF median sheet by survey quarter and check the calendar has no gap."""
    years = pd.to_numeric(frame["YEAR"], errors="coerce")
    quarters = pd.to_numeric(frame["QUARTER"], errors="coerce")
    if years.isna().any() or quarters.isna().any():
        raise SpfFileError(f"{label}: non-numeric YEAR or QUARTER")
    index = pd.PeriodIndex(
        [f"{int(y)}Q{int(q)}" for y, q in zip(years, quarters, strict=True)], freq="Q",
        name="survey",
    )
    _require_contiguous(index, label)
    out = frame.drop(columns=["YEAR", "QUARTER"]).apply(pd.to_numeric, errors="coerce")
    out.index = index
    return out


def _require_contiguous(index: pd.PeriodIndex, label: str) -> None:
    if index.has_duplicates:
        raise SpfFileError(f"{label}: duplicated quarters")
    if len(index) > 1:
        expected = pd.period_range(index[0], periods=len(index), freq="Q")
        if not index.equals(expected):
            raise SpfFileError(f"{label}: quarters are not consecutive from {index[0]}")


def read_spf(path: Path, required: Sequence[str]) -> pd.DataFrame:
    """Read a stored SPF median workbook, re-verifying it, indexed by survey quarter."""
    frame = verify_workbook(Path(path).read_bytes(), required=required, label=Path(path).name)
    return spf_table(frame, label=Path(path).name)


def read_rtdsm_first_release(content: bytes, *, label: str = RTDSM_ROUTPUT_FILE) -> pd.Series:
    """First-release real output growth by quarter, from the real-time data set file.

    The ``DATA`` sheet of ``routput_first_second_third.xlsx`` carries a header row
    ``Date | First | Second | Third | Most_Recent`` and quarters written ``YYYY:Qn``;
    values are annualised quarter-on-quarter growth in percentage points.

    Raises:
        SpfFileError: if the bytes are not that workbook.
    """
    require_zip(content, label)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            raw = pd.read_excel(io.BytesIO(content), sheet_name="DATA", header=None)
    except Exception as exc:  # noqa: BLE001
        raise SpfFileError(f"{label}: no readable DATA sheet ({exc})") from exc
    header = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq("Date")]
    if len(header) != 1:
        raise SpfFileError(f"{label}: no single 'Date' header row")
    body = raw.iloc[header[0] + 1 :].copy()
    body.columns = [str(c).strip() for c in raw.iloc[header[0]]]
    if "First" not in body.columns:
        raise SpfFileError(f"{label}: no 'First' column; has {list(body.columns)}")
    body = body.dropna(subset=["Date"])
    dates = body["Date"].astype(str).str.strip()
    if not dates.str.fullmatch(r"\d{4}:Q[1-4]").all():
        raise SpfFileError(f"{label}: dates are not YYYY:Qn")
    index = pd.PeriodIndex(dates.str.replace(":", "", regex=False), freq="Q", name="quarter")
    _require_contiguous(index, label)
    return pd.Series(
        pd.to_numeric(body["First"], errors="coerce").to_numpy(dtype=float),
        index=index, name="first_release",
    )


# ----------------------------------------------------------- release-date table

_ROW = re.compile(
    r"^\s*(?:(?P<year>\d{4})\s+)?Q(?P<q>[1-4])\s+"
    r"(?P<deadline>\d{1,2}/\d{1,2}/\d{2})(?P<n1>\**)\s+"
    r"(?P<release>\d{1,2}/\d{1,2}/\d{2})(?P<n2>\**)\s*$"
)


def _date_near(text: str, survey_year: int) -> pd.Timestamp:
    month, day, yy = (int(x) for x in text.split("/"))
    for year in (survey_year, survey_year + 1):
        if year % 100 == yy:
            return pd.Timestamp(year=year, month=month, day=day)
    raise SpfFileError(f"date {text!r} is not in survey year {survey_year} or the next")


def parse_release_dates(text: str) -> pd.DataFrame:
    """Parse the Philadelphia Fed's ``spf-release-dates.txt``.

    Returns:
        One row per survey, indexed by survey quarter, with ``deadline`` (the true
        deadline for returns), ``release`` (the news release date) and ``note`` (the
        asterisks of the source: ``*`` 1990Q2 not taken in real time, ``**`` delayed
        by a government shutdown, ``***`` 2025Q4 without most jump-off values).

    Raises:
        SpfFileError: on an HTML page, a missing title line, a gap in the survey
            calendar, or a date that is inconsistent with its survey.
    """
    head = text[:1024].lstrip().lower()
    if head.startswith(("<!doctype", "<html")) or "<html" in head:
        raise SpfFileError("release dates: an HTML page, not the text table")
    lines = text.replace("\r", "").split("\n")
    first = next((line.strip() for line in lines if line.strip()), "")
    if first != RELEASE_DATES_HEADER:
        raise SpfFileError(f"release dates: unexpected first line {first!r}")

    rows: list[dict[str, object]] = []
    year: int | None = None
    for line in lines:
        match = _ROW.match(line.expandtabs())
        if match is None:
            continue
        if match["year"]:
            year = int(match["year"])
        if year is None:
            raise SpfFileError(f"release dates: quarter before any year: {line!r}")
        rows.append({
            "survey": pd.Period(f"{year}Q{match['q']}", freq="Q"),
            "deadline": _date_near(match["deadline"], year),
            "release": _date_near(match["release"], year),
            "note": max(match["n1"], match["n2"], key=len),
        })
    if not rows:
        raise SpfFileError("release dates: no survey row parsed")
    table = pd.DataFrame(rows).set_index("survey")
    table.index = pd.PeriodIndex(table.index, freq="Q", name="survey")
    _require_contiguous(table.index, "release dates")

    lag = (table["release"] - table["deadline"]).dt.days
    if (lag < 0).any() or (lag > 21).any():
        raise SpfFileError("release dates: a release precedes its deadline or trails it by >21d")
    into = (table["release"] - table.index.start_time).dt.days
    if (into < 0).any() or (into > 200).any():
        raise SpfFileError("release dates: a release falls outside its survey's window")
    return table


def release_dates_csv(table: pd.DataFrame, source_url: str) -> pd.DataFrame:
    """The parsed table as stored: survey as ``YYYYQn`` text, ISO dates, the source URL."""
    return pd.DataFrame({
        "survey": table.index.astype(str),
        "deadline": table["deadline"].dt.strftime("%Y-%m-%d").to_numpy(),
        "release": table["release"].dt.strftime("%Y-%m-%d").to_numpy(),
        "note": table["note"].to_numpy(),
        "source_url": source_url,
    })


def read_release_dates(path: Path) -> pd.DataFrame:
    """Read the stored release-date CSV back into the parsed form."""
    raw = pd.read_csv(path, dtype={"note": str}, keep_default_na=False)
    table = pd.DataFrame({
        "deadline": pd.to_datetime(raw["deadline"]).to_numpy(),
        "release": pd.to_datetime(raw["release"]).to_numpy(),
        "note": raw["note"].to_numpy(),
    }, index=pd.PeriodIndex(raw["survey"], freq="Q", name="survey"))
    _require_contiguous(table.index, Path(path).name)
    return table


# -------------------------------------------------------------------- surprises


def growth_surprise(
    rgdp: pd.DataFrame, *, denominator: Literal["base", "nowcast"] = "base"
) -> pd.Series:
    """Declared growth surprise of each reference quarter ``q``, annualised points.

    ``(RGDP1_{q+1} - RGDP2_q) / D * 400`` with ``D = RGDP1_q`` (``"base"``, the level
    of ``q-1`` at survey ``q``: the declared formula and the draft script) or
    ``D = RGDP2_q`` (``"nowcast"``). ``D`` is positive, so the sign, and every label,
    is the same under both.
    """
    quarters = rgdp.index
    actual = rgdp["RGDP1"].reindex(quarters + 1).to_numpy(dtype=float)
    nowcast = rgdp["RGDP2"].to_numpy(dtype=float)
    base = rgdp["RGDP1" if denominator == "base" else "RGDP2"].to_numpy(dtype=float)
    out = pd.Series((actual - nowcast) / base * 400.0, index=quarters, name="growth")
    out.index.name = "quarter"
    return out


def inflation_surprise(cpi: pd.DataFrame) -> pd.Series:
    """Declared inflation surprise ``CPI1_{q+1} - CPI2_q``, percentage points annualised."""
    quarters = cpi.index
    actual = cpi["CPI1"].reindex(quarters + 1).to_numpy(dtype=float)
    out = pd.Series(actual - cpi["CPI2"].to_numpy(dtype=float), index=quarters, name="inflation")
    out.index.name = "quarter"
    return out


def growth_surprise_within_vintage(first_release: pd.Series, drgdp2: pd.Series) -> pd.Series:
    """Candidate growth surprise, growth rate against growth rate, each within its vintage.

    ``first-release growth of q - DRGDP2_q``. The first release (the advance estimate,
    published about a month after ``q`` ends) precedes survey ``q+1``, so this surprise
    is public no later than the declared one and shares its stamp. It carries no base
    revision and no rebasing.
    """
    out = first_release.reindex(drgdp2.index) - drgdp2.astype(float)
    out.name = "growth_within_vintage"
    out.index.name = "quarter"
    return out


def surprise_panel(
    rgdp: pd.DataFrame,
    cpi: pd.DataFrame,
    *,
    first_release: pd.Series | None = None,
    drgdp2: pd.Series | None = None,
) -> pd.DataFrame:
    """Every reference quarter with both declared surprises defined.

    Columns ``growth`` and ``inflation`` (declared), ``growth_nowcast_denominator``
    (same sign, other scale) and, when the real-time inputs are given,
    ``growth_within_vintage`` (may be missing where the first release is).
    """
    panel = pd.DataFrame({
        "growth": growth_surprise(rgdp),
        "inflation": inflation_surprise(cpi).reindex(rgdp.index),
        "growth_nowcast_denominator": growth_surprise(rgdp, denominator="nowcast"),
    })
    if first_release is not None and drgdp2 is not None:
        panel["growth_within_vintage"] = growth_surprise_within_vintage(
            first_release, drgdp2
        ).reindex(panel.index)
    panel = panel.dropna(subset=["growth", "inflation"])
    panel.index.name = "quarter"
    return panel


# ---------------------------------------------------------------- availability


def availability(
    quarters: pd.PeriodIndex,
    *,
    rule: StampRule,
    release_dates: pd.DataFrame | None = None,
    extra_months: int = 0,
) -> pd.DataFrame:
    """Date on which the surprise of each reference quarter became public.

    Returns:
        ``available_at`` (a calendar date) and ``source`` (``release``,
        ``fallback_quarter_end`` or ``draft``), indexed by reference quarter.
    """
    quarters = pd.PeriodIndex(quarters, freq="Q")
    survey = quarters + 1
    if rule == "release":
        if release_dates is None:
            raise ValueError("rule='release' needs the release-date table")
        stamps = pd.Series(release_dates["release"].reindex(survey).to_numpy(), index=quarters)
        known = stamps.notna()
        fallback = pd.Series(survey.end_time.normalize(), index=quarters)
        stamps = stamps.where(known, fallback)
        source = np.where(known.to_numpy(), "release", "fallback_quarter_end")
    elif rule == "draft":
        stamps = pd.Series(quarters.start_time + pd.DateOffset(months=4, days=14), index=quarters)
        source = np.full(len(quarters), "draft")
    else:
        raise ValueError(f"unknown stamp rule {rule!r}")
    if extra_months:
        stamps = stamps + pd.DateOffset(months=extra_months)
    out = pd.DataFrame({
        "available_at": pd.to_datetime(stamps).astype("datetime64[ns]").to_numpy(),
        "source": source,
    }, index=quarters)
    out.index.name = "quarter"
    return out


# ---------------------------------------------------------------------- labels


def quadrant_code(growth: pd.Series, inflation: pd.Series) -> pd.Series:
    """``2 * 1[growth > 0] + 1[inflation > 0]``; missing where either axis is missing.

    A value of exactly zero counts as negative, as the declared rule reads.
    """
    code = 2 * (growth > 0).astype("int64") + (inflation > 0).astype("int64")
    return code.astype("Int64").mask(growth.isna() | inflation.isna())


def quarterly_labels(
    panel: pd.DataFrame, stamps: pd.DataFrame, *, growth: str = "growth"
) -> pd.DataFrame:
    """One row per reference quarter with a defined label: label, stamp, stamp source.

    Raises:
        ValueError: if a later quarter is stamped before an earlier one, which would
            let an old label overwrite a newer one on the daily grid.
    """
    label = quadrant_code(panel[growth], panel["inflation"])
    out = pd.DataFrame({"label": label}).join(stamps, how="left")
    out = out.dropna(subset=["label"]).sort_index()
    if out["available_at"].isna().any():
        raise ValueError("a labelled quarter has no availability stamp")
    if (out["available_at"].diff().dt.days < 0).any():
        raise ValueError("availability stamps decrease with the reference quarter")
    return out


def daily_labels(
    quarterly: pd.DataFrame, sessions: pd.DatetimeIndex, *, lag: int = 1
) -> pd.Series:
    """Point-in-time label on each session, held from its stamp to the next stamp.

    A label is known on every session on or after its ``available_at`` date. The
    returned series is that known label shifted ``lag`` sessions later (``lag=1``:
    known at the close of T-1, traded at T). Sessions before the first stamp, and the
    first ``lag`` sessions, are missing. When two quarters share a stamp (1990Q1 and
    1990Q2, both released on 1990-08-31) the later quarter is the one held.
    """
    if lag < 0:
        raise ValueError("lag must be non-negative")
    sessions = pd.DatetimeIndex(sessions).astype("datetime64[ns]")
    table = quarterly.reset_index().sort_values(["available_at", quarterly.index.name or "index"])
    table = table.drop_duplicates("available_at", keep="last")
    stamps = table["available_at"].to_numpy(dtype="datetime64[ns]")
    labels = table["label"].to_numpy(dtype="int64")
    pos = np.searchsorted(stamps, sessions.to_numpy(), side="right") - 1
    known = pd.array(np.where(pos >= 0, labels[np.clip(pos, 0, None)], 0), dtype="Int64")
    known[pos < 0] = pd.NA
    return pd.Series(known, index=sessions, name="quadrant").shift(lag)


# ------------------------------------------------------------ label statistics


def run_lengths(values: np.ndarray) -> np.ndarray:
    """Lengths of the maximal runs of equal consecutive values."""
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    return np.diff(np.r_[starts, values.size]).astype(np.int64)


@dataclass(frozen=True)
class ClockStats:
    """Label-only statistics of a partition observed on a time index."""

    n: int
    counts: dict[int, int] = field(repr=False)
    transitions: int
    years: float
    per_year: float
    spell_median: float
    spell_max: int
    same_share: float
    same_share_n: float
    independent_same: float

    @property
    def occupancy(self) -> dict[int, float]:
        """Share of observations in each cell."""
        return {k: v / self.n for k, v in self.counts.items()}


def clock(labels: pd.Series) -> ClockStats:
    """Clock, occupancy, spells and persistence of a label sequence.

    ``labels`` is indexed by dates (stamps for a quarterly sequence, sessions for a
    daily one). Missing values are allowed only at the head, where they are dropped.
    ``years`` is the calendar span from the first to the last observation;
    ``same_share`` is P(same as previous) over the n-1 consecutive pairs;
    ``same_share_n`` is the draft's version, which divides by n; ``independent_same``
    is the sum of squared occupancies, P(same) under independent draws.
    """
    series = labels.copy()
    valid = series.notna().to_numpy()
    if not valid.any():
        raise ValueError("no labelled observation")
    first = int(np.argmax(valid))
    series = series.iloc[first:]
    if series.isna().any():
        raise ValueError("missing labels after the first labelled observation")
    values = series.to_numpy(dtype="int64")
    index = pd.DatetimeIndex(series.index)
    n = values.size
    same = values[1:] == values[:-1]
    counts = {k: int((values == k).sum()) for k in CELLS}
    years = float((index.max() - index.min()).days / 365.25)
    lengths = run_lengths(values)
    transitions = int((~same).sum())
    return ClockStats(
        n=n,
        counts=counts,
        transitions=transitions,
        years=years,
        per_year=transitions / years if years > 0 else float("nan"),
        spell_median=float(np.median(lengths)),
        spell_max=int(lengths.max()),
        same_share=float(same.mean()) if n > 1 else float("nan"),
        same_share_n=float(same.sum() / n),
        independent_same=float(sum((c / n) ** 2 for c in counts.values())),
    )


def composition(labels: pd.Series, groups: pd.Series | np.ndarray) -> pd.DataFrame:
    """Observations per cell within each group (e.g. a decade), missing labels dropped."""
    frame = pd.DataFrame({"label": labels.to_numpy(), "group": np.asarray(groups)})
    frame = frame.dropna(subset=["label"])
    table = pd.crosstab(frame["group"], frame["label"].astype("int64"))
    return table.reindex(columns=list(CELLS), fill_value=0)


def decade(index: pd.DatetimeIndex) -> np.ndarray:
    """Decade of each date, as ``"2000s"``."""
    return np.array([f"{y // 10 * 10}s" for y in pd.DatetimeIndex(index).year])


def quarter_end_sample(labels: pd.Series) -> pd.Series:
    """The label on the last session of each calendar quarter, indexed by that session."""
    index = pd.DatetimeIndex(labels.index)
    last = pd.Series(np.arange(len(index)), index=index).groupby(index.to_period("Q")).max()
    return labels.iloc[last.to_numpy()]


# -------------------------------------------------------- B1: market-implied axis


def market_panel(
    breakeven: pd.DataFrame,
    baa_spread: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    *,
    max_staleness: pd.Timedelta = MAX_STALENESS,
) -> pd.DataFrame:
    """The two B1 series as known on each session (point-in-time frames in, one each).

    Columns ``breakeven`` (FRED ``T10YIE``) and ``baa_spread`` (``fin_baa_spread``).
    Both are stored with ``available_at = period + 1 day``, so the value held on a
    session is the previous session's close. A value older than ``max_staleness`` is
    dropped rather than carried.
    """
    columns = {}
    for name, frame in (("breakeven", breakeven), ("baa_spread", baa_spread)):
        if frame["series_id"].nunique() != 1:
            raise ValueError(f"{name}: expected one series, got {frame['series_id'].unique()}")
        panel = build_panel(frame, sessions, max_staleness=max_staleness)
        columns[name] = panel.iloc[:, 0]
    return pd.DataFrame(columns, index=pd.DatetimeIndex(sessions).astype("datetime64[ns]"))


def b1_change_labels(panel: pd.DataFrame, window: int, *, lag: int = 1) -> pd.Series:
    """B1 candidate: signs of the change over ``window`` sessions, lagged ``lag`` sessions.

    growth+ when the BAA spread has **fallen** (credit stress easing); inflation+ when
    the ten-year breakeven has risen. An unchanged value counts as negative.
    """
    growth = -(panel["baa_spread"] - panel["baa_spread"].shift(window))
    inflation = panel["breakeven"] - panel["breakeven"].shift(window)
    return quadrant_code(growth, inflation).shift(lag).rename(f"b1_change_{window}")


def b1_level_labels(panel: pd.DataFrame, *, min_periods: int = 252, lag: int = 1) -> pd.Series:
    """B1 candidate: each series against its own expanding median, lagged ``lag`` sessions.

    growth+ when the BAA spread is below its expanding median; inflation+ when the
    breakeven is above its expanding median. The median includes the current session,
    which is known at that session.
    """
    baa = panel["baa_spread"]
    be = panel["breakeven"]
    growth = -(baa - baa.expanding(min_periods=min_periods).median())
    inflation = be - be.expanding(min_periods=min_periods).median()
    return quadrant_code(growth, inflation).shift(lag).rename("b1_level")
