"""Daily portfolio panels from the Ken French data library.

These panels are the study's source of *non-volatility* features — cross-sectional
dispersion, average pairwise correlation, factor concentration — without which the
placebo test T1 is lost in advance: a regime model fed only return-derived
volatility statistics can, by construction, only reproduce a volatility quantile.

Point-in-time treatment, stated because it is a judgement call rather than a fact.
The library republishes monthly, so the last few weeks of a file were not
downloadable in real time. We nonetheless set ``available_at = period``, because
the quantity being measured — how dispersed today's cross-section is — is
observable from any live price feed on the day itself; the library is a clean
historical proxy for it, not the only way to obtain it. A live implementation
would compute the same statistic from its own cross-section with a one-session
lag. Treating these panels as delayed by a month would understate what a real
desk knows, which is the opposite of the error this project guards against.
"""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import requests

from regime_lab.data.pit import validate

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

#: ``{name: (zip file, series_id prefix)}``
PANELS = {
    "industry_49": ("49_Industry_Portfolios_daily_CSV.zip", "ind"),
    "size_bm_25": ("25_Portfolios_5x5_Daily_CSV.zip", "szbm"),
    "factors_5": ("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", "ff"),
}

#: Sentinels the library uses for missing observations.
MISSING = (-99.99, -999.0, -99.99e2)


def _first_table(text: str) -> pd.DataFrame:
    """Return the first dated table of a Ken French CSV.

    The files hold several tables back to back — value-weighted then
    equal-weighted returns, for instance — separated by blank lines and a title.
    Only the first is read; taking the whole file would silently stack two
    different weightings of the same portfolios into one series.
    """
    lines = text.splitlines()
    header_at = next(i for i, line in enumerate(lines) if line.startswith(","))
    body: list[str] = []
    for line in lines[header_at + 1 :]:
        stripped = line.strip()
        if not stripped:
            break
        if not stripped[:8].isdigit():
            break
        body.append(line)

    table = pd.read_csv(io.StringIO("\n".join([lines[header_at], *body])))
    table = table.rename(columns={table.columns[0]: "period"})
    table["period"] = pd.to_datetime(table["period"].astype(str), format="%Y%m%d")
    return table.set_index("period")


def fetch(panel: str, *, start: str = "1990-01-01", timeout: int = 120) -> pd.DataFrame:
    """Download one panel as a point-in-time frame of daily percent returns."""
    if panel not in PANELS:
        raise ValueError(f"unknown panel {panel!r}; expected one of {list(PANELS)}")
    filename, prefix = PANELS[panel]

    response = requests.get(f"{BASE}/{filename}", timeout=timeout)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        text = archive.read(name).decode("latin-1")

    table = _first_table(text).loc[start:]
    table.columns = [f"{prefix}_{c.strip().lower().replace(' ', '_')}" for c in table.columns]
    for sentinel in MISSING:
        table = table.mask(table == sentinel)

    long = (
        table.stack(future_stack=True)
        .rename("value")
        .reset_index()
        .rename(columns={"level_1": "series_id"})
    )
    long["available_at"] = long["period"]
    return validate(long)
