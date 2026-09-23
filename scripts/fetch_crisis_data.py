"""Download the public series of the crisis-object study (`docs/PRESPEC_CRISE.md`).

Three sources, all free and public, nothing private:

1. **CBOE strategy indices**, daily closes (`cdn.cboe.com/api/global/us_indices/
   daily_prices/<SYMBOL>_History.csv`): BXM (S&P 500 BuyWrite, daily from
   2002-03-22) and PUT (S&P 500 PutWrite, daily from 2007-01-03; a handful of
   scattered earlier prints are dropped).
2. **CBOE Futures Exchange VIX futures**, daily settlement of every monthly contract
   from May 2004. Contracts that expire up to 2014 are read from the CFE archive
   (`CFE_<month code><yy>_VX.csv`) when it has them, those from 2013 on from the
   per-contract files named after their final settlement date
   (`VX/VX_<yyyy-mm-dd>.csv`); the archive wins where both exist, because the
   per-contract files carry a zero settle before 2013-05-20. Before 2007-03-26 the
   contract was quoted at ten times the index (the old VBI contract); those settles
   are divided by ten so that one series has one scale. Settles below 5 are
   dropped: zeros (listed, not traded) and a few 1.00 placeholders on listing days
   (2008-04-21), where the VIX itself has never closed below 9.
3. **Ken French**, the daily momentum factor UMD (`F-F_Momentum_Factor_daily_CSV
   .zip`), full history from 1926-11.

Everything lands in `data/raw/crisis/` (gitignored) through `regime_lab.data.store`,
with a manifest carrying the SHA-256, the row count and the period range. Market
data are stamped `available_at = period` (a close is known at the close).

Usage:
    .venv/bin/python scripts/fetch_crisis_data.py        # about 3 minutes
"""

from __future__ import annotations

import io
import time
import zipfile
from datetime import date, timedelta

import pandas as pd
import requests

from regime_lab.data import store

HEADERS = {"User-Agent": "Mozilla/5.0 (research; regime-lab)"}
CBOE_INDEX = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"
CFE_ARCHIVE = "https://cdn.cboe.com/resources/futures/archive/volume-and-price/CFE_{code}_VX.csv"
CFE_CONTRACT = "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_{day}.csv"
FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
MONTH_CODES = "FGHJKMNQUVXZ"
RESCALE_BEFORE = pd.Timestamp("2007-03-26")
FIRST_CONTRACT = (2004, 5)
ARCHIVE_UNTIL = 2014
CONTRACT_FILES_FROM = 2013
MIN_SETTLE = 5.0
DAILY_FROM = {"BXM": "2002-03-22", "PUT": "2007-01-03"}


def get(url: str) -> requests.Response | None:
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=60)
        except requests.RequestException:
            time.sleep(1.0 + attempt)
            continue
        if resp.status_code == 200:
            return resp
        if resp.status_code in (403, 404):
            return None
        time.sleep(1.0 + attempt)
    return None


def cboe_indices() -> pd.DataFrame:
    frames = []
    for symbol, start in DAILY_FROM.items():
        resp = get(CBOE_INDEX.format(symbol=symbol))
        if resp is None:
            raise RuntimeError(f"CBOE index {symbol} not reachable")
        table = pd.read_csv(io.StringIO(resp.text))
        table["period"] = pd.to_datetime(table["DATE"], format="%m/%d/%Y")
        table = table[table["period"] >= start]
        frames.append(pd.DataFrame({
            "series_id": f"cboe_{symbol.lower()}", "period": table["period"],
            "available_at": table["period"], "value": table[symbol].astype(float),
        }))
    return pd.concat(frames, ignore_index=True)


def third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    offset = (4 - first.weekday()) % 7
    return first + timedelta(days=offset + 14)


def rule_expiry(year: int, month: int) -> date:
    """The Wednesday 30 days before the third Friday of the following month."""
    ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
    return third_friday(ny, nm) - timedelta(days=30)


def parse_cfe(text: str) -> pd.DataFrame:
    """Trade date and settle of a CFE file. Some archive rows carry a trailing field,
    so the rows are split by hand and only the first and seventh fields are kept."""
    lines = text.splitlines()
    head = next(i for i, line in enumerate(lines) if line.startswith("Trade Date"))
    columns = lines[head].split(",")
    at = columns.index("Settle")
    rows = [line.split(",") for line in lines[head + 1:] if line.strip()]
    table = pd.DataFrame({"period": [r[0] for r in rows], "settle": [r[at] for r in rows]})
    table["period"] = pd.to_datetime(table["period"], format="mixed")
    table["settle"] = pd.to_numeric(table["settle"], errors="coerce")
    return table


def contract(year: int, month: int) -> tuple[pd.DataFrame, date] | None:
    """Daily settles of one monthly contract and its final settlement date.

    The per-contract files carry a zero settle on every trade date before
    2013-05-20, so a contract is read from the archive whenever the archive has it,
    and from the per-contract file otherwise; on a date both give, the archive
    settle wins, and a zero is never read as a price.
    """
    expected = rule_expiry(year, month)
    archive, recent, expiry = None, None, None
    if year <= ARCHIVE_UNTIL:
        resp = get(CFE_ARCHIVE.format(code=f"{MONTH_CODES[month - 1]}{year % 100:02d}"))
        if resp is not None:
            archive = parse_cfe(resp.text)
            expiry = archive["period"].max().date()
    if year >= CONTRACT_FILES_FROM:
        for back in range(4):
            day = expected - timedelta(days=back)
            resp = get(CFE_CONTRACT.format(day=day.isoformat()))
            if resp is not None:
                recent, expiry = parse_cfe(resp.text), day
                break
    if archive is None and recent is None:
        return None
    parts = [t[t["settle"] > 0] for t in (archive, recent) if t is not None]
    table = pd.concat(parts).drop_duplicates("period", keep="first").sort_values("period")
    table = table[table["period"] <= pd.Timestamp(expiry)]
    return table, expiry


def vix_futures() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, checks = [], []
    year, month = FIRST_CONTRACT
    today = date.today()
    while (year, month) <= (today.year + 1, today.month):
        got = contract(year, month)
        if got is not None:
            table, expiry = got
            table = table.copy()
            scaled = table["period"] < RESCALE_BEFORE
            table.loc[scaled, "settle"] = table.loc[scaled, "settle"] / 10.0
            table = table[table["settle"] >= MIN_SETTLE]
            if len(table):
                frames.append(pd.DataFrame({
                    "series_id": f"vx_{expiry.isoformat()}", "period": table["period"],
                    "available_at": table["period"], "value": table["settle"],
                }))
                checks.append({"contract": f"{year}-{month:02d}", "expiry": expiry,
                               "rule": rule_expiry(year, month), "rows": len(table),
                               "first": table["period"].min().date()})
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return pd.concat(frames, ignore_index=True), pd.DataFrame(checks)


def french_umd() -> pd.DataFrame:
    resp = get(FRENCH)
    if resp is None:
        raise RuntimeError("Ken French momentum file not reachable")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        lines = z.read(name).decode("latin-1").splitlines()
    head = next(i for i, line in enumerate(lines) if line.strip().startswith(","))
    body = []
    for line in lines[head + 1:]:
        if not line.strip()[:8].isdigit():
            break
        body.append(line)
    table = pd.read_csv(io.StringIO("\n".join([lines[head], *body])))
    table.columns = ["period", "umd"]
    table["period"] = pd.to_datetime(table["period"].astype(str), format="%Y%m%d")
    table = table[table["umd"] > -99.0]
    return pd.DataFrame({"series_id": "ff_umd", "period": table["period"],
                         "available_at": table["period"], "value": table["umd"] / 100.0})


def main() -> None:
    indices = cboe_indices()
    store.write(indices, "crisis", "cboe_indices", origin="CBOE, daily index history")
    for sid, one in indices.groupby("series_id"):
        print(f"cboe      {sid:<10} {len(one):>6,} rows  {one['period'].min():%Y-%m-%d} "
              f"-> {one['period'].max():%Y-%m-%d}")

    futures, checks = vix_futures()
    store.write(futures, "crisis", "vix_futures", origin="CBOE Futures Exchange, VX settles")
    off_rule = checks[checks["expiry"] != checks["rule"]]
    print(f"vix fut   {checks.shape[0]} monthly contracts, {len(futures):,} rows, "
          f"{futures['period'].min():%Y-%m-%d} -> {futures['period'].max():%Y-%m-%d}")
    print(f"          {len(off_rule)} final settlement dates off the Wednesday rule "
          "(holidays):")
    for _, row in off_rule.iterrows():
        print(f"            {row['contract']}  {row['expiry']}  (rule {row['rule']})")

    umd = french_umd()
    store.write(umd, "crisis", "french_umd", origin="Ken French, F-F_Momentum_Factor_daily")
    print(f"french    ff_umd     {len(umd):>6,} rows  {umd['period'].min():%Y-%m-%d} "
          f"-> {umd['period'].max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
