"""Download and verify the external files of the Bridgewater study into ``data/raw/spf/``.

Everything is free, public and needs no key. Every file is verified to be what it
claims before it is written, because the Philadelphia Fed server answers a wrong file
name with **HTTP 200 and an HTML page titled ``Error - 404``** (18,401 bytes for two
wrong SPF names on 2026-09-22; 18,386 bytes for a wrong real-time name on 2026-09-24).
A 200 is not a file.

1. **SPF median files** (`docs/PRESPEC_BRIDGEWATER.md` §1.1), from the lowercase path
   ``.../survey-of-professional-forecasters/data-files/files/``:
   ``median_rgdp_level``, ``median_cpi_level``, ``median_unemp_level``,
   ``median_indprod_level``; plus ``median_rgdp_growth`` (``DRGDP2``, the nowcast growth
   of the survey quarter), used only by the within-vintage growth candidate. Checks:
   ZIP signature, opens with pandas, ``YEAR``/``QUARTER`` and the named columns
   present, consecutive survey quarters from 1968Q4.
2. **Real-time data set**, ``routput_first_second_third.xlsx``: first-release real
   output growth, the other half of the within-vintage candidate. Checks: ZIP
   signature, a ``DATA`` sheet with a ``Date``/``First`` header, consecutive quarters.
3. **SPF release dates**, ``spf-release-dates.txt``: the true deadline and news release
   date of every survey from 1990Q2 (earlier dates are not known to the Philadelphia
   Fed). Kept verbatim and parsed to ``spf_release_dates.csv`` with its source URL.
   Checks: plain text, the exact title line, consecutive surveys, every release on or
   after its deadline and within 21 days of it.
4. **FRED ``T10YIE``**, the ten-year breakeven of level B1, daily from 2003-01-02, through
   the programme's lag path (``regime_lab.data.sources.fred.fetch_current``, stamped
   ``available_at = period + 1 day``, coverage-checked), as a point-in-time parquet.

A negative control requests ``median_cpi_growth.xlsx``, a name that does not exist,
and requires the verifier to refuse what comes back.

``data/raw/spf/manifest.json`` records, for every file, the URL, the size, the SHA-256,
the retrieval time and what was checked; the SPF files are also compared with the
hashes of the draft's download of 2026-09-22 (a new survey row every quarter changes
them, which is reported, not refused).

Usage:
    .venv/bin/python scripts/fetch_spf.py
    .venv/bin/python scripts/fetch_spf.py --skip-fred
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

from regime_lab.config import RAW
from regime_lab.construction import quadrant as qd
from regime_lab.data.sources import fred

TARGET = RAW / "spf"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; regime-lab)"}

#: SHA-256 of the files the draft opened on 2026-09-22 (survey 2026Q3 the latest row).
DRAFT_SHA256 = {
    "median_rgdp_level.xlsx": "312a047c8a98871c3beba42492a547afe2e796fbb2bbd5a4fb3d5f0b90f1d1c4",
    "median_cpi_level.xlsx": "da5c7b637545a8d9fdd0a410a7eb83ee3cabc567544b1ad47a407c6dbda19b9a",
    "median_unemp_level.xlsx": "491e09e1a98d11e2b00a850c26035f58263041d0e86b5e911c37c7fdb86af80b",
    "median_indprod_level.xlsx": "63f61dfae1dbbf1b478ca84480c88c35ac0de15eefbdeff3d736fe8de2db70ea",
}
NEGATIVE_CONTROL = "median_cpi_growth.xlsx"


def download(url: str) -> tuple[bytes, str]:
    """GET with retries; returns the body and its declared content type."""
    for attempt in range(4):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=120)
        except requests.RequestException:
            time.sleep(2.0 * (attempt + 1))
            continue
        if resp.status_code == 200:
            return resp.content, resp.headers.get("Content-Type", "")
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"not reachable: {url}")


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def write(name: str, content: bytes) -> Path:
    path = TARGET / name
    path.write_bytes(content)
    if sha256(path.read_bytes()) != sha256(content):
        raise RuntimeError(f"{name}: the bytes on disk differ from the bytes downloaded")
    return path


def entry(name: str, url: str, content: bytes, content_type: str, checks: str) -> dict:
    return {
        "path": f"raw/spf/{name}",
        "url": url,
        "bytes": len(content),
        "sha256": sha256(content),
        "content_type": content_type,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "checks": checks,
    }


def fetch_workbook(name: str, required: tuple[str, ...]) -> dict:
    url = f"{qd.SPF_DATA_URL}/{name}"
    content, ctype = download(url)
    frame = qd.verify_workbook(content, required=required, label=name)
    table = qd.spf_table(frame, label=name)
    write(name, content)
    row = entry(name, url, content, ctype,
                f"zip signature; opens with pandas; columns YEAR QUARTER {' '.join(required)}; "
                f"{len(table)} consecutive surveys {table.index[0]}-{table.index[-1]}")
    draft = DRAFT_SHA256.get(name)
    if draft is not None:
        row["same_bytes_as_draft_2026_09_22"] = row["sha256"] == draft
    print(f"{name:34} {len(content):>7,} B  {len(table)} surveys "
          f"{table.index[0]}-{table.index[-1]}  sha256 {row['sha256'][:16]}"
          + ("" if draft is None else
             f"  {'= draft' if row['same_bytes_as_draft_2026_09_22'] else '!= draft'}"))
    return row


def fetch_rtdsm() -> dict:
    url = qd.RTDSM_ROUTPUT_URL
    content, ctype = download(url)
    first = qd.read_rtdsm_first_release(content)
    write(qd.RTDSM_ROUTPUT_FILE, content)
    missing = [str(p) for p in first.index[first.isna()] if p.year >= 1981]
    print(f"{qd.RTDSM_ROUTPUT_FILE:34} {len(content):>7,} B  {len(first)} quarters "
          f"{first.index[0]}-{first.index[-1]}  first release missing since 1981: {missing}")
    return entry(qd.RTDSM_ROUTPUT_FILE, url, content, ctype,
                 f"zip signature; DATA sheet with Date/First header; {len(first)} consecutive "
                 f"quarters {first.index[0]}-{first.index[-1]}; first release missing for "
                 f"{missing}")


def fetch_release_dates() -> list[dict]:
    url = qd.SPF_RELEASE_DATES_URL
    content, ctype = download(url)
    if "html" in ctype.lower():
        raise qd.SpfFileError(f"release dates: served as {ctype!r}")
    text = content.decode("ascii")
    table = qd.parse_release_dates(text)
    write(qd.RELEASE_DATES_TEXT, content)
    csv = qd.release_dates_csv(table, url).to_csv(index=False).encode()
    write(qd.RELEASE_DATES_CSV, csv)
    notes = table.loc[table["note"] != "", "note"]
    print(f"{qd.RELEASE_DATES_TEXT:34} {len(content):>7,} B  {len(table)} surveys "
          f"{table.index[0]}-{table.index[-1]}  flagged: "
          + ", ".join(f"{p}{n}" for p, n in notes.items()))
    checks = (f"plain text; exact title line; {len(table)} consecutive surveys "
              f"{table.index[0]}-{table.index[-1]}; release 0-21 days after deadline")
    return [
        entry(qd.RELEASE_DATES_TEXT, url, content, ctype, checks),
        entry(qd.RELEASE_DATES_CSV, url, csv, "text/csv",
              "parsed from spf_release_dates.txt by quadrant.parse_release_dates"),
    ]


def fetch_t10yie() -> dict:
    frame = fred.fetch_current("T10YIE", frequency="daily", start="2003-01-01")
    path = TARGET / qd.T10YIE_FILE
    frame.to_parquet(path, index=False)
    content = path.read_bytes()
    print(f"{qd.T10YIE_FILE:34} {len(content):>7,} B  {len(frame):,} rows "
          f"{frame['period'].min():%Y-%m-%d} -> {frame['period'].max():%Y-%m-%d}")
    return entry(qd.T10YIE_FILE, f"{fred.FRED_CSV}?id=T10YIE&cosd=2003-01-01", content,
                 "application/x-parquet (written from FRED CSV)",
                 f"FRED lag path, available_at = period + 1 day, coverage-checked; "
                 f"{len(frame)} rows {frame['period'].min():%Y-%m-%d} to "
                 f"{frame['period'].max():%Y-%m-%d}")


def negative_control() -> dict:
    url = f"{qd.SPF_DATA_URL}/{NEGATIVE_CONTROL}"
    content, ctype = download(url)
    try:
        qd.verify_workbook(content, required=("CPI1", "CPI2"), label=NEGATIVE_CONTROL)
    except qd.SpfFileError as exc:
        print(f"negative control {NEGATIVE_CONTROL}: HTTP 200, {len(content):,} B, refused "
              f"({exc})")
        return {"url": url, "bytes": len(content), "sha256": sha256(content),
                "content_type": ctype, "refused": True, "reason": str(exc)}
    raise RuntimeError(f"negative control accepted: {url} is now a real workbook?")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-fred", action="store_true", help="do not fetch T10YIE")
    args = parser.parse_args()
    TARGET.mkdir(parents=True, exist_ok=True)

    files = [fetch_workbook(name, required) for name, required in qd.SPF_LEVEL_FILES.values()]
    files.append(fetch_workbook(*qd.SPF_GROWTH_FILE))
    files.append(fetch_rtdsm())
    files.extend(fetch_release_dates())
    if not args.skip_fred:
        files.append(fetch_t10yie())
    elif (TARGET / qd.T10YIE_FILE).exists():
        content = (TARGET / qd.T10YIE_FILE).read_bytes()
        files.append(entry(qd.T10YIE_FILE, "", content, "application/x-parquet",
                           "not refetched (--skip-fred); hash of the file on disk"))
    manifest = {
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "script": "scripts/fetch_spf.py",
        "files": files,
        "negative_control": negative_control(),
    }
    (TARGET / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: data/raw/spf/manifest.json ({len(files)} files)")


if __name__ == "__main__":
    main()
