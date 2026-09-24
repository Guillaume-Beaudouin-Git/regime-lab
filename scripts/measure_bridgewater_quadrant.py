"""Label-only measurement of the Bridgewater quadrant, before the lock.

**Blind.** This script reads no return. It opens the SPF files, the release-date table,
two market series for B1, and the session calendar of the trend universe (its index
and the first valid date of TIP, nothing else). Every number it prints is a property
of a label sequence: counts, dates, transitions, spells, persistence, occupancy.

Sections:

0. the files, their SHA-256 against the manifest written by `scripts/fetch_spf.py`;
1. the surprise panel: the draft's 180 quarters and standard deviations, the
   denominator question, the level-revision contamination of the growth axis and the
   within-vintage candidate;
2. the draft's §0/§1 figures, reproduced with the draft's own stamp and no lag;
3. the same with the real release dates and the declared T-1 lag, the draft's
   one-month sensitivity, and what moved;
4. the within-vintage growth candidate, same statistics;
5. per-decade cell composition of the declared object (the draft's killer 6);
6. the draft's comparison objects: the level quadrant and the H3 AR-proxy quadrant;
7. the B1 market-implied candidates: clock, persistence, occupancy.

Usage:
    .venv/bin/python scripts/fetch_spf.py                 # once, writes data/raw/spf/
    .venv/bin/python scripts/measure_bridgewater_quadrant.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime_lab.config import CACHE, RAW, ROOT
from regime_lab.construction import quadrant as qd
from regime_lab.data.pit import build_panel, validate

SPF = RAW / "spf"
UNIVERSE = CACHE / "trend_universe.parquet"
ASSET_START = pd.Timestamp("2003-12-05")
H3_RAW = ROOT / "chantiers" / "macro-momentum" / "data" / "raw" / "h3_macro"
MACRO = RAW / "macro"


@dataclass
class Check:
    quantity: str
    disclosed: str
    measured: str
    match: bool


CHECKS: list[Check] = []


def check(quantity: str, disclosed: float | int | str, measured: float | int | str,
          fmt: str = "{}", tol: float = 0.0) -> None:
    """Record a disclosed figure against its re-measurement."""
    if isinstance(disclosed, str) or isinstance(measured, str):
        ok = str(disclosed) == str(measured)
        text_d, text_m = str(disclosed), str(measured)
    else:
        ok = abs(float(disclosed) - float(measured)) <= tol + 1e-12
        text_d, text_m = fmt.format(disclosed), fmt.format(measured)
    CHECKS.append(Check(quantity, text_d, text_m, ok))


def section(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def cells(counts: dict[int, int]) -> str:
    return " / ".join(f"{counts[k]:>4}" for k in qd.CELLS)


def shares(counts: dict[int, int], n: int, digits: int = 3) -> str:
    return " / ".join(f"{counts[k] / n:.{digits}f}" for k in qd.CELLS)


# ------------------------------------------------------------------------ data


def load() -> dict[str, object]:
    needed = [SPF / name for name, _ in qd.SPF_LEVEL_FILES.values()] + [
        SPF / qd.SPF_GROWTH_FILE[0], SPF / qd.RTDSM_ROUTPUT_FILE, SPF / qd.RELEASE_DATES_CSV,
        UNIVERSE,
    ]
    absent = [p.relative_to(ROOT) for p in needed if not p.exists()]
    if absent:
        sys.exit(f"missing inputs {absent}; run scripts/fetch_spf.py first")
    rgdp = qd.read_spf(SPF / "median_rgdp_level.xlsx", ("RGDP1", "RGDP2"))
    cpi = qd.read_spf(SPF / "median_cpi_level.xlsx", ("CPI1", "CPI2"))
    growth_file = qd.read_spf(SPF / qd.SPF_GROWTH_FILE[0], qd.SPF_GROWTH_FILE[1])
    first = qd.read_rtdsm_first_release((SPF / qd.RTDSM_ROUTPUT_FILE).read_bytes())
    release = qd.read_release_dates(SPF / qd.RELEASE_DATES_CSV)
    universe = pd.read_parquet(UNIVERSE)
    calendar = pd.DatetimeIndex(universe.index).astype("datetime64[ns]")
    tip_start = universe["TIP"].first_valid_index()
    del universe  # the calendar is all this script keeps
    return {
        "rgdp": rgdp, "cpi": cpi, "drgdp2": growth_file["DRGDP2"], "first": first,
        "release": release, "calendar": calendar, "tip_start": pd.Timestamp(tip_start),
    }


# -------------------------------------------------------------------- sections


def files_section() -> None:
    section("0. Files and SHA-256 (data/raw/spf/manifest.json)")
    manifest = json.loads((SPF / "manifest.json").read_text())
    for row in manifest["files"]:
        path = RAW.parent / row["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "absent"
        same = "ok" if digest == row["sha256"] else "CHANGED SINCE FETCH"
        draft = row.get("same_bytes_as_draft_2026_09_22")
        tag = "" if draft is None else ("  same bytes as the draft" if draft else "  != draft")
        print(f"  {row['path']:40} {row['bytes']:>8,} B  {digest[:16]}  {same}{tag}")
    neg = manifest["negative_control"]
    print(f"  negative control: {neg['url'].rsplit('/', 1)[-1]} -> {neg['bytes']:,} B, "
          f"refused={neg['refused']}")


def surprise_section(data: dict) -> pd.DataFrame:
    section("1. The surprise panel (§1.1)")
    panel = qd.surprise_panel(data["rgdp"], data["cpi"], first_release=data["first"],
                              drgdp2=data["drgdp2"])
    print(f"  usable quarters {len(panel)}  {panel.index[0]} -> {panel.index[-1]}")
    check("usable quarters", 180, len(panel))
    check("first usable quarter", "1981Q3", str(panel.index[0]))
    check("last usable quarter", "2026Q2", str(panel.index[-1]))
    sd_g, sd_i = panel["growth"].std(), panel["inflation"].std()
    print(f"  growth surprise sd {sd_g:.3f} annualised points, inflation sd {sd_i:.3f} points")
    check("growth surprise sd", 35.481, round(sd_g, 3), "{:.3f}")
    check("inflation surprise sd", 1.333, round(sd_i, 3), "{:.3f}")

    print("\n  Denominator. The draft script divides by RGDP1 of survey q (level of q-1),")
    print("  which is what the §1.1 formula says. Dividing by RGDP2_q instead:")
    same_sign = bool((np.sign(panel["growth"]) == np.sign(panel["growth_nowcast_denominator"]))
                     .all())
    print(f"    sd {panel['growth_nowcast_denominator'].std():.3f}; every sign identical: "
          f"{same_sign} -> no label depends on the denominator")

    print("\n  Level revisions inside the declared growth axis. The ten largest |growth|:")
    top = panel["growth"].abs().sort_values(ascending=False).head(10).index
    for q in top:
        wv = panel.loc[q, "growth_within_vintage"]
        print(f"    {q}  declared {panel.loc[q, 'growth']:+9.3f}   within-vintage "
              f"{wv:+7.3f}" if np.isfinite(wv) else
              f"    {q}  declared {panel.loc[q, 'growth']:+9.3f}   within-vintage   n/a")
    big = panel.index[panel["growth"].abs() > 10]
    print(f"  |declared growth| > 10 points: {len(big)} quarters {[str(q) for q in big]}")
    trimmed = panel["growth"].drop(big)
    print(f"  declared growth sd without them: {trimmed.std():.3f}  "
          f"(within-vintage sd {panel['growth_within_vintage'].std():.3f}, "
          f"n={panel['growth_within_vintage'].notna().sum()})")
    wv = panel.dropna(subset=["growth_within_vintage"])
    disagree = np.sign(wv["growth"]) != np.sign(wv["growth_within_vintage"])
    missing = [str(q) for q in panel.index[panel["growth_within_vintage"].isna()]]
    print(f"  growth sign, declared vs within-vintage: {int(disagree.sum())} of {len(wv)} "
          f"quarters disagree; within-vintage undefined for {missing}")
    asset_q = wv.index[wv.index >= pd.Period("2003Q3", freq="Q")]
    print(f"    of which reference quarters 2003Q3 onward: "
          f"{int(disagree.loc[asset_q].sum())} of {len(asset_q)}")
    print("  Flags: reference quarter 2025Q3 is revealed by survey 2025Q4, which had no")
    print("  published jump-off value for real GDP (shutdown); RGDP1 of that survey is the")
    print("  forecasters' own estimate, so its growth 'surprise' is not against an outturn.")
    print("  Survey 1990Q2 was not taken in real time (collected in August 1990), which")
    print("  touches reference quarters 1990Q1 and 1990Q2 (full history only).")
    return panel


def on_sample(quarterly: pd.DataFrame, calendar: pd.DatetimeIndex, lag: int) -> pd.Series:
    """Daily labels built on the whole calendar, then cut at the sample start, so the
    first sample session already carries the label known the session before."""
    return qd.daily_labels(quarterly, calendar, lag=lag).loc[ASSET_START:]


def permutation_same(labels: np.ndarray, draws: int = 20_000, seed: int = 20260924) -> float:
    """Share of label permutations whose P(same) is at or below the observed one.

    Label-only: the null keeps the multiset of quarterly labels and destroys the order.
    """
    rng = np.random.default_rng(seed)
    observed = float((labels[1:] == labels[:-1]).mean())
    perms = np.array([rng.permutation(labels) for _ in range(draws)])
    null = (perms[:, 1:] == perms[:, :-1]).mean(axis=1)
    return float((null <= observed + 1e-12).mean())


def describe(name: str, quarterly: pd.DataFrame, daily: pd.Series, start: pd.Timestamp) -> dict:
    """Quarter-grid and session-grid statistics of one variant on the asset sample."""
    inside = quarterly.loc[quarterly["available_at"] >= start]
    seq = pd.Series(inside["label"].to_numpy(), index=pd.DatetimeIndex(inside["available_at"]))
    stats = qd.clock(seq)
    day = qd.clock(daily)
    return {"name": name, "q": stats, "d": day, "sessions": int(daily.notna().sum()),
            "first_session": daily.first_valid_index(), "daily": daily}


def print_variant(v: dict) -> None:
    s, d = v["q"], v["d"]
    print(f"  {v['name']}")
    print(f"    quarters stamped on/after the start {s.n}, span {s.years:.2f} yr, transitions "
          f"{s.transitions} -> {s.per_year:.3f}/yr, spell median {s.spell_median:.1f} max "
          f"{s.spell_max}")
    print(f"    P(same as last quarter) {s.same_share:.4f} over {s.n - 1} pairs "
          f"({s.same_share_n:.4f} dividing by n, the draft); independent draws "
          f"{s.independent_same:.4f}")
    print(f"    quarters per cell {cells(s.counts)}  shares {shares(s.counts, s.n)}")
    print(f"    sessions labelled {v['sessions']} from {v['first_session']:%Y-%m-%d}; per cell "
          f"{cells(d.counts)}  shares {shares(d.counts, d.n)}")
    print(f"    session grid: {d.transitions} changes over {d.years:.2f} yr -> {d.per_year:.3f}/yr")


def draft_section(data: dict, panel: pd.DataFrame, calendar: pd.DatetimeIndex) -> dict:
    section("2. The draft's figures, with the draft's stamp (15th of month 2 of q+1), no lag")
    stamps = qd.availability(panel.index, rule="draft")
    quarterly = qd.quarterly_labels(panel, stamps)
    full = qd.clock(pd.Series(quarterly["label"].to_numpy(),
                              index=pd.DatetimeIndex(quarterly["available_at"])))
    print(f"  full SPF history: {full.n} quarters, {full.years:.2f} yr, {full.transitions} "
          f"transitions -> {full.per_year:.3f}/yr")
    check("full-history transitions/yr", 2.95, round(full.per_year, 2), "{:.2f}")
    check("full-history span (years)", 44.7, round(full.years, 1), "{:.1f}")
    print(f"  first stamp {quarterly['available_at'].iloc[0]:%Y-%m-%d} "
          "(the text says the 14th; the code adds 14 days to the 1st, the 15th)")

    daily0 = on_sample(quarterly, calendar, lag=0)
    v = describe("draft stamp, lag 0 (the draft)", quarterly, daily0, ASSET_START)
    print_variant(v)
    s, d = v["q"], v["d"]
    check("asset-sample sessions", 5938, v["sessions"])
    check("asset-sample quarters", 91, s.n)
    check("asset-sample span (years)", 22.5, round(s.years, 1), "{:.1f}")
    check("transitions (asset)", 70, s.transitions)
    check("transitions/yr (asset)", 3.11, round(s.per_year, 2), "{:.2f}")
    check("median spell (quarters)", 1.0, s.spell_median, "{:.1f}")
    check("max spell (quarters)", 3, s.spell_max)
    check("P(same) as disclosed", 0.220, round(s.same_share_n, 3), "{:.3f}")
    check("P(same) independent", 0.266, round(s.independent_same, 3), "{:.3f}")
    for k, disclosed in zip(qd.CELLS, (841, 1792, 1630, 1675), strict=True):
        check(f"sessions {qd.CELLS[k]}", disclosed, d.counts[k])
    for k, disclosed in zip(qd.CELLS, (14.2, 30.2, 27.5, 28.2), strict=True):
        check(f"session share % {qd.CELLS[k]}", disclosed,
              round(100 * d.counts[k] / d.n, 1), "{:.1f}")
    for k, disclosed in zip(qd.CELLS, (13, 28, 25, 25), strict=True):
        check(f"quarters {qd.CELLS[k]}", disclosed, s.counts[k])
    for k, disclosed in zip(qd.CELLS, (0.143, 0.308, 0.275, 0.275), strict=True):
        check(f"P1 occupancy {qd.CELLS[k]}", disclosed, round(s.counts[k] / s.n, 3), "{:.3f}")
    check("(g-,i-) years", 3.34, round(d.counts[0] / 252, 2), "{:.2f}")
    check("quarters per fold at 5 folds", 18.2, round(s.n / 5, 1), "{:.1f}")
    check("sessions per fold at 5 folds", 1187, v["sessions"] // 5)
    print(f"  NB the session sample also holds the label stamped before {ASSET_START:%Y-%m-%d} "
          f"({quarterly.loc[quarterly['available_at'] < ASSET_START].index[-1]}), which the "
          "quarter counts leave out")
    print(f"  TIP first valid date {data['tip_start']:%Y-%m-%d} (the sample start)")
    check("TIP start", "2003-12-05", f"{data['tip_start']:%Y-%m-%d}")
    return v


def release_section(data: dict, panel: pd.DataFrame, calendar: pd.DatetimeIndex,
                    draft: dict) -> dict:
    section("3. Real release dates (spf-release-dates.txt) and the declared T-1 lag")
    rel = qd.availability(panel.index, rule="release", release_dates=data["release"])
    dft = qd.availability(panel.index, rule="draft")
    src = rel["source"].value_counts().to_dict()
    known = rel["source"] == "release"
    print(f"  stamp sources over the {len(panel)} usable quarters: {src}")
    print(f"  surveys with a real date: {data['release'].index[0]} -> "
          f"{data['release'].index[-1]}; reference quarters with a real stamp: "
          f"{rel.index[known][0]} -> {rel.index[known][-1]}")
    lead = (rel["available_at"] - dft["available_at"]).dt.days
    lk = lead[known]
    print(f"  real release minus draft stamp, days: median {lk.median():.0f}, min {lk.min()}, "
          f"max {lk.max()}; draft stamp BEFORE the release (look-ahead) on {(lk > 0).sum()} "
          f"of {len(lk)} quarters")
    window = rel.index[(rel["available_at"] >= ASSET_START) & known]
    late = lead.loc[window]
    print(f"  in the asset window ({len(window)} stamps): look-ahead on {(late > 0).sum()}, "
          f"worst {late.max()} days; the list:")
    for q in late.index[late > 0]:
        print(f"    reference {q}  survey {q + 1}  draft {dft.loc[q, 'available_at']:%Y-%m-%d}  "
              f"release {rel.loc[q, 'available_at']:%Y-%m-%d}  (+{late.loc[q]} d)")

    qrel = qd.quarterly_labels(panel, rel)
    full = qd.clock(pd.Series(qrel["label"].to_numpy(),
                              index=pd.DatetimeIndex(qrel["available_at"])))
    print(f"\n  full history with real dates (fallback = end of survey quarter before 1990Q2): "
          f"{full.n} quarters, {full.years:.2f} yr, {full.transitions} transitions -> "
          f"{full.per_year:.3f}/yr, P(same) {full.same_share:.4f} vs independent "
          f"{full.independent_same:.4f}")
    variants = {
        "release, lag 0": on_sample(qrel, calendar, lag=0),
        "release, lag 1 (DECLARED OBJECT)": on_sample(qrel, calendar, lag=1),
        "release, lag 2 (release time of day unknown)": on_sample(qrel, calendar, lag=2),
        "draft, lag 1": on_sample(qd.quarterly_labels(panel, dft), calendar, lag=1),
    }
    out = {}
    for name, daily in variants.items():
        stamps = rel if name.startswith("release") else dft
        v = describe(name, qd.quarterly_labels(panel, stamps), daily, ASSET_START)
        print_variant(v)
        out[name] = v
    for months in (1,):
        for rule in ("release", "draft"):
            st = qd.availability(panel.index, rule=rule, release_dates=data["release"],
                                 extra_months=months)
            ql = qd.quarterly_labels(panel, st)
            name = f"{rule} + {months} month, lag 1 (sensitivity)"
            v = describe(name, ql, on_sample(ql, calendar, lag=1), ASSET_START)
            print_variant(v)
            out[name] = v

    inside = qrel.loc[qrel["available_at"] >= ASSET_START, "label"].to_numpy(dtype="int64")
    p_low = permutation_same(inside)
    print(f"\n  persistence against order-destroying permutations of the {len(inside)} quarterly "
          f"labels: P(null P(same) <= observed) = {p_low:.3f} -> the quadrant is "
          f"{'anti-persistent at 5%' if p_low < 0.05 else 'indistinguishable from independent'}"
          )
    base = draft["daily"]
    print("\n  sessions whose label differs from the draft's (draft stamp, lag 0):")
    for name, v in out.items():
        diff = (v["daily"].astype("float") != base.astype("float")) & base.notna() \
            & v["daily"].notna()
        print(f"    {name:44} {int(diff.sum()):>5} of {int(base.notna().sum())} sessions")
    declared = out["release, lag 1 (DECLARED OBJECT)"]
    for k in qd.CELLS:
        delta = declared["d"].counts[k] - draft["d"].counts[k]
        print(f"    declared vs draft, {qd.CELLS[k]}: {declared['d'].counts[k]} sessions "
              f"({delta:+d})")
    return out


def within_vintage_section(data: dict, panel: pd.DataFrame, calendar: pd.DatetimeIndex,
                           declared: dict) -> None:
    section("4. Within-vintage growth candidate (first release - DRGDP2), real dates, lag 1")
    sub = panel.dropna(subset=["growth_within_vintage"])
    rel = qd.availability(sub.index, rule="release", release_dates=data["release"])
    ql = qd.quarterly_labels(sub, rel, growth="growth_within_vintage")
    full = qd.clock(pd.Series(ql["label"].to_numpy(), index=pd.DatetimeIndex(ql["available_at"])))
    print(f"  full history: {full.n} quarters, {full.transitions} transitions -> "
          f"{full.per_year:.3f}/yr, P(same) {full.same_share:.4f} vs {full.independent_same:.4f}")
    v = describe("within-vintage growth, release, lag 1", ql,
                 on_sample(ql, calendar, lag=1), ASSET_START)
    print_variant(v)
    a, b = v["daily"], declared["daily"]
    both = a.notna() & b.notna()
    print(f"  sessions labelled differently from the declared object: "
          f"{int((a[both].astype(int) != b[both].astype(int)).sum())} of {int(both.sum())}")
    print("  (the missing 2025Q3/2025Q4 quarters extend the previous label; that is a choice")
    print("   the lock must make if this candidate is adopted)")


def decade_section(declared: dict, panel: pd.DataFrame, data: dict) -> None:
    section("5. Cell composition by decade, declared object (killer 6), labels only")
    daily = declared["daily"]
    table = qd.composition(daily, qd.decade(daily.index))
    table.columns = [qd.CELLS[k] for k in table.columns]
    print("  sessions:")
    print("    " + table.to_string().replace("\n", "\n    "))
    rel = qd.availability(panel.index, rule="release", release_dates=data["release"])
    ql = qd.quarterly_labels(panel, rel)
    ql = ql.loc[ql["available_at"] >= ASSET_START]
    tq = qd.composition(pd.Series(ql["label"].to_numpy()),
                        qd.decade(pd.DatetimeIndex(ql["available_at"])))
    tq.columns = [qd.CELLS[k] for k in tq.columns]
    print("  quarters (by stamp date):")
    print("    " + tq.to_string().replace("\n", "\n    "))
    labelled = daily.dropna()
    fifths = np.minimum(np.arange(len(labelled)) * 5 // len(labelled), 4)
    tf = qd.composition(labelled, fifths)
    tf.index = [f"fifth {i + 1} {labelled.index[fifths == i][0]:%Y-%m}->"
                f"{labelled.index[fifths == i][-1]:%Y-%m}" for i in tf.index]
    tf.columns = [qd.CELLS[k] for k in tf.columns]
    print("  sessions by equal fifths of the sample (a provisional fold cut, not the lock's):")
    print("    " + tf.to_string().replace("\n", "\n    "))
    g0 = daily.eq(0).fillna(False)
    years0 = pd.Series(np.asarray(daily.index.year)[g0.to_numpy()]).value_counts().sort_index()
    print(f"  (growth-, inflation-) sessions by year: {years0.to_dict()}")


def level_quadrant_section() -> None:
    section("6a. Level quadrant (yoy vs expanding median), macro PIT store, quarterly grid")
    names = ["macro_indpro", "macro_payems", "macro_cpi", "macro_unrate"]
    pit = validate(pd.concat([pd.read_parquet(MACRO / f"{n}.parquet") for n in names],
                             ignore_index=True))
    grid = pd.bdate_range("1990-01-01", "2026-09-10")
    panel = build_panel(pit, grid)
    g_ip = panel["macro_indpro"].pct_change(252)
    g_pay = panel["macro_payems"].pct_change(252)
    comp_g = (g_ip / g_ip.expanding(756).std() + g_pay / g_pay.expanding(756).std()) / 2
    comp_i = panel["macro_cpi"].pct_change(252)
    gs = comp_g - comp_g.expanding(756).median()
    isr = comp_i - comp_i.expanding(756).median()
    lab = (2 * (gs > 0).astype(int) + (isr > 0).astype(int))
    lab = lab.loc[lab.index >= pd.Timestamp("1996-01-02")].dropna()
    q = lab.resample("QE").last().dropna()
    ch = int((q.diff().fillna(0) != 0).sum())
    same = float((q == q.shift()).mean())
    print(f"  {len(q)} quarters, {ch} transitions -> {ch / (len(q) / 4):.2f}/yr "
          f"(quarters/4, the draft's denominator); P(same) {same:.3f} (dividing by n)")
    check("level quadrant transitions/yr (quarterly)", 1.20, round(ch / (len(q) / 4), 2),
          "{:.2f}")


def h3_section() -> None:
    section("6b. H3 AR-proxy quadrant (committed chantiers/macro-momentum code), 63-day kernel")
    if not H3_RAW.exists():
        print("  h3_macro data absent; skipped")
        return
    from macro_momentum.surprise import decayed_score, first_releases, model_surprise

    transforms = {
        "payems": "growth", "unrate": "diff", "cpi": "growth", "core_cpi": "growth",
        "indpro": "growth", "houst": "growth", "permit": "growth", "dgorder": "growth",
        "retail": "growth", "sentiment": "diff", "capacity": "diff", "hours": "diff",
        "claims": "growth",
    }
    growth = {"payems": 1, "unrate": -1, "indpro": 1, "houst": 1, "permit": 1, "dgorder": 1,
              "retail": 1, "sentiment": 1, "capacity": 1, "hours": 1, "claims": -1}
    inflation = {"cpi": 1, "core_cpi": 1}
    events = {}
    for name, kind in transforms.items():
        rel = first_releases(pd.read_parquet(H3_RAW / f"{name}.parquet"))
        events[name] = model_surprise(rel, kind=kind).dropna(subset=["z"])
    grid = pd.bdate_range("1998-01-01", "2026-09-10")

    def axis(members: dict[str, int], hl: float) -> pd.Series:
        total = pd.Series(0.0, index=grid)
        for name, sign in members.items():
            ev = events[name].copy()
            ev["z"] = ev["z"] * sign
            total = total + decayed_score(ev, grid, half_life=hl)
        return total / len(members)

    for label, members in (("with claims", growth),
                           ("without claims", {k: v for k, v in growth.items()
                                               if k != "claims"})):
        g, i = axis(members, 63.0), axis(inflation, 63.0)
        lab = 2 * (g > 0).astype(int) + (i > 0).astype(int)
        start = max(max(events[n]["released_at"].min() for n in members),
                    max(events[n]["released_at"].min() for n in inflation))
        lab = lab.loc[lab.index >= start]
        ch = int((lab.diff().fillna(0) != 0).sum())
        q = lab.resample("QE").last().dropna()
        chq = int((q.diff().fillna(0) != 0).sum())
        print(f"  {label:15} start {start:%Y-%m-%d}: daily {ch} changes over "
              f"{len(lab) / 252:.1f} yr -> {ch / (len(lab) / 252):.2f}/yr; quarterly grid "
              f"{chq} over {len(q)} quarters -> {chq / (len(q) / 4):.2f}/yr")
        if label == "without claims":
            check("H3 AR-proxy quarterly grid /yr (no claims)", 2.27,
                  round(chq / (len(q) / 4), 2), "{:.2f}")
            check("H3 AR-proxy daily grid /yr (no claims)", 7.08,
                  round(ch / (len(lab) / 252), 2), "{:.2f}")


def b1_section(calendar: pd.DatetimeIndex) -> None:
    section("7. B1 candidates: T10YIE (inflation) and -BAA spread (growth), labels only")
    t10 = validate(pd.read_parquet(SPF / qd.T10YIE_FILE))
    baa = validate(pd.read_parquet(MACRO / "fin_baa_spread.parquet"))
    panel = qd.market_panel(t10, baa, calendar)
    print(f"  T10YIE {t10['period'].min():%Y-%m-%d} -> {t10['period'].max():%Y-%m-%d} "
          f"({len(t10):,} rows); BAA {baa['period'].min():%Y-%m-%d} -> "
          f"{baa['period'].max():%Y-%m-%d}")
    candidates = {f"change over {w} sessions": qd.b1_change_labels(panel, w)
                  for w in (21, 63, 126, 252)}
    candidates["level vs expanding median (min 252)"] = qd.b1_level_labels(panel)
    print(f"  {'candidate':38} {'first':>10} {'n':>5} {'/yr d':>6} {'spell d':>7} "
          f"{'/yr q':>6} {'P(same)q':>8} {'indep':>6}  occupancy (sessions)")
    for name, lab in candidates.items():
        lab = lab.loc[lab.index >= ASSET_START]
        first = lab.first_valid_index()
        lab = lab.loc[first:]
        d = qd.clock(lab)
        lengths = qd.run_lengths(lab.to_numpy(dtype="int64"))
        qs = qd.clock(qd.quarter_end_sample(lab))
        print(f"  {name:38} {first:%Y-%m-%d} {d.n:5d} {d.per_year:6.2f} "
              f"{np.median(lengths):7.0f} {qs.per_year:6.2f} {qs.same_share:8.3f} "
              f"{qs.independent_same:6.3f}  {shares(d.counts, d.n)}")
    for w in (21, 63, 126, 252):
        g = panel["baa_spread"] - panel["baa_spread"].shift(w)
        i = panel["breakeven"] - panel["breakeven"].shift(w)
        mask = panel.index >= ASSET_START
        print(f"  ties (unchanged value, counted negative) at {w:>3}: growth "
              f"{int((g[mask] == 0).sum())}, inflation {int((i[mask] == 0).sum())} sessions")
    print("  '/yr d' session-grid transitions per calendar year; 'spell d' median spell in")
    print("  sessions; '/yr q' and 'P(same)q' on the label at each calendar quarter's last")
    print("  session; 'indep' = sum of squared quarterly occupancies.")


def main() -> None:
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    data = load()
    calendar = data["calendar"]
    files_section()
    panel = surprise_section(data)
    draft = draft_section(data, panel, calendar)
    out = release_section(data, panel, calendar, draft)
    declared = out["release, lag 1 (DECLARED OBJECT)"]
    within_vintage_section(data, panel, calendar, declared)
    decade_section(declared, panel, data)
    level_quadrant_section()
    h3_section()
    b1_section(calendar)

    section("Disclosed figures against the re-measurement")
    width = max(len(c.quantity) for c in CHECKS)
    for c in CHECKS:
        print(f"  {c.quantity:{width}}  disclosed {c.disclosed:>10}  measured {c.measured:>10}"
              f"  {'ok' if c.match else 'DIFFERS'}")
    print(f"\n  {sum(c.match for c in CHECKS)} of {len(CHECKS)} reproduce")


if __name__ == "__main__":
    main()
