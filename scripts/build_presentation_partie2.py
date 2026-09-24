# ruff: noqa: E501  (slide copy is prose; wrapping it would hurt more than help)
"""Build part 2 of the deck: docs/presentation/presentation_partie2.{html,pdf}.

The regime filter applied to three trading strategies chosen by Guillaume on
2026-09-24 for their opposite profiles: equity momentum (the filter helps), the
month-end Treasury rebound (the filter is irrelevant) and crypto trend (the filter
hurts), and the six ways of using the filter that were tested on them.

Where every number comes from
    - tables and verdicts: transcribed from the single readings of each study, copied
      verbatim in docs/artifacts/partie2/lectures_trois_strategies.txt, from
      RESULTS_B1_COUPLAGE, RESULTS_CRISE, RESULTS_LONGHIST and RESULTS_BUDGET_RISQUE;
    - curves, and the momentum "reduce" and B1 "reduce" / "switch" rows at zero cost:
      recomputed here from public data with the studies' own constructions. The
      recomputation is checked against the published figures before anything is
      drawn (`check`), and a mismatch stops the build. Nothing here is a trial.
The crypto trend sleeve lives in the private neighbour repo: only aggregates appear.

Usage: .venv/bin/python scripts/build_presentation_partie2.py
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_presentation as bp  # noqa: E402
import run_b1_regime_coupling as b1  # noqa: E402
from run_crisis_coupling import WINDOWS, series  # noqa: E402
from run_m3_evaluation import cash_rate_daily, funding_charge  # noqa: E402

from regime_lab.config import CACHE, RAW  # noqa: E402
from regime_lab.extensions import crisis  # noqa: E402

ROOT = bp.ROOT
OUT_DIR = bp.OUT_DIR
BLUE, ORANGE, RED, GREY = bp.BLUE, bp.ORANGE, bp.RED, bp.GREY
INK, INK2, MUTED, RULE = bp.INK, bp.INK2, bp.MUTED, bp.RULE
fr, esc, svg_text, table = bp.fr, bp.esc, bp.svg_text, bp.table

UMD_SPAN = (pd.Timestamp("2002-10-29"), pd.Timestamp("2026-07-31"))
B1_SPAN = (pd.Timestamp("2004-10-05"), pd.Timestamp("2026-09-08"))

# --- transcribed from docs/artifacts/partie2/lectures_trois_strategies.txt --------------
# columns: Sharpe, ann. return, ann. vol, max DD, 2008, rebound 2009, Covid, rebound 2020, 2022
UMD_ROWS = {
    "Seule": (0.47, 5.0, 10.6, -26.2, 25.3, -17.4, 7.9, -8.9, 7.6),
    "Couper": (0.70, 6.8, 9.7, -20.9, 10.9, 2.3, 10.6, 2.0, 7.6),
    "Or": (0.77, 8.1, 10.5, -20.9, 8.2, 17.5, 7.1, 16.7, 7.6),
    "Obligations (TLT)": (0.66, 6.9, 10.5, -20.9, 19.1, -3.6, 12.7, 2.0, 7.6),
    "Options": (0.48, 5.2, 10.8, -28.9, 36.6, -20.0, 23.0, -13.9, 7.6),
}
B1_ROWS = {
    "Seule": (0.85, 8.0, 9.4, -17.7, 9.4, 14.0, 5.8, -6.3, 5.7),
    "Couper": (0.86, 7.4, 8.6, -17.7, 5.3, 5.0, 5.8, -5.1, 5.7),
    "Or": (0.89, 8.3, 9.4, -17.7, 4.8, 20.6, 2.5, 8.5, 5.7),
    "Obligations (TLT)": (0.75, 7.1, 9.4, -21.9, 13.4, -1.0, 7.8, -5.1, 5.7),
    "Options": (0.69, 6.8, 9.8, -28.4, 29.3, -17.9, 17.7, -19.9, 5.7),
}
# crypto: Sharpe, ann. return, ann. vol, max DD, Covid, rebound 2020, 2022 (None: not in that reading)
CRYPTO_ROWS = {
    "Seule": (1.13, 12.3, 10.8, -15.2, -9.7, 31.7, 1.0),
    "Couper": (0.89, 9.3, 10.4, -20.3, -8.6, -2.4, 1.0),
    "Réduire de moitié": (1.02, 10.8, 10.5, -15.2, -9.2, None, 1.0),
    "Basculer vers la tendance": (0.92, 10.0, 10.8, -15.2, -8.7, None, 1.0),
    "Or": (0.93, 10.1, 10.8, -17.4, -9.7, 12.7, 1.0),
    "Obligations (TLT)": (0.79, 8.6, 10.9, -27.3, -7.2, -1.6, 1.0),
    "Options": (0.71, 7.7, 10.9, -34.8, -5.7, -17.7, 1.0),
}
# Δ Sharpe computed before rounding (tests, or the checked recomputation for ° rows); the
# 10-minute deck shows the same values. Order: stop, halve, switch, gold, Treasuries, options.
DELTAS = {
    "Momentum actions": (0.23, 0.12, None, 0.30, 0.19, 0.00),
    "Rebond obligataire": (0.01, 0.02, -0.04, 0.04, -0.10, -0.16),
    "Tendance crypto": (-0.24, -0.11, -0.21, -0.20, -0.34, -0.43),
}
STRESS_CALM = {  # Sharpe of the strategy alone, by lagged state (descriptive, haven reading)
    "Momentum actions": (-1.13, 0.75),
    "Rebond obligataire": (0.41, 0.92),
    "Tendance crypto": (3.89, 0.93),
}
# verdict table: method, delta, MDE, t, placebo, vol rules (median / 80th), verdict
VERDICTS = {
    "Momentum actions": [
        ("Couper *", "+0,166", "0,395", "+1,05", "100 %", "+0,002 / +0,163", "Sous-puissant"),
        ("Réduire de moitié *", "+0,094", "0,185", "+1,05", "100 %", "+0,048 / +0,098", "Sous-puissant"),
        ("Or", "+0,298", "0,624", "+2,68", "100 %", "+0,349 / +0,213", "Sous-puissant"),
        ("Obligations (TLT)", "+0,187", "0,466", "+1,82", "100 %", "+0,050 / +0,173", "Sous-puissant"),
        ("Options", "+0,002", "0,572", "+0,11", "100 %", "−0,842 / −0,022", "Sous-puissant"),
    ],
    "Rebond obligataire": [
        ("Couper †", "+0,013", "0,229", "−0,70", "85 %", "−0,26 (médiane)", "Sous-puissant"),
        ("Réduire de moitié †", "+0,023", "0,106", "−0,70", "87 %", "−0,058 (médiane)", "Sous-puissant"),
        ("Basculer vers la tendance †", "−0,047", "0,328", "−0,41", "56 %", "−0,39 (médiane)", "Pas utile"),
        ("Or", "+0,041", "0,398", "+0,36", "83 %", "+0,026 / −0,089", "Sous-puissant"),
        ("Obligations (TLT)", "−0,095", "0,362", "−1,06", "49 %", "−0,254 / −0,127", "Pas utile"),
        ("Options", "−0,155", "0,722", "−0,95", "100 %", "−1,054 / −0,245", "Pas utile"),
    ],
    "Tendance crypto": [
        ("Couper", "−0,242", "0,898", "−2,56", "5 %", "−0,147 (médiane)", "Pas utile"),
        ("Réduire de moitié", "−0,110", "0,438", "−2,56", "3 %", "+0,002 (médiane)", "Pas utile"),
        ("Basculer vers la tendance", "−0,211", "0,693", "−2,08", "14 %", "−0,174 (médiane)", "Pas utile"),
        ("Or", "−0,200", "0,750", "−2,00", "14 %", "+0,151 / −0,002", "Pas utile"),
        ("Obligations (TLT)", "−0,344", "1,102", "−2,70", "4 %", "−0,275 / −0,150", "Pas utile"),
        ("Options", "−0,428", "1,365", "−2,89", "14 %", "−0,880 / −0,237", "Pas utile"),
    ],
}


# ---------------------------------------------------------------------------
# public recomputation, checked against the readings
# ---------------------------------------------------------------------------
def sharpe(x: pd.Series) -> float:
    v = x.dropna().to_numpy(float)
    return float(v.mean() / v.std(ddof=1) * np.sqrt(252))


def max_dd(x: pd.Series) -> float:
    curve = (1.0 + x.fillna(0.0)).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def onto(pnl: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Compound a daily P&L onto another calendar (as in the haven-switch study)."""
    live = pnl.dropna()
    wealth = (1.0 + live).cumprod()
    wealth = pd.concat([pd.Series(1.0, index=[live.index[0] - pd.Timedelta(days=1)]), wealth])
    on = wealth.reindex(wealth.index.union(index)).ffill().reindex(index)
    return on.pct_change().where(index > live.index[0])


def state_series() -> pd.Series:
    return pd.read_parquet(CACHE / "states.parquet")["A' sparse jump"]


def umd_arms() -> dict[str, pd.Series]:
    unit = series(RAW / "crisis" / "french_umd.parquet", "ff_umd")
    x = (crisis.vol_target_weight(unit) * unit).loc[UMD_SPAN[0]:UMD_SPAN[1]]
    stress = crisis.lagged_state(state_series(), x.index).eq(crisis.STRESS)
    prices = pd.read_parquet(CACHE / "trend_universe_m1.parquet")
    px = prices["GC=F"].dropna()
    gunit = (px.pct_change() - cash_rate_daily(px.index)).dropna()
    gold = onto(crisis.vol_target_weight(gunit) * gunit, x.index)
    return {"Seule": x, "Couper": x.where(~stress, 0.0),
            "Réduire de moitié": x.where(~stress, 0.5 * x), "Or": x.where(~stress, gold),
            "_stress": stress}


def b1_arms() -> dict[str, pd.Series]:
    p = b1.build()
    state = p["state"]

    def run(w: pd.DataFrame) -> pd.Series:
        gross = (w * p["returns"]).sum(axis=1)
        return (gross - funding_charge(w, p["rate"], mode="signed")).reindex(p["index"])

    out = {"Seule": run(p["base"])}
    for label, mode in (("Couper", "stop"), ("Réduire de moitié", "reduce"),
                        ("Basculer vers la tendance", "switch")):
        out[label] = run(b1.coupled(p["base"], state, mode, p["trend"]))
    out = {k: v.loc[B1_SPAN[0]:B1_SPAN[1]] for k, v in out.items()}
    out["_stress"] = state.reindex(out["Seule"].index).eq(b1.STRESS)
    return out


def row_of(x: pd.Series) -> tuple:
    """Sharpe, return %, vol %, max DD %, then the crisis windows in %, as in the readings."""
    x = x.dropna()
    wins = [float((1.0 + x.loc[a:b]).prod() - 1.0) * 100 for a, b in WINDOWS.values()]
    gfc, reb09, _feb18, covid, reb20, y22 = wins
    return (sharpe(x), x.mean() * 252 * 100, x.std(ddof=1) * np.sqrt(252) * 100, max_dd(x) * 100,
            gfc, reb09, covid, reb20, y22)


def check(umd: dict, bond: dict) -> dict[str, float]:
    """Stop the build if the public recomputation drifts from the published readings."""
    pairs = [("UMD alone", sharpe(umd["Seule"]), UMD_ROWS["Seule"][0]),
             ("UMD stop", sharpe(umd["Couper"]), UMD_ROWS["Couper"][0]),
             ("UMD gold", sharpe(umd["Or"]), UMD_ROWS["Or"][0]),
             ("B1 alone", sharpe(bond["Seule"]), B1_ROWS["Seule"][0]),
             ("B1 stop", sharpe(bond["Couper"]), B1_ROWS["Couper"][0])]
    for name, got, want in pairs:
        if abs(got - want) > 0.006:
            raise SystemExit(f"{name}: recomputed {got:.3f}, published {want:.2f}. Not drawing.")
        print(f"   check {name:<10} {got:.3f} ~ {want:.2f}")
    return {"umd_half": row_of(umd["Réduire de moitié"]),
            "b1_reduce": row_of(bond["Réduire de moitié"]),
            "b1_switch": row_of(bond["Basculer vers la tendance"])}


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------
def wealth_chart(arms: dict[str, pd.Series], stress: pd.Series, colours: dict[str, str],
                 *, width: int = 760, height: int = 330) -> str:
    left, right, top, bottom = 52, 118, 14, 30
    curves = {k: (1.0 + v.fillna(0.0)).cumprod() for k, v in arms.items()}
    idx = next(iter(curves.values())).index
    t0, t1 = idx.min(), idx.max()
    lo = math.log(min(c.min() for c in curves.values()) * 0.95)
    hi = math.log(max(c.max() for c in curves.values()) * 1.05)

    def x(d: pd.Timestamp) -> float:
        return left + (d - t0).days / (t1 - t0).days * (width - left - right)

    def y(v: float) -> float:
        return top + (hi - math.log(v)) / (hi - lo) * (height - top - bottom)

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    seg = (stress != stress.shift()).cumsum()
    for _, g in stress.groupby(seg):
        if bool(g.iloc[0]):
            x0, x1 = x(g.index[0]), x(g.index[-1])
            out.append(f'<rect x="{x0:.1f}" y="{top}" width="{max(x1 - x0, 1.5):.1f}" '
                       f'height="{height - top - bottom}" fill="{RED}" fill-opacity="0.14"/>')
    for level in (0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8):
        if lo < math.log(level) < hi:
            yy = y(level)
            out.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}" stroke="{RULE}"/>')
            out.append(svg_text(left - 6, yy, fr(level, "g") + " €", size=11, fill=MUTED, anchor="end"))
    years = range(t0.year + 1, t1.year + 1)
    step = 4 if len(years) > 14 else 2
    for yr in years:
        if yr % step == 0:
            out.append(svg_text(x(pd.Timestamp(f"{yr}-01-01")), height - 10, str(yr), size=11,
                                fill=MUTED, anchor="middle"))
    ends = []
    for name, c in curves.items():
        c = c.resample("W-FRI").last().dropna()
        pts = " ".join(f"{x(d):.1f},{y(v):.1f}" for d, v in c.items())
        wid = 2.2 if colours[name] != INK else 1.6
        out.append(f'<polyline points="{pts}" fill="none" stroke="{colours[name]}" stroke-width="{wid}"/>')
        ends.append([y(c.iloc[-1]), name, c.iloc[-1]])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 15)
    for yy, name, v in ends:
        out.append(svg_text(width - right + 8, yy, f"{esc(name)} · {fr(v, '.2f')} €", size=12,
                            fill=INK, weight=600 if colours[name] != INK else 400))
    out.append("</svg>")
    return "".join(out)


def stress_calm_chart() -> str:
    width, row, label_w = 640, 70, 170
    names = list(STRESS_CALM)
    height = row * len(names) + 34
    lo, hi = -1.5, 4.0
    plot = width - label_w - 60

    def x(v: float) -> float:
        return label_w + (v - lo) / (hi - lo) * plot

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    for v in (-1, 0, 1, 2, 3, 4):
        out.append(f'<line x1="{x(v):.1f}" y1="4" x2="{x(v):.1f}" y2="{height - 28}" '
                   f'stroke="{GREY if v == 0 else RULE}" stroke-width="{1.5 if v == 0 else 1}"/>')
        out.append(svg_text(x(v), height - 12, fr(v, "+.0f") if v else "0", size=12, fill=MUTED, anchor="middle"))
    for k, name in enumerate(names):
        s, c = STRESS_CALM[name]
        y0 = 8 + k * row
        out.append(svg_text(label_w - 12, y0 + 26, esc(name), size=15, anchor="end", weight=600))
        for j, (v, col) in enumerate(((c, BLUE), (s, ORANGE))):
            yy = y0 + 6 + j * 24
            a, b = sorted((x(0), x(v)))
            out.append(f'<rect x="{a:.1f}" y="{yy}" width="{max(b - a, 2):.1f}" height="19" rx="4" fill="{col}"/>')
            tx, anchor = (b + 7, "start") if v >= 0 else (a - 7, "end")
            out.append(svg_text(tx, yy + 10, fr(v, "+.2f"), size=13, anchor=anchor))
    out.append("</svg>")
    return "".join(out)


def crypto_bars() -> str:
    arms = [(k, v[4], v[5]) for k, v in CRYPTO_ROWS.items() if v[5] is not None]
    width, row, label_w = 620, 34, 200
    height = row * len(arms) + 34
    lo, hi = -25.0, 40.0
    plot = width - label_w - 30

    def x(v: float) -> float:
        return label_w + (v - lo) / (hi - lo) * plot

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    for v in (-20, -10, 0, 10, 20, 30, 40):
        out.append(f'<line x1="{x(v):.1f}" y1="4" x2="{x(v):.1f}" y2="{height - 28}" '
                   f'stroke="{GREY if v == 0 else RULE}" stroke-width="{1.5 if v == 0 else 1}"/>')
        out.append(svg_text(x(v), height - 12, f"{v:+d} %".replace("+0", "0").replace("-", "−"), size=11, fill=MUTED, anchor="middle"))
    for k, (name, _covid, reb) in enumerate(arms):
        y0 = 6 + k * row
        out.append(svg_text(label_w - 10, y0 + 15, esc(name), size=14, anchor="end",
                            weight=600 if name == "Seule" else 400))
        a, b = sorted((x(0), x(reb)))
        col = BLUE if name == "Seule" else GREY
        out.append(f'<rect x="{a:.1f}" y="{y0 + 5}" width="{max(b - a, 2):.1f}" height="20" rx="4" fill="{col}"/>')
        tx = b + 7 if reb >= 0 else x(0) + 7
        out.append(svg_text(tx, y0 + 15, fr(reb, "+.1f") + " %", size=13))
    out.append("</svg>")
    return "".join(out)


def covid_chart() -> str:
    spx = b1.us_prices("eq_us_large")
    spx.index = pd.to_datetime(spx.index)
    spx = spx.loc["2019-11-01":"2021-06-30"]
    st = state_series().reindex(spx.index).ffill()
    width, height, left, right, top, bottom = 760, 330, 56, 16, 26, 30
    t0, t1 = spx.index.min(), spx.index.max()
    lo, hi = math.log(2100), math.log(4500)

    def x(d: pd.Timestamp) -> float:
        return left + (d - t0).days / (t1 - t0).days * (width - left - right)

    def y(v: float) -> float:
        return top + (hi - math.log(v)) / (hi - lo) * (height - top - bottom)

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    stress = st.eq(0.0)
    seg = (stress != stress.shift()).cumsum()
    for _, g in stress.groupby(seg):
        if bool(g.iloc[0]):
            x0, x1 = x(g.index[0]), x(g.index[-1])
            out.append(f'<rect x="{x0:.1f}" y="{top}" width="{max(x1 - x0, 1.5):.1f}" '
                       f'height="{height - top - bottom}" fill="{RED}" fill-opacity="0.16"/>')
    for level in (2500, 3000, 3500, 4000):
        yy = y(level)
        out.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}" stroke="{RULE}"/>')
        out.append(svg_text(left - 6, yy, f"{level:,}".replace(",", " "), size=11, fill=MUTED, anchor="end"))
    for d, lab in (("2020-01-01", "janv. 2020"), ("2020-07-01", "juil. 2020"), ("2021-01-01", "janv. 2021"),
                   ("2021-06-01", "juin 2021")):
        out.append(svg_text(x(pd.Timestamp(d)), height - 10, lab, size=11, fill=MUTED, anchor="middle"))
    pts = " ".join(f"{x(d):.1f},{y(v):.1f}" for d, v in spx.items())
    out.append(f'<polyline points="{pts}" fill="none" stroke="{INK}" stroke-width="1.8"/>')
    marks = [("2020-02-19", "1", -16, 0), ("2020-03-11", "2", 0, 16), ("2020-03-23", "3", 16, 0),
             ("2021-04-01", "4", -18, 0)]
    for d, num, dy, dx in marks:
        dd = pd.Timestamp(d)
        v = spx.loc[:dd].iloc[-1]
        out.append(f'<circle cx="{x(dd):.1f}" cy="{y(v):.1f}" r="3.5" fill="{INK}"/>')
        cx, cy = x(dd) + dx, y(v) + dy
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="10" fill="{BLUE}" stroke="#fff" stroke-width="1.5"/>')
        out.append(svg_text(cx, cy + 0.5, num, size=12, fill="#fff", anchor="middle", weight=700))
    out.append("</svg>")
    return "".join(out)


def longhist_chart() -> str:
    rows = [("1937-1962", -0.008), ("1963-2001", 0.031), ("2002-2026", 0.145), ("1937-2026, tout", 0.045)]
    width, row, label_w = 620, 50, 170
    height = row * len(rows) + 34
    lo, hi = -0.05, 0.20
    plot = width - label_w - 40

    def x(v: float) -> float:
        return label_w + (v - lo) / (hi - lo) * plot

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    for v in (0.0, 0.05, 0.10, 0.15, 0.20):
        out.append(f'<line x1="{x(v):.1f}" y1="4" x2="{x(v):.1f}" y2="{height - 28}" '
                   f'stroke="{GREY if v == 0 else RULE}" stroke-width="{1.5 if v == 0 else 1}"/>')
        out.append(svg_text(x(v), height - 12, fr(v, "+.2f") if v else "0", size=12, fill=MUTED, anchor="middle"))
    xm = x(0.158)
    out.append(f'<line x1="{xm:.1f}" y1="4" x2="{xm:.1f}" y2="{height - 28}" stroke="{INK2}" stroke-dasharray="5 4" stroke-width="1.5"/>')
    out.append(svg_text(xm - 6, 12, "seuil de détection : 0,158", size=12, fill=INK2, anchor="end"))
    for k, (lab, v) in enumerate(rows):
        yc = 14 + k * row + row / 2 - 4
        last = k == len(rows) - 1
        out.append(svg_text(label_w - 12, yc, lab, size=14, anchor="end", weight=600 if last else 400))
        a, b = sorted((x(0), x(v)))
        out.append(f'<rect x="{a:.1f}" y="{yc - 12:.1f}" width="{max(b - a, 2):.1f}" height="24" rx="4" fill="{BLUE if last else GREY}"/>')
        tx, anchor = (b + 7, "start") if v >= 0 else (a - 7, "end")
        out.append(svg_text(tx, yc, fr(v, "+.3f"), size=13, anchor=anchor, weight=600 if last else 400))
    out.append("</svg>")
    return "".join(out)


def diversification_chart() -> str:
    rows = [("Moyenne des 9 stratégies seules", 0.58, GREY), ("La meilleure stratégie seule", 1.04, GREY),
            ("Les 9 ensemble, sans filtre", 1.91, BLUE)]
    width, row, label_w = 640, 64, 260
    height = row * len(rows) + 34
    plot = width - label_w - 60

    def x(v: float) -> float:
        return label_w + plot * v / 2.0

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    for v in (0.0, 0.5, 1.0, 1.5, 2.0):
        out.append(f'<line x1="{x(v):.1f}" y1="4" x2="{x(v):.1f}" y2="{height - 28}" '
                   f'stroke="{GREY if v == 0 else RULE}" stroke-width="{1.5 if v == 0 else 1}"/>')
        out.append(svg_text(x(v), height - 12, fr(v, ".1f"), size=12, fill=MUTED, anchor="middle"))
    for k, (lab, v, col) in enumerate(rows):
        yc = 8 + k * row + row / 2 - 4
        lead = col == BLUE
        out.append(svg_text(label_w - 12, yc, esc(lab), size=15, anchor="end", weight=600 if lead else 400))
        out.append(f'<rect x="{x(0):.1f}" y="{yc - 15:.1f}" width="{x(v) - x(0):.1f}" height="30" rx="4" fill="{col}"/>')
        out.append(svg_text(x(v) + 8, yc, fr(v, ".2f"), size=16, weight=600 if lead else 400))
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------
def pct(v: float | None, spec: str = "+.1f") -> str:
    return "—" if v is None else fr(v, spec) + " %"


def arm_table(rows: dict[str, tuple], extra: dict[str, tuple] | None = None, *, crypto: bool = False) -> str:
    """The readings' rows, with recomputed rows (marked °) inserted after « Couper »."""
    if crypto:
        head = ["Méthode", "Sharpe", "Rend./an", "Vol.", "Perte max", "Covid", "Rebond 2020", "2022"]
        body = [[f"<b>{k}</b>" if k == "Seule" else k, fr(v[0], ".2f"), pct(v[1]), pct(v[2], ".1f"),
                 pct(v[3]), pct(v[4]), pct(v[5]), pct(v[6])] for k, v in rows.items()]
        return table(head, body, num=set(range(1, 8)), highlight=0, cls="compact tight")
    head = ["Méthode", "Sharpe", "Rend./an", "Vol.", "Perte max", "2008", "Rebond 2009", "Covid", "Rebond 2020"]
    body = []
    for k, v in rows.items():
        body.append([f"<b>{k}</b>" if k == "Seule" else k, fr(v[0], ".2f"), pct(v[1]), pct(v[2], ".1f"),
                     pct(v[3]), pct(v[4]), pct(v[5]), pct(v[6]), pct(v[7])])
    for i, (k, v) in enumerate((extra or {}).items()):
        body.insert(2 + i, [k + " °", fr(v[0], ".2f"), pct(v[1]), pct(v[2], ".1f"), pct(v[3]),
                            pct(v[4]), pct(v[5]), pct(v[6]), pct(v[7])])
    return table(head, body, num=set(range(1, 9)), highlight=0, cls="compact tight")


def matrix(extra: dict[str, float]) -> str:
    methods = ["Couper", "Réduire de moitié", "Basculer vers la tendance", "Or", "Obligations (TLT)", "Options"]
    base = {"Momentum actions": UMD_ROWS["Seule"][0], "Rebond obligataire": B1_ROWS["Seule"][0],
            "Tendance crypto": CRYPTO_ROWS["Seule"][0]}
    vals = {
        "Momentum actions": {"Couper": UMD_ROWS["Couper"][0], "Réduire de moitié": extra["umd_half"][0],
                             "Basculer vers la tendance": None, "Or": UMD_ROWS["Or"][0],
                             "Obligations (TLT)": UMD_ROWS["Obligations (TLT)"][0], "Options": UMD_ROWS["Options"][0]},
        "Rebond obligataire": {"Couper": B1_ROWS["Couper"][0], "Réduire de moitié": extra["b1_reduce"][0],
                               "Basculer vers la tendance": extra["b1_switch"][0], "Or": B1_ROWS["Or"][0],
                               "Obligations (TLT)": B1_ROWS["Obligations (TLT)"][0], "Options": B1_ROWS["Options"][0]},
        "Tendance crypto": {m: CRYPTO_ROWS[m][0] for m in methods},
    }
    head = "<th>Méthode en stress</th>" + "".join(f'<th class="num">{s}<br><span class="sub">seule : {fr(base[s], ".2f")}</span></th>' for s in base)
    body = []
    for m in methods:
        cells = [f"<td>{m}</td>"]
        for s in base:
            v = vals[s][m]
            if v is None:
                cells.append('<td class="num cell na">non testé</td>')
                continue
            d = DELTAS[s][methods.index(m)]
            cls = "pos" if d >= 0.05 else ("neg" if d <= -0.05 else "flat")
            cells.append(f'<td class="num cell {cls}"><b>{fr(d, "+.2f")}</b><span class="lvl">{fr(v, ".2f")}</span></td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="matrix"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def build_html() -> str:
    print("recomputing the public series and checking them against the readings")
    umd, bond = umd_arms(), b1_arms()
    extra = check(umd, bond)
    s = []
    slide = bp.slide
    s.append("""
<section class="slide cover">
  <div class="cover-band"></div>
  <div class="cover-inner">
    <div class="eyebrow light">Projet Big Data · Master 2 · Partie 2</div>
    <h1 class="cover-title">Le filtre de régime appliqué à trois stratégies de trading</h1>
    <p class="cover-sub">Momentum actions, rebond obligataire de fin de mois, tendance crypto&nbsp;:
    six façons d'utiliser le filtre, comparées à la stratégie seule. <b>Le filtre change la forme
    du risque, rarement la performance.</b></p>
    <div class="cover-meta">Couper · réduire · basculer · or · obligations · options</div>
  </div>
</section>""")

    kpis = [("3", "stratégies aux profils opposés", "le filtre aide la première, laisse la deuxième indifférente, pénalise la troisième"),
            ("6", "façons d'utiliser le filtre", "couper, réduire, basculer vers la tendance, or, obligations, options"),
            ("+0,30", "le meilleur cas", "momentum : 0,47 → 0,77 en le remplaçant par de l'or en stress ; sous le seuil de détection"),
            ("0", "usage démontré", "aucun ne bat à la fois le hasard, le seuil de détection et une règle de volatilité")]
    tiles = "".join(f'<div class="kpi"><div class="kv">{v}</div><div class="kl">{lab}</div><div class="kd">{d}</div></div>' for v, lab, d in kpis)
    s.append(slide(2, "Synthèse", "Le filtre agit, mais pas toujours dans le bon sens",
        f"""<div class="kpis">{tiles}</div>
        <ol class="msgs">
          <li><b>L'effet dépend entièrement de la stratégie.</b> Le momentum perd en période de stress&nbsp;:
          le couper aide. La tendance crypto y gagne le plus&nbsp;: la couper la pénalise.</li>
          <li><b>Aucun usage ne bat de façon démontrable une simple règle de volatilité.</b> Même le
          meilleur cas reste sous le seuil de détection, et il fond quand on le vérifie sur 90 ans.</li>
          <li><b>La raison est comprise</b>&nbsp;: le modèle alerte tard et reste en mode crise pendant la
          reprise. Ce qui paie vraiment, c'est la diversification.</li>
        </ol>""", "docs/RESULTS_REFUGE.md, RESULTS_COUPLAGE_STRATEGIES.md, RESULTS_LONGHIST.md"))

    s.append(slide(3, "Avant de commencer", "Comment lire les chiffres de cette partie",
        """<div class="two even">
          <div><h3>Les mesures de performance</h3>
            <ul class="kvlist tall">
              <li><span>Sharpe</span><b>rendement par unité de risque, en excès du taux sans risque. Au-delà de 1, c'est très bon</b></li>
              <li><span>Rendement / vol.</span><b>moyenne annuelle et volatilité annuelle</b></li>
              <li><span>Perte max</span><b>la pire chute, d'un sommet au creux suivant</b></li>
              <li><span>Fenêtres de crise</span><b>2008 : 9 oct. 2007 → 9 mars 2009 · rebond 2009 : → 31 déc. 2009 · Covid : 19 févr. → 23 mars 2020 · rebond 2020 : → 31 déc. 2020</b></li>
              <li><span>Coûts</span><b>aucun, partout : c'est une hypothèse de simplicité du projet</b></li>
            </ul></div>
          <div><h3>Les garde-fous</h3>
            <ul class="kvlist tall">
              <li><span>Le filtre</span><b>l'état du Sparse Jump Model connu la veille au soir ; la position est prise le lendemain</b></li>
              <li><span>Δ Sharpe</span><b>Sharpe avec le filtre moins Sharpe de la stratégie seule</b></li>
              <li><span>Seuil de détection</span><b>le plus petit écart qu'on saurait distinguer du hasard sur cette durée</b></li>
              <li><span>Placebo</span><b>on place les jours de stress au hasard, 400 fois : le vrai filtre doit faire mieux</b></li>
              <li><span>Témoin</span><b>la même méthode pilotée par une règle d'une ligne : « la volatilité récente est-elle haute ? »</b></li>
            </ul></div>
        </div>
        <div class="verdicts">
          <div><span class="v ok">Utile</span>au-dessus du seuil, confirmé par le test et le placebo, et meilleur que le témoin</div>
          <div><span class="v under">Sous-puissant</span>positif, mais sous le seuil : ni oui ni non</div>
          <div><span class="v no">Pas utile</span>le filtre ne fait pas mieux que la stratégie seule</div>
        </div>""",
        "docs/RESULTS_CRISE.md (critère de verdict), docs/PROTOCOL_FREEZE.md"))

    methods = [("1", "Couper", "On ne fait pas tourner la stratégie en stress.", "Éviter les pertes des crises.", "Rater le rebond qui suit."),
               ("2", "Réduire de moitié", "La position est divisée par deux en stress.", "Un compromis entre garder et couper.", "La moitié des deux effets."),
               ("3", "Basculer vers une autre stratégie", "En stress, on tient un livre de tendance sur 46 marchés.", "La tendance gagne souvent dans les crises longues.", "Elle se retourne dans les rebonds en V."),
               ("4", "Remplacer par de l'or", "En stress, on tient de l'or à la place.", "La valeur refuge historique.", "L'or ne monte pas à chaque crise."),
               ("5", "Remplacer par des obligations d'État", "En stress, on tient des obligations américaines à 20 ans et plus (TLT).", "Elles montent quand les actions chutent.", "Pas quand l'inflation est la cause (2022)."),
               ("6", "Remplacer par des options", "En stress, on achète de la volatilité (un straddle couvert, reconstitué avec le VIX).", "Gagner quand la volatilité explose.", "Elle est déjà chère quand on l'achète.")]
    cards = "".join(f'<div class="mcard"><div class="mnum">{n}</div><h3>{t}</h3><p>{d}</p>'
                    f'<p class="pro"><b>L\'idée&nbsp;:</b> {i}</p><p class="con"><b>Le risque&nbsp;:</b> {r}</p></div>'
                    for n, t, d, i, r in methods)
    s.append(slide(4, "Les méthodes", "Six façons de tirer profit d'un filtre de crise",
        f"""<div class="mgrid">{cards}</div>
        <div class="note">Les jambes de remplacement (or, obligations, options) sont dimensionnées pour une
        volatilité de 10&nbsp;% par an, quelle que soit la stratégie remplacée. Une septième façon, le
        régime comme <b>budget de risque</b> entre plusieurs stratégies, est traitée page 13.</div>""",
        "docs/RESULTS_REFUGE.md, docs/RESULTS_COUPLAGE_STRATEGIES.md"))

    prof_rows = [["<b>Momentum actions</b>", "14 %", "perd (Sharpe −1,13)", "l'aider"],
                 ["<b>Rebond obligataire</b>", "12 %", "gagne, moins (0,41)", "être neutre"],
                 ["<b>Tendance crypto</b>", "8 %", "gagne le plus (3,89)", "la pénaliser"]]
    s.append(slide(5, "Le choix des trois stratégies", "Trois stratégies choisies pour leurs réactions opposées au stress",
        f"""<div class="two even">
          <div><div class="chart-title">Sharpe de chaque stratégie seule, selon l'état du filtre</div>
          <div class="legend"><span><i style="background:{BLUE}"></i>jours calmes</span>
          <span><i style="background:{ORANGE}"></i>jours de stress</span></div>{stress_calm_chart()}</div>
          <div>{table(["Stratégie", "Jours en stress", "En stress, elle…", "Le filtre devrait…"], prof_rows, cls="compact")}
          <p class="callout" style="margin-top:.25in">L'intuition à tester&nbsp;: <b>couper en stress ne peut aider
          qu'une stratégie qui y perd.</b> Ces trois-là couvrent les trois cas&nbsp;: elle y perd, elle y gagne
          un peu, elle y gagne beaucoup.</p></div>
        </div>""", "docs/artifacts/partie2/lectures_trois_strategies.txt (Sharpe par état, descriptif)"))

    s.append(slide(6, "Stratégie 1 — Momentum actions", "Le momentum&nbsp;: acheter les gagnants, vendre les perdants",
        f"""<div class="grid2top">
          <div><h3>Comment ça marche</h3>
            <p>Chaque mois, on achète les actions américaines qui ont le plus monté sur les 12 derniers mois
            (en sautant le dernier), et on vend celles qui ont le plus baissé. C'est une prime documentée
            depuis 1993 (Jegadeesh et Titman).</p></div>
          <div><h3>Son point faible&nbsp;: les rebonds violents</h3>
            <p>Après un krach, les actions les plus tombées remontent le plus vite&nbsp;; le momentum, vendeur
            de ces actions, s'effondre (2009, 2020). Données&nbsp;: facteur UMD de Ken French, toute la
            bourse américaine, ciblé à 10&nbsp;% de volatilité.</p></div>
        </div>
        <div class="chart-title">Les résultats, octobre 2002 → juillet 2026, sans coût</div>
        {arm_table(UMD_ROWS, {"Réduire de moitié": extra["umd_half"]})}
        <p class="small-note">Couper passe le rebond de 2009 de −17&nbsp;% à +2&nbsp;%, et l'or le porte à +18&nbsp;%.
        Les options gagnent +37&nbsp;% en 2008, puis perdent −20&nbsp;% au rebond. ° recalculé sur les mêmes dates.</p>""",
        "lectures_trois_strategies.txt ; « Réduire » recalculé sur les mêmes dates (descriptif)"))

    umd_curves = {k: umd[k] for k in ("Seule", "Couper", "Or")}
    s.append(slide(7, "Stratégie 1 — Momentum actions", "Le filtre aide le momentum, mais sans preuve suffisante",
        f"""<div class="two chart-left">
          <div><div class="chart-title">Ce que devient 1&nbsp;€ investi, échelle logarithmique</div>
          <div class="legend"><span><i style="background:{INK}"></i>seule</span><span><i style="background:{BLUE}"></i>couper en stress</span>
          <span><i style="background:{ORANGE}"></i>or en stress</span><span><i style="background:{RED};opacity:.3"></i>stress</span></div>
          {wealth_chart(umd_curves, umd["_stress"], {"Seule": INK, "Couper": BLUE, "Or": ORANGE})}</div>
          <div class="side"><h3>Le verdict des tests</h3>
            <p>Toutes les méthodes améliorent le momentum&nbsp;: <b>+0,30</b> avec l'or, +0,23 en coupant.
            Aucun des 400 placements de stress au hasard ne fait aussi bien.</p>
            <p>Mais <b>tous les écarts restent sous le seuil de détection</b> (0,62 pour l'or) : verdict
            sous-puissant.</p>
            <p class="callout">Et une règle de volatilité d'une ligne fait aussi bien&nbsp;: 0,82 avec l'or.</p>
          </div></div>""", "lectures_trois_strategies.txt ; courbes recalculées et vérifiées contre la lecture"))

    s.append(slide(8, "Stratégie 1 — L'épreuve des 90 ans", "Sur 90 ans, le gain du momentum fond de +0,17 à +0,05",
        f"""<div class="two chart-left">
          <div><div class="chart-title">Gain de Sharpe en coupant le momentum en stress, par période
          (modèle réestimé sur 1926-2026)</div>{longhist_chart()}</div>
          <div class="side"><h3>Ce qu'on a fait</h3>
            <p>Réestimer le même type de modèle sur un siècle, avec les variables qui existent depuis 1926,
            pour avoir 14 récessions au lieu de 2 et un test 2,5 fois plus précis.</p>
            <h3>Ce qu'on a trouvé</h3>
            <p>Le gain vient presque entièrement de 2002-2026. Sur 90 ans, il vaut +0,045, trois fois et
            demie sous le seuil, et il vient d'une baisse du risque, pas d'un gain de rendement.</p>
            <p class="callout">Une règle publiée fait mieux&nbsp;: « marché baissier + forte volatilité »
            (Daniel et Moskowitz) donne +0,11 et ramène la perte max de −37&nbsp;% à −27&nbsp;%.</p>
          </div></div>""", "docs/RESULTS_LONGHIST.md (modèle réduit à 30 variables, sans VIX ni macro)"))

    b1_curves = {k: bond[k] for k in ("Seule", "Couper")}
    s.append(slide(9, "Stratégie 2 — Rebond obligataire de fin de mois", "Le rebond obligataire&nbsp;: insensible au filtre",
        f"""<div class="grid2top">
          <div><h3>Comment ça marche</h3>
            <p>On achète des obligations d'État américaines longues (TLT) pendant les 3 dernières séances de
            chaque mois&nbsp;: les fonds de pension et les indices se rééquilibrent en fin de mois, ce qui crée
            une demande prévisible. <b>C'est un effet de calendrier, pas une prime de risque</b>&nbsp;: il gagne
            en 2008, en 2020 et en 2022. Il n'y a rien à protéger.</p></div>
          <div><div class="chart-title">1&nbsp;€ investi&nbsp;: seule et en coupant en stress (même Sharpe, un peu moins de rendement)</div>
            {wealth_chart(b1_curves, bond["_stress"], {"Seule": INK, "Couper": BLUE}, height=200)}</div>
        </div>
        <div class="chart-title">Les résultats, octobre 2004 → septembre 2026, sans coût</div>
        {arm_table(B1_ROWS, {"Réduire de moitié": extra["b1_reduce"], "Basculer vers la tendance": extra["b1_switch"]})}
        <p class="small-note">Tous les écarts de Sharpe sont entre −0,16 et +0,04&nbsp;: l'or et la coupure sont
        neutres, les options et les obligations dégradent. ° recalculé sans coût (les tests étaient à 1 pb).</p>""",
        "lectures_trois_strategies.txt ; RESULTS_B1_COUPLAGE.md ; lignes ° recalculées sans coût"))

    s.append(slide(10, "Stratégie 3 — Tendance crypto", "La tendance crypto gagne en stress&nbsp;: le filtre la pénalise",
        f"""<div class="grid2top">
          <div><h3>Comment ça marche</h3>
            <p>On suit la tendance du bitcoin et de l'ether&nbsp;: acheteur quand le prix monte depuis plusieurs
            semaines, hors du marché sinon, taille ajustée à la volatilité. <b>Son meilleur moment tombe dans le
            stress du modèle</b>&nbsp;: d'octobre 2020 à mars 2021, la crypto explose pendant que le modèle est
            encore en mode crise (Sharpe 3,89 en stress, 0,93 en calme).</p></div>
          <div><div class="chart-title">Rendement pendant le rebond 2020 (24 mars → 31 déc.)</div>{crypto_bars()}</div>
        </div>
        <div class="chart-title">Les résultats, décembre 2014 → juin 2026, sans coût</div>
        {arm_table(CRYPTO_ROWS, crypto=True)}
        <p class="small-note">Toutes les méthodes baissent le Sharpe (−0,11 à −0,43). Série de recherche du dépôt privé
        voisin&nbsp;: chiffres agrégés seulement&nbsp;; «&nbsp;—&nbsp;»&nbsp;: absent de la lecture.</p>""",
        "lectures_trois_strategies.txt (lectures du 23/09, sans coût)"))

    s.append(slide(11, "La comparaison", "Même filtre, même méthode, effets opposés",
        f"""<div class="two chart-left">
          <div>{matrix(extra)}
            <div class="legend" style="margin-top:.12in"><span><i style="background:#dbe9fa"></i>gain ≥ +0,05</span>
            <span><i style="background:#f1f0ec"></i>entre −0,05 et +0,05</span><span><i style="background:#fbe3e2"></i>perte ≤ −0,05</span></div></div>
          <div class="side"><h3>Lecture</h3>
            <p><b>Par ligne</b>&nbsp;: aucune méthode n'est bonne partout. Couper aide le momentum (+0,23) et
            pénalise la crypto (−0,24).</p>
            <p><b>Par colonne</b>&nbsp;: c'est la stratégie qui décide. Ce qui compte, c'est ce qu'elle fait
            en stress, pas la méthode.</p>
            <p><b>L'or</b> est la meilleure jambe de remplacement sur deux stratégies sur trois&nbsp;;
            <b>les options</b> détruisent partout sauf sur le momentum.</p>
            <p class="callout">Aucune case n'est un résultat démontré&nbsp;: les meilleures sont sous le seuil de
            détection, et une règle de volatilité fait aussi bien.</p></div>
        </div>""", "lectures_trois_strategies.txt ; en gras le Δ, calculé avant arrondi ; en petit le Sharpe"))

    s.append(slide(12, "Pourquoi le filtre ne paie pas", "Le modèle alerte tard et reste en crise pendant la reprise",
        f"""<div class="two chart-left">
          <div><div class="chart-title">S&amp;P 500 pendant le Covid, et l'état du modèle
          <span class="legend inline"><span><i style="background:{RED};opacity:.3"></i>stress</span></span></div>{covid_chart()}
          <div class="marks"><span><b>1</b> sommet, 19 févr.</span><span><b>2</b> le modèle bascule, 11 mars (−19&nbsp;%)</span>
          <span><b>3</b> creux, 23 mars (−34&nbsp;%)</span><span><b>4</b> sortie du stress, avril 2021 (+80&nbsp;% depuis le creux)</span></div></div>
          <div class="side">
            <p><b>1. Il alerte tard.</b> Le 11 mars 2020, le VIX, l'indice de la peur, était déjà passé de 14 à 54.</p>
            <p><b>2. Il reste en crise pendant la reprise.</b> Il sort du stress quand le marché a repris
            +80&nbsp;% depuis le creux&nbsp;: 97&nbsp;% de ses jours de stress Covid tombent après le creux
            (55&nbsp;% sur les trois épisodes).</p>
            <p><b>3. Le marché le savait déjà.</b> Au-delà du VIX, le modèle n'ajoute presque rien à la
            prévision du risque (+0,20 point).</p>
            <p class="callout"><b>Conséquence&nbsp;:</b> une couverture gagne dans la chute puis reperd dans le
            rebond. Seule une stratégie qui <b>perd dans les rebonds</b>, comme le momentum, en profite.</p>
          </div></div>""", "docs/presentation/PISTES_AMELIORATION.md §0, docs/RESULTS_CRISE.md (test P), data/cache/states.parquet"))

    s.append(slide(13, "Ce qui paie vraiment", "Le vrai levier&nbsp;: la diversification, pas le filtre",
        f"""<div class="two chart-left">
          <div><div class="chart-title">Sharpe brut, 2016-2026 (les 9 stratégies testées, toutes actives)</div>
          {diversification_chart()}</div>
          <div class="side"><h3>Le constat</h3>
            <p>On combine simplement les 9 stratégies testées, chacune pondérée selon sa volatilité, sans
            aucun filtre&nbsp;: <b>1,52</b> de Sharpe sur 2005-2026, <b>1,91</b> sur 2016-2026, perte max
            −10&nbsp;%.</p>
            <p>Elles sont presque indépendantes les unes des autres&nbsp;: ensemble, elles valent 3,3 fois
            leur moyenne.</p>
            <p>Utiliser le régime comme budget de risque entre elles n'ajoute rien.</p>
            <p class="limit"><b>À dire avec ses réserves&nbsp;:</b> sans coût, et avec des stratégies choisies
            par une recherche qui connaissait ces années. C'est un plafond, pas une performance démontrée.</p>
          </div></div>""", "docs/RESULTS_BUDGET_RISQUE.md"))

    s.append(slide(14, "Ce qu'il faut retenir", "Trois messages pour la partie 2",
        """<div class="concl">
          <div class="cl"><div class="cn">1</div><div><h3>Le filtre n'a pas un effet, il en a trois.</h3>
          <p>Il aide la stratégie qui perd en stress (momentum), laisse indifférente celle qui n'en dépend pas
          (obligataire) et pénalise celle qui y gagne (crypto).</p></div></div>
          <div class="cl"><div class="cn">2</div><div><h3>Aucune méthode n'est démontrée.</h3>
          <p>Couper, réduire, basculer, or, obligations, options&nbsp;: les meilleurs cas restent sous le seuil
          de détection, et une règle de volatilité d'une ligne fait aussi bien.</p></div></div>
          <div class="cl"><div class="cn">3</div><div><h3>On sait pourquoi.</h3>
          <p>Le modèle alerte tard et reste en crise pendant la reprise&nbsp;; le VIX le savait déjà. Ce qui
          paie, c'est la diversification.</p></div></div>
          <div class="cl"><div class="cn">!</div><div><h3>À ne pas dire.</h3>
          <p>« Le filtre protège le momentum » sans « sous le seuil, et +0,05 sur 90 ans »&nbsp;; les Sharpe
          comme des performances investissables (sans coût).</p></div></div>
        </div>
        <div class="banner">Un bon thermomètre du risque ne suffit pas à gagner de l'argent&nbsp;: il faut une
        stratégie qui <b>perd précisément</b> quand il sonne.</div>""", "synthèse des pages 5 à 13"))

    vrows = []
    for strat, rows in VERDICTS.items():
        vrows.append([f"<b>{strat}</b>", "", "", "", "", "", ""])
        for r in rows:
            vrows.append(list(r))
    s.append(slide(15, "Annexe A — Le détail des tests", "Chaque méthode, testée avec un critère écrit avant la lecture",
        table(["Méthode", "Δ Sharpe", "Seuil", "t", "Placebo", "Témoins (médiane / 80ᵉ)", "Verdict"], vrows,
              num={1, 2, 3, 4}, cls="compact annex")
        + """<div class="note small">* Tests sur avril 2002 → juillet 2026 (seule&nbsp;: 0,52), les autres lignes
        du momentum sur octobre 2002 (0,47). † Tests à 1 pb de coût (seule&nbsp;: 0,79). Placebo&nbsp;: part des 400
        placements au hasard que le vrai filtre bat. Seuils corrigés pour le nombre de tests.</div>""",
        "lectures_trois_strategies.txt, RESULTS_B1_COUPLAGE.md, docs/artifacts/crise/reading.txt", cls="appendix"))

    data_rows = [["Momentum actions", "Facteur UMD, Ken French (toute la bourse US)", "oct. 2002 – juil. 2026", "public"],
                 ["Rebond obligataire", "ETF TLT (Yahoo, ajusté des coupons), 3 dernières séances du mois", "oct. 2004 – sept. 2026", "public"],
                 ["Tendance crypto", "BTC et ETH, série de recherche", "déc. 2014 – juin 2026", "dépôt privé voisin, agrégés"],
                 ["Or", "Future GC=F (Yahoo), moins le taux sans risque", "2000 – 2026", "public"],
                 ["Obligations", "ETF TLT, moins le taux sans risque", "2002 – 2026", "public"],
                 ["Options", "Swap de variance à 1 mois reconstitué avec le VIX et le S&amp;P 500", "1990 – 2026", "public"],
                 ["Le filtre", "Sparse Jump Model, état filtré, connu la veille", "avr. 2002 – sept. 2026", "data/cache/states.parquet"]]
    s.append(slide(16, "Annexe B — Données et fichiers", "D'où viennent les données, où trouver le détail",
        table(["Élément", "Source et construction", "Période", "Accès"], data_rows, cls="compact data2")
        + """<div class="note small">Tout est sans coût, en excès du taux sans risque. Lectures brutes&nbsp;:
        <code>docs/artifacts/partie2/lectures_trois_strategies.txt</code>. Études&nbsp;: <code>docs/RESULTS_REFUGE.md</code>,
        <code>RESULTS_COUPLAGE_STRATEGIES</code>, <code>RESULTS_CRISE</code>, <code>RESULTS_LONGHIST</code>,
        <code>RESULTS_BUDGET_RISQUE</code>. Ce document&nbsp;: <code>scripts/build_presentation_partie2.py</code>.</div>""", "", cls="appendix"))

    extra_css = """
.sub { display:block; font: 500 9pt "Helvetica Neue", sans-serif; text-transform:none; letter-spacing:0; color:#898781; }
table.matrix { border-collapse: separate; border-spacing: 4px; }
table.matrix th { border-bottom: 1.5px solid #0f2a44; }
table.matrix td { border: none; padding: .09in .12in; font-size: 13pt; }
td.cell { border-radius: 5px; }
td.cell .lvl { display:block; font-size: 9.5pt; color:#52514e; }
td.pos { background:#dbe9fa; } td.neg { background:#fbe3e2; } td.flat { background:#f1f0ec; }
td.na { color:#898781; font-size:10.5pt; background:#fafaf8; }
.mgrid { display:grid; grid-template-columns: repeat(3, 1fr); gap: .18in .22in; }
.mcard { border:1px solid #e6e8ec; border-radius:6px; padding:.14in .18in; background:#f7f8fa; }
.mcard h3 { font-size: 13.5pt; margin-bottom: .04in; }
.mcard p { font-size: 11.5pt; line-height: 1.35; margin: 0 0 .04in; }
.mcard .pro { color:#1b1f24; } .mcard .con { color:#52514e; }
.mnum { font: 600 11pt "Avenir Next", sans-serif; color:#2a78d6; }
.small-note { font-size: 11pt; color:#52514e; margin-top:.1in; }
ul.kvlist.tall li { font-size: 12pt; padding: .07in 0; }
ul.kvlist.tall li span { min-width: 1.5in; }
.verdicts { margin-top:.3in; display:grid; grid-template-columns: repeat(3, 1fr); gap:.25in; font-size: 12pt; color:#52514e; }
.verdicts div { border-top: 1px solid #e1e0d9; padding-top: .1in; }
.v { display:inline-block; font: 600 10.5pt "Avenir Next", sans-serif; padding: .03in .1in; border-radius: 4px; margin: 0 .08in .06in 0; }
.grid2top { display:grid; grid-template-columns: 1fr 1fr; gap: .45in; margin-bottom: .1in; }
.grid2top p { font-size: 13pt; line-height: 1.4; }
.marks { display:flex; flex-wrap:wrap; gap:.06in .25in; font-size: 11pt; color:#52514e; margin-top:.04in; }
.marks b { display:inline-block; width:.2in; height:.2in; border-radius:50%; background:#2a78d6; color:#fff; text-align:center; font-size:9.5pt; line-height:.2in; margin-right:.05in; }
table.data2 td:nth-child(3) { white-space: nowrap; }
table.tight td { font-size: 11.5pt; padding: .045in .1in; }
table.tight th { padding: .05in .1in; }
.v.ok { background:#dbe9fa; color:#0f2a44; } .v.under { background:#f1f0ec; color:#0b0b0b; } .v.no { background:#fbe3e2; color:#0b0b0b; }
table.annex td { font-size: 9.5pt; padding: .02in .08in; line-height: 1.25; }
table.annex th { padding: .04in .08in; }
"""
    return bp.PAGE.replace("</style>", extra_css + "</style>").replace(
        "<title>Régimes de marché — présentation</title>",
        "<title>Régimes de marché — partie 2</title>").replace("{{SLIDES}}", "".join(s))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    page = OUT_DIR / "presentation_partie2.html"
    pdf = OUT_DIR / "presentation_partie2.pdf"
    page.write_text(build_html(), encoding="utf-8")
    subprocess.run([bp.CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", page.as_uri()], check=True, capture_output=True)
    print(f"{pdf.relative_to(ROOT)}  ({pdf.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
