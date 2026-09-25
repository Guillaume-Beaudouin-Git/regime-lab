# ruff: noqa: E501  (the handout is prose; wrapping it would hurt more than help)
"""Build the course handout: docs/presentation/dossier_regimes.{html,pdf}.

A4, in French, for the professor, after the oral of 25 September 2026. It follows the
order of the 10-minute deck and says, for every figure, where it comes from.

Where every number comes from
    - classification (layer 1): recomputed here from data/cache/states.parquet against
      the calendar-month NBER label, and checked against docs/RESULTS_FINAL.md; a
      mismatch stops the build (`check_layer1`);
    - information and portfolio layers: transcribed from docs/RESULTS_FINAL.md;
    - test P: docs/RESULTS_CRISE.md; the 90-year test: docs/RESULTS_LONGHIST.md;
    - the three strategies: the tables of scripts/build_presentation_partie2.py, whose
      public recomputation is re-run and checked here (`p2.check`);
    - sparsity of the sparse jump model: docs/artifacts/sjm_sparsity.txt, written by
      scripts/measure_sjm_sparsity.py;
    - the number of logged configurations: read live from data/trials.parquet.
The crypto trend sleeve lives in the private neighbour repo: only aggregates appear.

Usage: .venv/bin/python scripts/build_dossier.py
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_presentation as bp  # noqa: E402
import build_presentation_partie2 as p2  # noqa: E402

from regime_lab.config import CACHE  # noqa: E402
from regime_lab.data import store  # noqa: E402

ROOT, OUT_DIR = bp.ROOT, bp.OUT_DIR
REPO_URL = "https://github.com/Guillaume-Beaudouin-Git/regime-lab"
TAG = "dossier-2026-09-25"
BLUE, ORANGE, RED, GREY = bp.BLUE, bp.ORANGE, bp.RED, bp.GREY
INK, INK2, MUTED, RULE, NAVY = bp.INK, bp.INK2, bp.MUTED, bp.RULE, bp.NAVY
fr, esc, svg_text = bp.fr, bp.esc, bp.svg_text

OOS_START = "2002-04-01"
FAMILIES = {  # column in states.parquet -> label, published balanced accuracy, kappa, mean run
    "A' sparse jump": ("A′ Sparse Jump Model", 93.2, 0.53, 456),
    "A  jump": ("A Jump Model", 93.3, 0.49, 456),
    "B  filtered HMM": ("B HMM filtré", 84.7, 0.24, 128),
    "C  gradient boost": ("C Gradient boosting", 75.0, 0.12, 18),
    "C' HAR-RV": ("C′ HAR-RV", 78.2, 0.17, 14),
}


# ---------------------------------------------------------------------------
# live figures, checked against the published ones
# ---------------------------------------------------------------------------
def check_layer1() -> dict[str, dict]:
    """Recompute the classification table from the published states."""
    states = pd.read_parquet(CACHE / "states.parquet")
    nber = store.read("references", "ref_nber").set_index("period")["value"]
    out = {}
    for column, (label, ba_pub, kappa, run) in FAMILIES.items():
        s = states[column].loc[OOS_START:].dropna()
        rec = nber.reindex(s.index.to_period("M").to_timestamp()).to_numpy() == 1.0
        stress = (s == 0.0).to_numpy()
        recall, spec = stress[rec].mean(), (~stress[~rec]).mean()
        ba = 100 * (recall + spec) / 2
        if abs(ba - ba_pub) > 0.051:
            raise SystemExit(f"{label}: balanced accuracy {ba:.2f} recomputed, {ba_pub} published. Not building.")
        years = (s.index[-1] - s.index[0]).days / 365.25
        out[column] = {
            "label": label, "ba": ba, "recall": 100 * recall, "spec": 100 * spec, "kappa": kappa,
            "run": run, "switches": int((s.diff().abs() > 0).sum()), "years": years,
            "stress": 100 * stress.mean(), "precision": 100 * rec[stress].mean(),
            "n": len(s), "rec_days": int(rec.sum()), "rec_hits": int(stress[rec].sum()),
            "start": s.index[0], "end": s.index[-1],
        }
    a = out["A' sparse jump"]
    expected = {"n": 6377, "switches": 13, "rec_days": 435, "rec_hits": 419}
    for key, want in expected.items():
        if a[key] != want:
            raise SystemExit(f"A': {key} = {a[key]}, expected {want}. Not building.")
    print(f"   check A' {a['ba']:.2f} % on {a['n']} sessions, {a['switches']} switches, recall {a['rec_hits']}/{a['rec_days']}")
    return out


def sparsity() -> tuple[int, int, int, int]:
    """(min, max) non-zero weights and (min, max) top-10 share, from the committed artefact."""
    text = (ROOT / "docs" / "artifacts" / "sjm_sparsity.txt").read_text(encoding="utf-8")
    m = re.search(r"non-zero weights: (\d+) to (\d+) of 50; the ten largest carry (\d+)% to (\d+)%", text)
    if not m:
        raise SystemExit("docs/artifacts/sjm_sparsity.txt has no summary line: run scripts/measure_sjm_sparsity.py")
    return tuple(int(g) for g in m.groups())  # type: ignore[return-value]


def n_configs() -> int:
    return int(pd.read_parquet(ROOT / "data" / "trials.parquet")["config_hash"].nunique())


# ---------------------------------------------------------------------------
# figures, drawn for an A4 column (viewBox ~700 wide, text 11-13 units)
# ---------------------------------------------------------------------------
def svg_open(width: int, height: int, label: str) -> str:
    return f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{esc(label)}">'


def gridline(x: float, top: float, bottom: float, zero: bool = False) -> str:
    return (f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="{GREY if zero else RULE}" '
            f'stroke-width="{1.4 if zero else 1}"/>')


def fig_accuracy(l1: dict) -> str:
    others = sorted((r for k, r in l1.items() if k != "A' sparse jump"), key=lambda r: -r["ba"])
    rows = [l1["A' sparse jump"], *others]
    width, row, label_w, right = 700, 34, 170, 190
    height = row * len(rows) + 30
    plot = width - label_w - right

    def x(v: float) -> float:
        return label_w + plot * v / 100

    out = [svg_open(width, height, "Exactitude équilibrée face aux récessions NBER, par modèle")]
    for v in range(0, 101, 25):
        out.append(gridline(x(v), 2, height - 24, v == 0))
        out.append(svg_text(x(v), height - 11, f"{v} %", size=11, fill=MUTED, anchor="middle"))
    xc = x(50)
    out.append(f'<line x1="{xc:.1f}" y1="2" x2="{xc:.1f}" y2="{height - 24}" stroke="{MUTED}" stroke-dasharray="4 4"/>')
    for k, r in enumerate(rows):
        yc = 4 + k * row + row / 2 - 2
        lead = r["label"].startswith("A′")
        w = 600 if lead else 400
        out.append(svg_text(label_w - 10, yc, esc(r["label"]), size=13, anchor="end", weight=w))
        out.append(f'<rect x="{label_w}" y="{yc - 11:.1f}" width="{x(r["ba"]) - label_w:.1f}" height="22" rx="3" fill="{BLUE if lead else GREY}"/>')
        out.append(svg_text(x(r["ba"]) + 8, yc, fr(r["ba"], ".1f") + " %", size=13, weight=w))
        out.append(svg_text(x(r["ba"]) + 70, yc, "κ " + fr(r["kappa"], ".2f"), size=12, fill=INK2))
    out.append("</svg>")
    return "".join(out)


def fig_timeline() -> str:
    states = pd.read_parquet(CACHE / "states.parquet")["A' sparse jump"].loc[OOS_START:].dropna()
    spx = p2.b1.us_prices("eq_us_large")
    spx.index = pd.to_datetime(spx.index)
    spx = spx.loc[states.index.min():states.index.max()].resample("W-FRI").last().dropna()
    width, height, left, right, top, bottom = 700, 250, 44, 8, 22, 24
    t0, t1 = states.index.min(), states.index.max()
    lo, hi = math.log(spx.min() * 0.9), math.log(spx.max() * 1.08)

    def x(d) -> float:
        return left + (pd.Timestamp(d) - t0).days / (t1 - t0).days * (width - left - right)

    def y(v: float) -> float:
        return top + (hi - math.log(v)) / (hi - lo) * (height - top - bottom)

    out = [svg_open(width, height, "S&P 500 et états de stress du Sparse Jump Model, 2002-2026")]
    seg = (states != states.shift()).cumsum()
    for _, g in states.groupby(seg):
        if g.iloc[0] == 0:
            x0, x1 = x(g.index[0]), x(g.index[-1])
            out.append(f'<rect x="{x0:.1f}" y="{top}" width="{max(x1 - x0, 1.5):.1f}" height="{height - top - bottom}" fill="{RED}" fill-opacity="0.18"/>')
    for level in (1000, 2000, 4000):
        yy = y(level)
        out.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}" stroke="{RULE}"/>')
        out.append(svg_text(left - 6, yy, f"{level:,}".replace(",", " "), size=11, fill=MUTED, anchor="end"))
    for year in range(2004, 2027, 4):
        out.append(svg_text(x(f"{year}-01-01"), height - 10, str(year), size=11, fill=MUTED, anchor="middle"))
    pts = " ".join(f"{x(d):.1f},{y(v):.1f}" for d, v in spx.items())
    out.append(f'<polyline points="{pts}" fill="none" stroke="{INK}" stroke-width="1.3"/>')
    for label, start in (("2002-03", "2002-04-01"), ("2007-09", "2007-11-22"), ("2020-21", "2020-03-11")):
        out.append(svg_text(x(start) + 2, top - 10, label, size=11, fill=INK2, weight=600))
    low = spx.loc["2022"]
    xx, yy = x(low.idxmin()), y(low.min())
    out.append(f'<line x1="{xx:.1f}" y1="{yy + 5:.1f}" x2="{xx:.1f}" y2="{yy + 22:.1f}" stroke="{INK2}"/>')
    out.append(svg_text(xx + 5, yy + 26, "2022 : −25 %,", size=11, fill=INK2))
    out.append(svg_text(xx + 5, yy + 39, "non signalée", size=11, fill=INK2))
    out.append("</svg>")
    return "".join(out)


def fig_r2() -> str:
    rows = [("A′ Sparse Jump Model", 3.93, 0.030, True), ("A Jump Model", 3.47, 0.022, False),
            ("B HMM filtré", 2.26, 0.007, False), ("C Gradient boosting", 2.13, 0.023, False),
            ("C′ HAR-RV", 0.19, 0.008, False), ("Témoin de volatilité", 0.004, 0.000, False)]
    width, row, label_w, right = 700, 42, 170, 70
    height = row * len(rows) + 30
    plot = width - label_w - right

    def x(v: float) -> float:
        return label_w + plot * v / 4.0

    out = [svg_open(width, height, "R² incrémental sur la volatilité et sur les rendements futurs")]
    for v in range(5):
        out.append(gridline(x(v), 2, height - 24, v == 0))
        out.append(svg_text(x(v), height - 11, f"{v} pt", size=11, fill=MUTED, anchor="middle"))
    for k, (label, vol, ret, lead) in enumerate(rows):
        y0 = 4 + k * row
        w = 600 if lead else 400
        out.append(svg_text(label_w - 10, y0 + 17, esc(label), size=13, anchor="end", weight=w))
        wv, wr = max(x(vol) - label_w, 1.5), max(x(ret) - label_w, 1.5)
        out.append(f'<rect x="{label_w}" y="{y0 + 3}" width="{wv:.1f}" height="15" rx="2" fill="{BLUE}"/>')
        out.append(svg_text(label_w + wv + 6, y0 + 11, fr(vol, "+.2f" if vol >= 0.01 else "+.3f"), size=12, weight=w))
        out.append(f'<rect x="{label_w}" y="{y0 + 20}" width="{wr:.1f}" height="15" rx="2" fill="{ORANGE}"/>')
        out.append(svg_text(label_w + wr + 6, y0 + 28, fr(ret, "+.3f"), size=12, fill=INK2))
    out.append("</svg>")
    return "".join(out)


def fig_covid() -> str:
    spx = p2.b1.us_prices("eq_us_large")
    spx.index = pd.to_datetime(spx.index)
    spx = spx.loc["2019-11-01":"2021-06-30"]
    st = p2.state_series().reindex(spx.index).ffill()
    width, height, left, right, top, bottom = 700, 240, 44, 10, 14, 24
    t0, t1 = spx.index.min(), spx.index.max()
    lo, hi = math.log(2100), math.log(4400)

    def x(d) -> float:
        return left + (pd.Timestamp(d) - t0).days / (t1 - t0).days * (width - left - right)

    def y(v: float) -> float:
        return top + (hi - math.log(v)) / (hi - lo) * (height - top - bottom)

    out = [svg_open(width, height, "S&P 500 pendant le Covid et état du Sparse Jump Model")]
    stress = st.eq(0.0)
    seg = (stress != stress.shift()).cumsum()
    for _, g in stress.groupby(seg):
        if bool(g.iloc[0]):
            x0, x1 = x(g.index[0]), x(g.index[-1])
            out.append(f'<rect x="{x0:.1f}" y="{top}" width="{max(x1 - x0, 1.5):.1f}" height="{height - top - bottom}" fill="{RED}" fill-opacity="0.18"/>')
    for level in (2500, 3000, 3500, 4000):
        yy = y(level)
        out.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}" stroke="{RULE}"/>')
        out.append(svg_text(left - 6, yy, f"{level:,}".replace(",", " "), size=11, fill=MUTED, anchor="end"))
    for d, lab in (("2020-01-01", "janv. 2020"), ("2020-07-01", "juil. 2020"), ("2021-01-01", "janv. 2021"), ("2021-06-01", "juin 2021")):
        out.append(svg_text(x(d), height - 10, lab, size=11, fill=MUTED, anchor="middle"))
    pts = " ".join(f"{x(d):.1f},{y(v):.1f}" for d, v in spx.items())
    out.append(f'<polyline points="{pts}" fill="none" stroke="{INK}" stroke-width="1.5"/>')
    marks = [("2020-02-19", "1", -15, 0), ("2020-03-11", "2", 0, 15), ("2020-03-23", "3", 15, 0),
             ("2020-08-05", "4", -16, 0), ("2021-04-05", "5", -16, 0)]
    for d, num, dy, dx in marks:
        dd = pd.Timestamp(d)
        v = spx.loc[:dd].iloc[-1]
        out.append(f'<circle cx="{x(dd):.1f}" cy="{y(v):.1f}" r="3" fill="{INK}"/>')
        cx, cy = x(dd) + dx, y(v) + dy
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8.5" fill="{BLUE}" stroke="#fff" stroke-width="1.2"/>')
        out.append(svg_text(cx, cy + 0.5, num, size=11, fill="#fff", anchor="middle", weight=700))
    out.append("</svg>")
    return "".join(out)


def fig_bars(rows: list[tuple[str, float, bool]], lo: float, hi: float, ticks: list[float],
             label: str, *, mark: tuple[float, str] | None = None, spec: str = "+.3f") -> str:
    width, row, label_w, right = 700, 32, 200, 40
    height = row * len(rows) + (48 if mark else 30)
    top = 20 if mark else 4
    plot = width - label_w - right

    def x(v: float) -> float:
        return label_w + (v - lo) / (hi - lo) * plot

    out = [svg_open(width, height, label)]
    for v in ticks:
        out.append(gridline(x(v), top - 2, height - 24, v == 0))
        out.append(svg_text(x(v), height - 11, fr(v, "+.2f") if v else "0", size=11, fill=MUTED, anchor="middle"))
    if mark:
        xm = x(mark[0])
        out.append(f'<line x1="{xm:.1f}" y1="{top - 4}" x2="{xm:.1f}" y2="{height - 24}" stroke="{INK2}" stroke-dasharray="5 4" stroke-width="1.3"/>')
        out.append(svg_text(xm, 8, mark[1], size=11, fill=INK2, anchor="middle"))
    for k, (lab, v, lead) in enumerate(rows):
        yc = top + k * row + row / 2
        out.append(svg_text(label_w - 10, yc, esc(lab), size=13, anchor="end", weight=600 if lead else 400))
        a, b = sorted((x(0), x(v)))
        out.append(f'<rect x="{a:.1f}" y="{yc - 10:.1f}" width="{max(b - a, 1.5):.1f}" height="20" rx="3" fill="{BLUE if lead else GREY}"/>')
        tx, anchor, fill = (b + 6, "start", INK) if v >= 0 else (a - 6, "end", INK)
        if mark and v >= 0 and b - 4 < x(mark[0]) < b + 52:
            tx, anchor, fill = b - 6, "end", INK
        out.append(svg_text(tx, yc, fr(v, spec), size=12, anchor=anchor, fill=fill, weight=600 if lead else 400))
    out.append("</svg>")
    return "".join(out)


def fig_stress_calm() -> str:
    width, row, label_w = 700, 50, 170
    names = list(p2.STRESS_CALM)
    height = row * len(names) + 30
    lo, hi = -1.5, 4.2
    plot = width - label_w - 50

    def x(v: float) -> float:
        return label_w + (v - lo) / (hi - lo) * plot

    out = [svg_open(width, height, "Sharpe de chaque stratégie seule, en calme et en stress")]
    for v in (-1, 0, 1, 2, 3, 4):
        out.append(gridline(x(v), 2, height - 24, v == 0))
        out.append(svg_text(x(v), height - 11, fr(v, "+.0f") if v else "0", size=11, fill=MUTED, anchor="middle"))
    for k, name in enumerate(names):
        s, c = p2.STRESS_CALM[name]
        y0 = 4 + k * row
        out.append(svg_text(label_w - 10, y0 + 20, esc(name), size=13, anchor="end", weight=600))
        for j, (v, col) in enumerate(((c, BLUE), (s, ORANGE))):
            yy = y0 + 3 + j * 19
            a, b = sorted((x(0), x(v)))
            out.append(f'<rect x="{a:.1f}" y="{yy}" width="{max(b - a, 1.5):.1f}" height="15" rx="2" fill="{col}"/>')
            tx, anchor = (b + 6, "start") if v >= 0 else (a - 6, "end")
            out.append(svg_text(tx, yy + 8, fr(v, "+.2f"), size=12, anchor=anchor))
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------------------
# page furniture
# ---------------------------------------------------------------------------
def tbl(head: list[str], rows: list[list[str]], *, num: set[int] = frozenset(), cls: str = "",
        lead: int | None = None, widths: list[int] | None = None) -> str:
    cols = "<colgroup>" + "".join(f'<col style="width:{w}%">' for w in widths) + "</colgroup>" if widths else ""
    th = "".join(f'<th class="{"num" if i in num else ""}">{h}</th>' for i, h in enumerate(head))
    body = []
    for k, r in enumerate(rows):
        tr_cls = ' class="lead"' if k == lead else ""
        if len(r) == 1:
            body.append(f'<tr class="group"><td colspan="{len(head)}">{r[0]}</td></tr>')
            continue
        tds = "".join(f'<td class="{"num" if i in num else ""}">{c}</td>' for i, c in enumerate(r))
        body.append(f"<tr{tr_cls}>{tds}</tr>")
    return f'<table class="{cls}">{cols}<thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table>'


class Counter:
    def __init__(self) -> None:
        self.fig = 0
        self.tab = 0

    def figure(self, svg: str, caption: str, source: str, cls: str = "") -> str:
        self.fig += 1
        return (f'<figure class="{cls}">{svg}<figcaption><b>Figure {self.fig}.</b> {caption}'
                f'<span class="src">Source : {source}</span></figcaption></figure>')

    def table(self, caption: str, body: str, source: str, note: str = "") -> str:
        self.tab += 1
        note_html = f'<p class="tnote">{note}</p>' if note else ""
        return (f'<div class="tablewrap"><p class="tcap"><b>Tableau {self.tab}.</b> {caption}</p>{body}'
                f'{note_html}<p class="tsrc">Source : {source}</p></div>')


def pct(v: float, spec: str = ".1f") -> str:
    return fr(v, spec) + " %"


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "\u00a0")


def french_typography(html: str) -> str:
    """Non-breaking spaces where French typography wants them, in text only."""
    chunks = re.split(r"(<svg.*?</svg>)", html, flags=re.S)
    for j, chunk in enumerate(chunks):
        if chunk.startswith("<svg"):
            continue
        parts = re.split(r"(<[^>]+>)", chunk)
        for i, part in enumerate(parts):
            if part.startswith("<"):
                continue
            part = re.sub(r" ([%:;?!»])", "\u00a0\\1", part)
            part = part.replace("« ", "«\u00a0")
            part = re.sub(r"(\d) (\d{3})\b", "\\1\u00a0\\2", part)
            part = part.replace(" = ", "\u00a0=\u00a0")
            parts[i] = part
        chunks[j] = "".join(parts)
    return "".join(chunks)


CSS = """
@font-face { font-family: "NbspFix"; src: local("Helvetica Neue"), local("HelveticaNeue"), local("Arial"); unicode-range: U+00A0; }
@page { size: A4; margin: 19mm 18mm 18mm 18mm;
  @bottom-left { content: "Détecter les crises de marché · dossier de projet"; font: 7.5pt "Helvetica Neue", Arial, sans-serif; color: #898781; }
  @bottom-right { content: counter(page) " / " counter(pages); font: 7.5pt "Helvetica Neue", Arial, sans-serif; color: #898781; } }
@page cover { margin: 0; @bottom-left { content: none; } @bottom-right { content: none; } }
@page :first { margin: 0; @bottom-left { content: none; } @bottom-right { content: none; } }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body { font: 9.6pt/1.46 "NbspFix", Charter, "Iowan Old Style", Georgia, serif; color: #1b1f24;
  -webkit-print-color-adjust: exact; print-color-adjust: exact; hyphens: manual; }
svg text { font-family: "Helvetica Neue", Arial, sans-serif; }
.formula { text-align: center; font: 11.5pt "STIX Two Text", "STIXGeneral", "Times New Roman", serif; margin: 2mm 0 3mm; }
.cover { page: cover; height: 297mm; position: relative; background: #fff; break-after: page; overflow: hidden; }
.cover .band { position: absolute; left: 0; top: 0; height: 297mm; width: 14mm; background: #0f2a44; }
.cover .inner { position: absolute; left: 32mm; right: 22mm; top: 42mm; }
.cover .kicker { font: 600 9pt "Avenir Next", "Helvetica Neue", sans-serif; letter-spacing: .14em; text-transform: uppercase; color: #2a78d6; }
.cover h1 { font: 600 30pt/1.12 "Avenir Next", "Helvetica Neue", sans-serif; color: #0f2a44; margin: 7mm 0 6mm; letter-spacing: -.01em; }
.cover .sub { font: 12.5pt/1.45 "NbspFix", Charter, Georgia, serif; color: #52514e; max-width: 140mm; }
.cover .authors { margin-top: 16mm; font: 600 12pt "Avenir Next", "Helvetica Neue", sans-serif; color: #0f2a44; }
.cover .date { font: 10pt "Avenir Next", "Helvetica Neue", sans-serif; color: #52514e; margin-top: 1.5mm; }
.cover .repo { margin-top: 12mm; border-left: 3px solid #2a78d6; padding: 3mm 0 3mm 5mm; font: 9.5pt/1.5 "Helvetica Neue", Arial, sans-serif; color: #1b1f24; max-width: 150mm; }
.cover .repo a { color: #0f2a44; font-weight: 600; text-decoration: none; }
.cover .toc { position: absolute; left: 32mm; right: 22mm; bottom: 24mm; columns: 2; column-gap: 10mm;
  font: 8.8pt/1.6 "Helvetica Neue", Arial, sans-serif; color: #52514e; border-top: 1px solid #e1e0d9; padding-top: 4mm; }
.cover .toc b { color: #0f2a44; font-weight: 600; }
h1.sec { font: 600 15pt/1.2 "Avenir Next", "Helvetica Neue", sans-serif; color: #0f2a44; margin: 8mm 0 3.5mm;
  padding-top: 1mm; border-top: 2px solid #0f2a44; break-after: avoid; }
h1.sec.newpage { margin-top: 0; }
h1.sec .n { color: #2a78d6; margin-right: 2.5mm; }
.newpage { break-before: page; }
h2 { font: 600 10.8pt/1.3 "Avenir Next", "Helvetica Neue", sans-serif; color: #0f2a44; margin: 4.5mm 0 1.5mm; break-after: avoid; }
p { margin: 0 0 2.2mm; }
ul, ol { margin: 0 0 2.5mm; padding-left: 5mm; }
li { margin-bottom: 1.1mm; }
b { font-weight: 600; }
a { color: #0f2a44; }
code { font: 8.3pt "SF Mono", Menlo, monospace; color: #0f2a44; background: #f3f2ee; padding: 0 1px; border-radius: 2px; }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4mm; margin: 3mm 0 4.5mm; }
.kpi { border-top: 2.5px solid #2a78d6; padding-top: 2mm; }
.kpi .v { font: 600 17pt "Avenir Next", "Helvetica Neue", sans-serif; color: #2a78d6; line-height: 1.1; }
.kpi .l { font: 600 8.4pt/1.3 "Helvetica Neue", Arial, sans-serif; color: #0f2a44; margin-top: 1mm; }
.kpi .d { font: 7.8pt/1.35 "Helvetica Neue", Arial, sans-serif; color: #52514e; margin-top: 1mm; }
.box { background: #f1f5fa; border-left: 3px solid #0f2a44; padding: 3mm 4mm; margin: 3mm 0 3.5mm; break-inside: avoid; }
.box.warn { background: #fbf4ec; border-left-color: #eb6834; }
.box h3 { font: 600 9.4pt "Avenir Next", "Helvetica Neue", sans-serif; color: #0f2a44; margin: 0 0 1.5mm; }
.box p, .box li { font-size: 9pt; margin-bottom: 1.2mm; }
.box p:last-child, .box ul:last-child, .box ol:last-child { margin-bottom: 0; }
.tablewrap { margin: 3mm 0 4mm; break-inside: avoid; }
.tcap { font: 8.6pt/1.35 "Helvetica Neue", Arial, sans-serif; color: #1b1f24; margin: 0 0 1.2mm; text-align: left; }
.tnote, .tsrc { font: 7.4pt/1.35 "Helvetica Neue", Arial, sans-serif; color: #52514e; margin: 1.2mm 0 0; text-align: left; }
.tsrc { color: #898781; }
table { width: 100%; border-collapse: collapse; font: 8.1pt/1.3 "Helvetica Neue", Arial, sans-serif; font-variant-numeric: tabular-nums; }
th { text-align: left; font-weight: 600; color: #0f2a44; border-bottom: 1.2px solid #0f2a44; padding: 1.2mm 1.5mm; vertical-align: bottom; }
td { border-bottom: 0.6px solid #e1e0d9; padding: 1.1mm 1.5mm; vertical-align: top; }
td.num, th.num { text-align: right; }
td.num { white-space: nowrap; }
td code { overflow-wrap: anywhere; font-size: 7.4pt; }
tr.lead td { background: #eef4fb; }
tr.lead td:first-child { font-weight: 600; }
tr.group td { font-weight: 600; color: #0f2a44; background: #f6f5f1; }
table.small { font-size: 7.4pt; }
table.small td, table.small th { padding: 0.8mm 1.3mm; }
td.pos { background: #dbe9fa; } td.neg { background: #fbe3e2; } td.flat { background: #f1f0ec; }
td .lvl { color: #52514e; font-weight: 400; margin-left: 1.5mm; }
figure { margin: 3mm 0 4mm; break-inside: avoid; }
figure svg { display: block; }
figure.narrow svg { width: 82%; margin: 0 auto; }
figcaption { font: 8.2pt/1.35 "Helvetica Neue", Arial, sans-serif; color: #1b1f24; margin-top: 1.5mm; }
figcaption .src { display: block; color: #898781; font-size: 7.3pt; margin-top: 0.5mm; }
.legend { font: 7.8pt "Helvetica Neue", Arial, sans-serif; color: #52514e; margin: 0 0 1mm; }
.legend i { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin: 0 1.2mm 0 3mm; vertical-align: 0; }
.legend i:first-child { margin-left: 0; }
.marks { font: 7.8pt/1.5 "Helvetica Neue", Arial, sans-serif; color: #1b1f24; display: grid; grid-template-columns: 1fr 1fr; gap: 0 5mm; margin-top: 1mm; }
.marks b { display: inline-block; width: 4mm; height: 4mm; border-radius: 50%; background: #2a78d6; color: #fff; text-align: center; font-size: 6.8pt; line-height: 4mm; margin-right: 1.5mm; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 6mm; }
.refs p { text-indent: -5mm; padding-left: 5mm; margin-bottom: 1.1mm; text-align: left; font-size: 8.4pt; line-height: 1.38; }
.refs h2 { margin-top: 3mm; }
tr { break-inside: avoid; }
.annex h1.sec { border-top-color: #898781; }
.annex h1.sec .n { color: #52514e; }
.small-p { font-size: 8.6pt; }
"""


def build_html() -> str:
    print("checking the published figures against the data")
    l1 = check_layer1()
    umd, bond = p2.umd_arms(), p2.b1_arms()
    extra = p2.check(umd, bond)
    nz_lo, nz_hi, sh_lo, sh_hi = sparsity()
    configs = n_configs()
    a, b = l1["A' sparse jump"], l1["B  filtered HMM"]
    rec_share = 100 * a["rec_days"] / a["n"]
    c = Counter()
    s: list[str] = []

    # ------------------------------------------------------------------ cover
    s.append(f"""
<section class="cover"><div class="band"></div><div class="inner">
  <div class="kicker">Projet Big Data · Master 2 · Dossier de projet</div>
  <h1>Détecter les crises de marché&nbsp;:<br>est-ce que ça rapporte&nbsp;?</h1>
  <div class="sub">Cinq modèles de régimes comparés sur 24 ans de données publiques, hors échantillon et sous un
  protocole fixé avant les résultats, puis le meilleur appliqué comme filtre à trois stratégies de trading.</div>
  <div class="authors">Guillaume Beaudouin · Gabriel Golivet</div>
  <div class="date">25 septembre 2026</div>
  <div class="repo">Code, protocole, sorties brutes et historique complet&nbsp;:<br>
  <a href="{REPO_URL}">{REPO_URL.replace("https://", "")}</a><br>
  Version de référence de ce dossier&nbsp;: étiquette <b>{TAG}</b>. Tous les chiffres renvoient à un
  fichier du dépôt, cité sous chaque tableau et chaque figure.</div>
</div>
<div class="toc">
  <b>Résumé</b><br><b>1</b> La question et la démarche<br><b>2</b> Les données<br><b>3</b> Les modèles et le protocole<br>
  <b>4</b> Résultat 1 : il reconnaît les crises<br><b>5</b> Résultat 2 : le risque, pas la direction<br>
  <b>6</b> Application : trois stratégies<br><b>7</b> Pourquoi le filtre ne paie pas<br><b>8</b> Limites<br>
  <b>9</b> Conclusion et ouverture<br><b>10</b> Reproduire les résultats<br><b>Bibliographie</b><br>
  <b>Annexes</b> A. Les 50 variables · B. Écarts au protocole · C. Méthode statistique · D. Le test sur 90 ans · E. Les autres études
</div></section>""")

    # ------------------------------------------------------------------ summary
    s.append(f"""
<h1 class="sec">Résumé</h1>
<p>Les marchés alternent des périodes calmes et des périodes de crise. Nous avons posé deux questions. Un modèle
peut-il reconnaître la crise <b>pendant</b> qu'elle se produit, sans connaître la suite&nbsp;? Et cette
reconnaissance permet-elle de <b>mieux investir</b>&nbsp;? Les modèles, les règles et les critères de décision ont
été écrits et gelés avant le premier résultat&nbsp;; chaque étude ultérieure a eu son protocole commité avant
sa lecture.</p>
<div class="kpis">
  <div class="kpi"><div class="v">{fr(a['ba'], '.1f')} %</div><div class="l">d'exactitude face aux récessions officielles</div>
  <div class="d">Sparse Jump Model, {thousands(a['n'])} séances hors échantillon (avril 2002 – septembre 2026). Sur deux récessions seulement.</div></div>
  <div class="kpi"><div class="v">+3,93 pts</div><div class="l">de R² sur la volatilité à venir</div>
  <div class="d">au-delà d'une règle de volatilité passée (t = −3,40)&nbsp;; +0,20 seulement une fois le VIX ajouté.</div></div>
  <div class="kpi"><div class="v">+0,03 pt</div><div class="l">de R² sur les rendements à venir</div>
  <div class="d">t = 0,27&nbsp;: aucune information sur la direction du marché.</div></div>
  <div class="kpi"><div class="v">0 / 63</div><div class="l">usages du filtre démontrés utiles</div>
  <div class="d">arrêt, réduction, bascule, valeurs refuges, options, sur 13 stratégies.</div></div>
</div>
<ol>
  <li><b>Le classifieur fonctionne, dans un cadre étroit.</b> Parmi cinq modèles comparés sous un protocole
  identique, le Sparse Jump Model est le plus stable ({a['switches']} changements d'état en 24 ans) et le plus en
  accord avec les récessions (κ = 0,53). Mais la période de test ne contient que deux récessions&nbsp;; réestimée
  sur 1926-2026 avec les 30 variables qui existent depuis 1926, la même méthode n'en reconnaît que 6 sur 14.</li>
  <li><b>Il mesure le risque, pas la direction.</b> L'état prévoit l'ampleur des mouvements des semaines
  suivantes, pas leur sens. Et le marché des options, à travers le VIX, contient déjà presque toute cette
  information.</li>
  <li><b>Comme filtre de trading, il ne paie pas de façon démontrable.</b> Sur trois stratégies aux profils
  opposés, l'effet dépend de la stratégie&nbsp;: il aide le momentum actions (Sharpe 0,47 → 0,70 en coupant en
  stress, 0,77 avec de l'or à la place), laisse le rebond obligataire de fin de mois inchangé et pénalise la
  tendance crypto (1,13 → 0,89). Aucun gain ne dépasse le seuil de détection, une règle de volatilité d'une
  ligne fait aussi bien, et sur 90 ans le meilleur cas tombe à +0,045.</li>
</ol>
<p><b>Pourquoi.</b> Le modèle entre en stress tard (Covid&nbsp;: le 11 mars 2020, S&amp;P 500 déjà à −19 %, VIX passé de
14 à 54) et y reste pendant la reprise (sortie définitive le 5 avril 2021, +82 % au-dessus du point bas). Une
protection gagne donc dans la chute et reperd dans le rebond&nbsp;; seule une stratégie qui perd précisément dans
les rebonds, comme le momentum, en profite.</p>
<div class="box warn"><h3>Ce que ce dossier ne prétend pas</h3><ul>
  <li>Les Sharpe de la partie application (§6-7) sont <b>sans coûts de transaction</b>, par hypothèse du cours&nbsp;:
  ce ne sont pas des performances investissables.</li>
  <li>«&nbsp;Sans voir le futur&nbsp;» ne veut pas dire «&nbsp;avant le marché&nbsp;».</li>
  <li>Un écart inférieur au seuil de détection est «&nbsp;sous-puissant&nbsp;» — ni un oui, ni un non — et jamais
  «&nbsp;validé&nbsp;».</li>
</ul></div>""")

    # ------------------------------------------------------------------ 1 question
    s.append(f"""
<h1 class="sec newpage"><span class="n">1</span>La question et la démarche</h1>
<p>La question de recherche a été écrite avant toute mesure&nbsp;: <i>«&nbsp;Un modèle de régimes appris par machine
apporte-t-il une information exploitable au-delà de ce qu'une simple mesure de volatilité capte déjà&nbsp;?&nbsp;»</i>
Le témoin est volontairement modeste&nbsp;: si un modèle sophistiqué ne fait pas mieux qu'une règle d'une ligne,
il ne sert à rien.</p>
<h2>La démarche, en quatre temps</h2>
<ol>
  <li>Construire <b>50 indicateurs quotidiens</b> depuis 1990, chacun calculé avec l'information disponible ce
  jour-là (§2).</li>
  <li>Comparer <b>cinq modèles</b> sous un protocole identique, réestimés tous les six mois sur le seul passé (§3).</li>
  <li>Les juger en <b>trois couches</b>&nbsp;: classification (reconnaissent-ils les récessions&nbsp;?), information
  (prévoient-ils la volatilité ou les rendements au-delà du témoin&nbsp;?), portefeuille (améliorent-ils un
  60/40&nbsp;?) (§4-5).</li>
  <li>Appliquer le meilleur modèle comme <b>filtre</b> à des stratégies réelles (§6-7).</li>
</ol>
<h2>Les règles du protocole</h2>
<ul>
  <li><b>Pré-enregistrement.</b> Le cadrage (<code>docs/CHARTER.html</code>) a été gelé et scellé par empreinte
  SHA-256 avant le premier résultat. Chaque étude ultérieure a son protocole et son critère de décision
  commités avant la lecture, qui n'a lieu qu'une fois. Les écarts sont consignés, datés, dans
  <code>docs/PROTOCOL_FREEZE.md</code> (annexe B).</li>
  <li><b>Pas de regard vers l'avenir.</b> Signal calculé à la clôture de T−1, position prise en T. États
  <b>filtrés</b> (connus le jour même), jamais lissés.</li>
  <li><b>Hors échantillon.</b> Walk-forward à fenêtre croissante&nbsp;: 49 réestimations semestrielles d'avril 2002 à
  avril 2026, chacune sur les seules données antérieures.</li>
  <li><b>Un témoin partout.</b> «&nbsp;Stress si la volatilité réalisée sur 21 séances dépasse sa médiane
  passée&nbsp;», et sa variante au 80ᵉ centile.</li>
  <li><b>Statistiques.</b> Rendements en excès du taux sans risque à 3 mois&nbsp;; erreurs-types HAC (6 retards)&nbsp;;
  seuils de détection par bootstrap stationnaire par blocs&nbsp;; placebo&nbsp;; correction pour tests multiples&nbsp;;
  journal de tous les essais ({configs} configurations distinctes au 25 septembre 2026). Détails en annexe C.</li>
</ul>
<div class="box"><h3>Comment lire un verdict</h3>
<p><b>Seuil de détection (MDE).</b> Le plus petit écart de Sharpe que l'échantillon permet de distinguer du
hasard, au risque corrigé du nombre de tests de l'étude.</p>
<p><b>UTILE</b>&nbsp;: écart ≥ seuil, t HAC de même signe, au-dessus de 95 % du placebo, et meilleur que la même
règle pilotée par chacun des deux témoins de volatilité. <b>SOUS-PUISSANT</b>&nbsp;: écart positif mais sous le
seuil — ni oui ni non. <b>PAS UTILE</b>&nbsp;: écart nul ou négatif, sans l'être assez pour être mesuré.
<b>NUISIBLE</b>&nbsp;: écart ≤ −seuil, confirmé par le t et le placebo.</p></div>""")

    # ------------------------------------------------------------------ 2 data
    data_rows = [
        ["Marchés", "S&amp;P 500, Russell 2000, Nikkei 225, indice dollar, VIX, taux à 10 ans&nbsp;; pétrole WTI", "Yahoo Finance, FRED", "cours de clôture, connus le jour même"],
        ["Activité et prix", "Production industrielle, emplois non agricoles, taux de chômage, prix à la consommation", "ALFRED (FRED)", "<b>première publication</b>, à sa date réelle de parution"],
        ["Emploi hebdomadaire, conditions financières", "Inscriptions au chômage, indice NFCI de la Fed de Chicago", "FRED", "valeur actuelle, décalée de son délai de publication (révisions non neutralisées)"],
        ["Crédit et taux", "Écarts Baa et Aaa contre le 10 ans, pentes 10 ans – 2 ans et 10 ans – 3 mois, taux à 3 mois", "FRED", "séries de marché, jamais révisées"],
        ["Actions, en coupe", "49 secteurs, 25 portefeuilles taille × valeur, facteurs de Fama et French", "Ken French Data Library", "rendements quotidiens"],
        ["Référence", "Dates des récessions américaines (NBER, série USREC)", "FRED", "<b>jamais vue par les modèles</b>&nbsp;: sert seulement à les noter"],
    ]
    fam_rows = [
        ["Tendance", "7", "Rendement du S&amp;P 500 sur 1, 3 et 12 mois, accélération, pétrole et dollar sur 3 mois, part des marchés au-dessus de leur moyenne à 200 jours"],
        ["Volatilité", "9", "Volatilité réalisée sur 5, 21 et 63 séances, structure par terme, volatilité de la volatilité, ratio des semi-variances, part des sauts, VIX, prime de variance"],
        ["Asymétrie et mémoire", "6", "Asymétrie et aplatissement, distance au plus haut sur un an, exposant de Hurst, ratios de variance à 5 et 20 jours"],
        ["Coupe transversale", "8", "Dispersion entre secteurs et entre portefeuilles, corrélation moyenne, ratio d'absorption et sa variation, part des secteurs en hausse, facteurs taille et valeur"],
        ["Macroéconomie", "9", "Croissance de la production, de l'emploi et des prix (1 an et 3 mois), variation du chômage, règle de Sahm, inscriptions au chômage"],
        ["Crédit", "4", "Écart Baa, écart Baa − Aaa, et leurs variations sur 3 mois"],
        ["Conditions financières", "2", "NFCI et sa variation sur 13 semaines"],
        ["Taux", "5", "Pentes 10 ans – 2 ans et 10 ans – 3 mois, leurs variations, variation du taux à 3 mois sur un an"],
    ]
    s.append(f"""
<h1 class="sec"><span class="n">2</span>Les données</h1>
<p>Toutes les données sont publiques et quotidiennes (hebdomadaires ou mensuelles pour la macroéconomie, reportées
au jour). Elles couvrent 1990-2026. Un <b>contrat point-in-time</b> gouverne leur usage&nbsp;: une valeur n'entre
dans un modèle qu'à partir du jour où elle était réellement connue.</p>
{c.table("Les sources, et la façon dont chacune respecte le contrat point-in-time.", tbl(["Bloc", "Séries", "Source", "Traitement dans le temps"], data_rows, widths=[18, 38, 14, 30]), "regime_lab/data/universe.py ; inventaire fichier par fichier dans AVANCEMENT.md §3.")}
<h2>50 indicateurs, huit familles</h2>
<p>La littérature de référence n'utilise que trois indicateurs, tous tirés des rendements, donc tous de la
volatilité&nbsp;: un modèle ainsi nourri ne peut que redécouvrir un quantile de volatilité. Nous avons donc ajouté
des familles qui n'en sont pas (coupe transversale, crédit, taux, macroéconomie). La liste complète, avec les
définitions, est en annexe A.</p>
{c.table("Les 50 indicateurs, par famille.", tbl(["Famille", "Nombre", "Contenu"], fam_rows, num={1}), "regime_lab/features/ ; data/cache/features.parquet.")}
<h2>Standardisation</h2>
<p>Chaque indicateur est centré et réduit avec <b>son seul passé</b> (moyenne et écart-type sur une fenêtre qui
s'allonge, 252 séances au minimum), puis borné à ±5 écarts-types. La première ligne complète date du 2 mars 1992.
Les modèles ne voient donc jamais une moyenne ou une dispersion calculée avec des données futures.</p>
<div class="box warn"><h3>Les fuites résiduelles, déclarées</h3>
<p>Quatre séries macroéconomiques sont prises en première publication (ALFRED). Les inscriptions au chômage et le
NFCI, eux, sont pris dans leur version actuelle, décalée de leur délai de publication&nbsp;: leurs révisions
ultérieures ne sont pas neutralisées. Les séries de Yahoo et de Ken French peuvent aussi être révisées après
coup. Enfin, la jambe actions du portefeuille 60/40 de référence est l'indice de prix du S&amp;P 500, sans
dividendes&nbsp;: son rendement est sous-estimé d'environ 2 points par an, de la même façon dans toutes les
variantes comparées.</p></div>""")

    # ------------------------------------------------------------------ 3 models
    model_rows = [
        ["<b>A′ Sparse Jump Model</b>", "Regroupe les jours semblables en deux états, comme un k-moyennes, et fait payer une pénalité λ à chaque changement d'état&nbsp;; pondère les variables sous une contrainte L1", "2 états&nbsp;; λ dans {1, 3, 10, 30, 100, 300}&nbsp;; nombre effectif de variables&nbsp;: 10", "Modèle principal"],
        ["A Jump Model", "Même principe, sans pondération des variables", "2 états&nbsp;; même grille de λ", "Variante"],
        ["B HMM filtré", "Chaîne de Markov cachée, lois gaussiennes à covariance diagonale estimées par Baum-Welch&nbsp;; probabilité d'état <b>filtrée</b> (récursion avant, sans lissage)", "2 états&nbsp;; 200 itérations", "Référence classique, celle des articles du cours"],
        ["C Gradient boosting", "Pas d'état caché&nbsp;: prédit la volatilité réalisée des 21 séances suivantes du 60/40&nbsp;; «&nbsp;stress&nbsp;» si la prévision dépasse sa médiane d'entraînement", "LightGBM, 300 arbres, pas 0,05, 15 feuilles", "Défi de l'apprentissage automatique"],
        ["C′ HAR-RV", "Régression linéaire de la même cible sur les volatilités passées à 1, 5 et 21 séances&nbsp;; même seuil", "3 termes", "Référence de la littérature (Corsi, 2009)"],
    ]
    s.append(f"""
<h1 class="sec"><span class="n">3</span>Les modèles et le protocole</h1>
<p>Deux familles à état caché (A, A′, B) et deux prédicteurs directs de la volatilité (C, C′). Les seconds sont
<b>supervisés</b>&nbsp;: ils apprennent sur le passé à prévoir la volatilité future, que les premiers ne voient
jamais. C'est voulu&nbsp;: la comparaison dit si la notion d'«&nbsp;état caché&nbsp;» apporte quelque chose.</p>
{c.table("Les cinq modèles comparés.", tbl(["Modèle", "Principe", "Réglages", "Rôle"], model_rows, lead=0, widths=[15, 45, 22, 18]), "regime_lab/models/ ; scripts/run_phase2.py.")}
<h2>Le modèle principal&nbsp;: le Sparse Jump Model</h2>
<p>Le modèle (Nystrup, Kolm et Lindström, 2021&nbsp;; Aydınhan, Kolm, Mulvey et Shu, 2024) choisit une suite d'états
<i>s<sub>t</sub></i>, des centres <i>μ</i> et des poids de variables <i>w</i> qui minimisent</p>
<p class="formula">∑<sub><i>t</i></sub> ‖<i>x<sub>t</sub></i> − <i>μ</i><sub><i>s<sub>t</sub></i></sub>‖²<sub><i>w</i></sub>
&nbsp;+&nbsp; <i>λ</i> ∑<sub><i>t</i></sub> 𝟙{{<i>s<sub>t</sub></i> ≠ <i>s</i><sub><i>t</i>−1</sub>}},
&nbsp;&nbsp; avec ‖<i>w</i>‖<sub>2</sub> = 1 et ‖<i>w</i>‖<sub>1</sub> ≤ √10</p>
<ul>
  <li><b>Regrouper.</b> Chaque jour est affecté à l'état dont le profil moyen lui ressemble le plus.</li>
  <li><b>Stabiliser.</b> La pénalité λ fait payer chaque changement d'état&nbsp;: les régimes durent et le signal ne
  clignote pas.</li>
  <li><b>Sélectionner.</b> La contrainte L1 fixe à 10 le <b>nombre effectif</b> de variables,
  (Σ<i>w</i>)²/Σ<i>w</i>². En pratique, {nz_lo} à {nz_hi} des 50 variables gardent un poids non nul selon la
  fenêtre, et les dix premières portent {sh_lo} à {sh_hi} % du poids (<code>docs/artifacts/sjm_sparsity.txt</code>).</li>
</ul>
<h2>Le protocole commun</h2>
<ul>
  <li><b>Réestimation.</b> Les cinq modèles sont réestimés aux mêmes dates, tous les six mois, sur une fenêtre qui
  commence en 1992 et s'allonge&nbsp;: 49 réestimations. Entre deux, l'état est calculé chaque jour en ligne.</li>
  <li><b>Choix de λ.</b> Sur la seule fenêtre d'entraînement, λ maximise le Sharpe du portefeuille de référence
  piloté par les états, parmi les candidats qui changent d'état entre 0,5 et 12 fois par an. Il est recalculé
  toutes les quatre réestimations (tous les deux ans). Quand aucun candidat n'est admissible, le code retient
  20, la médiane de la grille, qui n'en fait pas partie&nbsp;: c'est le cas pour A′ depuis avril 2022 (écart déclaré,
  annexe B).</li>
  <li><b>Ordre des états.</b> L'état «&nbsp;stress&nbsp;» est celui dont la volatilité d'entraînement est la plus
  forte.</li>
  <li><b>Portefeuille de référence.</b> 60 % S&amp;P 500, 40 % obligation à 10 ans, rééquilibré chaque mois, en
  excès du taux à 3 mois. Coût de 2 points de base aller-retour dans l'étude principale.</li>
  <li><b>Règle de position.</b> Fixée d'avance&nbsp;: investi dans l'état calme, sans position en stress. Une règle de
  taille, proportionnelle à l'inverse de la volatilité de l'état, a été ajoutée ensuite et déclarée comme telle.</li>
</ul>""")

    # ------------------------------------------------------------------ 4 result 1
    fam_order = ["A' sparse jump", "A  jump", "B  filtered HMM", "C  gradient boost", "C' HAR-RV"]
    l1_rows = []
    for k in fam_order:
        r = l1[k]
        l1_rows.append([r["label"], pct(r["ba"]), pct(r["recall"]), pct(r["spec"]), fr(r["kappa"], ".2f"),
                        fr(r["switches"] / r["years"], ".1f" if r["switches"] / r["years"] >= 1 else ".2f"),
                        pct(r["stress"]), pct(r["precision"], ".0f"), f"{r['run']}"])
    s.append(f"""
<h1 class="sec"><span class="n">4</span>Résultat 1&nbsp;: il reconnaît les crises</h1>
<p>Chaque séance hors échantillon est notée contre la chronologie officielle des récessions américaines (NBER),
que les modèles ne voient jamais. L'<b>exactitude équilibrée</b> est la moyenne de deux taux&nbsp;: la part des
séances de récession classées en stress (rappel) et la part des séances normales classées en calme
(spécificité). Les deux comptent autant, alors que les récessions ne représentent que {fr(rec_share, '.1f')} % des
séances. 50 % correspond au hasard.</p>
{c.figure(fig_accuracy(l1), "Exactitude équilibrée face aux récessions NBER, avec le κ de Cohen (0 = hasard, 1 = accord parfait). Pointillé : le hasard.", "recalculé depuis data/cache/states.parquet ; docs/RESULTS_FINAL.md (couche 1).", "narrow")}
{c.table(f"Qualité de classification, {thousands(a['n'])} séances du {a['start']:%d/%m/%Y} au {a['end']:%d/%m/%Y}.", tbl(["Modèle", "Exactitude équilibrée", "Rappel", "Spécificité", "κ", "Changements / an", "Temps en stress", "Stress en récession", "Durée moy. d'un état (séances)"], l1_rows, num=set(range(1, 9)), lead=0, cls="small", widths=[19, 11, 8, 10, 6, 11, 10, 11, 14]),
    "recalculé depuis data/cache/states.parquet contre la série USREC ; κ et durées : docs/RESULTS_FINAL.md.",
    note="« Stress en récession » : part des séances de stress qui tombent pendant une récession officielle. Changements par année civile.")}
<ul>
  <li><b>Le Sparse Jump Model est le plus stable et le plus en accord avec les récessions</b> (κ = 0,53). Il met
  en stress {a['rec_hits']} des {a['rec_days']} séances de récession et laisse en calme {fr(a['spec'], '.0f')} % des
  séances normales. Le Jump Model simple fait jeu égal en exactitude (93,3 %), avec un κ plus faible.</li>
  <li><b>Le HMM ne manque aucune récession, mais crie souvent au loup</b>&nbsp;: il est en stress {fr(b['stress'], '.0f')} %
  du temps et seules {fr(b['precision'], '.0f')} % de ses séances de stress tombent en récession. Il n'est pas en
  retard&nbsp;: au Covid, il bascule le 27 février 2020 (S&amp;P 500 à −12 %), deux semaines avant le Sparse Jump
  Model.</li>
  <li><b>Les prédicteurs directs sont nerveux</b>&nbsp;: 14 à 18 changements d'état par an, 75 à 78 % d'exactitude.</li>
</ul>
{c.figure(fig_timeline(), "Le S&P 500 (échelle logarithmique) et, en rouge, les séances où le Sparse Jump Model est en stress. Treize changements d'état en 24 ans : l'après-bulle internet (tronqué au début de l'échantillon), la crise financière, le Covid.", "data/cache/states.parquet ; cours Yahoo Finance.")}
<div class="box warn"><h3>Trois réserves, sans lesquelles le 93 % trompe</h3><ul>
  <li><b>Deux récessions seulement</b> dans la période de test (2008-2009 et 2020, 20 mois). Sur 1926-2026, la même
  méthode réduite à 30 variables n'en reconnaît que 6 sur 14 (57,5 %, κ = 0,16), pas mieux qu'une règle de
  volatilité (annexe D).</li>
  <li><b>«&nbsp;Sans voir le futur&nbsp;» n'est pas «&nbsp;avant le marché&nbsp;».</b> Contre les dates officielles, la
  latence de détection est de 0 à 13 jours&nbsp;; contre le marché, le modèle arrive tard (§7).</li>
  <li><b>2022 n'a pas été signalée</b>&nbsp;: une baisse de 25 % due aux taux, lente, sans panique. Le modèle est
  calme depuis avril 2021.</li>
</ul></div>""")

    # ------------------------------------------------------------------ 5 result 2
    l2_rows = [["A′ Sparse Jump Model", "+3,93", "−3,40", "+0,030", "0,27"],
               ["A Jump Model", "+3,47", "−3,46", "+0,022", "0,24"],
               ["B HMM filtré", "+2,26", "−4,59", "+0,007", "−0,22"],
               ["C Gradient boosting", "+2,13", "−6,34", "+0,023", "0,52"],
               ["C′ HAR-RV", "+0,19", "−1,84", "+0,008", "−0,30"],
               ["Témoin de volatilité", "+0,004", "0,21", "+0,000", "−0,03"]]
    l3_rows = [["60/40 seul", "0,47", "−35,6 %", "—", "—"],
               ["A′ Sparse Jump Model", "0,55", "−22,3 %", "0,49", "−31,2 %"],
               ["A Jump Model", "0,54", "−22,3 %", "0,50", "−29,3 %"],
               ["B HMM filtré", "0,46", "−13,7 %", "0,49", "−30,4 %"],
               ["C Gradient boosting", "0,42", "−10,2 %", "0,50", "−32,5 %"],
               ["C′ HAR-RV", "0,50", "−13,4 %", "0,53", "−33,5 %"]]
    s.append(f"""
<h1 class="sec"><span class="n">5</span>Résultat 2&nbsp;: le risque, pas la direction</h1>
<p>On régresse, séance par séance, la volatilité réalisée des 21 séances suivantes du portefeuille 60/40 (puis
son rendement) sur le rang de la volatilité passée — le témoin — et sur l'état du modèle. Le <b>R² incrémental</b>
est ce que l'état ajoute au témoin.</p>
<div class="legend"><i style="background:{BLUE}"></i>volatilité des 21 séances suivantes<i style="background:{ORANGE}"></i>rendement des 21 séances suivantes</div>
{c.figure(fig_r2(), "R² incrémental au-delà du témoin, en points de pourcentage.", "docs/RESULTS_FINAL.md (couche 2).", "narrow")}
{c.table("R² incrémental et t HAC de l'état, par cible.", tbl(["Modèle", "Volatilité : R² (pts)", "t", "Rendement : R² (pts)", "t"], l2_rows, num={1, 2, 3, 4}, lead=0, cls="small"), "docs/RESULTS_FINAL.md (couche 2).")}
<ul>
  <li><b>Sur la volatilité</b>, quatre modèles sur cinq ajoutent 2 à 4 points de R² à ce que le témoin sait déjà
  (t de −3,4 à −6,3). Le résultat résiste aux effets fixes par pli de cinq ans (3,68 points, t = −3,51) et au
  retrait de n'importe quel pli (jamais sous 1,74 point, t = −2,42).</li>
  <li><b>Sur les rendements</b>, aucun modèle n'ajoute quoi que ce soit (t entre −0,3 et 0,5). L'état est une
  information de <b>dimensionnement</b>, pas de timing.</li>
  <li><b>Mais le VIX le savait déjà.</b> Dans un test séparé, commité avant sa lecture (test P, 6 129 séances), on
  prévoit le logarithme de la variance réalisée du S&amp;P 500 sur 21 séances. L'état y ajoute +4,48 points au-delà
  d'un rang de volatilité (t = −5,23), mais seulement <b>+0,20 point</b> une fois la variance implicite du VIX
  ajoutée (t = −1,34, seuil |t| ≥ 2,77). Le marché des options contient déjà l'information.</li>
</ul>
<h2>Au niveau d'un portefeuille, rien n'est décidable</h2>
{c.table("Le portefeuille 60/40 piloté par chaque modèle, avril 2002 – septembre 2026, net de 2 points de base par aller-retour.", tbl(["Modèle", "Tout ou rien : Sharpe net", "Perte maximale", "Taille : Sharpe net", "Perte maximale"], l3_rows, num={1, 2, 3, 4}, lead=1, cls="small"), "docs/RESULTS_FINAL.md (couche 3) ; docs/RESULTS_FALSIFICATION.md.",
    note="Les onze lignes de l'étude sont publiées, y compris celles qui ne favorisent pas le modèle principal.")}
<p>Toutes les améliorations restent sous le seuil de détection de cette comparaison, 0,271 à 0,399 de Sharpe. Les
contrôles de falsification confirment&nbsp;: aucun modèle ne bat le témoin de volatilité au niveau de la stratégie
(+0,04 à +0,06 de Sharpe, contre des seuils de 0,22 à 0,44), et le gain apparent passe par le dénominateur —
moins de risque, pas plus de rendement.</p>""")

    # ------------------------------------------------------------------ 6 application
    methods = ["Couper", "Réduire de moitié", "Basculer vers la tendance", "Or", "Obligations (TLT)", "Options"]
    base = {"Momentum actions": p2.UMD_ROWS["Seule"][0], "Rebond obligataire": p2.B1_ROWS["Seule"][0], "Tendance crypto": p2.CRYPTO_ROWS["Seule"][0]}
    levels = {
        "Momentum actions": {"Couper": p2.UMD_ROWS["Couper"][0], "Réduire de moitié": extra["umd_half"][0], "Basculer vers la tendance": None,
                             "Or": p2.UMD_ROWS["Or"][0], "Obligations (TLT)": p2.UMD_ROWS["Obligations (TLT)"][0], "Options": p2.UMD_ROWS["Options"][0]},
        "Rebond obligataire": {"Couper": p2.B1_ROWS["Couper"][0], "Réduire de moitié": extra["b1_reduce"][0], "Basculer vers la tendance": extra["b1_switch"][0],
                               "Or": p2.B1_ROWS["Or"][0], "Obligations (TLT)": p2.B1_ROWS["Obligations (TLT)"][0], "Options": p2.B1_ROWS["Options"][0]},
        "Tendance crypto": {m: p2.CRYPTO_ROWS[m][0] for m in methods},
    }
    recomputed = {("Momentum actions", "Réduire de moitié"), ("Rebond obligataire", "Réduire de moitié"), ("Rebond obligataire", "Basculer vers la tendance")}
    head = "<th>En stress, on…</th>" + "".join(f'<th class="num">{k}<br><span style="font-weight:400;color:#52514e">seule : {fr(v, ".2f")}</span></th>' for k, v in base.items())
    body = []
    for m in methods:
        cells = [f"<td>{m}</td>"]
        for strat in base:
            v = levels[strat][m]
            if v is None:
                cells.append('<td class="num" style="color:#898781">non testé</td>')
                continue
            d = p2.DELTAS[strat][methods.index(m)]
            cls = "pos" if d >= 0.05 else ("neg" if d <= -0.05 else "flat")
            mark = " °" if (strat, m) in recomputed else ""
            cells.append(f'<td class="num {cls}"><b>{fr(d, "+.2f")}</b><span class="lvl">({fr(v, ".2f")}){mark}</span></td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    matrix = f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'
    vrows = []
    for strat, rows in p2.VERDICTS.items():
        vrows.append([f"{strat}"])
        vrows.extend(list(r) for r in rows)
    strat_rows = [
        ["Momentum actions", "Achète les actions américaines qui ont le plus monté sur un an, vend celles qui ont le plus baissé (facteur UMD de Ken French)", "oct. 2002 – juil. 2026", "0,47", "public"],
        ["Rebond obligataire de fin de mois", "Acheteur d'obligations d'État américaines longues (ETF TLT) les 3 dernières séances de chaque mois&nbsp;: un effet de flux lié aux rééquilibrages", "oct. 2004 – sept. 2026", "0,85", "public"],
        ["Tendance crypto", "Suit la tendance du bitcoin et de l'ether&nbsp;: acheteur quand elle monte, sans position sinon", "déc. 2014 – juin 2026", "1,13", "dépôt privé voisin&nbsp;: agrégats seulement, non reproductible depuis le dépôt public"],
    ]
    s.append(f"""
<h1 class="sec"><span class="n">6</span>Application&nbsp;: trois stratégies</h1>
<p>Le modèle prévoit l'ampleur des mouvements, pas leur sens. La question devient&nbsp;: quelles stratégies ont
besoin d'un thermomètre du risque&nbsp;? Nous avons retenu trois stratégies aux profils opposés face aux crises, et
six façons d'utiliser le filtre quand le Sparse Jump Model est en stress (état connu la veille).</p>
{c.table("Les trois stratégies. Chacune est ciblée à 10 % de volatilité annuelle, en excès du taux sans risque.", tbl(["Stratégie", "Principe", "Période", "Sharpe seule", "Données"], strat_rows, num={3}), "docs/artifacts/partie2/lectures_trois_strategies.txt.")}
<p><b>Les six usages du filtre en stress</b>&nbsp;: <b>couper</b> la stratégie, la <b>réduire de moitié</b>, la
<b>remplacer</b> par un livre de tendance multi-actifs (46 marchés), par de l'<b>or</b>, par des <b>obligations
d'État longues</b> (TLT) ou par des <b>options</b> (volatilité achetée&nbsp;: un swap de variance synthétique à un mois,
équivalent d'un straddle couvert en delta). Chaque jambe de remplacement est elle aussi ciblée à 10 % de
volatilité.</p>
<p><b>Hypothèse de coûts&nbsp;: aucun coût, partout</b> (décision du cours). Les Sharpe sont bruts, en excès du taux
sans risque&nbsp;; ils surestiment ce qu'un investisseur obtiendrait.</p>
{c.table("Écart de Sharpe avec la stratégie seule (et, entre parenthèses, le Sharpe obtenu). Un échantillon par colonne, celui du tableau 7.", matrix, "docs/artifacts/partie2/lectures_trois_strategies.txt ; lignes ° recalculées sans coût par scripts/build_presentation_partie2.py et vérifiées par ce dossier.",
    note="Bleu : gain d'au moins 0,05 ; rouge : perte d'au moins 0,05 ; gris : entre les deux. Les écarts sont calculés avant arrondi. Aucune case n'est un résultat démontré : voir le tableau suivant.")}
{c.table("Chaque usage, testé avec un critère écrit avant la lecture.", tbl(["Usage en stress", "Δ Sharpe", "Seuil (MDE)", "t HAC", "Placebo", "Témoins (médiane / 80ᵉ)", "Verdict"], vrows, num={1, 2, 3, 4}, cls="small"),
    "docs/artifacts/partie2/lectures_trois_strategies.txt ; docs/artifacts/crise/reading.txt ; docs/RESULTS_B1_COUPLAGE.md.",
    note="* Test de l'étude de crise, sur avril 2002 – juillet 2026 (momentum seul : 0,52) ; les autres lignes du momentum portent sur octobre 2002 – juillet 2026 (0,47), d'où l'écart avec le tableau précédent (+0,23). † Tests à 1 point de base de coût (rebond seul : 0,79). Placebo : part des 400 placements aléatoires des séances de stress que le vrai filtre bat. Témoins : écart de Sharpe du même usage piloté par la règle de volatilité (médiane passée, puis 80ᵉ centile).")}
<h2>Ce qu'on lit</h2>
<ul>
  <li><b>C'est la stratégie qui décide, pas la méthode.</b> Le momentum s'améliore avec toutes les méthodes sauf
  les options&nbsp;; le rebond obligataire ne bouge presque pas&nbsp;; la tendance crypto se dégrade partout.</li>
  <li><b>Le meilleur cas du programme&nbsp;: momentum et or en stress, 0,47 → 0,77</b> (t = +2,68, meilleur que les 400
  placebos). Mais l'écart reste sous le seuil (0,62), l'essentiel vient du simple arrêt (0,70), et la règle de
  volatilité médiane fait autant (0,82 avec de l'or).</li>
  <li><b>Le stress n'est pas un mauvais moment pour toutes les stratégies.</b> Le momentum y perd (Sharpe −1,13 en
  stress contre 0,75 en calme)&nbsp;; la tendance crypto y gagne le plus (3,89 contre 0,93)&nbsp;: la couper en stress, c'est
  couper ses meilleurs jours.</li>
</ul>
<div class="legend"><i style="background:{BLUE}"></i>Sharpe en calme<i style="background:{ORANGE}"></i>Sharpe en stress</div>
{c.figure(fig_stress_calm(), "Sharpe de chaque stratégie seule, selon l'état du modèle la veille (descriptif).", "docs/artifacts/partie2/lectures_trois_strategies.txt.", "narrow")}""")

    # ------------------------------------------------------------------ 7 why
    lh = fig_bars([("1937-1962", -0.008, False), ("1963-2001", 0.031, False), ("2002-2026", 0.145, False), ("1937-2026, en tout", 0.045, True)],
                  -0.05, 0.20, [0.0, 0.05, 0.10, 0.15, 0.20], "Gain de Sharpe en coupant le momentum en stress, par période",
                  mark=(0.158, "seuil de détection : 0,158"))
    s.append(f"""
<h1 class="sec"><span class="n">7</span>Pourquoi le filtre ne paie pas</h1>
{c.figure(fig_covid(), "Le S&P 500 pendant le Covid et, en rouge, les séances de stress du Sparse Jump Model.", "data/cache/states.parquet ; cours Yahoo Finance ; docs/presentation/PISTES_AMELIORATION.md §0.")}
<div class="marks" style="margin-top:-2mm;margin-bottom:3mm">
  <span><b>1</b>sommet du marché, 19 février 2020</span><span><b>2</b>le modèle passe en stress, 11 mars (−19 %)</span>
  <span><b>3</b>point bas, 23 mars (−34 %)</span><span><b>4</b>première sortie, 5 août (+49 % depuis le point bas)</span>
  <span><b>5</b>sortie définitive, 5 avril 2021 (+82 %)</span></div>
<ol>
  <li><b>Il alerte tard.</b> Le 11 mars 2020, le VIX était déjà passé de 14 à 54. Le HMM, plus nerveux, avait
  basculé le 27 février.</li>
  <li><b>Il reste en crise pendant la reprise.</b> 97 % de ses séances de stress du Covid tombent le jour du point
  bas ou après&nbsp;; 55 % sur les trois épisodes de stress de la période. C'est le prix d'un modèle calme, qui ne
  change d'avis que rarement&nbsp;: il sort de la crise tard.</li>
  <li><b>Le marché le savait déjà</b>&nbsp;: au-delà du VIX, l'état n'ajoute presque rien à la prévision du risque
  (§5).</li>
</ol>
<p><b>Conséquence.</b> Une couverture gagne dans la chute puis reperd dans le rebond. Acheter de la volatilité en
stress gagne de 29 à 38 % pendant la chute de 2008 selon la stratégie remplacée, puis perd de 18 à 23 % au rebond
de 2009. Seule une stratégie qui perd <b>précisément</b> dans les rebonds, comme le momentum, en profite.</p>
<h2>L'épreuve des 90 ans</h2>
<p>Pour sortir des deux récessions, le même type de modèle a été réestimé sur 1926-2026 avec les 30 variables qui
existent depuis 1926 (sans VIX ni macroéconomie en première publication)&nbsp;: 14 récessions hors échantillon au
lieu de 2, et un test 2,5 fois plus précis (seuil 0,158 au lieu de 0,395).</p>
{c.figure(lh, "Gain de Sharpe du momentum quand on le coupe en stress, modèle réduit réestimé sur 1926-2026.", "docs/RESULTS_LONGHIST.md ; docs/artifacts/longhist/reading.txt.", "narrow")}
<p>Le gain vient presque entièrement de 2002-2026&nbsp;: le modèle réduit y gagne +0,145 (A′ complet&nbsp;: +0,17). Sur 90
ans, il vaut <b>+0,045, trois fois et demie sous le seuil</b>, et il passe par une baisse du risque, pas par un
gain de rendement (rendement annuel 12,19 % → 11,80 %). Une règle publiée fait mieux sans modèle&nbsp;: «&nbsp;marché
baissier et forte volatilité&nbsp;» (Daniel et Moskowitz, 2016) donne +0,106 et ramène la perte maximale de −37 % à
−27 %.</p>
<h2>Ce que disent les 63 usages testés</h2>
<p>Couper, réduire, basculer, or, obligations, options, sur 13 stratégies&nbsp;: <b>aucun usage n'est utile</b>. Les
meilleurs sont sous-puissants, et une règle de volatilité d'une ligne fait aussi bien. La portée exacte de ce
constat compte&nbsp;: il porte sur un classifieur à deux états, ordonné par la volatilité, qui change d'état environ
une fois tous les deux ans, utilisé pour couper, réduire ou remplacer une stratégie. Il ne dit rien d'un état
plus rapide, d'un état non lié à la volatilité, ni d'un usage en sélection de signaux ou en construction de
portefeuille.</p>
<p><b>À noter, et ce n'est pas un résultat de régime</b>&nbsp;: un livre de neuf stratégies peu corrélées, à risque
égal et sans aucun filtre, fait un Sharpe brut de 1,52 sur 2005-2026 (intervalle à 95 %&nbsp;: 1,06 à 1,96). C'est un
plafond&nbsp;: sans coût, avec des stratégies choisies en connaissant ces années, dont sept sur neuf vivent dans le
dépôt privé voisin. Utilisé comme budget de risque à l'intérieur de ce livre, le régime n'apporte rien de
démontrable (<code>docs/RESULTS_BUDGET_RISQUE.md</code>).</p>""")

    # ------------------------------------------------------------------ 8 limits
    s.append("""
<h1 class="sec"><span class="n">8</span>Limites</h1>
<ol>
  <li><b>Très peu de crises.</b> Deux récessions et trois épisodes de stress en 24 ans&nbsp;; les stratégies récentes
  n'en voient qu'un (2020). Le 93 % et la plupart des verdicts reposent sur ces quelques épisodes, et le test sur
  90 ans ne confirme pas le 93 %.</li>
  <li><b>Une référence datée après coup.</b> Les dates du NBER sont publiées six à dix-huit mois après les faits.
  Elles servent à noter les modèles, jamais à les entraîner&nbsp;; seuls les états des modèles sont «&nbsp;en temps
  réel&nbsp;».</li>
  <li><b>Des données imparfaitement point-in-time</b>&nbsp;: révisions non stockées, deux séries hebdomadaires en
  version actuelle, séries de marché susceptibles de révision, 60/40 sans dividendes (§2).</li>
  <li><b>Des écarts au protocole gelé</b>, tous déclarés (annexe B). Les plus importants&nbsp;: le sous-ensemble de
  variables commun à toutes les familles, prévu par le cadrage, n'a jamais été mis en œuvre, si bien que la
  comparaison A′ contre B mêle le modèle et la sélection des variables&nbsp;; le Jump Model continu prévu n'a jamais
  été estimé&nbsp;; λ vaut 20, hors de la grille déclarée, pour les neuf dernières réestimations d'A′&nbsp;; deux des cinq
  plis de cinq ans d'A′ ne contiennent qu'un état, ce qui rend incalculable la règle d'arrêt «&nbsp;trois plis sur
  cinq&nbsp;»&nbsp;; la règle d'arrêt n°2 (R² incrémental sur les rendements sous 0,2 point) s'est déclenchée et l'étude
  a continué sur la volatilité.</li>
  <li><b>Une application sans coûts.</b> Les Sharpe de la partie application sont bruts&nbsp;; la tendance crypto et
  sept des neuf stratégies du livre sont privées, publiées en agrégats seulement.</li>
  <li><b>Une puissance statistique faible.</b> Les seuils de détection vont de 0,1 à 1,4 de Sharpe selon les tests.
  Beaucoup de verdicts sont donc «&nbsp;sous-puissants&nbsp;»&nbsp;: ce n'est ni un oui ni un non.</li>
  <li><b>Des tests multiples à l'échelle du programme.</b> Chaque étude corrige pour ses propres tests (Bonferroni
  ou Holm)&nbsp;; aucune correction ne couvre l'ensemble des configurations journalisées du programme. Un résultat
  positif isolé devrait donc être lu avec encore plus de prudence&nbsp;; aucun usage du régime n'a d'ailleurs été jugé
  utile.</li>
  <li><b>Une portée étroite</b>&nbsp;: un classifieur lent, à deux états, ordonné par la volatilité, utilisé comme
  interrupteur ou multiplicateur de taille (§7).</li>
</ol>""")

    # ------------------------------------------------------------------ 9 conclusion
    s.append("""
<h1 class="sec"><span class="n">9</span>Conclusion et ouverture</h1>
<p><b>Détecter une crise ne suffit pas à gagner.</b></p>
<ol>
  <li><b>On sait détecter les crises</b>, avec un modèle stable et lisible, sans voir le futur — sur les deux
  récessions de la période, et sans devancer le marché.</li>
  <li><b>Le modèle prévoit le risque, pas la direction</b>&nbsp;: c'est un thermomètre, utile pour dimensionner, pas
  pour acheter ou vendre. Et le VIX en sait déjà presque autant.</li>
  <li><b>Comme filtre de trading, son effet dépend de la stratégie</b>, et aucune méthode ne gagne de façon
  démontrée. Il faudrait une stratégie qui perd précisément quand il sonne — dans les rebonds.</li>
</ol>
<h2>Avec plus de temps, trois pistes</h2>
<ul>
  <li><b>Un régime propre à chaque marché</b> (actions, or, crypto…), plutôt qu'un régime unique. Piste&nbsp;: un
  modèle de langage qui lit l'actualité propre à chaque actif.</li>
  <li><b>Prévoir les retournements plutôt que l'ampleur.</b> Par exemple, un modèle de durée des marchés haussiers
  et baissiers (Lunde et Timmermann, 2004), qui vise la fin d'un marché baissier.</li>
  <li><b>Des données alternatives</b>, plus rares et moins accessibles&nbsp;: par exemple les données de transactions
  qui suivent les ventes d'une enseigne comme Chipotle avant ses résultats officiels.</li>
</ul>""")

    # ------------------------------------------------------------------ 10 reproduce
    repro = [
        ["Les 50 indicateurs", "scripts/fetch_data.py, scripts/build_features.py", "data/cache/features.parquet"],
        ["Les états des cinq modèles (49 réestimations, 15 à 20 min)", "scripts/run_phase2.py", "data/cache/states.parquet"],
        ["Couches 1 et 2", "scripts/run_evaluation.py", "docs/RESULTS_FINAL.md"],
        ["Couche 3", "scripts/run_layer3.py", "docs/RESULTS_FINAL.md"],
        ["Contrôles de falsification T1, T3, T5", "scripts/run_t1_control.py, run_t3_control.py, run_t5_refit.py, run_t5_control.py", "docs/RESULTS_FALSIFICATION.md"],
        ["Stratégies de crise et test P", "scripts/run_crisis_coupling.py", "docs/artifacts/crise/reading.txt"],
        ["Valeurs refuges", "scripts/run_safe_haven_switch.py", "docs/artifacts/refuge/reading.txt"],
        ["Rebond obligataire", "scripts/run_b1_regime_coupling.py", "docs/RESULTS_B1_COUPLAGE.md"],
        ["Test sur 90 ans", "scripts/longhist_fetch.py, longhist_fit.py, longhist_validate.py, longhist_umd.py", "docs/artifacts/longhist/"],
        ["Parcimonie du Sparse Jump Model", "scripts/measure_sjm_sparsity.py", "docs/artifacts/sjm_sparsity.txt"],
        ["Ce dossier et les deux supports", "scripts/build_dossier.py, build_presentation.py, build_presentation_partie2.py", "docs/presentation/"],
    ]
    s.append(f"""
<h1 class="sec"><span class="n">10</span>Reproduire les résultats</h1>
<p>Tout le programme tient dans un seul dépôt public, avec son historique&nbsp;: <a href="{REPO_URL}">{REPO_URL.replace("https://", "")}</a>,
version de référence <b>{TAG}</b>. Il faut Python 3.12 et uv.</p>
<p><code>git clone {REPO_URL}.git</code><br>
<code>cd regime-lab &amp;&amp; git checkout {TAG}</code><br>
<code>uv sync --all-packages --extra dev</code><br>
<code>.venv/bin/python -m pytest -q</code> (849 tests, tous verts au 25 septembre 2026)</p>
<p>Les données ne sont pas versionnées (une cinquantaine de mégaoctets, régénérables)&nbsp;; une clé API FRED gratuite suffit à les
télécharger. L'inventaire exact, fichier par fichier, est dans <code>AVANCEMENT.md</code> §3. Les lectures des
études ont été faites une seule fois&nbsp;: leurs sorties intégrales sont commitées dans <code>docs/artifacts/</code>, et
le journal des essais est <code>data/trials.parquet</code>.</p>
{c.table("Où trouver chaque résultat.", tbl(["Résultat", "Script", "Sortie"], [[r[0], f"<code>{r[1]}</code>", f"<code>{r[2]}</code>"] for r in repro], cls="small", widths=[22, 40, 38]), "dépôt regime-lab.",
    note="Non reproductibles depuis le dépôt public : la tendance crypto et les sept stratégies privées du livre de neuf, qui vivent dans le dépôt privé voisin et ne sont publiées qu'en agrégats.")}""")

    # ------------------------------------------------------------------ bibliography
    s.append("""
<div class="refs">
<h1 class="sec">Bibliographie</h1>
<h2>Articles de référence du cours</h2>
<p>Mamon, R. S. et Elliott, R. J. (dir.) (2014). <i>Hidden Markov Models in Finance: Further Developments and Applications, Volume II</i>. Springer, International Series in Operations Research &amp; Management Science.</p>
<p>Nguyen, N. (2018). Hidden Markov Model for Stock Trading. <i>International Journal of Financial Studies</i>, 6(2), 36.</p>
<p>Sidhu, G. S., Metwaly, A. I. A., Tiwari, A. et Bhattacharyya, R. (2021). Short Term Trading Models Using Hurst Exponent and Machine Learning. SSRN, document 3824032.</p>
<p>Zhang, M., Jiang, X., Fang, Z., Zeng, Y. et Xu, K. (2019). High-order Hidden Markov Model for trend prediction in financial time series. <i>Physica A</i>, 517, 1-12.</p>
<h2>Modèles de régimes</h2>
<p>Aydınhan, A. O., Kolm, P. N., Mulvey, J. M. et Shu, Y. (2024). Identifying patterns in financial markets: extending the statistical jump model for regime identification. <i>Annals of Operations Research</i>.</p>
<p>Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. <i>Econometrica</i>, 57(2), 357-384.</p>
<p>Nystrup, P., Kolm, P. N. et Lindström, E. (2021). Feature selection in jump models. <i>Expert Systems with Applications</i>, 184, 115558.</p>
<p>Nystrup, P., Lindström, E. et Madsen, H. (2020). Learning hidden Markov models with persistent states by penalizing jumps. <i>Expert Systems with Applications</i>, 150, 113307.</p>
<p>Shu, Y. et Mulvey, J. M. (2024). Dynamic factor allocation leveraging regime-switching signals. arXiv 2410.14841.</p>
<p>Shu, Y., Yu, C. et Mulvey, J. M. (2024). Downside risk reduction using regime-switching signals: a statistical jump model approach. <i>Journal of Asset Management</i>.</p>
<p>Witten, D. M. et Tibshirani, R. (2010). A framework for feature selection in clustering. <i>Journal of the American Statistical Association</i>, 105(490), 713-726.</p>
<h2>Modèles de comparaison et indicateurs</h2>
<p>Barndorff-Nielsen, O. E. et Shephard, N. (2004). Power and bipower variation with stochastic volatility and jumps. <i>Journal of Financial Econometrics</i>, 2(1), 1-37.</p>
<p>Corsi, F. (2009). A simple approximate long-memory model of realized volatility. <i>Journal of Financial Econometrics</i>, 7(2), 174-196.</p>
<p>Ke, G. et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. <i>Advances in Neural Information Processing Systems</i>, 30.</p>
<p>Kritzman, M., Li, Y., Page, S. et Rigobon, R. (2011). Principal components as a measure of systemic risk. <i>Journal of Portfolio Management</i>, 37(4), 112-126.</p>
<p>Lo, A. W. et MacKinlay, A. C. (1988). Stock market prices do not follow random walks: evidence from a simple specification test. <i>Review of Financial Studies</i>, 1(1), 41-66.</p>
<h2>Méthode statistique</h2>
<p>Newey, W. K. et West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. <i>Econometrica</i>, 55(3), 703-708.</p>
<p>Politis, D. N. et Romano, J. P. (1994). The stationary bootstrap. <i>Journal of the American Statistical Association</i>, 89(428), 1303-1313.</p>
<h2>Stratégies et ouverture</h2>
<p>Daniel, K. et Moskowitz, T. J. (2016). Momentum crashes. <i>Journal of Financial Economics</i>, 122(2), 221-247.</p>
<p>Jegadeesh, N. et Titman, S. (1993). Returns to buying winners and selling losers: implications for stock market efficiency. <i>Journal of Finance</i>, 48(1), 65-91.</p>
<p>Lunde, A. et Timmermann, A. (2004). Duration dependence in stock prices: an analysis of bull and bear markets. <i>Journal of Business &amp; Economic Statistics</i>, 22(3), 253-273.</p>
<h2>Données</h2>
<p>Federal Reserve Bank of St. Louis, FRED et ALFRED (archive des publications successives) · Kenneth R. French Data Library · National Bureau of Economic Research, chronologie des cycles américains · Yahoo Finance · Cboe (VIX).</p>
</div>""")

    s.append(annexes(configs, nz_lo, nz_hi))
    body = french_typography("".join(s))
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<title>Détecter les crises de marché — dossier de projet</title><style>{CSS}</style></head><body>{body}</body></html>"""


# ---------------------------------------------------------------------------
# annexes
# ---------------------------------------------------------------------------
FEATURES = [
    ("mom_eq_21 / 63 / 252", "Tendance", "Rendement du S&amp;P 500 sur 21, 63 et 252 séances", "Yahoo"),
    ("mom_eq_accel", "Tendance", "Rendement sur 21 séances moins le douzième du rendement sur 252", "Yahoo"),
    ("mom_oil_63, mom_usd_63", "Tendance", "Rendement du pétrole WTI et de l'indice dollar sur 63 séances", "FRED, Yahoo"),
    ("mom_breadth_200d", "Tendance", "Part des marchés (S&amp;P 500, Russell 2000, Nikkei, pétrole) au-dessus de leur moyenne à 200 séances", "Yahoo, FRED"),
    ("vol_rv_5 / 21 / 63", "Volatilité", "Volatilité réalisée du S&amp;P 500 sur 5, 21 et 63 séances, annualisée", "Yahoo"),
    ("vol_term", "Volatilité", "Volatilité sur 5 séances rapportée à celle sur 63", "Yahoo"),
    ("vol_of_vol", "Volatilité", "Écart-type sur 63 séances de la volatilité à 21 séances", "Yahoo"),
    ("vol_semi_ratio", "Volatilité", "Somme des carrés des baisses sur celle des hausses, 63 séances", "Yahoo"),
    ("vol_jump_share", "Volatilité", "Part de la variance due aux sauts : variance réalisée moins variation bipuissance, sur 21 séances", "Yahoo"),
    ("vol_vix", "Volatilité", "Niveau du VIX", "Yahoo"),
    ("vol_vrp", "Volatilité", "Prime de variance : VIX moins volatilité réalisée sur 21 séances", "Yahoo"),
    ("asy_skew_63, asy_kurt_63", "Asymétrie et mémoire", "Asymétrie et aplatissement des rendements sur 63 séances", "Yahoo"),
    ("asy_drawdown_252", "Asymétrie et mémoire", "Distance au plus haut des 252 dernières séances", "Yahoo"),
    ("asy_hurst", "Asymétrie et mémoire", "Exposant de Hurst par analyse des fluctuations sans tendance (DFA)", "Yahoo"),
    ("asy_vr_5, asy_vr_20", "Asymétrie et mémoire", "Ratios de variance de Lo et MacKinlay à 5 et 20 jours, sur 252 séances", "Yahoo"),
    ("xs_dispersion_ind / szbm", "Coupe transversale", "Dispersion des rendements entre les 49 secteurs, puis entre les 25 portefeuilles taille × valeur, lissée sur 21 séances", "Ken French"),
    ("xs_avg_corr", "Coupe transversale", "Corrélation moyenne entre secteurs, 63 séances", "Ken French"),
    ("xs_absorption, xs_absorption_chg", "Coupe transversale", "Part de la variance des secteurs expliquée par les 5 premiers facteurs (252 séances), et sa variation sur 63", "Ken French"),
    ("xs_breadth_63", "Coupe transversale", "Part des secteurs en hausse sur 63 séances", "Ken French"),
    ("xs_ff_smb_63, xs_ff_hml_63", "Coupe transversale", "Facteurs taille et valeur cumulés sur 63 séances", "Ken French"),
    ("mac_indpro_yoy / 3m", "Macroéconomie", "Croissance de la production industrielle sur un an et sur trois mois", "ALFRED"),
    ("mac_payems_yoy / 3m", "Macroéconomie", "Croissance de l'emploi non agricole sur un an et sur trois mois", "ALFRED"),
    ("mac_cpi_yoy / 3m", "Macroéconomie", "Inflation sur un an et sur trois mois", "ALFRED"),
    ("mac_unrate_chg12m", "Macroéconomie", "Variation du taux de chômage sur un an", "ALFRED"),
    ("mac_unrate_sahm", "Macroéconomie", "Règle de Sahm : moyenne du chômage sur trois mois moins son minimum sur un an", "ALFRED"),
    ("mac_claims_yoy", "Macroéconomie", "Variation sur un an des inscriptions au chômage (moyenne sur 20 séances)", "FRED"),
    ("cre_baa, cre_baa_chg63", "Crédit", "Écart de rendement Baa contre le 10 ans, et sa variation sur 63 séances", "FRED"),
    ("cre_quality, cre_quality_chg63", "Crédit", "Écart Baa moins Aaa, et sa variation sur 63 séances", "FRED"),
    ("fin_nfci, fin_nfci_chg13w", "Conditions financières", "Indice NFCI de la Fed de Chicago, et sa variation sur 13 semaines", "FRED"),
    ("rat_slope_10y2y, _chg63", "Taux", "Pente 10 ans – 2 ans, et sa variation sur 63 séances", "FRED"),
    ("rat_slope_10y3m, _chg63", "Taux", "Pente 10 ans – 3 mois, et sa variation sur 63 séances", "FRED"),
    ("rat_cash_chg12m", "Taux", "Variation du taux à 3 mois sur un an", "FRED"),
]


def annexes(configs: int, nz_lo: int, nz_hi: int) -> str:
    feat_rows = [[f"<code>{a}</code>", b, d, e] for a, b, d, e in FEATURES]
    n_listed = sum(len(re.split(r",|/", a)) for a, *_ in FEATURES)
    if n_listed != 50:
        raise SystemExit(f"annex A lists {n_listed} features, not 50")
    deviations = [
        ["08/09", "Ordre des états", "Une erreur de signe dans l'enveloppe du modèle inversait les états « calme » et « stress ». Corrigée en ordonnant les états par leur volatilité d'entraînement ; la première explication consignée était fausse et a été rectifiée."],
        ["08/09", "Règle de position", "Une règle de taille (inverse de la volatilité de l'état) a été ajoutée à côté de la règle tout-ou-rien gelée ; les deux sont publiées."],
        ["08/09", "Grille de λ", "Non élargie malgré un optimum en bord de grille : un élargissement après avoir vu les résultats aurait été un ajustement a posteriori."],
        ["10/09", "Audit indépendant", "NBER noté contre le mois civil, et non avec jusqu'à 45 jours de recul ; témoin de volatilité causal ajouté ; information sur la volatilité future publiée. Exactitude revue de 95,1 % à 93,2 %."],
        ["13/09", "Contrôles T1, T3, T5", "Exécutés après les résultats principaux ; deux des trois affaiblissent l'étude (aucun modèle ne bat le témoin ; le gain passe par le dénominateur)."],
        ["13/09", "Plis", "Deux des cinq plis de cinq ans d'A′ ne contiennent qu'un état : la règle d'arrêt « trois plis sur cinq » est incalculable. Le walk-forward à 49 réestimations devient la référence."],
        ["—", "Règle d'arrêt n°2", "Déclenchée (R² incrémental sur les rendements de 0,007 à 0,030 point, sous 0,2) sans être annoncée ; l'étude a continué sur la volatilité."],
        ["—", "Seconde stratégie de base", "Le momentum à risque égal, prévu par le cadrage, n'a jamais été évalué."],
        ["—", "Révisions", "Les séries ne stockent que leur première publication : un panel passé n'est pas reconstruit exactement tel qu'il était connu."],
        ["22/09", "Réplication de Shu et al.", "Une ligne publiée sans code qui la produise a été retirée ; la conclusion repose sur le balayage de huit pénalités fixes."],
        ["23/09", "Réserve scellée", "La réserve 1971-1989, tenue à l'écart jusque-là, a été ouverte pour le test sur 90 ans, avant sa lecture."],
        ["25/09", "Sous-ensemble commun", "Jamais mis en œuvre : les 50 variables vont à A, A′, B et C ; seul A′ sélectionne, dans son propre ajustement. La comparaison A′ – B mêle modèle et sélection."],
        ["25/09", "Jump Model continu", "Prévu par le cadrage, jamais estimé."],
        ["25/09", "Choix de λ", "Recalculé toutes les quatre réestimations ; repli à 20 (médiane de la grille, hors grille) quand aucun candidat n'est admissible, ce qui vaut pour A′ d'avril 2022 à juillet 2026."],
        ["25/09", "Attribution d'un rappel", "Le rappel corrigé 419/435 attribué à A est celui d'A′ ; celui d'A est 431/435."],
        ["—", "Étiquette git", "Annotée mais non signée ; la date de publication sur GitHub fait foi."],
    ]
    phase_a = [["SJM réduit (30 variables)", "57,5 %", "0,16", "6 / 14", "2,05", "12,2 %"],
               ["Règle de volatilité médiane", "61,8 %", "0,11", "11 / 14", "8,85", "47,5 %"],
               ["Règle de volatilité au 80ᵉ centile", "60,4 %", "0,19", "9 / 14", "3,55", "16,0 %"],
               ["Panique de Daniel et Moskowitz", "64,6 %", "0,31", "6 / 14", "1,74", "11,3 %"]]
    phase_b = [["B1 arrêt du SJM réduit − seul", "+0,045", "0,158", "−0,78", "96ᵉ c.", "Sous-puissant"],
               ["B2 arrêt à sortie asymétrique − seul", "−0,013", "0,145", "−1,82", "70ᵉ c.", "Pas utile"],
               ["B3 arrêt du SJM réduit − arrêt de la règle au 80ᵉ c.", "+0,026", "0,153", "+1,47", "96ᵉ c.", "Sous-puissant"]]
    others = [
        ["Stratégies de crise", "Vente de variance, vente de futures VIX, momentum, achat-vente de call, coupées ou réduites en stress : 9 tests, aucun utile ; test P : l'état ne prévoit pas la volatilité au-delà du VIX.", "RESULTS_CRISE"],
        ["Actions en calme, refuge en stress", "Or, obligations longues ou volatilité achetée en stress à la place des actions : 3 tests, aucun utile ; l'or +0,04, sous le seuil, et une règle de volatilité fait mieux.", "RESULTS_REFUGE §1"],
        ["Refuge pour neuf stratégies", "9 stratégies × 3 jambes de remplacement : 27 tests, 8 sous-puissants, 19 pas utiles.", "RESULTS_REFUGE §2"],
        ["Couplage à huit stratégies", "Arrêt, réduction, bascule vers la tendance : 24 tests, aucun utile ; tendance crypto et prime overnight pénalisées.", "RESULTS_COUPLAGE_STRATEGIES"],
        ["Corrélation actions-obligations", "Son signe, pris comme régime, prévoit le risque d'un portefeuille à risque égal au-delà du VIX (+3,10 points de R², placebo pile au seuil de 95 %) ; il ne choisit pas mieux la couverture.", "RESULTS_COUVERTURE"],
        ["Régime en budget de risque", "À l'intérieur du livre de neuf stratégies : 3 tests, aucun utile.", "RESULTS_BUDGET_RISQUE"],
        ["Prévision de risque, 46 marchés", "L'état n'améliore la prévision de volatilité d'aucun marché (−0,8 % sur 43 marchés) ; le VIX l'améliore de 5,6 %.", "RESULTS_RISQUE"],
        ["Un régime par facteur", "Six régimes pour six facteurs (Shu et Mulvey, 2024), 1978-2026 : Sharpe 1,39 contre 1,51 pour les six facteurs tenus en permanence ; pas utile.", "RESULTS_FACTORSJM"],
        ["Sélection entre signaux", "Un contexte de marché, orthogonal à la volatilité, pour choisir entre dix signaux sectoriels : niveau A en échec (−0,110), B sous-puissant, C non montré.", "RESULTS_TWOSIGMA"],
        ["Vitesse des signaux de tendance", "Le régime ne change pas le classement des vitesses (niveau A, réfuté sur le signe) ; niveau B indécidable faute de puissance.", "RESULTS_AHL_LEVEL_A, _B"],
        ["Réplication de Shu, Yu et Mulvey (2024)", "Le S&amp;P 500 conservé se reproduit au chiffre près (Sharpe 0,48) ; le risque se reproduit, pas le rendement ; le chiffre publié (0,68) ne dépasse une cible de volatilité (0,61) que de 0,06.", "REPLICATION_SHU2024"],
        ["Hypothèses macro H1, H2, H3", "Momentum macro absolu, transversal sur 19 pays, surprises macro : falsifiées (H2 : la dispersion des taux courts de la zone euro a disparu en 1999).", "chantiers/macro-momentum"],
        ["Prime de retournement", "Sharpe de 3,64 dans les années 1990, −0,18 depuis 2020 : falsifiée.", "chantiers/reversal-lab"],
    ]
    return f"""
<div class="annex">
<h1 class="sec newpage"><span class="n">A</span>Les 50 variables</h1>
<p class="small-p">Toutes sont centrées et réduites sur leur seul passé puis bornées à ±5 (§2). Le code est dans
<code>regime_lab/features/</code>. Un facteur momentum prévu (<code>xs_ff_mom_63</code>) n'a jamais été construit, faute
d'être dans le fichier des cinq facteurs&nbsp;; son absence est consignée dans le code.</p>
{tbl(["Code", "Famille", "Définition", "Source"], feat_rows, cls="small")}

<h1 class="sec"><span class="n">B</span>Écarts au protocole et corrections</h1>
<p class="small-p">Le cadrage gelé n'est jamais modifié&nbsp;; chaque écart est daté et expliqué dans
<code>docs/PROTOCOL_FREEZE.md</code>, chaque correction de chiffre dans le document de résultats concerné, avec sa
date et sa raison.</p>
{tbl(["Date (2026)", "Objet", "Ce qui s'est passé"], deviations, cls="small")}

<h1 class="sec"><span class="n">C</span>Méthode statistique</h1>
<ul class="small-p">
  <li><b>Ratio de Sharpe</b>&nbsp;: moyenne sur écart-type des rendements quotidiens en excès du taux à 3 mois, multiplié
  par √252.</li>
  <li><b>Exactitude équilibrée</b>&nbsp;: (rappel + spécificité) / 2. <b>κ de Cohen</b>&nbsp;: (accord observé − accord
  attendu par hasard) / (1 − accord attendu).</li>
  <li><b>R² incrémental</b>&nbsp;: R² de la régression avec le témoin et l'état, moins R² avec le témoin seul, sur des
  fenêtres de 21 séances qui se chevauchent&nbsp;; t de l'état par erreurs-types HAC.</li>
  <li><b>t HAC</b>&nbsp;: Newey et West, 6 retards, sur la différence quotidienne entre la stratégie filtrée et la
  stratégie seule.</li>
  <li><b>Seuil de détection (MDE)</b>&nbsp;: bootstrap stationnaire apparié par blocs (longueurs moyennes de 21, 63 et
  126 séances, la plus défavorable retenue), sur des séries centrées, au niveau α = 0,05 divisé par le nombre de
  tests de l'étude (Bonferroni).</li>
  <li><b>Placebo</b>&nbsp;: la série des états décalée circulairement d'au moins 252 séances, 400 tirages&nbsp;; il garde
  la durée et le nombre des épisodes de stress et change leurs dates.</li>
  <li><b>Témoins</b>&nbsp;: stress quand la volatilité réalisée du S&amp;P 500 sur 21 séances dépasse sa médiane passée
  (environ la moitié du temps), ou son 80ᵉ centile passé (environ 22 % du temps, un ordre proche des
  16 % de stress du modèle).</li>
  <li><b>Ciblage de volatilité</b>&nbsp;: chaque stratégie et chaque jambe de remplacement est ramenée à 10 % de
  volatilité annuelle, estimée sur le passé.</li>
  <li><b>Verdicts</b>&nbsp;: voir l'encadré du §1. Une lecture qui contient une valeur non finie ne reçoit pas de
  verdict.</li>
  <li><b>Journal des essais</b>&nbsp;: chaque configuration évaluée est inscrite dans <code>data/trials.parquet</code>
  ({configs} configurations distinctes au 25 septembre 2026), pour permettre de corriger du nombre d'essais.</li>
</ul>

<h1 class="sec"><span class="n">D</span>Le test sur 90 ans</h1>
<p class="small-p">Même méthode (deux états, réestimation semestrielle sur une fenêtre croissante, états filtrés),
réestimée sur 1926-2026 avec les 30 variables calculables depuis 1926. C'est un cousin d'A′, pas A′&nbsp;: ni VIX, ni
NFCI, ni macroéconomie en première publication. Protocole commité avant l'ajustement du modèle
(<code>docs/PRESPEC_LONGHIST.md</code>).</p>
{tbl(["Phase A : classification, 1937-2026, 14 récessions", "Exactitude équilibrée", "κ", "Récessions détectées", "Transitions / an", "Temps en stress"], phase_a, num={1, 2, 3, 4, 5}, cls="small", lead=0)}
<p class="tnote" style="margin-bottom:3mm">Source : docs/RESULTS_LONGHIST.md §2.</p>
{tbl(["Phase B : couper le momentum en stress, 1937-2026", "Δ Sharpe", "Seuil (MDE)", "t HAC", "Placebo", "Verdict"], phase_b, num={1, 2, 3, 4}, cls="small")}
<p class="tnote">Source : docs/RESULTS_LONGHIST.md §3. Le momentum seul fait 1,12 de Sharpe sur la période (ciblé en
volatilité, sans coût)&nbsp;; la règle de Daniel et Moskowitz, témoin non testé, +0,106. Sans ciblage de volatilité, le
Sharpe passe de 0,51 à 0,69 avec l'arrêt&nbsp;: le ciblage fait l'essentiel du travail (0,51 → 1,12 à lui seul).
Leçon&nbsp;: un détecteur de crise calibré sur l'historique dépend de l'historique choisi — entraîné avec la Grande
Dépression, le modèle ne signale presque aucun stress de 1937 à 1959.</p>

<h1 class="sec"><span class="n">E</span>Les autres études du programme</h1>
<p class="small-p">Le document de chacune (dans <code>docs/</code>, ou dans le dossier indiqué) précise son protocole, la
date de son pré-enregistrement quand il y en a un, et sa lecture. Une étude de construction de portefeuille par quadrants macroéconomiques était en cours le 25
septembre et n'est pas incluse.</p>
{tbl(["Étude", "Résultat", "Document"], [[r[0], r[1], f"<code>{r[2]}</code>"] for r in others], cls="small", widths=[17, 52, 31])}
</div>"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    page = OUT_DIR / "dossier_regimes.html"
    pdf = OUT_DIR / "dossier_regimes.pdf"
    page.write_text(build_html(), encoding="utf-8")
    subprocess.run([bp.CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", page.as_uri()], check=True, capture_output=True)
    pages = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    n = re.search(r"Pages:\s+(\d+)", pages)
    print(f"{pdf.relative_to(ROOT)}  ({pdf.stat().st_size / 1e3:.0f} kB, {n.group(1) if n else '?'} pages)")


if __name__ == "__main__":
    main()
