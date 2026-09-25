# ruff: noqa: E501  (slide copy is prose; wrapping it would hurt more than help)
"""Build the presentation deck: docs/presentation/presentation_regimes.{html,pdf}.

Slides are HTML/CSS at 13.333 x 7.5 in (16:9), printed to PDF by headless Chrome.
Charts are hand-built inline SVG drawn from the data: the S&P 500 and the sparse jump
states come from data/; every other figure is transcribed from the committed result
document cited in the slide's footer (RESULTS_FINAL.md, RESULTS_DECOMPOSITION.md,
RESULTS_COUPLAGE_STRATEGIES.md), never from memory.

Colour roles, validated with the dataviz skill's validator (light surface):
    emphasis / volatility  #2a78d6   returns  #eb6834   stress  #e34948
    other models           #c3c2b7   ink #0b0b0b / #52514e / #898781

Usage: .venv/bin/python scripts/build_presentation.py
"""

from __future__ import annotations

import html
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_b1_regime_coupling as b1  # noqa: E402

from regime_lab.config import CACHE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "presentation"
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

BLUE, ORANGE, RED, GREY = "#2a78d6", "#eb6834", "#e34948", "#c3c2b7"
INK, INK2, MUTED, RULE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
NAVY = "#0f2a44"


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def fr(value: float, spec: str) -> str:
    """French number: decimal comma, true minus sign; '+0,00' and '−0,00' become '0,00'."""
    text = format(value, spec).replace(".", ",").replace("-", "−")
    if text.lstrip("+−").strip("0,") == "":
        text = text.lstrip("+−")
    return text


def svg_text(x: float, y: float, text: str, *, size: int = 14, fill: str = INK,
             anchor: str = "start", weight: int = 400) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}" dominant-baseline="middle">'
            f"{text}</text>")


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------
def accuracy_chart() -> str:
    """Balanced accuracy against NBER recessions, with kappa as a direct label."""
    rows = [("A′ Sparse Jump", 93.2, 0.53, True), ("A Jump model", 93.3, 0.49, False),
            ("B HMM filtré", 84.7, 0.24, False), ("C′ HAR-RV", 78.2, 0.17, False),
            ("C Gradient boosting", 75.0, 0.12, False)]
    width, row, label_w, right = 600, 62, 170, 150
    height = row * len(rows) + 34
    plot = width - label_w - right

    def x(v: float) -> float:
        return label_w + plot * v / 100

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    for v in range(0, 101, 25):
        out.append(f'<line x1="{x(v):.1f}" y1="4" x2="{x(v):.1f}" y2="{height - 28}" '
                   f'stroke="{GREY if v == 0 else RULE}" stroke-width="{1.5 if v == 0 else 1}"/>')
        out.append(svg_text(x(v), height - 12, f"{v} %", size=12, fill=MUTED, anchor="middle"))
    xc = x(50)
    out.append(f'<line x1="{xc:.1f}" y1="4" x2="{xc:.1f}" y2="{height - 28}" stroke="{MUTED}" '
               f'stroke-dasharray="4 4"/>')
    for k, (label, acc, kappa, lead) in enumerate(rows):
        yc = 6 + k * row + row / 2 - 4
        weight = 600 if lead else 400
        out.append(svg_text(label_w - 12, yc, esc(label), size=16, anchor="end", weight=weight))
        out.append(f'<rect x="{label_w}" y="{yc - 14:.1f}" width="{x(acc) - label_w:.1f}" '
                   f'height="28" rx="4" fill="{BLUE if lead else GREY}"/>')
        out.append(svg_text(x(acc) + 10, yc, fr(acc, ".1f") + " %", size=16, weight=weight))
        out.append(svg_text(x(acc) + 82, yc, "kappa " + fr(kappa, ".2f"), size=14, fill=INK2))
    out.append("</svg>")
    return "".join(out)


def grouped_r2() -> str:
    """Incremental R² on forward volatility vs forward returns, per model."""
    rows = [("A′ Sparse Jump", 3.93, 0.030, True), ("A Jump model", 3.47, 0.022, False),
            ("B HMM filtré", 2.26, 0.007, False), ("C Gradient boosting", 2.13, 0.023, False),
            ("C′ HAR-RV", 0.19, 0.008, False), ("Témoin volatilité", 0.004, 0.000, False)]
    width, row, label_w, right = 620, 60, 170, 70
    height = row * len(rows) + 34
    plot = width - label_w - right
    xmax = 4.0

    def x(v: float) -> float:
        return label_w + plot * v / xmax

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    for v in range(0, 5):
        out.append(f'<line x1="{x(v):.1f}" y1="4" x2="{x(v):.1f}" y2="{height - 28}" '
                   f'stroke="{GREY if v == 0 else RULE}" stroke-width="{1.5 if v == 0 else 1}"/>')
        out.append(svg_text(x(v), height - 12, f"{v} pt", size=12, fill=MUTED, anchor="middle"))
    for k, (label, vol, ret, lead) in enumerate(rows):
        y0 = 6 + k * row
        weight = 600 if lead else 400
        out.append(svg_text(label_w - 12, y0 + 24, esc(label), size=15, anchor="end",
                            weight=weight))
        wv, wr = max(x(vol) - label_w, 2), max(x(ret) - label_w, 2)
        out.append(f'<rect x="{label_w}" y="{y0 + 5}" width="{wv:.1f}" height="19" rx="4" '
                   f'fill="{BLUE}"/>')
        out.append(svg_text(label_w + wv + 8, y0 + 15, fr(vol, "+.2f" if vol >= 0.01 else "+.3f"),
                            size=14, weight=weight))
        out.append(f'<rect x="{label_w}" y="{y0 + 26}" width="{wr:.1f}" height="19" rx="4" '
                   f'fill="{ORANGE}"/>')
        out.append(svg_text(label_w + wr + 8, y0 + 36, fr(ret, "+.3f"), size=14, fill=INK2))
    out.append("</svg>")
    return "".join(out)


def timeline() -> str:
    """S&P 500 (log scale) with the sparse jump model's stress episodes shaded."""
    states = pd.read_parquet(CACHE / "states.parquet")["A' sparse jump"].dropna()
    spx = b1.us_prices("eq_us_large")
    spx.index = pd.to_datetime(spx.index)
    spx = spx.loc[states.index.min():states.index.max()].resample("W-FRI").last().dropna()
    width, height, left, right, top, bottom = 1180, 410, 56, 12, 30, 30
    t0, t1 = states.index.min(), states.index.max()
    lo, hi = math.log(spx.min() * 0.9), math.log(spx.max() * 1.08)

    def x(d: pd.Timestamp | str) -> float:
        d = pd.Timestamp(d)
        return left + (d - t0).days / (t1 - t0).days * (width - left - right)

    def y(v: float) -> float:
        return top + (hi - math.log(v)) / (hi - lo) * (height - top - bottom)

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    segment = (states != states.shift()).cumsum()
    for _, g in states.groupby(segment):
        if g.iloc[0] == 0:
            x0, x1 = x(g.index[0]), x(g.index[-1])
            out.append(f'<rect x="{x0:.1f}" y="{top}" width="{max(x1 - x0, 2):.1f}" '
                       f'height="{height - top - bottom}" fill="{RED}" fill-opacity="0.16"/>')
    for level in (800, 1000, 1500, 2000, 3000, 4000, 6000):
        if lo < math.log(level) < hi:
            yy = y(level)
            out.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}" '
                       f'stroke="{RULE}"/>')
            out.append(svg_text(left - 8, yy, f"{level:,}".replace(",", " "), size=12,
                                fill=MUTED, anchor="end"))
    for year in range(2004, 2027, 2):
        out.append(svg_text(x(f"{year}-01-01"), height - 12, str(year), size=12, fill=MUTED,
                            anchor="middle"))
    pts = " ".join(f"{x(d):.1f},{y(v):.1f}" for d, v in spx.items())
    out.append(f'<polyline points="{pts}" fill="none" stroke="{INK}" stroke-width="1.6"/>')
    for label, start in (("Après la bulle internet", "2002-04-01"),
                         ("Crise financière", "2007-11-22"), ("Covid", "2020-03-11")):
        out.append(svg_text(x(start) + 2, top - 13, label, size=13, fill=INK2, weight=600))
    low_2022 = spx.loc["2022"].min()
    xx, yy = x(spx.loc["2022"].idxmin()), y(low_2022)
    out.append(f'<line x1="{xx:.1f}" y1="{yy + 6:.1f}" x2="{xx:.1f}" y2="{yy + 34:.1f}" '
               f'stroke="{INK2}"/>')
    out.append(svg_text(xx, yy + 46, "2022 : baisse de 25 %, non signalée", size=13, fill=INK2,
                        anchor="middle"))
    out.append("</svg>")
    return "".join(out)


def coupling_chart() -> str:
    """STOP delta Sharpe per strategy, diverging around zero."""
    rows = [("Rebond obligataire fin de mois", 0.01, False),
            ("Cassure d'ouverture (ORB) Nasdaq", 0.00, False), ("Momentum USDJPY", -0.01, False),
            ("Tendance or", -0.04, False), ("Tendance or + argent", -0.06, False),
            ("Tendance énergie (long)", -0.11, False), ("Prime overnight Nasdaq", -0.12, True),
            ("Tendance crypto", -0.24, True)]
    width, row, label_w = 640, 46, 250
    height = row * len(rows) + 34
    lo, hi = -0.30, 0.10
    plot = width - label_w - 20

    def x(v: float) -> float:
        return label_w + (v - lo) / (hi - lo) * plot

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    for v in (-0.3, -0.2, -0.1, 0.0, 0.1):
        out.append(f'<line x1="{x(v):.1f}" y1="4" x2="{x(v):.1f}" y2="{height - 28}" '
                   f'stroke="{GREY if v == 0 else RULE}" stroke-width="{1.5 if v == 0 else 1}"/>')
        out.append(svg_text(x(v), height - 12, fr(v, "+.1f"), size=12, fill=MUTED,
                            anchor="middle"))
    for k, (label, v, penalised) in enumerate(rows):
        yc = 6 + k * row + row / 2 - 4
        colour = RED if penalised else GREY
        x0, x1 = sorted((x(0.0), x(v)))
        out.append(svg_text(label_w - 12, yc, esc(label), size=15, anchor="end"))
        out.append(f'<rect x="{x0:.1f}" y="{yc - 12:.1f}" width="{max(x1 - x0, 2):.1f}" '
                   f'height="24" rx="4" fill="{colour}"/>')
        tx, anchor = (x1 + 8, "start") if v >= 0 else (x0 - 8, "end")
        out.append(svg_text(tx, yc, fr(v, "+.2f"), size=14, anchor=anchor))
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------------------
# slides
# ---------------------------------------------------------------------------
def table(header: list[str], rows: list[list[str]], *, num: set[int] = frozenset(),
          highlight: int | None = None, cls: str = "") -> str:
    head = "".join(f'<th class="{"num" if i in num else ""}">{h}</th>'
                   for i, h in enumerate(header))
    body = []
    for r, cells in enumerate(rows):
        hl = ' class="hl"' if highlight == r else ""
        body.append(f"<tr{hl}>" + "".join(
            f'<td class="{"num" if i in num else ""}">{c}</td>' for i, c in enumerate(cells))
            + "</tr>")
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def slide(n: int, eyebrow: str, title: str, body: str, source: str = "", cls: str = "") -> str:
    src = f'<div class="src">Source : {source}</div>' if source else '<div class="src"></div>'
    return f"""
<section class="slide {cls}">
  <div class="eyebrow">{eyebrow}</div>
  <h1>{title}</h1>
  <div class="body">{body}</div>
  <footer>{src}<div class="brand">Régimes de marché · Projet Big Data M2</div>
  <div class="pn">{n}</div></footer>
</section>"""


def build_html() -> str:
    s = []
    s.append("""
<section class="slide cover">
  <div class="cover-band"></div>
  <div class="cover-inner">
    <div class="eyebrow light">Projet Big Data · Master 2 · Septembre 2026</div>
    <h1 class="cover-title">Régimes de marché&nbsp;: ce que la classification apporte vraiment</h1>
    <p class="cover-sub">Cinq modèles comparés sur 24 ans hors échantillon, un protocole fixé
    avant les résultats, et une conclusion nette&nbsp;: <b>le régime mesure le risque, pas la
    direction du marché.</b></p>
    <div class="cover-meta">Sparse Jump Model · HMM · Gradient boosting · HAR-RV · données publiques</div>
  </div>
</section>""")

    kpis = [("93,2 %", "d'exactitude face aux récessions officielles", "exactitude équilibrée "
             "contre le NBER, sans voir le futur ; 2 récessions dans la période (2008-09, 2020)"),
            ("+3,93 pts", "d'information sur le risque futur", "au-delà d'une règle de volatilité "
             "passée. Test séparé sur le S&P 500 : +4,48 sans le VIX, +0,20 avec"),
            ("+0,03 pt", "d'information sur la direction", "R² incrémental sur les rendements "
             "futurs : statistiquement nul (t = 0,27)"),
            ("0 / 62", "usages du filtre utiles", "arrêt, réduction, bascule, valeurs refuges et "
             "options, sur 13 stratégies : aucune amélioration démontrée")]
    tiles = "".join(f'<div class="kpi"><div class="kv">{v}</div><div class="kl">{lab}</div>'
                    f'<div class="kd">{d}</div></div>' for v, lab, d in kpis)
    s.append(slide(2, "Synthèse", "Le classifieur fonctionne&nbsp;; il prédit le risque, pas le sens",
        f"""<div class="kpis">{tiles}</div>
        <ol class="msgs">
          <li><b>Il reconnaît les crises sans voir le futur</b>, mais pas avant le marché&nbsp;: au
          Covid, il bascule le 11 mars 2020, quand le VIX est déjà passé de 14 à 54.</li>
          <li><b>Il informe sur la volatilité future, pas sur la direction</b>, et le VIX en savait
          déjà presque autant. C'est un thermomètre du risque, pas un signal de trading.</li>
          <li><b>Utilisé comme filtre, il n'améliore aucune stratégie de façon démontrable</b>&nbsp;:
          62 usages testés, avec un critère écrit avant chaque lecture.</li>
        </ol>""", "docs/RESULTS_FINAL.md, RESULTS_COUPLAGE_STRATEGIES, RESULTS_CRISE, RESULTS_REFUGE"))

    n_configs = f"{pd.read_parquet(ROOT / 'data' / 'trials.parquet')['config_hash'].nunique():,}"
    s.append(slide(3, "La question et la démarche",
        "Une question précise, testée pour pouvoir répondre non",
        f"""<div class="question">« Un modèle de régimes appris par machine apporte-t-il une
        information exploitable au-delà de ce qu'une simple mesure de volatilité capte déjà&nbsp;? »</div>
        <div class="grid4">
          <div class="card"><h3>Pré-enregistré</h3><p>Modèles, règles et critères écrits et gelés
          <b>avant</b> tout résultat, scellés par empreinte SHA-256.</p></div>
          <div class="card"><h3>Point-in-time</h3><p>Chaque donnée n'est utilisée qu'à partir du
          jour de sa publication réelle, en première version.</p></div>
          <div class="card"><h3>Hors échantillon</h3><p>Réestimés tous les 6 mois sur le passé
          seul&nbsp;; états <b>filtrés</b>, jamais lissés&nbsp;; signal en T−1, position en T.</p></div>
          <div class="card"><h3>Un témoin partout</h3><p>Chaque modèle doit battre une règle d'une
          ligne&nbsp;: «&nbsp;la volatilité récente est-elle sous sa médiane&nbsp;?&nbsp;»</p></div>
        </div>
        <div class="facts">
          <div><b>6 377</b><span>séances hors échantillon, avril 2002 – septembre 2026</span></div>
          <div><b>49</b><span>réestimations de chaque modèle, sur le passé seul</span></div>
          <div><b>5</b><span>modèles comparés sous un protocole identique</span></div>
          <div><b>{n_configs}</b><span>configurations testées dans le programme, toutes journalisées</span></div>
        </div>""", "docs/CHARTER.html, docs/PROTOCOL_FREEZE.md, data/trials.parquet"))

    data_rows = [
        ["<b>Prix de marché</b>", "S&amp;P 500, Russell 2000, Nasdaq, Nikkei, taux US 10 et 30 ans, "
         "dollar, VIX", "Yahoo Finance", "1990 – 2026", "Momentum, volatilité, asymétrie"],
        ["<b>Pétrole WTI</b>", "Prix spot quotidien", "FRED", "1990 – 2026", "Momentum des matières "
         "premières"],
        ["<b>Macroéconomie</b>", "Production industrielle, emploi, inflation, chômage (premières "
         "publications)&nbsp;; spreads de crédit, taux, NFCI", "FRED / ALFRED", "1990 – 2026",
         "Cycle, crédit, conditions financières"],
        ["<b>Panels d'actions</b>", "49 secteurs, 25 portefeuilles taille × valeur, 5 facteurs",
         "Ken French", "1990 – 2026", "Dispersion, corrélation, concentration"],
        ["<b>Récessions NBER</b>", "Indicateur mensuel des récessions US", "FRED", "1990 – 2026",
         "<b>Validation seulement</b> — jamais vu par les modèles"],
    ]
    s.append(slide(4, "D'où viennent les données",
        "Cinq sources publiques, transformées en 50 variables sur 36 ans",
        table(["Bloc", "Contenu", "Source", "Période", "Usage"], data_rows, cls="wide data")
        + """<div class="note">Les 50 variables couvrent 8 familles&nbsp;: momentum (7), volatilité
        (9), asymétrie et mémoire longue (6), transversal (8), macroéconomie (9), crédit (4),
        conditions financières (2) et taux (5). Un bloc de variables <b>indépendantes de la
        volatilité</b> (dispersion, corrélation moyenne, crédit, pente des taux) a été imposé pour
        que le modèle ne se réduise pas à un simple indicateur de volatilité. Détail en annexe.</div>""",
        "docs/DATA_NOTES.md, data/raw/, AVANCEMENT.md §3"))

    flow = [("Données", "5 sources publiques, point-in-time, depuis 1990"),
            ("50 variables", "standardisées sur le passé seul"),
            ("Modèle", "réestimé tous les 6 mois, fenêtre expansive"),
            ("État du jour", "stress ou calme, filtré, connu la veille"),
            ("Évaluation", "3 couches, toujours contre le témoin")]
    steps = "".join(f'<div class="step"><div class="sn">{i + 1}</div><h3>{a}</h3><p>{b}</p></div>'
                    + ('<div class="arrow">→</div>' if i < len(flow) - 1 else "")
                    for i, (a, b) in enumerate(flow))
    s.append(slide(5, "Comment on évalue", "Un protocole identique pour les cinq modèles",
        f"""<div class="flow">{steps}</div>
        <div class="grid3">
          <div class="card"><h3>Couche 1 — Classification</h3><p>Les états reconnaissent-ils les
          récessions officielles&nbsp;? Sont-ils stables et reconnus en temps réel&nbsp;?</p></div>
          <div class="card"><h3>Couche 2 — Information</h3><p>L'état apporte-t-il de l'information
          sur la volatilité et sur les rendements futurs <b>au-delà du témoin</b>&nbsp;?</p></div>
          <div class="card"><h3>Couche 3 — Portefeuille</h3><p>Un portefeuille 60/40 piloté par
          l'état (tout ou rien, ou taille) fait-il mieux que le 60/40 seul&nbsp;?</p></div>
        </div>
        <div class="note">Équité entre modèles&nbsp;: même calendrier de réestimation, mêmes 50
        variables (C′&nbsp;: ses trois termes HAR), même règle de position fixée d'avance, deux
        états ordonnés par la volatilité d'entraînement.</div>""",
        "regime_lab/models/protocol.py, docs/CHARTER.html"))

    model_rows = [
        ["<b>A′ Sparse Jump Model</b>", "Regroupe les jours semblables et <b>fait payer chaque "
         "changement d'état</b>&nbsp;; sélectionne lui-même ses variables", "2 états, pénalité λ "
         "recalibrée toutes les 4 réestimations, ≈ 10 variables effectives", "Modèle principal"],
        ["A Jump model", "Même principe, sans sélection de variables", "2 états, λ recalibrée "
         "toutes les 4 réestimations", "Variante"],
        ["B HMM gaussien", "Chaîne de Markov cachée, lois gaussiennes par état, probabilité "
         "<b>filtrée</b>", "2 états, matrice de transition", "Référence classique"],
        ["C Gradient boosting", "Pas d'état caché&nbsp;: prédit la volatilité à 21 jours, puis la "
         "coupe à sa médiane", "LightGBM, seuil appris sur le passé", "Défi du machine learning"],
        ["C′ HAR-RV", "Modèle linéaire de volatilité (1, 5, 21 jours), coupé de même",
         "Régression à 3 termes", "Référence de la littérature"],
    ]
    s.append(slide(6, "Les cinq modèles", "Deux familles à états cachés, deux prédicteurs directs",
        table(["Modèle", "Comment ça marche", "Réglages", "Rôle"], model_rows, highlight=0,
              cls="wide")
        + """<div class="note">C et C′ sont <b>supervisés</b>&nbsp;: ils apprennent, sur le passé,
        à prédire directement la volatilité future. A, A′ et B ne voient jamais cette cible. Les
        premiers sont donc avantagés sur la prévision de volatilité&nbsp;: c'est voulu, pour
        vérifier si la notion d'«&nbsp;état caché&nbsp;» apporte quelque chose.</div>""", "regime_lab/models/, docs/CHARTER.html §03"))

    s.append(slide(7, "Le modèle principal", "Le Sparse Jump Model&nbsp;: regrouper, stabiliser, sélectionner",
        """<div class="two">
          <div>
            <div class="formula">min<sub>&nbsp;états, centres, poids</sub> &nbsp;
            Σ<sub>t</sub> ‖ x<sub>t</sub> − μ<sub>s<sub>t</sub></sub> ‖<sup>2</sup><sub>w</sub>
            &nbsp;+&nbsp; <b>λ</b> · Σ<sub>t</sub> 1{ s<sub>t</sub> ≠ s<sub>t−1</sub> }</div>
            <div class="formula-caption">avec des poids de variables <i>w</i> contraints en
            norme L1&nbsp;: le nombre effectif de variables est fixé à 10&nbsp;; une vingtaine gardent un
            poids, dix en portent l'essentiel.</div>
            <ol class="steps3">
              <li><b>Regrouper.</b> Comme un k-moyennes, il affecte chaque jour à l'état dont le
              profil (centre μ) lui ressemble le plus.</li>
              <li><b>Stabiliser.</b> La pénalité λ fait «&nbsp;payer&nbsp;» chaque changement
              d'état&nbsp;: les régimes durent, le signal ne clignote pas.</li>
              <li><b>Sélectionner.</b> La contrainte L1 annule le poids des variables inutiles&nbsp;:
              le modèle reste lisible.</li>
            </ol>
          </div>
          <div class="side">
            <h3>Réglages retenus</h3>
            <ul class="kvlist">
              <li><span>États</span><b>2 (stress, calme)</b></li>
              <li><span>Pénalité λ</span><b>choisie dans {1, 3, 10, 30, 100, 300} toutes les 4
              réestimations&nbsp;; repli à 20 depuis 2022</b></li>
              <li><span>Variables</span><b>10 effectives sur 50</b></li>
              <li><span>Réestimation</span><b>tous les 6 mois, passé seul</b></li>
              <li><span>Ordre des états</span><b>par la volatilité d'entraînement</b></li>
            </ul>
            <h3>Références</h3>
            <p class="ref">Nystrup, Lindström &amp; Madsen (2020)&nbsp;; Nystrup, Kolm &amp; Lindström (2021)&nbsp;; Aydınhan, Kolm, Mulvey
            &amp; Shu (2024). Implémentation de référence des auteurs (jumpmodels).</p>
          </div>
        </div>""", "regime_lab/models/jump.py, regime_lab/models/calibrate.py"))

    side_rows = [["A′ Sparse Jump", "456 j", "0,53", "0,6 %"], ["A Jump model", "456 j", "0,53",
                 "0,9 %"], ["B HMM filtré", "128 j", "2,01", "0,7 %"], ["C Gradient boosting",
                 "18 j", "14,08", "0,0 %"], ["C′ HAR-RV", "14 j", "18,41", "0,0 %"],
                 ["Témoin volatilité", "34 j", "6,75", "0,0 %"]]
    s.append(slide(8, "Résultat 1 — Qualité de classification",
        "93&nbsp;% d'exactitude face aux récessions, sans voir le futur",
        f"""<div class="two chart-left">
          <div><div class="chart-title">Exactitude équilibrée contre les récessions NBER
          — pointillé&nbsp;: hasard (50&nbsp;%)</div>{accuracy_chart()}</div>
          <div><div class="chart-title">Stabilité des états</div>
          {table(["Modèle", "Durée moy.", "Chgts / an", "Instabilité"], side_rows, num={1, 2, 3},
                 highlight=0, cls="compact")}
          <div class="note small">Instabilité&nbsp;: part des étiquettes qui changeraient si l'on
          connaissait la suite du semestre. Latence mesurée contre les pics NBER&nbsp;: 0 à 13
          jours.</div>
          <div class="note small"><b>À garder en tête&nbsp;:</b> la période de test ne contient que
          <b>deux récessions</b> (2008-09 et 2020, 20 mois). «&nbsp;Sans voir le futur&nbsp;» ne veut
          pas dire avant le marché&nbsp;: au Covid, le VIX était déjà passé de 14 à 54.</div>
          </div></div>""", "docs/RESULTS_FINAL.md (couche 1), data/cache/states.parquet"))

    s.append(slide(9, "Résultat 1 — Les régimes dans le temps",
        "13 changements d'état en 24 ans, calés sur les grandes crises",
        f"""<div class="chart-title">S&amp;P 500, échelle logarithmique, avril 2002 – septembre 2026
        <span class="legend inline"><span><i style="background:{RED};opacity:.3"></i>état de
        stress du Sparse Jump Model</span></span></div>{timeline()}
        <div class="note">Le modèle est resté en état calme depuis avril 2021&nbsp;: la baisse de
        2022 (−25&nbsp;% sur le S&amp;P 500, choc de taux) n'a pas été classée comme un régime de
        stress. Treize décisions en 24 ans&nbsp;: c'est un indicateur de fond, pas un signal de
        trading.</div>""", "data/cache/states.parquet, data/raw/prices/cross_asset.parquet"))

    s.append(slide(10, "Résultat 2 — Ce que l'état apporte",
        "Il informe sur le risque futur, pas sur la direction",
        f"""<div class="two chart-left">
          <div><div class="chart-title">R² incrémental au-delà du témoin, en points, horizon 21 jours</div>
          <div class="legend"><span><i style="background:{BLUE}"></i>volatilité future</span>
          <span><i style="background:{ORANGE}"></i>rendements futurs</span></div>
          {grouped_r2()}</div>
          <div class="side">
            <h3>Lecture</h3>
            <p>Sur la <b>volatilité</b>, quatre modèles sur cinq ajoutent 2 à 4 points de R² au-delà
            de ce que le témoin sait déjà (t de −3,4 à −6,3). HAR-RV n'ajoute presque rien, le
            témoin rien du tout.</p>
            <p>Sur les <b>rendements</b>, aucun modèle n'ajoute quoi que ce soit (t entre −0,3 et
            0,5).</p>
            <p class="callout"><b>Mais le VIX le savait déjà.</b> Dans un test à part (test P),
            l'état ajoute +4,48 points au-delà d'un rang de volatilité, mais +0,20 seulement
            (t −1,34) une fois le VIX ajouté.</p>
          </div></div>""", "docs/RESULTS_FINAL.md (couche 2), docs/RESULTS_CRISE.md (test P)"))

    pf_rows = [["60/40 seul", "0,47", "—", "−35,6 %"],
               ["A′ Sparse Jump", "0,55", "0,49", "−22,3 %"],
               ["A Jump model", "0,54", "0,50", "−22,3 %"],
               ["B HMM filtré", "0,47", "0,50", "−13,7 %"],
               ["C Gradient boosting", "0,48", "0,52", "−10,2 %"],
               ["C′ HAR-RV", "0,57", "0,56", "−13,4 %"]]
    s.append(slide(11, "Résultat 3 — En portefeuille",
        "Moins de pertes en crise, mais aucun gain de Sharpe démontrable",
        f"""<div class="two chart-left">
          <div><div class="chart-title">Portefeuille 60/40 piloté par l'état, sans coût,
          avril 2002 – septembre 2026</div>
          {table(["Modèle", "Sharpe, tout ou rien", "Sharpe, taille", "Perte max., tout ou rien"],
                 pf_rows, num={1, 2, 3}, highlight=1, cls="compact")}
          <div class="note small">Tout ou rien&nbsp;: investi en calme, en liquidités en stress.
          Taille&nbsp;: exposition réduite en stress. À 2 pb aller-retour, les Sharpe baissent de
          0 à 0,07&nbsp;; les conclusions ne changent pas.</div></div>
          <div class="side">
            <h3>Lecture</h3>
            <p>Couper en stress <b>réduit la perte maximale</b> (−22&nbsp;% contre −36&nbsp;%),
            surtout parce qu'on est moins investi.</p>
            <p>Les écarts de Sharpe restent <b>sous le seuil détectable</b> sur cet échantillon
            (0,27 à 0,40)&nbsp;: aucun n'est démontré.</p>
            <p class="callout">La valeur vient du <b>ciblage de volatilité</b> (+0,44&nbsp;%/an),
            pas du régime lui-même (−0,04&nbsp;%/an avec recul, −0,25&nbsp;% net de la latence).</p>
          </div></div>""", "docs/RESULTS_FINAL.md (couche 3), docs/RESULTS_DECOMPOSITION.md"))

    s.append(slide(12, "Pourquoi le Sparse Jump Model",
        "Choisi avant de mesurer, confirmé par les mesures",
        """<div class="two even">
          <div class="card tall"><h3>Choisi avant de mesurer</h3>
            <ul class="ticks">
              <li>Désigné modèle principal dans le cadrage gelé, sur la base de la littérature
              récente sur les régimes (Nystrup et al., Aydınhan et al.).</li>
              <li><b>Persistance réglée explicitement</b> par λ&nbsp;: peu de changements, donc
              peu de rotation et peu de coûts.</li>
              <li><b>Sélection de variables intégrée</b>&nbsp;: l'essentiel du poids sur une dizaine de
              variables, donc un modèle lisible.</li>
              <li>Pas d'hypothèse de loi explicite, contrairement au HMM gaussien.</li>
            </ul></div>
          <div class="card tall"><h3>Confirmé par les mesures</h3>
            <ul class="ticks">
              <li><b>Meilleur accord avec les récessions</b>&nbsp;: kappa 0,53, le plus élevé des
              cinq&nbsp;; exactitude 93,2&nbsp;% (à égalité avec A).</li>
              <li><b>Plus forte information sur le risque futur</b>&nbsp;: +3,93 points de R²
              au-delà d'une règle de volatilité, t = −3,40 (mais +0,20 au-delà du VIX).</li>
              <li><b>Fiable en temps réel</b>&nbsp;: 0,6&nbsp;% d'étiquettes instables, 0 à 13
              jours de latence.</li>
              <li>Les modèles plus réactifs (C, C′) changent d'état 14 à 18 fois par an mais
              n'atteignent que 75-78&nbsp;% d'exactitude face aux récessions.</li>
            </ul></div>
        </div>
        <div class="limit"><b>Sa limite, à assumer&nbsp;:</b> il ne prédit pas la direction du
        marché, et sa lenteur (environ un changement tous les deux ans) en fait un thermomètre du
        risque plutôt qu'un signal de trading. Le rendre plus réactif le fait ressembler à une
        simple règle de volatilité.</div>""",
        "docs/CHARTER.html §03, docs/RESULTS_FINAL.md, docs/EXPLORATION_SJM_SPEED.md"))

    s.append(slide(13, "Application — Coupler le filtre à des stratégies réelles",
        "Couplé à 8 stratégies, le filtre n'en améliore aucune",
        f"""<div class="two chart-left">
          <div><div class="chart-title">Variation du Sharpe quand on coupe la stratégie en état de
          stress, sans coût</div>
          <div class="legend"><span><i style="background:{RED}"></i>pénalisée (t HAC &lt; −2)</span>
          <span><i style="background:{GREY}"></i>pas utile ou neutre</span></div>
          {coupling_chart()}</div>
          <div class="side">
            <h3>Protocole</h3>
            <p>Trois couplages testés par stratégie&nbsp;: arrêt, réduction de moitié, bascule vers
            un livre de tendance. Critère écrit avant la lecture, placebo, témoin de volatilité.</p>
            <h3>Résultat</h3>
            <p><b>0 couplage utile sur 24.</b> Les deux stratégies pénalisées sont celles qui
            <b>gagnent en période agitée</b>&nbsp;: la prime overnight (placebo&nbsp;: les 400
            placements aléatoires font mieux) et la tendance crypto (marché haussier de fin 2020).</p>
            <p class="callout">Le stress n'est pas un mauvais moment pour toutes les stratégies.</p>
          </div></div>""", "docs/RESULTS_COUPLAGE_STRATEGIES.md, docs/RESULTS_B1_COUPLAGE.md"))

    s.append(slide(14, "Conclusion", "Ce qu'il faut retenir",
        """<div class="concl">
          <div class="cl"><div class="cn">1</div><div><h3>La classification de régimes
          fonctionne.</h3><p>Le Sparse Jump Model reconnaît les crises sans voir le futur
          (93&nbsp;% d'exactitude contre le NBER, sur deux récessions), avec des états stables et
          lisibles.</p></div></div>
          <div class="cl"><div class="cn">2</div><div><h3>Elle mesure le risque, pas la
          direction.</h3><p>Il informe sur la volatilité future, pas sur les rendements, et le
          VIX en savait déjà presque autant.</p></div></div>
          <div class="cl"><div class="cn">3</div><div><h3>Comme filtre de trading, elle ne
          paie pas.</h3><p>Aucune amélioration démontrée sur 62 usages et 13 stratégies&nbsp;;
          une simple règle de volatilité fait aussi bien. Le meilleur cas, momentum + or en
          stress, reste sous le seuil de détection.</p></div></div>
          <div class="cl"><div class="cn">→</div><div><h3>Testé ensuite.</h3><p>Un historique depuis
          1926 (le modèle réduit n'y reconnaît que 6 récessions sur 14) et le régime comme budget
          de risque d'un portefeuille (aucun gain démontré)&nbsp;: voir le dossier.</p></div></div>
        </div>
        <div class="banner">Le régime décrit le <b>risque</b>, pas la <b>direction</b>&nbsp;:
        c'est un outil de dimensionnement, pas un signal d'entrée ou de sortie.</div>""",
        "synthèse des documents du dépôt regime-lab"))

    # --- appendix -----------------------------------------------------------
    fam_rows = [["Momentum", "7", "rendement actions 21/63/252 j, accélération, pétrole, dollar, largeur 200 j"],
                ["Volatilité", "9", "volatilité réalisée 5/21/63 j, structure par terme, vol de vol, "
                 "semi-variance, sauts, VIX, prime de variance"],
                ["Asymétrie et mémoire", "6", "skewness, kurtosis, drawdown 252 j, exposant de Hurst, "
                 "ratios de variance"],
                ["Transversal", "8", "dispersion sectorielle et taille × valeur, corrélation moyenne, "
                 "concentration factorielle, largeur, SMB, HML"],
                ["Macroéconomie", "9", "production industrielle, emploi, inflation, chômage (règle de "
                 "Sahm), inscriptions au chômage"],
                ["Crédit", "4", "spread Baa, qualité Baa − Aaa, et leurs variations"],
                ["Conditions financières", "2", "indice NFCI et sa variation sur 13 semaines"],
                ["Taux", "5", "pentes 10 ans − 2 ans et 10 ans − 3 mois et leurs variations, variation "
                 "du taux court sur 12 mois"]]
    s.append(slide(15, "Annexe A — Les 50 variables", "Huit familles de variables, toutes calculées sur le passé seul",
        table(["Famille", "Nb", "Contenu"], fam_rows, num={1}, cls="wide"),
        "data/cache/features.parquet, docs/FEATURE_NOTES.md", cls="appendix"))

    c_rows = [["Rebond obligataire fin de mois", "2004-2026", "0,85", "+0,01", "+0,02", "−0,04", "neutre"],
              ["Tendance or", "2007-2026", "0,62", "−0,04", "−0,01", "−0,05", "pas utile"],
              ["Tendance or + argent", "2004-2026", "0,58", "−0,06", "−0,02", "−0,07", "pas utile"],
              ["Momentum USDJPY", "2009-2026", "0,50", "−0,01", "0,00", "+0,03", "neutre"],
              ["Tendance énergie (long)", "2011-2026", "−0,17", "−0,11", "−0,06", "−0,02", "pas utile"],
              ["Tendance crypto", "2014-2026", "1,13", "−0,24", "−0,11", "−0,21", "pénalisée"],
              ["Prime overnight Nasdaq", "2010-2026", "1,10", "−0,12", "−0,03", "−0,11", "pénalisée"],
              ["Cassure d'ouverture (ORB) Nasdaq", "2010-2026", "0,64", "0,00", "+0,01", "+0,03", "neutre"]]
    s.append(slide(16, "Annexe B — Les 24 couplages", "Détail des couplages stratégie × filtre",
        table(["Stratégie", "Période", "Sharpe seule", "Δ arrêt", "Δ réduction", "Δ bascule",
               "Verdict"], c_rows, num={2, 3, 4, 5}, cls="wide")
        + """<div class="note">Brut de coûts (hypothèse de simplicité du projet). «&nbsp;Utile&nbsp;»
        exige quatre conditions écrites avant la lecture&nbsp;: gain au-delà de l'écart détectable
        corrigé du nombre de tests, test HAC de même signe, au-dessus de 95&nbsp;% d'un placebo,
        meilleur que la règle de volatilité. Aucun couplage ne les remplit.</div>""",
        "docs/RESULTS_COUPLAGE_STRATEGIES.md", cls="appendix"))

    s.append(slide(17, "Annexe C — Limites", "Ce que ces résultats ne disent pas",
        """<ul class="ticks big">
          <li><b>Trois épisodes de stress et deux récessions seulement</b> en 24 ans&nbsp;: les
          93&nbsp;% et la plupart des conclusions reposent sur très peu de crises.</li>
          <li><b>Le VIX sait déjà</b>&nbsp;: dans le test P, l'apport de l'état passe de +4,48
          à +0,20 point sur la volatilité future une fois le VIX ajouté.</li>
          <li><b>Calme continu depuis avril 2021</b>&nbsp;: la baisse de 2022 n'a pas été signalée,
          et les stratégies récentes ne voient qu'une crise, 2020.</li>
          <li><b>Couplages bruts de coûts</b>&nbsp;: les Sharpe des stratégies sont optimistes.</li>
          <li><b>Portée</b>&nbsp;: un classifieur à deux états, lent, utilisé pour couper ou
          dimensionner. La sélection entre signaux a été testée à part (plan Two Sigma, avec
          un autre classifieur) et ne marche pas mieux.</li>
          <li><b>Validation NBER rétrospective</b>&nbsp;: les dates de récession sont publiées après
          coup&nbsp;; seuls les états du modèle sont en temps réel.</li>
        </ul>""", "docs/RESULTS_FINAL.md, AVANCEMENT.md", cls="appendix"))

    s.append(slide(18, "Annexe D — Où trouver quoi", "Tout est reproductible depuis le dépôt regime-lab",
        table(["Pour…", "Fichier"], [
            ["Vue d'ensemble et inventaire des données", "AVANCEMENT.md, README.md"],
            ["Résultats de classification (3 couches)", "docs/RESULTS_FINAL.md"],
            ["Protocole gelé et écarts", "docs/CHARTER.html, docs/PROTOCOL_FREEZE.md"],
            ["Couplage aux stratégies", "docs/RESULTS_COUPLAGE_STRATEGIES.md, docs/RESULTS_B1_COUPLAGE.md"],
            ["Tableau de synthèse (tableur)", "docs/presentation/synthese_modeles_regimes.csv"],
            ["États quotidiens des 5 modèles", "data/cache/states.parquet"],
            ["Code des modèles", "regime_lab/models/"],
            ["Ce document", "scripts/build_presentation.py"]], cls="wide")
        + """<div class="note">Dépôt public&nbsp;: github.com/Guillaume-Beaudouin-Git/regime-lab.
        Environnement&nbsp;: <code>uv sync --all-packages --extra dev</code>, puis
        <code>.venv/bin/python -m pytest -q</code>.</div>""", "", cls="appendix"))
    return PAGE.replace("{{SLIDES}}", "".join(s))


PAGE = """<!doctype html><html lang="fr"><head><meta charset="utf-8">
<title>Régimes de marché — présentation</title>
<style>
@page { size: 13.333in 7.5in; margin: 0; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: #fff; }
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; color: #0b0b0b;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
  font-variant-numeric: tabular-nums; }
.slide { width: 13.333in; height: 7.5in; position: relative; overflow: hidden;
  padding: 0.5in 0.72in 0.62in; page-break-after: always; break-after: page; background: #fff; }
.slide::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 0.12in;
  background: #0f2a44; }
.slide.appendix::before { background: #898781; }
.eyebrow { font: 600 10.5pt "Avenir Next", "Helvetica Neue", sans-serif; letter-spacing: 0.12em;
  text-transform: uppercase; color: #2a78d6; margin-bottom: 0.08in; }
.appendix .eyebrow { color: #52514e; }
h1 { font: 600 28pt/1.15 "Avenir Next", "Helvetica Neue", sans-serif; color: #0f2a44;
  margin: 0 0 0.3in; letter-spacing: -0.01em; max-width: 11.8in; }
h3 { font: 600 14.5pt/1.25 "Avenir Next", "Helvetica Neue", sans-serif; color: #0f2a44;
  margin: 0 0 0.07in; }
p, li { font-size: 15pt; line-height: 1.45; color: #1b1f24; margin: 0 0 0.08in; }
b { font-weight: 600; }
.body { position: relative; }
footer { position: absolute; left: 0.72in; right: 0.72in; bottom: 0.26in; display: flex;
  align-items: baseline; gap: 0.3in; border-top: 1px solid #e1e0d9; padding-top: 0.08in; }
footer .src { flex: 1; font-size: 8.5pt; color: #898781; }
footer .brand { font-size: 8.5pt; color: #898781; letter-spacing: 0.04em; }
footer .pn { font: 600 9pt "Avenir Next", sans-serif; color: #0f2a44; min-width: 0.25in;
  text-align: right; }
table { border-collapse: collapse; width: 100%; }
th { font: 600 10pt "Avenir Next", sans-serif; text-transform: uppercase; letter-spacing: 0.06em;
  color: #52514e; text-align: left; padding: 0.07in 0.1in; border-bottom: 1.5px solid #0f2a44; }
td { font-size: 14pt; line-height: 1.34; padding: 0.1in 0.12in; border-bottom: 1px solid #e1e0d9;
  vertical-align: top; }
td.num, th.num { text-align: right; white-space: nowrap; }
tr.hl td { background: #eef4fc; }
tr.hl td:first-child { box-shadow: inset 3px 0 0 #2a78d6; font-weight: 600; }
table.compact td { font-size: 13pt; padding: 0.08in 0.1in; }
table.compact td:first-child { white-space: nowrap; }
table.data td:nth-child(3), table.data td:nth-child(4) { white-space: nowrap; }
.note { font-size: 13pt; line-height: 1.45; color: #52514e; margin-top: 0.24in;
  border-left: 3px solid #e1e0d9; padding-left: 0.14in; }
.note.small { font-size: 11.5pt; }
.card { background: #f7f8fa; border: 1px solid #e6e8ec; border-radius: 6px; padding: 0.22in 0.24in; }
.card p { font-size: 13.5pt; margin: 0; }
.card.tall { min-height: 3.7in; }
.grid4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.24in; }
.facts { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.24in; margin-top: 0.36in; }
.facts div { border-top: 3px solid #0f2a44; padding-top: 0.1in; }
.facts b { display: block; font: 600 28pt/1.1 "Avenir Next", sans-serif; color: #0f2a44; }
.facts span { display: block; font-size: 12.5pt; line-height: 1.35; color: #52514e; margin-top: 0.05in; }
.grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.24in; margin-top: 0.36in; }
.two { display: grid; grid-template-columns: 1.35fr 1fr; gap: 0.45in; align-items: start; }
.two.even { grid-template-columns: 1fr 1fr; }
.two.chart-left { grid-template-columns: 1.45fr 1fr; }
.side h3 { margin-top: 0.02in; }
.side p { font-size: 14pt; }
.callout { background: #eef4fc; border-left: 3px solid #2a78d6; padding: 0.1in 0.14in;
  border-radius: 0 4px 4px 0; }
.question { font: 500 20pt/1.35 "Avenir Next", sans-serif; color: #0f2a44; border-left: 4px solid
  #2a78d6; padding: 0.06in 0 0.06in 0.2in; margin: 0.05in 0 0.34in; max-width: 11in; }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.3in; margin-bottom: 0.45in; }
.kpi { border-top: 3px solid #0f2a44; padding-top: 0.12in; }
.kv { font: 600 38pt/1 "Avenir Next", sans-serif; color: #0f2a44; }
.kpi:first-child .kv, .kpi:nth-child(2) .kv { color: #2a78d6; }
.kl { font: 600 14pt/1.3 "Avenir Next", sans-serif; margin: 0.08in 0 0.05in; color: #0b0b0b; }
.kd { font-size: 12pt; line-height: 1.35; color: #52514e; }
ol.msgs { margin: 0; padding-left: 0.28in; }
ol.msgs li { margin-bottom: 0.16in; font-size: 16pt; }
.flow { display: flex; align-items: stretch; gap: 0.08in; }
.step { flex: 1; background: #0f2a44; color: #fff; border-radius: 6px; padding: 0.2in 0.2in; }
.step h3 { color: #fff; font-size: 15pt; }
.step p { color: #d5dde7; font-size: 12.5pt; margin: 0; line-height: 1.35; }
.step .sn { font: 600 9pt "Avenir Next", sans-serif; color: #86b6ef; margin-bottom: 0.04in; }
.arrow { align-self: center; color: #898781; font-size: 16pt; }
.formula { white-space: nowrap; font: 17pt/1.6 "Times New Roman", Times, serif; color: #0b0b0b; background: #f7f8fa;
  border: 1px solid #e6e8ec; border-radius: 6px; padding: 0.16in 0.22in; }
.formula-caption { font-size: 12.5pt; color: #52514e; margin: 0.06in 0 0.2in; }
ol.steps3 { margin: 0; padding-left: 0.28in; }
ol.steps3 li { margin-bottom: 0.14in; }
ul.kvlist { list-style: none; padding: 0; margin: 0 0 0.2in; }
ul.kvlist li { display: flex; justify-content: space-between; gap: 0.2in; border-bottom: 1px solid
  #e1e0d9; padding: 0.08in 0; font-size: 13pt; margin: 0; }
ul.kvlist li span { color: #52514e; white-space: nowrap; }
ul.kvlist li b { text-align: right; }
.ref { font-size: 12.5pt; color: #52514e; }
ul.ticks { padding-left: 0.2in; margin: 0; }
ul.ticks li { font-size: 14pt; margin-bottom: 0.14in; }
ul.ticks.big li { font-size: 16pt; margin-bottom: 0.22in; }
.limit { margin-top: 0.22in; background: #fdf1f0; border-left: 3px solid #e34948;
  padding: 0.16in 0.2in; font-size: 14pt; border-radius: 0 4px 4px 0; }
.chart-title { font: 600 13pt "Avenir Next", sans-serif; color: #52514e; margin-bottom: 0.1in; }
.legend { display: flex; gap: 0.3in; font-size: 12pt; color: #0b0b0b; margin: 0 0 0.08in; }
.legend.inline { display: inline-flex; margin: 0 0 0 0.3in; font-weight: 400; }
.legend i { display: inline-block; width: 0.16in; height: 0.16in; border-radius: 3px;
  vertical-align: -0.02in; margin-right: 0.07in; }
.concl { display: grid; grid-template-columns: 1fr 1fr; gap: 0.45in 0.5in; }
.cl { display: flex; gap: 0.18in; }
.banner { margin-top: 0.5in; background: #0f2a44; color: #fff; border-radius: 6px;
  padding: 0.22in 0.3in; font: 500 17pt/1.4 "Avenir Next", sans-serif; }
.banner b { color: #86b6ef; }
.cn { font: 600 30pt/1 "Avenir Next", sans-serif; color: #2a78d6; min-width: 0.36in; }
.cl p { font-size: 15pt; }
.cl h3 { font-size: 17pt; }
code { font: 11.5pt Menlo, monospace; background: #f1f2f4; padding: 0 0.04in; border-radius: 3px; }
.cover { background: #0f2a44; padding: 0; }
.cover::before { display: none; }
.cover-band { position: absolute; left: 0; right: 0; bottom: 0; height: 0.14in; background: #2a78d6; }
.cover-inner { position: absolute; left: 0.9in; right: 1.6in; top: 1.7in; }
.eyebrow.light { color: #86b6ef; }
.cover-title { font: 600 44pt/1.12 "Avenir Next", sans-serif; color: #fff; margin: 0.1in 0 0.3in;
  max-width: 10in; }
.cover-sub { font-size: 17pt; line-height: 1.45; color: #d5dde7; max-width: 9.6in; }
.cover-sub b { color: #fff; }
.cover-meta { margin-top: 0.5in; font: 500 10.5pt "Avenir Next", sans-serif; letter-spacing: 0.06em;
  color: #86b6ef; text-transform: uppercase; }
</style></head><body>{{SLIDES}}</body></html>"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    page = OUT_DIR / "presentation_regimes.html"
    pdf = OUT_DIR / "presentation_regimes.pdf"
    page.write_text(build_html(), encoding="utf-8")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", page.as_uri()], check=True, capture_output=True)
    print(f"{pdf.relative_to(ROOT)}  ({pdf.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
