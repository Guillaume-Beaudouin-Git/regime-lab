"""Write the presentation synthesis: inputs, how each model works, results, why A′.

One CSV, long format, meant to be filtered by its first column in a spreadsheet:
    section ; élément ; indicateur ; valeur ; explication ; source

Figures that can be recomputed from the data here (transitions per year, share of
time in stress, feature counts, the volatility rule's clock) are recomputed on every
run. Every other figure is transcribed from the committed results document named in
its `source` column, never from memory. Semicolon-separated, UTF-8 with BOM, so it
opens cleanly in a French-locale Excel.

Usage: .venv/bin/python scripts/export_synthese_modeles.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_b1_regime_coupling as b1  # noqa: E402

from regime_lab.config import CACHE  # noqa: E402
from regime_lab.evaluation.predictive import volatility_quantile_placebo  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "presentation" / "synthese_modeles_regimes.csv"
FINAL = "docs/RESULTS_FINAL.md"
CHARTER = "docs/CHARTER.html §03"
MODELS = ["A  jump", "A' sparse jump", "B  filtered HMM", "C  gradient boost", "C' HAR-RV"]
LABEL = {
    "A  jump": "A — Jump model",
    "A' sparse jump": "A′ — Sparse Jump Model (retenu)",
    "B  filtered HMM": "B — HMM gaussien filtré",
    "C  gradient boost": "C — Gradient boosting (LightGBM)",
    "C' HAR-RV": "C′ — HAR-RV",
    "placebo": "Témoin — règle de volatilité d'une ligne",
}


def fr(x: float, digits: int = 2, pct: bool = False) -> str:
    text = f"{x * 100:.1f} %" if pct else f"{x:.{digits}f}"
    return text.replace(".", ",").replace("-", "−")


def measured() -> dict:
    states = pd.read_parquet(CACHE / "states.parquet").dropna()
    years = (states.index[-1] - states.index[0]).days / 365.25
    out = {"start": states.index[0], "end": states.index[-1], "n": len(states), "years": years}
    for c in MODELS:
        x = states[c]
        transitions = int((x != x.shift()).sum() - 1)
        out[c] = {"per_year": transitions / years, "transitions": transitions,
                  "stress": float((x == 0).mean())}
    spx = b1.us_prices("eq_us_large")
    rule = volatility_quantile_placebo(spx.pct_change()).reindex(states.index).ffill()
    transitions = int((rule != rule.shift()).sum() - 1)
    out["placebo"] = {"per_year": transitions / years, "transitions": transitions,
                      "stress": float((rule == 0).mean())}
    features = pd.read_parquet(CACHE / "features.parquet")
    out["features"] = Counter(c.split("_")[0] for c in features.columns)
    out["n_features"] = features.shape[1]
    return out


def rows(m: dict) -> list[tuple[str, str, str, str, str, str]]:
    r: list[tuple[str, str, str, str, str, str]] = []
    add = r.append
    sessions = f"{m['n']:,}".replace(",", " ")
    period = (f"{m['start']:%d/%m/%Y} → {m['end']:%d/%m/%Y}, {sessions} séances, "
              f"{fr(m['years'], 1)} ans")

    s = "0. Mode d'emploi"
    add((s, "Ce fichier", "Contenu",
         "Données d'entrée, fonctionnement des modèles, résultats de chaque modèle, pourquoi le "
         "Sparse Jump Model", "Filtrer sur la colonne « section ». La colonne « source » dit d'où "
         "vient chaque chiffre.", "scripts/export_synthese_modeles.py"))
    add((s, "Question du projet", "Énoncé",
         "Un modèle de régimes appris par machine apporte-t-il une information exploitable au-delà "
         "d'une simple mesure de volatilité ?", "Et résiste-t-elle aux coûts, aux données "
         "disponibles en temps réel et au nombre de tests ?", "README.md"))
    add((s, "Échantillon d'évaluation", "Période", period,
         "Hors échantillon : aucune de ces séances n'a servi à entraîner les modèles au moment où "
         "ils les classent", "data/cache/states.parquet"))

    s = "1. Données d'entrée"
    data = [
        ("Prix de marché", "8 séries : S&P 500 (^GSPC), Russell 2000 (^RUT), Nasdaq (^IXIC), "
         "Nikkei (^N225), taux US 30 ans (^TYX) et 10 ans (^TNX), indice dollar (DX-Y.NYB), "
         "VIX (^VIX)", "Yahoo Finance, clôture ajustée", "01/01/1990 → 08/09/2026",
         "Variables de momentum, de volatilité et d'asymétrie ; rendements du portefeuille de "
         "référence", "data/raw/prices/cross_asset.parquet"),
        ("Pétrole WTI", "DCOILWTICO, quotidien", "FRED", "02/01/1990 → 01/09/2026",
         "Momentum du pétrole", "data/raw/prices/fred_daily.parquet"),
        ("Macroéconomie", "Production industrielle, emplois, inflation, chômage en **premières "
         "publications** (point-in-time) ; inscriptions au chômage, NFCI (hebdo) ; spreads Baa et "
         "Aaa, taux 3 mois, pentes 10a−2a et 10a−3m (quotidien)", "FRED / ALFRED",
         "1990 → 07-09/2026 selon la série",
         "Variables macro, crédit, conditions financières, taux ; taux 3 mois pour les rendements "
         "en excès", "data/raw/macro/*.parquet (11 fichiers)"),
        ("Panels d'actions", "5 facteurs Fama-French, 49 portefeuilles sectoriels, "
         "25 portefeuilles taille × valeur, quotidien", "Ken French Data Library",
         "02/01/1990 → 31/07/2026",
         "Variables transversales : dispersion, corrélation moyenne, concentration factorielle",
         "data/raw/panels/*.parquet"),
        ("Récessions NBER", "Indicateur mensuel USREC", "FRED", "01/1990 → 08/2026",
         "**Validation seulement** : les modèles ne le voient jamais", "data/raw/references/"
         "ref_nber.parquet"),
    ]
    for name, content, source, span, role, path in data:
        add((s, name, "Contenu", content, role, path))
        add((s, name, "Provenance et période", f"{source} — {span}", "", path))
    add((s, "Principe point-in-time", "Règle",
         "Chaque observation porte deux dates : la période décrite et la date où on pouvait la "
         "connaître. Une valeur n'entre dans un modèle qu'à partir de sa date de publication.",
         "Évite de regarder dans le futur via les révisions macro. Vérifié par tests/test_pit.py",
         "README.md, docs/DATA_NOTES.md"))
    add((s, "Régénérer les données", "Commandes",
         "scripts/fetch_data.py (clé API FRED gratuite dans .env), puis scripts/build_features.py",
         "Ou décompresser l'archive donnees_regime-lab_2026-09-22.zip à la racine du dépôt",
         "AVANCEMENT.md §3"))

    s = "2. Variables construites"
    fam = {
        "mom": ("Momentum", "rendement actions 21/63/252 j, accélération, pétrole, dollar, largeur "
                "200 j"),
        "vol": ("Volatilité", "volatilité réalisée 5/21/63 j, structure par terme, vol de vol, "
                "semi-variance, part des sauts, VIX, prime de variance"),
        "asy": ("Asymétrie et mémoire", "skewness, kurtosis, drawdown 252 j, exposant de Hurst, "
                "ratios de variance 5/20 j"),
        "xs": ("Transversal", "dispersion des secteurs et des portefeuilles taille × valeur, "
               "corrélation moyenne, absorption factorielle et sa variation, largeur, SMB, HML"),
        "mac": ("Macroéconomie", "production industrielle, emplois, inflation (sur 12 et 3 mois), "
                "chômage (variation, règle de Sahm), inscriptions au chômage"),
        "cre": ("Crédit", "spread Baa, qualité (Baa − Aaa) et leurs variations sur 63 j"),
        "fin": ("Conditions financières", "NFCI et sa variation sur 13 semaines"),
        "rat": ("Taux", "pentes 10a−2a et 10a−3m et leurs variations, variation du taux cash sur "
                "12 mois"),
    }
    add((s, "Ensemble", "Nombre de variables", f"{m['n_features']} variables, 8 familles",
         "Standardisées en fenêtre expansive (sans regarder le futur)",
         "data/cache/features.parquet, docs/FEATURE_NOTES.md"))
    for prefix, (name, what) in fam.items():
        add((s, f"{name} ({prefix}_*)", "Nombre et contenu", f"{m['features'][prefix]} variables",
             what, "data/cache/features.parquet"))

    s = "3. Protocole commun"
    for item, value, why in [
        ("Cadrage gelé", "Protocole, familles et règles de décision figés avant tout résultat",
         "Empreinte SHA-256 et étiquette publique charter-v2 ; écarts dans "
         "docs/PROTOCOL_FREEZE.md"),
        ("Réestimation", "Tous les 6 mois, fenêtre expansive depuis 1990",
         "Même calendrier pour toutes les familles, pour que la comparaison soit loyale"),
        ("Prédiction", "En ligne (filtrée) entre deux réestimations, jamais lissée",
         "Un état lissé utilise des données futures : c'est l'erreur la plus fréquente des "
         "backtests de régimes publiés"),
        ("Variables", "Le Sparse Jump Model sélectionne au plus 10 variables sur la fenêtre "
         "d'entraînement ; toutes les familles reçoivent ce même sous-ensemble",
         "Compare des modèles, pas des sélections de variables"),
        ("États", "2 états, ordonnés par la volatilité d'entraînement : 0 = stress, 1 = calme",
         "Ordonner par le rendement inversait le meilleur modèle (erreur corrigée le 08/09)"),
        ("Règle de position", "Tout ou rien (investi en calme, rien en stress), plus une règle de "
         "taille ajoutée et déclarée", "Fixée avant tout ajustement ; appliquée au portefeuille "
         "60/40 de référence"),
        ("Témoin obligatoire", "Règle d'une ligne : stress quand la volatilité sur 21 j du S&P 500 "
         "dépasse sa médiane expansive", "Tout résultat se lit à côté de ce que ce témoin obtient "
         "gratuitement"),
        ("Rendements et coûts", "En excès du taux 3 mois, 2 bp aller-retour",
         "Couche portefeuille"),
    ]:
        add((s, item, "Règle", value, why, CHARTER if item == "Cadrage gelé" else FINAL))

    s = "4. Modèles — comment ça marche"
    models = [
        ("A' sparse jump",
         "Même principe que A, plus une **sélection de variables intégrée** (pénalisation L1 : au "
         "plus 10 variables pondérées), refaite à chaque réestimation",
         "2 états ; λ choisi comme pour A ; au plus 10 variables",
         "**Modèle principal** ; c'est aussi lui qui choisit les variables données aux autres "
         "familles", "Nystrup, Kolm & Lindström (2020-2021) ; Aydınhan, Kolm, Mulvey & Shu (2024) "
         "— bibliothèque jumpmodels"),
        ("A  jump",
         "Regroupe les jours qui se ressemblent (type k-moyennes sur les variables "
         "standardisées), en **faisant payer une pénalité λ à chaque changement d'état**. Plus λ "
         "est grand, plus les régimes durent",
         "2 états ; λ choisi à chaque réestimation dans {1, 3, 10, 30, 100, 300} en maximisant le "
         "Sharpe d'entraînement, avec une borne de plausibilité sur la fréquence des changements",
         "Version sans sélection de variables", "Nystrup, Kolm & Lindström — regime_lab/models/"
         "jump.py"),
        ("B  filtered HMM",
         "Modèle de Markov caché : deux états cachés, chacun avec sa loi gaussienne, et une "
         "matrice de probabilités de transition. On lit la probabilité **filtrée** (passé "
         "seulement)", "2 états ; mêmes variables et même calendrier que A",
         "Référence exigée par le sujet, traitée loyalement", "regime_lab/models/hmm.py (hmmlearn "
         "+ récursion avant maison)"),
        ("C  gradient boost",
         "Pas d'état caché : **prédit directement la volatilité des 21 jours suivants** à partir "
         "des mêmes variables, puis coupe la prévision en deux à sa médiane d'entraînement",
         "LightGBM ; seuil = médiane de la prévision sur la fenêtre d'entraînement",
         "Teste si la couche « régime » est un intermédiaire inutile", "regime_lab/models/"
         "supervised.py"),
        ("C' HAR-RV",
         "Modèle linéaire de volatilité à partir de la volatilité récente sur 1, 5 et 21 jours, "
         "coupé en deux états de la même façon", "Régression linéaire, 3 composantes",
         "La référence que le machine learning bat rarement en prévision de volatilité",
         "Corsi (2009) — regime_lab/models/supervised.py"),
    ]
    for key, how, settings, role, ref in models:
        add((s, LABEL[key], "Principe", how, role, ref))
        add((s, LABEL[key], "Réglages", settings, "", ref))

    s = "5. Résultats par modèle"
    results = {
        # indicator: (values in MODELS order + placebo, explanation, source)
        "Durée moyenne d'un régime (jours)": (["456", "456", "128", "18", "14", "34"],
            "Plus c'est long, plus le régime est persistant (et moins il y a de trading)", FINAL),
        "Persistance par rapport au hasard": (["97×", "83×", "54×", "9×", "7×", "16×"],
            "Combien de fois plus persistant qu'une suite d'états tirée au hasard", FINAL),
        "Instabilité en temps réel": (["0,9 %", "0,6 %", "0,7 %", "0,0 %", "0,0 %", "0,0 %"],
            "Part des étiquettes qui changeraient si on connaissait la suite du semestre", FINAL),
        "Exactitude équilibrée contre les récessions NBER": (
            ["93,3 %", "93,2 %", "84,7 %", "75,0 %", "78,2 %", "—"],
            "Le stress tombe-t-il sur les récessions officielles ? (50 % = hasard)", FINAL),
        "Kappa contre NBER": (["0,49", "0,53", "0,24", "0,12", "0,17", "—"],
            "Accord corrigé du hasard (0 = hasard, 1 = parfait)", FINAL),
        "Écart de volatilité future entre états (21 j)": (
            ["−6,51 %", "−6,59 %", "−5,42 %", "−4,54 %", "−4,99 %", "−3,52 %"],
            "Calme moins volatil que stress : détecté pour tous", FINAL),
        "Écart de rendement futur entre états (21 j)": (
            ["−1,02 %", "−0,46 %", "−2,58 %", "−0,84 %", "−3,12 %", "−2,22 %"],
            "Non détecté pour aucun : les états ne prédisent pas le sens du marché", FINAL),
        "R² incrémental sur la volatilité future (t)": (
            ["+3,47 pts (t −3,46)", "+3,93 pts (t −3,40)", "+2,26 pts (t −4,59)",
             "+2,13 pts (t −6,34)", "+0,19 pt (t −1,84)", "+0,004 pt (t 0,21)"],
            "Ce que le modèle ajoute au-delà du témoin : **le résultat positif central de "
            "l'étude**", FINAL),
        "R² incrémental sur les rendements futurs (t)": (
            ["+0,022 pt (t 0,24)", "+0,030 pt (t 0,27)", "+0,007 pt (t −0,22)",
             "+0,023 pt (t 0,52)", "+0,008 pt (t −0,30)", "+0,000 pt (t −0,03)"],
            "Rien, pour tous : ni timing, ni prévision de direction", FINAL),
        "Sharpe net de la règle tout ou rien (60/40 seul : 0,47)": (
            ["0,54", "0,55", "0,46", "0,42", "0,50", "—"],
            "Toutes les différences restent sous le seuil détectable (0,27-0,40) : non décidable",
            FINAL),
        "Perte maximale de la règle tout ou rien (60/40 seul : −35,6 %)": (
            ["−22,3 %", "−22,3 %", "−13,7 %", "−10,2 %", "−13,4 %", "—"],
            "Réduit les pertes en crise, mais en réduisant aussi l'exposition", FINAL),
        "Sharpe net de la règle de taille": (["0,50", "0,49", "0,49", "0,50", "0,53", "—"],
            "Idem, non décidable", FINAL),
    }
    for c in MODELS:
        add((s, LABEL[c], "Changements d'état par an", fr(m[c]["per_year"]),
             f"{m[c]['transitions']} changements sur la période d'évaluation", "calculé ici "
             "(data/cache/states.parquet)"))
        add((s, LABEL[c], "Part du temps en stress", fr(m[c]["stress"], pct=True), "",
             "calculé ici (data/cache/states.parquet)"))
    add((s, LABEL["placebo"], "Changements d'état par an", fr(m["placebo"]["per_year"]),
         f"{m['placebo']['transitions']} changements", "calculé ici"))
    add((s, LABEL["placebo"], "Part du temps en stress", fr(m["placebo"]["stress"], pct=True),
         "Environ la moitié du temps, par construction (médiane)", "calculé ici"))
    for indicator, (values, why, source) in results.items():
        for key, value in zip([*MODELS, "placebo"], values, strict=True):
            add((s, LABEL[key], indicator, value, why, source))
    add((s, "Tous modèles", "Latence de détection", "0 à 13 jours calendaires",
         "Mesurée contre les pics NBER : les régimes sont reconnus quand ils arrivent", FINAL))
    add((s, "Tous modèles", "D'où vient la valeur (décomposition)",
         "Ciblage de volatilité +0,44 %/an ; apport propre du régime −0,04 % (avec recul), "
         "−0,25 % (net de la latence)", "La valeur vient de la gestion du risque, pas du régime",
         "docs/RESULTS_DECOMPOSITION.md"))

    s = "6. Pourquoi le Sparse Jump Model"
    for argument, evidence, source in [
        ("Choisi avant de mesurer",
         "Le cadrage gelé le désigne comme modèle principal avant tout résultat, sur la base de "
         "la littérature (Nystrup, Kolm & Lindström ; Aydınhan, Kolm, Mulvey & Shu)", CHARTER),
        ("Persistance réglée explicitement",
         "La pénalité λ fait payer chaque changement d'état : régimes longs, peu de rotation, "
         f"peu de coûts ({m[MODELS[1]]['transitions']} changements en {fr(m['years'], 1)} ans)",
         CHARTER),
        ("Sélection de variables intégrée",
         "La pénalisation L1 retient au plus 10 variables : modèle lisible (centroïdes "
         "interprétables), et même sous-ensemble fourni aux autres familles pour une comparaison "
         "loyale", CHARTER),
        ("Meilleur accord avec les récessions",
         "Kappa 0,53, le plus élevé des cinq ; exactitude 93,2 %, à égalité avec A (93,3 %), loin "
         "devant le HMM (84,7 %) et les modèles supervisés (75-78 %)", FINAL),
        ("Plus forte information sur le risque futur",
         "+3,93 points de R² sur la volatilité future au-delà du témoin (t −3,40), le plus élevé "
         "des cinq ; le témoin ajoute 0", FINAL),
        ("Fiable en temps réel",
         "0,6 % d'étiquettes instables, latence de 0 à 13 jours", FINAL),
        ("Limite à dire",
         "Il ne prédit pas le sens du marché (+0,03 point de R² sur les rendements, t 0,27), et "
         "sa lenteur (environ 0,5 changement par an) en fait un **thermomètre du risque**, pas un "
         "signal de trading. Le rendre plus réactif le fait ressembler à un simple filtre de "
         "volatilité (docs/EXPLORATION_SJM_SPEED.md)", FINAL),
    ]:
        add((s, argument, "Argument", evidence, "", source))

    s = "7. Couplage à des stratégies de trading"
    add((s, "8 stratégies × 3 couplages (arrêt, réduction, bascule)", "Verdict",
         "Aucun couplage utile ; tendance crypto et prime overnight Nasdaq pénalisées",
         "Ces stratégies gagnent justement en période de stress : le régime décrit le risque, pas "
         "le sens", "docs/RESULTS_COUPLAGE_STRATEGIES.md"))
    return r


def main() -> None:
    m = measured()
    table = rows(m)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["section", "élément", "indicateur", "valeur", "explication", "source"])
        writer.writerows([tuple(c.replace("**", "") for c in row) for row in table])
    print(f"{len(table)} rows -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
