# Détecter les crises de marché : est-ce que ça rapporte ?

**Projet Big Data — Master 2** · Guillaume Beaudouin et Gabriel Golivet · 25 septembre 2026

Ce dépôt (`regime-lab`) contient **tout le code du projet**, ses résultats et son
historique complet.

> **Vous venez du dossier PDF ?** Vous êtes au bon endroit.
> - Le dossier lui-même : [`docs/presentation/dossier_regimes.pdf`](docs/presentation/dossier_regimes.pdf).
> - La version exacte du code qui l'accompagne : l'étiquette
>   [`dossier-2026-09-25`](https://github.com/Guillaume-Beaudouin-Git/regime-lab/tree/dossier-2026-09-25).
> - Pour **lire** le code, un navigateur suffit : cliquez sur les fichiers cités
>   ci-dessous, rien n'est à installer.
> - Pour **relancer** les calculs, voir [Relancer les calculs](#relancer-les-calculs).

Le code et ses commentaires sont en anglais, comme les documents de résultats de l'étude
principale ; le dossier et les études de septembre 2026 sont en français. Le code a été écrit
avec l'aide d'un assistant de programmation (Claude Code) ; la question, le protocole, les
choix de recherche et l'interprétation sont les nôtres.

*English summary at the bottom of this page.*

## Lire le code en six fichiers

Pour comprendre le projet de bout en bout, dans l'ordre :

1. [`regime_lab/data/universe.py`](regime_lab/data/universe.py) — la liste de toutes les
   séries utilisées, avec leur source (Yahoo Finance, FRED, ALFRED, Ken French).
2. [`regime_lab/features/`](regime_lab/features/) — le calcul des 50 indicateurs
   (`market.py`, `asymmetry.py`, `crosssection.py`, `macro.py`) et leur standardisation sur
   le seul passé (`standardise.py`).
3. [`regime_lab/models/jump.py`](regime_lab/models/jump.py) — le Sparse Jump Model, modèle
   principal ; le choix de sa pénalité λ est dans
   [`regime_lab/models/calibrate.py`](regime_lab/models/calibrate.py).
4. [`scripts/run_phase2.py`](scripts/run_phase2.py) — les cinq modèles réestimés 49 fois
   sur le passé seul (walk-forward), qui produisent les états de chaque jour.
5. [`regime_lab/evaluation/`](regime_lab/evaluation/) — la notation des états :
   classification face aux récessions (`reliability.py`) et information sur la volatilité
   et les rendements futurs (`predictive.py`).
6. [`scripts/run_crisis_coupling.py`](scripts/run_crisis_coupling.py) — un exemple complet
   d'application à des stratégies, avec le critère de décision écrit en tête du fichier
   avant la lecture.

## Où est le code de chaque partie du dossier

| partie du dossier | code | résultat écrit |
|---|---|---|
| §2 Les données et le contrat point-in-time | [`regime_lab/data/`](regime_lab/data/) (`universe.py`, `pit.py`), [`scripts/fetch_data.py`](scripts/fetch_data.py) | `AVANCEMENT.md` §3 |
| §2 Les 50 indicateurs | [`regime_lab/features/`](regime_lab/features/), [`scripts/build_features.py`](scripts/build_features.py) | annexe A du dossier |
| §3 Sparse Jump Model et Jump Model | [`regime_lab/models/jump.py`](regime_lab/models/jump.py), [`calibrate.py`](regime_lab/models/calibrate.py) | [`docs/artifacts/sjm_sparsity.txt`](docs/artifacts/sjm_sparsity.txt) |
| §3 HMM filtré | [`regime_lab/models/hmm.py`](regime_lab/models/hmm.py) | — |
| §3 Gradient boosting et HAR-RV | [`regime_lab/models/supervised.py`](regime_lab/models/supervised.py) | — |
| §3 Walk-forward (49 réestimations) | [`regime_lab/models/base.py`](regime_lab/models/base.py), [`scripts/run_phase2.py`](scripts/run_phase2.py), [`export_states.py`](scripts/export_states.py) | [`docs/artifacts/etats_hors_echantillon.csv`](docs/artifacts/etats_hors_echantillon.csv) |
| §3 Portefeuille 60/40 de référence | [`regime_lab/strategies/`](regime_lab/strategies/) | — |
| §4 Résultat 1, classification | [`regime_lab/evaluation/reliability.py`](regime_lab/evaluation/reliability.py), [`scripts/run_evaluation.py`](scripts/run_evaluation.py) | [`docs/RESULTS_FINAL.md`](docs/RESULTS_FINAL.md) |
| §4 Témoin de volatilité face au NBER | [`scripts/measure_vol_rule_nber.py`](scripts/measure_vol_rule_nber.py) | [`docs/artifacts/temoin_nber.txt`](docs/artifacts/temoin_nber.txt) |
| §5 Résultat 2, risque et direction | [`regime_lab/evaluation/predictive.py`](regime_lab/evaluation/predictive.py), [`scripts/run_layer3.py`](scripts/run_layer3.py) | [`docs/RESULTS_FINAL.md`](docs/RESULTS_FINAL.md) |
| §5 Contrôles de falsification (T1, T3, T5) | [`scripts/run_t1_control.py`](scripts/run_t1_control.py), `run_t3_control.py`, `run_t5_refit.py`, `run_t5_control.py` | [`docs/RESULTS_FALSIFICATION.md`](docs/RESULTS_FALSIFICATION.md) |
| §5 Test P, §6 stratégies de crise | [`regime_lab/extensions/crisis.py`](regime_lab/extensions/crisis.py), [`scripts/run_crisis_coupling.py`](scripts/run_crisis_coupling.py) | [`docs/RESULTS_CRISE.md`](docs/RESULTS_CRISE.md) |
| §6 Rebond obligataire de fin de mois | [`scripts/run_b1_regime_coupling.py`](scripts/run_b1_regime_coupling.py) | [`docs/RESULTS_B1_COUPLAGE.md`](docs/RESULTS_B1_COUPLAGE.md) |
| §6 Valeurs refuges (or, obligations, options) | [`scripts/run_safe_haven_switch.py`](scripts/run_safe_haven_switch.py) | [`docs/RESULTS_REFUGE.md`](docs/RESULTS_REFUGE.md) |
| §7 Le test sur 90 ans | [`regime_lab/extensions/longhist.py`](regime_lab/extensions/longhist.py), `scripts/longhist_*.py` | [`docs/RESULTS_LONGHIST.md`](docs/RESULTS_LONGHIST.md) |
| Annexe B, écarts au protocole | — | [`docs/PROTOCOL_FREEZE.md`](docs/PROTOCOL_FREEZE.md) |
| Annexe C, bootstrap, placebo, puissance, journal des essais | [`regime_lab/analysis/`](regime_lab/analysis/) | `data/trials.parquet` (non versionné) |
| Les figures et tableaux du dossier | [`scripts/build_dossier.py`](scripts/build_dossier.py) | [`docs/presentation/`](docs/presentation/) |

La tendance crypto et sept des neuf stratégies du livre de la partie application viennent
d'un dépôt privé : le dossier n'en publie que des chiffres agrégés, et elles ne sont pas
ici. Tout le reste se relance depuis ce dépôt.

## Relancer les calculs

Il faut Python 3.12 et [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Guillaume-Beaudouin-Git/regime-lab.git
cd regime-lab
git checkout dossier-2026-09-25              # la version qui accompagne le dossier
uv sync --all-packages --extra dev           # un seul environnement pour tout le dépôt
.venv/bin/python -m pytest -q                # 851 tests, tous réussis le 25/09 ; sans données, ceux qui en ont besoin sont sautés
```

Utilisez toujours `.venv/bin/python`, jamais le Python du système.

**Les données ne sont pas dans git** (environ 50 Mo, régénérables). Une clé API FRED
gratuite suffit pour les télécharger : copiez `.env.example` vers `.env`, renseignez
`FRED_API_KEY`, puis lancez `scripts/fetch_data.py` et `scripts/build_features.py`.
L'inventaire exact, fichier par fichier, est dans [`AVANCEMENT.md`](AVANCEMENT.md) §3.
Les lectures des études n'ont été faites qu'une fois : leurs sorties complètes sont
commitées dans [`docs/artifacts/`](docs/artifacts/).

## Organisation du dépôt

```
regime_lab/      le code : données, indicateurs, modèles, évaluation, statistiques
scripts/         un script par résultat publié
tests/           les tests
docs/            le cadrage gelé, les résultats (RESULTS_*.md), les écarts au protocole,
                 les sorties brutes (artifacts/) et le dossier (presentation/)
chantiers/       deux études dérivées, falsifiées : macro-momentum (H1 à H3), reversal-lab
pilotage/        feuille de route, plans de recherche, notes de lecture
AVANCEMENT.md    l'état du projet, l'inventaire des données, ce qui reste
CLAUDE.md        les consignes données à l'assistant de programmation utilisé pendant le projet
```

## La question, et la réponse

> Un modèle de régimes appris par machine apporte-t-il une information exploitable
> au-delà de ce qu'une simple mesure de volatilité capte déjà ?

1. **Le classifieur fonctionne.** 93,2 % d'exactitude équilibrée contre les récessions
   officielles (NBER), sur 6 377 jours jamais vus à l'entraînement, contre 84,0 % pour la
   meilleure règle de volatilité d'une ligne (`docs/artifacts/temoin_nber.txt`). ⚠ La
   période de test ne contient que **deux récessions** (2008-09 et 2020) ; réestimée sur
   1926-2026 (hors échantillon : 1937-2026), une version réduite n'en reconnaît que 6 sur 14
   (`docs/RESULTS_LONGHIST.md`).
2. **Il prédit la volatilité, pas la direction.** Il ajoute +3,93 points de R² sur la
   volatilité future au-delà d'une règle de volatilité passée, et rien sur les
   rendements futurs (+0,03 point, t 0,27). ⚠ Dans un test séparé sur le S&P 500 (test P,
   `docs/RESULTS_CRISE.md`), l'état ajoute +4,48 points au-delà d'un rang de volatilité,
   mais +0,20 seulement, non significatif, une fois le VIX ajouté.
3. **Aucun usage de trading n'en tire un gain démontré.** Sur 62 usages et 13 stratégies,
   aucun n'est jugé utile, et une règle d'une ligne (« la volatilité récente est-elle sous sa
   médiane ? ») fait aussi bien. Le régime ne change que **13 fois en 24 ans**, entre en
   stress tard et reste en stress pendant une grande partie des reprises, en partie à
   cause des réestimations semestrielles.

Ce résultat négatif est le résultat. Il a été obtenu par des mesures conçues **à
l'avance** pour pouvoir dire non.

## Les règles de méthode

- Le signal est calculé en T−1 et la position prise en T.
- On juge sur des rendements **nets de coûts et en excès du taux sans risque**. Seule
  exception, déclarée : les couplages aux stratégies de la partie application sont
  calculés sans coût, par hypothèse du cours.
- Erreurs-types robustes (HAC), correction dès qu'il y a plusieurs tests, bootstrap par
  blocs.
- **Le critère de décision est écrit et commité avant de produire le chiffre.** Un effet
  plus petit que ce que l'échantillon peut détecter est déclaré « sous-puissant », jamais
  « validé ».
- Chaque configuration évaluée est journalisée, pour qu'on puisse corriger du nombre
  d'essais.

## English summary

A study of machine-learned market regimes under a point-in-time data contract. The
classifier works (93.2% balanced accuracy against NBER recessions, out of sample) and
carries information about **variance, not mean**: +3.93 points of incremental R² on
forward volatility beyond a past-volatility rule (in a separate test on the S&P 500, +4.48
points beyond a volatility rank but only +0.20 once the VIX is added), nothing on forward
returns. The test period holds two recessions only. No use of it as a trading filter beats
a one-line volatility rule; the state changes only 13 times in 24 years.
Four derived hypotheses are falsified in `chantiers/`. The English description of the
main study is in `docs/OVERVIEW_EN.md`.
