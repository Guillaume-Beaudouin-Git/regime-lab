# regime-lab — détection de régimes de marché, et ce qu'elle vaut vraiment

Projet de Big Data appliqué à la finance. On y apprend à une machine à reconnaître les
**régimes de marché** (calme, stress, crise…), puis on se demande honnêtement si cette
reconnaissance **rapporte quelque chose** une fois pris en compte les coûts, les données
réellement disponibles à chaque date, et le nombre de tests effectués.

*English summary at the bottom of this page.*

## Par où commencer

| si vous voulez… | lisez |
|---|---|
| savoir où en est le projet et ce qu'il reste à faire | **`AVANCEMENT.md`** |
| les résultats de l'étude principale | `docs/RESULTS_FINAL.md` |
| le protocole fixé avant toute mesure | `docs/CHARTER.html` (gelé), `docs/PROTOCOL_FREEZE.md` (écarts) |
| la liste détaillée des tâches | `pilotage/feuille_de_route/TACHES.md` |
| ce qui est établi, ce qui est mort | `pilotage/feuille_de_route/CONCLUSIONS.md` |

## La question, et la réponse

> Un modèle de régimes appris par machine apporte-t-il une information exploitable
> au-delà de ce qu'une simple mesure de volatilité capte déjà ?

1. **Le classifieur fonctionne.** 93,2 % d'exactitude équilibrée contre les récessions
   officielles (NBER), sur 6 377 jours jamais vus à l'entraînement. Cinq méthodes
   différentes s'accordent.
2. **Il prédit la volatilité, pas la direction.** Il ajoute +3,93 points de R² sur la
   volatilité future, et rien sur les rendements futurs (+0,03 point, t 0,27).
3. **Personne n'a réussi à en tirer de l'argent.** Sept dispositifs ont été testés et
   aucun ne bat une règle d'une ligne : « la volatilité récente est-elle sous sa
   médiane ? ». La raison est mécanique : le régime change **13 fois en 25 ans**, soit
   trop peu de décisions pour qu'un signal de trading en émerge.

Ce résultat négatif est le résultat. Il a été obtenu par des mesures conçues **à
l'avance** pour pouvoir dire non.

## Ce que contient le dépôt

```
README.md, AVANCEMENT.md       entrée et état d'avancement
regime_lab/                    le code de l'étude principale (données, variables, modèles, évaluation)
scripts/                       les scripts qui produisent chaque résultat publié
tests/                         114 tests
docs/                          cadrage, résultats, registre des écarts au protocole
chantiers/macro-momentum/      trois hypothèses dérivées (H1, H2, H3), toutes falsifiées
chantiers/reversal-lab/        la prime de retour à la moyenne, disparue depuis 2020
pilotage/                      feuille de route et plans de recherche
```

Trois familles de modèles sont comparées : des modèles à sauts statistiques (A, A′), un
modèle de Markov caché (B) et des prédicteurs supervisés (C, C′). Toutes les données macro
respectent un **contrat point-in-time** : une valeur n'entre dans un modèle qu'à partir du
jour où elle était réellement publiée.

## Installation

Il faut Python 3.12 et [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Guillaume-Beaudouin-Git/regime-lab.git
cd regime-lab
uv sync --all-packages --extra dev          # un seul environnement pour tout le dépôt
.venv/bin/python -m pytest -q               # 122 tests
```

Utilisez toujours `.venv/bin/python`, jamais le Python du système.

**Les données ne sont pas dans git** (36 Mo, régénérables). L'inventaire exact, fichier
par fichier (source, période, script qui le produit), est dans `AVANCEMENT.md` §3. Pour
les télécharger soi-même, une clé API FRED gratuite suffit : copiez `.env.example` vers
`.env` et renseignez `FRED_API_KEY`.

## Les règles de méthode

- Le signal est calculé en T−1 et la position prise en T.
- On juge sur des rendements **nets de coûts et en excès du taux sans risque**.
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
forward volatility, nothing on forward returns. None of seven devices built to monetise
it beats a one-line volatility rule, because the state changes 13 times in 25 years. Four
derived hypotheses are falsified in `chantiers/`. The English description of the main
study is in `docs/OVERVIEW_EN.md`.
