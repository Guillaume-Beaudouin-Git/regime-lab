# Avancement du projet — état au 23 septembre 2026

Ce document sert à partir du même point : **ce qui est fait, où sont les données, ce qui
reste et par quoi commencer.** Chaque chiffre renvoie à un document ou à un script du
dépôt. Le détail complet des tâches est dans `pilotage/feuille_de_route/TACHES.md`.

---

## 0. En une minute

**La question du projet.** Un modèle de régimes de marché appris par machine apporte-t-il
une information exploitable au-delà d'une simple mesure de volatilité ? Et si oui,
résiste-t-elle aux coûts, aux contraintes de données en temps réel et à la correction
pour tests multiples ?

**La réponse, établie.**
1. **Le classifieur marche comme classifieur.** 93,2 % d'exactitude équilibrée contre
   les récessions NBER, kappa 0,53, sur 6 377 jours hors échantillon. Cinq méthodes
   concordent.
2. **Il porte de la variance, pas de la moyenne.** Il ajoute +3,93 points de R² sur la
   volatilité future (t −3,40), contre +0,030 sur les rendements futurs (t 0,27). C'est
   donc un signal de **dimensionnement**, pas de **timing**.
3. **Aucun des sept dispositifs testés pour le monétiser ne bat une règle d'une ligne**
   (volatilité réalisée sous sa médiane). Le chiffre qui explique tout : l'état change
   **13 fois en 6 377 séances**, soit treize décisions en 25 ans.
4. **Quatre hypothèses dérivées ont été falsifiées** : H1, H2, H3 et la prime de
   retournement.

**Le livrable du cours.** C'est un projet de cours de M2, **pas un mémoire**. Il se
conclut par une **présentation d'une dizaine de minutes** des résultats, au format d'une
présentation en entreprise.

**L'objectif de fond, au-delà du cours.** Construire des **stratégies de trading
algorithmique dépendantes du régime**, une fois cette parenthèse macro refermée. On
cherche un résultat intéressant et prometteur. L'ambition est un Sharpe **net, en excès
du cash et hors échantillon, de l'ordre de 1 à 2**, à **coûts institutionnels**. Le seuil
de recherche du programme ne bouge pas : RESEARCH_PASS, soit un Sharpe supérieur à 0,7,
une perte maximale moins profonde que −25 % et plus de 50 % des plis de walk-forward
positifs.

**Où on en est par rapport à cet objectif.** Le meilleur livre mesuré ici fait +0,36 de
Sharpe net en excès. C'est une tendance sur 46 instruments, déjà facturée à des coûts de
futures institutionnels (environ 1 bp). Le régime n'y ajoute rien au-delà de la
volatilité. Viser 1 à 2 demande donc un autre objet que les sept dispositifs testés. Les
deux pistes qui n'ont pas encore été mesurées sont celles qui s'en approchent : un régime
qui **arbitre entre plusieurs signaux** (plan Two Sigma, en cours) et un régime qui entre
dans la **construction du portefeuille** (plan Bridgewater).

**L'hypothèse de coûts pour la suite est institutionnelle.** Le barème de tête du
programme, en aller-retour, est déjà de ce type : 1 bp pour les futures d'indices, de
taux et de change, 1,5 bp pour les matières premières, 5 bp pour les actions US et 7,5 bp
pour un ETF sans future équivalent. Les colonnes prudente (×2) et de stress (×4 à ×5)
restent rapportées à côté, mais ne décident pas. Les barèmes déjà verrouillés dans les
pré-enregistrements ne changent pas.

---

## 1. Ce qui est fait

### L'étude principale — racine du dépôt

| bloc | verdict | document |
|---|---|---|
| Cadrage pré-enregistré, gelé par empreinte SHA-256 et étiquette `charter-v2` | gelé | `docs/CHARTER.html`, `docs/PROTOCOL_FREEZE.md` |
| Couche 1 — l'étiquetage est fiable | ✅ 93,2 % / kappa 0,53 | `docs/RESULTS_CLASSIFIER.md` |
| Couche 2 — il sépare la variance, et seulement elle | ✅ +3,93 pts R² vol, t −3,40 | `docs/RESULTS_FINAL.md` |
| Couche 3 — règle de timing contre règle de taille | ❌ aucune règle ne bat le témoin d'une ligne | `docs/RESULTS_FINAL.md`, `RESULTS_T2.md` |
| Contrôles T1, T3, T5, les cinq plis | publiés, ils affaiblissent l'étude | `docs/RESULTS_FALSIFICATION.md`, `RESULTS_FOLDS.md` |
| Décomposition de la valeur | le ciblage de vol vaut +0,44 %, le régime −0,04 % | `docs/RESULTS_DECOMPOSITION.md` |
| Réplication Shu (2024) | partielle, une ligne retirée | `docs/REPLICATION_SHU2024.md` |
| Barrière propfirm | nul sur les nuls | `docs/RESULTS_BARRIER.md`, `NOTE_BARRIER_FR.md` |
| Véhicule de tendance, 46 instruments | **KILL** : Sharpe net en excess +0,359 | `docs/RESULTS_TREND_VEHICLE.md` |
| AHL niveau A — sélection de vitesse | **réfuté sur le signe**, Δ −0,068 contre un seuil de 2,12 | `docs/RESULTS_AHL_LEVEL_A.md` |
| AHL niveau B — rupture de corrélation | **indécidable**, non lu, aucun essai dépensé (22/09 au soir) | `docs/RESULTS_AHL_LEVEL_B.md` |

Les 12 points d'hygiène sont faits. 122 tests passent, `ruff` est propre, et le registre
d'essais compte 84 configurations distinctes (`data/trials.parquet`).

### Les chantiers dérivés — `chantiers/`

| chantier | question | verdict | document |
|---|---|---|---|
| `macro-momentum` H1 | le momentum macro absolu prédit-il la direction ? | falsifiée, Sharpe −0,38 | `chantiers/macro-momentum/docs/RESULTS_H1.md` |
| `macro-momentum` H2 | le transversal entre 19 pays ? | falsifiée, 0 cellule sur 9 ; la dispersion des taux courts de la zone euro est **nulle depuis mars 1999** | `RESULTS_H2.md`, `RESULT_DISPERSION.md`, `NOTE_DISPERSION_FR.md` |
| `macro-momentum` H3 | les surprises macro, à la Dedale ? | falsifiée, le placebo bat le signal | `RESULTS_H3.md` |
| `reversal-lab` | le retour à la moyenne court terme est-il exploitable ? | prime passée de 3,64 de Sharpe (années 90) à −0,18 (depuis 2020) | `chantiers/reversal-lab/docs/RESULTS.md` |

### Le 22 septembre au soir

- **Le niveau B d'AHL est clos.** Le critère a été commité avant le calcul du MDE
  (`9602526`). Le MDE est sorti à 49,6 % de l'erreur moyenne contre une limite de 25 %,
  donc la question est indécidable à l'échelle du livre. Le verdict ne dépend pas du
  type de rendement (52,2 % en rendements logarithmiques).
- **Les trois dépôts ne font plus qu'un**, avec leur historique : les identifiants de
  commit cités comme preuves de pré-enregistrement existent toujours. L'environnement
  Python est unique, les 122 tests passent, et la mesure du niveau B se reproduit à
  l'identique après la fusion.
- **120 fichiers de mesure ont été sauvés de `/private/tmp`**, que le redémarrage du Mac
  aurait effacés : les scripts des cinq plans de recherche, ceux des quatre
  contre-expertises A4, les données SPF et l'historique Ken French depuis 1926.

### Le 23 septembre

- **Nature du projet précisée** : un projet de cours de M2 qui se termine par une
  présentation de 10 minutes, pas un mémoire. L'objectif de fond est écrit en §0.
- **La phase 1 du plan Two Sigma est faite** par une autre session (`22abbe0`,
  `3b64ab2`, 262 tests). L'outillage est en place : 10 signaux sur les 49 secteurs US,
  le classifieur de contexte K-means, un placebo apparié exact et le protocole de test.
  **Aucun niveau de l'arbre n'a été lu et aucun essai n'a été dépensé.**
- **Le verrou Two Sigma est commité** (`docs/PRESPEC_TWOSIGMA.md`). Guillaume l'a approuvé
  le 23/09, puis deux relecteurs indépendants l'ont examiné : un second validateur sur le
  fond, un auditeur avant le commit. Leurs corrections sont les lignes 28 à 36 du §12.0 :
  entre autres, un succès est rétrogradé s'il ne tient pas sans la série NFCI, qui est
  révisée. Les deux ont confirmé le texte final.
- **Premier test « stratégie seule contre couplée au régime »** : le rebond obligataire de
  fin de mois (`docs/RESULTS_B1_COUPLAGE.md`). Il est **insensible au régime** : Sharpe 0,79
  seul, 0,82 avec réduction en stress (sous-puissant), 0,75 avec bascule vers la
  tendance (pas utile). Il est positif en 2008, 2020 et 2022. Sa pire perte (2025-2026)
  survient alors que le régime est calme depuis 2021.
- **Comparaison « stratégie seule contre couplée au régime » sur 8 stratégies, 24
  couplages** (`docs/RESULTS_COUPLAGE_STRATEGIES.md`, hypothèse zéro coût) : **aucun
  couplage n'est utile**. Deux stratégies sont pénalisées, la tendance crypto et la prime
  overnight Nasdaq, qui gagnent précisément en période de stress. C'est la matière de la
  présentation.
- **Remesurés à partir de code commité, plusieurs chiffres du brouillon étaient
  optimistes.** L'horloge tourne à 12,6 transitions par an hors échantillon, et non 8,3.
  Elle dépend davantage de la volatilité qu'annoncé : η² de 0,14, jusqu'à 0,6 sur un
  pli, contre 0,05 dans le brouillon. Le plafond de levier mord sur 70 % des séances. Le
  seuil de rentabilité en coûts tombe à 27-33 bp, au lieu de 93. Le test reste assez
  fin : il détecte un écart d'environ 0,26 de Sharpe.

---

## 2. Où est quoi

```
regime-lab/                         UN SEUL dépôt, public
├── README.md                       commencer ici
├── AVANCEMENT.md                   ce fichier
├── CLAUDE.md                       consignes pour les sessions Claude Code
├── regime_lab/  scripts/  tests/   l'étude principale : code, scripts, 114 tests
├── docs/                           cadrage gelé, résultats, registre d'amendements
├── chantiers/
│   ├── macro-momentum/             H1, H2, H3 (son propre code et ses docs)
│   └── reversal-lab/               la prime de retournement
├── pilotage/
│   ├── feuille_de_route/           TACHES, CONCLUSIONS, PISTES, synthèse A4
│   ├── plans_de_recherche/         cinq plans (AHL, AQR, Bridgewater, Rentec, Two Sigma) + ARBITRAGE
│   ├── REPRISE_PROJET.md           ancienne passation (historique ; §8bis encore utile)
│   └── mesures_brutes/             ⚠ NON versionné — voir §3.4
└── data/                           ⚠ NON versionné — voir §3
```

La littérature (4 PDF) reste **hors du dépôt**, pour des raisons de droits d'auteur :
`~/Desktop/M2/Projet_Big_Data_Regimes/3_Litterature/`.

### Mise en route sur une autre machine

```bash
git clone https://github.com/Guillaume-Beaudouin-Git/regime-lab.git
cd regime-lab
uv sync --all-packages --extra dev          # un seul .venv pour les trois études
.venv/bin/python -m pytest -q               # 122 tests, ~40 s
```

Les tests passent sans aucune donnée. Les scripts, eux, en ont besoin (§3).

---

## 3. Les données — inventaire exact

**Aucune donnée n'est dans git.** Elles vivent uniquement sur le Mac de Guillaume. Deux
façons de les obtenir :

- **Recommandé : l'archive prête à envoyer**,
  `~/Desktop/M2/Projet_Big_Data_Regimes/donnees_regime-lab_2026-09-22.zip` (36 Mo,
  176 fichiers). Elle contient les données des trois études et celles des plans (§3.4).
  On la décompresse à la racine du dépôt. C'est le seul moyen d'obtenir *les mêmes
  chiffres* : Yahoo révise ses prix ajustés, et un nouveau téléchargement peut déplacer
  les résultats publiés.
- Sinon, les régénérer avec les scripts `fetch*.py`. Il faut alors une clé API FRED
  (gratuite, sur fred.stlouisfed.org), à placer dans `.env` sous `FRED_API_KEY`.

Toutes les données macro sont stockées en **point-in-time** : chaque ligne porte `period`
(la date décrite) et `available_at` (la date à laquelle on pouvait la connaître).

### 3.1 Étude principale — `data/`

**Brut** (`data/raw/`, produit par `scripts/fetch_data.py`, environ 3 min, clé FRED requise)

| fichier | contenu | source | période | lignes |
|---|---|---|---|---|
| `prices/cross_asset.parquet` | 8 séries : `^GSPC` `^RUT` `^IXIC` `^N225` `^TYX` `^TNX` `DX-Y.NYB` `^VIX` | Yahoo Finance, clôture ajustée | 1990-01-01 → 2026-09-08 | 73 716 |
| `prices/fred_daily.parquet` | pétrole WTI (`DCOILWTICO`) | FRED | 1990-01-02 → 2026-09-01 | 9 217 |
| `macro/*.parquet` (11 fichiers) | INDPRO, PAYEMS, CPIAUCSL, UNRATE (premières publications ALFRED) ; ICSA, NFCI (hebdo) ; BAA10Y, AAA10Y, DTB3, T10Y2Y, T10Y3M (quotidien) | FRED / ALFRED | 1990 → 2026-07/09 selon la série | 438 à 9 178 |
| `panels/factors_5.parquet` | 5 facteurs Fama-French + taux sans risque, **sans momentum** | Ken French Data Library | 1990-01-02 → 2026-07-31 | 55 272 |
| `panels/industry_49.parquet` | 49 portefeuilles sectoriels, quotidien | Ken French | 1990-01-02 → 2026-07-31 | 451 388 |
| `panels/size_bm_25.parquet` | 25 portefeuilles taille × valeur | Ken French | 1990-01-02 → 2026-07-31 | 230 300 |
| `references/ref_nber.parquet` | récessions NBER (`USREC`), **validation seulement** | FRED | 1990-01 → 2026-08 | 440 |

**Dérivé** (`data/cache/`)

| fichier | contenu | produit par | période |
|---|---|---|---|
| `features.parquet` | 50 variables standardisées (9 572 × 50) | `build_features.py` | 1990-01-01 → 2026-09-08 |
| `states.parquet`, `states_offline.parquet` | états des 5 familles de modèles (9 007 × 5), filtrés et avec recul | `run_phase2.py`, **15-20 min, ne pas relancer à la légère** | 1992-03-02 → 2026-09-08 |
| `trend_universe.parquet` | **46 instruments** : 13 indices actions, 13 matières premières, 11 devises (dont l'indice dollar), 9 ETF obligataires et crédit, en **prix** (6 822 × 46) | `fetch_trend_universe.py` (Yahoo, `auto_adjust=True`) | 2000-07-17 → 2026-09-10 |
| `trend_universe_m1.parquet` | le même, avec 8 devises recalées d'une séance — **panneau de référence** | `apply_m1_realignment.py` | idem |
| `trend_universe_m1_verified.parquet` | sensibilité : 3 devises recalées seulement | idem | idem |
| `shu_*.parquet` | S&P 500 depuis 1970 (données Shiller) pour la réplication Shu | `replicate_shu2024.py` | 1970 → 2026-09-09 |
| `../trials.parquet` | **le registre d'essais** : 806 lignes, 84 configurations | chaque script de décision | — |

Échantillon de référence de toutes les études sur le livre de tendance : **2003-07-17 →
2026-09-10, 6 039 séances**. C'est la première séance où au moins 30 instruments ont 252
séances d'historique.

### 3.2 `chantiers/macro-momentum/data/`

| fichier | contenu | source | période |
|---|---|---|---|
| `raw/mm_prices/assets.parquet` | 20 actifs multi-classes (indices, ETF obligataires, futures, dollar) | Yahoo | 1990-01-01 → 2026-09-10 |
| `raw/mm_macro/*.parquet` | INDPRO, PAYEMS, CPI en premières publications ; BAA10Y ; DTB3 | FRED / ALFRED | 1990 → 2026-07/09 |
| `raw/h2/markets.parquet` | 14 indices actions nationaux + 10 devises (24 séries) | Yahoo | 1990-01-02 → 2026-09-10 (les devises à partir de 1996-2006) |
| `raw/h2/rates.parquet` | taux longs et courts de **19 pays** (38 séries), mensuel | FRED / OCDE | 1990-01 → 2026-06 |
| `raw/h3_macro/*.parquet` | 13 séries US **avec tous leurs millésimes** (INDPRO, PAYEMS, CPI, core CPI, UNRATE, ICSA, TCU, DGORDER, AWHMAN, HOUST, PERMIT, RSAFS, UMCSENT) | ALFRED | 1998-01 → 2026-07/08 |
| `raw/h3_assets/prices.parquet` | 4 séries : actions, obligations, or, dollar | Yahoo | 1998-01-01 → 2026-09-10 (l'or à partir de 2000-08) |

Produits par `chantiers/macro-momentum/scripts/fetch.py`, `fetch_h2.py` et `fetch_h3.py`.

### 3.3 `chantiers/reversal-lab/data/`

| fichier | contenu | source | période |
|---|---|---|---|
| `raw/french/*.parquet` | facteurs de retournement court et long terme, momentum, 5 facteurs, déciles et portefeuilles taille × rendement passé | Ken French | 1990-01-02 → 2026-07-31 |
| `raw/stocks/us_large.parquet` | 117 grandes capitalisations US, **biais de survie déclaré** | Yahoo | 1990-01-02 → 2026-09-09 |

Produits par `chantiers/reversal-lab/scripts/fetch.py`.

### 3.4 Données des plans de recherche — sauvées, pas encore intégrées

Dans `pilotage/mesures_brutes/`, **non versionné**, avec une copie sur iCloud
(`Sauvegardes_Recherche_Quant/mesures_brutes_2026-09-22/`) :

| fichier | contenu | source | période | sert à |
|---|---|---|---|---|
| `firmes/bridgewater/probe/median_{rgdp,cpi,indprod,unemp}_level.xlsx` | médianes des prévisionnistes, 4 variables, 232 enquêtes | Fed de Philadelphie, *Survey of Professional Forecasters* | 1968-T4 → 2026-T3 | plan Bridgewater |
| `firmes/aqr/verif/49_Industry_Portfolios_Daily.csv` | 49 secteurs, **historique complet** (20 Mo) | Ken French | 1926-07-01 → 2026-07-31 | holdout scellé 1971-1989 (AQR / Two Sigma) |
| `firmes/aqr/verif/F-F_Momentum_Factor_daily.csv` | le facteur momentum qui manque à `factors_5` | Ken French | 1926-11-03 → 2026-07-31 | AQR |
| `firmes/aqr/verif/25_Portfolios_5x5.csv`, `nfci_full.csv`, `anfci_full.csv`, `baa_full.csv` | panneaux et séries complètes | Ken French, FRED | historiques complets | AQR |

⚠ Les deux fichiers `/private/tmp/spf.xlsx` et `spf_cpi.xlsx` **ne sont pas des données**.
Ils font 18 401 octets chacun : ce sont des pages HTML « 404 » servies avec un code
HTTP 200, un piège que le plan Bridgewater décrit lui-même.

⚠ **Le plan Rentec utilise des données minute qui ne sont pas dans ce dépôt** : 27
instruments, 8,8 Go, dans un dépôt privé de Guillaume. Seul lui peut exécuter ce plan.

---

## 4. Ce qui reste, précisément

Classé par ordre d'intérêt. Chaque ligne donne l'effort, les données nécessaires et le
critère de fin.

| # | tâche | effort | données | c'est fini quand |
|---|---|---|---|---|
| ~~0~~ | ~~Fixer l'objectif final~~ **FAIT le 23/09** : une présentation de 10 minutes ; objectif de fond, des stratégies dépendantes du régime (§0) | — | — | — |
| **1** | **Construire la présentation de 10 minutes**, au format entreprise, en 8 à 10 diapositives. Le fil : la question → la méthode (données point-in-time, critère écrit avant le chiffre) → ce qui marche (classifieur à 93,2 %, qui prédit la variance et pas la direction) → ce que ça implique (un outil de dimensionnement, pas de timing ; 13 changements d'état en 25 ans) → ce qu'on a testé sans succès (7 dispositifs, 4 hypothèses) → la suite, vers des stratégies dépendantes du régime. Trois figures : **(a)** R² sur la volatilité contre R² sur les rendements, un point par famille ; **(b)** la frise des 13 transitions ; **(c)** les dispositifs face à la règle d'une ligne | 1-2 j | `data/cache/*` et `docs/RESULTS_*.md`, déjà calculés | les diapositives et un texte oral de 10 minutes, répétés une fois |
| **2** | **Plan Two Sigma : descendre l'arbre A → B → C** (le verrou est commité). C'est la piste la plus proche de l'objectif de fond : le régime choisit entre dix signaux. Le verrou impose d'abord d'écrire et de commiter **les trois instruments** (A, B, C), leurs seuils et les empreintes des données (§13.1). Ensuite viennent les lectures dans l'ordre A, B, C, **quel que soit le résultat de chacune** ; chaque lecture compte comme un essai. Le verdict alimente la dernière diapositive | ~2 j pour les instruments, puis quelques heures par lecture | `industry_49`, `factors_5`, `features` (§3.1) | verdicts écrits dans `docs/RESULTS_TWOSIGMA*.md` |
| **3** | **Fermer l'arbre AHL** : le MDE de C1, que le pré-enregistrement déclare sous-puissant d'avance, la note de fermeture, et la décision sur la lecture A1 principale sur (126,10), qui reste due | ½-1 j | `trend_universe_m1.parquet` | `docs/RESULTS_AHL_LEVEL_C.md` et la fermeture |
| **4** | **Nettoyer et commiter `pilotage/mesures_brutes/`** : retirer les chemins absolus et privés des scripts | 1-2 h | — | les scripts dans git, les données dans `data/` |
| 5 | **Plan Bridgewater** : le régime dans la construction du portefeuille, l'autre axe jamais testé (8,5 j). Porte G0 de Rentec (1,5 j, Guillaume seul, données privées) | plusieurs jours | §3.4 | — |
| 6 | Les notes écrites sur H3 et sur la dispersion **ne sont plus prioritaires** sans mémoire. Elles restent utiles comme diapositives de réserve | ½ j | aucune | — |
| 7 | Archiver sur GitHub les anciens dépôts privés `macro-momentum` et `reversal-lab` | 5 min | — | Guillaume |

---

## 5. Prochaine séance de travail, proposition

Deux pistes en parallèle : la présentation ne demande aucun calcul, Two Sigma en demande.

| durée | ensemble | piste A · Two Sigma | piste B · présentation |
|---|---|---|---|
| 30 min | arrêter le fil de la présentation et le choix des stratégies à comparer | | |
| 2 h 30 | | instruments A, B et C, avec tests et seuils commités | figures (a) et (b), diapositives 1 à 5 |
| 30 min | point d'étape | | |
| 2 h | | lecture du niveau A, puis B et C | diapositives 6 à 10, texte oral |
| 1 h | répétition chronométrée (10 min), commits, mise à jour de ce fichier | | |

**Par quoi on commence : la construction des trois instruments Two Sigma**, parce que
toute lecture en dépend. La présentation avance en parallèle : tout son matériau est déjà
calculé.

---

## 6. Règles du projet, à connaître avant de toucher au code

- Signal en T−1, trade en T. Rendements **nets, en excess** du taux cash. Erreurs HAC
  lag 6. Correction pour tests multiples dès qu'il y a plus d'un test.
- **On écrit le critère avant de produire le chiffre**, et on le commite. Un écart sous
  le MDE est « sous-puissant », **jamais** un succès.
- Tout script de décision **refuse de rendre un verdict** si une lecture n'est pas finie
  (NaN).
- Chaque lecture d'un résultat est un **essai**, journalisé dans `data/trials.parquet`.
- On ne modifie jamais un document verrouillé : toute déviation va dans
  `docs/PROTOCOL_FREEZE.md`.
- On commite tout, avec des messages qui donnent la raison. Un chiffre publié sans son
  code commité, c'est le piège n°1 de ce projet, et il s'est produit cinq fois.
