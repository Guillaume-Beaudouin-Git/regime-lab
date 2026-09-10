# Reprise du projet — tout ce qu'il faut savoir

Fichier de passation. La session qui l'a produit n'existe plus ; celle qui le lit
n'a aucune mémoire. Tout ce qui est nécessaire est ici, avec les chemins exacts.

À déplacer dans `~/Desktop/regime-lab/` après lecture.

---

## 1. Ce qu'il faut comprendre en trente secondes

Trois dépôts, trois hypothèses, **trois falsifications**. Aucune sur un backtest
raté : chacune sur une mesure conçue pour pouvoir dire non, qui l'a dit avant
qu'un seul paramètre soit ajusté.

| dépôt | question | verdict |
|---|---|---|
| `regime-lab` | la classification de régimes marche-t-elle, et que prédit-elle ? | **elle marche, elle porte la variance et pas la moyenne** — livré, audité, publié |
| `macro-momentum` | le momentum macro prédit-il la direction ? | H1 et H2 falsifiées, la coupe transversale n'existe plus |
| `reversal-lab` | le retour à la moyenne court terme est-il exploitable ? | **prime disparue**, kill mécanique, fermé |

Le projet principal (`regime-lab`) est un **projet de cours de M1 Big Data**,
terminé et défendable. Les deux autres sont des chantiers de recherche qui en
découlent.

---

## 2. Emplacements exacts

### Dépôts

```
~/Desktop/regime-lab/          projet principal, PUBLIC sur GitHub, 74 tests
~/Desktop/macro-momentum/      chantier 2, local uniquement
~/Desktop/reversal-lab/        chantier 3, local uniquement
~/Desktop/Projet régime de marché/   6 PDF de littérature (Nystrup, Kolm, Shu…)
```

GitHub : `https://github.com/Guillaume-Beaudouin-Git/regime-lab` (branche `main`,
tag `charter-v2`). Les deux autres dépôts n'ont **pas** de remote.

### Environnements Python

Chaque dépôt a son propre venv. **Toujours utiliser le venv du dépôt** :

```bash
~/Desktop/regime-lab/.venv/bin/python
~/Desktop/macro-momentum/.venv/bin/python
~/Desktop/reversal-lab/.venv/bin/python
```

`macro-momentum` et `reversal-lab` installent `regime-lab` en dépendance
éditable — ils réutilisent son contrat point-in-time, sa garde de couverture,
son bootstrap et son journal d'essais. **Modifier `regime-lab` casse les deux
autres.**

### Clé API

`~/Desktop/regime-lab/.env` contient `FRED_API_KEY`. Gratuite, lecture seule,
non commitée (vérifié : absente de tout l'historique git). Sans elle, la macro
tombe sur un chemin de repli documenté.

### Données déjà téléchargées — NE PAS RE-FETCH

```
~/Desktop/regime-lab/data/raw/prices/cross_asset.parquet      8 actifs, 1990-2026
~/Desktop/regime-lab/data/raw/prices/fred_daily.parquet       pétrole WTI depuis FRED
~/Desktop/regime-lab/data/raw/macro/*.parquet                 11 séries macro et crédit
~/Desktop/regime-lab/data/raw/panels/industry_49.parquet      49 secteurs Ken French
~/Desktop/regime-lab/data/raw/panels/size_bm_25.parquet       25 portefeuilles taille/valeur
~/Desktop/regime-lab/data/raw/panels/factors_5.parquet        5 facteurs + momentum
~/Desktop/regime-lab/data/raw/references/ref_nber.parquet     récessions NBER (VALIDATION SEULE)
~/Desktop/regime-lab/data/cache/features.parquet              50 variables standardisées
~/Desktop/regime-lab/data/cache/states.parquet                états filtrés, 5 familles
~/Desktop/regime-lab/data/cache/states_offline.parquet        états avec recul (latence)
~/Desktop/regime-lab/data/cache/refit_diagnostics.parquet     vol par état, par réestimation
~/Desktop/regime-lab/data/cache/trend_universe.parquet        46 instruments, 2000-2026
~/Desktop/regime-lab/data/trials.parquet                      journal d'essais

~/Desktop/macro-momentum/data/raw/mm_prices/assets.parquet    20 actifs multi-classes
~/Desktop/macro-momentum/data/raw/mm_macro/*.parquet          macro en premières publications
~/Desktop/macro-momentum/data/raw/h2/rates.parquet            38 séries de taux, 19 pays
~/Desktop/macro-momentum/data/raw/h2/markets.parquet          10 devises, 14 indices

~/Desktop/reversal-lab/data/raw/french/*.parquet              facteurs et déciles CRSP
~/Desktop/reversal-lab/data/raw/stocks/us_large.parquet       117 titres, BIAIS DE SURVIE
```

`data/` est gitignoré partout. Régénérable via les scripts `fetch*.py`.

### Documents publiés (pages web)

Sources dans `~/Desktop/regime-lab/docs/artifacts/`. URL des versions en ligne :

| page | URL |
|---|---|
| cadrage gelé | `https://claude.ai/code/artifact/d36272d5-2776-419f-ae66-cc0d29299c43` |
| journal de construction | `https://claude.ai/code/artifact/caed500b-26cb-4906-a677-b6e8829c1b34` |
| explication grand public | `https://claude.ai/code/artifact/5483a861-6c38-4117-8aa4-d1701d56b564` |
| note d'orientation | `https://claude.ai/code/artifact/ad45441d-fa54-470a-897d-c6b945d4f3a6` |

Pour mettre à jour une page : éditer le HTML dans `docs/artifacts/`, puis
republier avec l'outil Artifact en passant l'URL ci-dessus en paramètre `url`.

---

## 3. Les règles non négociables du projet

Elles viennent du `CLAUDE.md` de `~/Desktop/Algo_claude/` et s'appliquent partout.

- **Signal en T-1, trade en T.** Aucune exception.
- **Rendements en excess** du taux sans risque. HAC lag-6. Correction pour tests
  multiples dès `n_tests > 1`.
- **Walk-forward ≥ 5 plis**, jamais un split unique.
- **Ciblage de volatilité quotidien.**
- **N ≥ 30 instruments** pour toute stratégie transversale.
- **Maximum 3 modifications** par hypothèse de stratégie.
- **Bootstrap par blocs** (Politis-Romano), jamais iid quand les trades sont
  groupés.
- **Placebo apparié** obligatoire sur tout claim calendaire ou conditionnel.

Coûts aller-retour : crypto 0,17 % · futures ~0,01 % · actions 0,05-0,10 % ·
FX 0,15 %.

Politique de réouverture d'une famille archivée, selon la **nature** du kill :
mécanique (gross ≈ 0) → **reste fermé** ; véhicule, coût, puissance, géométrie →
**rouvrable**, mais uniquement avec un angle explicitement neuf, écrit avant de
toucher la donnée.

---

## 4. `regime-lab` — le projet principal

### Ce qu'il établit

**La classification de régimes fonctionne**, mesuré contre une référence externe :

- 93,2 % d'exactitude équilibrée contre les récessions NBER, kappa 0,53 (sparse
  jump model), sur 6 377 jours hors échantillon 2002-2026
- persistance 83 à 97 fois le hasard
- cinq méthodes indépendantes s'accordent, kappa 0,28 à 0,83
- latence réelle 0 à 13 jours

**Ce qu'elle porte est de la variance, pas de la moyenne.** C'est le résultat
central :

| information marginale au-delà d'un quantile de volatilité | rendements futurs | volatilité future |
|---|---|---|
| A′ sparse jump | +0,030 pt (t 0,27) | **+3,93 pt (t −3,40)** |
| A jump model | +0,022 pt | +3,47 pt (t −3,46) |
| C gradient boosting | +0,023 pt | +2,13 pt (t −6,34) |
| **· placebo d'une ligne** | +0,000 pt | **+0,004 pt (t 0,21)** |

Donc : **signal de dimensionnement, pas de timing.** Une règle ON/OFF est une
règle de timing, ce qui explique qu'elle ne batte rien.

**Au niveau portefeuille, la question est sous-puissante** d'un facteur deux à
quatre : il faudrait 46 à 101 ans de données, on en a 25. Sauf sur A′ sparse
jump, où l'échantillon résout 0,27 et l'écart mesuré est −0,03 — là c'est une
mesure, pas une absence de mesure.

### Architecture du code

```
regime_lab/config.py                chemins, clé FRED, SAMPLE_START
regime_lab/data/pit.py              LE CŒUR : contrat point-in-time
                                    validate / realtime_trace / as_of / build_panel
regime_lab/data/coverage.py         refuse une source qui rend moins que demandé
regime_lab/data/store.py            parquet + manifeste à empreinte
regime_lab/data/quality.py          écran de valeurs répétées
regime_lab/data/universe.py         catalogue des séries, chemins vintage vs lag
regime_lab/data/sources/            fred.py, prices.py, kenfrench.py
regime_lab/features/                50 variables, 8 familles
  standardise.py                    z-score en fenêtre EXPANSIVE (jamais full-sample)
  market.py, crosssection.py, asymmetry.py, macro.py, build.py
regime_lab/models/
  protocol.py                       plis, calendrier de réestimation
  base.py                           run_expanding : réestime, prédit en ligne
  jump.py                           famille A — Statistical Jump Model
  hmm.py                            famille B — HMM à probabilités FILTRÉES
  supervised.py                     famille C — gradient boosting + HAR-RV
  calibrate.py                      choix de λ sur l'entraînement seul
  mapping.py                        état → position, règle gelée
regime_lab/evaluation/
  reliability.py                    couche 1 : persistance, accord, validation externe
  predictive.py                     couche 2 : moments conditionnels, placebo
regime_lab/analysis/
  bootstrap.py                      bootstrap stationnaire par blocs
  power.py                          effet minimum détectable
  trials.py                         journal d'essais append-only
regime_lab/strategies/
  base.py                           60/40 et momentum équipondéré en risque (GELÉES)
  book.py                           définition unique du livre, EN EXCESS
  costs.py                          coûts déclarés : 2, 5, 10 bp
regime_lab/extensions/              APRÈS l'étude gelée, ne modifie aucun résultat
  concentration.py                  facteurs effectifs, ratio d'absorption
  trend.py                          livre TSMOM + atténuateur de Carver
  persistence.py                    ratio de variance Lo-MacKinlay VALIDÉ + DFA
```

### Ordre d'exécution

```bash
cd ~/Desktop/regime-lab
.venv/bin/python scripts/fetch_data.py            # ~3 min, données déjà présentes
.venv/bin/python scripts/build_features.py        # 50 variables + test d'admissibilité
.venv/bin/python scripts/phase0_report.py         # qualité + puissance
.venv/bin/python scripts/run_phase2.py            # ⚠ 15-20 MIN — lancer en arrière-plan
.venv/bin/python scripts/run_evaluation.py        # couches 1 et 2
.venv/bin/python scripts/run_t2.py                # test décisif
.venv/bin/python scripts/run_layer3.py            # timing contre dimensionnement
.venv/bin/python scripts/run_extensions.py        # les suites post-étude
.venv/bin/python -m pytest -q                     # 74 tests
```

`run_phase2.py` est le seul long. Il écrit `states.parquet`,
`states_offline.parquet` et `refit_diagnostics.parquet`, dont tout le reste
dépend.

### Discipline de protocole

- Cadrage gelé : `docs/CHARTER.html`, empreinte
  `5791c1887c41d1f6b0449bafad2125013b82dc6b3506b4d72564e14ec1abc0db`
- Tag git `charter-v2` poussé publiquement (date opposable)
- `docs/PROTOCOL_FREEZE.md` : journal d'amendements — **un refusé** (élargir la
  grille de λ après avoir vu quelle extrémité gagnait), trois appliqués
- 83 configurations distinctes journalisées dans `data/trials.parquet`

**Ne jamais éditer `docs/CHARTER.html`.** Toute déviation va au journal
d'amendements.

### Les neuf défaillances silencieuses trouvées

Toutes documentées dans `docs/DATA_NOTES.md`, `docs/FEATURE_NOTES.md` et
`docs/PROTOCOL_FREEZE.md`. Six sur neuf étaient dans le code du projet.

1. FRED tronque à 100 000 lignes en annonçant le vrai total ailleurs (NFCI :
   100 000 sur 579 084, couvrant 1990-1995)
2. FRED sert les spreads ICE BofA sur deux ans seulement (licence)
3. Le motif `data/` du gitignore excluait `regime_lab/data/` du dépôt public
4. Le test décisif v1 était arithmétiquement vide (Sharpe invariant d'échelle)
5. Phase 0 en excess, Phase 2 en brut
6. Une famille de variables promise n'avait jamais été écrite
7. **La convention d'étiquetage inversait le meilleur classifieur** : `jumpmodels`
   trie en décroissant, le wrapper supposait l'inverse → long du plus mauvais
   état pendant trois jours
8. `regime_lab/config.py` exclu par le `.gitignore` **global** de la machine
   (`~/.gitignore_global:3`) → le dépôt public ne démarrait pas, et les tests
   passaient quand même
9. `RESULTS_T2.md` jamais régénéré après la correction n° 7 → cinq chiffres
   publiés trop flatteurs

⚠ **Le point 8 peut se reproduire.** `~/.gitignore_global` contient `config.py`.
Tout nouveau `config.py` doit être ajouté avec `git add -f`.

### Ce qu'un audit indépendant a validé

Un auditeur externe a exécuté chaque script. **Aucune fuite d'information nulle
part** : matrice de variables tronquée à 2010 comparée au plein échantillon,
0 colonne sur 50 qui bouge ; états hors échantillon identiques avec variables
tronquées à 2015 ; récursion forward du HMM correcte à 1,2 × 10⁻¹⁴ près ;
décalage T-1 appliqué exactement une fois ; bootstrap réellement apparié ;
`run_phase2.py` relancé depuis un clone frais rendant un `states.parquet`
**bit-identique**.

### Limites déclarées, à ne pas re-découvrir

- **Les révisions ne sont pas stockées.** `output_type=4` ne rend que la première
  publication. `realtime_trace` n'est exercé que par les tests.
- **Trois des six contrôles de falsification du cadrage n'ont jamais été
  implémentés** : T1 au niveau stratégie, T3 (placebo à exposition appariée),
  T5 (ARI entre réestimations successives).
- **Les cinq plis sont déclarés et jamais évalués.** `walk_forward` n'est appelé
  que pour lire la date de début du premier. Le protocole réel est un bloc
  continu avec 49 réestimations semestrielles.
- **La deuxième stratégie de base gelée n'a aucun site d'appel**
  (`equal_risk_momentum`).
- **La règle d'arrêt n° 2 se déclenche** (R² incrémental sur rendements de 0,007
  à 0,030 point, seuil 0,2) et c'est déclaré dans `RESULTS_FINAL.md`.
- **Le DFA utilise 3 échelles au lieu de 4** sur une fenêtre de 252 jours :
  l'échelle 64 est abandonnée car 252//64 = 3 boîtes < 4. Décision prise de
  **ne pas corriger** (l'étude est gelée, et Hurst avait un poids de zéro dans
  la sélection parcimonieuse).

### Extensions post-étude — résultats

Univers de 46 instruments, 2000-2026, livre TSMOM de référence Sharpe **0,51**
sur tout l'échantillon, **0,44** sur l'échantillon commun (à partir de 09/2003,
après l'amorçage de 756 jours du percentile expanding).
⚠ **Sharpe BRUTS dans les extensions**, pas en excess — non comparables aux
0,47/0,55 de l'étude principale.
⚠ **Toute comparaison se fait sur l'échantillon commun** : contre le livre plein
échantillon les mêmes écarts se lisent +0,07 au lieu de +0,14, et la différence
est la période d'amorçage, pas le conditionnement.

Le régime de volatilité se réplique et plus fort qu'à la source :
tercile bas 0,82 · médian 0,67 · **haut −0,32** (Carver : 0,49 / 0,40 / 0,003).

Le dispositif passe son placebo, et perd contre une règle sans paramètre :
- atténuateur par instrument, gross libre : 0,55, écart +0,037, t **+0,18**
  (gross renormalisé à 1 = **la mauvaise construction**, il réalloue au lieu de
  dé-risquer : −0,003, t −0,35 ; conservé comme témoin)
- atténuateur au niveau du livre, L = 2 − 1,5Q : 0,58, écart **+0,144, t 2,60**,
  **placebo apparié au 99ᵉ percentile** (moyenne +0,428 sur un livre
  inconditionnel à 0,44 ⇒ placebo non biaisé, contrôle L58)
- **mais la cible de vol dynamique SANS PARAMÈTRE fait 0,68, écart +0,244,
  t 2,89** — elle bat le dispositif de régime de 0,10 de Sharpe. Et l'écart
  +0,144 est sous le MDE de 0,205 ⇒ **UNDERPOWERED, pas un PASS**.
- variante binaire : plus gros gain de Sharpe (+0,199) avec un t **négatif**
  (−0,47) ⇒ le gain arrive au **dénominateur**, pas au numérateur
- compteur de facteurs effectifs : tous les |t| sous 1,6, ne prédit ni ne décrit

⚠ **Ces chiffres remplacent ceux de la version du 10/09** (0,88/0,89/−0,45 ;
+0,053 t 2,59 ; placebo 94ᵉ percentile ; MDE 0,190). Le code qui les produisait
n'avait jamais été commité — `run_extensions.py` ne calculait que l'atténuateur
par instrument. Analyse réécrite dans le script et relancée : la section 1 se
reproduit au chiffre près (données inchangées), les t se reproduisent (2,60 vs
2,59), les niveaux de Sharpe non, parce que l'ancienne version comparait des
constructions mesurées sur des **échantillons différents**. Le verdict tient, sa
raison change : le dispositif ne rate pas son placebo, il perd contre un témoin
que l'ancienne version n'avait jamais lancé. Détail → `docs/EXTENSIONS.md`
§Amendment.

⚠ **Piège reproduit deux fois** : une version antérieure donnait t = 2,18 sur le
compteur de facteurs. Elle exigeait que les 46 instruments aient tous des
données à chaque date, ce qui démarrait l'échantillon en 2007 et jetait 8 ans.
**Le signal apparent était la troncature.**

---

## 5. `macro-momentum` — deux hypothèses falsifiées

### H1 — signal absolu, un seul pays

Carte des signes gelée dans `docs/PRESPEC.md` avant tout calcul.

- livre net : **Sharpe −0,38**
- placebo sur la **carte** (400 tirages) : carte réelle au **7ᵉ percentile**
- placebo sur le **signal** (thèmes mélangés) : signal réel au **99ᵉ percentile**
- accord carte gelée / signes empiriques : **7 cellules sur 16** (le hasard : 8)
- **une seule cellule sur 16** dépasse |t| = 2

**Le signal porte de l'information, la carte des signes fait pire qu'un tirage à
pile ou face.** Les signes n'ont **pas** été retournés — le faire après avoir vu
donnerait +0,38 et serait la fabrication que le pré-enregistrement empêche.

Cause : AQR font du **transversal entre pays** sur des **révisions de
prévisions**. H1 faisait de l'**absolu à un pays** sur des **données réalisées**.
C'était du market timing.

### H2 — transversal entre 19 pays, variables d'état de marché

Pré-enregistré dans `docs/PRESPEC_H2.md` avant toute donnée.

**Zéro cellule sur neuf** dépasse |t| = 1,96 (hasard : 0,5). Maximum 1,45.
Seuil de Šidák pour 9 tests : 2,77. **Aucun livre n'a été construit.**

Le diagnostic est le résultat :

| période | dispersion des taux courts |
|---|---|
| 1990-1998 | 2,30 % |
| 2016-2026 | 0,89 % |

Et chez les 9 pays de la zone euro, l'écart-type de leurs propres taux courts :
**2,35 % avant 1999, exactement 0,00 % après.**

**La stratégie n'a pas décayé, sa matière première a été supprimée par traité.**
Après 1999 c'est un kill mécanique, fermé. Avant 1999 c'est un défaut de
puissance (9 ans, 4 cellules), donc **rouvrable**.

### Exécution

```bash
cd ~/Desktop/macro-momentum
.venv/bin/python scripts/fetch.py       # H1
.venv/bin/python scripts/run.py         # H1, ~2 min (400 placebos)
.venv/bin/python scripts/fetch_h2.py    # H2
.venv/bin/python scripts/run_h2.py      # H2, rapide
```

---

## 6. `reversal-lab` — la prime a disparu

Mesuré sur les portefeuilles CRSP de Ken French, **sans biais de survie**.

| période | Sharpe | rendement | prime quotidienne |
|---|---|---|---|
| 1990-1999 | 3,64 | +61,2 % | +24,3 bp |
| 2000-2009 | 1,23 | +42,5 % | +16,9 bp |
| 2010-2019 | 0,54 | +8,9 % | +3,5 bp |
| **2020-2026** | **−0,18** | **−6,9 %** | **−2,7 bp** |

Positif **9 années sur 17** depuis 2010 (pile ou face : 8,5). Le chiffre des
années 1990 n'était pas récoltable — l'écart de cotation valait un huitième de
dollar.

Test de Nagel : depuis 2010, seul le tercile de VIX **haut** est positif, à
0,18-0,39 **brut**, dans le régime où les fourchettes sont les plus larges.

**Kill mécanique, fermé.** La couche de conditionnement par estimateur de
persistance n'a **délibérément pas** été lancée sur l'échantillon récent —
conditionner une prime morte, c'est chercher le sous-ensemble où elle survit.

```bash
cd ~/Desktop/reversal-lab
.venv/bin/python scripts/fetch.py
.venv/bin/python scripts/run_premium.py
```

---

## 7. Sur Hurst et le ratio de variance — établi, ne pas re-tester

Beaucoup de travail a été fait là-dessus. Résumé pour éviter de le refaire.

**Le code DFA est correct**, prouvé par calibration sur bruit gaussien
fractionnaire exact (plongement de Davies-Harte) : il retrouve un H connu à
±0,01 près sur 0,3 / 0,4 / 0,5 / 0,6 / 0,7.

**Pourquoi H sort de [0,1] :** c'est une régression linéaire ordinaire sur 3 à 5
points ; rien ne contraint une pente. Et **seules les queues épaisses** le
provoquent — Student t3 donne −0,48 à 1,51, alors que gaussien et GARCH ne
sortent jamais de [0,1]. Winsoriser avant d'appliquer.

**Le ratio de variance naïf est massivement biaisé** : sur marche aléatoire de
756 points il rend 0,813 à q=126 au lieu de 1,000, avec un écart-type de 0,41.
La correction Lo-MacKinlay l'efface : **0,999**, taux de rejet 5,2-7,8 % contre
5 % nominal. Implémentation validée dans
`regime_lab/extensions/persistence.py`.

**Ce que ça donne sur données réelles** (58 actifs, 5 classes, plein
échantillon, statistique robuste) : **3 rejets du marché aléatoire sur 58, quand
le hasard en donne 2,9.** Le rejet le plus fort de tout l'univers est SHY, un
ETF d'obligations courtes dont la « tendance » est le niveau des taux.

**Sélectionner des actifs par Hurst ne marche pas** : sur 49 secteurs, tri par
Hurst → tendance 12-1 à 0,35 contre 0,33 selon le tercile — aucune séparation.
Le ratio de variance sépare (0,10 contre 0,47) mais l'écart est dans le bruit.

**Le conditionnement au choc produit du bêta** : jambe longue R² 90 %, bêta 0,99,
alpha **−2,19 %/an** (t −2,01). Le marché équipondéré fait 0,75 sur la même
période, la stratégie 0,60. Elle sous-performe le buy-and-hold.

---

## 8. Ce que la pratique fait vraiment (scan praticien, sources non académiques)

Résumé de la note d'orientation. Utile pour ne pas réinventer.

- **Le socle universel n'est pas un modèle de régime, c'est le ciblage de
  volatilité.** Carver mesure Sharpe 0,92 avec contre 0,569 sans, sur 37 futures
  et 36 ans.
- **Two Sigma écrit de son propre modèle : « this model is not predictive ».**
  Dans leur produit, le régime ne s'active que pour les chocs > 3σ.
- **Bridgewater refuse explicitement de classer** — risque égal sur chaque
  scénario.
- **Man Numeric a abandonné les catégories discrètes** pour une mesure continue.
- **Le seul à revendiquer un timing directionnel** (Man Group 2025, Sharpe 0,82)
  admet un biais d'anticipation dans le choix de ses variables d'état, et sa
  méthode cousine tombe au **98ᵉ percentile de 160 000 tirages aléatoires**.
- **Ce qui prédit la moyenne chez les praticiens, ce sont les *variations* macro,
  pas la *classification* en états** (AQR, Sharpe 1,2 sur 1970-2016).
- **Le piège de data-mining est dans la carte régime → actifs**, pas dans le
  classifieur.
- **AlphaSimplex : c'est la DURÉE du drawdown qui compte, pas la profondeur.**
  Covid −34 % en 23 jours → trend −2,4 %. Bulle internet −47 % en 545 jours →
  trend +62 %. En dessous de ~12,5 % de drawdown, le trend n'a jamais été
  positif.
- **Aucune trace de déploiement de jump models en fonds.**
- **Personne de sérieux ne dépasse 4 états.**

À éviter selon la pratique : l'interrupteur ON/OFF (coût mesuré 2-4 %/an), la
table régime → actifs sans placebo, le timing factoriel par la valorisation,
multiplier les états, rebalancer à date fixe sans étaler (le *timing luck*
dépasse 100 bps annualisés).

---

## 9. Pistes ouvertes, par priorité

**1. Le notebook Dedale** — `https://github.com/brieuctrader/Dedale/blob/main/Python/macro_score_overview.ipynb`
C'était la prochaine tâche quand la session s'est arrêtée. Ni les métriques ni
les poids des indicateurs ne sont accessibles. L'approche prévue : lire le
notebook pour en extraire la **structure** (quels indicateurs, quelle
agrégation, quelle règle de décision), la reconstruire avec nos données et notre
discipline, et dire ce qui tient. Une partie a déjà été testée indirectement.

**2. Le régime branché sur la géométrie de barrière** — troisième orientation du
scan praticien, jamais exécutée. Elle a besoin du simulateur FTMO qui vit dans
`~/Desktop/Algo_claude/Portfolio/risk/trailing_dd_simulator.py`. Sortie utile :
**Δp(pass) et Δdurée**, pas ΔSharpe.

**3. Écrire le résultat de dispersion macro.** H1 + H2 forment un seul résultat
original : *la macro des pays développés n'a plus assez de dispersion
transversale pour être tradée en relatif, et en absolu c'est du market timing.*
Entièrement mesuré, il ne reste qu'à rédiger. C'est le résultat le plus fort du
programme après celui de `regime-lab`.

**4. Marchés émergents** — la seule réouverture légitime. Vérifié disponible et
gratuit : 12 devises depuis 2006, 12 indices depuis 2012, 6 ETF obligataires
locaux depuis 2013. Et la dispersion y **monte** pendant que celle du G10
s'effondre : rapport EM/G10 de 0,7× en 2006-2012 à **1,5× en 2020-2026**.
Obstacles : échantillon court, coûts 5-10× le G10, contrôles de capitaux
invisibles dans un backtest au comptant.

**5. L'argument d'arrêt.** Trois hypothèses, trois falsifications. Chaque essai
déflate tout ce qui suit. Le mémoire est fini, audité et défendable ; la valeur
marginale d'un quatrième kill est faible comparée à celle de bien écrire les
trois premiers.

---

## 10. Pièges spécifiques à cette machine

- **`~/.gitignore_global` contient `config.py`.** Tout nouveau fichier de ce nom
  doit être ajouté avec `git add -f`, sinon il disparaît du dépôt sans que
  `git status` ne dise rien.
- **Stooq est derrière un mur anti-bot** à preuve de travail JavaScript.
  Téléchargement en masse et endpoint CSV unitaire renvoient tous deux la page
  de défi. Ne pas y compter.
- **Les codes pays OCDE sur FRED sont à deux lettres** (`IRLTLT01DEM156N`), pas
  trois. Une vérification avec `DEU` renvoie « indisponible » à tort.
- **Les millésimes ALFRED ne passent pas par l'endpoint CSV public.** Toutes les
  formes documentées renvoient une page d'erreur HTML. Il faut l'API avec clé.
- **`yfinance` échoue par intermittence** sur des tickers parfaitement valides
  (`COST`, `WFC` ont échoué une fois avec « possibly delisted »). Relancer.
- **La profondeur des millésimes ALFRED varie énormément par série** :
  production industrielle 1990, inscriptions au chômage 2009, indice de stress
  de Saint-Louis 2022.

---

## 11. Style de travail attendu

Établi sur toute la session, à conserver.

- **Mesurer plutôt qu'affirmer.** Chaque affirmation technique du projet est
  adossée à une commande qui a réellement tourné.
- **Vérifier la disponibilité d'une source avant de la recommander.** Deux
  recommandations ont été faites sans test (Stooq, codes OCDE) et les deux
  étaient fausses.
- **Le placebo avant la conclusion.** Un effet sans son nul n'est pas un
  résultat.
- **Nommer ses propres erreurs explicitement**, y compris quand elles invalident
  quelque chose qui vient d'être présenté. Neuf défaillances documentées, six
  dans le code du projet.
- **Ne jamais ajuster après avoir vu.** Retourner les signes de H1 aurait donné
  +0,38 ; ne pas le faire est le résultat.
- **Rédaction en français**, code et documentation en anglais.
