# Bridgewater — quadrant croissance × inflation, parité des risques

Plan de recherche pré-enregistrable. Rédigé le 22 septembre 2026, **avant toute
mesure de rendement**. Aucun rendement de backtest n'a été lu pour l'écrire ;
les seules ouvertures de données ont servi à compter des lignes, des dates, des
cellules et des transitions. Les scripts de comptage sont à côté de ce fichier
(`measure_quadrant.py`, `measure_spf_quadrant.py`, `measure_level_quadrant.py`,
`measure_sleeves.py`, `measure_cells.py`, `measure_power.py`).

---

## a) Test d'admission

Les six réfutations partagent la même étroitesse : variable latente = volatilité,
0,51 transition par an, usage = dimensionnement ou timing, objet conditionné = un
livre de tendance unique. Voici où cette approche s'en écarte, chiffré.

### Axe 1 — la variable latente : franchi

Ce n'est plus la volatilité réalisée. C'est un couple **(surprise de croissance,
surprise d'inflation)** mesuré contre un **consensus réel**, celui du Survey of
Professional Forecasters de la Fed de Philadelphie.

Pour le trimestre *q*, le SPF publie `RGDP2` (sa prévision du niveau de PIB réel
de *q*) et, une enquête plus tard, `RGDP1` (le niveau de *q* tel qu'on le
connaissait alors). L'écart entre les deux est une vraie surprise de consensus,
et les deux membres étaient publics au moment où la seconde enquête est parue.
Même construction pour l'inflation, dont les colonnes `CPI1`/`CPI2` sont déjà des
taux annualisés.

Mesuré : **180 trimestres exploitables, 1981-T3 à 2026-T2**, écart-type de la
surprise de croissance 35,5 points annualisés, de la surprise d'inflation 1,333
point. Sur la fenêtre où les actifs existent, **91 trimestres, 22,5 ans**.

Ceci **résout la difficulté centrale** que la tâche signale. H3 avait contourné
l'absence de consensus par une autorégression validée à r = +0,667 contre le SPF.
Ici on n'a pas besoin du proxy : **le SPF lui-même est disponible, gratuit et
vérifié** — j'ai téléchargé et ouvert `median_rgdp_level.xlsx` (232 × 12),
`median_cpi_level.xlsx` (232 × 11), `median_unemp_level.xlsx` et
`median_indprod_level.xlsx`, tous des .xlsx réels. Le proxy H3 reste utile comme
**variante quotidienne** (voir §b), pas comme béquille.

⚠ **Mon erreur, nommée.** Mes deux premières URL SPF (`median_ngdp_level.xlsx`,
`median_cpi_growth.xlsx`) renvoient **HTTP 200 avec une page HTML « Error - 404 »
de 18 401 octets**, identique pour les deux (même md5). C'est exactement le piège
Stooq et les codes pays OCDE : un code de retour 200 ne dit pas qu'on a le
fichier. Les quatre noms retenus ont été vérifiés par `file` (archive ZIP) puis
ouverts avec pandas. Le chemin qui marche est celui que
`macro-momentum/scripts/validate_surprise.py:26` utilise déjà, en **minuscules**.

### Axe 2 — la constante de temps : franchi, 6,1×

| objet | transitions / an | P(même état au trimestre suivant) |
|---|---|---|
| A′ sparse jump — le classifieur réfuté | **0,51** | ≈ 0,99 |
| quadrant SPF, grille trimestrielle, échantillon actifs | **3,11** | **0,220** |
| quadrant SPF, historique complet 44,7 ans | 2,95 | — |
| quadrant proxy H3 (noyau 63 j), grille trimestrielle | 2,27 | — |
| quadrant proxy H3 (noyau 63 j), grille quotidienne | 7,08 | — |
| quadrant de **niveau** (glissement annuel contre médiane expansive) | 1,20 | 0,691 |

**3,11 transitions par an contre 0,51 : facteur 6,1.** Ce n'est pas H-b refait.

Mais le chiffre qui compte davantage est le second : **persistance 0,220, contre
0,266 pour des tirages indépendants à cette occupation**. Le quadrant de surprise
de consensus n'est pas un état persistant — il est **très légèrement
anti-persistant**, c'est-à-dire indiscernable d'une suite de chocs indépendants.
C'est ce qu'un consensus efficient doit produire, et c'est une propriété
mesurée, pas supposée.

Cette propriété **interdit** tout usage ex ante du quadrant. On ne peut pas le
prédire, on ne peut pas s'y positionner à l'avance. C'est précisément ce que
Bridgewater dit faire : All Weather **ne prédit pas le quadrant**, il équilibre
le risque à travers les quatre pour n'être ruiné par aucun. La non-prédictibilité
n'est pas un défaut de cette approche, c'en est la prémisse. Elle disqualifie en
revanche toute variante de timing, et le plan n'en contient aucune.

Occupation mesurée sur l'échantillon actifs (91 trimestres) :
(croissance−, inflation−) 14,3 % · (croissance−, inflation+) 30,8 % ·
(croissance+, inflation−) 27,5 % · (croissance+, inflation+) 27,5 %.
**Les quatre cellules sont peuplées**, ce qu'une matrice à quatre quadrants exige.

### Axe 3 — l'usage : franchi, et c'est le cœur

Les six réfutations testaient **dimensionnement** (T1, l'atténuateur de Carver,
H-b) et **timing** (T3, la barrière, Shu et al.). Aucune n'a testé la
**construction de portefeuille**.

Ici, le quadrant n'entre **jamais** dans une décision datée. Il entre dans la
**fonction objectif** : les poids de manches sont choisis pour que chacune des
quatre cellules contribue une part égale à la variance du portefeuille, la
covariance conditionnelle à la cellule étant estimée **sur le seul pli
d'entraînement**. Le portefeuille est ensuite détenu **sans savoir dans quelle
cellule on se trouve**. C'est All Weather, littéralement.

### Axe 4 — l'objet conditionné : franchi

Pas un livre de tendance 12-1 sur 46 instruments. Un **panel de cinq manches** —
actions, duration, indexé inflation, matières premières, métaux précieux —
couvrant **30 instruments**, du 05/12/2003 au 10/09/2026, **23,6 ans**.

**Quatre axes sur quatre. L'approche est admissible.**

---

## b) La donnée — ce qui existe, vérifié en l'ouvrant

### Sur disque, ouvert et compté

`regime-lab/data/raw/macro/` — onze séries point-in-time, colonnes
`series_id, period, available_at, value` :

| série | lignes | période couverte | `available_at` | fréquence |
|---|---|---|---|---|
| `macro_cpi` | 438 | 1990-01-01 → 2026-07-01 | 1990-02-21 → 2026-08-12 | mensuelle |
| `macro_indpro` | 439 | 1990-01-01 → 2026-07-01 | 1990-02-16 → 2026-08-18 | mensuelle |
| `macro_payems` | 440 | 1990-01-01 → 2026-08-01 | 1990-02-02 → 2026-09-04 | mensuelle |
| `macro_unrate` | 439 | 1990-01-01 → 2026-08-01 | 1990-02-02 → 2026-09-04 | mensuelle |
| `macro_claims` | 1 913 | 1990-01-06 → 2026-08-29 | 1990-01-13 → 2026-09-05 | hebdomadaire |
| `rate_cash_3m` | 9 177 | 1990-01-02 → 2026-09-08 | → 2026-09-09 | quotidienne |
| `rate_curve_10y2y` | 9 178 | 1990-01-02 → 2026-09-09 | → 2026-09-10 | quotidienne |
| `rate_curve_10y3m` | 9 178 | idem | idem | quotidienne |
| `fin_nfci` | 1 913 | 1990-01-05 → 2026-08-28 | → 2026-09-04 | hebdomadaire |
| `fin_aaa_spread` | 9 171 | 1990-01-02 → 2026-09-04 | → 2026-09-05 | quotidienne |
| `fin_baa_spread` | 9 171 | idem | → 2026-09-09 | quotidienne |

⚠ Ces onze séries **ne portent qu'une seule valeur par période** (pas de
révisions multiples : 438 lignes pour 438 périodes sur le CPI). Le contrat
point-in-time de `regime_lab/data/pit.py` est respecté au sens où `available_at`
existe, mais il s'agit d'un **décalage de publication appliqué à une série
révisée**, pas d'un véritable historique de millésimes. `macro_indpro` passe de
141,8 à 108,8 entre février et mars 1990, ce qui est une **rebasing**, signature
d'une série non millésimée. **Ces onze séries ne peuvent pas servir d'axe de
surprise.** Elles servent aux variantes de niveau et aux contrôles.

`macro-momentum/data/raw/h3_macro/` — **treize séries réellement millésimées**,
premières publications incluses (`payems` : 4 560 lignes pour 344 périodes, donc
4 216 révisions stockées). Premier z exploitable **2003-03-07** après le rodage
de l'autorégression ; `claims` seul commence en **2009-05-28**, ce qui coûterait
six ans — il est donc **exclu de l'axe croissance du titre** et gardé en
sensibilité sur 2009+.

`regime-lab/data/cache/trend_universe.parquet` — 6 822 séances × 46 colonnes,
17/07/2000 → 10/09/2026. Les manches et leur date contraignante :

| manche | n | date de départ | instrument contraignant |
|---|---|---|---|
| actions | 12 | 2000-07-17 | ^GSPC |
| duration | 4 | 2003-09-29 | AGG |
| indexé inflation | 1 | **2003-12-05** | TIP |
| matières premières | 11 | 2000-09-15 | ZS=F |
| métaux précieux | 2 | 2000-08-30 | GC=F |
| crédit | 3 | 2007-12-19 | EMB |
| change | 11 | 2006-05-16 | AUDUSD=X |

### La configuration retenue, et pourquoi

| configuration | départ | années | n instr. | plus petit Sharpe résoluble |
|---|---|---|---|---|
| 4 manches (act., dur., TIP, mat.) | 2003-12-05 | 23,6 | 28 | 0,632 |
| **5 manches (+ précieux) — le titre** | **2003-12-05** | **23,6** | **30** | **0,632** |
| 6 manches (+ crédit) | 2007-12-19 | 19,4 | 33 | 0,712 |
| 7 manches (+ change) | 2007-12-19 | 19,4 | 44 | 0,712 |
| 3 manches sans TIP | 2003-09-29 | 23,8 | 27 | 0,629 |

**Le titre prend les cinq manches, 30 instruments, 23,6 ans.** Ajouter le crédit
coûte 4,2 ans et fait passer le plus petit Sharpe résoluble de 0,632 à 0,712,
donc au-dessus de la porte à 0,70 : c'est exactement le piège de profondeur
temporelle que l'audit M2 a documenté le 21/09. Le crédit et le change sont
**des sensibilités sur 19,4 ans, jamais le titre**.

⚠ **N = 30 exactement.** La règle N ≥ 30 est satisfaite au niveau de
l'instrument et sans aucune marge. Au niveau de la **décision**, la largeur est
de **5 manches**, et je le déclare plutôt que de le laisser passer : c'est la
raison pour laquelle les plis de walk-forward sont minces.

### Ce qui manque, et ce qui a été testé plutôt que supposé

| besoin | source | statut |
|---|---|---|
| consensus croissance | SPF `median_rgdp_level.xlsx` | ✅ 232 × 12, 1968-T4 → 2026-T3, ouvert |
| consensus inflation | SPF `median_cpi_level.xlsx` | ✅ 232 × 11, ouvert |
| consensus chômage, prod. ind. | SPF `median_unemp_level.xlsx`, `median_indprod_level.xlsx` | ✅ ouverts |
| point mort d'inflation 10 ans | FRED `T10YIE` | ✅ HTTP 200, **6 188 lignes, 2003-01-02 → 2026-09-21**, CSV réel |
| TIPS réel 10 ans | FRED `DFII10` | ✅ 6 188 lignes, 2003-01-02 → 2026-09-18 |
| taux cash de financement | `rate_cash_3m` sur disque, **et** FRED `DTB3` | ✅ 9 177 lignes sur disque ; 18 971 en ligne depuis 1954 |
| PIB réel trimestriel | FRED `GDPC1` | ✅ 319 lignes, 1947 → 2026-04-01 |
| rendement total des indices actions | — | ❌ **manquant**, voir tueur n°5 |
| dates exactes de parution du SPF | table de parution de la Fed de Philadelphie | ⚠ **non testée**, voir tueur n°8 |

⚠ `openpyxl` **n'est pas dans le venv de regime-lab** (`ImportError`) ; il est
dans celui de macro-momentum. C'est le point d'hygiène **B5** déjà journalisé.
Toute lecture SPF passe par le venv de macro-momentum ou déclare la dépendance.

---

## c) La puissance a priori

Toutes les lignes ci-dessous sortent de `measure_power.py`, qui applique
l'inversion de `regime_lab/analysis/power.py` déjà utilisée par
`scripts/audit_vehicle_coverage.py`.

### Le statistique de moyenne est mort avant d'être testé

Plus petit Sharpe séparable de zéro, autonome, 80 % de puissance, bilatéral 5 % :

| échantillon | années | SR résoluble |
|---|---|---|
| panel 5 manches | 23,60 | **0,632** |
| cellule (croissance−, inflation+), 1 792 séances | 7,11 | 1,569 |
| cellule (croissance+, inflation+), 1 675 séances | 6,65 | 1,698 |
| cellule (croissance+, inflation−), 1 630 séances | 6,47 | 1,757 |
| **cellule (croissance−, inflation−), 841 séances** | **3,34** | **∞** |

**La plus petite cellule est indécidable sur la moyenne, et c'est un résultat à
coût nul obtenu avant d'écrire une ligne de stratégie.** En dessous de 3,92 ans
le dénominateur de l'inversion devient négatif : aucun Sharpe, si grand soit-il,
n'est séparable de zéro dans cette cellule. Toute variante dont le critère
primaire serait « le Sharpe de la pire cellule » est **non résoluble par
construction** et ne peut jamais rendre un PASS.

En apparié, la différence de deux constructions partageant les mêmes manches est
mieux résolue, parce qu'elles sont très corrélées
(MDE = 2,8016 × √(2(1−ρ)) × SE) :

| échantillon | SE | ρ=0,90 | ρ=0,95 | ρ=0,98 | ρ=0,99 |
|---|---|---|---|---|---|
| panel complet 23,6 ans | 0,2183 | 0,274 | **0,193** | 0,122 | 0,087 |
| un pli sur cinq (4,7 ans) | 0,4882 | 0,612 | 0,433 | 0,274 | 0,193 |
| plus grande cellule (7,1 ans) | 0,3977 | 0,498 | 0,352 | 0,223 | 0,158 |
| plus petite cellule (3,3 ans) | 0,5806 | 0,727 | 0,514 | 0,325 | 0,230 |

Repères mesurés par le programme : K3 dimensionnement, MDE 0,246, écart +0,068
→ UNDERPOWERED. K4 régime, résolution 0,043, écart +0,014 → UNDERPOWERED.

### Le statistique de variance, lui, est résoluble — et c'est celui que le programme a gagné le droit d'utiliser

Le programme a établi que cet objet **porte la variance et pas la moyenne** :
+3,93 points de R² incrémental sur la volatilité future (t −3,40) contre +0,030
point sur les rendements (t 0,27). Et la revendication d'All Weather est
elle-même une phrase sur la variance : « le risque est équilibré à travers les
quatre quadrants ». Le statistique primaire suit :

> **D = log(variance de la pire cellule) − log(variance de la meilleure cellule)**,
> la dispersion des quatre variances conditionnelles à la cellule.

Erreur-type de log(σ̂) ≈ 1/√(2n), en blocs de 63 séances parce que le trimestre
est l'unité d'information indépendante :

| échantillon | n_eff (blocs 63 j) | MDE sur le log-ratio |
|---|---|---|
| panel complet, 5 938 séances | 94,3 | **0,204** |
| plus grande cellule, 1 792 séances | 28,4 | 0,371 |
| plus petite cellule, 841 séances | 13,3 | 0,542 |

**MDE de 0,204 en log-ratio = une réduction de 22,6 % du rapport de variance
extrême.** Cible visée : une construction équilibrée qui ferait passer la
dispersion de 3:1 à 2:1 produit un écart de log(1,5) = **0,405**, soit **le
double du MDE**. De 3:1 à 2,5:1 donne log(1,2) = 0,182, **sous le MDE** et donc
déclaré UNDERPOWERED d'avance.

**Verdict de puissance, écrit avant la mesure : la question est décidable sur la
variance pour une réduction de dispersion d'au moins 23 %, et indécidable sur la
moyenne dans la plus petite cellule quelle que soit la taille de l'effet.**

---

## d) Les coûts, et le niveau où l'hypothèse meurt

Barème `regime_lab/extensions/vehicle.py`, aller-retour en points de base :
actions 1,0 · change 1,0 · taux 1,0 · matières premières 1,5 · véhicule cash 7,5.
Sur les 30 instruments du panel — TIP seul au tarif cash, 13 matières premières,
12 indices, 4 taux — **le mélange équipondéré paie 1,217 bp par aller-retour**.

| rotation supplémentaire de la jambe conditionnée | frein annuel | Sharpe perdu à 10 % de vol cible |
|---|---|---|
| +2 AR/an | 0,024 % | 0,0024 |
| +4 AR/an | 0,049 % | 0,0049 |
| +8 AR/an | 0,097 % | 0,0097 |
| +16 AR/an | 0,195 % | 0,0195 |

**Écrit avant de tester : l'hypothèse est déclarée morte par les coûts si le
frein différentiel dépasse 0,10 de Sharpe.** Au barème `headline` cela demande
**82 aller-retours supplémentaires par an** ; la jambe conditionnée en produira
au plus 4 (calendrier) + 3,11 (transitions) ≈ 8. **Cette hypothèse ne mourra
donc pas de frottement.** Elle mourra de puissance ou de placebo, et le dire
d'avance évite d'attribuer plus tard à des coûts un échec qui n'en vient pas.

Le seul barème capable de la tuer est la lecture littérale `all_cash` à 7,5 bp
sur les 30 : il suffit alors de **13,3 aller-retours par an**. Les trois colonnes
(`headline`, `conservative`, `stress`) et la variante `all_cash` sont rendues
côte à côte, et la règle du véhicule s'applique : **un résultat dont le signe
bouge entre `headline` et `conservative` est déclaré indécis, jamais un PASS**.

---

## e) Les placebos appariés

Trois, et chacun apparie sur une dimension nommée.

**P1 — placebo de partition.** Étiquettes de quadrant tirées au hasard en
conservant l'occupation mesurée (0,143 / 0,308 / 0,275 / 0,275) et la
distribution des durées de séjour (médiane 1 trimestre, maximum 3), 2 000
tirages par bootstrap stationnaire par blocs sur les trimestres. Il répond à la
question qui tue : **la réduction de dispersion vient-elle de ce quadrant-là, ou
n'importe quelle partition de même forme la donnerait-elle ?** Ce placebo est
sévère précisément parce que la persistance mesurée est 0,220, donc presque
indiscernable d'un tirage indépendant.

**P2 — placebo d'exposition, la leçon de T3.** Les deux jambes sont appariées
séance par séance sur l'exposition brute moyenne et sur la volatilité ex ante.
T3 a montré qu'un alpha apparent passait **entièrement par le bêta** — 0ᵉ
percentile en bêta, médian en moyenne ; l'étude de barrière a montré que
p(pass) s'achète avec de l'exposition à **0,2785 par unité**. Aucune différence
non appariée sur l'exposition ne sera lue.

**P3 — placebo de carte, pour le niveau B2.** 2 000 cartes manche → quadrant
tirées au hasard. C'est **exactement le test que H1 a échoué**, où la carte des
signes argumentée s'est retrouvée au **7ᵉ percentile** des cartes aléatoires. La
carte a priori ne vaut que si elle bat ses permutations.

---

## f) Les tueurs connus, sans les minimiser, avec le sens dans lequel ils poussent

1. **La plus petite cellule est indécidable sur la moyenne.** 841 séances,
   3,34 ans, SR résoluble infini. → pousse à faire du statistique de variance le
   primaire et du Sharpe par cellule un diagnostic rendu mais jamais capable de
   PASS.
2. **La parité des risques est peut-être déjà équilibrée par quadrant.** Les
   manches ont été choisies *parce qu'*elles couvrent croissance et inflation ;
   égaliser les contributions de variance par cellule pourrait retrouver presque
   les mêmes poids. → c'est le mode d'échec le plus probable du niveau A, et il
   pousse vers une différence nulle.
3. **Le quadrant est quasi indépendant d'un trimestre à l'autre** (0,220 contre
   0,266). → P1 devient un placebo presque parfait, et pousse contre l'idée que
   cette partition-là porte quoi que ce soit de spécifique.
4. **H1 est morte sur une carte des signes.** → pousse contre B2, et impose P3.
5. **Les indices actions sont des indices de prix, pas de rendement total**,
   alors que les neuf ETF obligataires et de crédit sont en rendement total
   (`auto_adjust=True`) et doivent être financés au taux cash. L'asymétrie vaut
   environ 2 points de dividende par an. → pousse mécaniquement les poids de
   parité des risques **vers les obligations**. Remède : financer les ETF avec
   `rate_cash_3m` comme la correction A4 l'a fait — elle valait 0,0414 de
   Sharpe — et rajouter un rendement de dividende à la manche actions en
   sensibilité déclarée.
6. **23,6 ans qui commencent en décembre 2003 contiennent un seul changement de
   régime d'inflation et un marché obligataire haussier séculaire.** La cellule
   (croissance−, inflation−) ne compte que 13 trimestres, concentrés autour de
   2008-2015. → la cellule risque d'être **une date déguisée en état**. Remède :
   rendre la composition par décennie de chaque cellule **avant** le statistique.
7. **La largeur de décision est 5, pas 30.** → plis de walk-forward minces :
   5 plis donnent 18,2 trimestres et 1 187 séances chacun.
8. **L'horodatage du SPF est approximé.** Je date la surprise du trimestre *q*
   au 14 du deuxième mois de *q*+1. Les dates réelles varient. Une erreur dans
   le mauvais sens est une fuite. → remède : utiliser la table de parution
   publiée, **que je n'ai pas testée**, et ajouter en sensibilité déclarée un
   mois entier de retard supplémentaire.

---

## g) L'arbre d'escalade

Trois niveaux, chaque critère chiffré d'avance, chaque niveau échappant à la
raison nommée pour laquelle le précédent est tombé.

### Niveau A — le quadrant comme partition d'évaluation, rien de plus

**Ce qu'on fait.** Deux constructions sur les mêmes 30 instruments, même cible de
volatilité quotidienne à 10 %, même barème de coût, même grille de rebalancement
trimestrielle : (i) parité des risques inconditionnelle sur les cinq manches ;
(ii) parité des risques **équilibrée par cellule**, les poids choisis pour
égaliser la contribution de variance des quatre cellules, covariance
conditionnelle estimée **sur le seul pli d'entraînement**. Le quadrant courant
n'est jamais lu en exploitation.

**Critère de falsification.** Primaire : D = log(var pire cellule) − log(var
meilleure cellule), sur les deux jambes, appariées. PASS exige les trois
conditions ensemble :
- réduction de D d'au moins **0,204** (le MDE mesuré) ;
- percentile ≥ 95 contre **P1** (2 000 partitions de même forme) ;
- survie à **P2** (appariement d'exposition et de volatilité ex ante) ;
- et le signe stable entre les colonnes `headline` et `conservative`.

Secondaire, rendu mais **incapable de PASS par déclaration** : le Sharpe net en
excess de chaque cellule, avec son MDE en face. La cellule (croissance−,
inflation−) est déclarée indécidable d'avance.

**Effort** 6 jours. **P(succès) ≈ 30 %.**

### Niveau B — si A tombe, et la branche dépend de *pourquoi*

**B1, si A tombe parce que P1 l'égale** (la partition ne porte rien de
spécifique). Alors le défaut est dans **l'axe**, pas dans l'usage : une surprise
de consensus est inprédictible par construction, persistance 0,220, donc elle
découpe l'échantillon en blocs quasi aléatoires — ce qui explique qu'une
partition aléatoire fasse aussi bien. **B1 remplace l'axe par un axe implicite
de marché, persistant** : inflation = variation du point mort 10 ans `T10YIE`
(6 188 observations quotidiennes depuis le 02/01/2003, vérifié) ; croissance =
variation de `fin_baa_spread` (9 171 observations sur disque). Même construction,
même critère. **B1 échappe à la raison de la chute de A parce que son axe est
persistant et parce que c'est l'anticipation que le portefeuille subit
réellement, pas celle que des économistes ont écrite.**

**B2, si A tombe parce que l'écart est sous le MDE** (l'échantillon ne voit
rien). Alors le défaut est dans **l'estimation** : égaliser quatre contributions
de variance à partir de 13 à 28 trimestres par cellule est trop bruité. **B2
supprime toute estimation conditionnelle** et la remplace par une **carte a
priori manche → quadrant**, déclarée dans le pré-enregistrement et jamais
ajustée : actions → croissance+ ; duration → croissance−/inflation− ; TIP et
matières premières → inflation+ ; précieux → inflation+/croissance−. Les poids
sortent d'une contrainte de signe, pas d'une covariance. **B2 échappe à la
raison de la chute de A parce qu'il n'estime plus rien sur les cellules minces.**
Placebo obligatoire **P3**, 2 000 cartes aléatoires, le test que H1 a échoué au
7ᵉ percentile.

**Effort** 1,5 jour par branche. **P(succès | atteint) ≈ 20 %.**

### Niveau C — si B tombe, le seul usage qui reste

**C.** La date de rebalancement. Le quadrant n'a pas besoin d'être prédit pour
qu'on rebalance **le jour où il change** plutôt qu'à date fixe. On compare la
même construction rebalancée sur les dates de transition (3,11 par an mesurées)
contre un calendrier trimestriel (4,0 par an), donc à fréquence presque
appariée. Placebo : dates tirées au hasard avec la même loi d'espacement. C'est
un test d'**exécution**, le seul des cinq usages que ni A, ni B, ni les six
réfutations n'ont touché. **Effort** 1 jour. **P(succès | atteint) ≈ 12 %.**

**Si C tombe : fermeture déclarée**, avec le motif écrit d'avance — *le couple
croissance × inflation contre attentes n'ajoute rien à un jeu de manches déjà
choisi pour le couvrir, testé comme étiquette d'évaluation, comme axe, comme
carte a priori et comme grille d'exécution.* Ce serait la **première fermeture du
programme sur l'axe construction**, et elle porterait sur quatre usages, pas un.

### Correction pour tests multiples, calculée sur l'arbre entier

L'arbre déclare **cinq tests** : A primaire, A secondaire, B1, B2, C. Au plus
trois seront exécutés, mais la correction couvre les cinq, parce que c'est le
choix de branche qui dépend du résultat.

| tests déclarés | α par test (Šidák, famille 0,05) | z bilatéral |
|---|---|---|
| 1 | 0,05000 | 1,960 |
| 3 | 0,01695 | 2,388 |
| **5 — cet arbre** | **0,01021** | **2,569** |
| 6 | 0,00851 | 2,631 |

**Seuil retenu : |t| ≥ 2,569, HAC Newey-West lag 6, sur bootstrap stationnaire
par blocs de Politis-Romano, jamais iid.** Ajouter un sixième test après coup
ferait passer le seuil à 2,631 et serait journalisé dans
`docs/PROTOCOL_FREEZE.md` comme amendement, jamais édité en place.

---

## Effort et probabilités, récapitulés

| niveau | effort | P(succès) | cumulé |
|---|---|---|---|
| socle de données et de construction | 3 jours | — | — |
| A | 3 jours | 30 % | 30 % |
| B (une branche) | 1,5 jour | 20 % | — |
| C | 1 jour | 12 % | — |
| **arbre entier** | **8,5 jours** | — | **≈ 30 %** |

Les trois niveaux partagent le même panel de manches et la même question ; leurs
échecs seraient corrélés. Je ne compose donc pas les probabilités
(1 − 0,70 × 0,80 × 0,88 = 51 % serait malhonnête) et je retiens **≈ 30 %** pour
l'arbre entier. Le taux de base du programme est six échecs sur six.

---

## h) Ce qu'on apprend même si tout l'arbre tombe

1. **La première constante de temps macro mesurée du programme.** 3,11
   transitions par an contre 0,51 pour A′ sparse jump, et surtout la persistance
   **0,220 contre 0,266 pour l'indépendance**. Cela tranche par un chiffre une
   question que le programme posait sans y répondre : un quadrant
   macroéconomique n'est **pas un régime**. C'est une suite de chocs quasi
   indépendants. Toute future tentative de le prédire est close d'avance.
2. **Le consensus réel est disponible et vérifié.** 232 enquêtes SPF, 1968-T4 à
   2026-T3, quatre fichiers ouverts. H3 a été obligée de modéliser la surprise et
   de la valider à r = +0,667 ; on sait maintenant qu'on n'était pas obligé. Cela
   nourrit directement **C2**, la rédaction de H3, qui doit dire ce qui était
   hors de portée et ce qui ne l'était pas.
3. **Une décomposition du risque par quadrant d'un livre diversifié à cinq
   manches sur 23,6 ans**, que le programme ne possède pas. Elle sert à toute
   étude d'allocation ultérieure, indépendamment du verdict.
4. **L'axe construction serait fermé**, le seul des quatre que les six
   réfutations n'avaient jamais touché. La phrase étroite du programme
   deviendrait légitimement plus large : non seulement le dimensionnement et le
   timing, mais aussi la construction.
5. **Le fait mesuré que ce n'est pas une question de coûts** : 0,0195 de Sharpe
   pour seize aller-retours supplémentaires par an. Cela évite d'attribuer à des
   frais un échec qui vient de la puissance.

---

## Ce qui est écrit d'avance, et ce qui est une lecture

- Je n'ai lu **aucun rendement**. Les scripts à côté de ce fichier ouvrent des
  parquets pour compter lignes, dates, cellules et transitions ; aucun ne calcule
  de performance. C'est vérifiable ligne à ligne.
- **Rien n'est écrit dans les dépôts.** Tout est dans le répertoire de brouillon.
- **Mes erreurs, nommées** : deux URL SPF renvoyant une page HTML « 404 » sous un
  code 200, et `openpyxl` absent du venv de regime-lab. Les deux corrigées avant
  usage, pas après.
