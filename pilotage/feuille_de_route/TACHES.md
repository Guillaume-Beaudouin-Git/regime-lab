# Ce qu'il reste à faire — mis à jour le 22 septembre 2026 au soir

> **Depuis le 22/09 au soir, la vue d'ensemble est `AVANCEMENT.md`** à la racine du dépôt :
> ce qui est fait, l'inventaire des données, ce qui reste et le plan des 7 heures du 23/09.
> Ce fichier-ci garde le détail et l'historique. Les chemins qu'il cite sont ceux du dépôt
> unique : `chantiers/`, `pilotage/`.

Destiné à l'autre session Claude Code. Tout ce qui est décrit ici est sur disque et
commité, sauf mention contraire. Les trois documents à lire avant de toucher quoi
que ce soit :

1. `pilotage/REPRISE_PROJET.md` — **section 8bis d'abord**, la
   section 9 est périmée.
2. `pilotage/feuille_de_route/CONCLUSIONS.md` et `PISTES.md` — ce qui est établi, ce qui est
   mort, les pistes classées par espérance.
3. `docs/PROTOCOL_FREEZE.md` — le registre
   d'amendements. Toute déviation y passe, le cadrage n'est jamais édité en place.

---

## A. L'étude ouverte — la seule piste de recherche active

`docs/PRESPEC_TREND_VEHICLE.md`, verrouillé le 21/09 avant toute mesure. Le livre de
tendance à 46 instruments devient le sujet. Trois modifications, pas une de plus.
Effort estimé 5-7 jours. P(Sharpe net en excess > 0,70) ≈ 35-45 %.

### A1. M1 — réalignement calendaire du change · **FAIT le 21/09**

Audit `scripts/audit_fx_alignment.py`, application `scripts/apply_m1_realignment.py`,
bibliothèque `regime_lab/extensions/repair.py`, tests `tests/test_repair.py`
(commit `ead0794`).

**Vérifié contre les instruments qui ont diagnostiqué le défaut**, pas affirmé :

| | stockée | réparée |
|---|---|---|
| EURUSD=X contre 6E | 0,0574 | **0,9040** |
| GBPUSD=X contre 6B | 0,1068 | **0,9120** |
| AUDUSD=X contre 6A | 0,0636 | **0,8955** |

Point d'ancrage daté : GBP au 24/06/2016 passe de **−1,56 %** à **−7,60 %**. Témoins
inchangés au chiffre près (^GSPC 0,9717, CL=F 0,9671, IEF 0,9256). Huit colonnes sur
46 changent ; CAD et MXN ne sont pas touchées. Coût : huit observations, la dernière
séance de chaque colonne décalée.

**Deux panneaux écrits** dans `data/cache/` (gitignoré, régénérable ; le
`trend_universe.parquet` stocké n'est jamais modifié) :
- `trend_universe_m1.parquet` — le titre, huit séries décalées
- `trend_universe_m1_verified.parquet` — la sensibilité obligatoire, trois seulement

⚠ Le décalage donne au livre une information **une séance plus fraîche**, ce qui est
le sens qui flatte un backtest. C'est pour ça que la sensibilité à trois n'est pas
optionnelle.

### A2. M2 — spécification du véhicule · **TRANCHÉ 21/09, journalisé**

Audit écrit, exécuté et commité : `scripts/audit_vehicle_coverage.py` (commit `d653f6f`).

La tension N ≥ 30 décrite ici auparavant n'était pas le vrai obstacle. Le vrai est
temporel : **les dix futures substituables n'ont pas la même profondeur**. Neuf
commencent au 01/06/2012, RTY au 10/07/2017, tous s'arrêtent au 12/06/2026.
L'intersection fait **8,9 ans** contre 23,2 pour l'échantillon verrouillé.

| échantillon | années | plus petit Sharpe résoluble |
|---|---|---|
| séries stockées, verrouillé | 23,2 | **0,639** — la porte à 0,70 est décidable |
| substitué, intersection | 8,9 | **1,253** — elle ne l'est plus |

**Décision : M2 se rétrécit comme M1.** La substitution de série de prix devient une
sensibilité sur la fenêtre où elle est possible et ne peut jamais être le titre. Le
titre garde les séries stockées pour les 46, les signale toutes les 46, et applique
le véhicule comme **barème de coût** — ce que D2 mesurait, et ce qui faisait basculer
le verdict de +0,244 t 2,91 à +0,106 t 1,23.

Trois raisons, toutes mesurées : substituer instrument par instrument sur sa propre
fenêtre reviendrait à **rabouter** une série stockée à une série future en milieu
d'échantillon, ce qui injecte une discontinuité de jointure — la classe de défaut que
M2 existe pour retirer ; la variante « sans signalés » du texte gelé ne compterait que
10 instruments, sous N ≥ 30 et loin sous la largeur effective de 11 à 21 déjà
déclarée ; et c'est exactement le piège de troncature où le programme est déjà tombé
deux fois.

**Deux corrections au passage.** D3(b) nommait trois matières premières brutes : il y
en a **deux**. GC=F et SI=F corrèlent à 1,0000 avec le front-month brut, coûtent 1,02
et −0,18 point de dérive annualisée et retournent le signe du signal 12-1 sur 7,5 %
et 5,6 % des séances. **CL=F n'est pas brut** (0,9671 contre le back-ajusté, 0,9252
contre le brut). Et la portée de M1 est désormais bornée par la mesure : indices
actions 0,9717 à 0,9766 en même jour, matières premières 0,9671 à 0,9901, IEF 0,9256
contre ZN — **le décalage est une propriété du point d'accès `=X` et de rien d'autre**.

### A3. M3 — ciblage de volatilité au niveau du portefeuille · **FAIT le 21/09**

`regime_lab/extensions/vehicle.py`, `scripts/run_m3_construction.py`,
`tests/test_vehicle.py` (commit `28644b9`). Intégrité de construction seulement —
aucun Sharpe, aucun critère d'arrêt décidé.

**La cible tient** : 10,84 % à 10,90 % de volatilité réalisée contre 10 % visé, sur
les trois panneaux. Multiplicateur décalé d'une séance derrière son propre
estimateur, et un test de causalité qui perturbe le futur et vérifie que le passé
ne bouge pas.

**K6 dépendait entièrement d'une lecture, et les deux ne diffèrent pas un peu :**

| lecture de « plafond de levier 3,0 » | mord sur |
|---|---|
| (a) sur le **multiplicateur** — la source que §5 nomme | **0,00 %** |
| (b) sur l'**exposition brute** | **52,24 %** |

Le brut médian tombe à **3,080**, pile sur le seuil. Tranché en faveur de (a) :
`MAX_LEVERAGE` dans `trend.py` plafonne un multiplicateur par instrument, jamais un
brut ; et K6 disqualifie (b) deux fois — un plafond qui mord sur 52 % est un défaut
de conception par sa propre définition, et plafonner le brut à 3,0 ramène la
volatilité de 10,90 % à **9,14 %**, donc M3 cesse de tenir la cible qu'il existe
pour tenir.

⚠ **Déclaré contre soi-même** : sous (a) le plafond est **inerte**. Le brut court à
3,08 de médiane, 5,30 au p95 et **43,8 au maximum** le 07/07/2004, où un estimateur
de 63 séances s'est effondré. Huit séances dépassent 10×, une dépasse 40×. « Le
plafond ne mord jamais » est l'absence de contrainte, pas une garantie.

**Deux autres lectures journalisées.** §6 « non-futures vehicle » est lu comme une
propriété du marché et non du cache : huit des 46 paient 7,5 bp (NOK, SEK, AGG, BWX,
EMB, HYG, LQD, TIP). La lecture littérale, que le rétrécissement de M2 aurait poussée
à 46 sur 46, est conservée comme barème `ALL_CASH`. Et la rotation passe de 15,36× à
**58,92×/an** là où §6 calibrait 24,71× — signalé, pas absorbé, §6 disant que c'est
attendu et non un motif d'altérer la construction.

### A4. L'évaluation contre les six critères · **FAIT le 22/09 — KILL**

`scripts/run_m3_evaluation.py`, `docs/RESULTS_TREND_VEHICLE.md` (commit `588202a`).
Quatre contre-expertises indépendantes, synthèse arbitrée dans
`SYNTHESE_A4_contre_expertises.md` à côté de ce fichier.

| | verdict | le chiffre |
|---|---|---|
| **K1** niveau | **KILL** | Sharpe net en excess **+0,359** contre une ligne à 0,50 |
| K2 walk-forward | PASS, à sa frontière | 4 plis positifs sur 5 |
| **K3** dimensionnement | **UNDERPOWERED**, H-a non établie | écart **+0,068** contre un MDE de 0,246 |
| **K4** régime | **UNDERPOWERED**, H-b non établie | écart **+0,014**, résolution propre 0,043 |
| K5 Sharpe déflaté | **PASS non établi** | l'excès court de −0,159 à +0,286 |
| K6 plafond | PASS sous la lecture déclarée | mord sur 0,00 % |

**RESEARCH_PASS non franchie** : rate le niveau (0,359 contre 0,70) et la perte
maximale (−29,6 % contre −25 %).

**LA LIGNE RÉGIME SE FERME — TELLE QUE CE PROGRAMME L'A CONSTRUITE.** Le qualificatif porte : voir `pilotage/plans_de_recherche/ARBITRAGE.md` §6. H-b était le sixième et dernier test, son a priori était
bas et écrit avant la mesure, et il échoue sur un écart valant 0,3× la résolution de sa
propre paire — un test assez fin pour dire que l'effet est plus petit que tout ce qu'il
peut voir.

**Trois chiffres publiés n'ont pas survécu**, dont un défaut qui était le mien : les
ETF sont téléchargés en `auto_adjust=True` donc en rendement total, et trois d'entre eux
n'étaient pas financés au taux cash alors que six identiques l'étaient. Coût : 0,041 de
Sharpe. Les deux autres sont l'affirmation « les deux placebos échouent à distinguer quoi
que ce soit » (fausse : P1 lit 99,8 % une fois le rodage purgé) et le PASS de K5.

**Ce que l'étude livre quand même** : le dénominateur qui manquait aux deux dépôts depuis
le début — un livre de référence audité, en excess, correctement facturé, sur 23,2 ans.

---

## B. Hygiène — **12 sur 12 FAITES** le 22/09, commits `9e507f4` et `c6fd`

| | quoi | état |
|---|---|---|
| **B1** | `REPLICATION_SHU2024.md` publie deux sections sans code | ✅ **ligne retirée**. Le bras manquant a été **écrit** (`select_penalty`) plutôt que de rétracter à l'aveugle — il donne 7,8 % / 0,54 / **762 %** contre 8,6 % / 0,50 / **112 %** publiés. Il ne reproduit pas. Le critère est **sous-déterminé** par le document (« par validation croisée » ne dit pas *sur quoi*), et chercher le critère qui rend 112 % serait ajuster la procédure à une réponse mémorisée. Écart publié à côté de la rétractation |
| **B2** | `equal_risk_momentum` non câblable | ✅ déclarée, non câblée, non supprimée. La raison est dans sa docstring : `pct_change()` sur des taux en pourcent vaut **+0,176 de Sharpe** d'erreur, et N=6 contre 30 |
| **B3** | `sh.xls`, 67,2 % du dépôt, aucun lecteur | ✅ dé-traqué, `*.xls` au gitignore, historique intact. **L'URL Shiller a été testée : HTTP 200** — la copie locale n'est donc pas la seule, ce que le rapport laissait ouvert |
| **B4** | `exposures.py` prétend qu'un test vérifie la carte | ✅ 8 tests écrits, qui **lisent le document gelé** au lieu d'en recopier les valeurs |
| **B5** | `xlrd` hors du graphe déclaré | ✅ déclaré |
| **B6** | `fetch_trend_universe` prérequis non documenté | ✅ documenté, et `run_extensions` échoue maintenant sur une instruction et non un chemin manquant |
| **B7** | `run_decomposition.py` jamais publié | ✅ **publié** — `docs/RESULTS_DECOMPOSITION.md`. Résultat dormant et fort : le ciblage de volatilité vaut **+0,44 %**, l'apport propre du régime **−0,04 % avec recul** et **−0,25 %** net de latence |
| **B8** | `regime-lab` non déclaré chez les dépendants | ✅ dépendance de chemin + `[tool.uv.sources]` dans les deux |
| **B9** | README affirme « exactly as it stood » | ✅ corrigé |
| **B10** | 3 erreurs de lint préexistantes | ✅ **ruff : All checks passed**, pour la première fois |
| **B11** | `xs_ff_mom_63` jamais construite, silencieusement | ✅ rendue visible. `factors_5.parquet` n'a pas de momentum — enregistré et **non réparé** : l'étude est gelée, aucun chiffre publié ne contenait la colonne |
| **B12** | départ N ≥ 30 ambigu | ✅ tranché le 23/09 : 2003-07-17, 6 039 séances |

**E1 est clos aussi** : `macro-momentum` et `reversal-lab` ont chacun un dépôt GitHub
**privé** et sont poussés. Les quatre falsifications ne vivent plus uniquement sur
cette machine.

---

## C. Rédaction et présentation

> **Précisé le 23/09 : ce n'est pas un mémoire.** Le projet se conclut par une
> présentation d'une dizaine de minutes au format entreprise. C1 devient la
> construction de cette présentation ; C2 et C3 ne servent plus que de diapositives de
> réserve. Voir `AVANCEMENT.md` §0 et §4.

**C1. La présentation de 10 minutes.** Le matériau est complet. Ce qui manque n'est pas
du résultat, c'est de la mise en évidence — voir `PISTES.md`, piste 3, P ≈ 90 %.

**C2. H3 n'a pas son écrit.** H1 et H2 ont `RESULT_DISPERSION.md` et
`NOTE_DISPERSION_FR.md`. H3 — les surprises macro, la construction de Dedale — n'a
que `RESULTS_H3.md`. Le programme compte quatre falsifications ; trois seulement
sont rédigées pour un lecteur extérieur.

**C3. Le résultat de dispersion est écrit mais pas intégré à la présentation.**

---

## D. Pistes non instruites

**D1. Marchés émergents.** La seule réouverture que la politique du projet autorise,
parce que le kill de H2 est géographiquement borné. **Mais les chiffres qui la
motivent ne sont vérifiés nulle part** : le rapport de dispersion EM/G10 de 0,7× à
1,5×, les 12 devises depuis 2006, les 12 indices depuis 2012, les 6 ETF obligataires
depuis 2013 — aucun code ni donnée émergente n'existe dans `macro-momentum`. **La
vérification passe avant le pré-enregistrement, pas après.** Tueurs connus :
échantillon court, coûts 5-10× le G10, contrôles de capitaux invisibles au comptant,
et un ETF n'est pas un future.

**D2. La géométrie FTMO à plancher statique.** Le seul chiffre que l'étude de
barrière laisse ouvert : le régime apparié allonge la survie médiane de **+36
sessions**. Mais la cible de vol sans paramètre en allonge **60**, et cette géométrie
passe par un adaptateur local et non par le simulateur audité. Ce n'est pas un
résultat ; ça demanderait un vrai simulateur FTMO, qui n'existe pas.

---

## E. Risques et choses à vérifier

**E1. Sauvegarde hors machine · FAIT le 21/09 pour la partie urgente.** Deux bundles
git, historique complet, vérifiés par restauration intégrale (historiques identiques,
6 et 1 commits, tous les documents présents) :
`~/Library/Mobile Documents/com~apple~CloudDocs/Sauvegardes_Recherche_Quant/`.
65 Ko et 8,8 Ko — les 464 Mo des dossiers sont du venv et de la donnée régénérable.

**Reste à faire, et c'est le vrai remède** : deux dépôts GitHub **privés**. `gh` est
authentifié sur le compte. Un bundle est un instantané ; un remote est vivant. Non
fait parce que publier deux dépôts jusqu'ici locaux est une action vers l'extérieur
qui demande l'accord de Guillaume.

**E2. Le déplacement du dossier casse les trois venvs.** C'est arrivé deux fois. Les
`.pth` des installations éditables codent le chemin absolu. Remède dans
`LISEZ_MOI.md` : les recréer, pas réparer les chemins.

**E3. Le simulateur propfirm est hors du dossier** et son chemin vit dans `.env`
(`SIMULATOR_PATH`), qui est gitignoré. Un clone frais ne l'a pas ; seul
`run_barrier.py` en dépend et il sort avec une explication.

**E4. Le dépôt public a été anonymisé le 21/09** mais l'historique antérieur contient
encore des références au dépôt privé voisin. Décision prise : pas de réécriture
d'historique, ce sont des chemins et non des identifiants.

---

## F. Ce qui est mort — ne pas y revenir

Le régime comme timing · le régime comme conditionneur de livre (T1, T3) ·
l'atténuateur de Carver · le régime dans la barrière propfirm · le compteur de
facteurs effectifs (le signal apparent était la troncature) · la sélection d'actifs
par Hurst · le conditionnement au choc de persistance · la prime de retournement
(kill mécanique) · le transversal macro G10 après mars 1999 (kill mécanique) · H1 ·
H3 · **Δp(pass) comme critère d'admission** — tué des deux côtés indépendamment,
p(pass) s'achète avec de l'exposition et corrèle à +0,51 avec l'autocorrélation du
candidat contre −0,06 avec le canal attendu.

Et le chiffre qui explique tout le reste : **A′ sparse jump change d'état 13 fois en
6 377 séances.** Vingt-cinq ans donnent treize décisions. Ce n'est pas un signal de
trading, c'est une variable de budget de risque.

---

## G. Ordre recommandé — au 22 septembre 2026

**La consigne de démarrage est `CLAUDE.md` à la racine du projet**, chargé
automatiquement par toute session ouverte ici. Ce fichier-ci en est le détail.

1. ~~E1 — copie hors machine~~ **FAIT.** Les deux chantiers ont un dépôt GitHub privé.
2. ~~A2 — la tension N ≥ 30 de M2~~ **FAIT**, `d653f6f`. Ce n'était pas la largeur,
   c'était la profondeur temporelle : l'intersection des futures fait 8,9 ans.
3. ~~A1 — le décalage du change~~ **FAIT**, `ead0794`. Vérifié : 0,057 → 0,904.
4. ~~A3 — la cible de volatilité~~ **FAIT**, `28644b9`.
5. ~~A4 — les six critères~~ **FAIT**, `588202a`. **KILL**, quatre contre-expertises.
6. ~~B1 à B12 — l'hygiène~~ **FAITS**, `9e507f4` et `53cd23f`. Douze sur douze.
7. ~~AHL jour 1 et niveau A~~ **FAITS**, `630ee16` et `f5b70c1`. **A1 réfuté sur le
   signe**, et pas par manque de puissance.

8. ~~AHL niveau B~~ **FAIT le 22/09 au soir**, `9602526`, `6812bca`, `4dd0eb6`. **B1
   indécidable**, non lu, aucun essai dépensé : le MDE vaut 49,6 % de l'erreur moyenne
   contre une limite de 25 % écrite avant le chiffre (52,2 % en rendements log).
9. ~~Un seul dépôt~~ **FAIT le 22/09 au soir** : fusion avec historique, venv unique,
   `pilotage/` expurgé et versionné, 120 fichiers de mesure sauvés de `/private/tmp`.

10. ~~L'objectif final~~ **FIXÉ le 23/09** : une présentation de 10 minutes ;
    l'objectif de fond est de construire des stratégies dépendantes du régime, avec
    l'ambition d'un Sharpe net de 1 à 2 à coûts institutionnels (`AVANCEMENT.md` §0).
11. **Plan Two Sigma, phase 1 FAITE le 23/09** (`22abbe0`, `3b64ab2`) par une autre
    session : l'outillage est construit, les chiffres du brouillon sont remesurés, le
    verrou est rédigé mais pas commité. Aucun niveau n'a été lu.

### ⟶ 12. ENSUITE — voir `AVANCEMENT.md` §4 et §5

Dans l'ordre proposé : la présentation de 10 minutes ; la relecture et le commit du
verrou Two Sigma, puis ses niveaux A, B et C ; la fermeture de l'arbre AHL (le MDE de C1,
et la décision sur la lecture A1 principale sur (126,10)) ; le nettoyage et le commit de
`pilotage/mesures_brutes/` ; puis le plan Bridgewater. La porte G0 de Rentec reste la
moins chère (1,5 j), mais ses données sont dans le dépôt privé voisin : Guillaume seul.
