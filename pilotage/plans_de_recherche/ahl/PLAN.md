# Man AHL — points de rupture, tendance contre consolidation, rupture de corrélation

Plan de recherche pré-enregistrable. Rédigé avant tout contact avec un rendement.
Aucune stratégie n'a été construite pour écrire ce document et aucun Sharpe n'a été
calculé sur l'angle proposé. Les seules mesures faites ici portent sur des prix, des
variables d'état latentes, leurs comptes de transition et leur corrélation avec la
volatilité — c'est exactement ce que la contrainte autorise.

Toutes les commandes sont dans `measure_timeconstant.py` et `measure_two.py`, à côté
de ce fichier, exécutées avec `regime-lab/.venv/bin/python`.

---

## a) TEST D'ADMISSION

Le programme a réfuté six dispositifs qui partagent quatre étroitesses. La question
est de savoir si celui-ci en diffère. Réponse mesurée : **il en diffère sur les quatre**.

### Axe 1 — la variable latente n'est plus la volatilité, et c'est chiffré

Les états du programme sont **ordonnés par la volatilité d'entraînement**
(`regime_lab/models/mapping.py`, docstring de `state_to_position`). La variable
latente d'AHL — l'efficacité directionnelle du chemin de prix, « tendance contre
consolidation » — est un autre objet. Reste à le prouver plutôt qu'à l'affirmer.

Mesure : ratio d'efficacité de Kaufman sur 63 séances,
`ER = |log P_t − log P_{t−63}| / Σ|Δ log P|`, par instrument, puis médiane
transversale des 46. Contre la volatilité réalisée 63 jours (médiane transversale
elle aussi), sur les 6 040 séances de l'échantillon N ≥ 30 :

| latente candidate | corr. Pearson avec VOL63 | Spearman | n |
|---|---|---|---|
| **ER63, médiane du livre (tendance/consolidation)** | **+0,096** | **−0,044** | 6 040 |
| ‖C₆₃ − C₂₅₂‖_F, distance entre matrices de corrélation | **+0,070** | **+0,118** | 5 725 |
| ‖C₆₃ − C₂₅₂‖_F résidualisée sur la vol (MCO expansif causal) | +0,009 | +0,075 | 4 780 |
| corrélation moyenne par paires, en **niveau** | +0,580 | +0,559 | 5 914 |
| part de la première valeur propre (parente de l'absorption) | +0,594 | +0,633 | 5 914 |

**Axe 1 franchi deux fois, et une troisième fois refusée.** L'efficacité
directionnelle est pratiquement orthogonale à la volatilité, la *rupture* de
corrélation aussi. Le *niveau* de corrélation ne l'est pas : à 0,58-0,59, c'est un
proxy bruité de la chose dont le programme sait déjà qu'elle ne porte que la variance.

**Et c'est la raison mesurée de l'échec de `concentration.py`.** Le document
`docs/EXTENSIONS.md` rapporte tous les |t| sous 1,6 pour le nombre effectif de
facteurs et le ratio d'absorption, sans expliquer pourquoi. L'explication est dans la
quatrième ligne du tableau : la part de la première valeur propre corrèle à +0,594
avec la volatilité réalisée. Ce n'était pas une mesure de structure, c'était une
mesure de volatilité en moins bon. Ce plan ne reprend donc **pas** un objet de niveau ;
il prend une distance entre matrices, et il la résidualise sur la volatilité avant de
la regarder.

### Axe 2 — la constante de temps, mesurée sur nos données

Référence recalculée par mes soins sur `data/cache/states.parquet`, 6 377 séances
communes, 24,4 ans :

| modèle du programme | transitions | par an |
|---|---|---|
| A jump | 13 | **0,53** |
| A′ sparse jump | 13 | **0,53** |
| B HMM filtré | 49 | 2,01 |
| C gradient boost | 344 | 14,08 |
| C′ HAR-RV | 450 | 18,41 |

Ce que produit un détecteur de points de rupture sur le même univers, échantillon
N ≥ 30 :

| latente | détecteur | ruptures/an |
|---|---|---|
| ER63 | BOCPD (Adams & MacKay 2007), hasard 1/250 | **11,04** |
| ER63 | BOCPD, hasard 1/60 | 11,94 |
| ER63 | CUSUM de Page, k=0,5 h=5 | 13,92 |
| ER63 | CUSUM, k=1,0 h=5 | 4,73 |
| ‖C₆₃ − C₂₅₂‖_F résiduelle | CUSUM, k=1,0 h=5 | 7,26 |
| ‖C₆₃ − C₂₅₂‖_F résiduelle | CUSUM, k=0,5 h=5 | 15,66 |
| corr. moyenne, niveau | BOCPD, hasard 1/250 | 6,92 |

**Axe 2 franchi : 5 à 16 ruptures par an contre 0,53.** Le chiffre de tête retenu est
**11,04/an**, soit **21×** la constante de temps de A′. Vingt-cinq ans donnent ici
environ 256 épisodes, pas treize décisions.

Un détail qui n'est pas un détail : le même ER63 découpé par un **seuil** (médiane
expansive, la forme du témoin gagnant du programme) donne **28,58 transitions/an**,
soit une durée moyenne d'état de neuf séances. C'est un objet intenable en coût. Le
détecteur bayésien n'est donc pas seulement « différent » du seuil : c'est lui qui
fait atterrir la constante de temps dans la bande utilisable, parce qu'il exige que
la preuve s'accumule avant de déclarer la rupture. La méthode fait partie de
l'hypothèse.

### Axe 3 — l'usage est la SÉLECTION, et elle est neutre en exposition

Les six réfutations testaient le **dimensionnement** (T1, T3, atténuateur de Carver,
H-b) ou l'**interrupteur** (barrière, Shu). Ici l'état ne touche jamais la taille. Il
décide **quel signal tourne sur chaque instrument** :
le `0,10/σ_63` par instrument et le multiplicateur `0,10/σ_livre` sont inchangés, et
seul le motif de signes change.

Conséquence structurelle, et c'est la défense contre le tueur identifié dans la
commande : un dispositif qui ne modifie pas l'exposition brute **ne peut pas** gagner
par le canal qui a tué T1, T3 et l'atténuateur. Ce n'est pas une promesse, c'est une
assertion vérifiable à la machine — voir le point e).

### Axe 4 — l'objet conditionné est une bibliothèque, pas un livre

Les six réfutations conditionnaient **un** livre : TSMOM 12-1 sur 46 instruments.
Ici l'objet est une bibliothèque de 3 vitesses × 46 instruments = 138 jambes
signal-instrument, plus une jambe de retour à la moyenne au niveau C. Le test de
premier rang est un test de **panneau** sur ces 138 jambes, pas une différence de
Sharpe de portefeuille.

### Verdict d'admission

**ADMISSIBLE sur les quatre axes.** Les deux qui comptent le plus sont l'axe 3
(l'usage, parce que c'est lui qui esquive la famille tuée) et l'axe 2 (la constante
de temps, parce que c'est elle qui rend l'axe 3 réalisable).

---

## b) LA DONNÉE — ce qui est sur disque, ouvert et compté

Rien ne doit être téléchargé. C'est une décision, pas une commodité : le programme
s'est fait prendre deux fois (Stooq derrière un mur anti-bot, les codes pays OCDE).
La seule recommandation de ce plan à propos de données externes est de **n'en chercher
aucune**.

Racine : la racine de ce dépôt (`regime-lab/`)

| fichier | forme | dates | fréquence |
|---|---|---|---|
| `data/cache/trend_universe.parquet` | 6 822 × 46 | 2000-07-17 → 2026-09-10 | quotidienne |
| `data/cache/trend_universe_m1.parquet` | 6 822 × 46 | idem, 8 colonnes FX recalées | quotidienne |
| `data/cache/trend_universe_m1_verified.parquet` | 6 822 × 46 | idem, 3 colonnes recalées | quotidienne |
| `data/cache/states.parquet` | 9 007 × 5 | 1992-03-02 → 2026-09-08 | quotidienne |
| `data/raw/macro/rate_cash_3m.parquet` | 9 177 lignes | 1990-01-02 → 2026-09-08 | quotidienne, `period` + `available_at` |
| `data/trials.parquet` | 805 lignes, 83 hachages distincts | — | — |

Complétude des colonnes du panneau : de 4 885 séances (EMB) à 6 822 (16 colonnes dont
`^GSPC`, `^FTSE`, `CT=F`, `KC=F`, `SB=F`, `DX-Y.NYB`).

**Échantillon actif.** Première séance où 30 instruments ont 252 séances d'historique :
**2003-07-16**, soit **6 040 séances, 23,15 ans**, identique sur les trois panneaux.
⚠ Le pré-enregistrement verrouillé du 21/09 (`docs/PRESPEC_TREND_VEHICLE.md` §3) écrit
**2003-07-17** et `RESULTS_TREND_VEHICLE.md` parle de 6 039 séances. Une séance d'écart.
Je ne sais pas laquelle des deux règles est la bonne et je ne la tranche pas ici : elle
doit être tranchée **avant** le verrouillage, pas après, et journalisée dans
`PROTOCOL_FREEZE.md`. C'est le genre d'écart d'une unité qui, laissé sans nom, devient
plus tard une explication commode.

**Bloc transversal complet** à partir de 2003-07-16 : **28 instruments** sans aucune
séance manquante (`CL=F, CT=F, DX-Y.NYB, GC=F, HG=F, HO=F, IEF, KC=F, LQD, NG=F, SB=F,
SHY, SI=F, TLT, ZC=F, ZS=F, ZW=F, ^AEX, ^AXJO, ^FCHI, ^FTSE, ^GDAXI, ^GSPC, ^GSPTSE,
^HSI, ^IBEX, ^NDX, ^RUT`). C'est le bloc sur lequel la matrice de corrélation est
estimée. **28 < 30**, et il faut le dire : la règle N ≥ 30 borne les *affirmations
transversales*, et la latente de corrélation n'en est pas une — c'est une entrée d'état
de marché. Le livre conditionné, lui, garde ses 46. La distinction est écrite ici pour
qu'un lecteur la vérifie plutôt que de la découvrir.

**Ce qui manque et qui n'est pas récupérable utilement.** Des futures back-ajustés pour
les matières premières brutes (GC=F et SI=F corrèlent 1,0000 avec le front-month brut).
La question a déjà été instruite et fermée par un audit exécuté,
`scripts/audit_vehicle_coverage.py` du 21/09 : l'intersection des dix substituables
fait 8,9 ans, ce qui porte le plus petit Sharpe résoluble de 0,639 à 1,253. Le
substitut est donc indécidable. J'hérite de cette décision au lieu de la refaire.

---

## c) LA PUISSANCE A PRIORI — et le seul endroit où ce plan a un avantage réel

### Au niveau portefeuille : l'échantillon ne peut pas voir ce que l'on cherche

`regime_lab/analysis/power.py`, bootstrap stationnaire par blocs, bloc moyen 126,
600 tirages, sur des jambes synthétiques appariées de volatilité 10 % :

| longueur | corr. entre jambes 0,98 | 0,95 | 0,90 | 0,80 |
|---|---|---|---|---|
| 23,2 ans | MDE 0,108 | 0,148 | 0,307 | 0,384 |
| 20,0 ans | 0,122 | 0,206 | 0,276 | 0,378 |
| 11,6 ans | 0,143 | 0,302 | 0,289 | 0,409 |

Ancrage sur données réelles, déjà mesuré par le programme pour exactement cette forme
de comparaison (deux constructions du même livre, 6 569 séances) : **MDE 0,246 à 0,273**
selon le bloc. Le tableau synthétique situe cet ancrage autour d'une corrélation entre
jambes de 0,92. Un livre commuté diffère du livre de base sur **35,9 %** des
instrument-séances (vitesse 63-5) ou **25,6 %** (vitesse 126-10) : sa corrélation avec
la base sera de l'ordre de 0,88-0,93, donc **le MDE de portefeuille sera autour de
0,28-0,31, c'est-à-dire pire que l'ancrage de 0,246.**

Et il faut encore corriger pour l'arbre entier — voir le point sur les tests multiples
plus bas : **le MDE de décision au niveau portefeuille est 0,331.**

**Dit avant la mesure, comme un résultat à coût nul : la question de portefeuille est
sous-puissante pour l'effet visé.** Un commutateur de vitesse conditionnel plausible
vaut 0,10 à 0,30 de Sharpe dans la littérature CTA. La borne de décision est 0,331.
L'échantillon ne peut pas trancher. Prétendre le contraire serait refaire l'atténuateur
de Carver, dont le +0,144 est mort sous un MDE de 0,205, et H-b, dont le +0,014 est mort
sous une résolution de 0,043.

### Au niveau panneau : c'est là que la puissance existe

Les six réfutations ont toutes réduit la question à **une** différence de Sharpe de
portefeuille, ce qui jette la transversale. Le programme a pourtant déjà démontré, sur
lui-même, qu'un test de panneau voit ce qu'un test de portefeuille ne voit pas : le
classifieur porte +3,93 points de R² incrémental sur la volatilité future avec t −3,40,
et cette statistique-là n'est pas sous-puissante.

Statistique de premier rang proposée : l'écart de justesse directionnelle relative,
`Δ = [hit(rapide|consolidation) − hit(lent|consolidation)] − [hit(rapide|tendance) − hit(lent|tendance)]`,
pondéré par l'exposition ajustée au risque, sur 46 instruments × 6 040 séances.

Le MDE dépend entièrement de la dépendance conservée dans le rééchantillonnage, et il
faut donner l'encadrement honnête plutôt qu'un chiffre :

- en regroupant par **date** (le produit signal × rendement est presque blanc, alors
  que le signal est très autocorrélé), n_eff ≈ 6 040 dates × 11,4 à 20,7 paris
  effectifs = 68 900 à 125 000 → **MDE ≈ 0,8 point de pourcentage de taux de réussite** ;
- en regroupant par **épisode de 63 séances** (le bloc moyen que le programme utilise
  partout), n_eff ≈ 96 épisodes × 11,4 à 20,7 = 1 090 à 1 990 → **MDE ≈ 4,5 à 6,0 points**.

Le vrai est entre les deux et je refuse de le deviner. **Le pré-enregistrement fixe
donc la procédure et non le nombre** : le MDE de décision sera mesuré par bootstrap
stationnaire sur la dimension date, en blocs de 63, la transversale entière conservée
dans chaque tirage, **sous l'hypothèse nulle et avant qu'aucun coefficient d'interaction
ne soit lu**. Attente déclarée : 2 à 4 points. Les trois longueurs de bloc 21/63/126
sont rapportées, comme partout ailleurs dans le programme.

C'est le seul endroit où ce plan a un avantage structurel sur les six qui l'ont précédé,
et il vaut mieux le dire ainsi que de prétendre que l'angle est plus malin.

---

## d) LES COÛTS, et à partir de quel niveau l'hypothèse est morte — écrit avant

Barème hérité de `docs/PRESPEC_TREND_VEHICLE.md` §6, implémenté dans
`regime_lab/extensions/vehicle.py`. Répartition des 46 instruments, comptée :
13 matières premières (1,5 bp), 13 indices actions (1,0), 9 devises (1,0),
8 véhicules non-futures (7,5), 3 taux (1,0).

- **Taux mélangé à poids égaux : 2,272 bp aller-retour** (colonne `headline`),
  4,543 bp en `conservative`.
- Pondéré par la rotation réelle : `RESULTS_TREND_VEHICLE.md` §5 mesure que *huit
  instruments sur 46 portent 69 % du coût*. Cela implique une part de rotation de 27 %
  sur les huit à 7,5 bp, donc **≈ 2,9 bp** de taux mélangé effectif. C'est le chiffre
  de décision.
- Rotation de base du livre M3 : **58,92×/an** (mesurée, `RESULTS_TREND_VEHICLE.md`),
  soit 1,71 %/an de traînée, soit **0,171 de Sharpe** à 10 % de volatilité cible.

### Rotation incrémentale du commutateur — dérivée de fréquences de retournement mesurées

Retournements de signe par instrument et par an, mesurés sur l'échantillon actif :

| signal | médiane | moyenne | total sur 46 |
|---|---|---|---|
| sign(252, 21) — le lent, inchangé | 6,88 | 6,75 | 309,2/an |
| sign(126, 10) | 10,05 | 10,15 | 467,6/an |
| sign(63, 5) | 15,12 | 14,92 | 692,1/an |
| sign(42, 1) | 17,94 | 17,72 | 822,8/an |
| sign(21, 1) | 24,83 | 24,78 | 1 151,4/an |

Accord de signe avec le lent : 74,4 % pour (126,10), 64,1 % pour (63,5), 52,8 % pour (21,1).

Avec 34 % du temps en état de consolidation, une exposition brute médiane de 3,08
(mesurée en A3) donc un poids moyen par instrument de 0,067, et 11,04 commutations/an :

| jambe rapide | flips supplémentaires | rotation de commutation | **rotation incrémentale** | traînée à 2,9 bp | en Sharpe |
|---|---|---|---|---|---|
| (63, 5) | 0,34 × (692,1 − 309,2) = 130/an | 11,04 × 0,359 × 46 × 2 × 0,067 = 24,4 | **+41,8×/an (+71 %)** | 1,21 %/an | **0,121** |
| **(126, 10)** | 0,34 × (467,6 − 309,2) = 53,9/an | 11,04 × 0,256 × 46 × 2 × 0,067 = 17,4 | **+24,6×/an (+42 %)** | 0,71 %/an | **0,071** |

### Le seuil de mort, écrit avant de tester

L'hypothèse A est morte en coût quand la traînée incrémentale atteint le MDE corrigé
pour l'arbre, 0,331 de Sharpe, soit 3,31 %/an :

- jambe rapide (126,10) : 3,31 % / 24,6 = **13,5 bp aller-retour mélangé** ;
- jambe rapide (63,5) : 3,31 % / 41,8 = **7,9 bp**.

**Conséquences, déclarées maintenant.** Sous le barème `headline` (2,9 bp effectif)
aucune des deux ne meurt. Sous la lecture littérale `ALL_CASH` de §6 — 7,5 bp
`headline`, 15 bp `conservative`, celle que `vehicle.py` conserve exprès comme borne
la plus conservatrice — **la jambe (63,5) est morte et la jambe (126,10) meurt en
colonne `conservative`**. Un résultat dont le signe bouge entre `headline` et
`conservative` est rapporté comme indécis, jamais comme un pass : c'est la règle du
programme et elle s'applique ici sans aménagement.

**La jambe rapide de tête est donc (126, 10), et la raison est écrite avant toute
mesure de rendement : c'est une décision de coût prise sur des fréquences de
retournement, pas sur une performance.** (63,5) devient une sensibilité déclarée.
Cette divulgation est obligatoire — elle est du même type que celle du §1 du
pré-enregistrement précédent, où le choix du véhicule futures avait été pris après
avoir vu D2.

---

## e) LE PLACEBO APPARIÉ

Trois, dont deux obligatoires et un qui est ici une assertion de construction.

**P-E — neutralité d'exposition, vérifiée à la machine et bloquante.** Le dispositif ne
touche ni `0,10/σ_i` ni `0,10/σ_livre`. Assertion pré-enregistrée :
`|brut moyen(commuté) − brut moyen(base)| / brut moyen(base) < 0,02`, et aussi sur le
brut médian et le p95. **Si elle est violée, la mesure est abandonnée et le défaut est
journalisé** — ce n'est pas un résultat à discuter, c'est une construction fausse. La
barrière a mesuré que p(pass) s'achète à 0,2785 par unité d'exposition ; tant que
l'exposition ne bouge pas, ce canal est fermé par construction.

**P-B — bêta sur le livre de base, obligatoire.** T3 a montré qu'un alpha apparent
passait entièrement par le bêta (0ᵉ percentile en bêta, médian en moyenne). On rapporte
donc systématiquement la régression du livre commuté sur le livre de base, avec HAC
lag 6 : α, β, et l'alpha résiduel. **Un gain dont le β s'écarte de 1 de plus de 0,05
est traité comme un gain d'exposition déguisé**, quelle que soit la neutralité brute.

**P1 — rotation circulaire de l'état, apparié sur la durée des épisodes.** L'observation
est la **séquence d'épisodes**, pas la séance. Chaque tirage fait tourner circulairement
la série d'état contre les dates : la loi des durées d'épisode et le taux de 11,04/an
sont préservés exactement, seul l'alignement calendaire est détruit. 400 tirages.
Contrôle d'absence de biais hérité d'`EXTENSIONS.md` : la moyenne du placebo doit
atterrir sur la statistique inconditionnelle, comme elle l'avait fait à +0,428 contre
0,44 — et si elle n'y atterrit pas, c'est le placebo qui est faux, pas le signal qui
est bon.

**P2 — signaux recalculés sur prix mélangés par blocs.** Jamais une permutation de la
colonne P&L. Le placebo hérite du bruit d'estimation du signal lui-même, y compris du
détecteur de rupture, qui est réestimé sur les prix mélangés.

Et une correction héritée, parce qu'elle a coûté cher la dernière fois : le rodage est
purgé **avant** de calculer un percentile de placebo. P1 lisait « 88,5ᵉ percentile » et
lit 99,8 % une fois les 63 premières séances retirées ; l'écart entier était une
statistique du rodage.

---

## f) LES TUEURS CONNUS, sans les minimiser, et leur sens

1. **Le tueur principal, nommé dans la commande : « réduire l'exposition en
   consolidation » est la famille tuée par T1, T3 et Carver.** Sens : vers le faux
   positif, parce que couper le risque améliore un Sharpe sans rien apporter. Canal
   par lequel ce plan diffère : il **ne coupe rien**. L'exposition est neutre par
   construction et l'assertion P-E est bloquante. Si un jour une variante de ce plan
   doit réduire l'exposition pour fonctionner, elle est hors périmètre et il faut le
   déclarer au lieu de l'ajouter.
2. **Le témoin d'une ligne gagne à chaque fois.** La cible de volatilité dynamique sans
   paramètre bat tout ce que le programme a construit (+0,244 contre +0,144 pour
   l'atténuateur). Sens : contre le dispositif. Remède : **le témoin d'une ligne est
   dans le protocole dès le niveau A**, pas ajouté après. Un dispositif qui ne le bat
   pas est refusé même s'il bat son placebo.
3. **Le coût.** Sens : contre. Chiffré au point d), seuil de mort 13,5 bp, et mort
   certaine en colonne `conservative` pour la jambe (63,5).
4. **La prime de retournement est morte** — 3,64 de Sharpe dans les années 90, **−0,18
   depuis 2020**, kill mécanique (`reversal-lab/docs/RESULTS.md`). Sens : contre le
   niveau C, fortement. Mais le document borne lui-même sa portée : « la mesure porte
   sur les actions cotées aux États-Unis ; rien ici ne parle du retournement sur
   d'autres marchés ». Le niveau C porte sur 46 instruments transversaux, ce n'est pas
   le même objet. Conséquence pratique : **C reçoit une porte inconditionnelle qui se
   ferme en une demi-journée**, avant tout conditionnement.
5. **Le niveau de corrélation est un proxy de volatilité** (+0,58 / +0,59), et c'est
   l'objet qui a fait échouer `concentration.py`. Sens : vers un faux positif qui serait
   en réalité l'effet de variance déjà connu. Remède : seule la *distance* entre
   matrices est utilisée, et elle est résidualisée causalement sur la volatilité avant
   d'être regardée (+0,070 brut → +0,009 résiduel).
6. **La troncature.** Le programme est tombé deux fois dans ce piège, et une fois il a
   produit un t de 2,18 qui n'était que la troncature (`EXTENSIONS.md` §1). Sens : vers
   le faux positif. Remède : le bloc complet à 28 instruments sert **uniquement** à la
   latente de corrélation ; le livre garde ses 46 et laisse les instruments entrer à
   leur cotation.
7. **La durée des épisodes.** À 11,04 ruptures/an, un épisode dure environ 23 séances.
   Une jambe rapide (126,10) n'a peut-être pas le temps de payer sa rotation 1,46× plus
   élevée sur 23 séances. Sens : contre. Non résolu, et c'est la raison la plus probable
   pour laquelle le niveau A tombera.
8. **La latence du détecteur.** Un BOCPD déclare la rupture après accumulation de
   preuve. Le programme a mesuré 0 à 13 jours de latence sur ses propres états ; sur un
   épisode de 23 séances une latence de 5 à 10 jours mange la moitié de l'épisode.
   Sens : contre. À mesurer explicitement et à rapporter, pas à découvrir.
9. **Mon propre biais de conception.** J'ai choisi ER63 après avoir vu qu'il corrèle à
   +0,096 avec la volatilité, c'est-à-dire après une mesure. C'est déclaré comme une
   mesure préalable au sens du §1 du pré-enregistrement précédent : aucun rendement
   n'a été vu, mais le choix n'est pas aveugle et un lecteur doit pouvoir l'escompter.

---

## g) L'ARBRE D'ESCALADE, l'effort et les probabilités

Un arbre déclaré à l'avance n'est pas du p-hacking. Les trois niveaux, leurs critères
de falsification et la correction pour tests multiples sont fixés ici, avant la mesure.

### Niveau A — sélection de vitesse conditionnée à la rupture de tendance

- **Latente** : BOCPD, hasard 1/250, sur ER63 médiane du livre, standardisé sur une
  fenêtre expansive. État = « consolidation » si la moyenne postérieure de ER63 depuis
  la dernière rupture est sous sa médiane expansive.
- **Dispositif** : en tendance chaque instrument porte sign(252,21) ; en consolidation
  il porte sign(126,10). Échelles de risque inchangées. Signal en T−1.
- **A1, le test qui décide** — panneau, 46 × 6 040 : l'état déplace-t-il la justesse
  directionnelle relative du rapide sur le lent, dans le sens annoncé ?
  **Falsification A1** : Δ sous le MDE mesuré par bootstrap de blocs sous la nulle, ou
  de signe contraire ⇒ A1 mort.
- **A2, rapporté mais non décisif** — portefeuille : Sharpe net en excess du livre
  commuté moins le livre de base ≥ **0,331** (MDE corrigé pour l'arbre).
  **Falsification A2** : écart sous 0,331 ⇒ UNDERPOWERED, jamais un PASS.
- **Témoin obligatoire au même niveau** : la cible de volatilité dynamique sans
  paramètre. A ne peut pas passer en la perdant.

### Niveau B — la rupture de corrélation comme entrée de CONSTRUCTION

**Pourquoi B échappe à la raison pour laquelle A tombe.** A conditionne le *signal* sur
la forme du *chemin*. Si A tombe parce que les deux vitesses sont trop proches
(accord 74,4 %) ou parce qu'un épisode de 23 séances est trop court pour amortir une
rotation plus rapide, B ne dépend d'aucun des deux : il ne change **pas** le signal, il
change **la manière dont le risque est réparti entre les 46**. Et cette répartition
n'existe pas aujourd'hui — `extensions/trend.py` met à l'échelle par σ_i seul puis
normalise par le brut ; **il n'y a aucun terme de covariance dans le livre**. B ajoute
un module absent, il ne règle pas un module présent.

- **Latente** : ‖C₆₃ − C₂₅₂‖_F sur le bloc complet à 28, résidualisée causalement sur
  la volatilité (corr. résiduelle mesurée +0,009). CUSUM k=1,0 h=5 → 7,26 ruptures/an.
- **Dispositif** : la fenêtre de l'estimateur de covariance se réinitialise à chaque
  rupture et s'allonge entre les ruptures ; cette covariance alimente une répartition
  du risque à signe de tendance imposé, renormalisée au même brut.
- **B1, le test qui décide** : une rupture prédit-elle une hausse de l'erreur du modèle
  de risque — c'est-à-dire de l'écart entre la volatilité du livre prévue et réalisée ?
  C'est une affirmation de **variance**, le seul canal où le programme a déjà une
  preuve positive (t −3,40). **Falsification B1** : t HAC lag 6 sous 1,96 après
  correction d'arbre, ou signe contraire ⇒ B1 mort.
- **B2** : Sharpe net en excess ≥ 0,331 d'écart, plus un secondaire obligatoire — la
  covariance réinitialisée tient-elle la cible de 10 % plus serré que la fixe ?

### Niveau C — bascule vers le retour à la moyenne, avec une porte inconditionnelle

**Pourquoi C échappe à la raison pour laquelle B tombe.** B demande encore au signal de
tendance d'avoir raison, et à la structure de risque d'être améliorable. C arrête de le
demander : en consolidation détectée, le mouvement à 5 jours de l'instrument est fadé
au lieu d'être suivi.

- **C0, porte inconditionnelle, une demi-journée** : la prime de retournement à 5 jours
  sur les 46 instruments transversaux, **brute de coûts**, sur 2016-2026, est-elle
  positive ? **Si elle est ≤ 0, C est mort sur la règle de kill mécanique que le
  programme a déjà appliquée aux actions américaines, et l'arbre se ferme.** Conditionner
  une prime qui n'existe pas est la recherche du sous-ensemble où un effet mort survit
  encore, et `reversal-lab` a explicitement refusé de le faire.
- **C1**, seulement si C0 passe : l'état de consolidation améliore-t-il la jambe de
  retournement au-delà du MDE ? Mêmes placebos, même correction.

### Niveau D — la fermeture, et ce qu'elle dit

Si A, B et C tombent, ce qui s'écrit est plus large que ce que le programme peut écrire
aujourd'hui : **l'échec ne tient ni à l'ordonnancement par la volatilité (franchi à
r = +0,096 / −0,044), ni à la constante de temps de 0,53/an (franchie à 11,04/an), ni
à l'usage de dimensionnement (remplacé par une sélection neutre en exposition), ni au
livre unique (remplacé par une bibliothèque de 138 jambes).** Les quatre échappatoires
qu'un lecteur opposerait à « les régimes ne se monétisent pas » sont fermées une par une,
avec un chiffre chacune.

### Correction pour tests multiples, calculée sur l'arbre entier

Tests primaires : A1, A2, B1, B2, C0, C1 = **6**. Holm-Bonferroni à un α familial de
0,05 : seuils 0,00833 / 0,0100 / 0,0125 / 0,0167 / 0,0250 / 0,0500 par rang. Le plus
strict, z bilatéral = **2,638** contre 1,960, soit un facteur **1,346** sur tout MDE.

| quantité | non corrigée | **corrigée pour l'arbre** |
|---|---|---|
| MDE portefeuille, bloc 126 (ancrage programme) | 0,246 | **0,331** |
| MDE portefeuille, bloc 63 | 0,264 | 0,355 |
| MDE portefeuille, bloc 21 | 0,273 | 0,367 |
| MDE panneau, attente déclarée | 2 à 4 pts | **2,7 à 5,4 pts** |

Les sensibilités **rapportées sans choix entre elles** — trois colonnes de coût, trois
longueurs de bloc, le panneau M1 à trois séries vérifiées, la jambe rapide (63,5) — ne
sont pas des essais séparés, par la même règle que §12 du pré-enregistrement précédent.
Si un choix est fait entre elles, elles deviennent des essais rétroactivement et la
correction est recalculée. **Les six primaires sont déclarées ici et leur nombre ne peut
pas augmenter sans amendement journalisé.**

Déflation du Sharpe : base de 83 configurations distinctes déjà dans
`data/trials.parquet`, plus chaque configuration de cet arbre, journalisée par
`regime_lab.analysis.trials`. **Et l'estimateur de variance est fixé ici** — variance
des Sharpe sur les hachages distincts, en prenant la **dernière** ligne par hachage —
parce que c'est précisément l'omission qui a fait s'effondrer K5 la fois précédente,
où l'excès déflaté courait de −0,159 à +0,286 selon un estimateur non déclaré.

### Effort et probabilités, honnêtes

| étape | jours | P(succès de l'étape) |
|---|---|---|
| détecteurs, tests unitaires, MDE sous la nulle | 2,0 | — |
| **A1** panneau + placebos | 2,0 | **0,40** |
| **A2** portefeuille + témoin d'une ligne | 1,5 | 0,25 sachant A1 ⇒ **0,10** conjoint |
| **B1** module de covariance + test de variance | 2,5 | **0,35** |
| **B2** portefeuille | 0,5 | 0,22 sachant B1 ⇒ **0,08** conjoint |
| **C0** porte inconditionnelle | 0,5 | **0,30** |
| **C1** conditionnel | 1,0 | 0,17 sachant C0 ⇒ **0,05** conjoint |
| rédaction, amendements, journal | 1,5 | — |
| **total** | **11,5 jours** | |

- P(au moins un étage de panneau passe, donc un mécanisme mesuré et publiable) ≈ **0,60**.
- P(un PASS de portefeuille au MDE corrigé, quelque part dans l'arbre) ≈ **0,20**.
- P(le livre commuté atteint lui-même RESEARCH_PASS, Sharpe net en excess > 0,70) ≈ **0,08**.
  Le livre de base lit +0,359 ; il faudrait exactement un MDE corrigé d'amélioration.

**Cet arbre n'est pas une route vers un livre à 0,70, et le prétendre serait malhonnête.**
C'est une route vers un mécanisme mesuré, et vers une phrase beaucoup plus large que
celle que le programme peut écrire aujourd'hui.

---

## h) CE QU'ON APPREND MÊME SI TOUT L'ARBRE TOMBE

1. **La généralisation devient soutenable.** Aujourd'hui le programme a le droit
   d'écrire une phrase étroite : « un classifieur ordonné par la volatilité, à 13
   transitions en 25 ans, en multiplicateur de taille sur un livre de tendance unique,
   n'apporte rien au-delà d'un quantile de volatilité ». Après cet arbre, quatre
   qualificatifs tombent, chacun avec un chiffre. C'est le produit principal, et il est
   acquis que l'arbre passe ou non.
2. **Une correction mesurée à un document publié.** `EXTENSIONS.md` §1 rapporte l'échec
   du compteur de facteurs effectifs sans en donner la cause. La cause est ici :
   corr(part de la première valeur propre, volatilité réalisée) = **+0,594** Pearson,
   **+0,633** Spearman. Ce n'était pas une mesure de structure. C'est déjà mesuré,
   ça ne coûte rien de plus, et ça ferme proprement une piste au lieu de la laisser
   « échouée sans raison ».
3. **Une frontière de coût pré-déclarée pour toute la famille.** La commutation de
   signaux sur cet univers meurt à **13,5 bp** aller-retour mélangé pour une jambe
   (126,10) et à **7,9 bp** pour une jambe (63,5). Ce nombre décide à l'avance si un
   futur dispositif de commutation vaut la peine d'être construit — y compris pour le
   dépôt voisin, dont les futures paient ~1 bp et les actions 5 à 10.
4. **Un dénominateur au niveau signal.** 46 instruments × 3 vitesses, audités, facturés,
   en excess, sur 23,15 ans — l'analogue au niveau signal de ce que
   `RESULTS_TREND_VEHICLE.md` a livré au niveau livre.
5. **Un résultat sur la méthode, pas seulement sur les régimes.** Si un test de panneau
   disposant de 50× l'échantillon effectif d'une différence de Sharpe de portefeuille ne
   voit toujours rien, alors les six verdicts « UNDERPOWERED » précédents ne sont pas de
   simples échecs de puissance : l'effet est **absent**, pas caché. C'est le résultat que
   le programme n'a jamais pu écrire, et c'est celui qui change la valeur de tous les
   autres.

---

## Contraintes d'exécution héritées

Ne jamais relancer `scripts/run_phase2.py` — 15 à 20 minutes et il réécrit
`states.parquet`. Ne rien retélécharger. Utiliser le venv du dépôt. Toute déviation va
dans `docs/PROTOCOL_FREEZE.md`, jamais dans le cadrage édité en place. Signal en T−1,
trade en T ; rendements en excess ; HAC lag 6 ; walk-forward ≥ 5 plis ; ciblage de
volatilité quotidien ; N ≥ 30 pour toute affirmation transversale ; bootstrap
stationnaire par blocs, jamais iid ; placebo apparié obligatoire ; **un écart sous le
MDE est UNDERPOWERED et jamais un PASS.**
