# Arbitrage — cinq plans, par quoi on commence

22 septembre 2026. Arbitre les cinq pré-enregistrements de `firmes/{rentec, bridgewater,
ahl, aqr, twosigma}`. Aucun rendement de backtest n'a été regardé, ici ni dans les cinq
instructions. Rien n'a été écrit dans les dépôts.

---

## 0. Ce que j'ai vérifié moi-même, avant d'arbitrer

Les cinq agents rapportent leurs propres mesures. Je ne les arbitre pas sur leur parole.
Ce qui suit vient de commandes exécutées dans cette session, sur
`.venv/bin/python`.

**La convention de puissance du programme se reproduit.** Avec l'erreur type de Lo,
`SE(Sharpe annualisé) = racine((1 + S²/2) / Y)`, Y en années, et `z + z_beta = 2,8016` :

| échantillon | années | plus petit Sharpe autonome résoluble | publié |
|---|---|---|---|
| référence du programme | 23,20 | **0,638** | 0,639 |
| porte RESEARCH_PASS | 19,90 | **0,701** | 0,70 |
| intersection futures (A2) | 8,90 | **1,256** | 1,253 |
| Bridgewater, 5 manches | 23,60 | 0,632 | 0,632 |
| Two Sigma, bibliothèque | 31,57 | 0,533 | — |
| AQR, 49 secteurs 1990-2026 | 36,57 | 0,490 | — |
| AQR, holdout scellé | 18,70 | 0,729 | — |
| Rentec, crypto | 8,40 / 5,60 | 1,324 / 2,164 | l'agent annonce 0,960 / 1,252 |

Le dénominateur devient positif à **3,924 ans** : en dessous, aucun effet n'est séparable
de zéro, quelle que soit sa taille. C'est le seuil qui tue la cellule
(croissance−, inflation−) de Bridgewater, à 3,34 ans — sa mesure est confirmée.

L'écart Rentec est une différence de convention, pas une erreur : ses 0,960 et 1,252 sont
des MDE **appariés** par bootstrap, les miens sont des seuils **autonomes**. Les deux
disent la même chose, la mienne plus sévèrement. À trancher au verrouillage, pas après.

**Les horloges du programme, recomptées sur `data/cache/states.parquet` (9 007 × 5) :**

| état | n | transitions | par an |
|---|---|---|---|
| A jump | 6 377 | 13 | **0,514** |
| A′ sparse jump | 6 377 | 13 | **0,514** |
| B HMM filtré | 6 377 | 49 | 1,936 |
| C gradient boost | 6 377 | 344 | 13,594 |
| C′ HAR-RV | 6 377 | 450 | 17,783 |

Le chiffre fondateur du programme tient au recompte. Les cinq agents l'écrivent 0,51,
0,514, 0,53 ou 0,532 selon leur convention d'année ; c'est la même mesure.

**Les panneaux, ouverts et comptés.** `trend_universe.parquet` 6 822 × 46,
2000-07-17 → 2026-09-10, identique en forme aux deux panneaux M1.
`industry_49.parquet` : 49 séries × 9 212 séances, 1990-01-02 → 2026-07-31, et
`available_at == period` sur **100,0 %** des lignes — le défaut hérité que Two Sigma
déclare est réel. `features.parquet` 9 572 × 50. `trials.parquet` : 805 lignes,
**83 hachages de configuration distincts**, 7 familles.

**Deux défauts confirmés à la main, dont un qui n'était dans aucun document du programme.**

1. `data/raw/panels/factors_5.parquet` ne contient que
   `['ff_cma','ff_hml','ff_mkt-rf','ff_rf','ff_rmw','ff_smb']`. Le momentum est absent.
   `regime_lab/features/crosssection.py` boucle sur `("ff_smb","ff_hml","ff_mom")`
   derrière un garde `if name in factors` : la feature `xs_ff_mom_63` **n'a jamais été
   construite**, silencieusement, et personne ne s'en était aperçu. La mesure d'AQR est
   exacte. À verser à la liste d'hygiène.
2. Le départ N ≥ 30 : le texte gelé de `PRESPEC_TREND_VEHICLE.md:118` écrit
   **2003-07-17**, le recompte permissif donne 2003-07-16. J'ai diagnostiqué la cause :
   la fenêtre de 252 séances compte-t-elle la séance courante ?
   - fenêtre incluant la séance courante → 2003-07-16, n = **6 040**
   - fenêtre exigeant 252 séances **antérieures** → 2003-07-17, n = **6 039**

   La seconde lecture est la causalement correcte — le signal se forme en T−1 — et c'est
   celle qui est gelée. L'écart que l'agent AHL déclarait « à trancher » est donc tranché,
   et il tranche **contre** son propre script.

**Les bibliothèques de signaux, mesurées sur les matrices que les agents ont laissées :**

| objet | rang effectif | 1ʳᵉ valeur propre | corr. \|moyenne\| |
|---|---|---|---|
| 12 signaux sur les 49 secteurs | **8,37** | 2,45 / 12 | 0,1005 |
| 11 signaux sur l'univers des 46 | **3,86** | 4,82 / 11 | 0,3363 |
| les 3 livres d'AQR seuls | 2,95 | 1,17 / 3 | 0,081 |

Et le fait qui commande la section 4 : **les trois livres d'AQR sont déjà dans la
bibliothèque de Two Sigma.** `IND_MOM_12_1`, `IND_LOWBETA`, `IND_LTREV` sont trois des
douze colonnes de `twosigma/corr_industries.csv`, et ils sont quasi orthogonaux entre eux
(|corr| max 0,133).

**Le patrimoine de Rentec existe** : 27 fichiers de barres à la minute, **8,8 Go**, dans le dépôt privé voisin, hors du
programme Big Data et hors de ce dépôt.

---

## 1. Le classement par espérance

Pas par intérêt intellectuel. Le numérateur est « probabilité d'un mécanisme mesuré et
publiable », pas « probabilité d'un Sharpe » — la section 5 explique pourquoi cette
seconde quantité est hors de portée pour les cinq.

| rang | plan | effort | P(PASS inférentiel) | P(résultat mesuré publiable) | mécanisme par jour | donnée manquante |
|---|---|---|---|---|---|---|
| **1** | **Man AHL** | 11,5 j | 0,20 (portefeuille) | **0,60** (panneau) | **0,052** | **aucune** |
| **2** | **Two Sigma** (fusionné avec AQR) | 10 j | 0,30 | ~1,00 | 0,030 | aucune |
| **3** | **Bridgewater** | 8,5 j | 0,30 | ~0,90 | 0,035 | dates de parution SPF (non testées), rendement total actions (introuvable gratuitement) |
| **4** | **Renaissance** | 11 j | 0,25 | 0,75 | 0,023 | aucune pour l'arbre ; carnet L2 hors budget |
| **5** | **AQR seul** | 9 j | 0,18 | 0,90 | 0,020 | aucune bloquante |

### 1. Man AHL — 11,5 jours, P(mécanisme) 0,60, P(PASS portefeuille) 0,20

Premier pour quatre raisons mesurées, pas pour son élégance.

C'est le seul des cinq qui franchisse les quatre axes **simultanément** : latente
pratiquement orthogonale à la volatilité (ER63 lit +0,096 Pearson et −0,044 Spearman
contre la vol réalisée 63 j, là où `mapping.py` ordonne explicitement les états
programme *par* la volatilité), horloge à 11,04 ruptures/an contre 0,514 recompté ici
(facteur 21), usage de **sélection neutre en exposition**, objet de 138 jambes.

L'axe 3 est celui qui compte, et il ne repose pas sur une promesse : le dispositif ne
coupe jamais le risque, et la neutralité est un verrou machine (`|brut(commuté) −
brut(base)| / brut(base) < 0,02`, une violation abandonne la mesure). C'est la seule
façon connue de fermer *par construction* le canal qui a tué T1, T3 et l'atténuateur de
Carver — T3 était au 0ᵉ percentile en bêta, et la barrière a mesuré que p(pass) s'achète
à 0,2785 par unité d'exposition. Un dispositif qui ne coupe rien ne peut pas gagner là.

C'est aussi le seul plan dont la réponse à « quelle donnée manque » est **aucune, et
c'est une décision** : rien à télécharger, sur un programme qui s'est fait avoir deux
fois (Stooq derrière un mur anti-bot, codes pays OCDE). Tout tourne sur
`trend_universe_m1.parquet`, que j'ai ouvert, et sur le venv du dépôt.

Et il a un avantage structurel que les quatre autres n'ont pas : les six réfutations ont
toutes réduit la question à **une** différence de Sharpe de portefeuille, en jetant la
transversale, alors que le programme a démontré sur lui-même qu'un panneau voit ce qu'un
portefeuille ne voit pas — +3,93 points de R² incrémental, t −3,40. AHL est le seul plan
dont le test de premier rang vit sur 46 × 6 039 et non sur 6 039.

Ce qu'on apprend même si tout tombe, et c'est le vrai produit : la phrase étroite du
programme perd ses quatre qualificatifs, chacun avec un chiffre en face. Plus un
sous-produit **déjà acquis à coût nul** : la part de la première valeur propre corrèle
à +0,594 avec la volatilité réalisée, ce qui donne enfin la cause mesurée de l'échec de
`concentration.py`, dont `EXTENSIONS.md` §1 rapporte tous les |t| sous 1,6 sans
l'expliquer.

Ce qu'il ne faut pas lui faire dire : P(le livre commuté atteigne lui-même
RESEARCH_PASS) = **0,08** par l'estimation de son propre auteur. Ce n'est pas une route
vers 0,70. C'est une route vers un mécanisme mesuré.

Coût du rang : c'est le plan le plus cher des cinq, et son tueur le plus probable n'est
pas résolu — à 11,04 ruptures/an un épisode dure ~23 séances, et la latence de détecteur
mesurée sur les états du programme va de 0 à 13 jours. Une latence de 5 à 10 jours mange
la moitié d'un épisode. L'auteur le nomme et ne le minimise pas.

### 2. Two Sigma — 10 jours, P 0,30, résultat rédigeable certain

Deuxième, et **fusionné avec AQR** (section 4) : ce sont deux familles déclarées sur un
seul panneau, et l'une des deux bibliothèques est un sous-ensemble de l'autre, mesuré
ci-dessus.

Son meilleur résultat est déjà encaissé et ne coûte plus rien : la dimension effective de
la bibliothèque du programme, mesurée pour la première fois — 3,86 sur l'univers des 46,
8,37 sur les 49 secteurs. Ce contraste éclaire rétrospectivement une partie des six
échecs : l'objet conditionné n'avait pas une dimension, il en avait environ quatre, dont
trois étaient la même tendance à trois horizons (`TSMOM_1M` et `XSREV_1M` corrèlent à
−0,990, `TSMOM_12_1` et `XSMOM_12_1` à +0,928 — vérifié).

Position de puissance la plus faible des cinq, et il faut le dire : l'auteur écrit
qu'il doit franchir 0,338 de Sharpe apparié après correction familiale et qu'**il ne peut
pas savoir si l'effet visé est au-dessus** sans toucher la donnée. C'est honnête et c'est
plus mauvais qu'AQR (marge 1,2× chiffrée), que Rentec (4 à 6× chiffrés) ou que
Bridgewater (2× chiffré sur la variance). Le plan compense en déplaçant la décision vers
le transfert de rang hors échantillon, adossé à 215 épisodes et 20 cellules — là où H-b
n'avait que treize décisions à observer.

### 3. Bridgewater — 8,5 jours, P 0,30, le moins cher

Le seul des cinq sur l'axe **construction de portefeuille**, que les six réfutations
n'ont jamais touché : le quadrant n'entre jamais dans une décision datée, il entre dans
la fonction objectif, et le portefeuille est détenu sans savoir dans quelle cellule on est.

Son meilleur résultat est déjà acquis et il est contre-intuitif : la persistance du
quadrant vaut **0,220 contre 0,266 sous indépendance**. Un quadrant macroéconomique
n'est pas un régime, c'est une suite de chocs quasi indépendants, légèrement
anti-persistants — ce qu'un consensus efficient doit produire. Cela **interdit** tout
usage ex ante, et n'interdit rien à All Weather, qui ne prédit pas le quadrant.

Il résout aussi la difficulté que la tâche signalait : le consensus réel est gratuit et
accessible — quatre fichiers SPF de la Fed de Philadelphie, 232 enquêtes chacun,
1968-T4 → 2026-T3, téléchargés et ouverts. L'agent s'est d'abord fait prendre par deux
URL rendant HTTP 200 sur une page HTML « 404 » de 18 401 octets, et il le nomme plutôt
que de le taire. C'est le piège Stooq, troisième instance.

Troisième et pas premier pour trois raisons. La persistance 0,220 qui rend l'objet
admissible rend aussi le placebo P1 (partitions aléatoires à occupation et durées de
séjour appariées) **presque parfait** : le niveau A est conçu pour être difficile à
passer, et son auteur le sait. Le tueur n°2 est structurel — les cinq manches ont été
choisies parce qu'elles couvrent croissance et inflation, donc égaliser les
contributions de variance par cellule pourrait retrouver presque les mêmes poids. Et
trois des 8,5 jours sont du socle : construire un livre à 5 manches et 30 instruments
qui n'existe pas, avec une asymétrie de véhicule qui mord (indices actions en prix,
neuf ETF en rendement total, ~2 points de dividende par an qui poussent mécaniquement
les poids de parité vers les obligations — la même classe de défaut que la correction A4
à 0,041 de Sharpe).

### 4. Renaissance — 11 jours, P 0,25, et le meilleur rapport puissance/question

Le seul plan dont l'échantillon soit **sur-puissant** sur sa cible : rapport prix/MDE de
3,9 à 5,7 en points de base, contre une sous-puissance d'un facteur ~2 en Sharpe sur le
*même* échantillon. Cette asymétrie mesurée est la justification entière de son choix
d'usage, et c'est le raisonnement le mieux construit des cinq.

Deux vérifications ont changé son plan en cours de route, ce qui est le signe qu'il a
réellement ouvert les fichiers : le patrimoine à la minute porte une **colonne de spread
flottant réel** sur 27 instruments (8,8 Go vérifiés ici), ce qui fait de la cible une
grandeur observée et non un mandataire ; et les 17 fichiers crypto ont une colonne de
spread identiquement nulle, donc la crypto ne peut pas héberger une étude de coût.

Le fait le plus utile au programme sort de là : **le témoin d'une ligne qui a battu les
six dispositifs est inopérant sur cette cible** — un quintile de volatilité réalisée
n'explique que −0,016 à +0,186 de la variance du log-spread, et corr(spread, |rendement|)
vaut −0,149 à +0,096. La prémisse « les fourchettes s'écartent dans la volatilité » est
fausse sur ce flux. Ce n'est pas une bonne nouvelle nette : le témoin change de forme et
devient une table heure-du-jour × niveau lent qui explique jusqu'à 0,802 de la même
variance. Le plan le nomme P2 et le déclare comme le vrai adversaire.

Quatrième pour une raison d'affectation, pas de qualité : **le livrable appartient au
dépôt privé voisin, pas au mémoire.** C'est un modèle de coût d'exécution calibré sur 27
instruments — précieux, et dont aucun des deux dépôts ne dispose — mais ce n'est pas une
falsification de l'axe régime pour l'étude principale. Guillaume fait tourner deux
sessions ; celle-ci est pour l'autre.

Avec une exception importante : **sa porte G0 est l'expérience la moins chère de tout le
lot.** 1,5 jour, **zéro essai dépensé**, P = 0,55 de fermer la branche entière et de
publier « un état de microstructure à 5 000-21 000 transitions/an n'est pas une partition
stable sur ce patrimoine ». Elle existe parce que l'instabilité est mesurée et déclarée
contre l'auteur lui-même : facteur cinq sur le nombre de transitions entre BTC 2022
(3 353/an) et BTC 2024 (18 960/an), à spécification et graine identiques, plus deux
avertissements de non-convergence. Le classifieur quotidien avait une stabilité de
partition à 73-137× le nul ; celui-ci ne l'a pas démontrée.

**À signaler à l'autre session, c'est un défaut d'intégrité** : les chiffres du gate-0
crypto (médiane 3,4 bp, p90 11, p99 27) viennent d'un fichier BTCUSD lu dans un bac à sable distant qui n'existe plus ;
la seule copie locale, dans le dépôt privé voisin, n'a pas de colonne de spread. **Ces
chiffres ne sont pas reproductibles localement.**

### 5. AQR seul — 9 jours, P 0,18

Dernier, et son contenu survit néanmoins : il entre dans le plan fusionné du rang 2.

Ses deux meilleurs résultats sont **déjà acquis à coût nul**, avant le
pré-enregistrement plutôt qu'après. L'écart de valorisation actualisé par les prix —
la variable d'état que la firme met réellement en avant — donne 0,12 changement de signe
par an, soit 4,3 transitions en 36 ans et une demi-vie AR(1) de 1,81 an : **cinq fois
plus lent que l'objet déjà réfuté**. Le scan praticien avait raison de déconseiller le
timing factoriel par la valorisation, et le programme sait maintenant pourquoi sur ses
propres données. Et le défaut `ff_mom` absent de `factors_5.parquet`, que j'ai confirmé
à la main.

Dernier parce que sa marge de puissance est la plus mince des cinq qui en aient chiffré
une : ΔIC visé ≈ 0,08 contre un MDE Šidák de 0,0666, soit **1,2×**. À P(A) = 12 % et
P(B) = 10 %, l'essentiel de son espérance est dans son niveau C-variance à 35 %, qui est
le même test que le niveau B de Two Sigma sur la même donnée.

Son actif rare est ailleurs et il faut le protéger : un **holdout scellé de 18,7 ans**
(1971-04 → 1989-12, 4 733 séances, 49 secteurs peuplés partout) que le programme n'a
jamais touché sur aucune des six réfutations. Il ne résout que 0,0853, donc c'est une
réplication de direction, pas une seconde chasse au p. Il ne doit être ouvert qu'une
fois, et par le plan fusionné, pas deux fois par deux plans.

---

## 2. Ce qui est inadmissible

**Aucun des cinq plans n'échoue au test d'admission.** Les cinq diffèrent sur au moins un
axe, chiffré. Ce qui est inadmissible, ce sont des **objets précis à l'intérieur** de ces
plans, plus une inadmissibilité de procédure. Chacun est mesuré, pas argumenté.

**La règle générale, écrite pour pouvoir être opposée à la prochaine idée.** Est un
septième dispositif tout objet qui réunit les quatre propriétés : latente ordonnée par la
volatilité, horloge sous ~2 transitions/an, usage de dimensionnement ou d'interrupteur, et
objet conditionné de dimension effective inférieure à 4. Il tombera sur le même placebo
d'une ligne, et le déclarer vaut mieux que de le découvrir.

1. **L'écart de valorisation d'AQR.** 0,12 changement de signe/an, 4,3 transitions en
   36 ans, demi-vie 456 séances. C'est **plus lent qu'A′ sparse jump** (0,514/an,
   recompté ici). L'axe 2 n'est pas seulement non franchi : il est franchi à l'envers.
   Écarté par son propre auteur ; je confirme l'exclusion.
2. **Le NIVEAU du NFCI** (1,27 transition/an, ρ = +0,639 avec la volatilité réalisée)
   **et le quadrant de NIVEAU de Bridgewater** (1,20/an). Ce sont le quantile de
   volatilité sous un autre nom, à la cadence de H-b. Les deux plans les gardent comme
   variantes de sensibilité ; ils ne peuvent jamais fonder un PASS.
3. **Les mélanges gaussiens de Two Sigma sur l'espace de contexte** : 0,90 à 2,43
   transitions/an selon K, contre 6,46 à 8,49 pour k-means sur le même espace
   (`twosigma/clock.csv`). Et **tout lissage qui ramène l'horloge sous ~2,6/an** — le
   lissage à 21 séances ramène k-means K=4 de 6,46 à 2,38. C'est la molette qui reconduit
   dans la famille réfutée ; déclarée comme sensibilité pré-enregistrée, elle est
   acceptable ; promue en titre, elle serait un septième dispositif.
4. **L'univers des 46 instruments comme objet conditionné d'une étude de sélection.**
   Rang effectif **3,86** sur 11 signaux, première valeur propre 4,82/11, `TSMOM_1M` et
   `XSREV_1M` à −0,990. L'axe 4 n'y est pas franchissable, et c'est mesuré. Toute étude
   de sélection entre signaux sur les 46 est close avant d'être écrite. *(Note : cela ne
   disqualifie pas AHL, dont l'objet est 138 jambes signal-instrument sur trois vitesses
   et non une bibliothèque de signaux décorrélés — mais c'est la borne qui l'encadre.)*
5. **Toute conversion d'un résultat de l'arbre Rentec en revendication de Sharpe sur
   BTC.** Un post-mortem antérieur du dépôt privé voisin déclare cet espace épuisé :
   n_trials honnête ~50, E[maxSR] 1,8-2,3, « tout nouveau candidat de cet espace naît
   mort ». Le pare-feu de portée du plan Rentec est obligatoire, pas décoratif.
6. **Δp(pass) comme critère d'admission.** Tué deux fois, des deux côtés, indépendamment.
   Aucun des cinq ne le propose ; c'est écrit pour que cela reste vrai.
7. **Inadmissibilité de procédure, et c'est la plus importante : AQR et Two Sigma
   déclarés comme deux familles séparées.** Même panneau (49 secteurs, 9 212 séances),
   même usage (sélection entre signaux), même canal de secours (la covariance entre
   signaux jugée sur la variance), et une bibliothèque qui est un sous-ensemble strict de
   l'autre — vérifié : `IND_MOM_12_1`, `IND_LOWBETA`, `IND_LTREV` sont trois des douze
   signaux de Two Sigma. Les faire tourner séparément, c'est déclarer 11 tests primaires
   sur un échantillon, compter la multiplicité deux fois à moitié, et ouvrir deux fois un
   holdout qui ne peut l'être qu'une. **Ils fusionnent ou l'un des deux ne tourne pas.**

---

## 3. La piste n°1, et ce qu'on fait dès demain

**Man AHL.** Et le premier jour ne dépense aucun essai du registre.

### Étape 1 — trancher le départ d'échantillon, et l'amender au journal (1 heure)

C'est le préalable : toute nouvelle étude hérite de cette définition, et deux documents
gelés disent une chose pendant que le script de mesure en dit une autre.

Fichier : `docs/PRESPEC_TREND_VEHICLE.md` §3, ligne 118.

```bash
# depuis la racine du dépôt
.venv/bin/python -c "
import pandas as pd
px = pd.read_parquet('data/cache/trend_universe_m1.parquet')
c  = px.notna().rolling(252).sum().ge(252).sum(axis=1)
print('seance courante incluse   :', c[c>=30].index[0].date(), len(px.loc[c[c>=30].index[0]:]))
print('252 seances ANTERIEURES   :', c.shift(1)[c.shift(1)>=30].index[0].date(), len(px.loc[c.shift(1)[c.shift(1)>=30].index[0]:]))
"
```

Sortie attendue, déjà produite ici : `2003-07-16 / 6040` et `2003-07-17 / 6039`.
**Décision : la seconde**, parce que le signal se forme en T−1 et qu'elle est celle qui
est gelée. Journaliser dans `docs/PROTOCOL_FREEZE.md` comme amendement — jamais éditer
le cadrage en place. Coût : une heure, et cela retire une ambiguïté que cinq plans
auraient héritée.

### Étape 2 — regeler les chiffres d'admission (1 heure)

```bash
.venv/bin/python pilotage/mesures_brutes/firmes/ahl/measure_timeconstant.py   # non versionné, voir AVANCEMENT.md §3.4
```

Ce script ne calcule aucun rendement de stratégie : prix, latentes, comptes de
transitions, corrélations avec la volatilité. Il regèle ER63 contre la vol
(+0,096 / −0,044), les 11,04 ruptures/an du BOCPD et la bande 4,73-15,66 des détecteurs.
Le figer avec la nouvelle date de départ.

### Étape 3 — LE PREMIER CHIFFRE À PRODUIRE, et c'est celui qui décide

**Le MDE du panneau, sous l'hypothèse nulle, avant qu'aucun coefficient d'interaction ne
soit lu.** Bootstrap stationnaire (Politis-Romano, jamais iid) sur la dimension **date**,
transversale entière conservée dans chaque tirage, blocs 21 / 63 / 126 tous les trois
rapportés, sur 46 × 6 039.

Statistique :
`Δ = [hit(rapide|consolidation) − hit(lent|consolidation)] − [hit(rapide|tendance) − hit(lent|tendance)]`,
pondérée par l'exposition ajustée au risque.

Le plan ne fixe pas le nombre, il fixe la procédure, et c'est la bonne façon de faire :
l'encadrement analytique va de ~0,8 point (regroupement par date) à 4,5-6,0 points
(regroupement par épisode de 63 séances), et deviner entre les deux serait du confort.
Attente déclarée : 2 à 4 points, 2,7 à 5,4 après la correction de Holm sur les 6 tests de
l'arbre.

**Le critère de décision du jour 1, écrit avant de le produire :** si le MDE mesuré sort
au bout « par épisode » de l'encadrement, c'est-à-dire au-dessus de ~4,5 points, alors
même la question de panneau est indécidable sur cet échantillon, A1 n'est pas posable, et
on l'écrit — **une demi-journée dépensée, zéro essai, un résultat à coût nul**. C'est
exactement la forme que la tâche demandait, et il faut l'accepter à l'avance pour qu'elle
ait une valeur.

### Étape 4, en parallèle et pour une demi-journée — la porte C0

La prime de retournement à 5 jours sur les 46 instruments transversaux, **brute de
coûts**, sur 2016-2026, est-elle positive ? Si ≤ 0, le niveau C est mort sur la règle de
kill mécanique que le programme a déjà appliquée aux actions américaines, et on ne
conditionne pas une prime qui n'existe pas. `reversal-lab` a explicitement refusé de le
faire, et son kill est géographiquement borné aux actions américaines — donc la question
se pose, et elle se ferme en une demi-journée.

### Et, dans l'autre session, le même jour — la porte G0 de Rentec

1,5 jour, **zéro essai dépensé du registre du programme**, P = 0,55 de fermer une branche
entière. Réestimer le HMM sur les 5 plis walk-forward, accord de partition inter-plis
(Rand ajusté) contre un nul de permutation par blocs de 21 jours, 1 000 permutations.
PASS si l'accord ≥ 10× le nul sur ≥ 4 paires de plis sur 5 — barre volontairement très
en dessous des 73-137× du classifieur quotidien, et déclarée basse.

C'est le meilleur taux de verdict par jour du lot entier : **0,37 verdict/jour** contre
0,020 à 0,052 mécanisme/jour pour les cinq arbres complets. Elle appartient au
dépôt privé voisin et n'entre pas dans le budget d'essais du mémoire.

---

## 4. Ce qui peut se mutualiser

Construire une fois vaut mieux que cinq. Par ordre de ce qu'on économise.

**1. Le placebo apparié horloge + occupation — une seule implémentation.**
Les cinq plans en ont besoin : P1 de Bridgewater (2 000 partitions aléatoires conservant
l'occupation 0,143/0,308/0,275/0,275 et les durées de séjour), P1 d'AQR (placebo à
transitions appariées), P-E et P-B d'AHL, P1 et P2 de Rentec, le nul de Two Sigma. Et il
est **déjà écrit et déjà payé deux fois** : `twosigma/measure_placebo.py` documente en
tête ses deux versions fausses — la première perd 29 % des transitions aux jointures
fusionnées, la seconde répare l'horloge et casse l'occupation de 0,304 — et seule la
troisième apparie exactement les deux, en permutant des paires (longueur, état) entières
puis en réparant les jointures par échange de paires, ce qui ne peut pas modifier le
multi-ensemble. **C'est le premier morceau à promouvoir en module du dépôt**, avec ses
tests. Économie : quatre réimplémentations, dont on sait maintenant que deux tentatives
sur trois sont fausses.

**2. La bibliothèque de signaux sur les 49 secteurs — une seule, et elle est déjà
choisie.** Les trois livres d'AQR (momentum 12-1, bêta faible, retournement long) sont
littéralement `IND_MOM_12_1`, `IND_LOWBETA`, `IND_LTREV`, trois des douze signaux de
Two Sigma, quasi orthogonaux entre eux (rang effectif 2,95 sur 3). Construire les douze
une fois, retenir les dix de la bibliothèque inférentielle de Two Sigma (`IND_REV_1W`
exclu avant tout contact pour sa rotation de 156,7×/an, `IND_ACCEL` pour sa redondance de
−0,912 avec `IND_MOM_12_1`), et faire d'AQR un sous-arbre de la même famille : sa carte
des signes déclarée devient un bras à trois signaux de la même bibliothèque, son
niveau C-variance et le niveau B de Two Sigma deviennent **un seul test** de covariance
conditionnelle, et le holdout scellé de 18,7 ans ne s'ouvre qu'une fois. Économie : 4 à
5 jours, et une famille de tests au lieu de deux qui se recouvrent.

**3. Le harnais de puissance, avec UNE convention.** Les cinq agents ont recalibré
`regime_lab/analysis/power.py` chacun de leur côté, et trois conventions différentes
circulent déjà (autonome de Lo, apparié par bootstrap, années contre séances 24/7). Rentec
annonce 0,960 à 8,4 ans là où la convention publiée du programme donne 1,324. Tant que ce
n'est pas unifié, **les MDE des cinq plans ne sont pas comparables entre eux**. Un seul
module, un seul étalonnage — celui reproduit en section 0 : 23,2 ans → 0,638 contre 0,639
publié.

**4. La couche de construction en excess.** `data/raw/macro/rate_cash_3m.parquet`
(9 177 lignes quotidiennes, 1990-01-02 → 2026-09-08), `regime_lab/extensions/vehicle.py`
issu de M3 (cible de vol quotidienne, tenue à 10,84-10,90 % contre 10 % visé sur les trois
panneaux), et le barème de coûts par classe d'actif. Quatre plans sur cinq en ont besoin,
et la correction A4 a déjà montré ce que coûte de l'oublier : 0,041 de Sharpe sur trois
ETF non financés.

**5. Le registre d'essais, un seul, et le compte global écrit d'avance.**
`data/trials.parquet` porte 805 lignes et 83 hachages distincts. Les cinq arbres déclarent
**12 (Rentec) + 5 (Bridgewater) + 6 (AHL) + 5 (AQR) + 6 (Two Sigma) = 34 essais
primaires**, plus les 15 sensibilités de Two Sigma si on les compte comme inférentielles,
soit 49. Ce nombre doit être inscrit avant, pas découvert après : un arbre déclaré à
l'avance n'est pas du p-hacking, mais cinq arbres dont on ne compte que celui qui passe
en serait. Règle héritée et à réaffirmer : interdit de re-pré-enregistrer un nul neuf pour
remettre le compteur à zéro.

**6. Le panneau des 49 secteurs et son défaut hérité.** `available_at == period` sur
**100,0 %** des lignes, vérifié. Le magasin n'encode aucun délai de parution ;
`kenfrench.py` argumente ce choix pour des features de contexte, l'argument est plus
faible pour des rendements de portefeuille qui servent de signaux. À corriger ou à
déclarer **une fois**, pas deux fois par deux plans.

Ce qui **ne** se mutualise **pas**, et il faut le dire aussi : les cinq détecteurs.
BOCPD sur ER63, CUSUM sur distance de matrices, HMM à la minute, k-means sur contexte
orthogonalisé, quadrant SPF — ce sont cinq latentes différentes, et c'est précisément
l'axe 1. Chercher à les unifier reviendrait à refaire le dispositif unique qu'on cherche
à quitter.

---

## 5. La contrainte de puissance, globalement

**Le fait le plus important de cet arbitrage : les cinq plans, sans exception, déclarent
le canal Sharpe indécidable sur leur échantillon et changent de statistique.** Aucun ne
l'a fait par élégance ; les cinq l'ont chiffré.

| plan | ce que le canal Sharpe exigerait | verdict | statistique de repli | marge |
|---|---|---|---|---|
| AHL | MDE corrigé 0,331, effet visé ≤ 0,30 | indécidable, déclaré | justesse directionnelle de panneau, 46 × 6 039 | procédure fixée, nombre mesuré sous la nulle |
| AQR | 0,615 de Sharpe autonome pour la surcouche | indécidable, déclaré | ΔIC conditionnel, 49 × 439 mois | **1,2×** (0,08 contre 0,0666) |
| Two Sigma | 0,338 apparié après Holm | **inconnu**, déclaré inconnu | transfert de rang, 215 épisodes × 20 cellules | non chiffrable avant la donnée |
| Rentec | 0,960-1,252 apparié (1,324-2,164 en autonome) | indécidable, déclaré | déficit d'exécution en points de base | **3,9 à 5,7×** |
| Bridgewater | cellule à 3,34 ans, sous le seuil de 3,924 ⇒ **infini** | indécidable par arithmétique | log-ratio de variance entre cellules | **2,0×** (0,405 contre 0,204) |

Trois conséquences, et elles commandent la façon de lire tout le reste.

**(a) Les cinq ne sont pas départageables sur le Sharpe, quelle que soit leur élégance.**
La référence du programme se reproduit : 23,2 ans résolvent **0,638** de Sharpe autonome,
19,9 ans résolvent 0,701. Aucun des cinq ne vise un effet incrémental de cet ordre. Tout
classement qui les comparerait sur un Sharpe attendu comparerait des quantités que
l'échantillon ne peut pas produire. C'est pourquoi la section 1 classe sur la probabilité
d'un mécanisme mesuré par jour, et pas sur autre chose.

**(b) Les cinq statistiques de repli ne sont pas commensurables entre elles.** Un point
de base de coût d'exécution, un log-ratio de variance, un ΔIC et un point de taux de
réussite de panneau ne se convertissent pas les uns dans les autres. **Aucun PASS dans un
canal ne peut être échangé contre un FAIL dans un autre**, et cela doit être écrit avant
les mesures, pas après, parce que c'est exactement le re-cadrage qu'on serait tenté de
faire à la fin.

**(c) Ce qui fait la puissance, ce n'est pas la durée, c'est la dimension.** Les
échantillons vont de 5,6 à 36,6 ans et aucun ne suffit en Sharpe, parce qu'un Sharpe vit
sur T seul. Les quatre replis qui fonctionnent vivent tous sur autre chose : un panneau
N × T (AHL, AQR, Two Sigma), une variance (Bridgewater), ou une grandeur mesurée à haute
fréquence (Rentec). Le programme l'avait déjà démontré sur lui-même sans en tirer la
conséquence : son seul résultat positif — +3,93 points de R² incrémental, t −3,40 — est
une statistique de panneau, et les six réfutations sont toutes des différences de Sharpe
de portefeuille. **Les six dispositifs n'ont peut-être pas échoué seulement parce que
l'objet était vide ; ils ont tous été mesurés avec l'instrument le moins puissant dont le
programme dispose.** Ce n'est pas une excuse rétrospective — les écarts observés (+0,068
contre un MDE de 0,246, +0,014 contre 0,043) restent petits même relativement — mais
c'est une contrainte de conception pour la suite, et elle est mesurée.

---

## 6. Ce que le programme doit retirer de ses conclusions

### Ce qui est écrit aujourd'hui, et qui va trop loin

`pilotage/feuille_de_route/CONCLUSIONS.md` §5 : « **Le régime ne marchera pas.** Cinq
dispositifs, cinq réfutations, une cause commune mesurée — treize transitions en
vingt-cinq ans donnent treize décisions, ce qui n'est pas un signal de trading. **Rouvrir
ce dossier serait un sixième essai qui déflaterait les cinq premiers sans rien
apporter.** »

`pilotage/feuille_de_route/TACHES.md` §A4 : « **LA LIGNE RÉGIME SE FERME.** »

La première phrase de chaque citation est une généralisation que les mesures ne
soutiennent pas. La seconde partie de la citation de CONCLUSIONS est en plus devenue
fausse dans sa forme : cinq dispositifs déclarés à l'avance, sur des axes que les six
n'ont jamais fait varier, avec un registre qui compte déjà 83 hachages et peut en
absorber 34 de plus, ne « déflatent » rien — c'est un arbre non déclaré qui déflate.

### La phrase juste, telle qu'elle devrait remplacer l'autre

> **Un classifieur d'états ordonné par la volatilité d'entraînement, qui change d'état
> 0,514 fois par an — 13 transitions en 6 377 séances, recompté le 22 septembre 2026 —
> utilisé comme multiplicateur de taille ou comme interrupteur au niveau du portefeuille,
> sur un livre unique de tendance 12-1 dont la dimension effective est de 3,86 pour 11
> signaux candidats, n'apporte rien au-delà d'une volatilité réalisée sous sa médiane
> expansive. Six dispositifs l'ont testé ; chacun est soit battu par ce témoin d'une
> ligne, soit sous-puissant contre sa propre résolution.**
>
> **Ces six mesures ne disent rien d'un état à plus d'une transition par an, rien d'une
> latente qui ne soit pas la volatilité, rien d'un usage de sélection entre signaux, de
> construction de portefeuille ou d'exécution, et rien d'un objet conditionné de
> dimension effective supérieure à quatre.**
>
> **Et sur cet échantillon, la question ne se tranche pas en Sharpe : 23,2 ans résolvent
> 0,638 de Sharpe autonome, et les cinq angles instruits visent tous des effets
> inférieurs. Ce qui reste décidable se mesure sur un panneau, sur une variance ou sur un
> coût.**

Ce qui reste vrai et ne bouge pas : les six réfutations, une par une, avec leurs chiffres.
Le classifieur qui marche comme classifieur (93,2 %, kappa 0,53). Le fait qu'il porte la
variance et pas la moyenne (+3,93 points de R², t −3,40 contre +0,030, t 0,27). Et
« treize transitions en vingt-cinq ans donnent treize décisions », qui est la bonne phrase
— ce qui doit changer, c'est la portée de ce qu'on en déduit.

### Deux points à verser à la liste d'hygiène, vérifiés ici

- **B11.** `data/raw/panels/factors_5.parquet` ne contient pas `ff_mom`.
  `regime_lab/features/crosssection.py` boucle sur `("ff_smb","ff_hml","ff_mom")` derrière
  un garde `if name in factors` : `xs_ff_mom_63` n'a jamais été construite, sans erreur ni
  avertissement. Même mode de défaillance que B1 et B4 — un code qui promet une sortie
  qu'il ne produit pas.
- **B12.** Le départ N ≥ 30 : `PRESPEC_TREND_VEHICLE.md:118` et `CONCLUSIONS.md` §3.1
  écrivent 2003-07-17 ; la lecture permissive de la fenêtre roulante donne 2003-07-16 et
  6 040 séances. La lecture stricte (252 séances antérieures) donne 2003-07-17 et 6 039,
  et c'est la causalement correcte. Amender au journal avant que la prochaine étude
  n'hérite de l'ambiguïté.

### Un point à transmettre à l'autre session

Les chiffres du gate-0 crypto du dépôt privé voisin (médiane 3,4 bp, p90 11, p99 27) sont
lus depuis un bac à sable distant qui n'existe plus. La seule copie locale a une
colonne `spread` identiquement nulle (max 0 sur 4,4 M de lignes). **Ces chiffres ne sont
pas reproductibles.** Le patrimoine à la minute (27 instruments, 8,8 Go, colonne de
spread flottant réel, vérifié ici) est le bon substrat, et c'est ce que le plan Rentec
propose.
