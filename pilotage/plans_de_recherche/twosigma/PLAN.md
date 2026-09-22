# Two Sigma — le régime comme sélecteur contextuel de signaux

Plan de recherche pré-enregistrable. Rédigé le 22 septembre 2026, **avant toute mesure
de rendement sur l'angle**. Chaque chiffre de ce document vient d'une commande
exécutée ; les scripts sont à côté, dans le même dossier, et se relancent tels quels
avec `regime-lab/.venv/bin/python`.

Ce qui a été mesuré ici : des dates, des comptages de transitions, des corrélations de
**vecteurs de position**, des rotations, et des erreurs-types de bootstrap sur des
séries **délibérément centrées**. Aucun Sharpe, aucune moyenne, aucun rendement à
terme n'a été multiplié par un signal. Le détail du protocole d'aveuglement est en
§ c).

---

## a) Test d'admission

L'approche diffère des six réfutations sur **les quatre axes**, dont un de façon
décisive. Chiffré :

| axe | les six réfutations | ici | écart mesuré |
|---|---|---|---|
| **3 · usage** | dimensionnement, ou interrupteur ON/OFF | **sélection entre signaux** : quelle part du budget va à quel signal | question qui n'existe pas avec un livre unique |
| **4 · objet** | un livre : TSMOM 12-1 sur 46 instruments | **bibliothèque de 11 signaux** sur 49 portefeuilles sectoriels | rang effectif **8,37** contre 1 |
| **1 · variable latente** | états ordonnés par la volatilité d'entraînement (`mapping.py`) | partition k-means sur un espace de contexte **orthogonalisé contre la volatilité réalisée** | η²(vol réalisée) = **0,048** contre un ordonnancement par la volatilité |
| **2 · constante de temps** | A′ sparse jump : **0,532** transition/an | k-means K=4 sur le contexte : **8,31**/an orthogonalisé, **6,46**/an brut | **×15,6** |

L'axe 3 est le vrai. Les axes 1, 2 et 4 sont des conditions nécessaires pour que
l'axe 3 soit posable, et chacun est chiffré plutôt qu'affirmé.

**Mais l'admission n'est acquise que sur un univers, et pas sur celui que le programme
attendait.** C'est le résultat le plus important de cette instruction, et il est
négatif sur la piste évidente.

### La bibliothèque de signaux : ce que j'ai trouvé, et ce que j'ai dû écarter

La mission demandait de chiffrer l'indépendance par la corrélation moyenne par paires
plutôt que par le compte brut. Fait, en **espace de position** : chaque signal produit
un vecteur de poids transversaux par séance ; la corrélation entre deux signaux est la
corrélation transversale de leurs vecteurs de poids, moyennée sur le temps. Aucun
rendement à terme n'entre dans ce calcul. Le rang effectif est le rapport de
participation des valeurs propres, `(Σλ)² / Σλ²`.

**Univers des 46 instruments — ne supporte pas l'approche.**
`data/cache/trend_universe_m1.parquet`, 11 signaux construits (tendance 12-1, 3 mois,
1 mois ; retournement 5 j et 1 j ; deux cassures de Donchian ; momentum et
retournement transversaux ; low-vol transversal ; retour à la moyenne à un an) :

```
signaux 11 | corr. moyenne -0,001 | corr. |moyenne| 0,336 | max |corr| 0,990
rang effectif 3,86 | valeurs propres [4,82 2,28 1,10 1,00 0,60 0,47 0,33 0,21 0,13 0,07 0,01]
6 composantes pour 90 % de la trace
paires les plus liées : TSMOM_1M/XSREV_1M −0,99 · TSMOM_12_1/XSMOM_12_1 +0,93 · TSMOM_3M/BRKOUT_100 +0,84
```

C'est exactement le cas que la mission demandait de déclarer : **trois ou quatre
directions, fortement corrélées**. La première valeur propre pèse 4,82 sur 11 — c'est
un seul facteur de tendance décliné à plusieurs horizons. Un sélecteur n'a rien à
sélectionner. **L'approche Two Sigma n'est pas transposable sur l'univers des 46
instruments, et le dire est le premier livrable.**

Cela éclaire rétrospectivement une partie des six échecs : l'objet conditionné n'avait
pas une dimension, il en avait environ quatre, dont trois étaient la même chose.

**Panel des 49 secteurs — supporte l'approche.**
`data/raw/panels/industry_49.parquet`, 12 signaux (momentum 12-1 et 6-1 ; retournement
1 mois et 1 semaine ; retournement long terme 60-13 ; low-vol ; momentum de
volatilité ; accélération ; skew ; low-beta ; low-corrélation ; saisonnalité
Heston-Sadka à 10 ans) :

```
corr. moyenne +0,013 | corr. |moyenne| 0,101 | max |corr| 0,912
rang effectif 8,37 | valeurs propres [2,45 1,70 1,52 1,32 1,18 1,00 0,84 0,80 0,54 0,50 0,10 0,05]
8 composantes pour 90 % de la trace
```

`IND_ACCEL` corrèle à **−0,912** avec `IND_MOM_12_1` : redondance de construction, elle
sort avant tout contact. **La bibliothèque retenue compte 11 signaux pour un rang
effectif d'environ 8.** Ce n'est pas « des milliers » ; c'est assez pour qu'un
sélecteur ait quelque chose à arbitrer, et le programme ne l'avait jamais mesuré.

**Conséquence de cadrage.** Le sujet de cette étude n'est pas le livre de tendance.
C'est le panel sectoriel, qui n'a jamais servi qu'à fabriquer des *features* de
dispersion (`xs_dispersion_ind`) et jamais de bibliothèque de signaux.

---

## b) La donnée — vérifiée en l'ouvrant

Tout est déjà sur disque. Rien à télécharger.

| fichier | forme | période | obs | fréquence |
|---|---|---|---|---|
| `data/raw/panels/industry_49.parquet` | 451 388 × 4, long | 1990-01-02 → **2026-07-31** | 9 212 par série × 49 | quotidienne |
| `data/raw/panels/size_bm_25.parquet` | 230 300 × 4 | 1990-01-02 → 2026-07-31 | 9 212 × 25 | quotidienne |
| `data/raw/panels/factors_5.parquet` | 55 272 × 4 | 1990-01-02 → 2026-07-31 | 9 212 × 6 (dont `ff_rf`) | quotidienne |
| `data/cache/features.parquet` | 9 572 × 50 | 1990-01-01 → 2026-09-08 | — | quotidienne |
| `data/cache/states.parquet` | 9 007 × 5 | 1992-03-02 → 2026-09-08 | — | quotidienne |
| `data/cache/trend_universe_m1.parquet` | 6 822 × 46 | 2000-07-17 → 2026-09-10 | — | quotidienne |

**Qualité du panel sectoriel, vérifiée et non supposée** : 0 sentinelle −99,99, 0
sentinelle −999, **100,00 % des séances ont les 49 colonnes renseignées**. Aucun
nettoyage n'est nécessaire, ce qui est rare et mérite d'être dit.

**Taux sans risque pour les rendements en excess** : `ff_rf`, 9 212 observations,
**aucun NaN**, moyenne 0,0107 %/jour soit 2,69 %/an. Il vient du même fichier que les
portefeuilles, donc l'excess est cohérent par construction. `rate_cash_3m.parquet`
(9 177 obs, 1990-01-02 → 2026-09-08) est disponible comme contrôle.

**Échantillon utile après rodage** : le signal le plus long est le retournement long
terme (1 260 séances). L'intersection où les 11 signaux existent simultanément fait
**7 946 séances, du 1995-01-04 au 2026-07-31, soit 31,57 ans**. À comparer aux 23,2 ans
de l'échantillon verrouillé du programme : **+8,4 ans**, et c'est de la puissance
gratuite.

**Ce qui manque, et que je ne recommande pas d'aller chercher.** Il n'y a sur disque
ni structure par terme de futures, ni donnée d'options : les familles *carry* et
*portage de volatilité* que la mission cite comme pistes **n'existent pas** et ne
peuvent pas être construites. C'est une des raisons de la pauvreté de l'univers des 46.

**Ce qui manque et qui est gratuit — testé, pas supposé.** Quatre fichiers quotidiens
de la bibliothèque Ken French répondent aujourd'hui en `HTTP 200`, vérifié par
`curl -sIL` le 2026-09-22 : `F-F_Momentum_Factor_daily_CSV.zip`,
`F-F_ST_Reversal_Factor_daily_CSV.zip`, `F-F_LT_Reversal_Factor_daily_CSV.zip`,
`49_Industry_Portfolios_daily_CSV.zip`. Ils ajouteraient trois livres tout faits.
**Le plan ne s'appuie sur aucun d'eux** : ce qui est sur disque suffit, et je ne
retélécharge rien.

**Un défaut hérité, déclaré.** `available_at == period` sur **100,0 %** des lignes du
panel sectoriel. `kenfrench.py` documente ce choix et l'argumente — la dispersion du
jour est observable depuis n'importe quel flux de prix, la bibliothèque n'étant qu'un
proxy historique propre. J'hérite de ce jugement pour les *features* de contexte. Pour
les **rendements de portefeuille** qui servent ici de signaux, l'argument est plus
faible : un rendement sectoriel du jour *t* est calculable le soir de *t*, mais le
fichier n'est téléchargeable qu'avec des semaines de retard. La règle T-1 couvre la
connaissance, pas la disponibilité du fichier. C'est le seul point où ce plan est plus
optimiste qu'un pupitre réel, et il est écrit ici plutôt que découvert ensuite.

---

## c) La puissance a priori

Calculée avec `regime_lab/analysis/power.py`, bootstrap stationnaire par blocs.

**Protocole d'aveuglement.** Les deux jambes sont **centrées** (moyenne retirée) avant
d'entrer dans `minimum_detectable_sharpe_difference`. Le champ `observed` vaut donc
`+0,0000` par construction — vérifié dans la sortie — et seule l'erreur-type sort,
qui ne dépend que des moments d'ordre deux. Je ne peux pas lire de performance
là-dedans, et c'est voulu.

**Le plus petit effet détectable dépend d'abord de la forme du sélecteur.** C'est la
décision de conception la plus importante du plan :

| forme du sélecteur | MDE (α 0,05, puissance 0,80, bloc moyen 63) |
|---|---|
| bascule dure, un signal à la fois | **0,474** |
| inclinaison des poids, écart-type 1,00 | 0,318 |
| **inclinaison, écart-type 0,50** | **0,221 – 0,294**, médiane **0,241** sur 8 tirages |
| inclinaison, écart-type 0,25 | 0,147 |

Un seul tirage donne 0,238 ; huit tirages donnent 0,221 à 0,294. **Mon premier chiffre
était le bas de la fourchette et le présenter seul aurait été trompeur** — c'est ma
propre erreur, corrigée ici.

La bascule dure, qui est la lecture littérale de « désactiver les stratégies exposées
négativement », coûte **le double** en résolution. Le plan retient donc l'inclinaison,
et le déclare avant mesure.

**Sensibilité à α, pour l'arbre entier** (même construction, bloc 63) :

| α | MDE |
|---|---|
| 0,05 | 0,272 |
| 0,05 / 6 = 0,00833 — les 6 tests primaires | **0,338** |
| 0,05 / 21 = 0,00238 — les 21 évaluations | 0,376 |

**Échelle temporelle** (sur la version bascule dure) : 10 ans → 0,868 · 15 ans → 0,669
· 20 ans → 0,584 · 23,2 ans → 0,559 · 31,3 ans → 0,478. La référence du programme,
0,639 pour 23,2 ans, se retrouve dans le même ordre de grandeur.

**Le verdict de puissance, écrit maintenant.** Le seuil à franchir est
**0,338 de Sharpe annualisé d'écart apparié**, après correction familiale. C'est
beaucoup pour un simple sélecteur. **Je ne prétends pas que l'effet visé soit
au-dessus** : je ne le sais pas et je ne peux pas le savoir sans toucher la donnée.
Ce que je peux faire — et que le plan fait — c'est déplacer la décision vers une
statistique mieux dimensionnée que l'écart de Sharpe du portefeuille.

**Cette statistique est le transfert de rang hors échantillon.** Sur l'échantillon de
la bibliothèque, la partition K=4 produit **215 épisodes** (46 / 71 / 20 / 78 par état),
de durée médiane 8 séances et moyenne 37,0. Avec 11 signaux et 4 états, cela fait
**44 cellules**, l'état le plus rare en recevant 20 épisodes. Un Spearman entre le
classement des signaux dans un état sur les plis d'entraînement et leur classement
dans le même état sur le pli de test, empilé sur 5 plis × 4 états = **20 cellules**,
porte beaucoup plus d'information que 31 ans d'une seule différence de Sharpe. C'est
par là que cet angle échappe au piège de puissance de H-b, dont la paire ne résolvait
que 0,043 et qui n'avait que 13 décisions à observer.

---

## d) Les coûts, et le seuil de mort écrit avant

Barème, conforme au programme : **actions 5 pb aller-retour en réaliste, 10 pb en
conservateur, 20 pb en stress**. Rotation mesurée en espace de position, `|Δw|`
sommé transversalement, annualisé.

| signal | rotation ×/an | coût à 5 pb | à 10 pb |
|---|---|---|---|
| IND_REV_1W | **156,7** | **7,83 %/an** | 15,67 %/an |
| IND_REV_1M | 76,1 | 3,81 %/an | 7,61 %/an |
| IND_MOM_6_1 | 33,4 | 1,67 %/an | 3,34 %/an |
| IND_SKEW | 31,0 | 1,55 %/an | 3,10 %/an |
| IND_VOLMOM | 23,3 | 1,16 %/an | 2,33 %/an |
| IND_MOM_12_1 | 21,8 | 1,09 %/an | 2,18 %/an |
| IND_SEASON | 16,6 | 0,83 %/an | 1,66 %/an |
| IND_LTREV | 10,5 | 0,52 %/an | 1,05 %/an |
| IND_LOWVOL | 10,4 | 0,52 %/an | 1,04 %/an |
| IND_LOWCORR | 7,5 | 0,37 %/an | 0,75 %/an |
| IND_LOWBETA | 3,9 | 0,19 %/an | 0,39 %/an |
| **mélange équipondéré — le témoin** | **21,6** | **1,08 %/an** | **2,16 %/an** |

**`IND_REV_1W` sort avant tout contact.** 7,83 %/an au barème réaliste, c'est-à-dire
dix fois ce qu'un retournement hebdomadaire sectoriel peut raisonnablement produire
brut. L'exclure après avoir vu son rendement aurait été un choix ; l'exclure ici est
une contrainte. La bibliothèque inférentielle passe donc à **10 signaux**.

**Le surcoût du sélecteur lui-même, mesuré :**

| inclinaison | rotation totale | surcoût vs témoin | à 5 pb | en Sharpe à 10 % de vol cible |
|---|---|---|---|---|
| sd 0,25 | 20,6 ×/an | −1,0 ×/an | −0,049 %/an | négligeable |
| **sd 0,50** | 24,2 ×/an | **+2,6 ×/an** | **+0,129 %/an** | **+0,013** |
| sd 1,00 | 31,8 ×/an | +10,2 ×/an | +0,512 %/an | +0,051 |

**Le seuil de mort, écrit avant la mesure.** Le surcoût de la surcouche atteint le MDE
de 0,241 lorsque `2,6 × bps / 10 000 / 0,10 ≥ 0,241`, soit **bps ≥ 92,7**. Il faudrait
un coût aller-retour de 93 points de base — dix fois le barème actions — pour que le
sélecteur meure de ses propres frais. **Le coût n'est pas la contrainte qui mord sur
cet angle ; la puissance l'est.** C'est une chose utile à savoir d'avance, et elle
ferme par la mesure une explication paresseuse qu'on pourrait invoquer après coup.

**Règle d'arrêt sur coût, pré-déclarée.** Si la carte ajustée s'avère plus agitée que
la carte aléatoire et que son **surcoût de rotation dépasse 12 ×/an** — niveau auquel
le barème de stress à 20 pb consomme 0,24 de Sharpe, soit le MDE entier — la surcouche
est déclarée morte sur le coût, quel que soit son écart brut.

---

## e) Le placebo apparié

Obligatoire, et il m'a fallu trois tentatives. Les deux premières sont fausses et
je les laisse écrites.

Le placebo doit apparier sur **la constante de temps et l'occupation**, parce que c'est
exactement ce que T3 a montré : un gain apparent peut passer entièrement par une
dimension qu'on n'a pas neutralisée.

| tentative | transitions/an (réel 6,46) | écart max d'occupation |
|---|---|---|
| 1 · permutation des épisodes `(durée, état)` | **4,79** — perd 29 % de l'horloge par jonctions fusionnées | 0,0000 |
| 2 · permutation indépendante des durées et des états | 6,77 — horloge bonne | **0,3043** — occupation détruite |
| 3 · **permutation des paires, réparation par échange** | **6,46 — exact** | **0,0000 — exact** |

La troisième garde les paires `(durée, état)` intactes, donc l'occupation est exacte
par construction, et répare les jonctions où deux épisodes voisins partagent un état
en **échangeant des paires entières**, ce qui ne peut pas altérer le multiensemble.
Résultat sur 300 tirages : **223 transitions dans chacun**, écart-type 0,0, déviation
d'occupation 0,0000, **300 tirages acceptés sur 300**.

**Trois contrôles, pas un.**
1. **Placebo d'horloge** — 1 000 partitions appariées sur l'horloge et l'occupation,
   sans contenu. Fournit la distribution nulle de la statistique de transfert de rang.
2. **Témoin équipondéré** — le mélange à poids fixes, qui est le vrai adversaire.
3. **Contrôle de bêta, la leçon de T3** — le bêta réalisé du sélecteur contre le
   marché (`ff_mkt-rf`) est reporté en percentile de la distribution placebo. Un
   sélecteur qui bat le témoin en se plaçant au 95ᵉ percentile de bêta n'a rien
   sélectionné ; il a acheté de l'exposition, à 0,2785 l'unité selon l'étude de
   barrière.

**Le seuil du placebo est le 99ᵉ percentile, pas le 95ᵉ.** Motif chiffré, et c'est le
programme lui-même qui le fournit : une méthodologie d'indice testée contre 160 000
combinaisons aléatoires place la version officielle au **98ᵉ** percentile. Le 95ᵉ est
précisément l'endroit où atterrit une carte qu'on a cherchée. Exiger le 99ᵉ coûte de
la puissance et c'est le prix de la crédibilité.

---

## f) Les tueurs connus, et le sens dans lequel ils poussent

1. **Le data-mining sur la carte régime → signal.** Le tueur principal, nommé comme
   tel par la mission. 10 signaux × 4 états = 40 cellules ; avec K ∈ {3,4,5,6} et le
   choix de l'espace de contexte, l'espace de recherche est grand. **Pousse vers un
   faux PASS.** Neutralisé par trois choses simultanées : la carte n'est ajustée que
   sur les plis d'entraînement, les plis de test ne la voient jamais ; le nul vient du
   placebo apparié sur l'horloge et l'occupation ; le seuil est le 99ᵉ percentile.
2. **Le canal de la moyenne est déjà mesuré vide.** +0,030 point de R² sur les
   rendements à terme, t 0,27. Le niveau A demande justement au contexte de prédire une
   moyenne. **Pousse le niveau A vers l'échec.** Ce n'est pas un biais, c'est un a
   priori, et il est la raison pour laquelle A n'est pas le seul niveau de l'arbre.
3. **La quasi-orthogonalité de la bibliothèque plafonne le niveau B.** Corrélation
   moyenne **signée** +0,013 sur 11 signaux : la part de la variance du mélange qui
   transite par la matrice de corrélation est `11×10×0,013 / (11 + 1,43)` =
   **11,5 %**. Les 88,5 % restants sont la diagonale. **Pousse vers un effet petit sur
   le canal corrélation** — et dicte que B conditionne d'abord la **diagonale**, la
   volatilité par signal et par état, qui est le canal où +3,93 points de R² sont
   établis.
4. **Le compromis horloge / bruit.** Une horloge à 8,31/an donne 288 épisodes mais de
   durée médiane 4 à 8 séances : chaque cellule est estimée sur peu de données.
   **Pousse vers le bruit.** Un lissage à 21 séances ramène l'horloge à 2,64/an et
   rallonge les épisodes ; il est déclaré comme **sensibilité pré-enregistrée**, pas
   comme molette.
5. **`available_at == period` sur 100 % des lignes.** Le magasin n'encode aucun délai
   de publication. **Pousse vers l'optimisme.** Déclaré en § b).
6. **Univers purement actions américaines.** Les 11 signaux partagent les 49 mêmes
   jambes et le même résidu de bêta marché. **Pousse vers un gain qui passe par le
   bêta**, exactement comme T3. C'est la raison d'être du troisième contrôle.
7. **Les jambes à rotation rapide.** `IND_REV_1W` à 156,7 ×/an. **Pousse vers un brut
   flatteur et un net mort.** Exclu avant contact, en § d).
8. **L'univers des 46 instruments ne peut pas porter cette étude.** Rang effectif 3,86.
   Si quelqu'un veut l'y refaire, la réponse est déjà écrite : non transposable.

---

## g) L'arbre d'escalade

Trois niveaux, chacun avec son critère chiffré, et une raison explicite pour laquelle
le niveau suivant échappe à ce qui a tué le précédent. **L'arbre entier est déclaré
avant la première mesure ; c'est ce qui le distingue du tripotage.**

Cadre commun à tous les niveaux : signal en T-1, trade en T · rendements en excess de
`ff_rf` · HAC lag 6 · walk-forward à **5 plis de test de 4,0 ans**, entraînement en
fenêtre extensive dont le premier fait 11,6 ans · ciblage de volatilité quotidien à
10 % · bootstrap stationnaire par blocs, blocs moyens 21 / 63 / 126 · N = 49 jambes,
au-dessus du seuil de 30.

### Niveau A — le canal de la moyenne : la transposition littérale

Partition k-means K=4 sur l'espace de contexte à 20 *features* orthogonalisé contre la
log-volatilité réalisée, réajustée à chaque pli. Carte état → poids estimée sur
l'entraînement seul, appliquée en test comme **inclinaison d'écart-type 0,50** autour
de l'équipondération.

**Falsification, deux verrous, les deux doivent céder :**
- écart de Sharpe apparié hors échantillon contre le mélange équipondéré
  **< 0,338** (MDE à α = 0,05/6) ⇒ ÉCHEC, et si l'écart est sous le MDE il est déclaré
  **UNDERPOWERED, jamais PASS** ;
- statistique de transfert de rang (Spearman entraînement → test, 20 cellules) sous le
  **99ᵉ percentile** du placebo apparié ⇒ ÉCHEC.

**P(succès) ≈ 12 %.** Faible et écrit d'avance, parce que A demande au contexte de
prédire une moyenne et que le programme a mesuré que ce canal est vide. A doit être
tourné quand même : c'est la méthode documentée de Two Sigma, et la sauter reviendrait
à supposer la réponse.

### Niveau B — le canal de la variance : la vraie chance

**Si A tombe, il tombe sur le premier moment.** B ne le touche jamais. La carte
n'estime plus « quel signal rapporte le plus dans l'état k » mais **la matrice de
covariance entre signaux, état par état**, qui alimente la construction du mélange
(parité de risque sur la covariance conditionnelle à l'état). C'est une question de
second moment, et le second moment est précisément l'endroit où le classifieur a
**+3,93 points de R² incrémental, t −3,40**, établis contre un quantile de volatilité.

Décomposition pré-calculée, qui dicte la conception : **88,5 % de la variance du
mélange est diagonale**. B conditionne donc d'abord la volatilité par signal et par
état, la corrélation ensuite.

**Falsification, deux verrous :**
- critère de prévision de covariance hors échantillon — la covariance conditionnelle à
  l'état doit battre une covariance unique poolée sur une perte de dispersion réalisée,
  au-delà du **99ᵉ percentile** du placebo apparié ⇒ sinon ÉCHEC ;
- écart de Sharpe apparié contre le mélange à covariance poolée **< 0,338** ⇒
  UNDERPOWERED, jamais PASS.

**P(succès) ≈ 25 %.** Plus haut que A parce que le canal est celui où le classifieur a
du contenu mesuré ; plafonné parce que la bibliothèque est déjà quasi orthogonale.

### Niveau C — le canal du coût, dont le verdict est déjà partiellement écrit

**Si B tombe, les deux moments du rendement sont épuisés.** Il ne reste que le
registre des frais : conditionner la **cadence de rééquilibrage** à l'état du contexte,
échanger moins dans les états où les signaux se décomposent vite.

**Et je peux pré-calculer que C est indécidable au niveau du portefeuille.** Le coût
total du mélange équipondéré est de 1,08 %/an à 5 pb, soit **0,108 de Sharpe** à 10 %
de volatilité cible. Même en supprimant **tout** le coût, le gain maximal
structurellement disponible est 0,108 — **sous le MDE de 0,241, et très sous les 0,338
corrigés**. Aucun résultat de Sharpe n'est atteignable sur ce niveau, et le découvrir
après six jours de travail serait une perte.

**C est donc pré-déclaré comme test de rotation, pas de Sharpe.** Critère : différence
appariée de rotation annuelle contre le témoin, dont le MDE est d'un tout autre ordre.
Traduction en Sharpe déclarée d'avance (`Δrotation × bps / 10 000 / 0,10`) et reportée
**comme borne, jamais comme PASS**.

**P(succès sur la statistique de rotation) ≈ 40 %. P(traduction en un gain de Sharpe
décidable) ≈ 0 %, par le calcul ci-dessus.**

### Fermeture, si les trois tombent

Motif à écrire, et il est étroit : *« une partition de contexte à 6–8 transitions par
an, orthogonale à la volatilité, ne transfère hors échantillon comme sélecteur sur une
bibliothèque de 10 signaux actions ni sur le premier ni sur le second moment, sur
31,6 ans, à une résolution de 0,338 de Sharpe. Le seul canal restant est le coût, et le
coût total du livre — 0,108 de Sharpe — est trop petit pour être décidable ici. »*

C'est une phrase qu'on peut défendre. Ce n'est pas « les régimes ne se monétisent pas ».

### Correction pour tests multiples, chiffrée d'avance sur l'arbre entier

**21 évaluations déclarées** : 6 tests primaires (3 niveaux × 2 verrous), 9
sensibilités sur K ∈ {3,5,6} (3 par niveau), 6 sensibilités sur l'inclinaison
∈ {0,25 ; 1,00} (2 par niveau).

- **Holm-Bonferroni sur les 6 primaires**, α familial 0,05, seuil le plus strict
  0,00833. **MDE correspondant 0,338**, contre 0,272 non corrigé.
- Les 15 sensibilités sont reportées et **ne peuvent jamais fonder un PASS** ; elles
  entrent au journal `trials.parquet` et donc dans la déflation.
- Seuil placebo au **99ᵉ percentile** pour absorber la recherche sur la carte.
- Si l'on voulait couvrir les 21 comme inférentiels, le MDE monterait à **0,376** —
  chiffré ici pour que le choix de n'en retenir que 6 soit visible et contestable.

---

## h) Ce qu'on apprend même si tout l'arbre tombe

1. **La dimension de la bibliothèque du programme, mesurée pour la première fois.**
   8,37 effectifs sur 12 pour le panel sectoriel ; **3,86 sur 11 pour l'univers des 46
   instruments**. Cela explique rétrospectivement une partie des six échecs : l'objet
   conditionné n'avait pas une dimension mais environ quatre, dont trois étaient la
   même tendance à des horizons différents. Résultat méthodologique publiable quel que
   soit le verdict de l'arbre.
2. **Une phrase étroite à la place d'une généralisation.** Le programme gagne un
   énoncé daté, chiffré et borné, au lieu de « ça ne se monétise pas ».
3. **Un placebo réutilisable.** La permutation d'épisodes appariée exactement sur
   l'horloge *et* sur l'occupation (223/223 transitions, 0,0000 d'écart d'occupation,
   300 tirages sur 300) est une brique que le programme n'avait pas, et les deux
   versions fausses sont documentées pour qu'on ne les refasse pas.
4. **La géométrie de coût de toute surcouche de sélection, fermée par la mesure.**
   +0,013 de Sharpe à 5 pb ; morte seulement au-delà de 93 pb aller-retour. Un échec
   futur d'une surcouche de sélection dans ce programme ne pourra plus être imputé aux
   frais.
5. **Le fait négatif sur l'univers des 46.** Non transposable, chiffré. Cela économise
   la prochaine tentative.

---

## Effort

| poste | jours |
|---|---|
| construction de la bibliothèque, exclusions, tests unitaires | 2,0 |
| module de contexte, réajustement walk-forward, placebo apparié | 2,0 |
| niveau A | 1,5 |
| niveau B | 2,0 |
| niveau C | 1,0 |
| rédaction, pré-enregistrement, amendement au registre de protocole | 1,5 |
| **total** | **10,0** |

**P(l'arbre produise au moins un PASS inférentiel) ≈ 30 %** — soit `1 − 0,88 × 0,75`,
arrondi vers le bas, le niveau C n'y contribuant pas par construction.

**P(l'arbre produise un résultat rédigeable) = 1**, par le § h).

---

## Ce que ce plan ne fait pas

Il ne construit aucune stratégie et n'a regardé aucun rendement de backtest. Les
corrélations sont des corrélations de vecteurs de poids ; les erreurs-types viennent de
séries centrées dont le champ `observed` est vérifié à `+0,0000` ; les rotations sont
des `|Δw|`. Ce qui distingue un signal d'un autre en régime — la seule chose qui
déciderait l'arbre — n'a pas été mesuré et ne doit pas l'être avant que le
pré-enregistrement soit verrouillé.

## Scripts

Tous dans ce dossier, relançables tels quels.
`inspect1.py` `inspect2.py` `inspect3.py` `inspect4.py` — inventaire disque et dates.
`measure_clock.py` — horloge du contexte brut. `measure_clock2.py` — horloge
orthogonalisée. `measure_library.py` — indépendance du panel sectoriel.
`measure_library2.py` — indépendance de l'univers des 46. `measure_power.py` /
`measure_power2.py` / `measure_fix.py` — MDE aveuglé. `measure_costs.py` — rotation et
coûts. `measure_amtp.py` — MDE corrigé. `measure_placebo.py` — placebo apparié.
Sorties : `clock.csv`, `clock_resid.csv`, `corr_industries.csv`, `corr_multiasset.csv`.
