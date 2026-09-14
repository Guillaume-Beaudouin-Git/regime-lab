# Le régime branché sur la géométrie de barrière

Piste n° 2 du document de passation, jamais exécutée, et la seule application du
résultat central qui ne soit pas déjà réfutée. Sortie : **Δp(pass) et Δdurée**.
Aucun Sharpe n'est rapporté comme résultat.

Tout est produit par un seul fichier, `regime_barrier.py`, exécuté dans le venv
du dépôt. Aucune écriture dans `regime-lab` ni dans `Algo_claude` ;
`scripts/run_phase2.py` n'a pas été relancé ; aucune donnée re-téléchargée ;
`docs/CHARTER.html` n'a pas été touché.

---

## 0. La thèse, et ce qui la décide

Un compte propfirm n'est pas un problème de Sharpe. C'est un problème de survie
face à une barrière de drawdown, où la variance décide directement de la
probabilité de toucher. Un signal qui ne prédit que la variance devrait donc
valoir quelque chose ici même s'il ne vaut rien dans un livre classique.

T3 impose la forme du test. Le gain apparent de ces dispositifs passe par le
bêta et par le dénominateur, pas par la moyenne. Une p(pass) qui monte parce que
le livre est moins — ou plus — exposé n'est pas un résultat. D'où trois
obligations tenues ici : l'exposition moyenne des deux côtés est rapportée, le
**prix de l'exposition nue est mesuré** sur un bras sans signal, et le placebo
par quantile de volatilité réalisée est le nul qui décide.

---

## 1. La géométrie réelle, citée et non supposée

`Algo_claude/Portfolio/risk/trailing_dd_simulator.py` — Tradeify Select Flex 25K.
Paramètres cités depuis `PropfirmConfig` :

| paramètre | valeur | en % du capital |
|---|---|---|
| `capital` | 25 000 $ | — |
| `profit_target` | 1 500 $ | 6,0 % |
| `trailing_dd_eval` | 1 000 $ | 4,0 % |
| `trailing_dd_funded` | 1 000 $ | 4,0 % |
| `lock_trigger_excess` | 100 $ | verrou à 26 100 $ |
| `locked_floor_offset` | 100 $ | plancher verrouillé 25 100 $ |
| `consistency_cap_pct` | 40 % | 600 $/jour comptabilisés, **eval seule** |
| `max_eval_days` | 180 | — |
| `funded_days` | 180 | — |

Nature du drawdown, lue ligne par ligne dans `simulate_run()` :

- **Glissant, pas statique.** `floor_e = max(floor_e, equity - trailing_dd)` : le
  plancher monte avec le pic d'equity de fin de journée et ne redescend jamais.
  En phase eval il ne se verrouille **jamais**.
- **Sur equity, pas sur balance.** `equity += pnl`, puis `if equity <= floor_e`.
- Le test de rupture arrive **avant** la mise à jour du plancher du jour : la
  perte nette du jour est traitée comme le pire creux intra-journalier.
- L'objectif exige **deux** conditions simultanées : `cum_capped >= 1500` (somme
  des jours positifs, chacun plafonné à 600) **et** `equity >= 26500`.
- **Contrainte de durée : aucune** dans les règles du prop. 180 jours dans le
  simulateur, puis 180 jours de phase financée.

Le simulateur est **directement utilisable sur un livre quotidien** : il prend un
tableau de P&L quotidien en dollars, jours sans trade inclus. Aucune géométrie
n'a donc été réimplémentée. L'adaptateur, c'est une ligne — `to_usd()`, qui met
le rendement fractionnaire du livre à l'échelle d'un niveau de risque déclaré.

**Vérifié avant usage.** Le docstring du simulateur annonce
« P(pass) ≈ 53.7%, P(bust) ≈ 46.2% (N=20k, seed=42, max_eval_days=180) ».
Obtenu ici : **53,735 % et 46,240 %**. Reproduit.

**Ce que la géométrie EOD coûte**, cité depuis le même docstring : *« It
overestimates survival for strategies with large intraday swings that recover by
EOD. »* Un livre quotidien n'a qu'un prix par jour. La p(pass) rapportée est donc
un **majorant**. Le biais est identique sur les deux bras : il déplace les
niveaux, pas le delta.

`propfirm_canonical.py` déclare une géométrie **différente** comme canonique :
FTMO, perte journalière 5 % du solde de départ, drawdown maximum 10 %
**statique**, cible 10 %, minimum 4 jours de trading. `trailing_dd_simulator.py`
ne peut pas l'exprimer — son plancher monte toujours. Elle passe donc en
**secondaire**, par un adaptateur de vingt lignes écrit ici, signalé comme tel à
chaque mention, et qui n'implémente que les quatre constantes de ce module.

`sizing.py` a été lu : ERC de Maillard-Roncalli-Teiletche et inverse-vol **entre
stratégies**, fenêtre 63 jours, plafond 0,60 par stratégie. Un seul livre ici.
Aucune de ses fonctions n'entre dans ce calcul, et c'est dit plutôt que passé
sous silence.

---

## 2. Le livre, l'échantillon, les coûts

Livre TSMOM de référence de `regime_lab/extensions/trend.py`, non modifié :
46 instruments, 6 822 sessions, 2000-07 à 2026-09, volatilité annualisée
**4,54 %**. Son Sharpe de 0,51 est cité pour identifier le livre, pas comme
sortie de ce travail.

**Le piège maison n'est pas reproduit.** Je n'exige à aucun moment que les 46
instruments aient de la donnée à chaque date : 41 instruments actifs en 2008-01,
46 à la fin, médiane 46, et l'échantillon du livre démarre en 2000-07 et non en
2007. C'est la contrainte qui avait jeté huit ans et fabriqué un t de 2,18 qui
n'était que la troncature.

**Échantillon commun à tous les bras et à tous les nuls : 6 128 sessions,
2003-03-12 à 2026-09-08, contigu.** La contiguïté est vérifiée par une assertion
dans le code, et cette assertion a effectivement sauté au premier passage (voir
§9, erreur 3). La date de départ est fixée par le plus tardif des amorçages
causaux, pas par un choix : toute comparaison se fait dessus, parce que la leçon
de l'amendement du 10/09 est que comparer des constructions mesurées sur des
échantillons différents fabrique des écarts qui sont de l'amorçage.

**Coûts déclarés avant tout résultat**, par classe d'actif, aller-retour, depuis
la table du projet : futures 1 bp, ETF obligataires 7,5 bp (milieu de la bande
actions 5-10 bp), FX 15 bp. Crypto 17 bp, absent de cet univers. Convention :
**la moitié de l'aller-retour déclaré par franchissement**, donc un aller-retour
complet coûte exactement le chiffre déclaré. Répartition : 13 futures et 14
indices actions tradables en futures à 1 bp, 9 ETF obligataires à 7,5 bp, 10
paires FX au comptant à 15 bp.

Mesuré et non supposé : **aller-retour moyen pondéré par la rotation réelle
6,96 bp**, ponction annuelle **0,51 % de rendement** sur le bras à taille
constante. Le coût est prélevé par instrument sur la variation de l'exposition en
dollars, donc il inclut la rotation supplémentaire que crée chaque règle de
dimensionnement.

---

## 3. Ce que j'ai borné, explicitement

Une troncature silencieuse se lit comme une couverture complète. Voici toutes les
bornes.

| ce qui est borné | valeur retenue |
|---|---|
| pseudo-comptes par estimation ponctuelle | **4 000** |
| pseudo-comptes à l'intérieur de chaque tirage de nul | **2 000** |
| bootstrap emboîté pour l'incertitude d'histoire | **300** pseudo-histoires × **300** fenêtres |
| nul 1, dimensionneurs par quantile de volatilité | **200** |
| nul 2, permutations à durée conservée | **400** |
| nul 3, rotations circulaires | **400** |
| nul familial (maximum sur 5 familles) | **150** tirages |
| horizon d'un compte simulé | 360 sessions (`max_eval_days` + `funded_days`) |
| bloc moyen du bootstrap | **63** sessions, sensibilité à 21 et 126 |
| échantillon | 6 128 sessions, un seul livre, un seul univers de 46 instruments |
| géométries | Tradeify (simulateur du dépôt) + FTMO (adaptateur local) |
| familles de régime | les 5 de `states_offline.parquet`, primaire déclarée avant lecture |

Ce que je n'ai **pas** fait : la phase financée n'est pas analysée pour
elle-même, les payouts sont laissés à `no_payout`, et la latence n'est pas
balayée — je prends `states_offline.parquet` tel quel, la version avec recul, la
seule honnête.

Un premier passage complet de bout en bout a tourné à **un quart de ces tirages**
(34 s) avant le passage complet (260 s). Les deux sorties sont conservées et les
verdicts sont identiques dans chaque direction, ce qui est vérifiable et non
affirmé :

| quantité | passage court (¼ des tirages) | passage complet |
|---|---|---|
| Δp(pass) régime apparié | +0,7 pt, MDE 6,7 | **+0,3 pt, MDE 6,1** |
| état − placebo vol | −2,8 pt | **−2,6 pt** |
| état − cible de vol | −2,3 pt | **−2,3 pt** |
| percentile nul 1 | 15ᵉ | **16ᵉ** |
| percentile nul 2 | 63ᵉ | **57ᵉ** |
| percentile nul 3 | 38ᵉ | **30ᵉ** |
| percentile nul familial | 25ᵉ | **17ᵉ** |
| barreaux gagnés sur 12 | 1 | **1** |
| verdict des quatre bras | UNDERPOWERED | UNDERPOWERED |

Tous les chiffres cités ailleurs dans ce rapport viennent du **passage complet**.

Le bootstrap est **stationnaire par blocs (Politis-Romano)**, via
`regime_lab/analysis/bootstrap.py`, jamais iid : les passages de barrière sont
massivement groupés dans le temps. Les deux bras d'une comparaison sont
resamplés sur le **même** chemin d'indices, donc toute comparaison est appariée.

---

## 4. Le niveau de risque, et deux amendements que je dois nommer

La p(pass) dépend du notionnel. Toute l'échelle est donc rapportée, bras constant :

| vol ann. du compte | levier sur le livre | p(pass) | p(bust) | p(timeout) | j. → cible | j. → échec |
|---|---|---|---|---|---|---|
| 1 % | 0,23 | 0,0 % | 0,1 % | 99,9 % | — | 148 |
| 2 % | 0,46 | 0,1 % | 5,9 % | 94,0 % | 135 | 131 |
| 3 % | 0,69 | 3,3 % | 19,7 % | 77,1 % | 135 | 102 |
| **4 %** | **0,91** | **10,9 %** | **35,4 %** | **53,7 %** | **128** | **90** |
| 6 % | 1,37 | 23,6 % | 55,4 % | 21,0 % | 102 | 70 |
| 8 % | 1,83 | 30,6 % | 62,4 % | 7,0 % | 79 | 50 |
| 10 % | 2,28 | 32,5 % | 65,6 % | 1,9 % | 58 | 39 |
| 12 % | 2,74 | 33,4 % | 66,0 % | 0,6 % | 43 | 30 |
| 16 % | 3,65 | 34,6 % | 65,4 % | 0,0 % | 29 | 20 |
| 20 % | 4,57 | 36,4 % | 63,5 % | 0,0 % | 19 | 14 |
| 25 % | 5,71 | 37,8 % | 62,2 % | 0,0 % | 14 | 10 |
| 32 % | 7,31 | 37,1 % | 62,9 % | 0,0 % | 10 | 7 |

Résultat autonome, avant tout conditionnement : **ce livre TSMOM ne passe pas une
eval Tradeify Select Flex 25K plus de 38 % du temps, quel que soit le levier.**
La p(pass) plafonne autour de 37 % et redescend ensuite. Au-delà de 8 % de
volatilité de compte, le prix d'une p(pass) plus haute est une durée de vie qui
s'effondre : 50 sessions à 8 %, 10 sessions à 25 %.

**Amendement 1, journalisé.** L'échelle déclarée s'arrêtait à 12 % et la règle de
calibrage déclarée était « le barreau dont la p(pass) du bras constant est la plus
proche de 0,50 ». Elle n'encadrait pas 0,50 — la p(pass) ne franchit un demi à
aucun levier — donc la règle sélectionnait le bord de la grille. Échelle
prolongée vers le haut.

**Amendement 2, journalisé, et c'est le plus important.** Prolongée, la règle est
devenue « le maximum » et a désigné 25 % de volatilité de compte, **où le compte
médian se résout en dix sessions**. Une barrière qui se résout en dix sessions ne
peut pas exprimer un régime dont les épisodes courent de 43 à 2 698 sessions : ce
barreau détruit le mécanisme testé. La règle de calibrage fondée sur un résultat
est donc **abandonnée** et remplacée par une ancre purement géométrique :
volatilité annualisée du compte = profondeur de la barrière =
`trailing_dd_eval / capital` = **4 %**. La barrière est alors à exactement une
année de volatilité, aucun résultat n'entre dans ce choix, et le levier implicite
sur le livre est de ×0,91.

Les deux amendements ont été faits après avoir lu l'échelle du **bras constant**,
qui ne contient aucune information de régime, et avant toute comparaison de bras.
Et pour qu'aucun barreau ne soit privilégié, le delta est rapporté à **tous** les
barreaux au §7.

### Le prix de l'exposition nue

Mesuré sur le bras constant seul, à l'ancre, en multipliant simplement son P&L :

| multiplicateur d'exposition | p(pass) | Δp |
|---|---|---|
| 0,900 | 7,3 % | −3,6 pt |
| 0,950 | 8,9 % | −2,0 pt |
| 1,000 | 10,9 % | — |
| 1,025 | 11,8 % | +0,9 pt |
| 1,050 | 11,4 % | +0,5 pt |
| 1,080 | 12,3 % | +1,4 pt |
| 1,120 | 13,7 % | +2,8 pt |

Pente : **+0,2785 de p(pass) par unité de multiplicateur**, soit +0,28 point de
p(pass) par point de pourcentage d'exposition supplémentaire. C'est le baromètre
contre lequel tout Δp(pass) doit être lu, et il n'existe que parce que T3 a montré
qu'il fallait le mesurer.

---

## 5. Les deux versions du même livre

États lus dans `data/cache/states_offline.parquet`, famille primaire **A′ sparse
jump**, déclarée avant lecture des résultats. Part du temps dans l'état turbulent
15,6 %. Le plafond de levier de 2,0 de `state_to_size` **ne mord sur aucune
session** (0,00 %), donc il ne déforme rien.

Géométrie Tradeify, ancre 4 %, 4 000 pseudo-comptes, bloc moyen 63 :

| bras | exposition moy. | écart-type expo | p(pass) | p(bust) | j. → cible | j. → échec | sous l'eau |
|---|---|---|---|---|---|---|---|
| taille constante | **1,000** | 0,000 | 10,9 % | 35,4 % | 128 | 90 | 186 |
| régime, gross libre | **1,113** | 0,175 | 13,9 % | 39,0 % | 121 | 89 | 184 |
| **régime, exposition appariée** | **1,025** | 0,171 | **11,2 %** | 35,5 % | **126** | **90** | **186** |
| placebo vol appariée, 2 seaux | **1,103** | 0,177 | 13,8 % | 38,0 % | 122 | 92 | 180 |
| cible de vol, sans paramètre | **1,116** | 0,365 | 13,5 % | 31,2 % | 126 | 106 | 175 |

Deltas contre taille constante, et ce que l'exposition explique déjà :

| bras | expo | Δp(pass) | attendu par l'exposition seule | **résidu** |
|---|---|---|---|---|
| régime, gross libre | 1,113 | +3,0 pt | +3,2 pt | **−0,2 pt** |
| **régime, exposition appariée** | 1,025 | **+0,3 pt** | +0,7 pt | **−0,4 pt** |
| placebo vol appariée | 1,103 | +2,9 pt | +2,9 pt | **+0,0 pt** |
| cible de vol, sans paramètre | 1,116 | +2,6 pt | +3,2 pt | **−0,6 pt** |

**L'arithmétique se ferme à moins d'un point près sur les quatre bras.** Chaque
Δp(pass) est entièrement acheté par l'exposition moyenne résiduelle du bras, et
rien d'autre. C'est l'illusion de dénominateur de T3, mesurée au niveau barrière,
et elle ne laisse aucun résidu à expliquer. Le bras de régime apparié est celui
dont le résidu est le plus négatif après la cible de vol.

Durées, avec leur incertitude d'histoire :

| quantité | constant | régime apparié | Δ | se hist. | MDE | verdict |
|---|---|---|---|---|---|---|
| jours → cible (médiane) | 128 | 126 | **−2** | — | — | — |
| jours → échec (médiane) | 90 | 90 | **+0,0** | 7,4 | **20,7** | **UNDERPOWERED** |
| plus longue période sous l'eau | 186 | 186 | **−0,5** | 9,5 | **26,6** | **UNDERPOWERED** |
| p(pass) | 10,9 % | 11,2 % | **+0,28 pt** | 2,18 pt | **6,10 pt** | **UNDERPOWERED** |

IC 95 % de Δp(pass) : **[−3,5 pt, +5,0 pt]**. Il contient zéro largement.

### Constantes de temps — l'explication structurelle

| quantité | valeur |
|---|---|
| durée de vie médiane d'un compte qui échoue | **90 sessions** |
| durée médiane jusqu'à la cible quand elle est atteinte | **128 sessions** |
| transitions d'état dans tout l'échantillon | **13**, sur 6 377 sessions |
| transitions moyennes dans une fenêtre de 90 sessions | **0,18** |
| part des fenêtres de 90 sessions **sans aucune transition** | **89 %** |
| part des fenêtres de 360 sessions sans aucune transition | **71 %** |

Voilà pourquoi le dispositif ne peut pas fonctionner ici, et c'est mesuré et non
supposé : **89 % des comptes qui échouent vivent et meurent à l'intérieur d'un
seul état.** Le signal de régime n'a rien à dire pendant la durée de vie d'un
challenge propfirm. Sa constante de temps est de l'ordre de l'année ; celle de la
barrière est de l'ordre du trimestre.

### Tête à tête — le test qui décide

Le bras de régime ne doit pas battre le livre à taille constante. Il doit battre
les témoins qui ne savent rien des régimes. Apparié sur les mêmes
pseudo-histoires :

| comparaison | Δp(pass) | se | MDE | verdict |
|---|---|---|---|---|
| état − placebo par quantile de vol | **−2,6 pt** | 2,0 pt | 5,6 pt | perd, UNDERPOWERED en magnitude |
| état − cible de vol sans paramètre | **−2,3 pt** | 2,8 pt | 7,7 pt | perd, UNDERPOWERED en magnitude |

---

## 6. Les trois nuls, avec les percentiles

### Nul 1 — dimensionner par un quantile de volatilité réalisée (200 tirages)

Le placebo qui a tué tous les effets de ce projet. Fenêtre tirée dans
{21, 42, 63, 126, 252}, seuil tiré parmi 40 points de [0,50 ; 0,95].

| quantité | nul : moyenne | écart-type | p95 | réel (état) | **percentile** |
|---|---|---|---|---|---|
| Δp(pass) | **+2,6 pt** | 1,4 pt | +4,1 pt | **+0,3 pt** | **16ᵉ** |
| Δ sous l'eau | −1,9 j | 3,4 | — | −0,5 j | 46ᵉ |
| Δ jours → échec | +8,3 j | 4,5 | — | +0,0 j | **4ᵉ** |

Exposition moyenne des placebos 1,088, contre 1,025 pour l'état.

Et le nul lui-même se lit avec le baromètre : exposition 1,088 × pente 0,2785 =
**+2,44 pt attendus**, contre +2,56 pt observés. **Le gain moyen du placebo est
lui aussi, entièrement, de l'exposition.** Ce n'est donc pas que le placebo batte
l'état par un meilleur conditionnement : c'est qu'à cette géométrie, *rien* — ni
l'état, ni aucun quantile de volatilité — n'achète quoi que ce soit au-delà de sa
propre exposition moyenne. La barrière s'achète avec de l'exposition, pas avec du
timing.

### Nul 2 — permuter les états en conservant leur durée (400 tirages)

L'état ne change que **14 fois** sur 6 377 sessions. C'est la taille
d'échantillon réelle de toute affirmation conditionnelle ici, et non 6 377 jours ;
`analysis/power.py` le dit déjà, le nombre d'épisodes indépendants est proche de
quatorze. Le nul permute l'**ordre** des épisodes et conserve chaque durée à
l'identique : le temps total dans chaque état et toute la distribution des durées
survivent exactement, seul l'alignement au calendrier tombe.

| quantité | nul : moyenne | écart-type | p95 | réel (état) | **percentile** |
|---|---|---|---|---|---|
| Δp(pass) | −0,1 pt | 1,3 pt | +2,0 pt | **+0,3 pt** | **57ᵉ** |
| Δ sous l'eau | +1,4 j | 1,6 | — | −0,5 j | 89ᵉ |
| Δ jours → échec | +1,4 j | 3,4 | — | +0,0 j | 35ᵉ |

La moyenne du nul à −0,1 pt sur un Δ réel de +0,3 pt est le contrôle que le nul
est non biaisé : permuter les épisodes ne fabrique ni gain ni perte.

### Nul 3 — rotation circulaire du profil de levier (400 tirages)

La forme de placebo établie dans le dépôt (contrôle L58 de `docs/EXTENSIONS.md`) :
l'autocorrélation du profil est préservée exactement, seul son alignement aux
dates est détruit.

| quantité | nul : moyenne | écart-type | réel (état) | **percentile** |
|---|---|---|---|---|
| Δp(pass) | +0,7 pt | 0,9 pt | **+0,3 pt** | **30ᵉ** |

---

## 7. Robustesse — les cinq façons dont le verdict pourrait changer d'avis

### Les cinq familles, avec correction pour tests multiples

| famille | p(pass) | Δp | j. → cible | j. → échec | sous l'eau | expo |
|---|---|---|---|---|---|---|
| A jump | 12,6 % | **+1,7 pt** | 126 | 92 | 185 | 1,07 |
| A′ sparse jump | 11,2 % | +0,3 pt | 126 | 90 | 186 | 1,03 |
| B filtered HMM | 11,6 % | +0,7 pt | 129 | 99 | 186 | 1,09 |
| C gradient boost | 10,5 % | −0,4 pt | 128 | 99 | 192 | 1,00 |
| C′ HAR-RV | 11,6 % | +0,7 pt | 126 | 104 | 178 | 1,06 |

Nul familial : le nul 2 refait 150 fois en prenant à chaque tirage le **maximum**
de Δp sur les cinq familles. Moyenne **+3,0 pt**, écart-type 1,4, p95 +5,6.
Maximum réel **+1,7 pt → 17ᵉ percentile.** Le meilleur des cinq classifieurs fait
**moins bien** que ce qu'un état permuté obtient par hasard quand on lui accorde
les mêmes cinq essais.

### Le même delta à chaque barreau de l'échelle

Δp(pass) contre taille constante, à tous les niveaux de risque :

| vol ann. | constant | régime libre | **régime apparié** | placebo vol | cible de vol |
|---|---|---|---|---|---|
| 1 % | 0,0 % | +0,0 | +0,0 | +0,0 | +0,0 |
| 2 % | 0,1 % | +0,0 | +0,1 | +0,1 | +0,4 |
| 3 % | 3,3 % | +1,6 | +0,0 | +1,3 | +1,4 |
| **4 % (ancre)** | 10,9 % | +3,0 | **+0,3** | +2,9 | +2,6 |
| 6 % | 23,6 % | +2,2 | −0,3 | +4,4 | +3,4 |
| 8 % | 30,6 % | +0,0 | −1,1 | −0,5 | −0,8 |
| 10 % | 32,5 % | +1,4 | +0,1 | +0,5 | +1,1 |
| 12 % | 33,4 % | +2,0 | +1,0 | +2,4 | +1,7 |
| 16 % | 34,6 % | +0,1 | −0,7 | +1,5 | −0,0 |
| 20 % | 36,4 % | +0,7 | −0,1 | +0,4 | −1,3 |
| 25 % | 37,8 % | −1,5 | −0,5 | −0,8 | −1,8 |
| 32 % | 37,1 % | +1,4 | +0,8 | −0,1 | +1,9 |

**Barreaux où le régime apparié battait les deux témoins : 1 sur 12.**

### Sensibilité à la longueur de bloc

| bloc moyen | Δp état | Δp placebo vol | Δp cible de vol |
|---|---|---|---|
| 21 | −0,1 pt | +2,6 pt | +2,0 pt |
| 63 | +0,5 pt | +3,2 pt | +2,5 pt |
| 126 | +1,6 pt | +3,2 pt | +4,4 pt |

L'état est dernier aux trois longueurs de bloc. L'ordre ne dépend pas du choix du
bloc.

### Les fenêtres historiques réelles, sans aucun reéchantillonnage

5 768 fenêtres consécutives de 360 sessions, telles qu'elles se sont produites.
Pas d'IC valide — elles se recouvrent — mais c'est la seule mesure qui contienne
les vraies séquences de 2008 et de 2020.

| bras | p(pass) | Δp | Δ j. → cible | Δ j. → échec | Δ sous l'eau |
|---|---|---|---|---|---|
| taille constante | 10,0 % | — | — | — | — |
| régime, gross libre | 16,3 % | +6,3 pt | −2 | +0 | −5 |
| **régime, exposition appariée** | 12,5 % | **+2,5 pt** | −2 | +0 | −5 |
| placebo vol appariée | 13,2 % | +3,3 pt | −6 | +2 | −7 |
| cible de vol, sans paramètre | 14,5 % | +4,5 pt | −4 | +24 | −14 |

Sur la vraie chronologie le régime apparié fait mieux qu'au bootstrap (+2,5 au
lieu de +0,3) — et reste **dernier des trois dispositifs**. L'ordre survit.

### La géométrie FTMO, plancher statique (adaptateur local)

| bras | p(pass) | p(bust) | j. → cible | j. → échec | Δp | Δ j. → échec |
|---|---|---|---|---|---|---|
| taille constante | 6,3 % | 3,6 % | 285 | 243 | — | — |
| régime, gross libre | 9,6 % | 2,7 % | 280 | 266 | +3,3 pt | +23 |
| **régime, exposition appariée** | 7,3 % | 2,6 % | 282 | 278 | **+1,0 pt** | **+36** |
| placebo vol appariée | 10,1 % | 2,5 % | 272 | 263 | +3,8 pt | +20 |
| cible de vol, sans paramètre | 9,3 % | 1,1 % | 270 | 303 | +3,0 pt | **+60** |

Sur un plancher statique — la géométrie où la durée compte le plus, puisqu'il n'y
a pas de cliquet qui remonte — le régime apparié **allonge la survie de 36
sessions**, et c'est la seule sortie de ce rapport où il fait quelque chose de
visible. La cible de volatilité sans paramètre en allonge **60**. L'ordre est le
même qu'ailleurs.

---

## 8. Verdict

**NULL sur les nuls. UNDERPOWERED sur la magnitude. Les deux, et pas l'un pour
l'autre.**

- **UNDERPOWERED sur la magnitude.** Δp(pass) = **+0,28 point**, incertitude
  d'histoire 2,18 points, IC 95 % [−3,5 ; +5,0], **MDE 6,10 points**. L'effet
  observé est vingt fois plus petit que ce que cet échantillon peut résoudre.
  Même chose sur les durées : Δ jours → échec = +0,0 contre un MDE de 20,7 ;
  Δ plus longue période sous l'eau = −0,5 contre un MDE de 26,6. Cet échantillon
  ne peut pas trancher un effet de cette taille, et le dire est le résultat, pas
  un échec.
- **NULL sur les nuls**, et c'est là que le dossier se referme. L'effet réel est
  **à l'intérieur** des trois distributions nulles et **sous la moyenne** de deux
  d'entre elles : 16ᵉ percentile du placebo par quantile de volatilité, 57ᵉ de la
  permutation à durée conservée, 30ᵉ de la rotation circulaire. Sur la durée
  jusqu'à l'échec, 4ᵉ percentile du nul 1. Le meilleur des cinq classifieurs est
  au 17ᵉ percentile de son nul familial. Un effet au 16ᵉ percentile de son placebo
  n'est pas un effet sous-puissant : c'est un non-effet.
- **Et la raison est arithmétique, pas statistique.** Les quatre dispositifs
  testés ont un Δp(pass) entièrement expliqué, à moins d'un point près, par leur
  seule exposition moyenne résiduelle. Le résidu du bras de régime apparié est
  **−0,4 point**. C'est l'illusion de dénominateur de T3, retrouvée telle quelle
  au niveau barrière : la p(pass) s'achète avec de l'exposition, pas avec du
  timing de variance.

**La thèse était bien posée et elle est fausse, pour une raison que la thèse
n'avait pas anticipée.** L'argument disait : la barrière est un problème de
variance, donc un signal de variance doit y valoir quelque chose. Ce qui manquait
à l'argument, c'est une constante de temps. **89 % des comptes qui échouent
vivent et meurent à l'intérieur d'un seul état de régime** — 0,18 transition en
moyenne sur les 90 sessions d'une vie de compte. Un signal dont les épisodes
durent 456 sessions en moyenne ne peut rien dire d'un objet qui se résout en 90.
Le régime porte bien la variance, mais il la porte sur une échelle de temps que
la barrière ne voit pas.

Il reste un chiffre qui n'est pas rien et qu'il serait malhonnête de ne pas
isoler : sur la géométrie **FTMO à plancher statique**, le régime apparié allonge
la durée médiane jusqu'à l'échec de **+36 sessions**. C'est cohérent avec
AlphaSimplex — sur un plancher fixe, survivre longtemps est exactement l'objectif
— et c'est la seule sortie où le dispositif fait quelque chose de visible. Mais
la cible de volatilité sans paramètre en allonge **60**, et cette géométrie passe
par mon propre adaptateur et non par le simulateur audité du dépôt. Ce n'est pas
un résultat ; c'est la seule direction que ce travail laisse ouverte, et elle
demanderait le simulateur FTMO qui n'existe pas encore.

**Cinquième arrivée indépendante, et la dernière que le programme avait en
réserve.** Le classifieur l'a mesuré directement, la réplication Shu 2024 l'a
retrouvé, l'atténuateur de Carver l'a retrouvé, T1/T3/T5 l'ont retrouvé, et la
géométrie de barrière — la seule application qui restait debout — le retrouve
aussi. Les régimes portent la variance et pas la moyenne, et **aucun dispositif
ne la monétise, pas même là où la variance est littéralement l'objectif.** Le
socle universel reste ce que le scan praticien disait déjà : le ciblage de
volatilité, sans paramètre, qui ne sait rien des régimes et fait mieux.

---

## 9. Mes erreurs, nommées

### Erreur 1 — le plafond de levier aurait handicapé les deux témoins, et seulement eux

Dans la première version du script, le placebo par quantile de volatilité et la
cible de volatilité sans paramètre calculaient tous deux leur levier comme
`VOL_TARGET / sigma_livre`, avec `VOL_TARGET = 0,10` — la constante de
`extensions/trend.py` — et le plafond de 2,0 de `state_to_size`.

Mais la volatilité propre de ce livre TSMOM est de 4,54 % annualisés, pas de
10 %. Mesuré : médiane de `sigma` = 0,0338, donc `0,10 / sigma` a une médiane de
**2,96** et **le plafond de 2,0 aurait mordu sur 76,8 % des sessions.** Les deux
témoins auraient été aplatis à un levier constant de 2,0 sur les trois quarts de
l'échantillon : leur écart-type de levier tombait à 0,271 au lieu de 0,446, soit
**−39 % du contraste même qui était testé**.

Le bras de régime, lui, n'était pas touché : `state_to_size` divise par la
volatilité **par état**, qui vaut 0,049 à 0,172 selon la réestimation et l'état,
donc son levier maximal est de 1,32 / 1,34 / 1,48 / 1,60 / 2,03 pour les cinq
familles et n'approche 2,0 que pour C′ HAR-RV. **Le bug n'handicapait que les
témoins, donc il poussait dans le sens qui m'arrangeait.**
C'est la direction la plus dangereuse pour un bug, et c'est pour ça qu'il est
écrit ici plutôt qu'effacé.

Correction : tout bras non gelé exprime son levier comme un rapport à la moyenne
expansive causale de la volatilité du livre, `ref_t / sigma_t`, plafonné au même
2,0 — donc le plafond mord au même point relatif (deux fois le risque moyen du
livre) sur tous les bras. Après correction, **le plafond ne mord sur 0,00 % des
sessions** du bras de régime.

### Erreur 2 — un facteur 100 dans la décomposition par l'exposition

`exp_dp = slope * (exposure - 1.0) * 100.0` portait un ×100 parasite. Il
rapportait que l'exposition expliquait **+17,7 points** de Δp(pass) pour un bras
à 1,113 d'exposition — un chiffre impossible, plus grand que la p(pass) entière.
Publié tel quel, il aurait dit que les quatre bras *détruisent* massivement la
p(pass) par rapport à ce que leur exposition achète, avec des résidus de −17 à
−21 points. Le ×100 est retiré : la pente est déjà en p(pass) par unité de
multiplicateur.

### Erreur 3 — l'échantillon commun construit comme un suffixe

Le premier passage s'est arrêté sur une assertion, « échantillon commun non
contigu : 2 trous ». Les états s'arrêtent au 2026-09-08 et le livre court jusqu'au
2026-09-10 ; je construisais l'échantillon commun comme un suffixe à partir de la
première date valide, sans borner la fin. L'assertion était volontaire et elle a
fait son travail. Corrigé en premier-à-dernier avec vérification de contiguïté au
milieu.

### Ce que j'ai refusé de faire

La règle de calibrage « le maximum de p(pass) » désignait 25 % de volatilité de
compte, où le Δp(pass) du régime apparié est de **−1,1 point**. La remplacer par
une ancre géométrique n'était donc pas un choix qui m'arrangeait, et l'échelle
complète est publiée au §7 précisément pour qu'on puisse le vérifier. Je n'ai pas
ajusté après avoir vu : les deux amendements ont été décidés sur le bras constant
seul, et journalisés au §4 avec ce qu'ils produisaient avant correction.

---

## 10. Fichiers

| fichier | contenu |
|---|---|
| `regime_barrier.py` | tout le calcul, déclarations en tête dans `DECLARED` |
| `RAPPORT_BARRIERE.md` | ce document |
| `barrier_results.json` | tous les chiffres du passage complet, et les tirages utilisés |
| `run_full.txt` | sortie intégrale du passage complet (260 s, tirages du §3) |
| `run_quick_kept.txt`, `barrier_results_quick.json` | le passage court préalable (34 s, un quart des tirages) — verdicts identiques |

Reproduction : `.venv/bin/python regime_barrier.py` depuis `~/Desktop/regime-lab`
avec le venv du dépôt. `BARRIER_SCALE=quick` pour le passage court. Graine unique
`20260913`, déclarée dans `DECLARED`.
