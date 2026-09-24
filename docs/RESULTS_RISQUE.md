# Le régime comme entrée d'une prévision de risque, sur 46 marchés : résultats

**En une phrase.** Utilisé là où son seul actif démontré devrait servir, la **prévision
du risque**, l'état du Sparse Jump Model n'améliore la prévision de volatilité d'**aucun**
des 46 marchés. Il la dégrade légèrement : −0,8 % sur les 43 marchés sans VIX propre, et
−1,8 % une fois le VIX dans le modèle. Le VIX, lui, améliore la même prévision de 5,6 %
sur ces 43 marchés. Dimensionner un livre avec la prévision augmentée de l'état ne change
pas son Sharpe (0,456 → 0,456).

- Pré-enregistrement : `docs/PRESPEC_RISQUE.md`, commit **`b12f193`**, avant toute
  lecture. Il porte les seuils, la famille de tests et les prédictions.
- Lecture, faite une fois : `scripts/run_risque.py --read`. La sortie intégrale est dans
  `docs/artifacts/risque/reading.txt` (commit `350f070`). Sa partie instrument est
  identique, octet pour octet, à `docs/artifacts/risque/instrument.txt`, commité avant
  la lecture.
- **16 essais** journalisés, famille `risque_forecast` de `data/trials.parquet`
  (2026-09-23, 22:41 UTC).
- **Aucun coût, nulle part.** Les Sharpe sont en excès du cash. La rotation est
  rapportée.

---

## 1. La question, en clair

Le modèle sait des choses sur **l'ampleur** des mouvements futurs, rien sur leur **sens**
(`RESULTS_FINAL.md`). Jusqu'ici, on l'a toujours jugé en Sharpe, à travers une stratégie :
couper, réduire, basculer. Ici, on le juge directement sur ce qu'il est censé savoir
faire : **prévoir le risque**.

Pour chacun des 46 marchés du livre de tendance, on prévoit la variance du mois suivant
(21 séances) avec un modèle standard, le **HAR** : la volatilité de la dernière semaine,
du dernier mois et du dernier trimestre. On lui ajoute l'état du modèle (« stress » ou
« calme ») et on regarde si la prévision devient meilleure. Tout est estimé **sur le passé
seul** : chaque jour, le modèle est réestimé sur les seules données déjà connues.

La note de prévision est la **QLIKE**, une mesure d'erreur standard pour les prévisions
de variance. Plus elle est basse, mieux c'est. On rapporte surtout son **gain relatif** :
+1 % veut dire que l'erreur baisse de 1 %.

Le VIX sait déjà ce que sait le modèle sur le S&P 500 (`RESULTS_CRISE.md` §5). **La
question intéressante porte donc sur les 43 autres marchés** : actions hors US, matières
premières, devises, obligations et crédit. Aucun d'eux n'a de VIX propre dans le dépôt.

## 2. Résultat principal

Fenêtre d'évaluation : 20/12/2005 → 12/08/2026, 5 385 séances. Elle contient deux
épisodes de stress (2008-2009, 2020-2021). Au moins 30 des 43 marchés sont présents
chaque jour.

| test | panneau | ajouté à | erreur de base (QLIKE) | gain de l'état | gain règle médiane | gain règle 80ᵉ c. | Δ | MDE | t (HAC 21) | placebo | plis > 0 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **P1** | 43 marchés sans VIX propre | HAR | 0,2986 | **−0,82 %** | +0,87 % | +1,47 % | −0,00246 | 0,00648 | −1,30 | 50,0 % | 2/5 | **pas utile** |
| **P2** | les mêmes 43 | HAR + VIX | 0,2818 | **−1,84 %** | −1,34 % | −1,04 % | −0,00518 | 0,00660 | −2,92 | 14,2 % | 1/5 | **pas utile** |
| **P3** | 3 indices actions US | HAR + VIX | 0,3039 | **−0,99 %** | −1,97 % | −0,36 % | −0,00302 | 0,01224 | −0,85 | 33,0 % | 3/5 | **pas utile** |

Pour lire le tableau :
- **Δ** est la baisse moyenne de l'erreur quand on ajoute l'état. Positif, l'état aide ;
  négatif, il nuit.
- Le **MDE** est le plus petit écart que l'échantillon permet de distinguer du hasard,
  corrigé pour les 3 tests de la famille. Il vaut 2,2 % de l'erreur de base pour P1.
- Le **placebo** fait tourner l'état au hasard dans le temps, 400 fois, en réestimant
  toutes les prévisions à chaque fois. 50 % veut dire que le vrai état fait exactement
  comme un état placé au hasard.
- Les **règles** sont les deux témoins d'une ligne du programme : « la volatilité du S&P
  500 est au-dessus de sa médiane passée », ou « de son 80ᵉ centile passé ».

**Ce que dit le tableau :**
- **P1.** Ajouté au modèle standard, l'état **dégrade** la prévision des 43 marchés
  (−0,82 %), exactement comme un état placé au hasard. Les deux règles d'une ligne font
  mieux que lui : elles l'améliorent un peu (+0,87 % et +1,47 %).
- **P2.** Une fois le VIX dans le modèle, l'état la dégrade davantage (−1,84 %), et
  les règles aussi. La dégradation est nette au sens statistique habituel : t −2,92,
  au-delà du seuil de 2,39 à α 0,05/3. Mais elle reste sous le MDE. Selon l'échelle
  écrite d'avance, c'est donc **pas utile, et non nuisible**.
- **P3.** Sur les actions US, là où le VIX s'applique, l'état n'ajoute rien au-delà
  du VIX. On retrouve le test P de l'étude de crise.

## 3. Par classe d'actifs

| test | classe | ajouté à | gain de l'état | gain règle médiane | gain règle 80ᵉ | Δ | MDE | t | placebo | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | actions hors US (10) | HAR | −0,71 % | −0,80 % | +0,85 % | −0,00227 | 0,01668 | −0,50 | 45,5 % | pas utile |
| C2 | actions hors US | HAR + VIX | −1,50 % | −2,34 % | −0,74 % | −0,00454 | 0,01227 | −1,35 | 15,0 % | pas utile |
| C3 | matières premières (13) | HAR | −1,68 % | −0,30 % | +0,02 % | −0,00364 | 0,00490 | −3,48 | 7,8 % | pas utile |
| C4 | matières premières | HAR + VIX | −1,66 % | −0,62 % | −0,66 % | −0,00357 | 0,00417 | −3,96 | 7,8 % | pas utile |
| C5 | devises (11) | HAR | −1,07 % | +0,09 % | +0,81 % | −0,00324 | 0,00814 | −1,36 | 49,0 % | pas utile |
| C6 | devises | HAR + VIX | −1,03 % | −1,25 % | −0,88 % | −0,00294 | 0,00875 | −1,24 | 50,5 % | pas utile |
| C7 | obligations et crédit (9) | HAR | **+0,05 %** | +3,87 % | +3,69 % | +0,00020 | 0,01556 | +0,05 | 66,5 % | sous-puissant |
| C8 | obligations et crédit | HAR + VIX | −3,33 % | −1,09 % | −2,08 % | −0,01199 | 0,02336 | −1,98 | 3,0 % | pas utile |

- Une seule case positive sur huit : obligations et crédit, au-delà du HAR, **+0,05 %**,
  très loin du MDE (3,9 %). Les deux règles d'une ligne y gagnent plus de 70 fois plus (+3,87 % et
  +3,69 %).
- **Matières premières.** La dégradation est statistiquement nette (t −3,5 et −4,0,
  au-delà du seuil de 2,73). Mais le placebo dégrade presque autant : son 95ᵉ centile
  est lui-même négatif (−0,00045). Ajouter une indicatrice 0/1, quelle qu'elle soit, y
  coûte du bruit d'estimation. L'état n'y fait pas pire qu'un état au hasard (7,8ᵉ
  centile).
- **Obligations et crédit, au-delà du VIX (C8).** L'écart est sous le 5ᵉ centile du
  placebo (3,0 %) et le t vaut −1,98. Mais |Δ| reste sous le MDE, donc le verdict est
  pas utile, pas nuisible.
- Ces tests portent sur 3 à 13 marchés : ce sont des **mesures par classe**, pas des
  affirmations transversales (règle N ≥ 30).

**Marché par marché** (46 tests de Diebold-Mariano par base, correction de Holm pour 46
marchés, déclarée d'avance) :
- **aucun marché n'est significativement amélioré par l'état, et aucun n'est
  significativement dégradé** ;
- l'écart n'est même positif que sur 10 marchés sur 46 au-delà du HAR, et 5 sur 46
  au-delà du HAR + VIX ;
- sur les matières premières, il est négatif pour les 13 marchés, contre les deux bases.

## 4. Ce que le VIX apporte, lui

Colonne « VIX adds » de la lecture : gain relatif du HAR + VIX sur le HAR seul. Ce n'est
pas un test de la famille, mais une mesure rapportée.

| panneau | le VIX améliore la prévision de |
|---|---|
| 43 marchés sans VIX propre | **+5,62 %** |
| actions US (3) | +13,74 % |
| obligations et crédit | +9,83 % |
| devises | +5,38 % |
| actions hors US | +5,15 % |
| matières premières | +1,17 % |

**Le prix public de la volatilité des actions US améliore la prévision du risque sur
toutes les classes, devises et obligations comprises. L'état du modèle n'améliore la
prévision d'aucune.**

## 5. La conséquence économique : les livres

Du 13/03/2006 au 10/09/2026, 5 347 séances, sans coût, en excès du cash. Chaque
instrument est dimensionné par la volatilité prévue, puis le livre entier est ciblé à
10 %, comme dans le véhicule du programme (M3).

**E1 — livre de tendance, 46 marchés**

| dimensionné par | Sharpe | rendement | volatilité | perte max. | rotation/an | erreur de suivi du risque |
|---|---|---|---|---|---|---|
| HAR | 0,456 | +4,77 % | 10,47 % | −24,4 % | 66,1 | 0,294 |
| **HAR + état** | **0,456** | +4,77 % | 10,48 % | −24,4 % | 65,3 | 0,297 |
| HAR + règle médiane | 0,458 | +4,79 % | 10,47 % | −24,3 % | 66,5 | 0,294 |
| HAR + règle 80ᵉ c. | 0,467 | +4,89 % | 10,46 % | −24,7 % | 66,0 | 0,294 |
| HAR + VIX | 0,488 | +5,09 % | 10,43 % | −23,3 % | 66,6 | 0,284 |
| véhicule M3 tel quel (σ63), pour situer | 0,465 | +4,91 % | 10,57 % | −24,1 % | 57,7 | 0,331 |

Δ −0,000, MDE 0,020, t +0,01, placebo 49,5 %, 3 plis sur 5 positifs. **Verdict : pas
utile.**

**E2 — parité de risque longue seule, 35 marchés hors devises**

| dimensionné par | Sharpe | rendement | volatilité | perte max. | rotation/an | erreur de suivi |
|---|---|---|---|---|---|---|
| HAR | 0,648 | +6,47 % | 9,98 % | −27,6 % | 16,1 | 0,310 |
| **HAR + état** | **0,640** | +6,38 % | 9,98 % | −27,2 % | 15,6 | 0,314 |
| HAR + règle médiane | 0,652 | +6,51 % | 9,98 % | −27,4 % | 16,2 | 0,310 |
| HAR + règle 80ᵉ c. | 0,655 | +6,54 % | 9,97 % | −27,9 % | 15,9 | 0,309 |
| HAR + VIX | 0,670 | +6,67 % | 9,94 % | −27,5 % | 15,9 | 0,307 |

Δ −0,008, MDE 0,029, t −1,09, placebo 6,2 %, 2 plis sur 5 positifs. **Verdict : pas
utile.**

- Les volatilités réalisées sont identiques d'un bras à l'autre (9,94 % à 10,48 %). Les
  écarts de Sharpe ne passent donc pas par le dénominateur (piège n° 5).
- L'« erreur de suivi du risque » est l'écart type du log de la volatilité mensuelle
  réalisée rapportée à la cible de 10 %. Plus elle est basse, plus le risque du livre
  est stable. L'état la dégrade à peine (0,294 → 0,297). Le VIX l'améliore (0,284).
- Dans les deux livres, **chacun des trois témoins fait mieux que l'état**.

## 6. Verdicts, selon le critère écrit avant la lecture

- **Famille primaire (P1, P2, P3) : trois fois pas utile.** L'état n'améliore la
  prévision du risque ni au-delà d'un modèle standard, ni au-delà du VIX, ni sur les
  marchés sans VIX propre.
- **Par classe (C1 à C8) : sept pas utile, un sous-puissant** (obligations et crédit
  au-delà du HAR, +0,05 %).
- **Par marché** : 0 amélioration et 0 dégradation significatives sur 46, pour chacune
  des deux bases.
- **Livres (E1, E2) : deux fois pas utile.**
- Aucune valeur non finie, et aucune ligne du registre en dehors des 16 déclarées.

**Ce qui était prédit (PRESPEC §9), et ce qui sort :**
- **P1 : la prédiction était fausse, et je le dis.** J'attendais un Δ positif avec une
  probabilité de 0,7 : il est négatif. L'argument était qu'un facteur commun de stress
  apporterait une information que le HAR d'un marché isolé ne voit pas. Les règles d'une
  ligne, qui sont aussi des facteurs communs, apportent un peu (+0,9 % et +1,5 %).
  L'état, non.
- P2 plus petit que P1 : oui. P3 proche de zéro : oui (−1 %). E1, E2 avec |Δ| < 0,05 et
  pas utiles : oui.
- Par classe, j'attendais le gain le plus probable sur les actions hors US et le crédit,
  et le plus faible sur les devises. Seul le crédit (au-delà du HAR) est positif. Les
  actions hors US sont négatives, et les matières premières sont la pire classe, pas les
  devises.

## 7. D'où vient la dégradation — lectures descriptives, qui ne décident rien

- **Elle se concentre sur les séances de stress.** Sur les 711 séances de stress, Δ vaut
  −0,00495 au-delà du HAR et −0,02173 au-delà du HAR + VIX. Sur les 4 674 séances calmes,
  il vaut −0,00208 et −0,00266. C'est cohérent avec le mécanisme du §0 de
  `PISTES_AMELIORATION.md` : l'état entre tard et sort très tard, donc il maintient des
  prévisions hautes pendant les reprises. **Je ne l'ai pas vérifié ici** : il faudrait
  séparer la sur-prévision et la sous-prévision, ce qui n'a pas été pré-enregistré.
- **Elle vient de quelques marchés.** Avec la médiane transversale au lieu de la
  moyenne, Δ vaut +0,00018 (P1) et −0,00016 (P2), soit à peu près zéro. Pour le marché
  typique, l'état ne change presque rien. Il fait perdre beaucoup sur quelques marchés
  (au-delà du HAR : EMB −193, MXN −175, HG −124, IBEX −125, en 10⁻⁴ de QLIKE).
- **La MSE de la variance dit un peu l'inverse** : +1,18 % pour P1 (t 1,46), +1,94 %
  pour P2 (t 1,32), non significatif. La MSE est dominée par les très gros jours ; la
  QLIKE, par l'erreur relative de tous les jours. L'état aide peut-être un peu sur les
  pics et nuit le reste du temps. La QLIKE a été déclarée d'avance comme la perte qui
  décide, parce qu'elle est robuste au bruit de la cible.
- **Mincer-Zarnowitz** (en niveaux, médianes par marché) : le R² baisse un peu avec
  l'état (0,203 → 0,201 pour P1 ; 0,243 → 0,233 pour P2).
- **Sensibilités déclarées** : sans CL=F, rien ne change (−0,00246 et −0,00517). Avec un
  EWMA calibré comme modèle de base, l'état dégrade aussi (−1,02 %, t −1,23).

**Une sensibilité qui compte pour la présentation.** Sur les trois indices actions US,
au-delà du HAR seul (sans VIX), l'état apporte **+0,10 %, t 0,06** : rien. Le résultat
central du programme (+3,93 points de R² sur la volatilité future) avait été mesuré
contre un **rang de volatilité passée à 21 jours**, en régression sur tout l'échantillon.
Contre un modèle de volatilité standard estimé sur le passé seul, il ne se retrouve pas
dans ce cadre. **Réserve** : la cible, la perte, la période et les marchés diffèrent
(trois indices, 2005-2026, QLIKE d'un log-HAR). C'est une indication, pas une
réfutation du chiffre publié.

## 8. Ce qu'on peut dire en présentation

1. **« Nous avons testé le modèle là où il devrait être le plus fort : la prévision du
   risque. Sur 46 marchés, il n'améliore la prévision d'aucun. »** Aucun marché sur 46
   après correction ; −0,8 % d'erreur sur les 43 marchés sans VIX propre.
2. **« Le prix des options sur le S&P 500, lui, améliore la prévision du risque de
   tous les marchés, devises et obligations comprises : +5,6 % en moyenne. »** Le marché
   fait gratuitement ce que le modèle ne fait pas. C'est le prolongement du message
   « le VIX sait déjà », étendu hors des actions US.
3. **« Même une règle d'une ligne fait mieux que le modèle. »** Au-delà du modèle
   standard, les deux règles de volatilité améliorent la prévision (+0,9 % et +1,5 %) ;
   l'état la dégrade (−0,8 %).
4. **« Dimensionner un portefeuille avec le modèle ne change rien »** : Sharpe 0,456 →
   0,456 sur le livre de tendance, 0,648 → 0,640 en parité de risque.

À ne jamais dire sans sa réserve : « le +3,93 disparaît ». Voir le §7 : le cadre n'est
pas le même.

## 9. Ce qui est prometteur, hors du régime

**Le dimensionnement par une prévision qui contient le VIX.** Ce n'est pas un résultat
de régime, et il est **hors famille** : le VIX n'était qu'un témoin des livres. Il n'a
ni placebo, ni MDE, ni correction qui lui soient propres.
- Livre de tendance : Sharpe 0,456 → 0,488 (+0,033), perte maximale −24,4 % → −23,3 %,
  erreur de suivi 0,294 → 0,284.
- Parité de risque : 0,648 → 0,670 (+0,022).
- Pour ordre de grandeur, le MDE de E1 était de 0,020.

Une étude propre la testerait avec les indices implicites propres à chaque classe (OVX,
GVZ, EVZ, MOVE), pré-enregistrée et à coûts institutionnels. Ce serait une piste pour
l'objectif de fond, **pas pour le projet « régimes »**.

## 10. Limites

- **Deux épisodes de stress** dans la fenêtre d'évaluation. L'épisode 2002-2003 ne sert
  qu'à l'entraînement. Le panneau multiplie les marchés, pas les épisodes.
- **Des prix Yahoo avec des trous** : PL=F n'a une prévision que sur 80 % des séances,
  et les ETF de crédit n'arrivent qu'en 2009-2010.
- **Une volatilité réalisée tirée des clôtures quotidiennes**, sans données
  intrajournalières.
- **Un HAR à composantes semaine, mois, trimestre**, faute de composante journalière
  exploitable (les jours fériés).
- **Un seul état, A′ filtré.** Aucune autre famille de modèles et aucune version
  continue de l'état n'ont été testées.
- **Le §7 est descriptif.** L'attribution de la dégradation à la sortie tardive est une
  lecture plausible, non vérifiée.
- **Aucun coût.** Les rotations sont rapportées (65 à 67 par an pour la tendance, 16 pour
  la parité de risque) et ne diffèrent presque pas d'un bras à l'autre.

## 11. Écarts et incidents

- **Deux interruptions de session** (limite de dépense) pendant l'étude. La lecture n'a
  été faite **qu'une fois** : ses 16 lignes portent toutes le même horodatage
  (22:41:52 UTC), et rien n'a été relancé.
- **Après le commit du PRESPEC, avant la lecture** (`571f1b5`), j'ai fait un
  changement mécanique : les 16 lignes sont écrites ensemble à la fin de la lecture,
  pour qu'un plantage ne laisse pas de lignes partielles. Juste avant, le chemin de
  lecture entier a tourné à blanc sur un **faux état** (une rotation fixe du vrai), sortie
  masquée et registre neutralisé. Aucun critère n'a changé.
- **Déclaré dans le PRESPEC** : l'instrument imprimait la QLIKE des modèles sans l'état,
  pour exprimer les MDE en pourcentage. On y voyait donc, avant la lecture, que le VIX
  améliore le HAR de 5,6 %.
- **P2 et C3, C4** : le t de Diebold-Mariano dépasse le seuil classique de sa famille
  (−2,92, −3,48, −3,96), mais |Δ| reste sous le MDE. Suivant l'échelle écrite d'avance,
  le verdict est « pas utile », pas « nuisible ». Pour un lecteur qui ne retient que le
  t, la dégradation est significative.
- La suite de tests complète passe : 689 tests, y compris les fichiers d'autres pistes
  présents sur le disque au moment du lancement. Les 28 tests de cette piste sont dans
  `tests/test_risque.py`. `ruff` est propre sur mes fichiers.
- Aucun fichier existant n'a été modifié.

## Reproduire

```bash
.venv/bin/python scripts/run_risque.py           # l'instrument seul, ~1 min, aucune lecture
# la lecture a été faite une fois ; la relancer ajoute 16 essais au registre :
# .venv/bin/python scripts/run_risque.py --read  # ~10 min (400 rotations réestimées)
```

Empreintes SHA-256 des fichiers lus : `PRESPEC_RISQUE.md` §4 et l'en-tête de
`docs/artifacts/risque/instrument.txt`. Yahoo révise ses prix ajustés : un nouveau
téléchargement peut déplacer les chiffres.
