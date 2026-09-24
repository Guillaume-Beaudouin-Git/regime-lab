# Le socle multi-stratégies, puis le régime en budget de risque — synthèse

**D'où vient cette étude.** C'est l'idée 4 de `docs/presentation/PISTES_AMELIORATION.md`.
Un Sharpe de 1 à 2 ne viendra pas d'un filtre posé sur une stratégie. Il viendra, s'il
vient, d'une **combinaison de stratégies peu corrélées**. On construit donc d'abord ce
livre sans régime, le **socle**. Ensuite seulement, on demande si le régime, utilisé comme
**budget de risque** à l'intérieur de ce livre, fait mieux que lui.

**Où.** Sept des neuf stratégies sont privées. L'étude vit donc dans le dépôt privé
voisin, comme `docs/RESULTS_COUPLAGE_STRATEGIES.md`. Ce fichier en est la synthèse
anonymisée.
- Le protocole, avec le critère et les seuils, a été commité **avant toute lecture**.
- Il y a eu une seule lecture, qui compte **6 essais `sjm_risk_budget`** dans
  `data/trials.parquet` : 3 lignes descriptives et 3 tests.
- Un incident a interrompu la première tentative dans une sensibilité descriptive,
  avant tout bras de test. Une stratégie long seule, plate pendant un an, avait une
  variance nulle. L'incident est consigné et corrigé avant la reprise, sans rien changer
  au critère.

**Hypothèse de coûts : aucun coût, partout.** Les Sharpe sont bruts. La rotation est
rapportée.

## Le livre

Neuf stratégies, les mêmes que dans les couplages précédents. **Aucune n'a été ajoutée ni
retirée selon son Sharpe** : la tendance énergie, négative seule, reste dans le livre.
- Chaque stratégie est ramenée à 10 % de sa volatilité de long terme (fenêtre expansive).
- Toutes reçoivent le même budget de risque : c'est la parité de risque diagonale,
  autrement dit l'inverse de la volatilité.
- Le livre est ciblé à 10 % chaque jour.
- Toute information est décalée de deux séances.
- Échantillon : **octobre 2005 à juin 2026, 5 385 séances**. Le livre compte 3
  stratégies en 2005-2007, 4 en 2008-2009, et 9 à partir de décembre 2015.

## 1. Le socle, sans régime (descriptif)

| | socle |
|---|---|
| Sharpe brut, 2005-2026 | **1,52**, intervalle à 95 % [1,06 ; 1,96] |
| perte maximale | −24,4 % |
| plis positifs | 5 sur 5 (de 0,86 à 2,12) |
| Sharpe brut, 2016-2026 (les neuf stratégies présentes) | **1,91**, perte maximale −10,0 % |

**La diversification**, sur 2016-2026 :

| stratégie seule | Sharpe |
|---|---|
| Prime overnight Nasdaq | 1,04 |
| Tendance crypto (BTC, ETH) | 1,01 |
| Cassure d'ouverture (ORB) Nasdaq | 0,99 |
| Rebond obligataire de fin de mois | 0,81 |
| Tendance or + argent | 0,59 |
| Tendance or | 0,52 |
| Momentum actions (UMD) | 0,38 |
| Momentum USDJPY | −0,05 |
| Tendance énergie, long seul | −0,06 |
| **moyenne des stratégies seules** | **0,58** |
| **socle** | **1,91** |

Le livre vaut **3,29 fois** la moyenne des stratégies qui le composent. Il fait mieux que
la meilleure d'entre elles, sans en choisir aucune. Les stratégies sont presque
indépendantes : le ratio de participation vaut 8,10 sur 9 (le livre de tendance du
programme : 3,86), et la corrélation médiane est nulle.

Deux variantes, descriptives, ne font pas mieux : la parité de risque en covariance
pleine (1,46) et l'ajout du livre de tendance de 46 instruments (1,49).

⚠ **Ce 1,52 est un plafond, pas un Sharpe hors échantillon.** Il est sans coût. Et les
stratégies ont été choisies par un programme de recherche qui connaissait ces années.

## 2. Le régime en budget de risque : le test

Trois façons d'utiliser l'état de stress A′, chacune comparée au socle, à alpha 0,05 / 3.
Le critère est celui de `scripts/run_crisis_coupling.py`, avec trois témoins : la règle
médiane de volatilité, la règle au 80ᵉ centile et le VIX au-dessus de son 80ᵉ centile.
Tout paramètre est estimé sur le passé seul.

| bras | en stress, on… | Δ Sharpe | seuil (MDE) | placebo | témoins (médiane / 80ᵉ / VIX) | perte max. | verdict |
|---|---|---|---|---|---|---|---|
| **R_VAR** | réalloue le risque selon la volatilité passée de chaque stratégie en stress | −0,041 | 0,073 | 16,8 % | −0,068 / −0,052 / −0,038 | inchangée | **pas utile** |
| **R_MEAN** | réalloue le risque vers les stratégies qui ont gagné par le passé en stress | +0,034 | 0,107 | 97,0 % | −0,022 / −0,012 / +0,007 | −24,4 → −22,6 % | **sous-puissant, contaminé** |
| **K_HALF** | divise le risque total du livre par deux | +0,002 | 0,209 | 77,5 % | −0,199 / −0,009 / −0,039 | **−24,4 → −17,4 %** | **sous-puissant** (nul) |

**Aucun bras n'est utile.**
- **Réallouer selon la variance (R_VAR) coûte un peu**, et les règles de volatilité aussi.
- **Réallouer selon la performance passée (R_MEAN) gagne un peu, mais c'est déclaré
  contaminé avant la lecture.** On savait déjà que la tendance crypto et l'overnight
  gagnent en stress. Pendant l'épisode 2020-2021, la règle l'apprend au bout de 63
  séances de stress, puis en profite jusqu'en mars 2021. Tout le gain vient de là :
  +9,2 % contre le socle d'octobre 2020 à mars 2021. En 2007-2009, l'écart est à peu près
  nul (−0,6 %, −0,2 % et +0,8 % selon le sous-épisode). **C'est une piste, pas un
  résultat.** Elle ne se confirmera que sur un épisode de stress que personne n'a encore
  vu.
- **Couper le risque total (K_HALF) ne change pas le Sharpe, mais la forme du risque** :
  la perte maximale passe de −24,4 % à −17,4 %. Il le paie au rebond de 2020 (−14,6 %
  contre le socle de mars à août 2020). **Les trois règles de volatilité obtiennent la
  même perte maximale** (−17,3 % à −17,4 %) : ce gain n'a rien de propre au régime.

**La règle opposable.** R_VAR et R_MEAN différaient sur deux axes, chiffrés avant la
lecture :
- **l'usage** : 11 % et 20 % du budget de risque déplacés entre stratégies en stress ;
- **la dimension de l'objet** : 8,10 contre un seuil de 4.

**Cela n'a pas suffi.** La latente et l'horloge restaient celles des sept dispositifs
précédents : 0,58 transition par an, et deux vrais épisodes dans le livre.

## Trois messages pour la présentation

1. **Le levier, c'est la diversification, pas le régime.** Neuf stratégies moyennes
   (0,58 de Sharpe en moyenne, seules) font un livre à 1,91 sur dix ans, sans aucun
   filtre. C'est la loi fondamentale de la gestion active : ce qui compte, c'est le
   nombre de paris indépendants.
2. **Le régime, même placé au bon endroit, ne change pas ce Sharpe.** Il peut réduire la
   perte maximale en coupant le risque en stress, mais une règle de volatilité d'une
   ligne en fait autant.
3. **Le chiffre de 1,5 à 1,9 est brut, et c'est un plafond.** La suite logique est de le
   remesurer à coûts institutionnels, et sur des dates postérieures à la conception de
   chaque stratégie.

## Limites

- Zéro coût. La rotation des expositions (10 par an pour le socle) ne compte pas la
  rotation interne de chaque stratégie.
- Les stratégies ont été choisies en connaissant les données.
- Le livre ne voit que deux épisodes de stress : 2007-2009 avec 3 ou 4 stratégies, et
  2020-2021 avec 9. L'épisode 2002-2003 tombe avant le début du livre.
- La cassure d'ouverture compte 1 R pour 0,5 % du capital. L'overnight est un proxy sur
  le future Nasdaq.
- Le détail, la construction des stratégies et la sortie brute sont dans le dépôt privé
  voisin. Ils ne sont pas publiés ici.
