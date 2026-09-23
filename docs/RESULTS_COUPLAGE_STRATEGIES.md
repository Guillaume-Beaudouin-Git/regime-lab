# Des stratégies de trading, seules ou couplées au régime — synthèse pour la présentation

**Question.** Le classifieur de l'étude (Sparse Jump Model, famille A′) améliore-t-il
de vraies stratégies algorithmiques quand on les couple à lui ?

**Trois façons de coupler**, figées avant chaque lecture. Chacune agit selon l'état de la
veille :
- **(a) Arrêt** : on ne trade pas en état de stress.
- **(b) Réduction** : position divisée par deux en stress.
- **(c) Bascule** : en stress, on tient le livre de tendance du programme (46
  instruments) à la place de la stratégie.

**Même critère pour tous**, écrit avant la lecture. Un couplage est « utile » seulement
s'il remplit quatre conditions :
1. il gagne plus de Sharpe que l'écart minimal détectable, corrigé pour le nombre de tests ;
2. le test HAC va dans le même sens ;
3. il fait mieux que 95 % d'un placebo qui déplace les jours de stress au hasard ;
4. il bat le même couplage fait avec une simple règle de volatilité.

**Hypothèse de coûts : aucun coût, partout.** C'est un choix de simplicité pour ce
projet, et les Sharpe ci-dessous sont donc **bruts**.

## Résultats — 8 stratégies, 24 couplages testés

| stratégie | période | crises vues par le filtre | Sharpe seule | Δ arrêt | Δ réduction | Δ bascule | verdict |
|---|---|---|---|---|---|---|---|
| Rebond obligataire de fin de mois | 2004-2026 | 2008, 2020 | 0,85 | +0,01 | +0,02 | −0,04 | **neutre** |
| Tendance or | 2007-2026 | 2008, 2020 | 0,62 | −0,04 | −0,01 | −0,05 | pas utile |
| Tendance or + argent | 2004-2026 | 2008, 2020 | 0,58 | −0,06 | −0,02 | −0,07 | pas utile |
| Momentum USDJPY | 2009-2026 | 2020 | 0,50 | −0,01 | −0,00 | +0,03 | neutre |
| Tendance énergie, long seul | 2011-2026 | 2020 | −0,17 | −0,11 | −0,06 | −0,02 | pas utile |
| **Tendance crypto** (BTC, ETH) | 2014-2026 | 2020 | 1,13 | **−0,24** | −0,11 | −0,21 | **pénalisée** |
| **Prime overnight Nasdaq** | 2010-2026 | 2020 | 1,10 | **−0,12** | −0,03 | −0,11 | **pénalisée** |
| Cassure d'ouverture (ORB) Nasdaq | 2010-2026 | 2020 | 0,64 | +0,00 | +0,01 | +0,03 | neutre |

Pour le rebond obligataire, les chiffres de ce tableau sont bruts, pour s'aligner sur le
reste. Sa lecture de référence, à 1 bp, est dans `docs/RESULTS_B1_COUPLAGE.md` (0,79
seule). Les verdicts sont identiques.

**Aucun couplage n'est utile, sur aucune stratégie.** Deux stratégies sont nettement
pénalisées :
- **la tendance crypto** : l'arrêt en stress fait passer son Sharpe de 1,13 à 0,89, avec
  un t de −2,56 et une perte maximale qui s'aggrave de −15 % à −20 % ;
- **la prime overnight Nasdaq** : t −2,37. Placer les jours de stress au hasard fait
  mieux dans 100 % des 400 tirages du placebo.

## Trois messages pour la présentation

1. **Le filtre reconnaît bien le stress, mais le stress n'est pas toujours un mauvais
   moment.** La prime overnight rémunère précisément le fait de porter le risque en
   période agitée. Et l'épisode de stress 2020-2021 du filtre couvre surtout le rebond
   et le marché haussier de la crypto. Couper en stress, c'est couper les meilleurs
   jours de ces stratégies.
2. **Une stratégie de flux, comme le rebond de fin de mois, est insensible au régime.**
   Elle gagne aussi en crise ; il n'y a rien à protéger.
3. **Le filtre peut protéger en crise sans améliorer la stratégie.** Pour l'or, l'arrêt
   transforme 2008 d'une perte de −1,8 % en un gain de +10,1 %, mais le Sharpe sur
   l'ensemble de la période baisse légèrement. *Cette observation a été faite après la
   lecture : c'est une illustration, pas un résultat.*

C'est le résultat central de l'étude, retrouvé sur de vraies stratégies : **le régime
décrit le risque, pas le sens des rendements.** C'est pourquoi il sert à dimensionner et
non à couper.

## Limites, à dire à l'oral

- Tous les Sharpe sont bruts de coûts.
- Le filtre ne connaît que trois épisodes de stress, et il est calme depuis avril 2021.
  Six stratégies sur huit ne voient qu'une seule crise, 2020.
- Choisir les stratégies les plus pénalisées gonfle mécaniquement leurs écarts. C'est
  pourquoi les verdicts et le placebo sont montrés avec les écarts.
- Le détail des sept stratégies privées, leur construction et leurs 21 lectures sont
  dans une étude du dépôt privé voisin, qui n'est pas publiée ici.
