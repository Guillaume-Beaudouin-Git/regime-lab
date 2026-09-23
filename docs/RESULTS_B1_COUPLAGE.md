# Rebond obligataire de fin de mois, seul et couplé au régime

**Question pour la présentation.** Le classifieur de l'étude (Sparse Jump Model, famille
A′) change-t-il ce que rapporte une stratégie simple et documentée quand on les couple ?

**Stratégie testée.** Le rebond de fin de mois des Treasuries, un effet de flux lié au
rééquilibrage des indices obligataires. On est long TLT sur les trois dernières séances
américaines de chaque mois. La position est ciblée en volatilité, le coût est de 1 bp
aller-retour, et les rendements sont en excès du taux cash.

Le protocole et le critère de décision ont été commités **avant tout calcul**
(`b737ea9`). Le script est `scripts/run_b1_regime_coupling.py`. La lecture a été faite
une fois, et trois essais sont journalisés sous `b1_regime_coupling` (le troisième,
l'arrêt, a été ajouté après lecture).

## Les couplages

- **(b) Réduction** : on divise la position par deux quand le régime de la veille est
  « stress ».
- **(c) Bascule** : en stress, on tient le livre de tendance du programme (46
  instruments, cible de volatilité de 10 %) au lieu de la stratégie de fin de mois. Ce
  choix a été fait avant la lecture : le suivi de tendance est la famille connue pour
  ses gains en crise.
- **(a) Arrêt** *(ajouté après la lecture de (b) et (c))* : aucune position quand le
  régime de la veille est « stress ».

Chaque couplage est comparé à la stratégie seule. Il est aussi refait avec la règle de
volatilité d'une ligne, et confronté à 400 rotations aléatoires du régime (placebo).

## Résultats — octobre 2004 à septembre 2026, 5 720 séances, nets de coûts

| | Sharpe | rendement/an | volatilité | perte max | pire mois | crise 2008 | Covid 2020 | 2022 |
|---|---|---|---|---|---|---|---|---|
| **Stratégie seule** | **0,79** | +7,5 % | 9,4 % | −18,3 % | −7,0 % | +8,7 % | +5,8 % | +5,4 % |
| (b) Réduction en stress | 0,82 | +7,2 % | 8,8 % | −18,3 % | −7,0 % | +7,1 % | +5,8 % | +5,4 % |
| (c) Bascule vers la tendance | 0,75 | +7,0 % | 9,4 % | −18,3 % | −7,0 % | **+17,6 %** | +2,7 % | +5,4 % |
| (a) Arrêt en stress *(ajouté après lecture)* | 0,81 | +7,0 % | 8,6 % | −18,3 % | −7,0 % | +5,2 % | +5,8 % | +5,4 % |
| (a) avec la règle de volatilité | 0,53 | +3,5 % | 6,7 % | −10,8 % | −7,0 % | 0,0 % | 0,0 % | 0,0 % |
| (b) avec la règle de volatilité | 0,74 | +5,5 % | 7,5 % | −14,4 % | −7,0 % | +4,5 % | +2,9 % | +2,7 % |
| (c) avec la règle de volatilité | 0,40 | +3,9 % | 9,8 % | −19,2 % | −6,6 % | +15,6 % | −13,8 % | +27,5 % |
| Livre de tendance seul | 0,41 | +4,3 % | 10,6 % | −29,6 % | −8,9 % | +18,1 % | −13,8 % | +27,5 % |

Les crises sont datées par les sommets et creux du S&P 500, pas par le régime : 2008 va
du 09/10/2007 au 09/03/2009, le Covid du 19/02 au 23/03/2020, et 2022 du 03/01 au
12/10/2022.

## Verdicts, selon le critère écrit avant la lecture

- **Réduction : SOUS-PUISSANT, donc pas utile.** Le Sharpe gagne +0,023, alors que
  l'écart détectable est de 0,106. Le test HAC est de signe opposé (t −0,70), le
  résultat se situe au 87e percentile du placebo, et la règle de volatilité fait moins
  bien (−0,058).
- **Bascule : PAS UTILE.** Le Sharpe perd 0,047 (et 0,056 à 2 bp). Le résultat se situe
  au 56e percentile du placebo.
- **Arrêt en stress : SOUS-PUISSANT, donc pas utile.** Ce couplage a été ajouté à la
  demande de Guillaume **après** la lecture des deux premiers. Il est déclaré comme tel
  (`f2a017a`), et son seuil est corrigé pour trois couplages, soit 0,05/3. Le Sharpe
  gagne +0,013, alors que l'écart détectable est de 0,229. Le test HAC est de signe
  opposé (t −0,70), le résultat se situe au 85e percentile du placebo, et la perte
  maximale ne change pas. S'arrêter en 2008 **coûte** 3,5 points (+5,2 % contre +8,7 %),
  parce que la stratégie y gagnait de l'argent. La même coupure avec la règle de
  volatilité réduit la perte maximale (−10,8 %), mais divise presque le Sharpe par deux
  (0,53), car elle coupe la stratégie la moitié du temps.

## Pourquoi : le mécanisme, mesuré

1. **La stratégie gagne dans les deux régimes.** Les jours investis en stress rapportent
   +12 bp par jour (98 séances), contre +23 bp en calme (715 séances). Elle est plus
   faible en stress, mais reste positive. La réduire de moitié en stress enlève autant
   de rendement que de risque, et le Sharpe ne bouge presque pas.
2. **Elle ne craint pas les crises.** Elle est positive en 2008, en 2020 et en 2022.
   C'est un effet de calendrier et de flux, pas une prime de risque : il n'y a rien à
   protéger.
3. **La bascule échange une crise contre une autre.** La tendance double le gain de 2008
   (+17,6 % contre +8,7 %). Mais elle perd au retournement en V du Covid : le gain passe
   de +5,8 % à +2,7 % sur la fenêtre Covid, et le livre de tendance seul y perd 13,8 %.
   Au total, la bascule ne fait rien gagner.
4. **La pire perte de la stratégie, −18,3 %, est récente** : d'août 2025 à septembre
   2026. Or le régime est en état calme sans interruption depuis avril 2021. Aucun
   couplage ne pouvait donc la voir. Elle vient d'un affaiblissement de l'effet : la
   fenêtre de fin de mois a rapporté en moyenne **+3 bp en 2025-2026, contre +36 bp sur
   2004-2024**, sur les rendements bruts de TLT.

## Ce qu'on peut en dire en présentation

C'est un cas **« stratégie insensible au régime »** : le filtre ne l'améliore pas et ne
la protège pas, parce qu'elle n'a pas de risque de crise à couper. Le résultat est
honnête et utile dans une comparaison à trois cas, où il tient le rôle du cas neutre.
Il faut le placer à côté d'une stratégie que le filtre aide et d'une qu'il pénalise.

**Limites.** TLT remplace le future (le coût et la duration diffèrent). L'échantillon
commence en octobre 2004, ce qui écarte l'épisode de stress 2002-2003, car le livre de
tendance ne vit qu'à partir de là. Le régime n'a connu que deux épisodes de stress dans
cet échantillon (2007-2009 et 2020-2021).
