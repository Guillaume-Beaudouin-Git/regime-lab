# Actions en temps calme, valeur refuge en stress : le test

**Question de Guillaume, 23/09.** Un détecteur de crises rapporte-t-il davantage si l'on
**achète** en stress un actif qui gagne dans les crises, plutôt que de simplement couper
une exposition ? Deux idées étaient sur la table : un actif refuge (or, obligations
d'État) ou un straddle.

**Protocole.** Il est écrit dans le script `scripts/run_safe_haven_switch.py`, commité
**avant lecture** en `5872d6b`. Il y a eu une seule lecture, soit trois essais
`safe_haven_switch` dans `data/trials.parquet`. La sortie brute est dans
`docs/artifacts/refuge/reading.txt`.
- **Base** : les actions américaines (facteur marché de Ken French, dividendes compris),
  ciblées à 10 % de volatilité.
- **Bascule** : on garde les actions quand l'état connu la veille est calme. En stress,
  on tient à la place l'une de trois jambes, chacune ciblée à 10 % de volatilité :
  - **TLT**, des obligations d'État US à plus de 20 ans ;
  - **l'or**, via le future GC=F ;
  - **de la volatilité achetée** : le swap de variance synthétique de l'étude de crise,
    acheté au lieu de vendu. C'est l'équivalent d'un straddle couvert en delta.
- **Hypothèses** : aucun coût, rendements en excès du taux sans risque.
- **Période** : octobre 2002 à juillet 2026, 5 977 séances. TLT n'existe que depuis
  juillet 2002, et sa volatilité doit être estimée sur 63 séances.
- **Critère** : le même que pour l'étude de crise. La bascule doit dépasser le seuil de
  détection, avec un t HAC de même signe et au-dessus de 95 % du placebo. Elle doit aussi
  battre la même bascule pilotée par les règles d'une ligne sur la volatilité (la
  médiane, et le 80ᵉ centile).

## Résultats

| en stress, on tient… | Sharpe (actions seules : 0,64) | Δ | seuil | perte max. (seules : −24,9 %) | 2008 | rebond 2009 | Covid | verdict |
|---|---|---|---|---|---|---|---|---|
| des liquidités (référence) | 0,60 | −0,04 | — | −17,3 % | −6,8 % | −0,6 % | −13,0 % | — |
| **des obligations d'État (TLT)** | 0,57 | −0,07 | 0,43 | −24,7 % | **+0,1 %** | −6,3 % | −11,3 % | pas utile |
| **de l'or** | **0,68** | **+0,04** | 0,31 | **−18,5 %** | −9,0 % | +14,2 % | −15,7 % | sous-puissant |
| **de la volatilité (straddle)** | 0,39 | −0,25 | 0,73 | −28,5 % | **+14,8 %** | −22,3 % | −3,2 % | pas utile |

Pour comparer, les actions seules font −22,9 % en 2008, +20,3 % au rebond de 2009 et
−17,0 % pendant le Covid.

**Les témoins.** La bascule vers l'or pilotée par la simple règle médiane de volatilité
fait **mieux** que le modèle : Sharpe 0,77 contre 0,68. Pour TLT et pour la
volatilité, les règles de volatilité et le modèle échouent tous.

## Pourquoi ça ne marche pas : les périodes de stress contiennent aussi le rebond

La ligne la plus instructive de la lecture est descriptive. Une fois ciblées en
volatilité, les **actions rapportent autant en stress qu'en calme** : Sharpe 0,62 contre
0,64. Les états de stress du modèle couvrent la chute, mais aussi les rebonds qui suivent :
2003, mars à septembre 2009, avril 2020 à mars 2021.
- Basculer vers les **obligations** protège parfaitement en 2008 (+0,1 % au lieu de
  −22,9 %), mais fait rater le rebond de 2009 : −6,3 % au lieu de +20,3 %.
- Acheter de la **volatilité** gagne +14,8 % en 2008, mais la volatilité se paie cher en
  stress, puisqu'elle y perd 11,7 % par an en moyenne. On perd −22,3 % au rebond de 2009.
  Le modèle bascule aussi tard pour le Covid : il passe en stress le 11 mars 2020, quand
  le VIX est déjà passé de 14 à 54 (son pic est à 83, le 16 mars). La volatilité est donc
  déjà chère au moment où on l'achète.
- **L'or** est la seule jambe qui gagne davantage en stress (Sharpe 0,90) qu'en calme
  (0,63). D'où le seul Δ positif et la meilleure perte maximale. Mais l'écart est huit fois
  sous le seuil de détection, et une règle de volatilité fait mieux.

## Ce qu'on peut dire en présentation

1. **Le modèle protège bien pendant la chute.** Avec les obligations en stress, 2008
   passe de −22,9 % à 0 %. Avec la volatilité achetée, 2008 devient même +14,8 %.
2. **Mais il reste en mode crise pendant le rebond, et c'est là qu'il perd ce qu'il avait
   gagné.** C'est le prix d'un modèle calme, qui ne change d'avis que rarement : il sort
   de la crise tard.
3. **L'or est la meilleure valeur refuge des trois**, mais sans résultat démontrable, et
   une simple règle de volatilité fait aussi bien.

## Limites

- Tout est sans coût. L'achat de volatilité est synthétique : le prix d'exercice est égal
  au VIX, et il n'y a pas de vraies options.
- L'or est pris sur le future continu de Yahoo, non ajusté. Sa variation quotidienne
  approche celle du prix spot.
- Il n'y a que trois épisodes de stress, et le premier est tronqué au début de
  l'échantillon (octobre 2002).
- La règle opposable de `CLAUDE.md` s'applique pleinement. La latente est ordonnée par la
  volatilité, l'horloge fait moins de 2 transitions par an, l'usage est une bascule, et
  l'objet est de dimension 1. Ce test est donc un nouveau cas de la même famille, et il
  tombe de la même façon.
