# Des stratégies de crise, seules et couplées au régime — résultats

**En une phrase.** Sur quatre stratégies choisies parce qu'elles vivent ou meurent par
la volatilité et les crises, le filtre de régime **change beaucoup la forme du risque et
presque pas le Sharpe**. Il évite l'essentiel des pertes de 2008, mais il rate la prime
qui suit la crise. Le seul cas où il améliore nettement la stratégie est le **momentum**
(Sharpe de 0,52 à 0,69), mais une simple règle de volatilité fait autant. Et la raison
de fond est mesurée : **le VIX sait déjà ce que sait le modèle.**

- Pré-enregistrement : `docs/PRESPEC_CRISE.md`, commit **`7d5b95c`**, avant toute lecture.
- Lecture, faite une fois : `scripts/run_crisis_coupling.py --read`, sortie intégrale
  dans `docs/artifacts/crise/reading.txt` (commit `d3e7c48`).
- Neuf essais journalisés, famille `crise_coupling` de `data/trials.parquet`.
- **Aucun coût, partout** (décision du 23/09). Les Sharpe sont bruts et en excès du taux
  sans risque. La rotation est rapportée.

---

## 1. La question, en clair

Le modèle (Sparse Jump Model) reconnaît bien les périodes de stress. Mais il prévoit
**l'ampleur** des mouvements futurs, pas leur **sens**. Jusqu'ici, on l'a branché sur des
stratégies dont le gain ne dépend pas de l'ampleur des mouvements (actions, obligations,
tendance) : il ne pouvait que réduire la taille des positions, et une règle d'une ligne
sur la volatilité le faisait aussi bien.

L'idée testée ici : le brancher sur des stratégies **dont le gain dépend directement de
la volatilité ou des crises**. Sur ces objets, savoir que la volatilité va monter, c'est
savoir que la stratégie va perdre.

## 2. Ce qui a été testé

| code | stratégie | ce qu'elle gagne, ce qu'elle risque | période |
|---|---|---|---|
| **VRP** | vente de volatilité sur le S&P 500 (swap de variance à 1 mois, reconstitué) | encaisse l'écart entre la volatilité annoncée par le VIX et celle qui se réalise ; perd lourdement quand la volatilité explose | 2002-2026 |
| **VIXF** | vente de futures VIX à 1 mois (reconstitués depuis la bourse CBOE) | encaisse la décote des futures ; perte maximale en crise, y compris en février 2018 | 2006-2026 |
| **UMD** | momentum actions (facteur de Ken French) | achète les gagnants, vend les perdants ; s'effondre au rebond qui suit un marché baissier (2009) | 2002-2026 |
| **BXM** | indice CBOE « achat-vente de call » | le produit public de la vente d'options ; perd comme les actions en crise, plafonné à la hausse | 2002-2026 |

Chaque stratégie est **ciblée à 10 % de volatilité**, comme partout dans le programme. Le
filtre agit de deux façons, selon l'état de la veille :

- **arrêt** : aucune position en stress ;
- **moitié** : position divisée par deux en stress.

À côté, un **test de prédiction (P)** : l'état prévoit-il la volatilité future **mieux que
le VIX** ? C'est la façon la plus directe de « mieux utiliser la prédiction ». Si oui, on
sait vendre de la volatilité au bon moment.

## 3. Résultat principal — arrêt en stress

| stratégie | Sharpe seule | Sharpe avec filtre | Δ | MDE | Δ avec la règle de vol. (médiane / 80ᵉ c.) | perte max. seule → avec filtre | verdict |
|---|---|---|---|---|---|---|---|
| VRP, vente de variance | 1,30 | 1,30 | −0,005 | 0,426 | −0,269 / +0,090 | −49,3 % → −42,0 % | pas utile |
| VIXF, vente de futures VIX | 0,35 | 0,32 | −0,028 | 0,348 | −0,242 / +0,018 | −36,4 % → −33,4 % | pas utile |
| **UMD, momentum** | **0,52** | **0,69** | **+0,166** | 0,395 | +0,002 / **+0,163** | −26,2 % → −20,9 % | **sous-puissant** |
| BXM, achat-vente de call | 0,44 | 0,40 | −0,039 | 0,259 | −0,203 / −0,053 | −31,4 % → −28,9 % | pas utile |

Pour lire le tableau :
- Le **MDE** est le plus petit écart de Sharpe que l'échantillon permet de distinguer du
  hasard, corrigé pour les 9 tests. Un écart plus petit ne prouve rien.
- La **règle de volatilité** est le témoin. La « médiane » coupe quand la volatilité du
  S&P 500 dépasse sa médiane passée, soit la moitié du temps. Le « 80ᵉ c. » coupe au-delà
  du 80ᵉ centile passé, soit environ 22 % du temps, un ordre proche des 16 % de stress
  du modèle.

### Réduction de moitié en stress (levier dépendant du régime)

| stratégie | Sharpe seule | avec filtre | Δ | MDE | Δ règle de vol. (médiane / 80ᵉ c.) | perte max. avec filtre | verdict |
|---|---|---|---|---|---|---|---|
| VRP | 1,30 | 1,32 | +0,017 | 0,203 | −0,046 / +0,077 | −45,7 % | sous-puissant |
| VIXF | 0,35 | 0,34 | −0,009 | 0,167 | −0,091 / +0,016 | −34,3 % | pas utile |
| UMD | 0,52 | 0,62 | +0,094 | 0,185 | +0,048 / +0,098 | −20,9 % | sous-puissant |
| BXM | 0,44 | 0,43 | −0,012 | 0,124 | −0,058 / −0,013 | −29,9 % | pas utile |

Le +0,017 de VRP passe **entièrement par le dénominateur**. Le rendement annuel baisse
(19,8 % → 19,1 %) et le t HAC de l'écart quotidien est négatif (−1,12) : c'est moins de
risque, pas plus de rendement.

## 4. Le P&L dans chaque crise — là où le filtre agit vraiment

Rendement cumulé de chaque stratégie ciblée, **seule → avec arrêt en stress**. Les
fenêtres sont datées par le S&P 500, pas par le modèle.

| | crise 2008 (10/07 → 03/09) | rebond 2009 (03/09 → 12/09) | fév. 2018 | Covid (02/20 → 03/20) | rebond 2020 (03/20 → 12/20) | 2022 |
|---|---|---|---|---|---|---|
| VRP | −20,4 % → **+1,0 %** | +36,6 % → **+7,4 %** | −35,1 % → −35,1 % | −35,7 % → −27,8 % | +22,8 % → +4,0 % | +3,2 % → +3,2 % |
| VIXF | −18,5 % → −4,3 % | +25,0 % → +3,3 % | −25,5 % → −25,5 % | −17,3 % → −14,1 % | +8,4 % → +0,8 % | −5,8 % → −5,8 % |
| UMD | +25,3 % → +10,9 % | **−17,4 % → +2,3 %** | +0,6 % → +0,6 % | +7,9 % → +10,6 % | −8,9 % → +2,0 % | +7,6 % → +7,6 % |
| BXM | −19,7 % → −3,5 % | +25,2 % → +1,1 % | −17,4 % → −17,4 % | −18,7 % → −15,0 % | +15,2 % → +2,3 % | −14,2 % → −14,2 % |

**Février 2018 et 2022 sont identiques avec ou sans filtre** : le modèle était en état
calme. Le krach de la volatilité de février 2018 coûte 35 % à la vente de variance et
25,5 % à la vente de futures VIX, et le filtre ne le voit pas.

## 5. Le test P — le VIX sait déjà ce que sait le modèle

On prévoit la volatilité réalisée des 21 séances suivantes (en log), sur 6 129 séances
hors échantillon.

| ce qu'on met dans la régression | apport de l'état (R² incrémental) | t de l'état |
|---|---|---|
| rang de la volatilité passée seulement | **+4,48 points** | −5,23 |
| rang de la volatilité passée **et le VIX** | **+0,20 point** | −1,34 |

Seuil pré-enregistré : |t| ≥ 2,77. L'apport est au 79ᵉ centile du placebo, qui
demandait le 95ᵉ. En échantillon mensuel sans chevauchement : t −0,77.

**Verdict : l'état NE PRÉDIT PAS la volatilité au-delà du VIX.** Sans le VIX, on retrouve,
sous une autre forme (volatilité en log, autre échantillon), le résultat central de
l'étude : l'état sait des choses sur la volatilité future qu'un quantile de volatilité
passée ne sait pas. Mais le VIX, qui est le prix de marché de
cette volatilité, contient déjà presque tout (4,48 points ramenés à 0,20). C'est pour
cela qu'on ne peut pas vendre de la volatilité « mieux » grâce au modèle : **le marché
des options a déjà intégré l'information.**

## 6. Verdicts, selon le critère écrit avant la lecture

- **Test P : ne prédit pas au-delà du VIX.** La prémisse qui devait distinguer la vente de
  variance des sept échecs tombe. Le pré-enregistrement l'avait écrit (§4.1) : dans ce
  cas, VRP et VIXF redeviennent des septièmes dispositifs. C'est ce qu'on observe. Ni
  l'un ni l'autre ne bat la règle du 80ᵉ centile (VRP −0,005 contre +0,090 ; VIXF −0,028
  contre +0,018).
- **VRP, VIXF, BXM : pas utiles.** Seule exception de forme : la moitié sur VRP est
  « sous-puissante » (+0,017), et ce gain passe par le seul dénominateur (§3). Tous les
  écarts sont très sous le MDE et entre le 47ᵉ et le 68ᵉ centile du placebo.
- **UMD : sous-puissant, arrêt comme moitié.** C'est le seul écart franchement positif :
  +0,166, au **100ᵉ centile du placebo** (le 95ᵉ est à +0,073). Il reste sous le MDE de
  0,395, et la règle du 80ᵉ centile obtient le même gain (+0,163). Le filtre aide le
  momentum, bien au-delà du hasard, mais pas au-delà d'une règle de volatilité bien
  calibrée. Le gain passe pour partie par le rendement (+5,6 % → +6,6 % par an, t HAC
  1,05, non significatif) et pour partie par le risque (volatilité 10,6 % → 9,6 %).
- **Aucun couplage n'est « pénalisant »** au sens du critère.

Ce que l'on attendait, écrit avant la lecture (`PRESPEC_CRISE.md` §8) : P prédit avec une
probabilité d'environ 0,25, il ne prédit pas ; UMD sous-puissant avec un Δ entre 0 et
+0,3, c'est ce qui sort ; BXM septième dispositif, c'est ce qui sort. Pour VRP et VIXF, on
attendait un |Δ| « grand, de signe incertain ». Il est presque nul, parce que la perte
évitée en crise et le gain manqué au rebond se compensent.

## 7. Pourquoi : le mécanisme

**Pour la vente de volatilité (VRP, VIXF, BXM), le filtre déplace le P&L sans le créer.**
Le modèle entre en stress le 14 janvier 2008 et en sort en septembre 2009, puis
définitivement en novembre 2009. Il évite ainsi la crise
(VRP : −20,4 % → +1,0 %). Mais il reste dehors pendant le rebond, quand la volatilité
annoncée est encore très haute et que la volatilité réalisée retombe : c'est là que la
prime est la plus riche (VRP : +36,6 % → +7,4 %). Même chose en 2020. En stress, la vente
de variance garde un Sharpe positif (0,60, contre 1,41 en calme). VIXF et BXM gagnent
autant en stress qu'en calme (0,38 contre 0,34 ; 0,47 contre 0,43). Couper en stress,
c'est couper une période qui rapporte encore.

**Pour le momentum, le filtre coupe une vraie mauvaise période.** Seul, UMD a un Sharpe
de **−0,58** sur les séances de stress et de **+0,75** en calme. Les états de stress du
modèle recouvrent les marchés baissiers où le momentum s'effondre au rebond (Daniel et
Moskowitz 2016) : κ 0,50 avec l'état baissier, contre 0,31 pour la règle de volatilité
(mesuré avant la lecture, `PRESPEC_CRISE.md` §4.2). En 2009, l'arrêt transforme −17,4 %
en +2,3 %. Mais la règle du 80ᵉ centile coupe aussi 2009 et fait le même Sharpe.

**Le ciblage de volatilité fait déjà une grande partie du travail.** Sur le facteur UMD
**brut**, sans ciblage (lu à titre descriptif), le filtre fait passer le Sharpe de 0,15 à
0,52, la perte maximale de −63,3 % à −26,9 %, et le rebond 2009 de −54,9 % à +2,8 %. Le
seul ciblage de volatilité porte déjà le Sharpe à 0,52. Barroso et Santa-Clara (2015)
l'avaient montré ; le filtre ajoute +0,17 par-dessus.

## 8. Ce qu'on peut dire en présentation

Trois stratégies se prêtent à la diapositive « stratégie seule contre stratégie +
filtre », avec un message chacune.

1. **Momentum : le filtre aide, et c'est au-delà du hasard.** Sharpe 0,52 → 0,69, perte
   maximale −26 % → −21 %, rebond de 2009 −17 % → +2 %. Aucun des 400 placements
   aléatoires du stress ne fait aussi bien. *À dire aussi* : l'écart reste sous le seuil
   de détection, et une règle de volatilité bien réglée fait autant.
2. **Vente de volatilité : le filtre change la forme du risque, pas la rentabilité.**
   Crise de 2008 : −20 % → +1 %. Rebond de 2009 : +37 % → +7 %. Sharpe 1,30 → 1,30. Le
   filtre évite la crise et rate la prime qui la suit. Et il ne voit pas février 2018
   (−35 % dans les deux cas).
3. **Pourquoi : le VIX sait déjà.** Sans le VIX, le modèle ajoute 4,5 points de R² sur la
   volatilité future ; avec le VIX, 0,2 point. L'information du modèle est déjà dans le
   prix des options.

La phrase de synthèse : **le modèle décrit bien le risque ; là où ce risque a un prix de
marché (la volatilité), le prix le contient déjà ; là où il n'en a pas (le momentum), le
modèle aide, autant qu'une bonne règle de volatilité.**

## 9. Ce qui est prometteur pour l'objectif de fond

- **Le momentum conditionné à l'état de panique** est, dans cette étude, le seul objet
  où l'état porte une latente mesurée autre que la volatilité (κ 0,50 avec le marché
  baissier) et où l'effet sort du placebo (100ᵉ centile). Suite possible, à pré-enregistrer : tester
  l'indicateur de Daniel et Moskowitz lui-même sur 1926-2026 (UMD est disponible depuis
  1926), avec la règle du 80ᵉ centile comme témoin et des coûts réalistes. Le facteur
  n'est pas investissable tel quel.
- **La vente de variance** fait le Sharpe le plus élevé des quatre (1,30), mais à coût
  nul, sur un objet synthétique, et **ce n'est pas un résultat de régime**. Elle
  mériterait une étude propre (coûts, strike réel des swaps, futures plutôt qu'indice),
  avec le risque de février 2018 au centre.
- **Pour les couplages de type interrupteur, la règle médiane du programme est un témoin
  trop faible** : elle coupe la moitié du temps. La règle du 80ᵉ centile, à part de stress
  comparable, fait mieux que l'état sur VRP et VIXF, autant sur UMD (+0,163 contre
  +0,166 pour l'arrêt) et un peu moins bien sur BXM (−0,053 contre −0,039, les deux
  perdent). C'est le témoin à garder pour la suite.

## 10. Limites

- **Aucun coût.** UMD tourne son poids 2,9 fois par an ici, mais le facteur lui-même se
  rééquilibre chaque mois sur tout l'univers, et ce coût n'est pas compté. La rotation
  interne de VRP (1/21 des tranches chaque jour) et le roulement de VIXF non plus.
- **VRP est synthétique.** Le strike est pris égal à VIX² (un vrai swap se traite un peu
  au-dessus). La réévaluation quotidienne suppose une structure par terme plate. Seul le
  gain final de chaque tranche est exact. Le ciblage vise 10 % mais réalise 15,2 %, à
  cause des sauts.
- **VIXF est reconstitué** depuis les règlements CBOE. Il suit le VXX avec une
  corrélation hebdomadaire de 0,987 sur 2018-2026 (`scripts/check_vix_futures_proxy.py`,
  sortie dans `docs/artifacts/crise/vix_futures_check.txt`). Mais sur les 5 et 6 février
  2018, il fait +45,8 % contre +31,3 % pour le VXX. Le décalage d'horaire entre le
  règlement et la clôture n'explique qu'une partie de l'écart ; le reste n'est pas
  expliqué. Ce jour-là, l'état était calme : les deux bras de chaque couplage portent la
  même perte, et l'effet sur l'écart de Sharpe n'est que de second ordre (non mesuré).
  Il pèse en revanche sur le niveau et la perte maximale de VIXF.
- **Trois épisodes de stress** (2002-2003, 2008-2009, 2020-2021), dont deux seulement
  pour VIXF. Le modèle est calme depuis avril 2021 : février 2018 et 2022 lui échappent.
- **Les MDE sont grands** (0,12 à 0,43), à cause des 9 tests (Bonferroni) et du petit
  nombre d'épisodes. Pour qu'un arrêt soit déclaré utile, il fallait un gain de 0,26 à
  0,43 de Sharpe selon la stratégie.
- **Le test P** utilise, comme l'étude, un rang de volatilité calculé sur tout
  l'échantillon. C'est un test de contenu prédictif, pas une règle de trading.
- **Les fenêtres de crise** ont été fixées avant la lecture. Les explications du §7, elles,
  ont été écrites après : ce sont des lectures du mécanisme, pas des résultats testés.

## Reproduire

```bash
.venv/bin/python scripts/fetch_crisis_data.py          # données publiques, ~2 min
.venv/bin/python scripts/run_crisis_coupling.py        # l'instrument, aucune lecture
.venv/bin/python scripts/check_vix_futures_proxy.py    # contrôle de VIXF contre le VXX
# la lecture a été faite une fois ; la relancer ajoute 9 essais au registre :
# .venv/bin/python scripts/run_crisis_coupling.py --read
```

Yahoo, CBOE et Ken French révisent parfois leurs séries : un nouveau téléchargement peut
déplacer les chiffres. Les empreintes SHA-256 des fichiers lus sont dans l'instrument et
dans `PRESPEC_CRISE.md` §5.
