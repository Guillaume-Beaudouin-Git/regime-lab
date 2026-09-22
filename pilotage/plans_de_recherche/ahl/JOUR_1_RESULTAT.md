# AHL — jour 1 : l'instrument est vérifié, et il voit

23 septembre 2026. **Aucun essai du registre n'a été dépensé.** Reproductible par
`regime-lab/scripts/run_ahl_panel_power.py` (commit `630ee16`).

## Étape 1 — le départ d'échantillon, tranché

**2003-07-17, 6 039 séances** : lecture stricte, 252 séances *strictement avant* t.
La permissive, qui compte t, donne 2003-07-16 et 6 040. Le texte gelé nomme la
première et c'est la causalement correcte — un signal formé en T−1 ne peut pas
utiliser la 252ᵉ séance si cette séance est t. Journalisé (B12 résolu).

## Étape 2 — les chiffres d'admission tiennent

| | mesuré | référence du programme |
|---|---|---|
| ER63 contre volatilité réalisée | **+0,096** Pearson, −0,044 Spearman | les états sont *ordonnés* par la volatilité |
| ruptures par an, détecteur bayésien | **11,04** | A′ sparse jump : **0,53** |
| désaccord signal rapide / lent | **35,9 %** des couples instrument-séance | — |

Axe 1 franchi (autre latente), axe 2 franchi d'un facteur **21**.

## Étape 3 — LE CHIFFRE QUI DÉCIDAIT

Critère écrit **avant** : MDE au-dessus de ~4,5 points ⇒ question indécidable,
niveau A non posable.

| lecture | MDE, points de taux de réussite |
|---|---|
| bootstrap dates, bloc 21 | 1,27 |
| bloc 63 | 1,30 |
| bloc 126 | **1,33** |
| nul par rotation de l'étiquette | 1,27 |

Nul centré à **−0,022 point** : non biaisé par la construction de la statistique.

**⇒ LA QUESTION DE PANNEAU EST DÉCIDABLE**, avec 3,4× de marge sur le critère, et
sous l'attente déclarée de 2-4 du plan. Elle atterrit à l'extrémité « regroupement
par date » de la fourchette analytique (0,8 contre 4,5-6,0), ce qui confirme que le
produit signal × rendement est proche du bruit blanc — c'est tout l'avantage
structurel d'un test de panneau sur les six dispositifs qui ont réduit la question à
un Sharpe unique.

**La statistique observée n'a pas été lue.** La lire, c'est le niveau A, et le
niveau A est un essai.

## Étape 4 — la porte C0

Prime de retournement transversale à 5 jours, brute de coûts, 46 instruments :

| période | Sharpe brut | t |
|---|---|---|
| plein échantillon | +0,130 | 0,63 |
| **2016-2026 (la porte)** | **+0,067** | **0,22** |
| 2003-2015 | +0,187 | 0,67 |

La porte demande « > 0 » et c'est satisfait — **à la lettre seulement**. Un t de 0,22
brut de coûts est indiscernable de zéro. Le niveau C reste ouvert par la règle, pas
par la force, et c'est la branche la plus faible de l'arbre.

## Une erreur, nommée

Le premier passage rendait **NaN sur les quatre lectures** et imprimait « question
non décidable » — un instrument cassé qui se lisait comme un résultat.
`(ratio > median).shift(1)` donne un dtype `object`, où `~True` vaut −2 et non False ;
les totaux de poids partaient à **−367 992**. Corrigé, et le script **refuse
désormais d'émettre un verdict** quand une lecture n'est pas finie.

## Ce qui vient ensuite

Le niveau A est posable. Le lire est un essai et il va au registre. Avant ça, la
porte G0 de Rentec — 1,5 j, zéro essai, P 0,55 de fermer sa branche — reste la
dépense la moins chère du lot et peut tourner dans l'autre session.
