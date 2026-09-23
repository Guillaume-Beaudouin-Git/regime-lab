# PRESPEC — l'état du Sparse Jump Model comme entrée d'une prévision de risque, sur 46 marchés

**Statut : pré-enregistrement, écrit et commité avant toute lecture d'une perte ou d'un
rendement qui fait intervenir l'état.** Ce document ne se modifie plus après son commit.
Toute déviation ira dans `docs/RESULTS_RISQUE.md`, section « écarts » (la session
principale consolidera dans `docs/PROTOCOL_FREEZE.md`).

- Piste : idée 6 de `docs/presentation/PISTES_AMELIORATION.md`, nom de piste `risque`.
- Script : `scripts/run_risque.py`. Sans `--read`, il n'imprime que l'instrument ;
  sortie intégrale dans `docs/artifacts/risque/instrument.txt`.
- Bibliothèque : `regime_lab/extensions/risque.py`, tests `tests/test_risque.py`
  (données synthétiques uniquement).
- Registre d'essais : famille `risque_forecast`, 16 lignes, **une seule lecture**.
- **Aucun coût, nulle part** (décision du 23/09). Seule la rotation est rapportée.

---

## 1. La question

Le seul actif démontré du classifieur (A′ sparse jump) est une information sur la
**variance future**. Il ajoute +3,93 points de R² sur la volatilité future au-delà d'un
quantile de volatilité (`docs/RESULTS_FINAL.md`), mais seulement +0,20 point au-delà du
VIX sur le S&P 500 (`docs/RESULTS_CRISE.md` §5). Jusqu'ici, il a toujours été jugé en
Sharpe, à travers un interrupteur ou un multiplicateur de taille.

**On le juge ici là où cet actif devrait servir : une prévision de risque.**

- Question 1, la mesure : l'état, ajouté à un modèle standard de prévision de la
  volatilité estimé sur le passé seul, **améliore-t-il la prévision de la variance à
  un mois** des 46 marchés de `data/cache/trend_universe_m1.parquet` ?
- Le VIX a déjà absorbé l'information sur le S&P 500. La question intéressante porte
  donc sur les **43 autres marchés** : actions hors US, matières premières, devises,
  obligations et crédit. Aucun d'eux n'a d'indice de volatilité implicite dans le dépôt.
- Question 2, la conséquence économique, en second temps : un livre dimensionné avec la
  prévision augmentée de l'état fait-il mieux que le même livre dimensionné avec la
  prévision standard ?

## 2. Ce qui existe déjà

- **Couche 2** (`RESULTS_FINAL.md`) : l'état porte la variance, pas la moyenne. Mesuré
  sur le seul S&P 500, par un R² incrémental en échantillon complet (rang de volatilité
  calculé sur tout l'échantillon).
- **Test P** (`RESULTS_CRISE.md` §5) : au-delà du VIX et du rang de volatilité, +0,20
  point de R² (t −1,34, seuil 2,77). Toujours sur le seul S&P 500.
- **H-b** (`scripts/run_m3_evaluation.py`, `RESULTS_TREND_VEHICLE.md`) : l'état est entré
  comme covariable de la **volatilité du livre entier** de tendance. Réfuté, c'est l'un
  des sept dispositifs.
- **Two Sigma, niveau B** (`RESULTS_TWOSIGMA_LEVEL_B.md`) : une covariance par état
  (K-means de contexte, pas le SJM) prévoit **moins bien** que la covariance unique.

**Ce qui est neuf ici.** Personne n'a mesuré l'état comme régresseur d'une **prévision de
volatilité par marché, estimée sur le passé seul**, jugée par une perte de prévision en
panneau. Personne ne l'a mesuré non plus hors du S&P 500.

## 3. La règle opposable, axe par axe, chiffrée avant la lecture

Chiffres de l'instrument (`docs/artifacts/risque/instrument.txt`, §2 et §3), qui ne lit
aucune perte liée à l'état.

| axe | ici | franchi ? |
|---|---|---|
| 1. latente ordonnée par la volatilité | même état A′ | non |
| 2. horloge sous ~2 transitions par an | 0,56 transition par an sur la fenêtre d'évaluation | non |
| 3. usage de dimensionnement ou d'interrupteur | question 1 : une **mesure** de prévision, pas un usage ; question 2 : dimensionnement | question 1 hors du champ de la règle ; question 2 non |
| 4. objet conditionné de dimension effective < 4 | panneau des volatilités de 46 marchés : ratio de participation des log-VR21 de fin de mois **4,14 en niveau**, 6,40 en variations mensuelles, première valeur propre 21,7 sur 46 | **oui, de justesse** (+0,14) |

La mesure de l'axe 4 a été refaite avant la lecture, comme le conseiller le demandait.
Elle suit exactement sa méthode :
- sur son échantillon (à partir du 17/07/2003) : 4,14 et 6,42, ses deux chiffres ;
- sur la fenêtre d'évaluation de cette étude : 4,14 et 6,40, sur 218 mois
  (01/2008 → 08/2026, les mois où les 46 marchés sont tous présents).

**Conséquences, écrites d'avance.**
- La question 1 n'est pas un « dispositif » : elle mesure un contenu prédictif, comme la
  couche 2. La règle opposable ne s'y applique pas.
- La question 2 franchit l'axe 4 de 0,14 seulement, et aucun autre axe. La règle
  écrite ici était la suivante : si la dimension en niveau tombait sous 4, E1 et E2
  seraient déclarés septièmes dispositifs avant lecture. Elle ne tombe pas sous 4, donc
  ils ne le sont pas. Mais **la marge est mince, et la prédiction du §9 les traite comme
  quasi-septièmes dispositifs.**

## 4. Données et échantillon

| fichier | rôle | SHA-256 (16 premiers) |
|---|---|---|
| `data/cache/trend_universe_m1.parquet` | prix des 46 marchés (**prix**, pas des rendements : piège n° 4) | `700f606051109195` |
| `data/raw/prices/cross_asset.parquet` | `eq_us_large` (règles de volatilité), `vol_vix` | `570bb4c9f31732e0` |
| `data/raw/macro/rate_cash_3m.parquet` | taux cash, pour l'excès des livres | `87e2d3b7c7912a48` |
| `data/cache/states.parquet` | colonne `"A' sparse jump"`, filtrée, 0 = stress | `0622c72fa75115cb` |

- **Rendements.** Rendements logarithmiques pour la prévision. Un prix manquant ou non
  positif ne donne aucun rendement : le pétrole (CL=F) a réglé à −37,63 le 20/04/2020, et
  ses séances concernées sortent du panneau, elles ne valent pas zéro. Pour les livres,
  ce sont les rendements simples de `vehicle.vol_targeted_book` (`pct_change`), pour rester
  comparable au véhicule.
- **Calendrier.** Le calendrier commun du panneau, où les jours fériés propres à chaque
  marché sont des rendements nuls (environ 4 % des séances). Fenêtres strictes : une
  fenêtre qui contient un rendement manquant est manquante.
- **L'état, les règles et le VIX** sont pris à la dernière valeur connue à la date t ou
  avant (`asof`). La prévision formée à la clôture de t vise les séances t+1 à t+21 :
  c'est « signal en T−1, position en T ».
- **Fenêtre d'évaluation, fixée par une règle et non choisie** : du lendemain de la
  dernière séance où moins de 30 des 43 marchés ont une prévision et une cible
  définies, jusqu'à la dernière séance où ils sont au moins 30. Résultat : **du
  20/12/2005 au 12/08/2026, 5 385 séances**. Au moins 30 marchés sur 43 chaque jour, 43
  en médiane.
- **Couverture par classe**, en nombre de marchés présents : actions hors US de 9 à 10,
  matières premières de 12 à 13, devises de 3 à 11 (AUDUSD n'arrive qu'en 08/2008),
  obligations et crédit de 4 à 9 (HYG, BWX et EMB arrivent en 2009-2010). PL=F n'a une
  prévision que sur 80 % des séances, à cause de trous dans ses prix avant 2010.
- La fenêtre contient les épisodes de stress 2007-2009 et 2020-2021. **L'épisode
  2002-2003 ne sert qu'à l'entraînement.**

## 5. Les prévisions

Pour chaque marché, séparément :

- **Cible** : y_t = log Σ_{j=1..21} r²_{t+j}, la variance réalisée des 21 séances
  suivantes.
- **HAR** (Corsi 2009) : y ~ 1 + log RV5 + log RV22 + log RV66, où RVw est la moyenne
  des r² sur les w dernières séances.
  - La composante journalière du HAR classique est remplacée par une composante
    trimestrielle : le log d'un rendement nul (jour férié) vaut −∞.
  - Ce choix a été fait sur ce seul argument, avant toute estimation.
- **HARVIX** : HAR + log VIX_t. Le VIX sert de régresseur commun à tous les marchés :
  c'est le témoin le plus exigeant, la jauge de peur publique que tout le monde a.
- **+ état** : une indicatrice, qui vaut 1 si A′ est en stress à t.
- **+ médiane**, **+ 80ᵉ** : les indicatrices des deux règles d'une ligne du programme.
  Elles valent 1 quand la volatilité à 21 séances du S&P 500 dépasse sa médiane
  expansive (`volatility_quantile_placebo`), ou son 80ᵉ centile expansif
  (`crisis.volatility_tail_rule`).

**Estimation, sur le passé seul.**
- Les coefficients utilisés à la clôture de t viennent d'un MCO sur les lignes
  s ≤ t − 21, dont la cible est entièrement réalisée à t.
- Les lignes d'entraînement commencent au 01/04/2002, premier état hors échantillon, et
  il en faut au moins 504.
- Réestimation à **chaque séance**, en fenêtre expansive.
- **Tous les bras d'un marché s'entraînent sur exactement les mêmes lignes.**
- Prévision de variance : exp(x′b + s²/2), correction log-normale par la variance
  résiduelle d'entraînement.
- **Abstention.** Si l'une des deux modalités d'une indicatrice compte moins de 63 lignes
  d'entraînement, le bras augmenté rend la prévision du bras de base. C'est la convention
  H-b. Aucune séance n'est retirée.

**Perte : QLIKE**, L = σ²/h − log(σ²/h) − 1 (Patton 2011). Elle est robuste au bruit de
la variance réalisée utilisée comme cible, et elle pénalise davantage la sous-prévision.

## 6. La famille de tests — fixée d'avance, rien n'y sera ajouté

**Panneaux de prévision.** Pour chaque panneau, on calcule
d_t = moyenne sur les marchés du panneau de [QLIKE(base) − QLIKE(base + état)], puis
Δ = moyenne de d_t sur les 5 385 séances. **Un Δ positif veut dire que l'état aide.**

| code | panneau | base | α |
|---|---|---|---|
| **P1** | 43 marchés sans VIX propre | HAR | 0,05/3 |
| **P2** | les mêmes 43 | HARVIX | 0,05/3 |
| **P3** | les 3 indices actions US (^GSPC, ^NDX, ^RUT), là où le VIX s'applique | HARVIX | 0,05/3 |
| C1, C2 | actions hors US (10) | HAR, HARVIX | 0,05/8 |
| C3, C4 | matières premières (13) | HAR, HARVIX | 0,05/8 |
| C5, C6 | devises (11) | HAR, HARVIX | 0,05/8 |
| C7, C8 | obligations et crédit (9) | HAR, HARVIX | 0,05/8 |

- **P1 à P3 forment la famille primaire : ce sont eux qui font le verdict de l'étude.**
- C1 à C8 donnent le résultat par classe d'actifs. Ils forment une famille secondaire,
  corrigée à part. Sous 30 marchés, ce sont des **mesures par classe, pas des
  affirmations transversales**.
- **Par marché, correction pour 46 marchés déclarée d'avance.** Pour chaque base (HAR,
  HARVIX), on fait 46 tests de Diebold-Mariano (t HAC à 21 retards), corrigés par Holm au
  niveau de famille 0,05. On ne rapporte que des décomptes : amélioration significative,
  dégradation significative, Δ positif. **Aucun verdict par marché.**

**Livres (question 2).** Famille de 2 tests, à α = 0,05/2.
- **E1** : livre de tendance sur les 46 marchés. C'est `vehicle.vol_targeted_book`, où
  σ_63 par instrument est remplacé par la volatilité prévue (signe de la tendance 12-1 ×
  min(0,10/σ̂, 3), décalé d'une séance), avec la même cible de volatilité du livre (M3).
- **E2** : parité de risque longue seule par inverse de la volatilité prévue, sur les 35
  marchés hors devises, avec la même cible M3. Les devises sont exclues parce qu'une
  position « longue » n'y a pas de sens commun (USD/JPY contre EUR/USD).
- Dans les deux cas, on compare le livre dimensionné par HAR + état au livre dimensionné
  par HAR. Excès du cash (financement signé des ETF, comme M3), **coût nul**, du
  13/03/2006 au 10/09/2026, 5 347 séances.

**Treize tests à verdict (P1-P3, C1-C8, E1-E2), plus deux lignes de décomptes par marché
et une ligne de sensibilités.**

## 7. Le critère, écrit avant le chiffre

Le critère reprend celui de `scripts/run_crisis_coupling.py`, avec une condition de plus
(les plis). Il est codé dans `risque.verdict`.

- **MDE** : pour les prévisions, l'erreur type du bootstrap stationnaire par blocs de la
  moyenne de d_t, **démoyenné** (`risque.blinded_mean_se`), blocs de 21, 63 et 126
  séances, 2 000 tirages ; on retient le plus grand, multiplié par z(1 − α/2) + z(0,80).
  Pour les livres, `selection.protocol.blinded_mde` et `mde_at`, mêmes blocs.
- **t** : t HAC de d_t à **21 retards**, parce que les cibles à 21 séances se
  chevauchent. C'est la convention du test P. Le t à 6 retards sur une séance sur 21
  (sans chevauchement) est rapporté à côté. Pour les livres, `paired_hac_t` à 6 retards.
- **Placebo** : 400 rotations circulaires de l'indicatrice d'état, d'au moins 252
  séances, graine 20260925. **Chaque rotation réestime toutes les prévisions**, puis
  reconstruit les deux livres.
- **Témoins** : le même gain obtenu par la règle médiane et par la règle du 80ᵉ centile
  ajoutées à la même base. Pour les livres, le livre dimensionné par HAR + médiane, par
  HAR + 80ᵉ et par HARVIX.
- **Plis** : 5 tranches contiguës de la fenêtre d'évaluation.

| verdict | condition |
|---|---|
| **UTILE** | Δ ≥ MDE, t de même signe, Δ ≥ 95ᵉ centile du placebo, Δ au-dessus de chaque témoin, Δ > 0 sur au moins 3 plis sur 5 |
| **NON MONTRÉ** | Δ ≥ MDE, mais une des autres conditions échoue |
| **SOUS-PUISSANT** | 0 < Δ < MDE — **jamais un succès**, même si le t est significatif |
| **PAS UTILE** | −MDE < Δ ≤ 0 |
| **NUISIBLE** | Δ ≤ −MDE, t de même signe, Δ sous le 5ᵉ centile du placebo |
| **PAS UTILE (négatif non confirmé)** | tous les autres cas |

**Aucun verdict si une valeur lue n'est pas finie.** L'instrument l'a vérifié : 0
cellule non finie sur les 11 séries de d_t, et les deux livres sont finis.

**Seuils mesurés sous le nul, avant lecture** (instrument §4 et §5) :

| test | MDE | en % de la perte de base |
|---|---|---|
| P1 | 0,00648 | 2,17 % |
| P2 | 0,00660 | 2,34 % |
| P3 | 0,01224 | 4,03 % |
| C1 / C2 (actions hors US) | 0,01668 / 0,01227 | 5,22 % / 4,05 % |
| C3 / C4 (matières premières) | 0,00490 / 0,00417 | 2,26 % / 1,95 % |
| C5 / C6 (devises) | 0,00814 / 0,00875 | 2,69 % / 3,06 % |
| C7 / C8 (obligations, crédit) | 0,01556 / 0,02336 | 3,89 % / 6,48 % |
| E1 (tendance) | 0,020 de Sharpe | — |
| E2 (parité de risque) | 0,029 de Sharpe | — |

**Un point de transparence.** Pour exprimer le MDE en pourcentage, l'instrument imprime
la QLIKE moyenne des modèles **sans l'état** : HAR 0,2986 et HARVIX 0,2818 sur les 43
marchés, HARVIX 0,3039 sur les 3 indices US, et les niveaux par classe. On y voit, avant
le critère, que **le VIX améliore déjà le HAR d'environ 5,6 % sur les 43 marchés**. Ce
n'est pas un résultat conditionné à l'état, mais c'est une information vue avant
d'écrire ce document : elle est déclarée ici.

Les MDE des livres sont petits (0,020 et 0,029) : les deux jambes ne diffèrent que par
leur dimensionnement et sont presque identiques. Un petit écart de Sharpe y est donc
détectable, et **les témoins (règles, VIX) deviennent la condition qui compte.**

## 8. Rapporté à côté, sans jamais décider

- **Niveaux et gains relatifs** : QLIKE de chaque bras, gain relatif de l'état, de la
  règle médiane et de la règle du 80ᵉ, et ce qu'apporte le VIX seul au HAR.
- **d_t selon l'état de la veille** : séances de stress contre séances calmes. Cela dit
  d'où vient le gain.
- **Autres critères** :
  - la MSE de la variance, chaque marché ramené à sa variance réalisée moyenne, avec son t
    HAC 21 ;
  - la régression de Mincer-Zarnowitz en niveaux par marché : pente et R² médians, pour
    P1 à P3.
- **Sensibilités déclarées** :
  - P1 et P2 sans CL=F ;
  - P1 et P2 avec la médiane transversale au lieu de la moyenne ;
  - P1 avec un EWMA calibré (λ 0,94, log EWMA × 21 régressé comme le HAR) à la place du
    HAR ;
  - les actions US au-delà du HAR seul, soit le résultat connu, retrouvé dans ce cadre.
- **Livres** : pour chaque bras, Sharpe, rendement, volatilité, perte maximale, rotation
  annuelle, levier brut moyen, et l'erreur de suivi du risque (écart type du log de la
  volatilité mensuelle réalisée rapportée à 10 %). Plus le véhicule M3 tel quel (σ_63),
  pour situer. **Le rendement et la volatilité sont montrés pour vérifier qu'un gain de
  Sharpe ne passe pas par le seul dénominateur (piège n° 5).**

## 9. Ce que j'attends, écrit avant la lecture

Ce sont des jugements, pas des mesures.

- **P1 (au-delà du HAR, 43 marchés).**
  - Δ positif, avec une probabilité d'environ 0,7. L'état est un facteur commun de
    stress, et un HAR par marché ne voit pas le stress des autres marchés.
  - Au-delà du MDE (2,2 %), avec une probabilité d'environ 0,3.
  - UTILE, avec une probabilité d'environ 0,15 : la règle du 80ᵉ centile, facteur commun
    elle aussi, devrait faire presque autant.
  - Le mécanisme de sortie tardive joue contre l'état. Pendant les reprises, il maintient
    des prévisions hautes (sur-prévision), que la QLIKE pénalise moins qu'une
    sous-prévision. À l'entrée, il arrive après le HAR.
- **P2 (au-delà du HARVIX).** Δ plus petit que P1, avec une probabilité d'environ 0,8.
  UTILE avec une probabilité d'environ 0,05.
- **P3 (actions US, au-delà du VIX).** Δ proche de zéro : c'est le test P de l'étude de
  crise, refait en prévision. PAS UTILE ou SOUS-PUISSANT.
- **Par classe.** Le gain le plus probable est sur les actions hors US et sur le crédit,
  le plus faible sur les devises.
- **E1, E2.** |Δ| < 0,05. Le multiplicateur de portefeuille M3 absorbe en 63 séances la
  partie commune de l'effet de l'état. UTILE avec une probabilité d'environ 0,05, ce que
  le conseiller estimait déjà.

## 10. Ce qui sera enregistré

Tout va dans la famille `risque_forecast` de `data/trials.parquet`, à une seule lecture :
- 13 lignes à verdict (P1-P3, C1-C8, E1, E2) ;
- 2 lignes de décomptes par marché (HAR, HARVIX) ;
- 1 ligne de sensibilités.

Sortie intégrale dans `docs/artifacts/risque/reading.txt`.

## 11. Limites connues d'avance

- **Deux épisodes de stress** dans la fenêtre d'évaluation (2008-2009, 2020-2021). Le
  gain d'un facteur commun se joue sur eux. Le panneau multiplie les marchés, pas les
  épisodes : les 43 marchés sont fortement corrélés en stress (première valeur propre
  21,7 sur 46).
- **Des prix Yahoo, avec des trous** : PL=F sur 80 % des séances, ETF de crédit à partir
  de 2009-2010. Quand un marché manque, il est retiré de la moyenne transversale du
  jour ; il ne compte pas pour zéro.
- **La variance réalisée est calculée sur des rendements quotidiens de clôture**, sans
  données intrajournalières. C'est un substitut bruité : la QLIKE est choisie pour cela.
- **Un seul état (A′), filtré.** La version avec recul (`states_offline`) n'est pas
  utilisée : elle voit l'avenir.
- **Le VIX est un indice actions US.** Pour les 43 autres marchés, il n'est qu'une jauge
  mondiale de peur. Des indices implicites propres (OVX, GVZ, EVZ, MOVE) existent
  ailleurs, mais ne sont pas dans le dépôt et ne sont pas testés ici.

## 12. Choix déclarés, et écarts à la consigne

- **Composantes HAR (5, 22, 66) au lieu de (1, 5, 22)** : voir §5, pour cause de rendements
  nuls des jours fériés.
- **Un HAC à 21 retards pour les prévisions**, au lieu des 6 de la règle dure. C'est plus
  strict, parce que les cibles se chevauchent. Le t à 6 retards, sans chevauchement,
  est rapporté à côté.
- **Une condition de plis ajoutée au critère de l'étude de crise** : 3 plis sur 5. C'est
  plus strict, et cela répond à la règle du walk-forward en au moins 5 plis. La
  réestimation expansive quotidienne est, elle, le walk-forward des prévisions.
- **Le VIX est ajouté en témoin pour tous les marchés** (P2, C pairs), et pas seulement
  pour les actions US. C'est plus strict que la consigne, qui le demandait « là où il
  s'applique ». P1 répond à la consigne telle qu'écrite.
- **Aucune modification d'hypothèse n'est prévue** (0 sur 3).
