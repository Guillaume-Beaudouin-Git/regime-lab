# Pré-enregistrement — un Sparse Jump Model par facteur (piste `factorsjm`)

Idée 5 de `docs/presentation/PISTES_AMELIORATION.md`. Ce document est commité **avant**
toute lecture d'un rendement conditionné à un état. Il fixe les données, le modèle, la
famille de tests, les métriques, les verdicts et la correction pour tests multiples. Tout
écart ultérieur sera déclaré dans `docs/RESULTS_FACTORSJM.md`, section « écarts au
protocole ».

Code : `regime_lab/extensions/factorsjm.py` (construction, testée sur données
synthétiques par `tests/test_factorsjm.py`) et `scripts/factorsjm_run.py` (trois phases :
`--fit`, instrument, `--read` une seule fois). Données : `scripts/factorsjm_fetch.py`.

---

## 1. La question

Le programme a testé un **interrupteur unique** : un classifieur ordonné par la
volatilité, qui change d'état environ une fois tous les deux ans, posé sur un livre de
dimension effective inférieure à 4. Il ne bat jamais une règle de volatilité d'une ligne.

Shu et Mulvey (*Dynamic Factor Allocation Leveraging Regime-Switching Signals*, arXiv
2410.14841, *Journal of Portfolio Management* 51(3)) font autre chose : **un modèle par
facteur**, estimé sur les variables du rendement actif de ce facteur, avec des états
étiquetés « haussier » et « baissier » par le **rendement cumulé** de chaque état, et six
décisions qui pilotent une allocation entre facteurs.

Deux questions, dans cet ordre :

1. **Réplication partielle.** Le dispositif du papier, appliqué aux facteurs de Ken
   French, reproduit-il le signe des Sharpe publiés pour la stratégie long-short
   mono-facteur, sur la fenêtre du papier (2007-2024) ?
2. **Le test du programme.** Un portefeuille de six facteurs, chacun exposé selon son
   propre régime, bat-il le même portefeuille tenu en permanence, les mêmes décisions
   prises par des règles d'une ligne propres à chaque facteur, et le placebo ?

## 2. Les données

| série | fichier | source | période | SHA-256 (16 premiers) |
|---|---|---|---|---|
| Mkt-RF, SMB, HML, RMW, CMA, RF, quotidien, en % | `data/raw/factorsjm/factors_5_full.parquet` | Ken French, `F-F_Research_Data_5_Factors_2x3_daily` | 1963-07-01 → 2026-07-31 | `f5fff1d98f4a69f7` |
| UMD, quotidien, en décimal | `data/raw/crisis/french_umd.parquet` | Ken French, `F-F_Momentum_Factor_daily` | 1926-11-03 → 2026-07-31 | `302303df13aee16d` |
| taux 1 an (`DGS1`) | `data/raw/factorsjm/dgs1.parquet` | FRED, indexé par date de publication (J+1) | 1962-01-02 → 2026-09-22 | `5862e06942a64ffe` |
| taux 10 ans (`DGS10`) | `data/raw/factorsjm/dgs10.parquet` | FRED, idem | 1962-01-02 → 2026-09-22 | `4290d7ba675cb581` |

- **Historique retéléchargé, déclaré.** `data/raw/panels/factors_5.parquet` commence en
  1990 ; le fichier complet commence en 1963. L'instrument imprime l'écart maximal entre
  les deux fichiers sur leur recouvrement 1990-2026 et le nombre de séances où UMD manque
  sur le calendrier des cinq facteurs ; il refuse de continuer si l'un des deux n'est pas
  nul.
- **Holdout AQR 1971-1989.** Il a été ouvert pour l'étude `longhist` le 23/09
  (`docs/PROTOCOL_FREEZE.md`, commit `d1bc935`). Cette étude le lit aussi. Rien de plus
  n'est perdu : il n'est plus scellé.
- Les facteurs long-short sont autofinancés, `Mkt-RF` est déjà en excès : **tous les
  rendements sont en excès du cash.** Coût : **zéro** (décision du 23/09). La rotation
  est rapportée.

## 3. Le modèle, par facteur

**Les six facteurs** : marché (`Mkt-RF`), taille (`SMB`), valeur (`HML`), rentabilité
(`RMW`), investissement (`CMA`), momentum (`UMD`). Le rendement « actif » d'un facteur
long-short est le facteur lui-même ; pour le marché, c'est `Mkt-RF`.

**Variables propres au facteur** (celles du papier, calculées sur le rendement actif) :
- moyenne exponentielle du rendement, demi-vies 8, 21 et 63 séances ;
- RSI sur 8, 21 et 63 séances (moyennes glissantes simples des gains et des pertes) ;
- %K stochastique sur 8, 21 et 63 séances, sur l'indice cumulé en log ;
- MACD (8, 21) et (21, 63), différence de moyennes exponentielles (demi-vies) de
  l'indice cumulé en log ;
- log de la déviation à la baisse, demi-vie 21 ;
- bêta actif sur le marché, demi-vie 21 (absent pour le marché lui-même).

**Variables de marché**, avec des substituts de long historique, déclarés :
- rendement du marché, moyenne exponentielle demi-vie 21 (comme le papier ; absent du
  modèle du marché, où il ferait doublon) ;
- **VIX → volatilité réalisée du marché** : log de `sqrt(EWMA(r², demi-vie 10))`,
  différencié, moyenne exponentielle demi-vie 21. Le VIX commence en 1990 ;
- **taux 2 ans → taux 1 an** (`DGS1`), différencié, moyenne exponentielle demi-vie 21 ;
- **pente 10 ans − 2 ans → 10 ans − 1 an**, différenciée, moyenne exponentielle demi-vie
  21. Le taux 2 ans commence en 1976.

Soit 17 variables par facteur long-short, 15 pour le marché.

**Estimation.**
- Sparse Jump Model (`jumpmodels`, l'implémentation de référence), 2 états,
  **κ² = 9,5 fixé** (la valeur d'exemple du papier), 10 initialisations, graine 0.
- Variables écrêtées à ±3 écarts-types puis standardisées, **avec les statistiques de la
  fenêtre d'entraînement seule**.
- **Fenêtre d'entraînement** : toutes les séances antérieures à la réestimation, au plus
  12 ans, au moins 8 ans (comme le papier).
- **Réestimation tous les 6 mois**, première séance de janvier et de juillet, à partir de
  janvier 1972. Entre deux réestimations, inférence en ligne à centroïdes fixes : le
  filtre avance depuis le début de la fenêtre d'entraînement, et seules les séances à
  partir de la réestimation sont gardées.
- **Étiquettes** : haussier (1) = l'état au plus fort rendement actif cumulé sur la
  fenêtre d'entraînement ; baissier (0) = l'autre. Pas d'ordre par la volatilité.
- **Délai** : un état daté de la séance *d* est utilisé sur la séance *d* + 2 (le délai
  d'un jour du papier, plus strict que la règle T−1 / T du programme).

**La grille de λ, déclarée avant tout ajustement évalué** : `(10, 20, 50, 100, 200,
500)`. Elle a été choisie pour couvrir la plage de rotation du papier (0,64 à 3,66
changements par an), après un seul contrôle qui n'a lu aucun rendement de stratégie :
le nombre de changements d'état par an sur une fenêtre d'entraînement (HML, 1978-1990)
pour quelques valeurs de λ. Ce contrôle a été fait dans un fichier temporaire ; ses
chiffres ne sont pas cités. La rotation effective, hors échantillon, est rapportée par
l'instrument.

**Le réglage, règle du papier.** Pour chaque facteur et chaque λ, le chemin hors
échantillon complet est calculé. Tous les 6 mois à partir de janvier 1978, on retient le
λ dont la stratégie long-short mono-facteur du papier (§7, facteur brut) a eu le meilleur Sharpe **sur les 6 ans
précédents de son propre chemin hors échantillon**, à coût nul. Ce λ fournit les états
jusqu'au réglage suivant. En cas d'égalité, le plus petit λ est retenu. Un λ dont le
chemin ne couvre pas toute la fenêtre de validation n'est pas éligible.

Premier état réglé : janvier 1978. **Échantillon de test : de la première séance où
toutes les jambes sont finies (début janvier 1978) au 31/07/2026, soit environ 48,5
ans.** Fenêtre du papier, pour la réplication : 03/01/2007 → 28/06/2024.

## 4. Des états aux expositions

Deux correspondances, fixées ici :
- **marche/arrêt** : exposition 1 en état haussier, 0 en état baissier ;
- **celle du papier** : exposition `clip(μ_état / 5 %, −1, 1)`, où `μ_état` est le
  rendement annualisé moyen de la jambe sur les séances de la fenêtre d'entraînement que
  le modèle a classées dans cet état. Linéaire entre −5 % et +5 %.

**Les jambes** : chaque facteur est tenu à `min(10 % / σ₆₃, 3)`, σ₆₃ étant la volatilité
réalisée sur les 63 séances qui finissent la veille (`crisis.vol_target_weight`, ciblage
de volatilité quotidien). **Le livre** : la moyenne des six jambes, chacune multipliée par
son exposition datée de *d* − 2. Un facteur exposé à 0 n'est pas remplacé et le livre
n'est pas recalé en volatilité : un gain ne peut pas venir d'un effet de levier
reporté sur les autres facteurs.

## 5. La famille de tests (8 tests, α = 0,05 / 8 = 0,00625 dans tous les MDE)

| test | objet | jambe comparée |
|---|---|---|
| **B1, principal** | livre des six jambes, marche/arrêt par le régime de chaque facteur | les six jambes toujours tenues |
| B2 | livre des six jambes, correspondance du papier | idem |
| F_mkt, F_smb, F_hml, F_rmw, F_cma, F_umd | une jambe, correspondance du papier | la même jambe toujours tenue |

**Sensibilités, journalisées, jamais décisives** : B1 et B2 à λ fixe = 50, sans réglage
(S1, S2) ; B1 et B2 avec le marché tenu en permanence, puisque le papier ne synchronise
pas le marché (S3, S4).

**Les témoins, par le même code, avec la même correspondance et le même délai :**
- la règle médiane de volatilité **propre à chaque facteur** : stress quand la
  volatilité réalisée sur 21 séances du facteur dépasse sa médiane expansive
  (`volatility_quantile_placebo`) ;
- la même au 80ᵉ centile (`crisis.volatility_tail_rule`) ;
- **la règle de tendance du facteur** : haussier quand la somme de ses 252 derniers
  rendements est positive. Le SJM est ordonné par le rendement, pas par la
  volatilité ; son témoin d'une ligne naturel est donc le momentum temporel du facteur
  (Ehsani et Linnainmaa, 2022). Ce témoin est **ajouté** à ceux du programme : il rend
  le verdict plus strict ;
- pour la correspondance du papier seulement, **le témoin statique** :
  `clip(μ / 5 %, −1, 1)` avec la moyenne de toute la fenêtre d'entraînement, sans
  état. Il isole ce que le régime ajoute à une simple repondération des facteurs par
  leur rendement passé.

Pour la correspondance du papier, les `μ` des règles d'une ligne sont estimés comme ceux
du SJM : aux mêmes dates de réestimation, sur la même fenêtre d'entraînement, par état de
la règle.

**Le VIX n'est pas un témoin ici.** Les latentes sont propres à chaque facteur, le VIX est
une variable du marché, et il commence en 1990.

**Le placebo** : la matrice des expositions tenues de la jambe testée, décalée
circulairement d'un même nombre de séances pour tous les facteurs, 400 tirages, décalage
entre 252 et n − 252 séances, graine 20260923. Il garde la distribution des expositions
et leurs dépendances entre facteurs, et détruit leur calage sur les dates.

## 6. Les métriques et le verdict, écrits avant le chiffre

- `delta` = Sharpe(régime) − Sharpe(toujours tenu), annualisés, mêmes séances.
- **MDE** : `selection.protocol.blinded_mde`, bootstrap stationnaire par blocs apparié sur
  jambes démoyennées, blocs 21, 63 et 126, **le plus grand**, à α = 0,05 / 8, puissance
  0,80 (`mde_at`).
- **t** : t HAC à 6 retards de la différence quotidienne des deux jambes, **chacune
  divisée par son propre écart-type sur l'échantillon**, pour que la différence des
  moyennes soit proportionnelle à la différence des Sharpe (sans cette mise à l'échelle,
  une jambe moins exposée aurait un t négatif à Sharpe égal).
- **pct** : part des 400 placebos dont le delta est sous le delta réel.
- **deltas des témoins** : Sharpe(témoin) − Sharpe(toujours tenu).

| verdict | condition |
|---|---|
| **UTILE** (USEFUL) | delta ≥ MDE, t de même signe, pct ≥ 95 %, et delta > le delta de **chaque** témoin |
| NON MONTRÉ (NOT SHOWN) | delta ≥ MDE mais une des trois autres conditions échoue |
| **SOUS-PUISSANT** (UNDERPOWERED) | 0 < delta < MDE — **jamais un succès** |
| PAS UTILE (NOT USEFUL) | −MDE < delta ≤ 0 |
| NUISIBLE (HARMFUL) | delta ≤ −MDE, t de même signe, pct ≤ 5 % |
| PAS UTILE, négatif au-delà du MDE, non confirmé | sinon |

**Aucun verdict si une valeur n'est pas finie.** L'instrument refuse de passer à la
lecture si une jambe, un témoin ou un MDE contient un NaN.

**Rapportés à côté, jamais décisifs** : rendement et volatilité annualisés, perte
maximale, rotation annuelle des expositions, exposition brute moyenne ; les deltas sur 5
plis de longueur égale (part de plis positifs) ; le delta sur 2007-2024, avant 1990 et
après 1990 ; le ratio d'information de (régime − toujours tenu), l'analogue du « IR
contre le portefeuille équipondéré » du papier.

## 7. La réplication (famille `factorsjm_replication`, 6 essais)

La stratégie long-short mono-facteur du papier : exposition `clip(μ_état / 5 %, −1, 1)`
avec `μ` calculé sur le **facteur brut**, multipliée par le facteur brut, **sans ciblage de
volatilité**, délai de 2 séances, coût nul, sur **03/01/2007 → 28/06/2024**.

Les chiffres publiés, avec la correspondance déclarée :

| papier (indice MSCI ou Russell long seulement, actif contre le marché) | ici (Ken French long-short) | Sharpe publié | changements/an publiés |
|---|---|---|---|
| Value | HML | 0,39 | 3,16 |
| Size | SMB | 0,20 | 2,57 |
| Momentum | UMD | 0,16 | 3,66 |
| Quality | RMW | 0,21 | 0,64 |
| Low Vol, Growth | pas d'équivalent | 0,30, 0,37 | 1,78, 1,31 |
| — | CMA, Mkt | rapportés, sans comparaison | — |

**Verdict par facteur apparié** : SIGNE REPRODUIT si notre Sharpe > 0, sinon SIGNE NON
REPRODUIT ; COMPATIBLE si |le nôtre − le publié| ≤ 1,96 × l'erreur type de Lo
`sqrt((1 + SR²/2) / années)`.
**Verdict d'ensemble** : REPRODUIT EN SIGNE si 4 sur 4 sont positifs (le papier affirme un
Sharpe positif pour tous les facteurs) ; PARTIELLEMENT REPRODUIT de 1 à 3 ; NON REPRODUIT
à 0. Rapportés à côté : la corrélation moyenne entre les six stratégies (le papier :
0,05 à 0,48), les changements d'état par an, le Sharpe du facteur toujours long, et le
Sharpe de la même stratégie sur tout l'échantillon de test.

**Ce qui n'est pas répliqué, déclaré d'avance** : les indices long seulement (MSCI,
Russell, Bloomberg), non disponibles gratuitement ; le VIX et le taux 2 ans (remplacés,
§3) ; le réglage conjoint de λ et κ (seul λ est réglé) ; la réestimation mensuelle (ici
semestrielle, pour garder le calcul raisonnable : 6 facteurs × 6 λ × ≈ 110
réestimations) ; les coûts de 5 pb (ici zéro, règle du projet) ; l'étape Black-Litterman
et ses cibles d'écart de suivi de 1 à 4 %, qui supposent des actifs long seulement. B2
en est l'analogue en espace de facteurs, et rien de plus.

## 8. Les quatre axes de la règle opposable, mesurés par l'instrument avant la lecture

- **Axe 1 (latente ordonnée par la volatilité ?)** : kappa entre l'état baissier et, pour
  chaque facteur, sa propre règle médiane, sa règle au 80ᵉ centile, la règle médiane du
  marché et sa règle de tendance ; corrélation avec le log de la volatilité réalisée du
  marché. Lecture fixée : kappa < 0,2 avec la règle médiane propre = latente distincte de
  la volatilité ; 0,2 à 0,4 = en partie ; ≥ 0,4 = ordonnée par la volatilité en pratique.
- **Axe 2 (horloge)** : changements d'état par an, hors échantillon, par facteur et au
  total. Franchi au-dessus de 2 par an.
- **Axe 3 (usage)** : une allocation entre six facteurs, pas un interrupteur. Déclaré.
- **Axe 4 (dimension)** : ratio de participation des six facteurs bruts sur 1990-2026
  (le conseiller : 4,71), des six jambes hors échantillon, et des six chemins
  d'exposition. Franchi au-dessus de 4.

Les axes servent à l'interprétation. **La famille de tests est lue quel que soit leur
résultat.**

## 9. Discipline

- Trois phases, à la main : `--fit` (écrit `data/cache/factorsjm_paths.parquet`, puis
  `factorsjm_states.parquet` et `factorsjm_tuning.parquet` ; `--select` rejoue le réglage
  seul sur les chemins en cache, sans rien lire de plus), l'instrument (sortie dans
  `docs/artifacts/factorsjm/instrument.txt`, aucun rendement conditionné imprimé), puis
  `--read` **une seule fois** (sortie dans `docs/artifacts/factorsjm/reading.txt` et
  `reading.json`). Le script refuse une seconde lecture dès que le registre contient une
  ligne de ses familles.
- **Essais déclarés : 18** — 12 dans `factorsjm` (les 8 tests et les 4 sensibilités), 6
  dans `factorsjm_replication`.
- Au plus 3 modifications de l'hypothèse. Toute modification après lecture est un nouvel
  essai, déclarée comme telle.
- Avant la lecture, la chaîne entière est exécutée sur des facteurs simulés
  (`--synthetic`), qui n'écrit rien sous `data/` ni `docs/`.

## 10. Ce qu'on dira, selon l'issue (écrit d'avance)

- **B1 UTILE** : premier dispositif du programme qui bat à la fois le portefeuille tenu,
  les règles de volatilité, la règle de tendance et le placebo. À confirmer sur une autre
  période ou un autre univers avant tout usage.
- **B1 SOUS-PUISSANT** : positif mais sous le seuil de détection, même avec 48 ans. Ce
  n'est pas un succès.
- **B1 NON MONTRÉ parce que la règle de tendance fait aussi bien** : le SJM par facteur
  est, en pratique, du momentum de facteur ; l'appareil ne sert à rien de plus qu'une
  ligne de code.
- **B1 PAS UTILE ou NUISIBLE** : le résultat du papier ne transfère pas aux facteurs de
  Ken French sur 48 ans, et la conclusion du programme (le régime porte le risque, pas le
  rendement) s'étend à l'allocation entre facteurs.
