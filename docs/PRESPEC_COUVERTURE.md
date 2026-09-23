# Pré-enregistrement — piste `couverture` : le régime choisit *avec quoi* se couvrir

**Écrit et commité le 23/09/2026, avant toute lecture d'un rendement ou d'une variance
conditionnés au label.** L'idée vient de `docs/presentation/PISTES_AMELIORATION.md`,
idée 3, qui en donnait un brouillon de critère ; ce document le reprend et le durcit.

- Code : `regime_lab/extensions/couverture.py` (19 tests dans `tests/test_couverture.py`,
  sur données synthétiques) et `scripts/run_couverture.py`.
- Sortie de l'instrument, commitée avec ce document : `docs/artifacts/couverture/instrument.txt`.
  Tous les chiffres ci-dessous en viennent. Aucun ne dépend d'un rendement conditionné.
- La lecture se fait **une seule fois** (`--read`). Le script refuse une seconde lecture
  dès que la famille `couverture` figure dans `data/trials.parquet`.

---

## 1. L'idée, et ce qu'on juge

La corrélation entre actions et obligations d'État change de signe au fil des décennies.
Elle est positive dans les années 1990 et de nouveau depuis 2021, négative de 2000 à
2020. Les obligations ne couvrent un portefeuille d'actions que dans le second monde :
c'est pour cela qu'elles ont protégé en 2008 et en 2020, et pas en 2022.

**Le label.** L_t = 1 (régime « positif ») quand la corrélation entre les rendements
logarithmiques du S&P 500 (^GSPC) et l'opposé de la variation du taux à 10 ans (^TNX),
calculée sur les 63 dernières séances où les deux existent, est positive, avec une
hystérésis de 21 séances : le label ne change qu'après 21 séances consécutives de l'autre
signe. L_t = 0 sinon. **Aucun paramètre n'est estimé** : 63, 21 et le seuil 0 sont ceux de
la note du conseiller, fixés avant cette étude. Le label est causal.

**Deux questions, jugées séparément.**
- **V, la variance** : le label améliore-t-il la prévision du risque d'un portefeuille
  actions + obligations, au-delà de la volatilité et du VIX ? C'est le canal où le
  programme a de la puissance.
- **S, le Sharpe** : une poche défensive choisie par le label (obligations si L = 0, or si
  L = 1) fait-elle mieux qu'une poche fixe ?

---

## 2. Ce que l'on sait avant de lire (instrument, propriétés du label seulement)

| mesure | valeur |
|---|---|
| séances où la corrélation est définie | 9 113 (04/1990 → 09/2026) |
| part des séances en régime positif (label) | 40,5 % |
| changements de label par an | 0,93 (3,46 pour le signe brut, sans hystérésis) |
| années 100 % positives | 1990 à 1996 (1997 : 94 %) |
| années à 0 % | 2001-2002, 2008-2012, 2017-2020 |
| 2021 → 2026 | 38 %, 48 %, 62 %, 70 %, 25 %, 85 % |

**Un écart avec la construction du conseiller, déclaré.** `scripts/describe_pistes.py`
aligne d'abord les séries, puis calcule la corrélation glissante. Chaque jour férié du
marché obligataire (31 sur l'échantillon) efface alors les 63 corrélations suivantes, et
il ne reste que 7 710 séances sur 9 238. Ici, les séances où l'une des deux séries manque
sont retirées **avant** la fenêtre glissante : 9 113 séances. L'horloge ne change pas
(0,93 changement par an contre 0,93 chez le conseiller).

---

## 3. La règle opposable : sur quels axes la piste diffère, chiffrés avant lecture

| axe | les sept échecs | ce label | franchi ? |
|---|---|---|---|
| 1. latente ordonnée par la volatilité | état A′ : Spearman −0,49 avec la volatilité réalisée à 21 séances du S&P 500, −0,53 avec le VIX (2002-2026) | Spearman −0,23 avec la volatilité réalisée, −0,23 avec le VIX (la corrélation continue : −0,27 et −0,23) ; accord avec les règles de volatilité κ **+0,08** (médiane), **+0,09** (80ᵉ centile), **+0,09** (VIX), et avec l'état A′ κ **+0,10** | **oui** |
| 2. horloge sous ~2 transitions par an | 0,514 par an | 0,93 par an | non |
| 3. usage : taille ou interrupteur | multiplicateur ou interrupteur | **choix entre deux couvertures** (S) ; **entrée d'une prévision de risque** (V) | **oui** |
| 4. objet de dimension effective < 4 | 3,86 | 2,82 sur 3 (actions, TLT, or, ciblés en volatilité, sans condition) | non |

La piste franchit donc les axes 1 et 3, et reste en deçà sur les axes 2 et 4. Le label
est presque indépendant des règles de volatilité (κ ≈ 0,09) : si un témoin de
volatilité fait aussi bien que lui, ce ne sera pas parce qu'il le recopie.

---

## 4. La combinaison avec le Sparse Jump Model : vide par construction

La tâche demandait aussi de combiner le label avec l'état A′ : garder les actions en
calme, et tenir en stress la jambe défensive choisie par le label (objet **W**, la
bascule de `scripts/run_safe_haven_switch.py`). L'instrument montre que **sur les 832
séances de stress d'A′ entre 2002 et 2026, le label choisit l'or une seule fois** (le
05/04/2021) :

| épisode de stress A′ | séances | dont label positif |
|---|---|---|
| 28/10/2002 → 28/05/2003 | 146 | 0 |
| 26/11/2007 → 13/12/2007 | 14 | 0 |
| 15/01/2008 → 17/09/2009 | 423 | 0 |
| 02/10/2009 → 11/11/2009 | 29 | 0 |
| 12/03/2020 → 05/08/2020 | 102 | 0 |
| 02/10/2020 → 22/03/2021 | 117 | 0 |
| 05/04/2021 | 1 | 1 |

Toutes les crises détectées par A′ sur 2002-2026 tombent dans le monde à corrélation
négative. La bascule W avec jambe choisie **est** donc la bascule vers TLT, déjà lue dans
`docs/RESULTS_REFUGE.md`, à une séance près. W contre TLT fixe n'a rien à tester. W contre
l'or fixe reviendrait à relire la comparaison TLT contre or de ce document.

**Décision, prise avant lecture : W n'est pas un test.** Il est calculé et journalisé
comme **référence**, sans verdict, pour que le registre garde la trace de la lecture.
C'est en soi la réponse à la question « combiner avec A′ » : sur cet échantillon, les
deux latentes ne se recouvrent pas. Une version antérieure du script, jamais lue, avait
7 tests dont W_TLT et W_GOLD ; la famille passe à 5, ce qui relâche α de 0,05/7 à 0,05/5.
Ce changement ne dépend que de propriétés du label.

---

## 5. Les données

| jambe | source | période |
|---|---|---|
| S&P 500 (^GSPC), taux 10 ans (^TNX), VIX | `data/raw/prices/cross_asset.parquet` (Yahoo) | 1990 → 09/2026 |
| obligation 10 ans reconstituée depuis le taux : `r = y/252 − 7,5 · Δy` (`strategies.base.bond_return`, proxy déclaré : sans convexité ni roulement) | idem | idem |
| actions (facteur marché de Ken French, dividendes compris, déjà en excès) | `data/raw/panels/factors_5.parquet` | → 31/07/2026 |
| TLT (obligations d'État US > 20 ans), or (future GC=F non ajusté) | `data/cache/trend_universe_m1.parquet` (en **prix**) | 07/2002 et 08/2000 → 09/2026 |
| taux sans risque 3 mois, estampillé à sa date de disponibilité | `data/raw/macro/rate_cash_3m.parquet` | 1990 → |
| état A′ filtré | `data/cache/states.parquet`, **lu, jamais réécrit** | 2002-04 → 09/2026 hors échantillon |

Tout est en excès du cash. **Aucun coût** (décision du 23/09) : la rotation annuelle est
rapportée. **Pas de reconstitution avant 1990** : les années 1970-1980 de Campbell,
Pflueger et Viceira restent hors champ. La question V couvre déjà 1991-1997, sept années
entièrement positives. La question S ne peut pas commencer avant TLT (octobre 2002 une
fois sa volatilité estimée), et l'or n'est pas sur disque avant 2000.

---

## 6. La famille de tests : 5 tests, Bonferroni α = 0,05/5 = 0,01

### Question V — prévision de la variance (2 tests, 29/01/1991 → 07/08/2026, 8 945 séances)

**Portefeuilles.**
- **V1** : le 60/40 figé du programme (S&P 500 et obligation 10 ans, rééquilibré chaque
  mois, en excès du cash).
- **V2** : actions et obligation 10 ans à risque égal, chacune pondérée par
  min(0,10 / σ₆₃, 3), avec σ₆₃ connu en T−1, puis la moyenne des deux.

**Cible.** y_t = log Σ_{j=t+1..t+21} r_j², la variance réalisée des 21 séances suivantes.

**Contrôles (« la volatilité et le VIX »)** :
- log K_t, la variance implicite à un mois tirée du VIX ;
- le rang expansif, donc causal, de la volatilité à 21 séances du S&P 500 : la règle du
  programme ;
- log RV21_t, la variance réalisée du portefeuille lui-même sur 21 séances.

**Régression.** On ajoute L_t. Le t est HAC à 21 retards, la convention du programme
pour une cible à 21 séances qui se chevauche.

**Seuil de détection, mesuré en aveugle avant lecture** (`couverture.regression_mde` :
le résultat ne dépend pas de l'effet, test à l'appui). Coefficient détectable à 80 % de
puissance :

| portefeuille | coefficient détectable | R² incrémental détectable |
|---|---|---|
| V1 | 0,170 point de log | 0,72 point |
| V2 | 0,176 point de log | 1,40 point |

À comparer à l'état A′ : +3,93 points au-delà d'un rang de volatilité, +0,20 au-delà du
VIX.

**Verdict, par portefeuille :**
- **PREDICTS** : coefficient > 0, t ≥ 2,58 (z à 1 − 0,01/2), percentile du placebo par
  rotation de L ≥ 95 % (400 rotations d'au moins 252 séances), **et** gain de R² hors
  échantillon > 0. Le hors échantillon se fait avec des réestimations expansives toutes
  les 252 lignes après 2 520 (premières prévisions en janvier 2001). Chaque modèle
  n'apprend que sur les cibles déjà réalisées.
- **NOT SHOWN** : coefficient > 0 et t ≥ 2,58, mais le placebo ou le hors échantillon
  échoue.
- **UNDERPOWERED** : coefficient > 0, t < 2,58. Ce n'est jamais un succès.
- **DOES NOT PREDICT** : coefficient ≤ 0.
- Aucune valeur non finie ne donne de verdict.

**Qualificatif, qui décide de la formulation et non du verdict.** Le label ajoute-t-il
encore quelque chose quand la variance du portefeuille **sur 63 séances**, sa propre
fenêtre, entre dans les contrôles ? Mêmes seuils : t ≥ 2,58 et gain hors échantillon
> 0. Si la réponse est non, on écrira que le label n'apporte que la mémoire de la
covariance récente, et non une information de régime.

**Rapportés, sans décider :**
- l'échantillon mensuel sans chevauchement (une séance sur 21, HAC 6) ;
- la corrélation continue à la place du label ;
- sur 2002-2026, l'apport de l'état A′ au-delà des mêmes contrôles, et celui du label
  au-delà des contrôles et de l'état A′.

### Question S — le Sharpe d'une poche défensive (3 tests, 28/10/2002 → 31/07/2026, 5 977 séances)

**Jambes**, exactement celles de `run_safe_haven_switch.py` :
- EQ, le facteur marché de Ken French ;
- TLT ;
- l'or (GC=F).

Chacune est en excès du cash, pondérée par min(0,10 / σ₆₃, 3), avec σ₆₃ connu en T−1.

**Objet P** : 0,5 × EQ + 0,5 × poche. La poche est TLT si le label **connu en T−1** vaut
0, l'or s'il vaut 1. Le label choisit l'or sur 24,3 % des séances et change 27 fois dans
la fenêtre.

**Les trois tests** comparent P-label à la même construction avec une poche fixe :

| test | comparateur | MDE (le plus grand des blocs 21 / 63 / 126, α 0,01) |
|---|---|---|
| P_TLT | poche 100 % TLT | 0,356 |
| P_GOLD | poche 100 % or | 0,577 |
| P_MIX | poche fixe 50 % TLT / 50 % or, le témoin du brouillon | 0,361 |

**Verdict par test** : la fonction `verdict` de `run_crisis_coupling.py`, rendue plus
stricte.
- δ = Sharpe(P-label) − Sharpe(comparateur), à coût nul.
- t HAC à 6 retards de la différence quotidienne.
- Placebo : le label tourne par rotation circulaire (400 rotations d'au moins 252
  séances).
- Trois témoins : la même poche, choisie cette fois par une règle de volatilité (TLT
  quand la volatilité est haute, l'or sinon, puisque la corrélation est plus souvent
  négative quand la volatilité monte). Les trois règles sont la médiane expansive de la
  volatilité à 21 séances du S&P 500, son 80ᵉ centile, et le VIX au-dessus de sa médiane
  expansive.

**USEFUL** exige : δ ≥ MDE, t de même signe, percentile du placebo ≥ 95 %, et δ
supérieur au δ de **chacun** des trois témoins. Si les deux premiers témoins sont battus
mais pas le VIX, le verdict est NOT SHOWN. Un δ positif sous le MDE est UNDERPOWERED,
jamais un succès.

**Verdict de l'idée sur P** : « le label choisit la bonne couverture » n'est établi que
si **les trois** tests sont USEFUL. C'est une affirmation d'intersection : battre la
poche fixe qui a perdu ne suffit pas, choisir après coup la poche fixe à battre serait
de la sélection sur le résultat.

**Rapportés, sans décider :**
- le Sharpe, le rendement, la volatilité, la perte maximale, le pire mois et la rotation
  de chaque bras ;
- les fenêtres de crise de l'étude de crise (2008, rebond de 2009, février 2018, Covid,
  rebond de 2020, 2022) ;
- le Sharpe de chaque jambe selon le label ;
- un bras de référence « TLT si L = 0, liquidités si L = 1 » (l'« or ou cash » de la
  tâche) : c'est une référence, jamais promue en test après lecture ;
- la statistique du brouillon (§7).

---

## 7. Ce que le brouillon disait, et ce qui change

| brouillon du conseiller | ici | pourquoi |
|---|---|---|
| statistique primaire : rapport des variances du portefeuille sur les 5 % pires mois des actions, contre la poche 50/50 ; PASS sous le 5ᵉ centile du placebo | **rapportée, sans décider**, avec son percentile de placebo | 5 % des mois, c'est 14 mois sur 24 ans : aucun MDE n'est calculable. Et une faible variance dans les mauvais mois récompense une poche qui bouge peu, pas une poche qui paie : des liquidités y gagneraient |
| MDE « mesuré sous le nul avant lecture » | MDE en aveugle pour les 5 tests, commité ici | — |
| témoins : poche 50/50, bascule pilotée par le 80ᵉ centile | poche 50/50, **plus** TLT fixe, or fixe (affirmation d'intersection), **plus** trois témoins de volatilité (médiane, 80ᵉ, VIX) | un gain face à une seule poche fixe peut venir de la faiblesse de celle-ci |
| la variance « de baisse » comme canal puissant | la **prévision** de la variance d'un portefeuille actions + obligations, au-delà de la volatilité et du VIX, en échantillon **et** hors échantillon | c'est là que le programme a mesuré sa seule information (+3,93 points), et c'est un test à 8 945 séances plutôt qu'à 14 mois |
| α non précisé | Bonferroni 0,05/5 sur toute la famille | — |

---

## 8. Garde-fous

- **Une seule lecture.** Chaque test est un essai `couverture` dans
  `data/trials.parquet` : 5 tests et 1 référence (W), soit 6 lignes.
- **Modifications de l'hypothèse après lecture : 0 sur 3.** Toute modification sera
  déclarée dans `docs/RESULTS_COUVERTURE.md`, avec son numéro.
- **Aucun paramètre estimé sur l'échantillon de test** : le label n'en a pas, les poids
  viennent du passé seul (σ₆₃ en T−1), et le hors échantillon de V n'apprend que sur des
  cibles réalisées.
- **Pas de verdict sur une valeur non finie** : le script s'arrête avant.

**Attente déclarée avant lecture**, sans valeur de critère :
- V2 a de bonnes chances d'être PREDICTS au-delà de la volatilité à 21 séances, parce
  que la variance d'un portefeuille à risque égal dépend mécaniquement de la
  corrélation. Le qualificatif à 63 séances est plus incertain.
- V1 est moins probable : le risque d'un 60/40 vient surtout des actions.
- Sur P, les MDE (0,36 à 0,58) dépassent de loin les écarts plausibles, et un seul
  épisode décisif (2022) est dans l'échantillon : UNDERPOWERED ou NOT USEFUL sont les
  issues attendues.

---

## 9. Reproduire

```bash
.venv/bin/python scripts/run_couverture.py          # l'instrument, ~70 s, ne lit rien de conditionné
.venv/bin/python scripts/run_couverture.py --read   # la lecture, une fois ; refuse la seconde
```
