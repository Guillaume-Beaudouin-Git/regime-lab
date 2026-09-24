# PRESPEC — un Sparse Jump Model réduit sur cent ans (1926-2026), et le momentum

**Statut : pré-enregistrement, écrit et commité le 24/09/2026, avant l'ajustement du
modèle, avant toute mesure de validation et avant toute lecture d'un rendement
conditionné à un état.** Ce document ne se modifie plus après son commit. Toute
déviation va dans `docs/RESULTS_LONGHIST.md` (section « écarts au protocole ») ; la
session principale consolidera dans `docs/PROTOCOL_FREEZE.md`.

- Idée 1 de `docs/presentation/PISTES_AMELIORATION.md`, avec l'idée 2 comme bras
  déclaré. Ce document reprend et durcit le brouillon de critère du conseiller.
- Données : `scripts/longhist_fetch.py` (commit `ba62a0c`).
- Bibliothèque : `regime_lab/extensions/longhist.py`, tests `tests/test_longhist.py`
  (commit `9762836`).
- Ajustement : `scripts/longhist_fit.py`. Validation : `scripts/longhist_validate.py`.
  Test : `scripts/longhist_umd.py` (sans `--read`, il n'imprime que l'instrument).
- Registre d'essais : familles `longhist_calibration` (le choix de λ, comme la phase 2
  d'A′) et `longhist_umd` (la lecture, 4 lignes, une seule fois).
- **La réserve scellée AQR (1971-04 → 1989-12) est ouverte par cette étude**, sur
  autorisation de Guillaume du 23/09, et consignée avant toute lecture dans
  `docs/PROTOCOL_FREEZE.md` (commit `d1bc935`). Le plan AQR perd sa réserve.

---

## 1. La question

Tous les verdicts du programme butent sur la puissance : la fenêtre hors échantillon
d'A′ (2002-2026) ne contient que **2 récessions NBER** et **3 épisodes de stress**.
« 93 % des récessions » repose sur deux récessions ; le seul usage qui a montré quelque
chose, l'arrêt du momentum en stress (UMD 0,52 → 0,69, au 100ᵉ centile du placebo,
`docs/RESULTS_CRISE.md`), reste sous son MDE (0,395), et la règle de volatilité au
80ᵉ centile fait autant.

On réestime **la même méthode** (Sparse Jump Model, deux états ordonnés par la
volatilité d'entraînement, walk-forward expansif, réestimation semestrielle, états
filtrés) sur les seules variables disponibles depuis 1926, et on pose trois questions :

1. **Validation, sans aucun rendement de stratégie.** Sur 1937-2026, soit **14
   récessions NBER hors échantillon** (145 mois), le classifieur reconnaît-il les
   récessions, avec quelle latence, à quelle cadence, et mieux qu'une règle de
   volatilité ?
2. **Le test pré-enregistré.** L'arrêt du momentum UMD en stress améliore-t-il son
   Sharpe au-delà du MDE, du placebo, des plis, et des règles d'une ligne ?
3. **Le bras de l'idée 2.** Une sortie asymétrique, qui quitte le stress dès que la
   volatilité se normalise, fait-elle mieux ?

C'est **un cousin d'A′, pas A′** : pas de VIX avant 1990, pas de macro en premières
publications, pas de NFCI, ni pétrole, ni dollar. On teste la méthode, pas le
classifieur figé.

## 2. Les données

Toutes publiques, gratuites et sans clé. Elles sont dans `data/raw/longhist/`, qui est
gitignoré.

| fichier | contenu | source | période | SHA-256 (16 premiers) |
|---|---|---|---|---|
| `ff3_daily` | Mkt-RF, SMB, HML, RF quotidiens | Ken French | 1926-07-01 → 2026-07-31, 26 296 séances | `87f4d694a996e411` |
| `industry49_daily` | 49 secteurs, pondérés par la capitalisation | Ken French | idem ; 39 à 43 secteurs peuplés avant 1964, 49 dès 1970 | `709086bfbac92b73` |
| `size_bm25_daily` | 25 portefeuilles taille × valeur, pondérés | Ken French | idem | `844f7f9478496c8d` |
| `fred_monthly` | AAA, BAA (Moody's), INDPRO (millésime courant) | FRED | 1919-01 → 2026-08 | `35e574f574736cc3` |
| `usrec` | indicateur de récession NBER, **validation seulement** | FRED | 1854-12 → 2026-08 | `cd0f4aca1f918e7b` |
| `crisis/french_umd` | facteur momentum UMD quotidien | Ken French | 1926-11-03 → 2026-07-31 | `302303df13aee16d` |
| `cache/states.parquet` | A′ publié, pour la comparaison 2002-2026 seulement | le programme | 2002-04 → 2026-09 | `0622c72fa75115cb` |

**Horodatage.** Une clôture est connue à la clôture. La moyenne mensuelle AAA ou BAA du
mois m entre le 8 du mois m+1. INDPRO du mois m entre le dernier jour du mois m+1. Le
taux court du mois m (le rendement du bon à un mois de Ken French) est connu dès la
première séance du mois m, puisque le bon s'achète en début de mois.

**Ce qui n'existe pas, et ce que l'on en fait.**
- **Pas de VIX avant 1990**, ni de NFCI avant 1971, ni de premières publications macro
  dans le dépôt avant 1990. Le modèle n'en a pas.
- **INDPRO n'existe qu'en millésime révisé.** Il est aussi l'une des séries que le
  comité du NBER regarde pour dater les récessions. L'utiliser rendrait la validation
  NBER en partie circulaire, et il introduirait l'information des révisions. **Il est
  donc exclu du modèle de tête**, et n'entre que dans une sensibilité déclarée (§3.4),
  qui ne décide rien.
- **Pas d'or coté avant 1971.** Le prix était fixé à 35 $ l'once jusqu'en 1968-1971, et
  la détention privée était interdite aux États-Unis de 1933 à 1974. Aucune série
  quotidienne gratuite ne couvre 1971-2000 dans le dépôt : GC=F commence en 2000 et
  les fixings LBMA ont été retirés de FRED. **Il n'y a donc pas de bascule vers l'or
  dans la famille de tests.** L'étude refuge avait d'ailleurs montré que l'essentiel du
  gain venait de l'arrêt (0,70 sur les 0,77).
- **Le NYSE cotait le samedi jusqu'en 1952** : 1 158 séances du samedi. Les fenêtres
  en séances (21, 63, 252) couvrent donc un peu moins de temps calendaire avant 1953.
  Les fenêtres de 24 mois sont en mois calendaires (§4).

## 3. Le modèle

### 3.1 Les variables : 30, soit les 28 variables d'A′ calculables depuis 1926, plus 2 substituts

Mêmes définitions qu'A′ (`features/market.py`, `features/asymmetry.py`,
`features/crosssection.py`), calculées sur l'indice de rendement total du marché CRSP
pondéré (Ken French) au lieu du S&P 500 prix.

| bloc | variables |
|---|---|
| marché, momentum (4) | `mom_eq_21`, `mom_eq_63`, `mom_eq_252`, `mom_eq_accel` |
| marché, volatilité (7) | `vol_rv_5`, `vol_rv_21`, `vol_rv_63`, `vol_term`, `vol_of_vol`, `vol_semi_ratio`, `vol_jump_share` |
| marché, forme (6) | `asy_skew_63`, `asy_kurt_63`, `asy_drawdown_252`, `asy_hurst`, `asy_vr_5`, `asy_vr_20` |
| transversal (8) | `xs_dispersion_ind`, `xs_dispersion_szbm`, `xs_avg_corr`, `xs_absorption`, `xs_absorption_chg`, `xs_breadth_63`, `xs_ff_smb_63`, `xs_ff_hml_63` |
| crédit et taux (5) | `cre_quality` = BAA − AAA (DEF de Fama et French 1989), `cre_quality_chg63`, `rat_slope_term` = AAA − taux court (TERM de Fama et French 1989), `rat_slope_term_chg63`, `rat_cash_chg12m` |

- Les statistiques transversales (corrélation moyenne, ratio d'absorption, largeur)
  sont calculées sur les secteurs **complets dans chaque fenêtre**. Les versions du
  dépôt suppriment toute ligne où manque un secteur, et ne renverraient rien avant
  1964. Un test vérifie qu'elles coïncident sur un panneau complet.
- 28 variables sont celles d'A′, à la définition près. Les 2 variables TERM sont des
  substituts : elles remplacent les 4 pentes de la courbe des taux d'A′.
- Les 22 variables d'A′ absentes ici :
  - VIX et prime de variance (2) ;
  - pétrole et dollar (2) ;
  - largeur sur 4 actifs (1) ;
  - les 9 variables macro en premières publications : INDPRO, emploi, CPI et chômage,
    chacun sur 2 horizons, plus les inscriptions au chômage ;
  - BAA − taux 10 ans et sa variation (2) ;
  - NFCI et sa variation (2) ;
  - les pentes 10 ans-2 ans et 10 ans-3 mois et leurs variations (4).
- Standardisation : z-score expansif (au moins 252 séances), puis écrêtage à ±5, comme
  `features/build.py`. Les variables sont complètes à partir du **24/05/1928**.

### 3.2 La méthode, identique à A′ sauf trois points déclarés

- `models.jump.JumpRegimes`, 2 états, `max_features = 10`, états **ordonnés par la
  volatilité d'entraînement** (état 0 = stress), prédiction **en ligne** (filtrée) entre
  deux réestimations, contexte de 252 séances d'entraînement avant chaque bloc
  (`models.base.run_expanding`, inchangé).
- Fenêtre expansive, **réestimation tous les 6 mois**, première réestimation le
  **01/01/1937**.
- λ est choisi **sur la fenêtre d'entraînement seule**, dans la grille figée
  `LAMBDA_GRID` = (1, 3, 10, 30, 100, 300), toutes les 4 réestimations (tous les deux
  ans), par `models.calibrate.choose_jump_penalty`, inchangé : Sharpe du marché tenu en
  état calme sur l'entraînement, entre 0,5 et 12 transitions par an. Les candidats sont
  journalisés (famille `longhist_calibration`), comme la phase 2 d'A′ l'a fait.

**Les trois différences avec A′, déclarées :**
1. **Les variables** (§3.1).
2. **La série de rendement** qui ordonne les états et note la calibration : le marché
   CRSP en excès du bon (Ken French). A′ utilisait un livre 60/40, qui n'existe pas
   avant 1990. **Ce n'est jamais UMD** : l'objet testé n'entre pas dans l'estimation.
3. **Le rodage.** La première réestimation dispose de 8,6 ans de variables complètes
   (1928-05 → 1936-12), et non de 10 ans. Ce choix place la récession de 1937-1938 hors
   échantillon, soit 14 récessions au lieu de 13. La Grande Dépression est dans
   l'entraînement.

Enfin, la programmation dynamique du modèle passe par `extensions.longhist.fast_dp`,
**identique au bit près** à celle de la bibliothèque de référence (tests), et bien plus
rapide : la version de référence fait une réduction numpy par séance, ce qui rend cent
ans de walk-forward impraticables.

### 3.3 Sorties

`data/cache/longhist_states.parquet` (états en ligne et avec recul),
`longhist_features.parquet`, `longhist_refits.parquet` (λ et poids des variables à
chaque réestimation). **`states.parquet` n'est ni réécrit ni recalculé.**

### 3.4 Une sensibilité déclarée, qui ne décide rien

Le même modèle avec INDPRO (croissance sur 12 mois et sur 3 mois, millésime courant),
dans des fichiers `*_indpro`. Seules les mesures de validation de la phase A lui sont
appliquées, avec la réserve écrite au §2 (révisions, circularité). Il n'entre dans aucun
test de rendement.

## 4. Les étiquettes et les témoins

Convention du programme : 1 = calme, 0 = stress. Tout est causal. Une étiquette formée à
la clôture de d est tradée en d+1 (`extensions.crisis.lagged_state`).

| étiquette | définition |
|---|---|
| **SJM long** | l'état en ligne du §3 |
| **sortie asymétrique (idée 2), k = 10** | le stress du SJM long, **levé** sur toute séance où la volatilité réalisée à 21 séances du marché est sous celle à 63 séances depuis **10 séances consécutives** ; le stress revient dès que la condition cesse, tant que le SJM est en stress. Entrée : celle du SJM. Sortie : la première de la sortie du SJM et de la levée |
| règle médiane | volatilité à 21 séances du marché au-dessus de sa médiane expansive (au moins 252 séances) = stress (`evaluation.predictive.volatility_quantile_placebo`) |
| règle du 80ᵉ centile | idem au-dessus du 80ᵉ centile expansif (`extensions.crisis.volatility_tail_rule`) |
| **panique de Daniel et Moskowitz** | stress si le marché est sous son niveau d'il y a **24 mois calendaires** **et** si sa variance sur 126 séances (leur fenêtre) dépasse sa médiane expansive (au moins 252 séances) |
| marché baissier seul | marché sous son niveau d'il y a 24 mois calendaires (mesure de l'axe 1 seulement) |

Le marché est ici le rendement total CRSP pondéré, rendements simples.

**La règle de sortie est choisie avant toute lecture, et aucun paramètre n'y est
estimé.** k = 10, soit deux semaines de bourse, est fixé par un argument externe, comme
l'écrivait le brouillon du conseiller. Les fenêtres 21 et 63 sont celles des variables
du programme. Aucune recherche sur k : **k = 5 est une sensibilité déclarée**, journalisée
et sans pouvoir de décision. **Contamination déclarée** : le conseiller a vu, sur
2002-2026, la courbe des futures VIX et les dates des creux avant d'écrire l'idée 2, et
il a mesuré la cadence du label « A′ sauf si RV21 < RV63 depuis k séances » (1,27
transition par an pour k = 10). Aucun rendement conditionné à ce label n'a été lu.
Avant 2002, rien n'a été vu.

## 5. Phase A — la validation, sans aucun essai de rendement

**Fenêtre** : toutes les séances hors échantillon, du 04/01/1937 au 31/07/2026.
**Référence** : `USREC` de FRED, rapporté à chaque séance par son mois calendaire, sans
délai de publication, puisque c'est une validation. `USREC` marque les mois qui suivent
le pic jusqu'au creux, soit 14 récessions et 145 mois sur la fenêtre. La première
commence en juin 1937, la dernière en mars 2020.

### 5.1 Mesures, pour chaque étiquette du §4

- exactitude équilibrée et κ de Cohen du stress contre la récession
  (`evaluation.reliability.external_validation`), rappel et précision ;
- **détection par récession** : une récession est détectée si au moins **21 séances de
  stress** tombent dans ses mois NBER ;
- **latence d'entrée** : première séance de stress dans [premier mois de récession −
  183 jours, fin du dernier mois] moins le premier jour du premier mois de récession, en
  jours calendaires (négatif = en avance) ; médiane sur les récessions détectées ;
- transitions par an, part de stress ;
- κ avec la règle du 80ᵉ centile, avec la règle médiane, avec la panique de Daniel et
  Moskowitz, avec le marché baissier seul.

### 5.2 Critères, écrits avant la mesure

- **PASS-A** si les trois conditions tiennent sur la fenêtre entière :
  1. exactitude équilibrée ≥ 85 % ;
  2. κ ≥ 0,40 ;
  3. au moins **12 des 14** récessions détectées.
  Sinon **FAIL-A**, avec la condition qui manque. Une mesure non finie : aucun verdict.
- **A2, « mieux qu'une règle de volatilité »** : Δκ = κ_NBER(SJM long) − κ_NBER(règle
  du 80ᵉ centile), avec un intervalle à 95 % par bootstrap stationnaire par blocs,
  apparié (`analysis.bootstrap.stationary_indices`, blocs moyens de 252 séances,
  2 000 tirages, graine 0 ; `extensions.longhist.kappa_difference_ci`). **MIEUX** si la borne basse est > 0,
  **MOINS BIEN** si la borne haute est < 0, **INDISCERNABLE** sinon. Même mesure,
  rapportée sans décider, contre la règle médiane et contre la panique de Daniel et
  Moskowitz.
- Rapporté sans décider : les mêmes mesures sur cinq plis consécutifs d'environ 18 ans,
  et la sortie asymétrique (k = 10 et k = 5) : ses transitions par an et son κ avec la
  règle du 80ᵉ centile, comme le demande l'idée 2.

### 5.3 La comparaison avec A′ (50 variables) porte sur la période commune 2002-2026

**Déclaré ici : toute comparaison entre le SJM long et le modèle actuel à 50 variables
se fait sur la seule période commune**, du 01/04/2002 (premier état hors échantillon
d'A′) au 31/07/2026 (dernière donnée Ken French). On y rapporte, pour les deux :
l'exactitude équilibrée, le κ contre NBER, le κ entre eux, les transitions par an et la
part de stress. Deux récessions seulement : c'est descriptif, cela ne décide rien.

### 5.4 La porte de l'axe 1, avant toute lecture de la phase B

La règle opposable (`CLAUDE.md`) : un objet qui a une latente ordonnée par la
volatilité, une horloge sous ~2 transitions par an, un usage d'interrupteur et une
dimension inférieure à 4 est un **septième dispositif**. Le test UMD n'y échappe que par
l'axe 1, si l'état recouvre le marché baissier mieux que la règle de volatilité (κ 0,50
contre 0,31 sur 2002-2026, `PRESPEC_CRISE.md` §4.2), et par la puissance.

- On mesure **Δκ_baissier = κ(SJM long, baissier) − κ(règle du 80ᵉ centile,
  baissier)** sur 1937-2026. **S'il est inférieur à 0,10, la phase B est déclarée
  septième dispositif avant sa lecture.** Elle est lue quand même, mais la prédiction
  écrite devient alors : B3 ne sera pas « utile ».
- On rapporte aussi l'axe 2 : la cadence du SJM long. Au-dessus de 2 transitions par an,
  l'objet diffère aussi sur cet axe.
- **La phase B est lue quel que soit le verdict de la phase A.** Les deux questions sont
  distinctes : un classifieur imparfait contre NBER peut encore bien couper le momentum.

## 6. Phase B — le test UMD, une famille de 3 essais

### 6.1 L'objet et l'échantillon

- **UMD** de Ken French, quotidien, long-short. C'est déjà un rendement en excès.
- **Ciblé à 10 % de volatilité**, comme partout dans le programme : poids
  min(0,10 / σ₆₃, 3), où σ₆₃ est la volatilité réalisée des 63 séances qui finissent la
  veille (`extensions.crisis.vol_target_weight`).
- **Échantillon** : de la première séance où l'état décalé, les trois témoins et le
  poids existent tous (au plus tôt le 05/01/1937) au 31/07/2026, soit environ 22 000
  séances.
- **Aucun coût** (décision du 23/09 pour tout le projet). La rotation annuelle du poids
  est rapportée. Le brouillon du conseiller prévoyait 5 bp : c'est un écart déclaré.

### 6.2 Les bras

- **seul** : w·u ;
- **arrêt** : w·u sur les séances dont l'étiquette de la veille est calme, 0 sinon,
  piloté par chacune des étiquettes du §4.

### 6.3 Les trois tests (α = 0,05 / 3)

| test | Δ | témoins requis |
|---|---|---|
| **B1** | Sharpe(arrêt SJM long) − Sharpe(seul) | Δ au-dessus du même arrêt piloté par la règle médiane, par la règle du 80ᵉ centile et par la panique de Daniel et Moskowitz |
| **B2** | Sharpe(arrêt sortie asymétrique k = 10) − Sharpe(seul) | idem |
| **B3** | Sharpe(arrêt SJM long) − Sharpe(arrêt règle du 80ᵉ centile), apparié | aucun (c'est la comparaison au témoin) |

### 6.4 Les mesures

- **MDE** : `selection.protocol.blinded_mde` sur les deux jambes du test, démoyennées,
  bootstrap stationnaire apparié par blocs moyens de 21, 63 et 126 séances, 2 000
  tirages, graine 0, lu à α = 0,05/3 et puissance 0,80 par `mde_at` ; on retient **le plus
  grand** des trois. Calculé dans l'instrument, avant la lecture.
- **t HAC** à 6 retards de la différence quotidienne des deux jambes
  (`selection.protocol.paired_hac_t`) ; valeur critique z(1 − α/2) = 2,394.
- **Placebo** : l'étiquette décalée tournée circulairement contre les dates, 400
  rotations d'au moins 252 séances, graine 20260924 ; on rapporte le centile de Δ dans
  la distribution des Δ placebo. Pour B3, c'est l'état du SJM long qui tourne, la règle
  du 80ᵉ centile restant fixe.
- **Plis** : 5 blocs consécutifs d'effectif égal (environ 18 ans chacun) ; Δ par pli.

### 6.5 Le verdict (`extensions.longhist.verdict`), plus strict que celui de l'étude de crise

- **USEFUL** si toutes ces conditions tiennent :
  - Δ ≥ MDE ;
  - un t HAC de même signe **et** |t| ≥ 2,394 ;
  - Δ au-dessus du 95ᵉ centile du placebo ;
  - au moins **3 plis sur 5** avec Δ > 0 ;
  - Δ strictement au-dessus du Δ de chaque témoin du tableau §6.3.
- **NOT SHOWN** si Δ ≥ MDE mais qu'une autre condition échoue.
- **UNDERPOWERED** si 0 < Δ < MDE. **Jamais un succès.**
- **NOT USEFUL** si −MDE < Δ ≤ 0.
- **HARMFUL** si Δ ≤ −MDE avec un t de même signe au-delà de 2,394 et Δ sous le 5ᵉ
  centile du placebo ; **NOT USEFUL** (négatif au-delà du MDE, non confirmé) sinon.
- **Une valeur non finie, n'importe où : aucun verdict.**

Pour B3, USEFUL se lit « le SJM long bat la règle du 80ᵉ centile ».

### 6.6 Rapporté à côté, sans décider

Toutes ces valeurs sont journalisées dans les lignes des essais.
- Pour chaque bras : rendement annuel, volatilité, Sharpe, perte maximale, rotation
  annuelle du poids. **Pour montrer qu'un gain n'est pas un gain par le dénominateur**
  (piège n° 5) : le rendement annuel de chaque bras, et le t HAC de la différence
  quotidienne, qui teste la moyenne.
- Δ par sous-période : 1937-1962, 1963-2001, 2002-2026.
- Sur 2002-04 → 2026-07, l'arrêt piloté par A′ à côté de l'arrêt piloté par le SJM
  long. Cette période a déjà été lue pour A′ dans l'étude de crise.
- UMD brut (poids 1, sans ciblage), seul et arrêté par le SJM long, à titre descriptif.
- **Sensibilité k = 5** de la sortie asymétrique : Δ, MDE, t, placebo, plis,
  journalisés dans une 4ᵉ ligne marquée `sensitivity`, sans verdict qui compte.

### 6.7 Prédictions, écrites avant la lecture

- Phase A : PASS-A avec une probabilité d'environ 0,6 ; A2 indiscernable ou mieux.
- B1 positif, autour du MDE (probabilité d'un USEFUL : environ 0,15).
- B2 proche de B1 ou un peu au-dessus. La sortie asymétrique rend au momentum une partie
  des reprises, mais ce sont précisément les reprises qui le font chuter (Daniel et
  Moskowitz 2016). Le signe de B2 − B1 est donc incertain.
- B3 proche de zéro, avec une probabilité d'environ 0,10 d'être « utile ».

## 7. Ce que l'étude pourra dire, et ne pas dire

- Validation : un chiffre de reconnaissance des récessions mesuré sur **14 récessions**,
  hors échantillon, pour la méthode ; pas pour A′ lui-même.
- UMD : l'arrêt aide, n'aide pas ou est indécidable, **à coût nul**, sur un facteur qui
  n'est pas investissable tel quel.
- Rien sur l'or, rien sur les coûts, rien sur une stratégie investissable.

## 8. Ordre des opérations

1. Ce document est commité. Les scripts de validation et de test sont écrits et
   commités.
2. `scripts/longhist_fit.py`, puis `--indpro` : les états.
3. `scripts/longhist_validate.py` : la phase A et la porte de l'axe 1, avec la sortie
   dans `docs/artifacts/longhist/validation.txt`, commitée.
4. `scripts/longhist_umd.py` sans `--read` : l'instrument (échantillon, NaN, MDE), avec
   la sortie dans `docs/artifacts/longhist/instrument.txt`, commitée **avant** la
   lecture.
5. `scripts/longhist_umd.py --read`, **une seule fois** : 4 lignes `longhist_umd`, la
   sortie dans `docs/artifacts/longhist/reading.txt`. Le script refuse toute seconde
   lecture si le registre contient déjà la famille.
6. `docs/RESULTS_LONGHIST.md`.

Au plus 3 modifications de l'hypothèse. Ce document en fixe une seule : le SJM long et
sa sortie asymétrique. Si un défaut de code est trouvé après la lecture, il est corrigé,
déclaré et relu, et le premier chiffre est publié à côté du second.
