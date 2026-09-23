# PRESPEC — des objets de crise, seuls et couplés à l'état du Sparse Jump Model

**Statut : pré-enregistrement, écrit et commité le 23/09/2026 avant toute lecture d'un
rendement conditionné à l'état.** Ce document ne se modifie plus après son commit.
Toute déviation va dans `docs/PROTOCOL_FREEZE.md`.

- Script : `scripts/run_crisis_coupling.py` (sans `--read`, il n'imprime que l'instrument).
- Bibliothèque : `regime_lab/extensions/crisis.py`, tests `tests/test_crisis.py`.
- Données : `scripts/fetch_crisis_data.py`.
- Registre d'essais : famille `crise_coupling`, 9 lignes, une seule lecture.

---

## 1. La question

Le classifieur de l'étude (A′ sparse jump) porte de l'information sur la **variance**
future et aucune sur la **moyenne** (+3,93 points de R² sur la volatilité future, t −3,40 ;
+0,030 point sur les rendements, t 0,27 ; `docs/RESULTS_FINAL.md`). Tous les objets qu'on
lui a couplés jusqu'ici avaient un gain à peu près linéaire dans le marché : le 60/40, le
livre de tendance, huit stratégies (`docs/RESULTS_COUPLAGE_STRATEGIES.md`,
`docs/RESULTS_B1_COUPLAGE.md`). Sur ces objets, une information de variance ne peut que
redimensionner la position, et la règle d'une ligne sur la volatilité fait aussi bien.

**Question.** L'état a-t-il plus d'impact, dans un sens ou dans l'autre, sur des objets
dont le rendement **dépend lui-même de la variance**, ou qui s'effondrent dans un **état
de marché précis** ? Et, pour « mieux utiliser la prédiction » : l'état prédit-il la
**prime de variance** au-delà de ce que le VIX et un quantile de volatilité savent déjà ?

## 2. Ce qui existe déjà, et d'où l'on part

- **Couplages déjà mesurés** : 8 stratégies × 3 couplages, aucun utile ; la tendance
  crypto et la prime overnight sont pénalisées parce qu'elles gagnent en stress. Même
  critère à quatre conditions que celui repris ici (§7).
- **T1, T2, T3, décomposition** : sur le 60/40, la règle d'une ligne fait jeu égal ; la
  valeur vient du ciblage de volatilité (+0,44 %), l'apport propre du régime est négatif
  (`docs/RESULTS_DECOMPOSITION.md`).
- **Extensions** : l'atténuateur de Carver passe son placebo mais perd contre une cible
  de volatilité sans paramètre (`docs/EXTENSIONS.md`).
- **Retournement court terme** : falsifié par `chantiers/reversal-lab` (kill mécanique ;
  le test de Nagel par tercile de VIX y est déjà fait). Écarté ici, voir §3.
- **AQR** (`pilotage/plans_de_recherche/aqr/PRESPEC_AQR.md`) cite Daniel et Moskowitz
  (2016) pour le momentum, mais comme signe d'une inclinaison entre trois livres
  sectoriels, pas comme couplage du facteur UMD. Rien dans le dépôt ne mesure le facteur
  UMD, la prime de variance, les futures VIX ni un indice CBOE.

## 3. La liste restreinte, et ce qui est écarté

**Retenues : quatre stratégies et un test de prédiction.**

| code | objet | pourquoi l'état devrait compter | données | échantillon évalué |
|---|---|---|---|---|
| **VRP** | vente d'un swap de variance à 1 mois sur le S&P 500, synthétique | le gain **est** « variance implicite − variance réalisée » : une prévision de la variance est une prévision du gain | `^GSPC`, `^VIX` (Yahoo) | 2002-04-02 → 2026-09-08, 6 149 séances |
| **VIXF** | vente de futures VIX à maturité constante 1 mois | le contrat converge vers le VIX ; perte maximale en crise ; février 2018 inclus | CBOE Futures Exchange | 2006-12-04 → 2026-09-08, 4 970 séances |
| **UMD** | facteur momentum de Ken French | krachs du momentum dans les « états de panique » (Daniel et Moskowitz 2016) : marché baissier, forte volatilité, puis rebond | Ken French | 2002-04-02 → 2026-07-31, 6 123 séances |
| **BXM** | indice CBOE BuyWrite (achat du S&P 500, vente du call à la monnaie) | version investissable et publique de la vente d'options ; **témoin** : prédit comme septième dispositif (§4) | CBOE | 2002-06-24 → 2026-09-08, 6 088 séances |
| **P** | test de prédiction : l'état prédit-il la variance réalisée future au-delà du VIX ? | c'est la condition pour que la prémisse de la règle tombe sur VRP et VIXF (§4) | `^GSPC`, `^VIX` | 2002-04-01 → 2026-08-07, 6 129 séances |

**Écartées, avec la raison.**

- **Retournement court terme (Nagel 2012).** `reversal-lab` l'a classé en kill
  mécanique : la prime brute passe de +24 bp par jour (années 1990) à −2,7 bp (depuis
  2020), avant tout coût. Le partage par tercile de VIX y est déjà mesuré : depuis 2010,
  seul le tercile haut est positif, à 0,18-0,39 de Sharpe brut. Coupler cette prime à
  l'état, ce serait chercher le sous-ensemble où un effet mort survit encore, ce que
  `reversal-lab` refuse explicitement. Et l'état n'a qu'un épisode de stress après 2010.
- **PUT (CBOE PutWrite).** Économiquement proche de BXM (parité call-put, à la monnaie),
  et son historique quotidien ne commence qu'en 2007 : il manque l'épisode 2002-2003.
  Il est téléchargé mais **ne sera pas lu**.
- **Carry de change.** Le G10 donne 9 paires, sous la règle N ≥ 30. Il n'existe pas de
  points à terme quotidiens publics et gratuits, et les séries `=X` de Yahoo ont le
  défaut d'alignement réparé par M1.
- **Couverture de queue** (achat de futures VIX ou de puts). C'est le miroir exact de
  VIXF : ce qu'elle gagne en stress est ce que VIXF y perd, et cela se lit dans le même
  tableau (colonnes de crise et Sharpe par état). La tester à part compterait deux fois
  la même information.
- **État en probabilité ou en distance continue.** `states.parquet` ne contient que des
  étiquettes. Une distance aux centroïdes demande de réestimer les 49 fenêtres, donc de
  toucher au protocole figé, et elle change l'horloge. C'est une piste ouverte, non
  testée ici.
- **Levier dépendant du régime.** Il n'est pas écarté : c'est le couplage HALF (§6).

## 4. La règle opposable, axe par axe, chiffrée avant la lecture

La règle (`CLAUDE.md`) : est un septième dispositif tout objet qui réunit (1) une latente
ordonnée par la volatilité, (2) une horloge sous ~2 transitions par an, (3) un usage de
dimensionnement ou d'interrupteur, et (4) un objet conditionné de dimension effective
inférieure à 4.

**Sur les quatre axes tels qu'écrits, aucune piste ne diffère franchement.** Même état
A′ (axe 1), même horloge de **0,514 transition par an** (axe 2), usage d'interrupteur
(STOP) ou de dimensionnement (HALF) (axe 3), un seul objet par stratégie, soit une
dimension de **1** (axe 4). Ce qui suit dit en quoi elles diffèrent quand même, et le
chiffre sur lequel l'argument repose. Tous les chiffres de cette section sortent de
l'instrument (`run_crisis_coupling.py` sans `--read`), qui ne lit aucun rendement
conditionné à l'état.

### 4.1 La prémisse de la règle, pour VRP et VIXF

La règle marche parce que, sur les objets testés, **une information de variance ne change
pas la moyenne**. C'est mesuré sur le 60/40 : +0,030 point de R² sur les rendements
futurs, t 0,27. La vente de variance échappe à cette prémisse par construction, car son
gain par unité est une identité :

    gain = K_t − RV_{t→t+21},   avec K_t = (VIX_t / 100)² × 21/252

Une prévision de RV au-delà de K est donc une prévision de la **moyenne** du gain.
**L'argument ne vaut qu'à une condition : que l'information de l'état sur la variance ne
soit pas déjà dans le VIX.** Le quantile de volatilité contre lequel l'état a gagné
+3,93 points de R² est un prévisionniste plus pauvre que le VIX. D'où le test P, qui
tranche la prémisse avant de lire les couplages :

- si P conclut **PRÉDIT**, la prémisse tombe pour VRP et VIXF, et leur couplage n'est pas
  un septième dispositif ;
- si P conclut **NE PRÉDIT PAS AU-DELÀ DU VIX**, VRP et VIXF redeviennent des septièmes
  dispositifs, et la prévision écrite ici est qu'ils ne battront pas la règle de
  volatilité.

**Un chiffre qui ne fonde pas l'argument, publié pour ne pas être caché.** Voici le R² des
rendements sur 21 séances, régressés sur la variance réalisée du S&P 500 sur les mêmes 21
séances (blocs disjoints, hors échantillon, sans état) :

| objet | R² | t HAC 6 | blocs |
|---|---|---|---|
| livre 60/40 (T1-T3, décomposition) | 12,6 % | −7,98 | 292 |
| livre de tendance, 46 instruments (AHL, H-b) | 3,5 % | −1,43 | 277 |
| VRP, ciblée en volatilité | 27,2 % | −5,73 | 292 |
| VIXF, ciblée | 11,0 % | −5,12 | 236 |
| UMD, ciblé | 0,1 % | +0,72 | 291 |
| BXM, ciblé | 8,0 % | −4,81 | 289 |

Le 60/40 co-varie déjà avec la variance réalisée du même mois (effet de levier : les
baisses s'accompagnent de volatilité). **La co-variation contemporaine ne sépare donc pas
les candidats des objets déjà réfutés.** Ce qui les sépare est l'identité ci-dessus, et
elle ne vaut que pour VRP (exactement) et pour VIXF (approximativement, par convergence
vers le VIX). Nous ne la revendiquons pas pour BXM.

### 4.2 La latente, pour UMD

Pour le momentum, ce n'est pas la variance qui compte, c'est **l'état de panique** :
Daniel et Moskowitz (2016) le définissent par un marché baissier sur 24 mois, où la
volatilité est haute et le rebond imminent. Nous avons mesuré le recouvrement de l'état
de stress avec ce marché baissier (`^GSPC` sous son niveau d'il y a 504 séances), hors
échantillon, **sans aucun rendement de stratégie** :

| étiquette | part de stress | P(baissier \| stress) | P(stress \| baissier) | κ avec l'état baissier |
|---|---|---|---|---|
| **A′ sparse jump** | 15,9 % | **64,5 %** | 54,2 % | **+0,50** |
| règle médiane | 48,9 % | 30,4 % | 78,5 % | +0,23 |
| règle du 80ᵉ centile | 21,8 % | 42,1 % | 48,4 % | +0,31 |

À part de stress comparable, l'état A′ recouvre l'état de panique de Daniel et Moskowitz
nettement mieux qu'une règle de volatilité (κ 0,50 contre 0,31). **Sur l'axe 1, pour
UMD, la latente n'est pas « la volatilité sous un autre nom ».** C'est l'axe sur lequel
UMD diffère des sept échecs, et c'est ce chiffre qui l'autorise.

### 4.3 BXM : témoin, prédit comme septième dispositif

BXM co-varie moins avec la variance que le 60/40 (8,0 % contre 12,6 %), et son rendement
moyen est surtout la prime actions, que l'état ne prédit pas. **Nous prédisons, avant de
lire, que tout couplage de BXM est un septième dispositif.** Il reste dans la famille
parce que c'est le produit public, investissable, de la vente d'options. Il dira si ce que
l'on trouve sur VRP et VIXF survit dans un vrai produit.

## 5. Données

Toutes publiques et gratuites. Stockées sous `data/` (gitignoré) avec un manifeste
(`data/manifests/crisis__*.json`) : SHA-256, nombre de lignes, période.

| fichier | contenu | source | période | lignes | SHA-256 (16) |
|---|---|---|---|---|---|
| `raw/prices/cross_asset.parquet` | `^GSPC` et `^VIX`, clôtures (déjà dans le dépôt) | Yahoo | 1990-01-02 → 2026-09-08 | — | `570bb4c9f31732e0` |
| `raw/crisis/vix_futures.parquet` | règlements quotidiens de 274 contrats mensuels VX | CBOE Futures Exchange (archive `CFE_*_VX.csv` jusqu'en 2014, fichiers par contrat à partir de 2013) | 2004-03-26 → 2026-09-22 | 47 238 | `31a98fe879a3c661` |
| `raw/crisis/french_umd.parquet` | facteur UMD quotidien | Ken French, `F-F_Momentum_Factor_daily_CSV.zip` | 1926-11-03 → 2026-07-31 | 26 195 | `302303df13aee16d` |
| `raw/crisis/cboe_indices.parquet` | BXM (quotidien depuis 2002-03-22) et PUT (depuis 2007-01-03, non lu) | CBOE, `*_History.csv` | → 2026-09-22 | 6 162 + 4 961 | `985d2f9a094e3dbe` |
| `raw/macro/rate_cash_3m.parquet` | taux à 3 mois, pour l'excès de BXM | FRED (déjà dans le dépôt) | — | — | `87e2d3b7c7912a48` |
| `cache/states.parquet` | états filtrés, colonne `A' sparse jump` | `run_phase2.py` (jamais relancé ici) | 2002-04-01 → 2026-09-08 | — | `0622c72fa75115cb` |

Trois traitements, déclarés :
- avant le 2007-03-26, les contrats VX étaient cotés à dix fois l'indice (l'ancien
  contrat VBI) : les règlements sont divisés par dix sur ces dates ;
- les règlements inférieurs à 5 sont retirés : ce sont des zéros (contrat listé, non
  échangé) et quelques valeurs de 1,00 aux dates de listage (2008-04-21), alors que le VIX
  n'a jamais clôturé sous 9 ;
- les fichiers par contrat portent un règlement nul avant le 2013-05-20 : on lit alors
  l'archive.

## 6. Construction, figée

- **VRP.** Chaque séance ouvre une tranche de strike K_i = (VIX_i/100)² × 21/252 sur la
  variance réalisée des 21 séances suivantes (somme des carrés des rendements log de
  `^GSPC`). Chaque tranche est réévaluée chaque jour avec le VIX courant comme variance
  implicite de sa vie restante (structure par terme plate). Quelle que soit cette
  approximation, les variations quotidiennes d'une tranche somment exactement à son gain
  final K_i − RV_i (testé). L'échelle de 21 tranches donne un rendement quotidien qui ne
  dépend que de K_{s−1}, K_s et r_s :
  `R_s = c × (K_{s−1}/21 − r_s² − (10/21) × (K_s − K_{s−1}))`. Un swap ne coûte rien à
  l'entrée : c'est un rendement en excès.
- **VIXF.** Position longue sur les deux premiers contrats, pondérés comme l'indice S&P
  500 VIX Short-Term Futures. Le poids du premier contrat vaut la part de la période de
  roulement qui reste à courir ; il passe de 1 à 1/période. On prend l'opposé pour la
  position vendeuse. L'échantillon commence le 2006-09-01 : c'est la première date après
  laquelle les deux premiers contrats ont toujours un règlement (l'instrument vérifie
  qu'aucune valeur ne manque ensuite). Un future ne se finance pas : c'est un rendement
  en excès.
- **UMD.** Le facteur tel quel, long-short : un rendement en excès.
- **BXM.** Variation de l'indice moins le taux à 3 mois du jour, pris à sa date de
  disponibilité.
- **Dimensionnement, identique pour les quatre** (la convention du programme) : poids
  `min(0,10 / σ₆₃, 3)`, où σ₆₃ est la volatilité réalisée annualisée du rendement unitaire
  de la stratégie sur les 63 séances qui finissent la veille. Signal en T−1, position en T.
- **État.** `A' sparse jump` filtré ; la séance d utilise le dernier état daté au plus
  tard de d−1.
- **Couplages.** **STOP** : poids × 0 en stress. **HALF** : poids × 0,5 en stress (levier
  dépendant du régime).
- **Échantillon de chaque stratégie** : de la première séance où le poids, l'état et les
  deux règles de volatilité existent (au plus tôt le 2002-04-01) à la plus proche de la
  dernière donnée et du dernier état.
- **Coûts : aucun**, partout (décision de Guillaume du 23/09). La rotation annuelle du
  poids est rapportée pour chaque bras. La rotation interne de l'échelle de VRP (1/21 des
  tranches renouvelé chaque jour) et le roulement de VIXF ne sont pas comptés.

## 7. La famille de tests et les critères, écrits avant le chiffre

**Famille : 9 tests**, soit 4 stratégies × 2 couplages, plus P. Correction de Bonferroni :
**α = 0,05/9 = 0,0056**.

### 7.1 Couplages

Pour chaque stratégie X et couplage C, sur les mêmes séances, à coût nul :

- **Δ** = Sharpe(X couplée) − Sharpe(X seule) ;
- **MDE** = `selection.protocol.blinded_mde`, bootstrap stationnaire par blocs apparié
  sur jambes démoyennées, blocs 21, 63 et 126, on garde le plus grand, à α = 0,05/9 ;
- **t** = t HAC de lag 6 de la différence quotidienne (`paired_hac_t`) ;
- **placebo** : l'état décalé circulairement contre les dates, 400 rotations d'au moins
  252 séances. Il garde l'horloge et la part de stress et détruit le calage ;
- **témoins** : le même couplage construit sur la règle médiane du programme
  (`volatility_quantile_placebo`, VR 21 j de `^GSPC` au-dessus de sa médiane expansive)
  et sur la règle du 80ᵉ centile expansif (`volatility_tail_rule`), dont la part de
  stress (22 %) est de l'ordre de celle de l'état (16 %).

| verdict | condition |
|---|---|
| **UTILE** | Δ ≥ MDE, t de même signe, Δ au-dessus du 95ᵉ centile du placebo, et Δ supérieur au Δ du même couplage fait avec **chacune** des deux règles de volatilité |
| NON ÉTABLI | Δ ≥ MDE mais une des trois autres conditions échoue |
| SOUS-PUISSANT | 0 < Δ < MDE. Jamais utile |
| PAS UTILE | −MDE < Δ ≤ 0 |
| **PÉNALISANT** | Δ ≤ −MDE, t de même signe, Δ sous le 5ᵉ centile du placebo |
| PAS UTILE (négatif au-delà du MDE, non confirmé) | Δ ≤ −MDE sans le t ou sans le placebo |

Une lecture qui contient une valeur non finie ne reçoit **aucun verdict**.

**MDE mesurés par l'instrument** (jambes démoyennées, aucune moyenne lue) :

| | STOP | HALF |
|---|---|---|
| VRP | 0,426 | 0,203 |
| VIXF | 0,348 | 0,167 |
| UMD | 0,395 | 0,185 |
| BXM | 0,259 | 0,124 |

### 7.2 Test P

Sur chaque séance hors échantillon t qui a 21 séances devant elle :

    y_t = log( Σ_{j=t+1..t+21} r_j² )
    base :    y ~ 1 + log K_t + rang_vol_t
    complet : y ~ 1 + log K_t + rang_vol_t + état_t     (état 1 = calme)

`rang_vol` est le rang centile de la volatilité réalisée sur 21 séances, sur
l'échantillon, comme dans `incremental_information`. Avec log K_t dans les deux
régressions, le coefficient de l'état est le même que la cible soit log RV ou la prime de
variance log(RV/K).

- **PRÉDIT** si |t_état| ≥ z(1 − α/2) = **2,77** (HAC de 21 retards, la convention du
  programme pour des cibles de 21 séances qui se chevauchent) **et** si le R²
  incrémental dépasse le 95ᵉ centile de 400 rotations de l'état.
- Sinon, **NE PRÉDIT PAS AU-DELÀ DU VIX.**
- Sensibilité, qui ne décide jamais : une séance sur 21 (sans chevauchement), HAC 6.
- Descriptif : la même régression sans log K, qui mesure ce que le VIX absorbe.
- Le signe se lit ainsi. Un coefficient **négatif** sur « calme » veut dire qu'en calme
  la variance réalisée est plus basse que ce que dit le VIX, donc que la prime est plus
  grosse en calme, et que STOP devrait aider la vente de variance.

### 7.3 Rapporté à côté, sans jamais décider

- Rendement et volatilité annuels, perte maximale, pire mois, rotation.
- P&L dans six fenêtres datées par le S&P 500 et non par l'état : GFC (2007-10-09 →
  2009-03-09), rebond 2009 (2009-03-10 → 2009-12-31), février 2018 (2018-01-26 →
  2018-02-08), Covid (2020-02-19 → 2020-03-23), rebond 2020 (2020-03-24 → 2020-12-31),
  2022 (2022-01-03 → 2022-10-12).
- Sharpe de chaque stratégie seule sur les séances de stress et sur les séances calmes.
- Pour UMD et BXM, dont l'unité est une position financée à 1× : le même STOP **sans**
  ciblage de volatilité, pour mesurer la part du travail que le ciblage fait déjà
  (Barroso et Santa-Clara 2015).

## 8. Ce que nous attendons, écrit avant la lecture

- **P** : probabilité de PRÉDIT d'environ 0,25. Le VIX est un bien meilleur
  prévisionniste de la variance qu'un quantile de volatilité passée, et l'essentiel des
  +3,93 points devrait y être déjà.
- **VRP et VIXF, STOP** : |Δ| grand, de signe incertain. Arrêter en stress évite l'essentiel
  de 2008 (l'état passe en stress le 2008-01-14), mais manque la prime très riche d'après
  le krach de 2020 (stress du 2020-03-11 au 2020-08-04, puis du 2020-10-01 au
  2021-03-21). Février 2018 est en état calme : les deux bras y perdent autant. Nous
  prévoyons |Δ| sous le MDE.
- **UMD, STOP** : Δ positif attendu, car le krach de 2009 (mars à mai) et celui de
  novembre 2020 tombent en stress. Mais STOP coupe aussi les gains de 2002 et de 2008, et
  le ciblage de volatilité fait déjà une partie du travail. Nous prévoyons Δ entre 0 et
  +0,3, donc sous le MDE de 0,395 : SOUS-PUISSANT est le verdict le plus probable.
- **BXM** : septième dispositif (§4.3), |Δ| petit.
- **HALF** : partout, un Δ d'environ la moitié de celui de STOP, avec un MDE deux fois
  plus petit.

## 9. Ce qui a été vu avant ce verrou, déclaré

- La sortie complète de l'instrument (§4, §5, §7.1). Aucune moyenne, aucun Sharpe, aucun
  rendement conditionné à l'état. Les échantillons, les parts de stress, la volatilité
  réalisée des jambes seules (VRP 15,2 %, VIXF 12,0 %, UMD 10,6 %, BXM 11,7 % pour une
  cible de 10 %), les leviers médians (0,08 ; 0,16 ; 0,87 ; 1,15), la part de séances où
  le plafond mord (0 % ; 0 % ; 0 % ; 2,7 %).
- **Un diagnostic de construction sur VRP, fait avant d'écrire l'instrument, pour
  vérifier que le dimensionnement ne ruine pas la stratégie** : ses six pires séances
  (la pire : −28,5 % le 2018-02-05) et sa perte maximale inconditionnelle (−49 %). Ce sont
  des rendements, non conditionnés à l'état. Ils n'ont rien changé à la construction : la
  convention de dimensionnement du programme est gardée telle quelle.
- Sur les données : la couverture des contrats VX, les dates d'échéance hors règle (jours
  fériés), la liste des règlements aberrants.

## 10. Modifications

Aucune à ce jour (**0 sur 3**). Toute modification après la lecture est une nouvelle
hypothèse déclarée comme telle, journalisée dans `docs/PROTOCOL_FREEZE.md`, et comptée
dans la limite de trois. La famille ne s'agrandit pas : aucune cinquième stratégie,
aucun troisième couplage.
