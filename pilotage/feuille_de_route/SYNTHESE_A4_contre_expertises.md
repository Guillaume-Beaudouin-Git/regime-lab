# Synthèse d'arbitrage — le livre de tendance à 46 instruments comme sujet

Arbitrage de l'évaluation `run_m3_evaluation.py` / `RAPPORT_EVALUATION.md` après quatre
contre-expertises (fuite temporelle, convention de rendement, puissance et déflation,
validité des placebos). Venv du dépôt. **Rien n'a été écrit dans le dépôt.**

Ce que j'ai mesuré moi-même se limite à ce qui départage deux affirmations contradictoires
et à la lecture du code là où un vérificateur accuse le code plutôt que la prose :
`vehicle.py:40-67` (liste `NO_FUTURES_VEHICLE`, branche par défaut de `cost_class`),
`fetch_trend_universe.py:23` (`auto_adjust=True`), `run_m3_evaluation.py:689` (le collapse
de `data/trials.parquet`), `verif_fuite/placebo_form.py:113-131` et
`verif_placebo/verif_placebo.py:144-170` (les deux placebos de régime corrigés).

---

## 1. Les réfutations qui invalident un chiffre publié

**Aucune des quatre attaques ne sauve le livre.** La construction est causalement propre
au bit près — c'est le seul angle où rien n'est tombé — et la lecture la plus favorable
qu'un vérificateur ait pu construire (+0,4510, échantillon purgé de son rodage) reste
sous 0,50 et loin de 0,70. La conclusion de l'étude tient : KILL sur K1, RESEARCH_PASS
non franchie, H-a non établie, H-b non établie et la ligne régime se ferme.

Trois chiffres publiés, en revanche, ne tiennent pas. Ils passent avant le reste.

### 1.1 Le Sharpe du titre n'est pas un Sharpe net en excess

`scripts/fetch_trend_universe.py` télécharge avec `auto_adjust=True` : les neuf colonnes
ETF du panneau sont des séries de **rendement total**. `NO_FUTURES_VEHICLE` n'en retient
que six (AGG, BWX, EMB, HYG, LQD, TIP) ; `cost_class` renvoie **TLT, IEF et SHY** sur
`fixed_income` par sa branche par défaut, donc §7 ne leur facture jamais le taux 3 mois.
J'ai vérifié les trois faits dans le code, ils ne sont pas en discussion.

Six ETF obligataires sont financés, trois ETF obligataires identiques en nature ne le sont
pas. Ce n'est pas une convention défendable, c'est une incohérence d'implémentation : la
liste des huit a été construite pour §6 par une propriété du marché (une exposition IEF se
traite en ZN, donc au tarif future) et réutilisée pour §7, où la question porte sur ce que
la série contient déjà. SHY, ETF Treasury 1-3 ans, dérive à 1,87 %/an contre un taux cash
moyen de 1,77 % : la série encaisse le cash, et l'étude le compte comme du rendement.

| grandeur | publié | les trois financés |
|---|---|---|
| M3, construction commitée | **+0,4029** | **+0,3614** |
| net annuel | 4,392 % | 3,940 % |
| t HAC(6) | +1,725 | +1,548 |
| livre épinglé | +0,2719 | +0,2411 |
| écart K3 | +0,1310 | +0,1204 |
| écart K4 | +0,0138 | +0,0143 |
| construction fidèle au §3 (6 039 séances) | +0,5048 | **+0,4634** |

Conséquence directe : **la réserve que l'étude place en tête de ses NON ÉTABLI se ferme**.
Le rapport écrit qu'il n'est pas établi que le livre soit sous 0,50, parce que la
construction fidèle au §3 lit +0,5048 et rate le seuil de 0,0048. L'omission de
financement vaut 0,0414, huit fois plus large et en sens inverse. Corrigées, les deux
constructions passent sous 0,50. Le rapport, `evaluation.txt` et le script ne mentionnent
nulle part `auto_adjust`, rendement total, coupon, ni TLT/IEF/SHY : la question n'a pas été
tranchée, elle n'a pas été posée.

### 1.2 Le PASS de K5 est produit par une ligne de code non déclarée

`run_m3_evaluation.py:689` :

```python
distinct = trials.dropna(subset=["m_sharpe"]).groupby("config_hash")["m_sharpe"].mean()
```

`data/trials.parquet` porte 5 à 10 Sharpe par configuration. Les moyenner écrase la
dispersion d'un facteur 4,90 (sd 0,2274 sur les Sharpe journalisés, 0,0471 après
moyennage), et c'est cette dispersion qui devient `V[SR]`, la quantité qui décide
entièrement K5. Le §12 fixe le **compte** d'essais ; il ne dit rien de l'estimateur de
variance, et le bloc `DECLARED` non plus.

| pool, N = 87 dans les quatre cas | sd[SR] | SR₀ | SR − SR₀ | K5 |
|---|---|---|---|---|
| moyenne par configuration (la lecture prise) | 0,0471 | +0,117 | +0,286 | PASS |
| médiane par configuration | 0,0986 | +0,245 | +0,158 | PASS |
| première ligne par configuration | 0,1574 | +0,391 | +0,012 | PASS de justesse |
| les 480 Sharpe tels que journalisés | 0,2265 | +0,562 | **−0,159** | **KILL** |
| dispersion publiée par le §1 du pré-enregistrement (D2) | 0,1611 | +0,400 | +0,003 | nul |

Le seul contrôle de robustesse que le rapport porte sur la déflation est le compte d'essais
(N = 87 contre 91), qui déplace la réponse de 0,001 ; la variance, jamais variée, la
déplace de 0,45. S'ajoute le fait que la lecture du critère — §8 écrit « DSR ≤ 0 », ce
qu'une probabilité ne peut pas satisfaire — a été tranchée après avoir vu les deux, et que
c'est celle qui ne mord pas qui a été retenue : sous la convention usuelle (DSR > 0,95),
le critère se déclenche sous les **quatre** pools testés.

Arbitrage : je ne déclare pas K5 KILL, parce que trois collapses défendables sur quatre
donnent un excès positif. Je déclare que **le PASS de K5 n'est pas établi** : la quantité
court de −0,159 à +0,286 selon un estimateur que le pré-enregistrement n'a pas fixé, et
elle traverse zéro. Sans conséquence sur H, que K1 tue déjà ; avec conséquence sur la porte
du §9, qui exige DSR > 0.

### 1.3 Les deux placebos

Deux défauts, tous deux confirmés par deux vérificateurs indépendants.

**Le placebo du bras régime fait tourner le ratio du mauvais côté de la division.** Le bras
construit `clip(V / (sigma_63 × rho), 3)` ; le placebo construit `roll(rho) × clip(V / sigma_63, 3)`.
Le ratio divise dans l'un, multiplie dans l'autre. Mesuré :
max|clip(V/sigma63)/rho − multiplicateur réel| = 2,8e-17, contre 3,9e-02 sous la forme
expédiée. Sur les 11,9 % de séances où rho > 1, le bras coupe le levier et chaque tirage
l'augmente. Le tirage identité k = 0 lit +0,3812 contre +0,4180 pour le bras : la valeur
comparée n'appartient pas à la famille des tirages, et le contrôle de non-biaisitude
déclaré porte sur un autre dispositif.

*Contradiction arbitrée.* Les deux vérificateurs corrigent l'inversion et rapportent des
percentiles différents, 80,2 % et 80,8 %. J'ai lu les deux scripts : ils tirent les **mêmes**
400 décalages (même graine 20260921, même longueur d'échantillon) et diffèrent sur la seule
convention de coût interne au tirage — l'un facture la position d'ouverture comme le fait
`rotation_placebo` (`prepend=0.0`), l'autre ne la facture pas comme le fait
`charge_by_instrument`. La valeur réelle, elle, ne paie pas cette ouverture : le chiffre
cohérent est donc **80,2 %**, et 80,8 % est le même calcul sous la convention expédiée. Ni
l'un ni l'autre n'atteint le 95e percentile ; aucun verdict ne bouge. Le désaccord était de
deux tirages sur 400.

**Le percentile de 88,5 % de M3 est une statistique des 63 premières séances.**
`unscaled_weights` renvoie `.fillna(0.0)`, donc la série de rendements passée à
`portfolio_scalar` est identiquement nulle sur les 253 séances antérieures au 2004-07-07 :
sigma_63 est estimé sur des zéros et `0,10/sigma` explose. Médiane du multiplicateur
0,0717, première valeur 2,0805, 21 séances au-dessus de 3× la médiane, toutes antérieures
au 2004-08-04. Le brut maximum de 43,77 est cela, et rien d'autre : il tombe à 7,96 dès
qu'on retire ces 63 séances.

| échantillon | réel | moyenne placebo | sd | percentile |
|---|---|---|---|---|
| tel qu'expédié | +0,4029 | +0,2646 | 0,1215 | 88,5 % |
| moins les 63 premières séances | **+0,4510** | +0,2970 | **0,0485** | **99,8 %** |

Sur l'échantillon que l'évaluation elle-même déclare purgé d'un artefact de découpe, **P1
franchit le 95e percentile**, et la phrase de conclusion « les deux placebos échouent à
distinguer quoi que ce soit » est fausse pour M3. La fragilité est le rodage et non le bruit
d'échantillonnage : graines 1 / 7 / 12345 donnent 89,0 / 89,0 / 88,8 %, 4 000 tirages
donnent 89,0 %.

À quoi s'ajoute que **P3 n'est pas un test**. L'invariance d'échelle étant exacte à
1,110e-16, `sharpe(épinglé × k) = sharpe(épinglé)` pour tout k : les deux lignes de P3 sont
K3 et K4 recalculés, et P3 ne peut pas échouer. Son témoin, `épinglé × 3,2022`, ne fait pas
varier la seule exposition — le livre épinglé renormalise par le brut variable du livre
brut, ce qui est un second dispositif de conditionnement. Le livre réellement porté à levier
constant lit **+0,3345**, pas +0,2719, ce qui décompose l'écart de K3 en **+0,0626 de
dé-épinglage** et **+0,0684 de ciblage de volatilité proprement dit**.

---

## 2. Les six critères après attaque

| | verdict de l'évaluation | après arbitrage |
|---|---|---|
| K1 | KILL | **KILL, tient et se durcit** — le chiffre du titre tombe |
| K2 | PASS | **PASS, tient** — sur une règle plus faible qu'elle ne se lit, et à sa frontière |
| K3 | UNDERPOWERED | **UNDERPOWERED, tient et se durcit** |
| K4 | RÉFUTÉ, formellement UNDERPOWERED | **tient** — le placebo qui l'appuyait tombe, le mécanisme est mal étiqueté |
| K5 | PASS | **tombe** — PASS non établi ; un pool défendable donne un KILL |
| K6 | PASS | **PASS sous la lecture retenue** — ses chiffres de lecture brute ne sont vérifiés par personne |

**K1 — tient.** Le seuil de §8 porte sur le Sharpe mesuré, et il se déclenche sous toutes
les constructions une fois le financement corrigé : commitée +0,3614, fidèle au §3 +0,4634,
purgée du rodage +0,4510 (non corrigée du financement — personne n'a mesuré la correction
jointe). Ce qui tombe, c'est +0,403 comme grandeur en excess, et la réserve que le rapport
plaçait en tête de ses non-établis.

**K2 — tient, sous une réserve de lecture.** Aucun vérificateur n'a trouvé d'erreur de
calcul : les plis sont partitionnés et imprimés avant lecture, 4 positifs sur 5, un seul
retournement. Mais un retournement étant défini comme un pli de signe opposé au signe plein
échantillon, lui-même positif, « ≥ 3 positifs ET < 2 retournements » se réduit à « ≥ 4
positifs sur 5 » : les deux conditions n'en font qu'une, la taille du test sous un nul
symétrique est 0,1875 et non 0,5000, et le résultat est exactement à la frontière. Le
défaut d'entraînement sur dix ans reste : les plis ne couvrent que 2014-2026, et la
première décennie n'est jugée que par K1.

**K3 — tient, et l'écart rétrécit deux fois.** +0,1310 publié, +0,1204 une fois les trois
ETF financés, et **+0,0684** une fois retirée la part de dé-épinglage que le témoin de
l'étude confondait avec du ciblage de volatilité. Soit 0,28× le MDE verrouillé de 0,246 et
0,23× la résolution propre de la paire (0,300). H-a n'est pas établie, sur un chiffre
environ deux fois plus petit que celui rapporté.

**K4 — tient ; ce qui l'appuyait ne tient pas.** L'écart +0,0138 (+0,0143 financé) vaut
0,06× le MDE verrouillé et 0,3× la résolution propre de sa paire, qui est 0,043. Le placebo
censé le placer sous le nul était inversé ; corrigé, il lit 80,2 % et ne distingue toujours
rien. Le mécanisme annoncé doit être réécrit : les 163,9 % contre 151,6 % décrivent le livre
**non mis à l'échelle**, ce que le corps du rapport dit et que sa conclusion omet, et au
niveau du bras réellement évalué la covariable **augmente** la volatilité du livre (10,7148 %
contre 10,5741 % pour M3 sur les mêmes 5 535 séances), tout son gain non significatif
passant par la moyenne. Le sens du verdict ne change pas ; sa justification mesurée, si.

**K5 — tombe.** Voir §1.2. Le PASS repose sur un estimateur de variance non déclaré, et la
lecture du critère a été choisie après avoir vu les deux.

**K6 — tient sous la lecture retenue, sans preuve sous l'autre.** Le plafond est inerte
(0,00 % des séances) : vérifié indépendamment sur le multiplicateur de M3 **et** sur celui
du bras régime. En revanche les 52,24 % de morsure et les 9,14 % de volatilité sous la
lecture du brut sont des constantes codées en dur dans `K6_FROZEN`, recalculées par
personne. Et la déclaration contre soi-même du rapport doit être renforcée plutôt
qu'atténuée : le brut maximum de 43,8 au 2004-07-07 n'est pas seulement un artefact de
découpe, c'est un artefact de rodage dont la cause est identifiée (sigma estimé sur 253
séances de zéros) et qui disparaît à 7,96 dès qu'on retire 63 séances.

---

## 3. Les portes du §9

**RESEARCH_PASS : non franchie.** Quatre conditions, deux échouent sur le point estimé et
une troisième n'est plus établie.

| condition | seuil | mesuré | état |
|---|---|---|---|
| Sharpe net en excess | > 0,70 | +0,403 publié, +0,361 financé | **échoue** |
| perte maximale | > −25 % | −29,5 % | **échoue** |
| plis positifs | ≥ 3 / 5 | 4 / 5 | satisfait, à sa frontière |
| DSR | > 0 | de −0,159 à +0,286 selon le pool | **non établi** |

PROPFIRM_PASS : non, et le §9 ne l'attendait pas.

**La réserve que la conclusion doit retirer.** Le rapport écrit « ce que l'étude établit,
sans réserve : ce livre ne franchit pas une porte de recherche à 0,70 ». Mesuré sur sa
propre série, 2 000 tirages, bootstrap stationnaire : IC95 [−0,047 ; +0,854] au bloc 21,
[−0,018 ; +0,812] à 63, [−0,015 ; +0,779] à 126. **0,70 est à l'intérieur aux trois
longueurs de bloc**, avec P(tirage > 0,70) = 0,093 / 0,085 / 0,057. La perte maximale n'est
publiée qu'en point estimate ; au même bootstrap son IC95 est [−52,6 % ; −20,9 %] et 12,0 %
des rééchantillons rendent une perte moins profonde que la limite.

La porte est une règle de décision sur la grandeur mesurée, et à ce titre elle échoue deux
fois — le point estimé ne s'approche de 0,70 sous aucune construction admissible. Mais
l'étude applique un standard inférentiel au seuil 0,50 (« non établi ») et un standard de
point au seuil 0,70 (« sans réserve ») dans le même paragraphe. C'est le second standard
qui doit être aligné sur le premier : la porte n'est pas franchie, et l'échantillon ne peut
pas exclure que le vrai Sharpe soit au-dessus d'elle.

---

## 4. Ce qui est sous son MDE reste sous son MDE

**UNDERPOWERED n'est ni un PASS ni un NULL.** K3 et K4 sont dans cet état, et ils n'en
sortent sous aucune attaque. La distinction entre les deux mérite d'être portée, parce
qu'elle n'est pas de même nature :

- **K3 est un non-établissement grossier.** La résolution propre de la paire est **0,300**,
  le MDE verrouillé 0,246, l'écart +0,1204 après correction et +0,0684 une fois le
  dé-épinglage retiré. Cet échantillon ne distingue pas cet écart de zéro, ni de 0,25. On ne
  peut rien conclure sur le ciblage de volatilité, ni qu'il apporte, ni qu'il n'apporte pas.
- **K4 est un non-établissement fin.** La résolution propre de sa paire est **0,043** : le
  dispositif voit des effets sept fois plus petits que le MDE verrouillé, et il mesure
  +0,0138. L'effet du régime, s'il existe, est plus petit que 0,3× ce que ce dispositif sait
  voir. C'est un vide mesuré, pas un vide supposé.

Le §8 fait d'un écart sous le MDE la réfutation de H-a et de H-b ; le §10 interdit de
l'appeler un passage. Les deux lectures sont portées. Pour K4 la réfutation est celle que
les chiffres soutiennent, parce que la paire résout à 0,043. Pour K3, la lecture honnête
est le non-établissement seul.

---

## 5. Ce qui n'a pas pu être établi

1. **Que le livre soit sous 0,50 au sens inférentiel.** 0,50 tombe dans l'IC95 des deux
   côtés. Ce qui se ferme, en revanche, c'est la réserve telle que l'étude l'a écrite : après
   correction du financement, la construction fidèle au §3 lit +0,4634 et non +0,5048.
2. **Que le ciblage de volatilité apporte ou n'apporte rien.** +0,0684 à +0,1204 contre une
   résolution de 0,300.
3. **Que le régime apporte quelque chose.** +0,0138. Voir §4.
4. **La robustesse au coût.** La colonne stress met les cinq bras en négatif, épinglé compris.
   Et deux colonnes présentées comme conservatrices ne le sont pas : `funding=gross` facture
   le brut de huit instruments (0,898) et non celui du livre (3,202) — la vraie borne est
   −0,0690 et non +0,3288 — et `ALL_CASH` n'applique la lecture littérale qu'à §6 ; rendue
   cohérente sur §6 et §7, elle donne +0,0075 pour M3 et −0,0021 pour l'épinglé.
5. **Le différentiel de taux du compartiment change.** Dix séries `=X`, 25,9 % du brut et
   32,9 % de la rotation, portent du spot sans points de terme. Le signal de tendance est
   corrélé au carry : ce n'est pas un bruit centré. Aucun taux court étranger n'existe dans
   `data/raw/macro/` et le §14 interdit de re-télécharger. Sens inconnu, taille inconnue, non
   nulle. Symétriquement NOK=X et SEK=X se voient facturer le taux **USD** sur une série spot
   qui ne l'a jamais encaissé (−0,028 %/an).
6. **La correction dividende du compartiment actions.** Treize indices `^`, 21,1 % du brut,
   sont des indices de prix. Estimation indicative −0,169 %/an, obtenue en appliquant le seul
   rendement du S&P (+1,89 %/an) aux treize.
7. **L'alignement des cinq séries FX inférées** (CHF, JPY, NOK, NZD, SEK), inférées par
   lead-lag contre l'indice dollar. NOK et SEK passent par des ratios de 1,04 et 1,07, ce qui
   est mince. Mesuré en faveur de M1 : le panneau à trois colonnes vérifiées lit +0,431
   contre +0,4029, donc les cinq inférées **coûtent** 0,028 au lieu d'en acheter, et l'échelle
   de fraîcheur est non monotone (amplitude 0,045) contre +0,171 pour une vraie avance d'une
   séance — le canal de fuite est étroit.
8. **La substitution de série de prix de M2.** Possible sur 10 des 46 instruments et 8,9 ans,
   où le plus petit Sharpe résoluble est 1,253 contre une porte à 0,70. Barème de coût, pas
   véhicule.
9. **P2, le placebo qui testerait le signal, est absent, et son absence n'est pas déclarée.**
   Le §11 le spécifie ; `grep -i shuffl` ne rend rien dans le script. P1 et P3 sondent tous
   deux le chemin d'exposition ; rien ne donne une distribution nulle au signal de tendance.
   Défaut du pré-enregistrement lui-même, à journaliser : le §11 écrit « Both are mandatory »
   puis énumère trois placebos.
10. **Aucune correction pour tests multiples sur les t rapportés**, alors que le §13 l'exige
    dès que n_tests > 1 : ni bonferroni, ni holm, ni benjamini, ni AMTP dans le fichier. Les
    trois t primaires (+1,73, +1,86, +0,97) échouent tous à un Bonferroni sur trois tests
    (|t| > 2,39). Direction conservatrice.
11. **La construction jointe n'a été mesurée par personne** : purge du rodage **et** trois ETF
    financés. Les deux corrections vont en sens inverse, la première vaut +0,048 et la seconde
    −0,041 sur la construction commitée. Aucun choix n'est fait entre ces constructions, ce que
    le §12 autorise tant qu'aucune n'est retenue comme titre.
12. **La journalisation des essais.** Les quatre configurations évaluées ne sont pas écrites
    dans `data/trials.parquet`. Le §12 en rend d'autres exigibles : le bras régime tronqué
    défectueux, abandonné après un regard, et les deux constructions du §8(k) dès lors qu'un
    choix est fait entre elles — et le rapport fait ce choix en décidant de ne pas déplacer le
    titre. N défendable ≥ 90 ; mesuré, cela déplace SR₀ de +0,117 à +0,117.
13. **La dernière séance de l'échantillon** porte 46 instruments à poids non nul mais 38
    contributions : `shift(-1)` laisse un NaN en queue sur les huit colonnes FX de M1. Une
    séance sur 5 785, dans le sens qui retire de l'exposition.

---

## 6. Les chiffres qu'aucun vérificateur n'a pu reproduire

| chiffre | pourquoi il n'est pas reproduit |
|---|---|
| K6 lu sur le brut : 52,24 % de morsure, volatilité 9,14 % | constantes codées en dur dans `K6_FROZEN`, jamais recalculées à l'exécution (deux vérificateurs le disent) |
| MDE verrouillé 0,246 / 0,264 / 0,273 | calculés par le §10 sur une autre paire (6 569 séances, 2001-07-05) que ce run ne reconstruit pas ; entrent comme constantes de `DECLARED` |
| corrélations M1 contre les futures CME (0,057 → 0,904 EUR, 0,107 → 0,912 GBP, 0,064 → 0,896 AUD) et ratios de lead-lag 1,04 / 1,07 pour NOK et SEK | aucune série CME dans le cache ; `scripts/audit_fx_alignment.py` n'a pas pu être rejoué |
| M2 : 14 ans d'échantillon perdus, plus petit Sharpe résoluble 0,639 → 1,253 | non remesuré ; par la formule de Lo un vérificateur retrouve 1,256 et non 1,253 |
| le 0,145 de D1 (« si tout le brut doit être financé ») | aucune colonne publiée ne le reproduit : celle qui prétend le porter donne +0,3288, la vraie grandeur −0,0690 |
| 151,6 % de volatilité en état calme | un vérificateur mesure 149,64 % sur l'échantillon actif complet (ratio 1,0953 au lieu de 1,081) et n'a pas reproduit la valeur sur l'échantillon du bras |
| 93,2 % de précision équilibrée contre le NBER, kappa 0,53 | hérité du classifieur, non remesuré ici |
| la colonne stress comme barème observé | la colonne se reproduit (−0,074 pour M3) ; que 5 bp futures / 30 bp cash corresponde à un coût réellement observé n'est vérifié par personne |
| élasticité 0,2785 de p(pass) par unité d'exposition, percentiles de T3 | hérités de l'étude de barrière, non remesurés |
| t de −11,5 du biais de placebo | le biais se retrouve (−0,0699 contre +0,3345) ; le t suppose 400 tirages iid alors que 388 seulement sont distincts |

Se reproduisent en revanche, par des chemins de code indépendants : la sortie complète au
bit près (trois relances, `diff` vide), brut 6,672 %, coût 1,569 %, financement 0,710 %,
net 4,392 %, rotations 15,358 / 58,921, volatilité 10,90 %, brut médian 3,080, l'inertie du
plafond à 0,00 %, l'usage de `available_at` (+1 jour sur 9 177 lignes, 0 identique), le
bootstrap comme Politis-Romano apparié, les 12 t en HAC lag 6, et les 83 configurations de
`data/trials.parquet`.

---

## 7. Ce que cela veut dire pour le programme

**H-b était le sixième et dernier test de la thèse régime, et il échoue.** L'écart est
+0,0138, soit 0,3× la résolution de sa propre paire, laquelle vaut 0,043 : ce n'est pas un
test trop grossier pour voir, c'est un test assez fin pour dire que l'effet est plus petit
que tout ce qu'il peut voir. Il échoue sous les états **avec recul**, qui favorisent H-b ;
le recul vaut +0,0061 de Sharpe et sa portée réelle est de 6 séances au plus (37 séances
divergentes sur 6 377, soit 0,58 %), ce qui rend l'argument a fortiori plus étroit
qu'annoncé mais le laisse valide. Il échoue avec une sensibilité de remplacement négative
(−0,045) et une sensibilité d'états en ligne nulle (+0,008). **La ligne régime se ferme,
comme le §8 K4 et le §15 l'ont prévu avant la mesure.** C'est un résultat, publiable tel
quel : six dispositifs indépendants, un mécanisme connu — 13 transitions d'état en 6 377
séances — et une classification qui porte la variance et non la moyenne.

Une correction doit accompagner cette fermeture : le mécanisme publié n'est pas celui du
bras évalué. Les 163,9 % contre 151,6 % décrivent le livre non mis à l'échelle, et le bras
régime, lui, **augmente** la volatilité du livre traité (10,7148 % contre 10,5741 %). La
ligne se ferme sur un vide mesuré, pas sur le mécanisme que la conclusion invoque.

**Le livre lui-même n'est pas le benchmark que le programme espérait.** Correctement facturé
et en excess, il lit entre +0,36 et +0,46 selon la construction, pour une perte maximale de
−29,5 % et 58,9× de rotation annuelle. Le §15 prévoyait cette issue : « H échoue, H-a
échoue » donne le résultat de dépendance au véhicule — le seul dispositif que ce programme
n'avait pas battu perd sa significativité sous son propre barème de coût déclaré, et la
couche de base que la pratique dit universelle n'est pas démontrée comme telle net de coût
sur cet univers. Avec une réserve à porter : huit instruments sur 46 portent 69 % du coût du
livre, et ce sont ceux dont la classification §7 vient d'être trouvée fausse.

**Ce qui reste à faire, et qui n'est pas de la recherche.** La correction du §7 sur
TLT/IEF/SHY, la déclaration de l'estimateur de variance de K5 et de la lecture de son
critère, la purge du rodage des 63 premières séances, l'absence de P2, et l'écriture des
essais dans `data/trials.parquet` sont des amendements à journaliser dans
`docs/PROTOCOL_FREEZE.md`. Aucun ne ressuscite le livre ; tous conditionnent la valeur du
prochain pré-enregistrement.

---

## 8. Verdict

**KILL.** Le livre meurt sur K1 sous toutes les constructions admissibles une fois le
financement corrigé, et il manque la porte RESEARCH_PASS sur le niveau et sur la perte
maximale. K2 tient à sa frontière, K6 tient sous la lecture déclarée, K5 ne tient pas. H-a
n'est pas établie à une résolution trop grossière pour trancher. H-b n'est pas établie à
une résolution qui, elle, tranche : la ligne régime se ferme.

Le chiffre du titre, +0,403, doit être retiré et remplacé par le Sharpe réellement net en
excess. La phrase « les deux placebos échouent à distinguer quoi que ce soit » doit être
retirée. La phrase « sans réserve » sur la porte à 0,70 doit être retirée. La mort du livre
ne dépend d'aucune des trois.
