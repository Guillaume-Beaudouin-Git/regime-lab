# Le régime choisit-il *avec quoi* se couvrir ? — résultats de la piste `couverture`

**Idée 3 de `docs/presentation/PISTES_AMELIORATION.md`, testée le 23/09/2026.**
- Protocole : `docs/PRESPEC_COUVERTURE.md`, commité **avant** toute lecture en `42c02c8`.
- Une seule lecture (`f449bb3`), soit 6 lignes `couverture` dans `data/trials.parquet` :
  5 tests et 1 référence.
- Sorties brutes : `docs/artifacts/couverture/instrument.txt` et `reading.txt`. Tous les
  chiffres ci-dessous en viennent.
- Aucun coût, rendements en excès du cash.

## En une minute

La corrélation entre actions et obligations change de signe au fil des décennies :
positive dans les années 1990 et depuis 2021, négative de 2000 à 2020. On en fait un
**label de régime**, sans aucun paramètre estimé : le signe de la corrélation sur 63
séances, avec une hystérésis de 21 séances. On pose deux questions.

1. **Ce régime aide-t-il à prévoir le risque d'un portefeuille actions + obligations ?
   Oui, pour un portefeuille à risque égal**, au-delà de la volatilité et du VIX, dans
   l'échantillon comme hors échantillon.
   - Le label ajoute +3,10 points de R², avec un t de 5,50.
   - Sur le même portefeuille, avec les mêmes contrôles, le Sparse Jump Model n'ajoute
     que **0,05 point**.
   - Réserve : le placebo passe **pile au seuil** (95,0 %).
   - Pour le 60/40, le signal est significatif mais échoue au placebo (92,8 %).
2. **Choisit-il mieux la couverture ? Non.**
   - Une poche défensive choisie par le label (obligations ou or) fait un Sharpe de 0,78.
   - Une poche fixe mi-obligations mi-or fait 0,95, une poche fixe en or 0,92.
   - Le label ne bat que la poche 100 % obligations, de +0,06, loin sous le seuil de
     détection (0,36).

**La leçon rejoint celle du programme.** Ce second régime, presque indépendant de la
volatilité, porte lui aussi du **risque** (de la covariance) et pas de **rendement**.

---

## 1. Ce qui a été testé

**Le label** est le signe de la corrélation, sur 63 séances, entre le S&P 500 et
l'obligation à 10 ans (l'opposé de la variation du taux). Il vaut 1 quand la corrélation
est positive : les obligations baissent avec les actions et ne couvrent pas. Il vaut 0
quand elle est négative : les obligations couvrent.
- Il change **0,93 fois par an**.
- Il est positif 40,5 % du temps depuis 1990 : 100 % de 1990 à 1996, 0 % en 2008-2012 et
  en 2017-2020, 48 % en 2022, 70 % en 2024.

**La règle opposable** (`CLAUDE.md`) dit qu'une piste neuve doit s'écarter des sept
échecs sur au moins un axe, chiffré. Celle-ci s'en écarte sur deux :
- **axe 1, la latente** : le label est presque indépendant des règles de volatilité
  (κ +0,08 à +0,09) et de l'état du Sparse Jump Model (κ +0,10) ;
- **axe 3, l'usage** : il choisit entre des couvertures et sert d'entrée à une prévision.

Elle ne s'en écarte pas sur les deux autres : l'horloge (0,93 changement par an) et la
dimension de l'objet (2,82).

**Question V, la variance.** On prévoit la variance réalisée des 21 séances suivantes,
en logarithme, pour deux portefeuilles, sur 1991-2026 (8 945 séances) :
- **V1** : le 60/40 figé du programme ;
- **V2** : actions et obligations à risque égal.

Les contrôles sont :
- le VIX ;
- le rang de volatilité du S&P 500 ;
- la variance récente du portefeuille lui-même, sur 21 séances.

Est-ce que le label ajoute quelque chose ? Le jugement combine un t HAC, un placebo qui
fait tourner le label dans le temps, et une vérification hors échantillon avec des
réestimations annuelles.

**Question S, le Sharpe.** On compare trois poches défensives, sur 2002-2026
(5 977 séances) :

| poche | ce qu'elle tient |
|---|---|
| **P-label** | 50 % actions + 50 % de poche : obligations longues (TLT) quand le label connu la veille vaut 0, or quand il vaut 1 |
| poches fixes | TLT, or, ou 50/50 TLT/or |

Les trois témoins choisissent la poche avec une règle de volatilité (médiane, 80ᵉ
centile, VIX) au lieu du label.

**La combinaison avec le Sparse Jump Model** (objet W) est la bascule « actions en calme,
poche en stress ». Elle a été retirée des tests **avant lecture** : l'instrument montrait
que le label ne choisit l'or que **sur 1 des 832 séances de stress** du modèle. Elle est
rapportée comme référence.

---

## 2. Question V : la prévision du risque

| | V1 : 60/40 | V2 : risque égal |
|---|---|---|
| R² des seuls contrôles (VIX, rang de vol, variance récente) | 53,7 % | 31,8 % |
| **coefficient du label** (log de la variance) | +0,192 | **+0,284** |
| t HAC 21 (seuil 2,58) | +3,88 | **+5,50** |
| R² incrémental (détectable avant lecture : 0,72 / 1,40) | 0,94 pt | **3,10 pt** |
| placebo par rotation, percentile (seuil 95 %) | **92,8 %** (p95 du placebo 1,05 pt) | **95,0 %** (p95 3,05 pt) |
| hors échantillon, 2001-2026 : réduction de l'erreur quadratique | 2,21 % | **5,48 %** |
| Clark-West t | +3,28 | +4,81 |
| encore présent avec la variance sur 63 séances en contrôle ? | oui (t +3,91) | oui (t +5,30) |
| sensibilité, une séance sur 21, HAC 6 | t +3,02 | t +4,33 |
| *2002-2026 : l'état du Sparse Jump Model, mêmes contrôles* | *0,08 pt (t −0,88)* | *0,05 pt (t +0,47)* |
| *2002-2026 : le label, mêmes contrôles* | *0,39 pt (t +2,20)* | *1,34 pt (t +3,03)* |
| *2002-2026 : le label au-delà des contrôles **et** de l'état* | *0,47 pt (t +2,41)* | *1,31 pt (t +3,04)* |
| *la corrélation continue au lieu du label (descriptif)* | *1,23 pt (t +4,43)* | *5,49 pt (t +6,82)* |
| **verdict** | **NOT SHOWN** : significatif, mais sous le placebo | **PREDICTS**, placebo au seuil exact |

**Lecture pour un non-spécialiste.** En régime de corrélation positive, la variance à
venir du portefeuille à risque égal est plus forte que ce qu'annoncent le VIX et sa
volatilité récente. Le coefficient de +0,284 en logarithme correspond à une variance
multipliée par exp(0,284) ≈ 1,33. C'est logique : quand obligations et actions baissent
ensemble, la diversification disparaît. Le VIX ne le voit pas, parce qu'il ne mesure
que les actions.

**Ce qui distingue ce résultat du Sparse Jump Model.** Sur le même portefeuille, avec les
mêmes contrôles et la même période (2002-2026), l'état du modèle n'ajoute **rien** :
0,05 point, t 0,47. Le label ajoute 1,34 point, t 3,03, et reste présent quand on tient
compte de l'état. C'est la première latente du programme qui apporte une information sur
le risque **au-delà du VIX**.

---

## 3. Question S : choisir la couverture

| portefeuille, 2002-2026 | Sharpe | vol. | perte max. | 2008 | rebond 2009 | Covid | 2022 | rotation/an |
|---|---|---|---|---|---|---|---|---|
| actions seules | 0,64 | 10,7 % | −24,9 % | −22,9 % | +20,3 % | −17,0 % | −13,3 % | 2,4 |
| **P-label** (TLT ou or selon le label) | **0,78** | 6,6 % | −16,0 % | −3,9 % | +6,3 % | −4,2 % | −14,4 % | 3,2 |
| P poche TLT | 0,72 | 6,4 % | −21,1 % | −3,9 % | +6,3 % | −4,2 % | −16,0 % | 2,4 |
| P poche or | 0,92 | 7,6 % | −12,8 % | −6,7 % | +17,0 % | −9,4 % | −9,6 % | 2,3 |
| **P poche 50/50 TLT/or** | **0,95** | 6,1 % | −14,1 % | −5,2 % | +11,6 % | −6,8 % | −12,8 % | 2,4 |
| P choisie par la vol. médiane | 0,79 | 6,9 % | −20,1 % | −4,4 % | +6,6 % | −4,2 % | −16,0 % | 7,4 |
| P choisie par la vol. au 80ᵉ c. | 0,93 | 7,2 % | −15,7 % | −0,1 % | +11,9 % | −4,1 % | −13,4 % | 5,2 |
| P choisie par le VIX | 0,75 | 6,9 % | −16,9 % | −2,0 % | +6,3 % | −3,2 % | −15,1 % | 13,5 |
| *réf. : TLT ou liquidités selon le label* | *0,74* | 5,9 % | −15,8 % | −3,9 % | +6,3 % | −4,2 % | −13,0 % | 2,6 |

| test (α 0,05/5) | δ Sharpe | seuil (MDE) | t | placebo | témoins de vol. (médiane / 80ᵉ / VIX) | verdict |
|---|---|---|---|---|---|---|
| P-label contre poche TLT | +0,061 | 0,356 | +0,74 | 54,2 % | +0,067 / +0,214 / +0,031 | **UNDERPOWERED** |
| P-label contre poche or | −0,139 | 0,577 | −1,60 | 54,2 % | −0,134 / +0,013 / −0,169 | **NOT USEFUL** |
| P-label contre poche 50/50 | −0,166 | 0,361 | −1,02 | 54,2 % | −0,161 / −0,014 / −0,196 | **NOT USEFUL** |

**L'idée n'est pas établie sur P**, qui exigeait trois USEFUL. Le label ne fait mieux
que la poche fixe qui perdait déjà, TLT, et encore dans le bruit. Même là, une règle de
volatilité choisit mieux que lui : médiane +0,067, 80ᵉ centile +0,214. En 2022,
l'année qui motivait l'idée, P-label perd −14,4 %, contre −16,0 % avec TLT et −9,6 %
avec l'or. Le label n'était positif que 48 % de l'année : il a basculé tard.

**La statistique du brouillon du conseiller** est rapportée sans décider. Sur les 15
pires mois des actions, la variance de P-label vaut 1,62 fois celle de la poche 50/50.
74,0 % des rotations du placebo font mieux, alors que le brouillon demandait au plus
5 %. En moyenne sur ces mois, P-label perd le moins (−1,65 % contre −2,16 % pour
50/50), mais de façon plus dispersée.

**W, la bascule du Sparse Jump Model** (référence, pas un test) fait 0,57 avec la jambe
choisie par le label, exactement comme avec TLT fixe (0,57, déjà lu dans
`RESULTS_REFUGE.md`). Avec l'or fixe, elle fait 0,68, et 0,64 avec 50/50.

---

## 4. Pourquoi : le label sépare la covariance, pas les rendements

La ligne descriptive clé est le Sharpe de chaque jambe selon le label connu la veille :

| jambe | régime positif (1 454 séances) | régime négatif |
|---|---|---|
| actions | 0,50 | 0,69 |
| TLT | 0,22 | 0,24 |
| or | 0,60 | 0,69 |

**TLT ne rapporte pas moins en régime positif, et l'or ne rapporte pas plus.** Le label
change la façon dont les jambes **bougent ensemble**. Il ne change pas ce qu'elles
**rapportent**. Choisir la jambe selon le label ne peut donc pas relever le Sharpe : on
remplace une jambe par une autre de rendement équivalent, et on perd la diversification
permanente de la poche 50/50.

C'est exactement la leçon du Sparse Jump Model (variance +3,93 points, moyenne +0,03).
Elle se retrouve ici sur une latente presque indépendante de la volatilité.

**Et la combinaison avec le Sparse Jump Model est vide.** Toutes les crises qu'il détecte
sur 2002-2026 (2002-2003, 2008-2009, 2020-2021) tombent dans le monde à corrélation
négative, où les obligations couvrent. Les épisodes de corrélation positive (2005-2006,
2021-2026, 2022 compris) sont des périodes où le modèle est calme. Les deux latentes ne
se recouvrent pas.

---

## 5. Ce qu'on peut dire en présentation

1. **« Nous avons trouvé une information de régime que le VIX n'a pas. »** Le signe de la
   corrélation actions-obligations prévoit le risque d'un portefeuille actions +
   obligations à risque égal au-delà du VIX et de sa volatilité récente : +3,1 points de
   R², t 5,5, et une erreur de prévision 5,5 % plus faible hors échantillon. Sur le même
   portefeuille, le Sparse Jump Model n'ajoute rien (0,05 point). *À dire en même temps :*
   le placebo passe au seuil exact, et pour le 60/40 le résultat ne passe pas le placebo.
2. **« Mais ce régime ne dit pas avec quoi se couvrir. »** La poche choisie par le régime
   fait 0,78 de Sharpe, la poche fixe mi-obligations mi-or 0,95. Le régime décrit comment
   les actifs bougent ensemble, pas ce qu'ils rapportent.
3. **« Les crises du modèle et le changement de corrélation ne se croisent jamais. »** Sur
   832 séances de stress, le régime de corrélation ne recommande l'or qu'une fois. 2022,
   l'échec des obligations, est une année où le modèle était calme.
4. **Le message unificateur** : deux régimes très différents (l'un proche de la
   volatilité, l'autre presque indépendant, κ 0,09) portent tous deux du **risque**, pas
   du **rendement**. Leur place est dans le dimensionnement et la construction du
   portefeuille, pas dans le choix d'actifs.

---

## 6. Ce qui est prometteur pour l'objectif de fond

- **Un intrant de modèle de risque, pas un sélecteur d'actifs.** Le résultat V2 suggère
  que la corrélation actions-obligations devrait entrer dans la prévision de variance
  d'un livre multi-actifs à risque équilibré. La corrélation continue fait encore mieux
  que le label : 5,49 points au lieu de 3,10, à titre descriptif.
- **Le test suivant**, à pré-enregistrer : un livre actions-obligations à risque égal,
  ciblé en volatilité au niveau du livre, avec et sans la corrélation dans la prévision
  de variance, jugé sur la stabilité de la volatilité réalisée puis sur le Sharpe.
  Honnêtement, le programme a mesuré que le ciblage de volatilité lui-même vaut peu :
  +0,44 % d'équivalent certain (γ = 5) sur le 60/40, dans `RESULTS_DECOMPOSITION.md`.
  Un meilleur dénominateur ne mènera pas seul à un Sharpe de 1 à 2, mais il sert tout
  livre futur.

---

## 7. Limites

- **Le placebo est le point faible.** V2 passe à 95,0 %, soit 380 rotations sur 400
  sous le vrai label, exactement le seuil écrit (≥ 95 %). V1 échoue à 92,8 %. Un label
  qui change 0,93 fois par an s'aligne souvent par hasard avec les régimes de
  volatilité. Le t, la sensibilité mensuelle et le hors échantillon sont nets ; le
  placebo, lui, est limite.
- **Une part est mécanique.** La variance d'un portefeuille à risque égal dépend de la
  corrélation de ses jambes. Le contrôle par la variance du portefeuille sur 63 séances
  ne fait pas disparaître l'effet (t 5,30), mais **aucun modèle de covariance complet**
  (volatilité de chaque jambe + corrélation glissante) n'a servi de contrôle. « Au-delà
  d'un modèle de risque standard » n'est **pas établi**. Que la corrélation continue
  fasse mieux que le label indique que l'information utile est la mesure de corrélation
  elle-même, pas son découpage en régime.
- L'obligation de V est reconstituée depuis le taux à 10 ans (durée 7,5, sans convexité).
  Le VIX ne mesure que les actions : aucune volatilité implicite obligataire (MOVE) n'est
  sur disque.
- S ne couvre que 2002-2026. Le régime positif n'y apparaît qu'en 2005-2006, par moments
  en 2013 et en 2021-2026. L'épisode décisif (2022) est unique, et les seuils de
  détection (0,36 à 0,58) sont bien plus grands que les écarts mesurés.
- Tout est à coût nul (rotation de 2,3 à 13,5 par an, rapportée). Les coefficients de V
  sont estimés sur tout l'échantillon, ce qui est la convention du programme pour
  l'information incrémentale. Le hors échantillon, lui, ne réestime que sur le passé.

---

## 8. Écarts

1. **Label sur séances appariées** (déclaré avant lecture) : la construction du
   conseiller effaçait 63 corrélations après chacun des 31 jours fériés obligataires. On
   garde 9 113 séances au lieu de 7 710, et l'horloge ne change pas.
2. **W retiré de la famille avant lecture** (7 → 5 tests, α 0,05/7 → 0,05/5), sur une
   propriété du label seul : 1 séance sur 832. Il est journalisé comme référence.
3. **Libellé du gain hors échantillon.** La sortie affiche « R2 gain … pt ». Le calcul
   est 1 − SSE_complet / SSE_base, c'est-à-dire une réduction relative de l'erreur
   quadratique du modèle de base, et non des points de R² sur la variance totale. Le
   pré-enregistrement parlait d'un « gain de R² hors échantillon > 0 ». Le critère, un
   signe, n'en dépend pas ; les valeurs se lisent en pourcentage de l'erreur du modèle
   de base.
4. **La statistique du brouillon porte sur 15 mois, et non 14 comme annoncé** : le
   quantile à 5 % inclut les ex-aequo au seuil. Elle ne décide rien.
5. **Le percentile du placebo est identique (54,2 %) pour les trois tests P.** C'est
   voulu : les portefeuilles tournés ne dépendent pas du comparateur, qui ne fait que
   décaler vrai et placebo de la même constante.
6. **Déroulement.** La session a été coupée deux fois par une limite de dépense. La
   lecture, lancée une fois, est allée au bout sur une machine très chargée. Les 6
   lignes sont horodatées du 23/09 à 22:35 UTC, et `--read` n'a pas été relancé.
7. **Modifications de l'hypothèse après lecture : 0 sur 3.**

## Reproduire

```bash
.venv/bin/python scripts/run_couverture.py          # l'instrument seul
.venv/bin/python scripts/run_couverture.py --read   # refuse : la famille est déjà au registre
```
