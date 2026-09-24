# Cent ans au lieu de vingt-quatre : le Sparse Jump Model réduit sur 1926-2026 — résultats

**En une phrase.** On a appliqué la même méthode à cent ans de données, avec les seules
variables qui existent depuis 1926. Elle ne reconnaît plus que **6 récessions sur 14**
(exactitude équilibrée 57,5 %, κ 0,16), pas mieux qu'une règle de volatilité d'une
ligne. Le « meilleur cas » du programme, l'arrêt du momentum en stress, **fond de +0,166
à +0,045** de Sharpe une fois mesuré sur 90 ans avec un test 2,5 fois plus précis, et
passe par le dénominateur.

- Pré-enregistrement : `docs/PRESPEC_LONGHIST.md`, commit **`440c796`**, avant
  l'ajustement du modèle, avant la validation et avant toute lecture.
- Ouverture de la réserve scellée AQR (1971-1989) : `docs/PROTOCOL_FREEZE.md`, commit
  `d1bc935`, avant toute lecture de cette fenêtre.
- Validation (phase A), aucun essai de rendement : `docs/artifacts/longhist/validation.txt`
  (commit `bf6a580`).
- Instrument de la phase B, commité avant la lecture : `docs/artifacts/longhist/instrument.txt`
  (commit `aa25750`).
- Lecture unique : `docs/artifacts/longhist/reading.txt` (commit `9779db2`), 4 lignes
  `longhist_umd` dans `data/trials.parquet`.
- **Aucun coût, partout** (décision du 23/09). Les Sharpe sont bruts, en excès du taux
  sans risque. La rotation est rapportée.

---

## 1. La question, en clair

Tous les verdicts du programme butent sur le même mur : la période hors échantillon du
modèle (2002-2026) ne contient que **2 récessions** et **3 épisodes de stress**. Les
« 93 % des récessions reconnues » reposent sur deux récessions. Le seul usage qui ait
montré quelque chose, l'arrêt du momentum en stress (0,52 → 0,69), restait sous son
seuil de détection.

L'idée : réestimer **la même méthode** (Sparse Jump Model, deux états, réestimation
tous les six mois sur une fenêtre qui s'allonge, états filtrés en temps réel) sur
**1926-2026**. On garde les 30 variables calculables depuis 1926 : volatilités,
tendance et forme du marché, dispersion et corrélation des 49 secteurs, écarts de crédit
et de taux. On obtient ainsi 14 récessions hors échantillon au lieu de 2 (1937-2026).
Trois questions :

1. le modèle reconnaît-il les récessions sur cent ans ?
2. couper le momentum en stress l'améliore-t-il, avec un test enfin puissant ?
3. une sortie plus rapide du stress (idée 2) fait-elle mieux ?

C'est un **cousin** du modèle A′, pas A′ : il n'a ni VIX, ni NFCI, ni données macro en
premières publications.

## 2. Phase A — le classifieur reconnaît-il les récessions sur cent ans ?

**Non.** Période 1937-2026, 23 171 séances, 14 récessions NBER, sans aucun rendement de
stratégie.

| étiquette | exactitude équilibrée | κ | récessions détectées | latence médiane | transitions / an | part de stress |
|---|---|---|---|---|---|---|
| **SJM long (30 variables)** | **57,5 %** | **0,16** | **6 / 14** | −81 j | 2,05 | 12,2 % |
| SJM long, sortie asymétrique k = 10 | 55,7 % | 0,14 | 6 / 14 | −81 j | 1,96 | 8,3 % |
| SJM long, sortie asymétrique k = 5 | 54,9 % | 0,12 | 6 / 14 | −81 j | 2,05 | 7,3 % |
| règle de volatilité médiane | 61,8 % | 0,11 | 11 / 14 | −181 j | 8,85 | 47,5 % |
| règle de volatilité au 80ᵉ centile | 60,4 % | 0,19 | 9 / 14 | −151 j | 3,55 | 16,0 % |
| panique de Daniel et Moskowitz | 64,6 % | **0,31** | 6 / 14 | +28 j | 1,74 | 11,3 % |
| marché baissier sur 24 mois seul | 62,8 % | 0,24 | 6 / 14 | +28 j | 2,26 | 15,3 % |

Pour lire le tableau :
- une récession est **détectée** si au moins 21 séances de stress tombent dans ses mois ;
- la **latence** est l'avance (négative) ou le retard sur le premier mois de récession ;
- la **panique de Daniel et Moskowitz** signale le stress quand le marché est sous son
  niveau d'il y a deux ans **et** que sa volatilité est au-dessus de sa médiane.

**Verdicts pré-enregistrés.**
- **FAIL-A.** Les trois conditions manquent : exactitude équilibrée 57,5 % contre 85 %
  exigés, κ 0,16 contre 0,40, 6 récessions détectées contre 12.
- **A2, mieux qu'une règle de volatilité ? Indiscernable.** L'écart de κ avec la règle
  du 80ᵉ centile vaut −0,037, avec un intervalle à 95 % de [−0,123 ; +0,034] (bootstrap
  stationnaire par blocs, 2 000 tirages). Contre la panique de Daniel et Moskowitz,
  −0,158 [−0,364 ; +0,029] : indiscernable aussi, mais du mauvais côté.

**Récession par récession.** Le SJM long détecte 1973-75, 1981-82, 1990-91, 2001,
2008-09 et 2020. Il rate les huit autres : 1937-38, 1945, 1948-49, 1953-54, 1957-58,
1960-61, 1970 (20 séances de stress, une de moins que le seuil) et 1980.

**Pourquoi : le modèle a appris que « stress » veut dire « Grande Dépression ».** C'est
vérifié, pas supposé (section 10 de la validation, ajoutée après lecture des sections 1
à 9 et descriptive) :
- hors échantillon, le SJM long n'est **jamais** en stress de 1937 à 1959, et 0,3 % du
  temps dans les années 1960. Son premier épisode de stress d'au moins 21 séances
  commence le 14/10/1974 ;
- réestimé sur 1928-1936, il classe en stress 92 % des séances de 1930 et 100 % de celles
  de 1931 : la volatilité du marché y est de 34,2 %, contre 25,0 % dans l'état calme ;
- réestimé sur 1928-1969, il classe encore en stress presque 100 % de 1929-1933, mais
  seulement 19 % de 1937 et 26 % de 1938.

La fenêtre d'entraînement s'allonge, et elle contient toujours 1929-1933. L'état
« stress » est donc défini par la pire crise vue, et rien entre 1937 et 1973 ne lui
ressemble assez. Le score s'améliore au fil des cinq plis d'environ 18 ans : non défini
sur 1937-1952 (aucune séance de stress), puis 51,1 %, 59,1 %, 71,5 % et 85,0 %
d'exactitude équilibrée.

**Sur la période commune 2002-2026** (déclarée dans le PRESPEC, 2 récessions, descriptif) :

| | exactitude équilibrée | κ | transitions / an | part de stress |
|---|---|---|---|---|
| A′, 50 variables | 93,2 % | 0,53 | 0,53 | 16,0 % |
| SJM long, 30 variables | 85,4 % | 0,32 | 2,63 | 24,5 % |
| règle du 80ᵉ centile | 84,1 % | 0,27 | 4,77 | 28,8 % |

Le κ entre A′ et le SJM long vaut 0,448. Les 22 variables d'A′ absentes ici (VIX, NFCI, macro en
premières publications) expliquent une partie des 93 % d'A′, **et la période aussi** :
sur 2002-2026, même la règle du 80ᵉ centile atteint 84 %.

**Le bras de l'idée 2 (sortie asymétrique).** Sa cadence est de **1,96 transition par
an** pour k = 10 (2,05 pour k = 5). Son κ avec la règle du 80ᵉ centile vaut **0,50**
(0,47 pour k = 5), et son κ avec le SJM long 0,79. Il reconnaît un peu moins bien les
récessions que le SJM dont il dérive.

**La porte de l'axe 1, avant la phase B.** L'écart de κ avec le marché baissier, entre
le SJM long et la règle du 80ᵉ centile, vaut **−0,051**, sous la porte de +0,10 (0,145
contre 0,195). Le SJM long ne recouvre pas mieux que la règle de volatilité l'état où le
momentum s'effondre. **La phase B a donc été déclarée « septième dispositif » avant sa
lecture**, avec pour prédiction : B3 ne sera pas utile. Axe 2 : l'horloge tourne à 2,05
transitions par an, juste au-dessus du seuil de ~2.

## 3. Phase B — couper le momentum en stress

Momentum UMD de Ken French, ciblé à 10 % de volatilité, du 04/01/1937 au 31/07/2026,
23 170 séances. Trois tests à α = 0,05/3.

**Le gain de puissance est bien là.** Le seuil de détection (MDE) du test principal passe
de **0,395** sur 2002-2026 à **0,158** sur 1937-2026. Pour B2 et B3, il vaut 0,145 et
0,153.

| bras (arrêt en stress) | Sharpe | rendement / an | volatilité | perte max. | rotation / an | exposé |
|---|---|---|---|---|---|---|
| UMD seul | 1,120 | 12,19 % | 10,88 % | −37,1 % | 4,4 | 100 % |
| arrêt SJM long | 1,166 | 11,80 % | 10,12 % | −37,1 % | 6,3 | 87,8 % |
| arrêt sortie asymétrique k = 10 | 1,107 | 11,41 % | 10,31 % | −37,1 % | 6,2 | 91,7 % |
| arrêt règle médiane (témoin) | 0,749 | 5,69 % | 7,60 % | −34,5 % | 15,1 | 52,5 % |
| arrêt règle du 80ᵉ centile (témoin) | 1,140 | 11,12 % | 9,76 % | −33,6 % | 7,3 | 84,0 % |
| arrêt panique de Daniel et Moskowitz (témoin) | **1,226** | 12,64 % | 10,31 % | **−26,6 %** | 5,6 | 88,7 % |

| test | Δ Sharpe | MDE | t HAC 6 | placebo | plis > 0 | verdict |
|---|---|---|---|---|---|---|
| **B1** arrêt SJM long − seul | **+0,045** | 0,158 | −0,78 | 96ᵉ c. | 2 / 5 | **SOUS-PUISSANT** |
| **B2** arrêt sortie asymétrique − seul | −0,013 | 0,145 | −1,82 | 70ᵉ c. | 2 / 5 | **PAS UTILE** |
| **B3** arrêt SJM long − arrêt règle du 80ᵉ c. | +0,026 | 0,153 | +1,47 | 96ᵉ c. | 2 / 5 | **SOUS-PUISSANT** |
| sensibilité : sortie asymétrique k = 5 − seul | −0,027 | 0,139 | −2,12 | 58ᵉ c. | 2 / 5 | pas utile (ne décide pas) |

Témoins, écart de Sharpe du même arrêt : règle médiane −0,371, règle du 80ᵉ centile
+0,019, panique de Daniel et Moskowitz **+0,106**.

**Ce que disent ces chiffres.**
- **B1 est positif mais trois fois et demie sous son seuil, et son gain passe par le
  dénominateur.** Le rendement annuel **baisse** (12,19 % → 11,80 %) ; seule la
  volatilité baisse davantage (10,88 % → 10,12 %). Le t de la différence quotidienne est
  négatif (−0,78). C'est le piège n° 5 : moins de risque, pas plus de rendement. Le
  premier pli (1937-1952) vaut exactement 0, puisque le modèle n'y est jamais en stress.
- **Le résultat de 2002-2026 ne se généralise pas.** Par sous-période, B1 vaut −0,008
  (1937-1962), +0,031 (1963-2001) et +0,145 (2002-2026). Sur la fenêtre d'A′, l'arrêt
  piloté par A′ refait exactement le +0,166 de l'étude de crise, et le SJM long +0,138.
  L'effet vient donc surtout de 2002-2026. L'étude de crise l'y attribuait au rebond de
  2009 (−17 % → +2 % avec A′) ; cette étude ne l'a pas remesuré pour le SJM long.
- **La sortie asymétrique (idée 2) n'aide pas** : −0,013. Interprétation, non testée
  ici : rendre au momentum les reprises, c'est lui rendre ses krachs, qui s'y
  concentrent selon Daniel et Moskowitz (2016).
- **Le meilleur arrêt est le témoin publié** : la panique de Daniel et Moskowitz fait
  +0,106 et ramène la perte maximale de −37 % à −27 %. C'est un témoin, pas une
  hypothèse testée : il n'a pas de verdict.
- Décrit seulement : sans ciblage de volatilité (UMD brut), le Sharpe passe de 0,51 à
  0,69 avec l'arrêt, et la perte maximale de −63 % à −55 %. Comme en 2002-2026, **le
  ciblage de volatilité fait l'essentiel du travail** : 0,51 → 1,12 à lui seul.

Prédictions écrites avant la lecture (PRESPEC §6.7) : PASS-A avec une probabilité de 0,6
(c'est FAIL) ; B1 positif autour du MDE (positif, mais bien en dessous) ; B3 proche de
zéro (+0,026) ; après la porte de l'axe 1, B3 pas utile (sous-puissant, donc pas utile
au sens du critère).

## 4. Ce qu'on peut dire en présentation

1. **« 93 % des récessions » ne survit pas à cent ans.** La même méthode, sur les
   variables qui existent depuis 1926, reconnaît 6 récessions sur 14 (57,5 %). Elle ne
   fait pas mieux qu'une règle de volatilité d'une ligne. C'est la réponse à la question
   « sur combien de récessions ? » : sur 2, le chiffre est beau ; sur 14, il s'effondre.
2. **Pourquoi : un modèle de régime apprend « la pire crise qu'il a vue ».** Entraîné
   avec la Grande Dépression, il ne signale plus aucun épisode de stress d'un mois
   pendant 37 ans. La diapositive : la part de stress par décennie, 0 % de 1937 à 1959,
   0,3 % dans les années 1960, puis 5 % à 37 %. Leçon utile en
   entreprise : **un détecteur de crise calibré sur l'historique dépend de l'historique
   choisi**.
3. **Plus de données, test plus fin, effet plus petit.** Le seuil de détection du test
   momentum passe de 0,395 à 0,158. L'effet passe de +0,166 à +0,045, et ce qui reste est
   une baisse de risque, pas un gain de rendement. Le « meilleur cas » du programme
   était surtout un effet de 2002-2026 (+0,145 sur cette sous-période, −0,008 sur
   1937-1962).
4. **Le générique fait au moins aussi bien.** La règle publiée de Daniel et Moskowitz
   (marché baissier × forte volatilité) coupe le momentum mieux que notre modèle en
   point (+0,106 contre +0,045). L'écart n'est pas testé : c'était un témoin.

À ne jamais dire : « le modèle protège le momentum depuis 1937 » ; les Sharpe de 1,1 à
1,2 comme des performances investissables (UMD à coût nul, non investissable tel quel) ;
« 57 % » sans « méthode réduite, sans VIX ni macro ».

## 5. Ce qui est prometteur pour l'objectif de fond

- **Rien ici ne porte le régime.** L'arrêt piloté par le modèle ne bat ni le seuil, ni
  la règle de Daniel et Moskowitz.
- **Le momentum ciblé en volatilité** fait 1,12 de Sharpe sur 90 ans à coût nul
  (Barroso et Santa-Clara 2015). C'est un socle, pas un résultat de régime. Il faudrait
  le mesurer avec un véhicule investissable (futures ou ETF factoriels) et des coûts
  institutionnels.
- **La règle de panique de Daniel et Moskowitz** est la sortie qui améliore le plus le
  Sharpe (+0,106) et la perte maximale (−37 % → −27 %). La règle du 80ᵉ centile fait
  +0,019 et −34 %. Ce sont des chiffres de témoins, sans verdict. Elle mérite un
  pré-enregistrement propre, comme hypothèse et non comme témoin. Ce serait une piste neuve, pas une
  modification de celle-ci.
- **Une piste de méthode, non testée** : un état de stress défini par rapport à un
  passé récent (fenêtre glissante, ou volatilité rapportée à son niveau des dix
  dernières années) éviterait qu'une seule crise extrême définisse « stress » pour des
  décennies. Ce serait la deuxième modification de l'hypothèse (sur trois permises), à
  pré-enregistrer avant toute donnée.

## 6. Limites

- **Un cousin d'A′, pas A′.** Pas de VIX, pas de NFCI, pas de macro en premières
  publications. Le verdict porte sur la méthode avec 30 variables de marché, de
  transversal et de taux, pas sur le classifieur figé du programme.
- **Le marché CRSP** (Ken French) remplace le S&P 500 ; le taux court est le bon à un
  mois de Ken French ; la pente des taux est AAA − bon (Fama et French 1989).
- **Samedis de bourse jusqu'en 1952** : les fenêtres en séances couvrent moins de temps
  calendaire avant 1953.
- **UMD** n'est pas investissable tel quel, et sa construction avant 1963 repose sur un
  univers plus petit. Aucun coût n'est compté.
- **14 récessions**, c'est mieux que 2 mais peu : les intervalles de κ restent larges
  (±0,08 à ±0,2).
- **Les dates NBER sont rétrospectives** : c'est une validation, pas un signal.
- **La période 1971-1989 était la réserve scellée du plan AQR** ; elle est maintenant
  vue.

## 7. Écarts au protocole

1. **Deux ajustements interrompus.** Les premiers passages du modèle de tête et de la
   sensibilité INDPRO sont morts avec la session, après avoir journalisé 192 candidats de
   calibration chacun (32 calibrations × 6 valeurs de λ), et avant d'écrire leurs états.
   Le script a été modifié (commit `88a8a8d`) pour ne pas rejournaliser un candidat dont
   le hash de configuration est déjà au registre. Le calcul est déterministe : la relance
   a retrouvé les mêmes λ. Le registre compte **270 lignes `longhist_calibration`**
   (45 × 6), **sans doublon** : 78 écrites par la relance, 192 sautées. Pour INDPRO, voir
   le point 6.
2. **λ hors de la grille déclarée.** Dans 8 calibrations sur 45, aucun candidat ne
   passait la bande de 0,5 à 12 transitions par an. Ces calibrations ont des fins
   d'entraînement en 1964, 1966, 1968, 1970, 1972, 1974, 1978 et 1982, et couvrent 32
   réestimations sur 180. Tous leurs candidats étaient trop persistants : de 0,055 à
   0,495 transition par an à l'entraînement (validation, section 8). `choose_jump_penalty`, inchangé, retombe alors sur la médiane de la
   grille, **λ = 20**, qui n'est pas une valeur de la grille. C'est le comportement du
   dépôt depuis la phase 2, mais le PRESPEC ne l'avait pas écrit.
3. **Section 10 de la validation**, ajoutée après avoir vu les sections 1 à 9. Elle est
   descriptive et ne change aucun verdict. Elle vérifie la cause au lieu de l'affirmer
   (piège n° 6).
4. **Dates de début.** Le PRESPEC annonçait le 04/01/1937 comme première séance hors
   échantillon ; la validation commence le 02/01/1937 (un samedi de bourse, 23 171
   séances), le test UMD le 04/01/1937 (23 170 séances). Aucune conséquence.
5. **Coûts** : le brouillon du conseiller prévoyait 5 bp ; l'étude est à coût nul
   (décision du 23/09), déclaré dans le PRESPEC.
6. **Sensibilité INDPRO** : voir §8.

## 8. Sensibilité INDPRO (déclarée, ne décide rien)

*À compléter à la fin de la relance ; voir `docs/artifacts/longhist/validation.txt`,
section 9.*

## Reproduire

```bash
.venv/bin/python scripts/longhist_fetch.py                 # données publiques, quelques minutes
.venv/bin/python scripts/longhist_fit.py                   # ~1 h, écrit data/cache/longhist_*
.venv/bin/python scripts/longhist_fit.py --indpro          # sensibilité, ~1 h
.venv/bin/python scripts/longhist_validate.py              # phase A, ~1 min
.venv/bin/python scripts/longhist_umd.py                   # instrument, ~4 min
# la lecture a été faite une fois ; le script refuse une seconde lecture :
# .venv/bin/python scripts/longhist_umd.py --read
```

Ken French et FRED révisent parfois leurs séries : un nouveau téléchargement peut
déplacer les chiffres. Les empreintes SHA-256 des fichiers lus sont dans le PRESPEC §2 et
en tête de chaque artefact.
