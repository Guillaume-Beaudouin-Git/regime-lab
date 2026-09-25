# Pistes d'amélioration, et comment présenter honnêtement la partie 2

Regard extérieur d'un conseiller, 23 septembre 2026. Deux questions de Guillaume :
peut-on faire **mieux marcher** le projet, et comment le faire **mieux rendre** en
présentation sans rien cacher ?

**Ce que ce document n'a pas fait.** Il n'a lu aucun rendement de stratégie conditionné
au régime, n'a ajouté aucun essai à `data/trials.parquet` et n'a rien écrit dans
`data/cache/`. Les idées qui méritent un test sont accompagnées d'un **brouillon de
critère**, non lancé : c'est Guillaume qui décide.

**D'où viennent les chiffres.** Ils ont deux sources, et seulement deux :
- les documents commités, cités à chaque fois ;
- des mesures **descriptives** faites pour cette note, qui ne touchent aucun rendement
  de stratégie : dates des états, S&P 500 et VIX autour des transitions, structure de
  labels candidats (cadence, part de stress, accord avec la règle de volatilité),
  corrélations et dimensions effectives. Le script est `scripts/describe_pistes.py`, sa
  sortie intégrale est dans `docs/artifacts/pistes/description.txt`.

Toutes les probabilités données plus bas sont **mon jugement**, pas des mesures.

---

## 0. Le diagnostic, avant les idées

Cinq constats commandent tout le reste. Les trois premiers sont mesurés ici.

**1. Le modèle entre tard et, surtout, sort très tard.** Sur les 1 013 séances de
stress du Sparse Jump Model (A′), **553, soit 55 %, tombent après le creux du S&P 500 de
leur épisode**. Le creux est repéré avec le recul : c'est une description, pas une règle.

| épisode de stress | entrée | creux du S&P 500 | sortie | séances de stress après le creux |
|---|---|---|---|---|
| 2002-2003 (tronqué : l'échantillon commence le 01/04/2002) | — | 09/10/2002 | 27/05/2003, S&P +22,5 % au-dessus du creux | 165 sur 302 (55 %) |
| 2007-2009 | 22/11/2007, S&P −9,5 % sous son plus haut sur un an, VIX 26,8 | 09/03/2009 | 10/11/2009, S&P +61,6 % au-dessus du creux | 167 sur 482 (35 %) |
| 2020-2021 | 11/03/2020, S&P −19,0 %, VIX 53,9 (13,7 trois semaines plus tôt) | 23/03/2020 | 02/04/2021, S&P +79,7 % au-dessus du creux | 221 sur 229 (97 %) |

Le S&P 500 était déjà **20 % au-dessus de son creux le 23/03/2009** et **le 08/04/2020**.
Le modèle est resté en stress encore sept mois en 2009 et presque un an en 2020.

**2. Le marché, lui, sonnait la fin de l'alerte bien plus tôt.** La courbe des futures
VIX est **inversée** quand le contrat le plus proche cote au-dessus du suivant : c'est le
signe d'une panique en cours. Voici la part des séances où elle l'était :

| mois | 04/2009 | 05/2009 | 07/2009 → 09/2009 | 11/2009 | 05/2020 | 07/2020 | 11/2020 → 03/2021 |
|---|---|---|---|---|---|---|---|
| courbe inversée | 43 % | 5 % | 0 % | 0 % | 30 % | 0 % | 0 à 5 % |
| état de stress A′ | 100 % | 100 % | 100 % (52 % en septembre) | 35 % | 100 % | 100 % | 100 % (65 % en mars) |

La courbe est redevenue normale en mai-juillet 2009 et en juillet 2020. À l'inverse, elle
s'est inversée pendant 78 séances en 2011, 88 en 2018, 62 en 2022 et 47 en 2025, quatre
années où A′ n'a jamais été en stress.

**3. Pourquoi les couvertures échouent.** C'est la conséquence directe des points 1 et
2. Une couverture tenue « en stress » gagne dans la chute, puis rend son gain dans la
reprise :
- les options achetées font +29 % à +38 % en 2008, puis −18 % à −23 % au rebond de 2009
  (`docs/RESULTS_REFUGE.md`) ;
- ciblées en volatilité, les actions rapportent autant en stress qu'en calme : Sharpe
  0,62 contre 0,64 (même source) ;
- **le seul objet aidé est celui qui perd précisément dans les reprises : le momentum**
  (Daniel et Moskowitz 2016).

**4. La puissance est le vrai plafond.** La fenêtre hors échantillon ne contient que
**2 récessions NBER, soit 20 mois de récession** (FRED `USREC`, recompté ici). Elle
compte trois épisodes de stress. Les 93,2 % d'exactitude reposent sur ces deux
récessions, et chaque test de Sharpe est sous-puissant par construction : les MDE vont
de 0,12 à 0,73 selon l'étude (`RESULTS_CRISE.md` §10, `RESULTS_REFUGE.md`).

**5. Conditionner un objet qui ne gagne pas ne le fait pas gagner.** C'est la leçon du
plan Two Sigma, qui franchissait pourtant les quatre axes de la règle opposable :
- la latente était un K-means sur 20 variables, orthogonalisé contre la volatilité ;
- l'horloge tournait à 12,5 changements d'état par an hors échantillon
  (`AVANCEMENT.md`) ;
- l'usage était la sélection entre signaux ;
- l'objet était une bibliothèque de signaux sectoriels de rang effectif 8,37
  (`ARBITRAGE.md` §0).

Résultat : niveau A **FAIL**, B **UNDERPOWERED**, C **NOT SHOWN**
(`docs/RESULTS_TWOSIGMA.md`), et un témoin de volatilité d'une ligne fait au moins aussi
bien sur chaque canal (`AVANCEMENT.md`). Le livre témoin avait lui-même un Sharpe net
**négatif** (−0,26 à 5 bp, `RESULTS_TWOSIGMA_LEVEL_A.md`).

**Ce qui distingue vraiment A′ d'une règle de volatilité**, et qui oriente les idées
ci-dessous :
- **il recouvre le marché baissier de Daniel et Moskowitz** nettement mieux : κ 0,50
  contre 0,31 pour la règle du 80ᵉ centile (`PRESPEC_CRISE.md` §4.2) ;
- **ajusté à trois états, il sépare la *gravité*, pas la *phase*** (une fois sur 1992-2006,
  filtré ensuite, §8 de la sortie). Il isole la crise du crédit, 2008-2009 : spread Baa à
  +2,70 écarts-types, drawdown à −2,46. Il met à part les chocs de volatilité : 2011,
  2016, 2020, 2022, 2025. **Aucun troisième état « reprise » n'apparaît spontanément** :
  le rebond de 2009 reste dans l'état de crise jusqu'au 15/09/2009. Réserve : les deux
  états agités ont presque la même volatilité d'entraînement (0,00807 et 0,00797 par
  séance), donc leur ordre est fragile ; c'est un seul ajustement, sans réestimation.

Autrement dit, A′ est un bon **détecteur de récession et de crise du crédit**. C'est un
mauvais **chronomètre de sortie**. Et son information sur la variance est déjà dans le VIX.

---

## A. Faire mieux marcher le projet : six idées classées

**Critère de classement** : la probabilité d'un résultat utile et mesurable par jour
d'effort, pondérée par la proximité avec l'objectif de fond. Les idées écartées, avec
leur raison, suivent le tableau.

| rang | idée | axe de la règle opposable qui diffère, chiffré | effort | P(résultat utile) |
|---|---|---|---|---|
| 1 | Un SJM réduit sur cent ans (1926-2026) : validé sur 14 récessions au lieu de 2, testé sur le momentum | axe 1 pour UMD (κ 0,50 contre 0,31, à remesurer) ; surtout la **puissance** : MDE ≈ 0,15-0,18 au lieu de 0,395 | 3-4 j | 0,6 pour la validation ; 0,35-0,45 pour « l'arrêt aide UMD » ; ≈ 0,10 pour « bat la règle de volatilité » |
| 2 | Sortir plus vite : une sortie asymétrique (« convalescence ») | **aucun, seule** : 1,2-1,6 transition/an, κ 0,35-0,41 avec la règle du 80ᵉ c. ; admissible seulement dans l'idée 1, sur UMD | +0,5 j dans l'idée 1 | 0,10-0,15 |
| 3 | Le régime choisit **quelle** couverture : la corrélation actions-obligations | axe 1 (ρ avec la VR −0,27, contre +0,64 pour le NFCI) et axe 3 (sélection entre couvertures) | 2-3 j | 0,30 sur la variance ; 0,05-0,10 sur le Sharpe |
| 4 | Le socle d'abord : un livre multi-stratégies, le régime en budget de risque | axes 3 (construction) et 4 (9 manches, dimension à mesurer) | 2-3 j | ≈ 0,05 pour l'apport du régime ; le socle lui-même n'est pas chiffrable ici |
| 5 | Un SJM par facteur (Shu et Mulvey) : allocation entre facteurs | axes 3 et 4 (6 facteurs, dimension effective 4,71) ; axe 1 à mesurer | 5-7 j | 0,10-0,15 |
| 6 | Le régime comme entrée d'une prévision de risque sur 46 marchés, jugée en panneau | axe 4, de justesse (dimension effective des volatilités 4,14 en niveau, 6,42 en variation) | 1-2 j | 0,35 comme mesure ; ≈ 0,05 en Sharpe |

---

### Idée 1 — Cent ans au lieu de vingt-quatre : un Sparse Jump Model réduit sur 1926-2026

**L'idée.** On réestime un SJM en walk-forward sur 1926-2026, avec les seules variables
qui existent depuis 1926 : volatilités réalisées, drawdown, rendement sur 24 mois,
dispersion et corrélation moyenne des 49 secteurs, écart Baa-Aaa mensuel décalé d'un
mois. On s'en sert deux fois : d'abord pour **valider le classifieur sur 14 récessions
au lieu de 2**, ensuite pour rejouer **le seul usage qui a montré quelque chose**,
l'arrêt du momentum en stress, sur quatre fois plus de crises.

**Pourquoi ça pourrait marcher là où le reste a échoué.**
- Le mur de tous les verdicts est la puissance, pas forcément l'absence d'effet. Sur
  1927-2026, l'ordre de grandeur du MDE du test UMD passe de **0,395 à ≈ 0,15** pour un
  test unique, et à ≈ 0,18 pour une famille de trois (§6 de la sortie). C'est une
  extrapolation en 1/√T, qui suppose une erreur type par an inchangée avant 2002 : un
  ordre de grandeur, pas un seuil.
- Le mécanisme est documenté sur cette période. Daniel et Moskowitz (2016) montrent que
  les krachs du momentum se concentrent dans les reprises qui suivent un marché
  baissier. Daniel, Jagannathan et Kim (*Tail Risk in Momentum Strategy Returns*, NBER
  w18169, puis *A Hidden Markov Model of Momentum*) montrent qu'un état caché
  « turbulent » d'un HMM anticipe les grosses pertes du momentum sur 1929-2010. C'est
  exactement la famille de modèles du projet.
- Le gain observé sur 2002-2026 existe : +0,166 de Sharpe, au 100ᵉ centile du placebo
  (`RESULTS_CRISE.md` §6). Il échoue sur deux choses seulement : le MDE, et la règle
  du 80ᵉ centile qui fait autant (+0,163).
- **Pour la présentation, c'est le gain le plus sûr** : « 93 % des récessions » deviendrait
  un chiffre mesuré sur 14 récessions, et non sur 2. FRED `USREC` en compte 16 de
  1926-07 à 2026-09 (201 mois), dont 14 après un rodage de dix ans (1937-01 → 2026-09,
  145 mois).

**Axe de la règle opposable.** L'idée ne diffère que sur l'**axe 1**, et seulement pour
UMD : l'état recouvre le marché baissier à κ 0,50, contre 0,31 pour la règle de
volatilité. **Ce κ doit être remesuré sur 1926-2026 avant toute lecture.** S'il tombe au
niveau de la règle de volatilité, le test UMD est un septième dispositif et doit être
déclaré comme tel. Son vrai apport se situe hors des quatre axes : la puissance.

**Données.** Tout est gratuit, et presque tout est déjà sur disque.
- UMD quotidien depuis 1926 : `data/raw/crisis/french_umd.parquet`.
- 49 secteurs quotidiens depuis 1926 et Baa mensuel depuis 1919 : `pilotage/mesures_brutes/`,
  non versionné.
- Le marché quotidien depuis 1926 (`F-F_Research_Data_Factors_daily`) est à télécharger ;
  l'URL a été vérifiée ici : HTTP 200, une vraie archive zip.
- `USREC` sur FRED, sans clé.
- **Pas de VIX avant 1990, pas de macro en premières publications** : c'est un cousin de
  A′, pas A′. On teste la méthode, pas le classifieur figé.

⚠ **Le holdout scellé d'AQR (1971-04 → 1989-12, 49 secteurs) n'a jamais été ouvert**
(`PRESPEC_TWOSIGMA.md` §14). Des variables de dispersion sectorielle y toucheraient. Il
faut trancher avant : ou bien cette étude l'ouvre, une fois et en le déclarant, ou bien
elle se passe des variables sectorielles. **C'est à Guillaume de décider.**

**Effort.** 3 à 4 jours :
- 1 jour pour les variables ;
- 0,5 jour pour les réestimations, environ 180 semestres × 6 valeurs de λ, soit 1 à 2 h
  de calcul ;
- 0,5 jour pour la validation NBER ;
- 1 à 1,5 jour pour le pré-enregistrement et la lecture.

**Probabilité honnête.**
- ≈ 0,6 que la validation tienne, à plus de 85 % d'exactitude équilibrée sur 14
  récessions.
- 0,35 à 0,45 que l'arrêt en stress améliore UMD au-delà du MDE.
- **≈ 0,10 qu'il batte la règle de volatilité appariée.** Sur 2002-2026, l'écart entre les
  deux est de +0,003.

**Risque principal.** Plus de données mesurent plus finement une différence entre A′ et
la règle de volatilité qui vaut peut-être zéro. Il y a aussi le momentum d'avant 1963
(construction, liquidité), et les coûts : UMD n'est pas investissable tel quel.

**Brouillon de critère (non lancé, à commiter avant toute donnée).**
- **Phase A — le classifieur seul, aucun essai.**
  - Réestimation tous les 6 mois, fenêtre expansive, 10 ans de rodage, λ choisi sur la
    fenêtre d'entraînement comme dans l'étude.
  - PASS-A si :
    - l'exactitude équilibrée contre NBER est ≥ 85 % et le κ ≥ 0,40, sur les mois
      postérieurs au rodage ;
    - au moins 12 des 14 récessions postérieures au rodage contiennent au moins un mois
      de stress.
  - On rapporte aussi l'horloge, le κ avec l'état baissier et le κ avec la règle du 80ᵉ
    centile.
  - Si κ(SJM, baissier) − κ(règle, baissier) < 0,10, la phase B est déclarée septième
    dispositif **avant** d'être lue.
- **Phase B — UMD, une famille de 3 essais.**
  - B1 : UMD ciblé à 10 % seul, contre le même avec arrêt en stress.
  - B2 : la même chose avec la sortie asymétrique de l'idée 2, k = 10 séances en tête et
    k = 5 en sensibilité, qui n'est pas un essai.
  - B3 : l'arrêt piloté par le SJM contre l'arrêt piloté par la règle du 80ᵉ centile,
    apparié.
  - Critère : les quatre conditions de `RESULTS_CRISE.md`.
    - Δ ≥ MDE à α 0,05/3, par bootstrap apparié en aveugle, blocs 21, 63 et 126 ; on
      retient le plus grand.
    - Un t HAC 6 de même signe.
    - Δ au-dessus du 95ᵉ centile du placebo par rotation.
    - Δ au-dessus du même couplage piloté par la médiane et par le 80ᵉ centile.
  - Walk-forward en 5 plis d'environ 18 ans (1937-2026) : au moins 3 plis sur 5 doivent
    avoir un Δ positif.
  - Coûts : 5 bp sur les changements de poids. Le rééquilibrage interne du facteur est
    déclaré non compté.
  - Prédiction écrite d'avance : B1 positif, autour du MDE ; B3 proche de zéro.

---

### Idée 2 — Sortir plus vite : une sortie asymétrique, ou état de « convalescence »

**L'idée.** On garde l'entrée en stress du SJM : c'est ce qu'il fait bien, reconnaître
une vraie crise. On y met fin dès qu'une mesure rapide du marché dit que la tempête est
passée. Pour ne pas multiplier les tests, deux déclinaisons au plus :
- depuis 2006, la courbe des futures VIX redevenue normale pendant k séances ;
- depuis 1926, la volatilité à un mois redescendue sous celle à trois mois (RV21 < RV63)
  pendant k séances.

**Pourquoi ça pourrait marcher.**
- C'est la réponse directe au mécanisme mesuré au §0 : 55 % des séances de stress sont
  des séances de reprise, et la courbe s'est normalisée 4 à 9 mois avant la sortie de
  A′.
- La littérature distingue une phase de reprise. Guidolin et Timmermann (2007, *JEDC*)
  ont besoin de quatre régimes (krach, croissance lente, marché haussier, reprise) pour
  décrire conjointement actions et obligations. Johnson (2017, *JFQA*) montre que la
  pente de la courbe du VIX porte le prix du risque de variance, au-delà des autres
  mesures.
- **Un troisième état ajusté ne suffit pas** : il découpe la gravité, pas la phase
  (§0). La sortie doit donc être une règle explicite, pas un état de plus.

**Axe de la règle opposable, et c'est là que l'idée est faible.** Mesuré ici (§3 de la
sortie), le label combiné présente les caractéristiques suivantes :

| label | depuis | transitions/an | part de stress | κ avec la règle du 80ᵉ c. |
|---|---|---|---|---|
| A′ publié | 2002 | 0,53 | 15,9 % | 0,50 |
| règle du 80ᵉ centile | 2002 | 4,26 | 21,8 % | 1 |
| A′, sauf si RV21 < RV63 depuis 5 / 10 séances | 2002 | 1,64 / 1,27 | 8,7 % / 9,9 % | 0,39 / 0,41 |
| A′, sauf si courbe normale depuis 5 / 10 séances | 2006-09 | 1,40 / 1,20 | 6,7 % / 7,9 % | 0,35 / 0,40 |

Le label reste sous 2 transitions par an, sa latente reste liée à la volatilité, il sert
d'interrupteur et son objet est de dimension 1. **Seule, sur une stratégie quelconque,
c'est un septième dispositif**, et il faut l'écrire ainsi. Elle n'est admissible que sur
UMD, par l'axe 1 de l'idée 1, et en y ajoutant la puissance de l'idée 1. D'où sa place
de bras B2 dans l'idée 1.

**Données.** Disponibles : `vix_futures.parquet` (courbe exploitable depuis le
01/09/2006) et les prix du S&P 500. La variante « courbe » ne voit que **deux** épisodes
de stress, ce qui la condamne à la sous-puissance. La variante RV se calcule sur cent
ans.

**Effort.** 0,5 jour dans le pipeline de l'idée 1, 1 jour seule.

**Probabilité.** 0,10 à 0,15. La règle du 80ᵉ centile sort déjà vite de la crise, et la
règle médiane fait mieux que le modèle sur la bascule vers l'or (0,77 contre 0,68,
`RESULTS_REFUGE.md`).

**Risque principal : la contamination, et je la déclare.** J'ai vu le tableau mensuel
de la courbe et les dates des creux avant d'écrire cette idée. Le choix de la variable
est donc informé par la donnée. Pour que le test garde une valeur :
- k doit être fixé par un argument externe (une ou deux semaines de bourse), avec k = 10
  en tête ;
- aucune recherche sur k n'est permise ;
- aucune mesure du chevauchement entre le label candidat et les reprises ne doit être
  faite avant le pré-enregistrement. Je ne l'ai pas faite.

---

### Idée 3 — Le régime choisit *quelle* couverture, pas seulement *quand*

**L'idée.** On ajoute une seconde latente, qui n'est pas la volatilité : **le signe de la
corrélation récente entre actions et obligations** (63 séances). Il sert à choisir, en
continu, l'actif défensif d'un portefeuille :
- obligations d'État quand la corrélation est négative, ce qui correspond à un monde
  déflationniste comme en 2008 ou en 2020 ;
- or, liquidités ou tendance quand elle est positive, ce qui correspond à un monde
  inflationniste comme dans les années 1990 ou en 2022.

**Pourquoi ça pourrait marcher.**
- 2022 est l'échec que tout le monde a vu : A′ était calme, et les obligations n'ont pas
  couvert.
- Mesuré ici (§4 de la sortie), la corrélation est positive sur 100 % des séances de 1990
  à 1997. Elle l'est sur 0 % des séances de 2008 à 2012 et de 2017 à 2020. Elle remonte à
  53 % en 2022, 62 % en 2023 et 65 % en 2024.
- Campbell, Pflueger et Viceira (2020, *JPE*) relient le signe de cette corrélation à celle
  entre inflation et écart de production.
- La bascule vers TLT a sauvé 2008 (+0,1 % au lieu de −22,9 %, `RESULTS_REFUGE.md`) : la
  corrélation y était négative.
- Le test se juge sur la **variance** du portefeuille, le canal où le programme a de la
  puissance (`ARBITRAGE.md` §5c).

**Axes de la règle opposable.**
- **Axe 1 franchi** : la corrélation de Spearman avec la volatilité réalisée à 21 séances
  vaut −0,27 en niveau et −0,25 pour le signe. Pour comparaison, le NFCI, déclaré
  « volatilité sous un autre nom », est à +0,639 (`ARBITRAGE.md` §2).
- **Axe 3 franchi** : l'usage est une sélection entre couvertures, pas une taille ni un
  interrupteur.
- **Axe 2 non franchi** : 3,30 changements de signe par an bruts, 0,93 avec une
  hystérésis de 21 séances.

**Données.** Le S&P 500 et le taux 10 ans depuis 1990 sont dans le dépôt, TLT depuis
2002 et l'or (GC=F) depuis 2000 sont dans `trend_universe_m1.parquet`. Pour remonter à
1962 et couvrir les années 1970-1980, où Campbell, Pflueger et Viceira situent une
corrélation positive (non vérifié ici), il faut `DGS10` sur FRED et le S&P 500
quotidien.

**Effort.** 2 à 3 jours.

**Probabilité.** ≈ 0,30 pour une réduction mesurée de la variance de baisse ; 0,05 à 0,10
pour un gain de Sharpe au-delà du MDE.

**Risque principal.**
- Ce n'est pas le SJM : c'est un autre régime. À présenter comme une suite, pas comme un
  sauvetage du modèle.
- Peu de bascules longues en 36 ans : la fin des années 1990, un épisode positif en
  2005-2006, puis 2021-2022. Le signe est lent, et l'échantillon ne voit que quelques
  changements de régime.
- Recouvrement possible avec le plan Bridgewater (quadrant croissance × inflation) : à
  coordonner.

**Brouillon de critère.**
- **Objet** : actions ciblées à 10 % de volatilité, plus une poche défensive de poids fixe.
- **Règle, déclarée** : TLT si la corrélation sur 63 séances, avec une hystérésis de 21
  séances, est négative ; l'or sinon.
- **Témoins** : une poche fixe 50/50 TLT/or, et la même bascule pilotée par la règle du
  80ᵉ centile.
- **Statistique primaire** : le rapport des variances du portefeuille sur les 5 % pires
  mois des actions, contre le témoin 50/50.
  - PASS si ce rapport est sous le 5ᵉ centile d'un placebo qui fait tourner les labels
    de corrélation.
  - Le MDE est mesuré sous le nul avant la lecture.
- **Le Sharpe est rapporté**, avec le verdict UNDERPOWERED s'il reste sous son MDE.

---

### Idée 4 — Le socle d'abord : un livre multi-stratégies, le régime en budget de risque

**L'idée.** Un Sharpe de 1 à 2 ne viendra pas d'un filtre posé sur une stratégie. Il
viendra de la **combinaison de stratégies positives et peu corrélées**, ciblée en
volatilité. Les 9 stratégies du couplage existent déjà. Le régime n'y entre qu'en
**budget de risque entre les manches**, jamais en interrupteur.

**Pourquoi.**
- Le plan Two Sigma montre qu'on ne sauve pas un objet perdant en le conditionnant (§0,
  point 5).
- La décomposition montre où est la valeur : le ciblage de volatilité rapporte +0,44 %,
  l'apport propre du régime −0,04 % (`RESULTS_DECOMPOSITION.md`).
- La loi fondamentale de la gestion active dit que c'est le nombre de paris indépendants
  qui compte.

**Axes.** Axe 3 (construction de portefeuille) et axe 4 (9 manches, dimension effective à
mesurer sur les rendements **inconditionnels**).

⚠ **Deux pièges déjà là.**
- **La contamination.** Les Sharpe conditionnés au stress des 9 stratégies ont déjà été
  lus : deux gagnent surtout en stress, le momentum y perd (`RESULTS_REFUGE.md`). Donner
  plus de budget en stress à celles qui y ont gagné serait un choix fait après coup.
  Seule une combinaison **inconditionnelle** est propre. Une surcouche de régime devrait
  être mécanique : par exemple une covariance conditionnelle à l'état. Or au niveau B de
  Two Sigma, la covariance par état prévoit **moins bien** que la covariance unique
  (B-1, p 0,92, `RESULTS_TWOSIGMA_LEVEL_B.md` et `AVANCEMENT.md`).
- **Le périmètre.** Sept des neuf stratégies vivent dans le dépôt privé voisin. Le livre
  ne peut pas être publié en détail ici.

**Effort.** 2 à 3 jours pour le livre inconditionnel.

**Probabilité.** ≈ 0,05 que le régime y apporte quelque chose. Je ne chiffre pas la
probabilité que le socle atteigne RESEARCH_PASS, parce que je n'ai lu aucun rendement.
**C'est l'idée la plus utile pour l'objectif de fond, et la moins « régime ».**

**Risque.** Confondre ce résultat avec un résultat du projet. Il faut le présenter comme
la suite, et le régime comme une surcouche.

---

### Idée 5 — Un SJM par facteur : le prolongement publié de Shu et al.

**L'idée.** On estime un SJM **par facteur** (valeur, taille, momentum, rentabilité,
investissement, faible volatilité), sur les variables du rendement actif de ce facteur
plus un contexte de marché. On incline ensuite un portefeuille long seulement par
Black-Litterman. C'est l'article de suite des auteurs mêmes du SJM (Shu et Mulvey,
*Dynamic Factor Allocation Leveraging Regime-Switching Signals*, arXiv 2410.14841,
*Journal of Portfolio Management* 51(3)). Le programme a déjà répliqué leur article
précédent : « le risque se reproduit, le rendement non » (`REPLICATION_SHU2024.md`).

**Pourquoi ça pourrait marcher.** Chaque facteur a sa propre latente, et non la
volatilité du marché. L'objet de base, le marché plus des inclinaisons, a une prime
positive, contrairement au livre sectoriel de Two Sigma.

**Axes.** Axe 3 (allocation) et axe 4 : la dimension effective des six facteurs Fama-French
quotidiens vaut **4,71 sur 6** sur 1990-2026 (§9 de la sortie). L'axe 1 est à mesurer :
la corrélation de chaque latente avec la volatilité du marché. L'axe 2 est inconnu avant
l'ajustement.

**Données.** Les cinq facteurs quotidiens depuis 1990 (`factors_5.parquet`) et UMD sont
déjà dans `data/raw/`. L'historique depuis 1963 et les portefeuilles longs par
caractéristique, qu'il faut pour un portefeuille long seulement, sont gratuits chez Ken
French mais pas encore téléchargés, tout comme une source de faible volatilité (les
portefeuilles de French triés par variance).

**Effort.** 5 à 7 jours.

**Probabilité.** 0,10 à 0,15. La littérature sur le timing factoriel est dégrisante
(Asness et al., 2017, *Contrarian Factor Timing is Deceptively Difficult*). Et le
niveau A de Two Sigma, un sélecteur par régime entre dix signaux, est un **FAIL**.

**Risque.** Réglage de λ dans l'échantillon chez les auteurs. Et l'effet visé, une
inclinaison d'information ratio, est petit devant le MDE.

---

### Idée 6 — Le régime comme entrée d'une prévision de risque, jugée sur 46 marchés

**L'idée.** On ajoute l'état A′ comme variable d'un modèle HAR de volatilité pour chacun
des 46 instruments du livre de tendance. On le juge par la **perte QLIKE en panneau**
(Diebold-Mariano regroupé par date), pas en Sharpe.

**Pourquoi.**
- Le seul résultat positif du programme porte sur la variance.
- Pour le S&P 500, le VIX la contient déjà : +0,20 point au-delà du VIX
  (`RESULTS_CRISE.md` §5). Mais la plupart des 46 instruments (matières premières,
  change, obligations, indices hors US) n'ont pas de VIX dans le dépôt.
- La puissance vit dans les statistiques de panneau (`ARBITRAGE.md` §5c).

**Axes.** Axe 4 seulement, et de justesse : la dimension effective du panneau des log-VR
à 21 jours vaut **4,14** en niveau et 6,42 en variations mensuelles, avec une première
valeur propre de 21,7 sur 46 (§5 de la sortie). L'usage reste le dimensionnement. **C'est
admissible comme mesure, pas comme stratégie.**

**Effort.** 1 à 2 jours.

**Probabilité.** ≈ 0,35 pour un gain QLIKE significatif ; ≈ 0,05 pour qu'il devienne du
Sharpe (le barreau « régime » de la décomposition est négatif).

**Risque.** Le niveau B d'AHL, un test de panneau sur la corrélation, s'est révélé
indécidable (MDE à 49,6 % de l'erreur moyenne). Le MDE doit être mesuré sous le nul avant
toute lecture.

---

### Écartées, avec la raison

- **Acheter la reprise** : « stress et courbe normalisée » comme signal d'achat
  d'actions. C'est une affirmation sur la moyenne, tirée de trois épisodes, et
  contaminée par ce qu'on a déjà vu. Indécidable, et pas honnête.
- **Surpondérer en stress les stratégies qui y ont gagné** : choix fait après lecture
  (idée 4).
- **Un SJM plus rapide** : plus rapide, il ressemble davantage au filtre de volatilité
  (κ 0,70 à 0,74 contre 0,40, `EXPLORATION_SJM_SPEED.md`).
- **Mieux timer la vente de variance** : le VIX sait déjà ce que sait le modèle.
- **Le pairs trading à régimes** (Elliott et Bradrania 2018 ; Endres et Stübinger, dans
  `pilotage/notes_de_lecture/papiers_a_tester/`) : il franchit les quatre axes, mais son objet de base est un
  retournement de court terme. Or ce retournement est tué sur les actions US
  (`chantiers/reversal-lab` : Sharpe de la prime −0,18 depuis 2020), et le seul univers d'actions du
  dépôt a un biais de survie déclaré.
- **Des régimes macro par K-means sur FRED-MD** (Oliveira et al. 2026, dans
  `pilotage/notes_de_lecture/papiers_a_tester/`) : trop proche du niveau A de Two Sigma (contexte K-means puis
  sélection, FAIL) et du plan Bridgewater. À ne rouvrir que si Bridgewater tourne.

---

## B. Mieux rendre la partie 2, honnêtement

### B.1 D'abord, retirer les réserves cachées du matériel actuel

Ces six points ne demandent aucun calcul. Tant qu'ils ne sont pas corrigés, la
présentation cache une réserve.

1. **`PARTIE1_CHEMINEMENT.md` §5, question « Pourquoi ne pas juste regarder le
   VIX ? ».** La réponse actuelle dit : « C'est exactement notre témoin. Le modèle en sait
   plus que lui : +3,93 points de R². » Elle est fausse deux fois. Le témoin est la
   volatilité réalisée sous sa médiane, pas le VIX. Et au-delà du VIX, le modèle
   n'ajoute que **+0,20 point** (t −1,34, `RESULTS_CRISE.md` §5).
   - Réponse proposée : « Le modèle en sait plus qu'une simple règle sur la volatilité
     passée (+3,9 points). Mais le VIX, le prix des options, contient déjà presque tout
     ce qu'il sait (+0,2 point). C'est justement un résultat de la partie 2. »
2. **La tuile « +3,93 pts » du support** (diapositives 2, 10 et 14 de
   `presentation_regimes.pdf`) : ajouter « au-delà d'une règle de volatilité ; +0,2 pt
   au-delà du VIX ». C'est la première question que posera la salle.
3. **« 93 % des récessions »** : dire qu'il y a **2 récessions (20 mois) dans la fenêtre
   hors échantillon**. L'idée 1 réglerait ce point.
4. **« En temps réel, 0 à 13 jours de latence »** : c'est vrai contre les dates du NBER,
   mais trompeur face au marché. Le 11/03/2020, le VIX était déjà à 53,9 et le S&P à
   −19 %. Il faut dire les deux.
5. **La diapositive 3 annonce « 109 configurations testées »** : le registre en compte
   **168** distinctes, sur 890 lignes, au soir du 23/09. Le nombre bouge à chaque
   lecture : le relire au moment de générer le support.
6. **`RESULTS_REFUGE.md`, 2ᵉ partie**, annonce « 12 positives sous-puissantes, 15 pas
   utiles ». Le registre (`sjm_haven_switch`, 27 lignes) en donne **8 UNDERPOWERED et 19
   NOT USEFUL**, ce que confirme le tableau du même document (8 Δ non négatifs). À
   corriger par son auteur. Je ne l'ai pas modifié.

### B.2 Le fil de la partie 2, en cinq diapositives (≈ 4 à 5 minutes)

La partie 1 se termine sur : « un thermomètre du risque, pas une boule de cristal ». La
partie 2 répond à la question qui suit : **à quoi sert un thermomètre du risque ?**

| # | titre (la phrase que la salle doit lire) | contenu | figure |
|---|---|---|---|
| 1 | **« Nous avons fixé les règles du match avant de jouer »** | Trois usages : couper, réduire, basculer vers un refuge. Quatre conditions écrites avant la donnée. Le témoin d'une ligne. Le placebo. | l'entonnoir (F1) |
| 2 | **« 63 usages testés, aucun ne bat une règle d'une ligne »** | 62 couplages et 1 test de prédiction. 18 positifs mais sous le seuil de détection, 44 pas utiles, 0 utile. | F1, suite |
| 3 | **« Le thermomètre reste bloqué sur "fièvre" pendant la convalescence »** | 55 % des jours de stress sont après le creux. La courbe VIX se normalise des mois avant. Les options : +29 à +38 % en 2008, puis −18 à −23 % en 2009. | la convalescence (F2) et chute contre rebond (F3) |
| 4 | **« Là où le risque a un prix, le marché le connaît déjà »** | +4,48 points sans le VIX, +0,20 avec. Là où le risque n'a pas de prix (le momentum), le modèle aide, autant qu'une bonne règle de volatilité. | deux barres (F4) |
| 5 | **« Le meilleur cas, avec ses trois réserves à l'écran »** | Momentum + or : 0,47 → 0,77, mais sous le MDE (0,62), surtout grâce à l'arrêt (0,70), et égalé par la règle de volatilité (0,82). Puis les 3 messages et la suite (idée 1). | l'échelle (F5) |

Si le temps manque, fusionner les diapositives 1 et 2.

### B.3 Les figures, et ce qui les produit

Aucune n'est encore dans `scripts/build_presentation.py`. Il faudra les y ajouter, dans
le style de `timeline()` et `coupling_chart()`. Je n'ai pas modifié ce script.

- **F1, l'entonnoir.** 168 configurations au registre → 63 tests du SJM sur de vraies
  stratégies → 18 positifs sous le seuil → 1 objet au-delà du hasard (le momentum, au
  100ᵉ centile du placebo), égalé par la règle de volatilité → **0 utile**.
  - Source : `data/trials.parquet`, colonne `m_verdict`, familles `b1_regime_coupling`
    (3), la famille des sept stratégies du dépôt privé voisin (21 lignes),
    `crise_coupling` (9), `safe_haven_switch` (3) et `sjm_haven_switch` (27). Lire les
    verdicts, jamais les rendements.
- **F2, la convalescence**, la figure la plus parlante du projet.
  - Contenu : le S&P 500 sur 2007-2010 et 2020-2021, les bandes rouges de stress A′, un
    repère au creux, et en dessous un ruban « courbe VIX inversée ».
  - Message : la bande rouge continue longtemps après la fin du ruban.
  - Sources : `data/cache/states.parquet`, `data/raw/prices/cross_asset.parquet`
    (`eq_us_large`) et `data/raw/crisis/vix_futures.parquet`. La logique est dans
    `scripts/describe_pistes.py` (`vix_curve`, `stress_episodes`).
- **F3, chute contre rebond.** Des barres appariées par stratégie : crise de 2008 et
  rebond de 2009, seule puis avec filtre. **Toujours montrer la chute et le rebond
  ensemble.**
  - Source : `RESULTS_CRISE.md` §4 (par exemple la vente de variance : −20,4 % → +1,0 %
    dans la chute, +36,6 % → +7,4 % dans le rebond, Sharpe 1,30 → 1,30).
- **F4, le VIX sait déjà.** Deux barres, +4,48 pts (t −5,23) et +0,20 pt (t −1,34).
  - Source : `RESULTS_CRISE.md` §5.
- **F5, l'échelle du momentum.** Les barres 0,47 (seul) → 0,70 (arrêt en stress) → 0,77
  (+ or), et à côté 0,82 (règle de volatilité). Le Δ de +0,30 est tracé avec une bande
  grise « zone indiscernable du hasard » jusqu'à 0,62.
  - Source : `RESULTS_REFUGE.md`, 2ᵉ partie.

### B.4 Présenter le sous-puissant et le négatif comme une force, sans maquillage

- **Montrer le seuil, pas seulement l'écart.** Chaque Δ s'affiche avec sa bande de MDE,
  dans le graphique et non en note de bas de page. La phrase pour la salle : « notre
  microscope ne voit pas les écarts plus petits que ça ; celui-ci est plus petit ».
  « Sous-puissant » veut dire « ni oui ni non », jamais « presque oui ».
- **L'analogie de l'essai clinique** : critère écrit avant, groupe placebo et
  comparateur. Le « médicament générique » est la règle d'une ligne. La valeur pour une
  entreprise : **ne pas payer, ni déployer, un modèle complexe quand le générique fait
  aussi bien.**
- **L'horodatage comme preuve.** Montrer une fois le commit du critère, puis celui de la
  lecture (par exemple `7d5b95c` puis `d3e7c48` pour l'étude de crise). C'est la preuve
  qu'aucun chiffre n'a été choisi après coup.
- **Dire la portée** : ce qui n'a pas été testé (la règle opposable, en une phrase). Cela
  rend crédibles les « non ».
- **À ne jamais faire** :
  - « 0,47 → 0,77 » sans ses trois réserves ;
  - « 2008 : −22,9 % → +0,1 % » sans « 2009 : +20,3 % → −6,3 % » ;
  - « +3,93 » sans le VIX ;
  - « 93 % » sans « 2 récessions » ;
  - « en temps réel » sans le 11 mars 2020 ;
  - un Sharpe de la partie 2 présenté comme net : ils sont tous **à coût nul**.

### B.5 Les trois messages qu'un non-spécialiste doit retenir

1. **« Notre modèle reconnaît les crises, mais il tarde à lever l'alerte : plus de la
   moitié de ses jours "de crise" sont en réalité des jours de reprise. »**
2. **« Ce qu'il sait sur le risque, le marché le sait déjà : le prix des options contient
   presque toute son information. »**
3. **« Nous avons testé 63 usages avec des règles écrites à l'avance ; aucun ne bat une
   règle d'une ligne. Savoir dire non, preuves à l'appui, c'est ce qui évite de mettre en
   production une stratégie qui ne marche pas. »**

---

## C. S'il ne fallait en faire que trois

Les corrections du §B.1 passent avant tout, mais elles ne comptent pas : une heure, sans
calcul, et sans elles la présentation cache une réserve.

1. **L'idée 1, avec l'idée 2 comme bras déclaré B2.**
   - Elle attaque la cause commune de tous les verdicts : trois épisodes.
   - Sa phase A ne dépense aucun essai et donne à la présentation son chiffre le plus
     solide : un classifieur validé sur 14 récessions au lieu de 2.
   - Sa phase B teste le seul objet où l'état a montré quelque chose, avec la seule
     correction que le mécanisme suggère.
   - Préalable : la décision sur le holdout 1971-1989.
2. **L'idée 3, la couverture choisie par la corrélation actions-obligations.**
   - C'est la seule idée dont le franchissement de l'axe 1 est déjà mesuré (ρ −0,27 avec
     la volatilité), en plus de l'axe 3, et ses données sont presque toutes sur disque.
   - Elle se juge sur la variance, où le programme a de la puissance.
   - Elle répond à la question que 2022 pose à tout le projet.
3. **L'idée 4, le socle multi-stratégies inconditionnel**, dans le dépôt privé voisin.
   - Ce n'est pas un résultat de régime. Mais c'est la seule route plausible vers un
     Sharpe de 1 à 2.
   - Toute surcouche de régime future a besoin d'un objet qui gagne déjà : le plan Two
     Sigma l'a montré.

**Dans cet ordre**, parce que :
- la première renforce à la fois la présentation et la seule piste vivante ;
- la deuxième ouvre une latente neuve pour 2 à 3 jours ;
- la troisième est la plus utile à terme, mais n'est pas un résultat du projet.

---

## Reproduire

```bash
.venv/bin/python scripts/describe_pistes.py     # ~1 min ; lit data/, n'écrit rien ; FRED en ligne pour le §7
```

Aucune ligne ajoutée à `data/trials.parquet`, rien écrit dans `data/cache/`.
