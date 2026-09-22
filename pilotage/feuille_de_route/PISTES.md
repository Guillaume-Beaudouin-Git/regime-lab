# Pistes, classées par espérance de gain

Cinq pistes retenues, quatre écartées. Le classement est par espérance, pas par
intérêt. Chaque piste retenue porte : l'hypothèse exacte, pourquoi elle échappe
aux cinq réfutations acquises, le calcul de puissance a priori, les coûts, le
critère de falsification écrit à l'avance, le placebo apparié, l'effort, une
probabilité de succès honnête, et ce qu'on apprend même en cas d'échec.

Les mesures citées viennent de `CONCLUSIONS.md` §3, produites pendant cette
session. Aucune n'est un rendement de backtest sur une piste neuve : ce sont des
audits de constructions déjà publiées et des descriptions de données.

---

## Piste 1 — Le livre de tendance comme sujet, sur un véhicule correct

**Espérance : la plus haute du programme. Effort : une semaine. P(RESEARCH_PASS)
honnête : 35-45 %.**

Cette piste fusionne les directions 1, 2 et 3 de la commande, parce qu'elles sont
la même piste : retourner le sujet vers le ciblage de volatilité, exploiter les
46 instruments, et travailler le livre plutôt que l'overlay reviennent à une
seule opération.

### L'hypothèse

> Un livre de tendance multi-actifs sur les 46 instruments de
> `trend_universe.parquet`, sur un échantillon conforme à N ≥ 30 (2003-07-17 à
> 2026-09-10, 23,2 ans), avec un véhicule correctement spécifié et un ciblage de
> volatilité de portefeuille quotidien, atteint un Sharpe net **en excess**
> supérieur à 0,70 sur l'ensemble de l'échantillon.

Trois réparations, et trois seulement — c'est la limite de trois modifications
par hypothèse du `CLAUDE.md`.

1. **Réaligner les onze séries de change** sur la session de l'évènement
   (§3.2). Ce n'est pas un paramètre : c'est une correction de calendrier
   vérifiée contre un instrument externe, et elle retire deux jours de retard au
   signal sans rien ajuster.
2. **Back-ajuster les treize futures de matières premières**, ou les remplacer
   par les séries back-ajustées déjà validées contre Databento dans
   le dépôt privé voisin (§3.3).
3. **Poser le ciblage de volatilité au niveau du portefeuille**, qui est la règle
   dure du programme et que le livre de référence ne respecte pas (§3.4).

Le régime entre comme **une covariable parmi d'autres**, jamais comme un
interrupteur, et uniquement dans le module de budget de risque. Il doit battre la
cible de vol sans paramètre pour être conservé ; sinon il sort, et cette sortie
est le sixième constat, pas une déception.

### Pourquoi elle échappe aux cinq réfutations

Les cinq réfutations portent toutes sur le **conditionnement par le régime** :
T1 et T3 testent un conditionneur, l'atténuateur de Carver est un conditionneur,
la couche 3 est un conditionneur, la géométrie de barrière teste un conditionneur.
Aucune ne teste la construction du livre lui-même. Le livre a servi de véhicule
d'essai dans les cinq et de sujet dans aucune.

Concrètement : les cinq réfutations disent toutes « le régime n'ajoute rien
au-delà du ciblage de volatilité ». Aucune ne dit « le ciblage de volatilité ne
suffit pas », et §3.6 montre que sur véhicule futures il produit +0,228 de Sharpe
pour t +2,71, au-dessus du MDE. Cette piste prend au sérieux le seul résultat
positif que le programme ait produit cinq fois et ne s'est jamais donné la peine
de mesurer proprement.

Le risque de réfutation numéro six existe et il faut le nommer : si le livre
réparé rend 0,64 comme le livre non réparé, alors le programme aura établi que
l'univers ne porte pas de prime de tendance suffisante, ce qui est un résultat.

### Puissance a priori

Échantillon conforme : 2003-07-17 à 2026-09-10, **23,2 ans**.

Erreur type d'un Sharpe seul, formule de Lo (2002), `SE = sqrt((1+SR²/2)/T)` :

| vrai Sharpe | SE | Sharpe mesuré nécessaire pour t > 1,96 |
|---|---|---|
| 0,50 | 0,226 | 0,44 |
| 0,70 | 0,232 | 0,45 |
| 1,00 | 0,254 | 0,50 |

Années nécessaires pour résoudre un Sharpe vrai de 0,70 contre zéro à 80 % de
puissance et 5 % : **19,9 ans**. Disponible : 23,2. **La question est
puissante** — ce qui n'était vrai d'aucune question du programme jusqu'ici.

Pour toute comparaison entre deux constructions corrélées du même livre, en
revanche, le MDE par bootstrap par blocs mesuré ici est de **0,246 à 0,273** de
Sharpe selon la longueur de bloc (21, 63, 126 jours). Toute comparaison interne
en dessous de cet écart doit être déclarée sous-puissante et non passée, comme
l'a été l'atténuateur.

À déclarer d'avance : le nombre effectif de paris est **11,4** par ratio de
participation et **20,7** par entropie du spectre, pour 46 instruments. Ce n'est
pas 46. Toute invocation de la loi fondamentale doit utiliser ces chiffres-là.

### Coûts

Barème du programme, appliqué par classe. Rotation annuelle mesurée sur la
construction de référence : 14,79 fois le livre, dont 5,13 sur le change.

| classe | véhicule actuel | bp AR | véhicule à spécifier | bp AR |
|---|---|---|---|---|
| change | comptant de détail | 15,0 | futures CME (6E, 6B, 6A, 6J, 6C…) | 1,0-2,0 |
| obligataire | ETF (AGG, TLT, LQD…) | 7,5 | futures (ZN, ZB, ZF, FGBL…) | 1,0-2,0 |
| actions | indices non tradables | 1,0 | futures (ES, NQ, RTY, FESX…) | 1,0 |
| matières premières | front-month brut | 1,5 | futures back-ajustés | 1,0-1,5 |

Le point mort du livre de référence est à 15,7 bp aller-retour ; le changement de
véhicule fait passer la dérive annuelle de 1,130 % à environ 0,15-0,30 %. **Le
gain de véhicule est plus grand que tout gain de signal jamais mesuré dans ce
programme**, et il ne coûte pas un degré de liberté.

Contrainte à déclarer : un véhicule futures impose une taille minimale de
contrat. Sur les micros (MES, MNQ, M6E…), le dépôt privé voisin a mesuré qu'un livre de
100 k$ sur 7 micros voit son Sharpe tomber de 0,50 à 0,31 par seule granularité.
Le livre de recherche doit donc être évalué en poids continus, et la contrainte
de lot rapportée séparément comme une dégradation d'implémentation.

### Critère de falsification, écrit à l'avance

Écrire ces quatre lignes dans le pré-enregistrement avant toute mesure, et les
tenir.

- **K1.** Sharpe net en excess ≤ 0,50 sur l'échantillon complet → KILL. Le livre
  réparé n'aura alors pas fait mieux que le livre non réparé et la thèse du
  véhicule est fausse.
- **K2.** Moins de 3 plis sur 5 positifs en walk-forward → KILL.
- **K3.** Le ciblage de volatilité de portefeuille n'améliore pas le Sharpe net
  d'au moins le MDE de 0,246 → la direction 1 de la commande est réfutée, et le
  ciblage de volatilité n'est pas le socle que la pratique annonce. C'est un
  résultat publiable qu'il faut écrire comme tel.
- **K4.** Le régime, entré comme covariable de budget de risque, n'améliore pas
  le Sharpe net d'au moins le MDE → il sort du livre définitivement, sixième et
  dernier constat.

Et une règle de discipline : **le rendement net en excess est la seule métrique
de décision.** Les extensions ont été jugées sur du brut pendant un an.

### Placebo apparié

Deux, obligatoires, tous deux déjà implémentés dans le dépôt.

1. **Rotation circulaire du profil de levier** (`run_extensions.py` lignes
   ~170-190). Elle préserve exactement l'autocorrélation du levier et ne détruit
   que son alignement calendaire. C'est le placebo qui a fait passer l'atténuateur
   au 99ᵉ percentile — il est non biaisé, vérifié par le fait que sa moyenne de
   +0,428 tombe sur le Sharpe inconditionnel de 0,44.
2. **Signaux re-dérivés sur rendements mélangés par blocs**, jamais permutation
   de la colonne de P&L. C'est la convention de le dépôt privé voisin (K2 de son
   pré-enregistrement) et elle est plus stricte.

Plus une exigence de transparence : la décision de passer au véhicule futures
a été prise **après** avoir vu que le mélange de détail tue l'effet (§3.6). Ce
n'est pas hors échantillon et il faut l'écrire, exactement comme `EXTENSIONS.md`
a écrit que le choix de conditionner sur la volatilité de marché venait d'avoir
vu l'écart des terciles.

### Effort

Cinq à sept jours de travail effectif.

| étape | durée | note |
|---|---|---|
| pré-enregistrement gelé et commité | 0,5 j | avant toute donnée |
| réparation du véhicule (change, rolls, prints) | 1,5 j | les séries back-ajustées existent déjà dans le dépôt privé voisin |
| construction du livre, ciblage de vol, coûts par classe | 1,5 j | `extensions/trend.py` fournit 80 % du squelette |
| walk-forward 5 plis, bootstrap, MDE, DSR | 1,5 j | `analysis/` fournit tout |
| placebos et rédaction | 1,5 j | |

Aucune donnée à télécharger. Aucun long calcul : `run_phase2.py` n'est pas
concerné et ne doit pas être relancé.

### Probabilité honnête et ce qu'on apprend en cas d'échec

Point de départ mesuré : **0,638 net à 1 bp, brut de l'ajustement en excess,
sans aucune réparation**. Il faut 0,70 en excess.

Les trois réparations vont dans le bon sens mais aucune n'est chiffrable à
l'avance sans faire la mesure, et l'ajustement en excess va dans le mauvais sens
(sur un livre entièrement en futures il est neutre, puisque les rendements de
futures sont déjà des excess — c'est un argument de plus pour le véhicule
futures). Contre : les 2 246 à 7 722 ans de T1 rappellent à quel point ce
programme a surestimé ce qu'il pouvait voir, et le dépôt privé voisin a déjà tué une
version à 10 instruments sur 12,5 ans avec un DSR de −0,33.

**P(Sharpe net en excess > 0,70) ≈ 35-45 %. P(> 1,00, PROPFIRM_PASS) ≈ 5 %.**

En cas d'échec, on obtient trois choses qui manquent au programme depuis le
début :

1. **Un livre de référence audité, à coûts corrects, en excess, sur 23,2 ans et
   46 instruments.** C'est le dénominateur de toutes les comparaisons futures des
   deux dépôts. Aujourd'hui il n'existe pas : le témoin de toutes les mesures
   publiées est brut, sans ciblage de vol, et 24 % de ses instruments sont
   décalés.
2. **La réponse à la direction 1** : le ciblage de volatilité est-il un socle ou
   un artefact de coût ? Le résultat est publiable dans les deux sens, et §3.6
   montre qu'il est serré.
3. **Le sixième et dernier constat sur le régime**, obtenu cette fois sur un
   livre qui mérite d'être conditionné.

---

## Piste 2 — Écrire le résultat de dispersion macro

**Espérance : haute. Effort : deux à trois jours. P(succès) ≈ 85 %.**

### L'hypothèse

Il n'y en a pas, et c'est la force de la piste. H1 et H2 sont entièrement
mesurées. Elles forment un seul résultat original :

> *La macro des pays développés n'a plus assez de dispersion transversale pour
> être tradée en relatif, et en absolu c'est du market timing.*

Avec la cause datée : écart-type des taux courts des neuf pays de la zone euro,
2,35 % avant mars 1999, **exactement 0,00 %** après. Dispersion G10 de 2,30 % en
1990-1998 à 0,89 % en 2016-2026.

### Pourquoi elle échappe aux réfutations

Elle ne cherche rien. C'est de la rédaction d'une mesure acquise. Le seul risque
est de mal l'écrire.

### Puissance, coûts, falsification, placebo

Sans objet : rien de neuf n'est mesuré. Les placebos existent déjà — carte des
signes de H1 au 7ᵉ percentile sur 400 tirages, signal au 99ᵉ ; zéro cellule sur
neuf au-dessus de |t| 1,96 pour H2 contre un seuil de Šidák de 2,77.

Une seule chose à ne pas faire : ne pas retourner les signes de H1. Les retourner
donnerait +0,38 et détruirait le résultat.

### Ce qu'on apprend en cas d'échec

Il n'y a pas d'échec possible au sens de la mesure. Le risque est éditorial : le
résultat peut ne pas trouver de lecteur. C'est la piste avec la plus haute
probabilité de produire quelque chose qui « marche » au sens où un travail de
recherche marche — un énoncé original, causalement expliqué, daté, et qu'on peut
défendre.

Elle est indépendante de la piste 1 et doit tourner en parallèle.

---

## Piste 3 — Ce qui manque au dossier pour convaincre

**Espérance : haute. Effort : deux à quatre jours. P(succès) ≈ 90 %.**

Le mémoire est correct. Ce qui lui manque pour porter n'est pas du résultat, c'est
de la mise en évidence. Quatre manques, du plus au moins coûteux.

**1. Le résultat central n'a pas de figure.** « Le régime porte la variance et
pas la moyenne » est le seul énoncé que le jury retiendra, et il vit dans un
tableau à quatre lignes. Une figure : R² incrémental sur les rendements contre
R² incrémental sur la volatilité, un point par famille, le placebo d'une ligne à
l'origine. Cinq points, deux axes, l'argument entier. Le dépôt impose déjà de
sortir matplotlib de la logique métier, donc c'est un module de tracé séparé.

**2. Le chiffre qui structure tout n'est écrit nulle part au premier plan.**
Treize transitions en 6 377 jours. Vingt-cinq ans donnent treize décisions. Il
est enterré dans `RESULTS_BARRIER.md`. C'est l'explication mécanique de toutes
les réfutations et il devrait ouvrir le mémoire.

**3. Les neuf défaillances silencieuses sont le meilleur matériau du dossier et
elles sont en annexe.** Six sur neuf étaient dans le code du projet. La n° 7 —
la convention d'étiquetage inversait le meilleur classifieur, donc long du plus
mauvais état — est exactement ce qu'un jury veut entendre d'un candidat qui vise
un fonds systématique. Un candidat qui a trouvé et publié l'erreur qui rendait
ses propres résultats trop flatteurs vaut plus qu'un candidat dont tout marche.

**4. L'audit du véhicule de cette session.** §3.2 à §3.6 : onze instruments
décalés d'une session détectés par lead-lag contre un instrument externe, une
vérification sur un évènement daté (Brexit), des métaux non back-ajustés
identifiés par corrélation avec une série back-ajustée validée, et un témoin
central qui perd sa significativité dès qu'on lui applique le barème de coûts de
son propre projet. C'est une démonstration de méthode, et elle est neuve.

Rien de tout cela ne demande un calcul nouveau. Ce sont des décisions de
rédaction sur du matériau acquis.

---

## Piste 4 — Le plancher fixe FTMO

**Espérance : faible. Effort : trois à cinq jours. P(succès) ≈ 10-15 %.**

C'est la seule direction que `RESULTS_BARRIER.md` laisse explicitement ouverte :
sur géométrie à plancher fixe, le bras régime apparié allonge la survie médiane
de **+36 sessions**. Cohérent avec AlphaSimplex — sur un plancher fixe, survivre
longtemps *est* l'objectif.

Trois raisons de la classer bas, dont deux sont décisives.

- La cible de vol sans paramètre allonge la même survie de **+60 sessions**. Le
  régime perd déjà, sur la métrique qui compte, dans la mesure qui a produit le
  +36.
- Le chantier d'admission marginale du dépôt privé voisin a fait tourner le contrôle
  nul que cette piste exigerait : **une série à moyenne exactement nulle et à
  persistance de 10 jours achète +14,95 points de Δp(pass)**. Un filtre de régime
  a une persistance de l'ordre de l'année. Il achèterait Δp(pass) par le canal
  que le contrôle nul montre gratuit. Le même chantier mesure que l'autocorrélation
  du candidat corrèle à **+0,51** avec Δp(pass), quand la couverture de calendrier
  — le canal attendu — corrèle à **−0,06**.
- La géométrie passe par un adaptateur local et non par le simulateur audité.

Ce qu'il faudrait pour rouvrir : un vrai simulateur FTMO, et un placebo décalé
qui préserve moyenne, distribution et autocorrélation. Sans les deux, ne pas y
toucher. **Ne la garder que si le simulateur devient disponible pour une autre
raison.**

---

## Piste 5 — Dispersion des marchés émergents

**Espérance : faible. Effort : deux à trois semaines. P(succès) ≈ 15 %.**

La seule réouverture légitime de H2, puisque le kill est mécanique après 1999
mais que la matière première existe ailleurs. Vérifié disponible et gratuit :
12 devises depuis 2006, 12 indices depuis 2012, 6 ETF obligataires locaux depuis
2013. Le rapport de dispersion EM/G10 passe de 0,7× en 2006-2012 à **1,5× en
2020-2026**.

Trois obstacles, et le troisième est structurel.

- **Puissance.** Vingt ans au mieux, souvent treize. Pour résoudre un Sharpe de
  0,70 contre zéro à 80 % de puissance il faut 19,9 ans (§ piste 1). Sur les
  indices et l'obligataire local on ne les a pas.
- **Coûts.** Cinq à dix fois le G10. Sur un livre dont §3.5 montre que le point
  mort est à 15,7 bp, c'est probablement fatal — et ce calcul doit être fait
  **avant** de construire quoi que ce soit, comme le veut la leçon n° 3 du
  programme.
- **Contrôles de capitaux invisibles dans un backtest au comptant.** Un backtest
  ne voit pas qu'on n'a pas pu sortir.

À ne rouvrir que si les pistes 1 à 3 sont closes et qu'il reste du temps. Le
calcul de coût de dix lignes doit précéder tout le reste ; s'il tue la piste,
c'est deux heures pour économiser trois semaines.

---

## Ce qui est mort, et qu'il ne faut pas rouvrir

Repris de `CONCLUSIONS.md` §2, en une liste, pour que ce soit citable.

| famille | nature du kill | statut |
|---|---|---|
| Régime comme signal de timing (couche 3, 11 règles) | mesure, règle d'arrêt déclenchée | **fermé** |
| Régime comme conditionneur de livre (T1, T3) | mesure + placebo, 0/5 plis | **fermé** |
| Atténuateur de Carver, par instrument et au niveau du livre | battu par un témoin sans paramètre | **fermé** |
| Régime dans la barrière propfirm (Tradeify) | nul sur les nuls ET sous-puissant | **fermé** |
| Compteur de facteurs effectifs | le signal apparent était la troncature | **fermé** |
| Sélection d'actifs par exposant de Hurst | 3 rejets sur 58 pour 2,9 attendus | **fermé** |
| Conditionnement au choc de persistance | produit du bêta, alpha −2,19 %/an | **fermé** |
| Prime de retournement court terme | mécanique, gross ≈ 0 | **fermé définitivement** |
| Transversal macro G10 après mars 1999 | mécanique, dispersion = 0,00 % | **fermé définitivement** |
| H1 momentum macro absolu, un pays | carte des signes au 7ᵉ percentile | **fermé** |
| H3 surprises macro à noyau exponentiel | sous le seuil de Šidák, placebo bat le signal | **fermé** |
| Δp(pass) comme critère d'admission | achetable sans edge : +14,95 pts sur une série à moyenne nulle | **fermé, des deux côtés** |

Deux clarifications sur ce qui n'est **pas** mort, pour éviter la sur-fermeture.

- **Le classifieur n'est pas mort.** Il marche, il est validé contre une
  référence externe, et il reste l'objet du mémoire. Ce qui est mort est sa
  monétisation.
- **H2 avant mars 1999 n'est pas mort** — c'est un défaut de puissance (9 ans,
  4 cellules), donc rouvrable par la politique du programme. Mais neuf ans ne
  permettent pas de résoudre un Sharpe de 0,70, donc rouvrable ne veut pas dire
  prometteur.


---

## Ordre d'exécution recommandé

```
semaine 1   piste 1, jours 1-5      + piste 2 en parallèle (rédaction)
semaine 2   piste 1, verdict        + piste 3 (figures et mise en évidence)
ensuite     piste 5 seulement si le calcul de coût de dix lignes survit
            piste 4 seulement si un vrai simulateur FTMO apparaît
```

Les pistes 2 et 3 sont quasi certaines et peu coûteuses ; elles ne doivent pas
attendre le verdict de la piste 1. La piste 1 est la seule qui puisse produire
un résultat déployable, et elle produit un résultat publiable même en échouant.
