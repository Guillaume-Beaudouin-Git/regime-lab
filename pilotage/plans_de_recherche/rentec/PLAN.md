# Renaissance Technologies — HMM sur micro-régimes
## Plan de recherche pré-enregistrable · 22 septembre 2026 · état DRAFT

Chaque chiffre de ce document vient d'une commande exécutée sur cette machine. Les scripts
sont à côté de ce fichier. Aucun rendement de stratégie n'a été calculé : l'angle est écrit
avant que la donnée ne soit interrogée sur l'hypothèse, et les mesures ci-dessous ne portent
que sur des comptages, des dates et des coûts.

---

## a) TEST D'ADMISSION

Le programme a réfuté six dispositifs. Tous partagent quatre étroitesses. Voici où celui-ci
diffère, chiffré.

### Axe 2 — LA CONSTANTE DE TEMPS · franchi de quatre à cinq ordres de grandeur

Référence du programme : A′ sparse jump change d'état **13 fois en 6 377 séances**, soit
**0,514 transition par an**.

Mesuré ici. HMM gaussien K=3, diagonal, six variables de microstructure, barres d'une minute.
Comptage de transitions uniquement (`measure_timeconstant.py`, `measure_stability_cost.py`) :

| support | fenêtre | n barres | transitions | /an | /séance | durée médiane d'état |
|---|---|--:|--:|--:|--:|--:|
| EURUSD | 2023-2025 | 699 551 | 63 662 | **21 258** | 84,4 | 2 min |
| US500 | 2023-2025 | 951 801 | 18 194 | **6 081** | 24,1 | 12 min |
| XAUUSD | 2023-2025 | 1 050 850 | 16 449 | **5 497** | 21,8 | 16 min |
| GBPUSD | 2023-2025 | 883 616 | 14 600 | **4 879** | 19,4 | 11 min |
| BTC | 2024 | 525 600 | 18 947 | **18 960** | — | 6 min |
| SOL | 2024 | 525 600 | 36 183 | **36 208** | — | 4 min |
| BTC | 2022 | 524 160 | 3 342 | **3 353** | — | 84 min |

Le plus lent de ces objets, GBPUSD à 4 879/an, tourne à **9 492 fois** la cadence de A′.
Le plus rapide, SOL 2024, à **70 443 fois**. Et l'agrégat le plus grossier de l'arbre — l'état
modal de la journée, qui est l'objet du niveau C — donne encore **86 à 147 transitions par an**,
soit **167 à 286 fois** A′. Même le niveau le plus lent de cet arbre ne refait pas H-b.

⚠ Déclaré contre moi-même : la constante de temps **n'est pas stable**. Même spécification,
même graine, BTC 2020 et 2022 donnent 3 721 et 3 353 transitions/an, BTC 2024 et 2025 en donnent
17 340 et 18 960. Un facteur cinq. Deux ajustements ont imprimé un avertissement de
non-convergence. C'est pourquoi la porte G0 ci-dessous existe et précède toute dépense d'α.

### Axe 3 — L'USAGE · l'axe réellement neuf

Les six réfutations testaient le **dimensionnement** au niveau du portefeuille et
l'**interrupteur ON/OFF**. Cet arbre teste l'**EXÉCUTION** (niveaux A et B) puis la
**CONSTRUCTION de portefeuille** (niveau C). Aucun des trois ne produit de signal directionnel.

La conséquence méthodologique est le cœur du plan : **la grandeur de sortie n'est pas un Sharpe,
c'est un coût en points de base.** Voir le (c), où l'on montre que le même échantillon est
sous-puissant en Sharpe et puissant en coût.

### Axe 1 — LA VARIABLE LATENTE · franchi sur la cible, partiellement sur l'état

Deux questions distinctes, et il faut les séparer.

**L'état.** Il n'est pas réductible à un quantile de volatilité, mais il n'en est pas
orthogonal. Accord entre l'état HMM et un tercile de volatilité réalisée de même fréquence :
0,570 (EURUSD), 0,743 (GBPUSD), 0,799 (XAUUSD), 0,816 (US500) ; hasard 0,333. À K=4 sur BTC
1-min, 0,558 contre un hasard de 0,250. Il y a donc de l'information non-volatilité, mais
l'aveu est net : une partie de l'état **est** de la volatilité.

**La cible.** C'est là que l'axe est franchi sans ambiguïté. La quantité prédite n'est plus un
rendement mais le **spread effectif coté**, et un quintile de volatilité réalisée de même
fréquence explique de sa variance en log :

| | EURUSD | GBPUSD | XAUUSD | US500 | GER40 |
|---|--:|--:|--:|--:|--:|
| quintile de volatilité | **−0,003** | +0,014 | +0,009 | **−0,016** | +0,186 |

Le témoin d'une ligne qui a battu les six dispositifs est **inopérant sur cette cible**.
Corrélation mesurée entre spread et |rendement| à la minute : −0,149 à +0,096 selon
l'instrument. La croyance « le spread s'écarte quand ça bouge » est fausse sur ce flux.

⚠ Ce n'est pas une bonne nouvelle nette. Le témoin d'une ligne change simplement de forme :
ici c'est **l'heure de la journée plus un niveau lent**. Voir le (f), tueur n°1.

### Axe 4 — L'OBJET CONDITIONNÉ

Les six conditionnaient **un seul livre**, du TSMOM 12-1 sur 46 instruments. Ici l'objet
conditionné est la **liste d'ordres obligatoire d'un livre déjà gelé** et son coût — pas un
livre, pas un signal, pas un rendement.

### Verdict d'admission

**ADMISSIBLE sur les quatre axes**, dont deux sans réserve (2 et 3), un sur la cible et non sur
l'état (1), un par construction (4). Le plan ne serait pas un septième dispositif tombant sur le
même placebo, parce que le placebo en question — le quantile de volatilité — a été mesuré
inopérant sur la grandeur visée.

---

## b) LA DONNÉE

### Ce qui existe sur disque, vérifié en l'ouvrant

**Le patrimoine à la minute, 27 instruments, avec une colonne de spread flottant réel.**
Dans le dépôt privé voisin, hors de ce dépôt : un fichier `*_M1.csv` par instrument.

C'est la découverte qui décide du véhicule. Ces fichiers portent
`timestamp_utc,open,high,low,close,tick_volume,spread,real_volume` et la colonne `spread` est
un vrai spread flottant : 89 à 1 017 valeurs distinctes selon l'instrument. **129 237 891 barres
d'une minute**, couverture `spread>0` de 0,61 à 1,00.

| instrument | lignes | début | fin | spr>0 | spread médian 2021+ (bp) |
|---|--:|---|---|--:|--:|
| XAGUSD | 7 290 493 | 2003-08-08 | 2026-06-05 | 1,00 | — |
| AUDJPY | 7 264 227 | 2007-01-01 | 2026-06-05 | 1,00 | — |
| EURJPY | 7 263 644 | 2007-01-01 | 2026-06-05 | 1,00 | — |
| GBPUSD | 7 260 253 | 2007-01-01 | 2026-06-05 | 0,90 | **0,36** |
| EURUSD | 7 259 299 | 2007-01-01 | 2026-06-05 | **0,61** | **0,19** |
| EURGBP | 7 259 257 | 2007-01-01 | 2026-06-05 | 1,00 | — |
| USDJPY | 7 256 114 | 2007-01-01 | 2026-06-05 | 0,99 | **0,26** |
| USDCAD | 7 255 578 | 2007-01-01 | 2026-06-05 | 1,00 | — |
| NZDUSD | 7 250 320 | 2007-01-01 | 2026-06-05 | 1,00 | — |
| USDCHF | 7 242 013 | 2007-01-01 | 2026-06-05 | 1,00 | — |
| AUDUSD | 7 125 523 | 2007-01-01 | 2026-06-05 | 1,00 | — |
| XAUUSD | 6 886 976 | 2007-01-01 | 2026-06-05 | 1,00 | **0,63** |
| UKOIL | 4 579 620 | 2010-12-02 | 2026-06-05 | 1,00 | — |
| USOIL | 4 434 667 | 2013-01-01 | 2026-06-05 | 1,00 | **6,53** |
| COPPER | 4 325 524 | 2012-03-02 | 2026-06-07 | 1,00 | **10,19** |
| US100 | 4 081 738 | 2013-01-01 | 2026-06-05 | 1,00 | **0,73** |
| US500 | 4 034 473 | 2012-01-16 | 2026-06-05 | 1,00 | **0,87** |
| US30 | 3 954 770 | 2013-09-30 | 2026-06-05 | 0,99 | — |
| GER40 | 3 840 291 | 2013-09-30 | 2026-06-05 | 1,00 | **0,74** |
| NGAS | 3 725 891 | 2012-09-02 | 2026-06-05 | 1,00 | **203,76** |
| DIESEL, SOYBEAN, COTTON, COFFEE, SUGAR, COCOA, OJUICE | 0,5 à 2,4 M chacun | 2017-10 à 2017-12 | 2026-06-05 | 1,00 | non mesuré |

Manifeste d'intégrité déjà audité sur disque, dans ce même dépôt privé : densité de barres,
fenêtres d'anomalie classées, corrélations de recouvrement 0,99997 à 1,0, porte de couture PASS
sur US500/US100/GER40 après la réparation du 2026-06-10.

**Le patrimoine crypto, 17 instruments, sans colonne de coût.**
Même dépôt privé — `timestamp_utc,open,high,low,close,tick_volume,spread`.
BTC 4 418 408 lignes du 2018-01-01 au 2026-06-01 ; ETH 4 417 842 ; SOL 3 051 844 depuis 2020-08-11 ;
XMR s'arrête au **2024-02-20** (délisté), les seize autres vont au 2026-06-01. Intersection des
seize : 2020-10-15 → 2026-06-01, **5,63 ans**.
**La colonne `spread` y vaut identiquement zéro** : maximum 0, fraction non nulle 0,0000 sur
4,4 M de lignes BTC. Vérifié, pas supposé. **La crypto ne peut donc pas être le terrain d'une
étude de coût** — c'est la deuxième raison, après la puissance, de déplacer le véhicule.

**Ce que je croyais pouvoir utiliser et qui n'est pas là.** Le fichier BTCUSD à spread
flottant dont se sert une étude antérieure du dépôt privé voisin est lu depuis un bac à sable
distant qui n'existe plus. La seule version locale n'a pas de spread. **Les chiffres du gate-0 crypto (médiane 3,4 bp, p90 11, p99 27) ne sont pas
reproductibles localement.** C'est ma première erreur potentielle, attrapée avant de la promettre.

**Le reste, vérifié :** une série NQ quotidienne back-ajustée du même dépôt privé,
4 112 lignes quotidiennes du 2010-06-07 au 2026-05-29, colonnes
`open/high/low/close/raw_close/volume/is_roll`. Utile comme instrument de contrôle back-ajusté,
hors du périmètre à la minute.

### Travail intraday déjà fait, à ne pas refaire

- Une exploration antérieure du dépôt privé voisin (juin 2026) — gate-0 spread contre mouvement, **4 axes sur 5
  tués**, le 5e (réversion de cascade de liquidations) réel mais décroissant et négatif en 2025.
  Le fait décisif pour moi : « le tueur crypto-intraday est l'absence de mouvement capturable,
  pas le spread ». **Toute branche directionnelle à la minute est déjà morte.** C'est précisément
  pourquoi cet arbre est un arbre de coût.
- Une réplication antérieure, même dépôt — réversion intraday à 15 min sur 4
  indices et 3 devises, **falsifiée brut et bilatérale**, n_tests ≈ 13. Leçon L33b : il faut plus
  de 3 bp de brut par trade ; le meilleur observé était 0,2 bp.
- Un post-mortem antérieur, même dépôt — **le tueur administratif**. L'espace
  prix/régime BTC est déclaré épuisé : `n_trials` honnête ≈ 50, E[maxSR] ≈ 1,8-2,3, DSR des deux
  specs promues 0,00 et 0,02-0,075. « Tout nouveau candidat de cet espace naît mort. »
  Conséquence pour ce plan, écrite maintenant : **aucun niveau de cet arbre ne peut être converti
  en revendication de Sharpe sur BTC sans hériter de ces 50 essais.**

### Ce qui manque, et ce qui est récupérable — testé, pas supposé

Nous n'avons **pas** de carnet d'ordres niveau 2. Réponse franche à la question posée :

Un HMM sur des variables de microstructure dérivées de l'OHLCV est un substitut défendable **de
l'état de coût**, et **pas** de l'état du carnet. La différence est nette. Le déséquilibre du
carnet exige du flux signé, que l'OHLCV ne contient pas : le signe des rendements n'est pas le
signe du flux. En revanche le patrimoine porte la grandeur qui nous intéresse **directement** —
le spread coté flottant — ce qui rend tout estimateur de substitution inutile ici.

⚠ Et j'ai vérifié que l'estimateur de substitution usuel ne tiendrait pas : le Corwin-Schultz à
la minute sur BTC est **non positif sur 42,4 % des barres**. Écrêté à zéro, son p25 intra-heure
vaut 0 et la dispersion « moyenne − p25 » se confond avec « moyenne − 0 ». Ma première mesure de
puissance a produit un prix oracle et un prix réaliste **identiques** (1,02 et 1,02 bp sur BTC) :
c'était le symptôme, je l'ai rattrapé et corrigé avec l'amplitude de Parkinson, strictement
positive. Corwin-Schultz a été conçu pour des barres quotidiennes ; à la minute il est dégénéré.

**Source externe testée par requête réelle** (`curl -sI`, en-tête seulement, aucun téléchargement) :

| ressource | HTTP | taille | verdict |
|---|--:|--:|---|
| `data.binance.vision` futures um daily **bookDepth** BTCUSDT 2023-06-01 | **200** | 337 443 o | disponible |
| idem 2024-06-03 | **200** | 465 194 o | disponible |
| idem 2026-06-01 | **200** | 520 918 o | disponible |
| idem 2022-06-01 | 404 | — | **avant l'historique** |
| idem 2021-01-04 | 404 | — | avant l'historique |
| futures um daily **aggTrades** BTCUSDT 2025-01-02 | 200 | 18 947 850 o | disponible mais lourd |
| spot monthly aggTrades BTCUSDT 2025-01 | 200 | 756 026 457 o | **hors de question** |
| futures um monthly bookTicker 2025-01 | 404 | — | pas à ce chemin |
| spot daily bookTicker 2025-01-02 | 404 | — | pas à ce chemin |

Donc : une profondeur de carnet réelle **est** disponible gratuitement, légère (≈ 170 Mo par
instrument-an), et commence entre juin 2022 et juin 2023 — j'ai testé cinq dates, pas la borne
exacte. Mais elle ne couvre que les perpétuels Binance, c'est-à-dire le terrain qui n'a **pas**
de colonne de coût sur disque. Elle est donc **nommée comme branche séparée avec son propre
pré-enregistrement**, et n'entre pas dans le budget α de cet arbre. Trois ans et trois mois
donnent un MDE de Sharpe d'environ 1,6 : cette branche serait à son tour à mener en espace de
coût, pas en espace de rendement.

---

## c) LA PUISSANCE A PRIORI

Machinerie validée d'abord. Avec `regime_lab/analysis/bootstrap.stationary_indices`, bloc moyen
63, α 0,05, puissance 0,80, je retrouve la référence publiée du programme :

| échantillon | MDE sur une différence de Sharpe |
|---|--:|
| 23,2 ans (8 468 séances) | **0,637** — le programme publie 0,639 |
| 8,4 ans (3 066 séances, 24/7) | 0,960 |
| 5,6 ans (2 043 séances, 24/7) | **1,252** |

**C'est le résultat à coût nul demandé par la consigne, et il est décisif pour le cadrage.**
Sur la fenêtre intraday disponible, rien en dessous de Sharpe ≈ 1,0 n'est décidable. Une étude
de cet angle menée en espace de rendement serait **sous-puissante avant d'avoir commencé**, et
le déclarer vaut mieux que le découvrir.

En espace de coût, sur le même échantillon, le rapport s'inverse. Amplitude intra-heure de la
difficulté d'exécution, mesurée sur 47 450 fenêtres de 60 minutes et 1 978 jours, bootstrap
stationnaire à bloc moyen 21 jours :

| support | jours | prix moyen (bp) | SE bootstrap | **MDE** | prix / MDE |
|---|--:|--:|--:|--:|--:|
| BTC | 1 978 | 1,024 | 0,0826 | **0,231 bp** | **4,4×** |
| ETH | 1 978 | 1,336 | 0,1011 | **0,283 bp** | **4,7×** |
| SOL | 1 978 | 2,282 | 0,1423 | **0,399 bp** | **5,7×** |
| DOGE | 1 978 | 2,194 | 0,1996 | **0,559 bp** | **3,9×** |

Corrigé du biais Corwin-Schultz signalé au (b), avec l'amplitude de Parkinson strictement
positive : dispersion intra-heure « moyenne − p25 » de **2,68 bp sur BTC (41 % du niveau de
6,95 bp)** et **5,02 bp sur SOL (34 % de 14,83 bp)**. L'ordre de grandeur du prix tient, la
mesure dégénérée ne le fondait pas.

**Conclusion de puissance, écrite avant toute mesure d'arme : le même échantillon est
sous-puissant d'un facteur ~2 en espace de Sharpe et sur-puissant d'un facteur 4 à 6 en espace
de coût.** C'est la justification entière du choix d'usage. Le MDE opératoire de l'arbre sera
recalculé sur l'échantillon gelé du patrimoine avant le premier bras, et figé à ce moment-là ;
l'ordre de grandeur attendu est **0,2 à 0,6 bp par côté**.

### Traduction en Sharpe, pour que personne ne s'illusionne

La rotation mesurée du livre de tendance est de **58,92 ×/an** (TACHES.md §A3). Une économie de
0,5 bp par côté vaut 0,5 × 2 × 58,92 = **59 bp/an**, sur un livre dont le rendement en excess à
10 % de volatilité et Sharpe 0,359 fait 3,59 %/an — soit **+0,059 de Sharpe**.

Écrit à l'avance : 0,059 est **au-dessus** de la résolution propre de la paire K4 (0,043) et
**très en dessous** du MDE de dimensionnement K3 (0,246). Un gain d'exécution de cette taille est
**visible en points de base et invisible en Sharpe**. Quiconque demandera après coup « et ça fait
quoi sur le Sharpe ? » devra lire cette ligne : la réponse est « moins que ce que le Sharpe peut
voir », et ce n'est pas un échec, c'est le cadrage.

---

## d) LES COÛTS ET LE POINT MORT

Ici le coût n'est pas une charge déduite d'une hypothèse : **le coût est l'hypothèse**. Mesuré
sur le patrimoine, 2021 et après, spread coté converti en points de base :

| instrument | p10 | médiane | p75 | p90 | p99 | moyenne | corr(spread,|rdt|) |
|---|--:|--:|--:|--:|--:|--:|--:|
| EURUSD | 0,09 | **0,19** | 0,43 | 1,04 | 4,50 | 0,51 | −0,010 |
| USDJPY | 0,09 | **0,26** | 0,46 | 0,99 | 4,61 | 0,51 | −0,006 |
| GBPUSD | 0,09 | **0,36** | 0,53 | 1,33 | 5,95 | 0,61 | −0,004 |
| XAUUSD | 0,36 | **0,63** | 0,82 | 1,02 | 1,51 | 0,66 | +0,073 |
| US100 | 0,40 | **0,73** | 1,13 | 1,42 | 1,91 | 0,84 | −0,090 |
| GER40 | 0,45 | **0,74** | 1,90 | 4,01 | 5,99 | 1,56 | −0,149 |
| US500 | 0,63 | **0,87** | 1,17 | 1,47 | 1,87 | 0,97 | +0,096 |
| USOIL | 5,23 | **6,53** | 7,52 | 8,33 | 11,53 | 6,75 | −0,081 |
| COPPER | 7,16 | **10,19** | 11,72 | 12,88 | 14,93 | 10,11 | −0,063 |
| NGAS | 99,11 | **203,76** | 327,42 | 415,05 | 596,37 | 237,89 | −0,155 |

**Exclusion ex ante, écrite maintenant :** tout instrument dont le spread médian 2021+ dépasse
**5 bp** est hors de l'échantillon gelé. Cela sort NGAS, COPPER, USOIL, et par extension les sept
produits agricoles que je n'ai pas mesurés — ils seront mesurés à la congélation et sortis sous
la même règle, pas après avoir vu un résultat.

Barème de référence du programme, aller-retour : futures ~0,01 %, actions 0,05-0,10 %, FX 0,15 %,
crypto 0,17 %. Note importante : le spread mesuré ici (0,19 à 0,87 bp) est **très en dessous** du
barème FX de 15 bp du programme, parce que le barème couvre un aller-retour complet avec
commission et glissement, et que la colonne MT5 ne porte que la fourchette cotée d'une classe de
liquidité. **Je la traite comme un mandataire de classe LP, exactement comme l'avait
fait l'étude crypto antérieure.** Elle sous-estime le coût vrai ; elle sous-estime donc probablement aussi sa
dispersion, ce qui va contre l'hypothèse et non pour elle.

**Point mort, écrit avant de tester.** L'hypothèse est morte si, sur le livre groupé, l'économie
nette de déficit d'exécution face au témoin P2 est inférieure au MDE gelé, soit un ordre de
grandeur de **0,25 bp par côté**. En dessous, le résultat est UNDERPOWERED et **jamais** un PASS —
règle dure du programme, appliquée sans discussion. Et si l'économie de spread est positive mais
que le terme de dérive de prix la mange, le verdict est également KILL : c'est le déficit **net**
qui compte, pas la fourchette économisée.

---

## e) LE PLACEBO APPARIÉ

Obligatoire, et il y en a trois, appariés sur trois dimensions différentes. T3 a montré qu'un
gain apparent peut passer entièrement par le bêta ; la barrière a montré que p(pass) s'achète
avec de l'exposition à 0,2785 par unité. Ici l'équivalent est : un gain apparent de coût peut
passer entièrement par l'heure de la journée, ou par la dérive du prix.

- **P1 — minute aléatoire.** Apparié sur *l'occasion* : même instrument, même fenêtre, même
  nombre d'exécutions, minute tirée uniformément, 200 tirages par fenêtre. Il isole le fait
  d'avoir choisi.
- **P2 — table heure-du-jour × niveau lent.** Apparié sur *l'information triviale*. C'est le
  témoin d'une ligne, l'analogue du quantile de volatilité qui a battu les six. Part mesurée de
  la variance du log-spread : heure du jour seule 0,007 (XAUUSD) à 0,625 (GER40) ; heure du jour
  plus niveau lent 0,407 (US500) à 0,802 (XAUUSD). **C'est le vrai adversaire.**
- **P3 — quintile de volatilité réalisée.** Apparié sur *l'ancien vainqueur*. Mesuré à −0,016 à
  +0,186 de part de variance, donc attendu inopérant — inclus précisément pour que sa défaite
  soit consignée plutôt que supposée.

Contrôle structurel supplémentaire au niveau B, et il n'est pas facultatif : **l'heure moyenne
pondérée d'exécution du bras doit rester à ±2 minutes de celle du TWAP.** C'est ce qui garantit
mécaniquement que le gain de B ne passe pas par une exposition à la dérive. C'est l'équivalent
direct du test d'exposition appariée de T3, appliqué au temps au lieu du bêta.

---

## f) LES TUEURS CONNUS, ET LE SENS DANS LEQUEL ILS POUSSENT

1. **Le témoin d'une ligne a simplement changé de forme.** Heure du jour plus niveau lent explique
   0,41 à 0,80 de la variance du log-spread. Sur GER40, l'heure du jour seule fait 0,625. → contre
   A et contre B. C'est le tueur principal, et il est exactement de la même famille que celui qui
   a eu les six précédents.
2. **Le spread est fortement autocorrélé.** Lag 1 : 0,800 à 0,947. Lag 60 min : 0,625 à 0,896.
   Lag une journée : 0,426 à 0,879. Un AR(1) sur le spread lui-même pourrait battre le HMM
   d'emblée. → contre les trois niveaux. Intégré à P2 comme composante « niveau lent » plutôt que
   comme quatrième témoin, pour ne pas gonfler le compte d'essais.
3. **La sélection adverse.** Une minute bon marché est bon marché parce qu'il ne s'y passe rien,
   et « il ne se passe rien » inclut « le prix est déjà parti ». → contre A spécifiquement, et
   c'est la raison pour laquelle B existe.
4. **La colonne `spread` est un mandataire de classe LP**, pas un coût exécutable : pas de
   commission, pas de glissement, une seule contrepartie. → sous-estime le niveau, donc
   probablement la dispersion, donc contre l'hypothèse.
5. **corr(spread, |rendement|) ≈ 0**, de −0,149 à +0,096. Le mécanisme que tout le monde
   supposerait — les fourchettes s'écartent dans la volatilité — est faux sur ce flux. → deux sens :
   cela protège du placebo de volatilité, et cela retire la source de dispersion la plus évidente.
   À noter que l'étude crypto antérieure avait mesuré l'inverse sur BTC MT5 (p90 24,7 bp sur les minutes
   de cascade) : le fait dépend du flux, et je ne généralise pas.
6. **La colonne `spread` d'EURUSD n'est peuplée qu'à 0,61**, et tombe à 0,31-0,37 sur 2021-2023.
   Une absence non aléatoire est possible. → EURUSD passe en strate séparée, ou sort.
7. **N = 27 < 30.** La règle dure ne mord que sur le transversal et cette étude ne l'est pas,
   mais je le déclare : **le niveau C ne pourra pas être converti en revendication transversale.**
   Côté crypto, 16 instruments utilisables après la mort de XMR, même remarque.
8. **L'espace prix/régime BTC est épuisé** : n_trials ≈ 50, E[maxSR] ≈ 1,8-2,3, DSR 0,00 et 0,02.
   → tout glissement de cet arbre vers une revendication directionnelle sur BTC naît mort.
9. **Le mouvement capturable à la minute n'existe pas**, mesuré par l'étude crypto antérieure : ratio
   |mouvement|/spread ≈ 0,9-1,1 à chaque heure UTC, pic médian FOMC/CPI de 6 bp et non 2-5 %.
   → interdit toute branche directionnelle. Va dans le sens de l'arbre tel qu'il est construit.
10. **L'ajustement HMM lui-même est instable** : facteur cinq sur le nombre de transitions entre
    BTC 2022 et BTC 2024, à spécification et graine identiques, et deux avertissements de
    non-convergence. Le classifieur quotidien avait une stabilité de partition à 73-137× le nul ;
    celui-ci ne l'a **pas** démontrée. → c'est la porte G0, et elle passe avant tout le reste.

---

## g) L'ARBRE D'ESCALADE, L'EFFORT, ET LES PROBABILITÉS

**Un arbre déclaré à l'avance n'est pas du p-hacking ; c'est un arbre non déclaré qui l'est.**
Les trois niveaux, leurs critères de falsification et la correction pour tests multiples sont
figés avant la première mesure d'arme, dans `PRESPEC_RENTEC.md`.

### G0 — porte de stabilité · 1,5 jour · P(passe) = 0,55

Avant de dépenser un seul essai : réestimer le HMM sur cinq plis walk-forward, mesurer l'accord
de partition entre plis contre un nul de permutation par blocs, exiger un accord supérieur à
**10× le nul** (contre les 73-137× du classifieur quotidien — une barre volontairement basse, et
déclarée basse). Si G0 échoue, l'arbre s'arrête ici, **zéro essai dépensé**, et le résultat
publié est « un état de microstructure à 5 000-21 000 transitions/an n'est pas une partition
stable sur ce patrimoine ».

### Niveau A — quelle minute · 4 essais · 3 jours · P(passe | G0) = 0,18

Le livre doit exécuter une fois dans une fenêtre de 60 minutes imposée. Le bras choisit la minute
dont l'état filtré en t−1 porte le coût conditionnel attendu le plus bas. Sortie : déficit
d'exécution **net** en bp contre le prix d'arrivée — fourchette payée **moins** dérive subie.

**Critère de falsification A.** PASS exige, simultanément : économie nette contre P1 **et** contre
P2 au-dessus du MDE gelé, p bootstrap par blocs sous le seuil de Holm, et ≥ 4 plis positifs sur 5.
Si l'économie contre P2 est sous le MDE → A tombe, motif *redondance avec l'information triviale*.
Si l'économie de fourchette est positive mais le déficit net nul ou négatif → A tombe, motif
*sélection adverse*.

### Niveau B — quel rythme · 4 essais · 2 jours · P(passe | A tombe) = 0,22

**Pourquoi B échappe à la raison pour laquelle A est tombée.** Si A tombe par sélection adverse,
c'est parce qu'attendre expose à la dérive. B n'attend jamais : l'ordre est découpé en douze
tranches de 5 minutes, toutes exécutées, et l'état ne module que le **poids** de chaque tranche,
avec un plancher à 1/(3M) et des poids sommant à 1. L'exposition à la dérive est celle du TWAP
**par construction**, et c'est vérifiable : heure moyenne pondérée d'exécution à ±2 min du TWAP.
Le canal par lequel A échouait est donc fermé mécaniquement, pas espéré fermé.

**Critère de falsification B.** Même barre de MDE contre le TWAP uniforme et contre la même
programmation dérivée de P2. Si le contrôle d'heure moyenne dévie de plus de 2 minutes, B est
disqualifié quel qu'en soit le résultat. Si B tombe, motif attendu : *prévision trop bruitée à
l'échelle de la tranche de 5 minutes*.

### Niveau C — quelle bande · 4 essais · 2,5 jours · P(passe | B tombe) = 0,15

**Pourquoi C échappe à la raison pour laquelle B est tombée.** Si B tombe par bruit de prévision
à 5 minutes, C cesse de prévoir à 5 minutes : il agrège l'état à la journée (état modal, mesuré
à 86-147 transitions/an, soit encore 167-286× A′) et change d'usage — **construction de
portefeuille**. Une bande de non-négociation dont la largeur vaut k × E[coût | état] : le livre
ne se rééquilibre que si l'écart de poids dépasse le coût attendu de l'état du jour.

Sortie : **rotation annuelle à erreur de suivi constante**, et non un Sharpe. La rotation de
référence est 58,92 ×/an. L'erreur de suivi au livre non bandé doit rester dans ±0,5 % annualisé ;
au-delà, C est disqualifié.

**Critère de falsification C.** Réduction de rotation sous le MDE à erreur de suivi tenue, ou
erreur de suivi dépassée → C tombe. Motif attendu : *l'agrégat quotidien n'est qu'un niveau lent
déguisé*, ce que P2 dira.

### Niveau D — la fermeture

Si G0 passe et que A, B et C tombent tous les trois, la déclaration de fermeture est :
**« un état latent estimé sur OHLCV et fourchette cotée, à 5 000-21 000 transitions par an,
n'apporte rien au coût d'exécution au-delà d'une table heure-du-jour et d'un niveau lent, sous
trois usages et trois constantes de temps. »** Étroite, mesurée, et symétrique de celle que le
programme tient déjà à l'autre bout de l'axe.

### Budget, correction, effort

| | essais | jours | P(passe) |
|---|--:|--:|--:|
| G0 | 0 | 1,5 | 0,55 |
| A | 4 | 3,0 | 0,18 |
| B | 4 | 2,0 | 0,22 |
| C | 4 | 2,5 | 0,15 |
| rédaction, pré-enregistrement, contre-expertise | — | 2,0 | — |
| **total** | **12** | **11,0** | |

**Correction pour tests multiples, chiffrée à l'avance et couvrant l'arbre entier.**
Holm-Bonferroni sur les **12** essais, α global 0,05 : le plus petit p doit passer sous
**0,05/12 = 0,00417**, soit z = 2,87. L'espérance du maximum de 12 normales indépendantes vaut
**≈ 2,16 écarts-types** : tout effet apparent sous 2,16 σ est au plafond de bruit et se lit comme
tel. Tous les p sont des p de bootstrap stationnaire par blocs (Politis-Romano, bloc moyen 21
jours), **jamais iid**. Le registre global du programme est incrémenté de 12, quoi qu'il arrive.

**Règle de process héritée du postmortem voisin, et je m'y soumets explicitement :** il est
interdit de re-pré-enregistrer un nouveau nul pour remettre le compteur à zéro. Si l'arbre tombe,
il tombe ; une quatrième idée sur le même patrimoine hérite des 12.

**P(au moins un niveau passe) = 0,55 × [1 − 0,82 × 0,78 × 0,85] ≈ 0,25.**
P(fermeture publiée) ≈ 0,75, dont ≈ 0,45 dès G0 ou A.

---

## h) CE QU'ON APPREND MÊME SI TOUT L'ARBRE TOMBE

1. **La phrase étroite du programme gagne sa deuxième borne.** Aujourd'hui on sait qu'un état
   ordonné par la volatilité à 0,5 transition par an n'apporte rien au dimensionnement. Demain on
   saurait aussi qu'un état de microstructure à 5 000-21 000 transitions par an n'apporte rien au
   coût d'exécution. Deux extrémités de l'axe des constantes de temps, séparées par quatre ordres
   de grandeur, tombant pour des raisons mesurément différentes. Cela transforme un soupçon en
   **encadrement**, et c'est exactement ce que la généralisation non soutenue réclamait.
2. **Un modèle de coût d'exécution calibré pour les 27 instruments** — heure du jour × niveau lent
   × instrument, sur 129 M de barres — dont les deux dépôts ont besoin et qu'aucun ne possède.
   Même nature de sous-produit que le livre de tendance audité : c'est le dénominateur qui
   manquait. Il survit à l'échec de l'hypothèse.
3. **Le fait mesuré corr(spread, |rendement|) ≈ 0 sur le patrimoine FX/indices**, contre le p90 à
   24,7 bp mesuré par l'étude crypto antérieure sur BTC. La prémisse « les fourchettes s'écartent dans
   la volatilité » dépend du flux. Utile directement à la leçon L33b, qui est écrite comme si elle
   était universelle.
4. **La règle de conception qui sort de la puissance** : 5,6 ans donnent un MDE de 1,252 en Sharpe
   et un rapport prix/MDE de 3,9 à 5,7 en coût. Quand l'échantillon est court, déplacer la question
   des rendements vers les coûts multiplie la résolution par quatre à six. C'est réutilisable bien
   au-delà de cet angle.
5. **La non-reproductibilité du gate-0 crypto est consignée** : les chiffres de spread de
   `exploration_2026-06.md` dépendent d'un fichier qui n'est plus sur disque. À signaler à l'autre
   session, parce que c'est la cinquième instance du mode de défaillance maison — un résultat
   publié dont le code ne peut plus tourner.

---

## Fichiers produits

- `PLAN.md` — ce document
- `PRESPEC_RENTEC.md` — le pré-enregistrement, en anglais, état DRAFT, couvrant l'arbre entier
- `measure_timeconstant.py` · `timeconstant_btc.csv` — constante de temps par barre et par K
- `measure_stability_cost.py` · `stability.csv` — stabilité inter-années et inter-instruments
- `measure_power.py` — MDE en espace de coût et référence en espace de Sharpe

Rien n'a été écrit dans les dépôts. `scripts/run_phase2.py` n'a pas été lancé. Aucune donnée de
marché n'a été téléchargée : les seules requêtes réseau sont neuf `curl -sI` d'en-têtes.
