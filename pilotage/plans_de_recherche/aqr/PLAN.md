# AQR — timing factoriel conditionnel : plan de recherche pré-enregistrable

État : proposition. Rien n'est verrouillé tant que `PRESPEC_AQR.md` n'est pas passé de
DRAFT à LOCKED et journalisé dans `docs/PROTOCOL_FREEZE.md`.

Toutes les mesures citées ici sortent d'une commande exécutée pendant la rédaction.
Les scripts sont à côté de ce fichier : `measure_states.py`, `measure_states2.py`,
`power_aqr.py`, `design_numbers.py`, `ic_power.py`, `holdout_power.py`. Aucun rendement
de stratégie n'a été calculé — voir la section « ce que je me suis interdit ».

---

## a) Test d'admission

L'approche doit différer des six réfutations sur au moins un des quatre axes. Elle
diffère sur **deux axes pleinement (3 et 4), un partiellement (1), et elle ÉCHOUE sur
l'axe 2 pour la variable que la description d'AQR met en avant** — ce dernier point est
mesuré et il commande tout le reste du plan.

### Axe 4 — l'objet conditionné : **DIFFÉRENT**, et c'est l'axe principal

Les six réfutations conditionnaient un objet unique : un livre TSMOM 12-1 sur 46
instruments, ou un 60/40. Ici l'objet est une **bibliothèque de trois livres
transversaux** construits sur le même univers de 49 secteurs :

| livre | signal | direction attendue sous stress, d'après la littérature |
|---|---|---|
| B1 | momentum transversal 12-1 | vulnérable (les krachs de momentum arrivent aux rebonds) |
| B2 | bêta faible, type BAB | défensif (prime de contrainte de levier) |
| B3 | retournement long terme 60-12 mois, proxy de valeur | signe ambigu |

Chiffrage : 1 livre → 3 livres, sur N = 49 instruments contre 46, mais surtout **trois
signaux dont deux ont des comportements de régime opposés**. Conditionner un panier où
le momentum s'effondre là où le bêta faible paie n'est pas la même question que
conditionner un livre directionnel unique : dans le second cas, le régime ne peut que
changer la taille ; dans le premier, il peut changer la **composition** à taille
constante.

### Axe 3 — l'usage : **DIFFÉRENT**

Les six réfutations testaient le dimensionnement (multiplicateur de taille, covariable
de budget de risque) et le timing directionnel (interrupteur ON/OFF). Ici l'usage est la
**sélection relative entre signaux, à exposition brute constante** :

```
w_k,t = 1/3 + δ · s_k · clip(z_t, −2, +2)/2      Σ_k s_k = 0,  Σ_k |s_k| = 2,  δ = 0,25
```

Σ_k w_k = 1 à toutes les dates, et chaque livre est lui-même neutre en dollars et brut
unitaire. Conséquence structurelle : **l'exposition brute et le bêta marché sont
appariés par construction**, pas seulement testés après coup. C'est la fermeture directe
du canal qui a tué T3 (l'alpha passait par le bêta, 0ᵉ percentile) et du canal qui a tué
Δp(pass) (la p(pass) s'achète avec de l'exposition, 0,2785 par unité). Un dispositif qui
ne peut pas acheter d'exposition ne peut pas acheter le résultat par ce chemin-là.

### Axe 1 — la variable latente : **PARTIELLEMENT DIFFÉRENT**, mesuré

Ce n'est plus une partition ordonnée par la volatilité d'entraînement. C'est l'**impulsion
de conditions financières** : la variation à 13 semaines du NFCI, z-scorée sur historique
expansif. Corrélation mesurée avec la volatilité réalisée à 63 jours (le z-score en
cache, `features.parquet`), sur le calendrier des panneaux :

| variable d'état candidate | ρ avec la vol réalisée | demi-vie AR(1) |
|---|---|---|
| **NFCI, variation 13 semaines** | **+0,323** | 281 séances |
| NFCI, niveau | +0,639 | — (gonflée par le report hebdomadaire) |
| spread de qualité BAA−AAA | +0,514 | 407 séances |
| `xs_dispersion_szbm` (au disque) | +0,687 | 215 séances |
| `xs_dispersion_ind` (au disque) | +0,733 | 254 séances |
| écart de valorisation actualisé par les prix | **+0,106** | 456 séances |
| *placebo* `vol_rv_63` | +1,000 | 423 séances |

Deux lectures, toutes deux utiles.

**La dispersion transversale n'est pas une variable orthogonale à la volatilité.** Les
deux features de dispersion déjà au disque lisent ρ = +0,69 et +0,73 contre la
volatilité réalisée. Un dispositif bâti dessus serait, pour l'essentiel, un septième
passage sur le même placebo. Elles sortent du plan.

**L'écart de valorisation est bien orthogonal — ρ = +0,106 — et c'est la seule
candidate qui le soit.** Mais il échoue sur l'axe 2, ci-dessous, et il sort donc lui
aussi de l'arbre confirmatoire.

### Axe 2 — la constante de temps : **mesurée, et elle tranche**

Chaque candidate binarisée sur sa propre médiane expansive, sur le calendrier des
panneaux Ken French :

| état | transitions/an brut | +21 séances de séjour min. | +63 séances | épisodes complets sur la fenêtre |
|---|---|---|---|---|
| A′ sparse jump *(l'objet réfuté)* | 0,53 | — | — | **6,5 en 24,4 ans** |
| **NFCI, variation 13 semaines** | **2,42** | **2,42** | **2,42** | **43,9 en 36,3 ans** |
| NFCI, niveau | 1,27 | 1,21 | 0,95 | 23,2 |
| spread BAA−AAA | 4,51 | 1,56 | 1,16 | 21,0 |
| `xs_avg_corr` | 3,11 | 1,92 | 1,44 | 26,2 |
| **écart de valorisation, bande ±0,5 z** | **0,12** | — | — | **2,2 en 36,0 ans** |

La variation du NFCI produit **2,42 transitions par an, soit 4,6× A′**, et elle est la
seule candidate dont le compte ne bouge pas quand on impose un séjour minimum de 21 ou
63 séances : elle ne bavarde pas autour de son seuil, elle change vraiment d'état.

**L'écart de valorisation produit 0,12 changement de signe par an — 4,3 en 36 ans.**
C'est cinq fois plus lent qu'A′ sparse jump. Sa demi-vie AR(1) est de 456 séances, soit
1,81 an. Utilisé en variable de timing, il ferait de H-b une deuxième fois, en pire.

**Ce que ça répond à la question du brief.** Le programme avait noté que « ce qui prédit
la moyenne chez les praticiens, ce sont les *variations* macro, pas la *classification*
en états ». Sur nos données, avant tout rendement : la **variation** du NFCI est 1,9×
plus rapide que son **niveau** (2,42 contre 1,27 transitions/an) et deux fois moins
corrélée à la volatilité réalisée (+0,323 contre +0,639). Sur les deux axes qui comptent,
la variation domine la classification. C'est ce qui fixe la variable d'état, et c'est
une mesure, pas une préférence.

### Verdict d'admission

**ADMISSIBLE**, sur les axes 4 (objet : bibliothèque de trois signaux transversaux
contre un livre de tendance unique) et 3 (usage : sélection relative à brut constant
contre dimensionnement). L'axe 1 est partiel : ρ = +0,323 contre la volatilité n'est pas
l'orthogonalité, c'est environ 10 % de variance partagée. L'axe 2 est amélioré d'un
facteur 4,6 en transitions et 6,8 en épisodes.

**Et l'axe que la description d'AQR met en avant — la dispersion des valorisations —
est écarté par la mesure.** Je ne l'escamote pas : la firme qu'on m'a donnée fait du
timing factoriel par la valorisation, et je viens de mesurer que sur notre échantillon
cette variable contient 2,2 épisodes complets. Elle reste dans le plan comme **mesure
descriptive à coût nul** (section h), jamais comme test confirmatoire.

---

## b) La donnée

### Ce qui existe sur disque, ouvert et vérifié

Tous les panneaux sont en format long `(series_id, period, available_at, value)`,
rendements quotidiens **en pourcent**, `available_at = period` (choix documenté et
argumenté dans `regime_lab/data/sources/kenfrench.py`).

| fichier | séries | observations | début | fin | années |
|---|---|---|---|---|---|
| `data/raw/panels/industry_49.parquet` | **49 secteurs** | 451 388 lignes, 9 212 séances | 1990-01-02 | 2026-07-31 | 36,57 |
| `data/raw/panels/size_bm_25.parquet` | 25 portefeuilles taille/valeur | 230 300 lignes, 9 212 séances | 1990-01-02 | 2026-07-31 | 36,57 |
| `data/raw/panels/factors_5.parquet` | **6** : `ff_mkt-rf, smb, hml, rmw, cma, rf` | 55 272 lignes, 9 212 séances | 1990-01-02 | 2026-07-31 | 36,57 |
| `data/raw/macro/fin_nfci.parquet` | NFCI hebdomadaire | 1 913 | 1990-01-05 | 2026-08-28 | délai de publication 7 jours, déjà encodé |
| `data/raw/macro/fin_baa_spread.parquet` | spread BAA quotidien | 9 171 | 1990-01-02 | 2026-09-08 | délai 1 jour |
| `data/raw/macro/fin_aaa_spread.parquet` | spread AAA quotidien | 9 171 | 1990-01-02 | 2026-09-08 | délai 1 jour |
| `data/cache/features.parquet` | 50 features, **z-scores expansifs écrêtés à ±5** | 9 572 × 50 | 1990-01-01 | 2026-09-08 | — |
| `data/cache/states.parquet` | 5 partitions, dont A′ | 9 007 × 5 | 1992-03-02 | 2026-09-08 | A′ non nul sur 6 377 séances |

Aucune valeur manquante dans les trois panneaux sur 1990-2026 : 49 secteurs sur 49
peuplés à **toutes** les 9 212 séances.

### Trois corrections au descriptif de la tâche, vérifiées en ouvrant les fichiers

1. **`factors_5.parquet` ne contient pas le momentum.** La tâche annonce « 5 facteurs
   Fama-French plus momentum » ; le fichier contient six séries, et la sixième est
   `ff_rf`, le taux sans risque, pas `ff_mom`. `regime_lab/features/crosssection.py:78`
   boucle d'ailleurs sur `("ff_smb", "ff_hml", "ff_mom")` avec un garde `if name in
   factors`, donc la feature `xs_ff_mom_63` n'a jamais été construite et personne ne
   s'en est aperçu. Sans conséquence pour le plan : le momentum de tête est transversal
   sur les 49 secteurs, il ne vient pas d'un fichier de facteur.
2. **La dispersion de valorisation n'est PAS dérivable de `size_bm_25.parquet`.** Ce
   panneau ne contient que des rendements. Les ratios BE/ME vivent dans un autre fichier
   de la bibliothèque Ken French, qui n'est pas au disque.
3. **Le BE/ME publié est mensuel en apparence et annuel en substance.** Le fichier
   `25_Portfolios_5x5_CSV.zip` contient bien une section BE/ME de **1 201 lignes
   mensuelles, 1926-07 à 2026-07, pour les 25 portefeuilles**. Mais le numérateur
   BE(exercice t−1)/ME(déc. t−1) est figé par titre pour toute l'année juin-juin ; seule
   la pondération ME(mois) bouge. Mesuré : le |Δlog BE/ME| moyen vaut **0,1939 en
   juillet** (mois de reformation) contre **0,0037 sur les onze autres mois**, un
   rapport de **52,4×**. L'information arrive une fois par an.

### Ce qui manque, et la disponibilité testée et non supposée

Le programme s'est fait avoir deux fois (Stooq derrière un mur anti-bot, codes pays
OCDE). J'ai donc téléchargé chaque source avant de la recommander. Tout est dans
`verif/`.

| source | URL | test | résultat |
|---|---|---|---|
| Momentum FF quotidien | `…ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip` | téléchargé, dézippé | **OK**, 85 202 octets, 26 211 lignes, finit au **20260731** — même date que les panneaux |
| BE/ME des 25 portefeuilles | `…ken.french/ftp/25_Portfolios_5x5_CSV.zip` | téléchargé, section parsée | **OK**, 548 334 octets, 1 201 lignes mensuelles × 25 |
| 49 secteurs quotidiens, historique complet | `…ken.french/ftp/49_Industry_Portfolios_daily_CSV.zip` | téléchargé, parsé avec `_first_table` du dépôt | **OK**, 4 186 243 octets, **26 296 séances, 1926-07-01 → 2026-07-31** |
| NFCI, historique complet | `fred.stlouisfed.org/graph/fredgraph.csv?id=NFCI` | téléchargé | **OK**, sans clé d'API, **2 906 points hebdomadaires depuis 1971-01-08** |
| ANFCI (NFCI ajusté du cycle) | `…fredgraph.csv?id=ANFCI` | téléchargé | **OK**, même profondeur, 1971-01-08 |

Couverture du panneau secteurs sur les fenêtres candidates, comptée :

| fenêtre | séances | mois | secteurs peuplés (moy / min) |
|---|---|---|---|
| 1985-01 → 2026-07 (tampon de formation + échantillon) | 10 475 | 499 | 49,00 / **49** |
| **1990-01 → 2026-07 (échantillon de tête)** | **9 212** | **439** | 49,00 / **49** |
| **1971-04 → 1989-12 (holdout scellé)** | **4 733** | **225** | 49,00 / **49** |
| 1966-01 → 1989-12 (tampon de formation du holdout) | 6 033 | 288 | 48,86 / 48 |
| 1926-07 → 2026-07 (fichier entier) | 26 296 | — | 46,44 / 39 — **inutilisable avant 1966** |

### L'écart de valorisation actualisé par les prix, construit et mesuré

Puisque le BE/ME brut n'apporte qu'un point par an, je l'ai actualisé comme le fait la
littérature (Cohen-Polk-Vuolteenaho, et AQR dans ses propres notes) : on part du BE/ME
de la reformation de juin, on le traîne au quotidien en le divisant par le rendement
cumulé de la jambe depuis la formation. Jambe valeur = moyenne des cinq portefeuilles
HiBM, jambe croissance = les cinq LoBM.

Résultat : **9 065 observations quotidiennes, 1990-08-01 → 2026-07-31, 36,0 ans**,
37 reformations de juin dans la fenêtre, niveau log entre 1,529 et 2,846.

Et le verdict, qui est négatif et qu'il valait mieux obtenir avant le
pré-enregistrement qu'après : demi-vie AR(1) **456 séances (1,81 an)**, **0,12 changement
de signe par an** dans une bande ±0,5 z, **4,3 transitions en 36 ans**. La construction
marche ; la variable est trop lente pour porter un test.

### Ce que l'échantillon de tête sera

- **Formation** : panneau 49 secteurs re-téléchargé depuis **1985-01-01** (tampon de 60
  mois pour B3), écrit sous un nouveau nom, jamais en écrasant le panneau stocké.
- **Évaluation walk-forward** : **1990-01-02 → 2026-07-31, 9 212 séances, 439 mois**,
  cinq plis expansifs.
- **Holdout scellé** : **1971-04-08 → 1989-12-31, 4 733 séances, 225 mois**, formation
  depuis 1966. Ouvert **une seule fois**, à la fin, seulement si un niveau de l'arbre
  passe. C'est 18,7 ans que ce programme n'a jamais touchés, sur aucune des six
  réfutations — le seul holdout réel dont il dispose.

Les rendements des panneaux sont totaux, pas en excess. Les trois livres sont neutres en
dollars et à brut unitaire, donc leur rendement est en excess par construction ; le solde
de trésorerie n'est **pas** crédité au taux sans risque, ce qui est le sens conservateur.
`ff_rf` et `rate_cash_3m` sont au disque si un contrôle l'exige.

---

## c) La puissance a priori

Calculs dans `power_aqr.py`, `design_numbers.py`, `ic_power.py`, `holdout_power.py`.

### Étalonnage contre la référence du programme

La référence citée — 23,2 ans donnent 0,639 de Sharpe résoluble, il faut 19,9 ans pour
résoudre 0,70 — se reproduit **exactement** avec l'erreur standard de Lo,
SE = √((1+S²/2)/T), et (z_{α/2}+z_β) = 2,8016 :

| années | ce que je calcule | ce que le programme publie |
|---|---|---|
| 23,2 | **0,638** | 0,639 |
| 19,9 | **0,701** | 0,70 |

L'étalonnage tient. Tout ce qui suit est sur la même échelle.

### Correction pour tests multiples, sur l'arbre ENTIER

L'arbre déclare **5 tests confirmatoires** : A, B, C-variance, C-Sharpe (condition-
nel à C-variance), et l'ouverture du holdout. Šidák à α = 0,05 :

α₁ = 1 − 0,95^(1/5) = **0,010206**, d'où (z_{α₁/2}+z_β) = **3,4104** contre 2,8016.

### Le canal Sharpe est trop grossier, et je le dis avant de le tester

Plus petit Sharpe *autonome* résoluble pour la surcouche de timing (formule de Lo,
Šidák sur 5 tests) :

| fenêtre | non corrigé | corrigé Šidák |
|---|---|---|
| 1990-01 → 2026-07 (36,6 ans) | 0,490 | **0,615** |
| 1971-04 → 2026-07 (55,3 ans) | 0,391 | 0,485 |

Contre-vérification par bootstrap stationnaire, avec `regime_lab.analysis.power` sur des
séries appariées **synthétiques** (aucun rendement réel lu), blocs moyens de 63 séances :
le MDE sur la *différence* appariée de Sharpe vaut 0,210 à ρ = 0,90, 0,149 à ρ = 0,95,
0,094 à ρ = 0,98 et 0,067 à ρ = 0,99, sur 36,0 ans. Après l'inflation Šidák de **+21,7 %** que
l'arbre à cinq tests impose, seule une corrélation de jambes ≥ 0,98 laisse voir un
incrément de +0,15 : à ρ = 0,98 le MDE apparié corrigé vaut **0,114**, à ρ = 0,95 il
vaut 0,181. Et cette exigence est **invariante d'échelle** : l'ampleur du tilt δ
multiplie l'effet et le MDE dans le même rapport, donc jouer sur δ n'achète aucune
puissance. Ce qui reste est la contrainte nue : **la surcouche doit valoir 0,615 de
Sharpe autonome pour être vue.**

La littérature d'AQR elle-même ne rapporte rien de cette taille pour le timing
factoriel. Le Sharpe 1,2 sur 1970-2016 est le **niveau** d'un portefeuille multi-facteurs
statique — c'est-à-dire notre jambe de contrôle — et non l'**incrément** dû au timing.
Donc : **le critère de tête ne sera pas un Sharpe.**

### Le canal qui a la puissance : le coefficient d'information conditionnel

Une hypothèse de *sélection* est une affirmation sur des IC conditionnels, qui vivent sur
un panneau de N × T et non sur un Sharpe qui ne vit que sur T. Avec N = 49 secteurs,
l'erreur standard d'un IC transversal mensuel vaut 1/√48 = **0,1443**. Sur T mois
répartis entre deux états :

| fenêtre | mois | SE de la différence d'IC (HAC ×1,3) | **MDE, Šidák n=5** |
|---|---|---|---|
| **1990-01 → 2026-07** | **439** | 0,0179 | **0,0611** |
| holdout 1971-04 → 1989-12 | 225 | 0,0250 | 0,0853 |
| les deux fenêtres réunies | 664 | 0,0146 | 0,0497 |

Sous un partage 30/70 des états au lieu de 50/50, le MDE de tête passe de 0,0611 à
0,0666 ; c'est la sensibilité la plus défavorable et c'est elle qui sera pré-enregistrée.

**L'effet visé est-il au-dessus ?** L'IC transversal inconditionnel du momentum sectoriel
est de l'ordre de 0,03 à 0,05. L'hypothèse d'AQR — le momentum s'effondre aux
retournements — suppose un IC qui passe d'environ +0,06 en conditions calmes à environ
−0,02 sous resserrement, soit ΔIC ≈ 0,08. **0,08 > 0,0666**, donc l'effet visé est
au-dessus du MDE, mais d'un facteur 1,2 seulement. Si l'effet réel est 0,03, nous ne le
verrons pas et nous écrirons UNDERPOWERED, jamais PASS.

### Un critère que j'écarte parce qu'il est sous-puissant, et je le dis maintenant

Le test des signes sur les épisodes complets d'état, qui serait le test naturel d'un
effet épisodique, demanderait un taux de réussite de **0,757 sur 44 épisodes** (0,708 sur
67 en fenêtre étendue). Aucune cible plausible n'atteint ça. Il reste **descriptif**, il
n'entre pas dans le décompte des 5 tests.

---

## d) Les coûts, et le niveau auquel l'hypothèse meurt — écrit avant de tester

Barème du programme, aller-retour : futures ~0,01 %, **actions 0,05-0,10 %**, FX 0,15 %,
crypto 0,17 %. Les secteurs Ken French sont des portefeuilles papier sans coût incorporé.

**Coût incrémental de la surcouche.** La rotation à brut constant ne change que les poids
entre livres. Turnover mesuré du tilt continu bâti sur la variation NFCI 13 semaines :
**1,84 unité de |Δw| par an**. Avec δ = 0,25 et Σ|s_k| = 2, le brut incrémental échangé
vaut 0,25 × 2 × 1,84 = **0,92 par an**.

| coût aller-retour | drag annuel | en Sharpe, cible de vol 10 % |
|---|---|---|
| 0,05 % | 0,046 %/an | 0,0046 |
| 0,10 % | 0,092 %/an | 0,0092 |
| 0,20 % | 0,184 %/an | 0,0184 |

**Seuil de mort, écrit avant la mesure :** l'hypothèse conditionnelle est morte quand le
drag incrémental égale le MDE apparié. À δ = 0,25, c'est un aller-retour de **1,24 %**
(contre le MDE apparié de 0,114). À δ = 0,50, c'est **0,62 %**. Le barème actions laisse
donc une marge de **12× à 25×**. **Les coûts ne sont
pas la contrainte qui mord ici — la puissance l'est.** Le dire à l'avance évite
d'attribuer plus tard un échec aux frais.

**Coût du niveau, distinct.** Les trois livres ont leur propre rotation mensuelle, que je
n'ai pas mesurée parce que la mesurer voudrait dire les construire. Elle s'annule dans le
contraste apparié mais pas dans le niveau. Déclaré : le niveau sera publié à 0,05 %,
0,10 % et 0,20 %, et si le composite statique ne franchit pas RESEARCH_PASS, l'incrément
reste un résultat scientifique valide mais **n'est pas un résultat négociable**, et ce
sera écrit ainsi.

---

## e) Les placebos appariés

Obligatoires, et chacun apparie sur une dimension nommée. T3 a montré qu'un gain apparent
peut passer entièrement par le bêta ; la barrière a montré que p(pass) s'achète avec de
l'exposition à 0,2785 par unité.

| | placebo | apparie sur | règle de rejet |
|---|---|---|---|
| **P1** | état nul à transitions appariées : même nombre de transitions et même loi des durées de séjour, phase aléatoire, 1 000 tirages par bootstrap stationnaire de la séquence d'état | **la constante de temps** | le ΔIC réel doit dépasser le 95ᵉ percentile de P1 |
| **P2** | tilt **statique** dont le vecteur de poids égale la moyenne temporelle du tilt conditionnel | **l'exposition moyenne aux facteurs** | si le conditionnel ne bat pas P2, c'est un pari statique déguisé — c'est exactement la critique d'Asness et al. sur le timing par la valorisation, et elle s'applique ici telle quelle |
| **P3** | état = volatilité réalisée de l'indice sectoriel équipondéré sous sa médiane expansive, **même usage, même δ** | **l'usage** | la règle d'une ligne a battu cinq dispositifs ; elle doit être battue, pas seulement le zéro |
| **P4** | contrôle d'exposition : bêta marché et brut réalisés des deux jambes | **l'exposition** | \|Δβ\| ≤ 0,02 et brut identique à 1e−9, sinon le niveau est annulé |
| **P5** | permutation de la carte des signes sur les 3! = 6 ordonnancements | **la carte des signes** | descriptif seulement — avec trois livres ce test est faible, et le dire vaut mieux que de le présenter comme fort |

P4 est en grande partie structurel : le tilt conserve Σw = 1 et chaque livre est neutre
en dollars. On le mesure quand même, parce que « par construction » a déjà été faux dans
ce programme.

---

## f) Les tueurs connus, sans les minimiser, avec le signe dans lequel ils poussent

**1. Le scan praticien liste « le timing factoriel par la valorisation » parmi les choses
à éviter.** Pousse vers le nul. **Je ne prétends pas y échapper : je l'adopte, et je
l'ai mesuré sur nos données avant de pré-enregistrer.** L'écart de valorisation
actualisé par les prix a une demi-vie de 1,81 an et donne 4,3 changements de signe en
36 ans. La pratique a raison, et nous savons maintenant *pourquoi* sur notre échantillon :
il n'y a pas assez d'observations indépendantes pour résoudre quoi que ce soit. La
valorisation sort de l'arbre confirmatoire. Ce qui reste d'AQR dans ce plan, c'est le
*timing factoriel conditionnel* — l'objet et l'usage — et non la variable d'état que la
firme met en avant.

**2. Leçon n°6 : petit univers + correction pour tests multiples = tout échoue, N ≥ 30
minimum.** Pousse vers le nul. Satisfaite en tête : **N = 49**, avec 49 secteurs sur 49
peuplés à toutes les 9 212 séances de 1990-2026 et à toutes les 4 733 séances du holdout.
**Non satisfaite** par la bibliothèque FF-5 + momentum (N = 6), qui est donc déclarée
**sensibilité non confirmatoire** et n'entre pas dans les 5 tests.

**3. Le quantile de volatilité a battu cinq dispositifs.** Pousse vers le nul. Notre
état lit ρ = +0,323 contre la volatilité réalisée — mieux que le niveau NFCI (+0,639) et
bien mieux que les features de dispersion (+0,687 et +0,733), mais ce n'est pas zéro,
c'est environ 10 % de variance partagée. P3 est le juge.

**4. Les krachs de momentum sont concentrés sur très peu de mois** (mars-avril 2009,
2000-2001). Pousse vers un faux positif. Parade déclarée : leave-one-year-out sur le
ΔIC ; si retirer une seule année fait passer le ΔIC sous le MDE, le résultat est déclaré
**fragile** et n'est pas un PASS.

**5. Les portefeuilles Ken French sont des portefeuilles papier.** Pousse vers
l'optimisme sur le niveau. Vendre 25 secteurs à découvert en 1971 n'était pas gratuit et
pas toujours possible. Le holdout 1971-1989 est donc lu comme **confirmation de signe**,
jamais comme mesure de rentabilité.

**6. Les 49 secteurs et les 25 portefeuilles taille/valeur recouvrent les mêmes titres.**
La sensibilité sur le panneau taille/valeur n'est pas une preuve indépendante. Déclaré.

**7. B3 est un proxy de valeur, pas de la valeur.** Le retournement 60-12 mois est le
substitut classique quand le BE/ME manque, mais si B3 ne porte rien, on ne pourra pas
distinguer « la valeur ne répond pas à l'état » de « notre proxy n'est pas la valeur ».
C'est une limite d'interprétation, pas un défaut, et elle est écrite d'avance.

**8. Le NFCI est lui-même construit sur ~105 séries financières, dont des mesures de
volatilité.** ρ = +0,323 le confirme. Variante de robustesse pré-déclarée : **ANFCI**,
la version ajustée du cycle, vérifiée disponible (2 906 points depuis 1971-01-08).

**9. Timing luck.** Le scan praticien mesure plus de 100 bp annualisés pour un
rebalancement à date fixe. Parade dans la construction : étalement sur 5 jours ouvrés.

**10. Le NFCI est publié avec 7 jours de retard.** Déjà encodé dans le parquet
(`available_at − period = 7` partout). L'état lu en T−1 n'utilise que ce qui était publié
en T−1. La règle « signal en T−1, trade en T » s'applique au-dessus de ce décalage, pas
à sa place.

---

## g) L'arbre d'escalade

Trois niveaux et une fermeture, déclarés avant toute mesure. Un arbre déclaré à l'avance
n'est pas du p-hacking ; c'est un arbre non déclaré qui l'est.

### Niveau A — carte des signes déclarée

**Ce qu'on teste.** État = variation NFCI 13 semaines, z-scorée, en T−1. Carte des signes
fixée avant la mesure et issue de la littérature, pas de nos données :
s = (momentum **−1**, bêta faible **+1**, retournement long **0**). Le momentum baisse
quand les conditions se resserrent, parce que les krachs de momentum arrivent aux rebonds
(Daniel-Moskowitz 2016) ; le bêta faible monte, parce que c'est une prime de contrainte
de levier qui paie quand le levier se retire (Frazzini-Pedersen 2014) ; le retournement
long garde un poids constant, parce que nous ne pouvons pas justifier son signe et que
déclarer un signe injustifiable est exactement le mode d'échec de H1.

**Statistique de tête** : ΔIC, la différence entre les IC transversaux mensuels du signal
dans les deux états, HAC lag 6.

**Critère de falsification, chiffré** : A tombe si **l'une** de ces conditions est vraie —
- aucun signal n'atteint \|ΔIC\| ≥ **0,0666** avec le signe déclaré ;
- le ΔIC maximal sur les trois livres ne dépasse pas le **95ᵉ percentile de P1** ;
- moins de **4 plis sur 5** portent le même signe ;
- le ΔIC de P3 (état de volatilité) est **égal ou supérieur** à celui de l'état réel ;
- retirer une seule année fait passer le ΔIC sous 0,0666.

**Critère économique secondaire**, rapporté mais non confirmatoire : l'incrément de
Sharpe apparié du tilt contre le composite statique équipondéré, avec son MDE. **Déclaré
UNDERPOWERED d'avance sauf s'il dépasse 0,114.**

*Effort 4 jours. P(succès) ≈ 12 %.*

### Niveau B — carte des signes estimée, walk-forward

**Pourquoi B échappe à la raison pour laquelle A tombe.** A peut tomber pour trois motifs
distincts : (i) la carte des signes déclarée est fausse ; (ii) l'état ne porte aucune
information de sélection ; (iii) le tilt est trop petit. Le motif (iii) est écarté par
construction — l'exigence de puissance est invariante d'échelle (section c). B traite
(i) : il n'affirme aucun signe, il **estime** s_k par l'IC conditionnel **à l'intérieur
du pli d'entraînement seulement**, et l'applique au pli de test. Si B tombe aussi, la
conclusion n'est plus « nous avions mal deviné les signes », elle est « aucune carte
stable n'existe » — un énoncé strictement plus fort.

**Critère supplémentaire** : le signe de s_k estimé doit s'accorder sur **≥ 4 plis
d'entraînement sur 5**. S'il alterne, on déclare **« aucune carte conditionnelle stable
n'existe »** et on passe à C sans même regarder le test.

*Effort 1,5 jour. P(succès | A tombe) ≈ 10 %.*

### Niveau C — l'état conditionne la covariance entre signaux, pas leurs moyennes

**Pourquoi C échappe à la raison pour laquelle B tombe.** Si A et B tombent, le motif
résiduel est (ii) : l'état ne porte pas la moyenne. Ce serait cohérent avec tout ce que
le programme a mesuré — +0,030 point de R² sur les rendements, t 0,27. Mais le programme
a aussi mesuré que l'état porte **la variance** : +3,93 points de R² incrémental, t
−3,40. C pose donc la seule question que les mesures acquises soutiennent, **à un niveau
d'agrégation jamais testé** : non pas la taille du portefeuille (ce que les six
réfutations testaient) mais la **matrice de covariance entre les trois livres**. Le
momentum et le bêta faible décorrèlent-ils sous resserrement ? C'est la question d'AHL
sur les ruptures de corrélation, posée sur l'objet d'AQR.

**Critère, sur la variance et non sur le Sharpe** : la variance réalisée hors échantillon
du composite à risque égal sous covariance conditionnelle doit battre celle sous une
covariance statique à rétrécissement de Ledoit-Wolf, sur **≥ 4 plis sur 5**, au-delà du
MDE calculé sur la log-variance réalisée. **Test conditionné** : si et seulement si la
variance passe, on demande si le gain se convertit en Sharpe à risque cible constant. Si
la conversion échoue, on publie un résultat de variance et **aucune affirmation de
Sharpe**.

*Effort 2,5 jours. P(variance) ≈ 35 %. P(variance ET conversion) ≈ 10,5 %.*

### Holdout — ouvert une fois

Si un niveau passe : **1971-04 → 1989-12, 225 mois, 4 733 séances**, formation depuis
1966, 49 secteurs peuplés partout. Critère : **accord de signe** du ΔIC et \|ΔIC\| ≥ la
moitié du seuil de tête. Le holdout résout 0,0853 seulement ; ce n'est pas une seconde
chasse au p, c'est une réplication de direction sur 18,7 ans que ce programme n'a jamais
touchés.

### Fermeture, si tout tombe

Déclaration, avec son motif : **« le régime ne porte ni la moyenne ni la covariance entre
signaux d'une bibliothèque transversale actions ; le canal variance est épuisé à tous les
niveaux d'agrégation que notre échantillon résout. »** C'est strictement plus fort que
l'énoncé actuel du programme, qui ne porte que sur un livre de tendance et un 60/40.

### Décompte et correction

5 tests confirmatoires : A, B, C-variance, C-Sharpe, holdout. Šidák α₁ = **0,010206**,
(z+z_β) = **3,4104**. La multiplicité *à l'intérieur* d'un niveau (trois livres) est
absorbée par la statistique du maximum contre la distribution de P1, pas par un terme de
Šidák supplémentaire — déclaré, parce que compter deux fois la même multiplicité est
aussi malhonnête que ne pas la compter. Le Sharpe déflaté lit en plus
`regime_lab/analysis/trials.py`, qui journalise **toutes** les évaluations, y compris les
tournes de mise au point.

### Effort total et probabilité

| poste | jours |
|---|---|
| données : re-téléchargement depuis 1985 et 1966, NFCI depuis 1971, PIT, manifestes sha256 | 1,0 |
| construction des trois livres, ciblage de vol quotidien, étalement, coûts, tests unitaires | 2,0 |
| niveau A, trois placebos, cinq plis, bootstrap | 2,0 |
| niveau B | 1,5 |
| niveau C | 1,5 |
| rédaction, amendement `PROTOCOL_FREEZE.md`, document de résultats | 1,0 |
| **total** | **9,0** |

P(au moins un niveau donne un PASS confirmatoire) ≈ **18 %**. Les trois niveaux ne sont
pas indépendants — si l'état ne porte rien, ils tombent ensemble — donc le produit naïf
(29 %) surestime. P(l'arbre produit un résultat publiable, quel que soit le verdict)
≈ **90 %**.

---

## h) Ce qu'on apprend même si tout l'arbre tombe

1. **Un livre de référence transversal actions, audité, en excess, correctement facturé,
   sur 36,6 ans et N = 49.** Le programme vient de se doter du même objet du côté
   tendance (l'étude A4). Il n'en a aucun du côté actions transversales. Tous les
   chiffres actions futurs auront enfin un dénominateur.
2. **Une raison mesurée, et non folklorique, de ne pas faire de timing factoriel par la
   valorisation** : demi-vie 1,81 an, 0,12 changement de signe par an, 4,3 transitions
   en 36 ans, contre un plancher résoluble de 0,615 de Sharpe autonome. Le scan praticien
   l'affirmait ; nous le chiffrons sur nos propres données. C'est un résultat à coût nul,
   déjà obtenu pendant la rédaction de ce plan.
3. **La supériorité mesurée des *variations* macro sur les *classifications* d'état comme
   conditionneurs** : 2,42 contre 1,27 transitions par an, ρ +0,323 contre +0,639 avec la
   volatilité réalisée. C'est transférable aux quatre autres angles de firme, et ça
   confirme sur nos données une remarque que le programme avait notée sans la mesurer.
4. **Le premier test du régime hors du monde tendance/60-40.** Les six réfutations
   portent toutes sur un livre directionnel. Une fermeture au niveau C généraliserait le
   résultat central du programme d'un cran entier.
5. **Le holdout 1971-1989 reste scellé** s'il n'est pas ouvert : 18,7 ans, 225 mois,
   49 secteurs peuplés partout, que le programme peut garder pour une hypothèse future.

---

## Ce que j'ai mesuré et ce que je me suis interdit de mesurer

**Mesuré** : nombres de lignes, de séries et de dates dans les trois panneaux et les onze
fichiers macro ; nombre de transitions par an de huit variables d'état candidates, sous
trois filtres de séjour ; corrélations entre variables d'état et avec la volatilité
réalisée ; demi-vies AR(1) ; turnover d'un tilt continu ; disponibilité et contenu de
cinq sources externes ; couverture du panneau secteurs sur cinq fenêtres ; puissance a
priori sur des séries **synthétiques**.

**Interdit et non fait** : aucun rendement de livre, aucun IC réalisé, aucun Sharpe,
aucune moyenne conditionnelle d'un facteur par état. Les trois livres ne sont pas
construits. Le ΔIC dont ce plan fixe le seuil n'a jamais été calculé. C'est la règle du
programme — l'angle s'écrit avant de toucher la donnée — et la respecter est ce qui
donne leur valeur aux quatre falsifications acquises.

## Mes propres erreurs, pendant cette rédaction

1. **J'ai pris le logarithme des features en cache sans vérifier ce qu'elles sont.** Ce
   sont des z-scores expansifs écrêtés à ±5 (`regime_lab/features/standardise.py`), donc
   négatifs la moitié du temps ; `np.log` a renvoyé des NaN et la première table
   d'orthogonalité tournait sur 3 720 observations au lieu de ~9 100. Elle lisait
   ρ = −0,118 pour l'écart de valorisation là où le bon chiffre est **+0,106**, et
   surtout elle lisait ρ = +0,785 pour `vol_rv_63` **contre elle-même**, ce qui devait
   valoir 1,000 et aurait dû m'arrêter sur-le-champ. Corrigé dans `measure_states2.py` ;
   tous les chiffres de ce plan viennent de la seconde passe.
2. **J'ai d'abord cru la section BE/ME annuelle.** Elle est mensuelle — 1 201 lignes — mais
   la variation mensuelle n'est que de la repondération : 52,4× moins de mouvement hors
   du mois de reformation. La conclusion « l'information arrive une fois par an » tient,
   pour une raison différente de celle que j'avais supposée, et c'est la mesure qui a
   tranché, pas l'intuition.
3. **J'ai nommé un script `numbers.py`**, qui masque le module standard `numbers` et a
   cassé l'import de numpy. Renommé `design_numbers.py`.
