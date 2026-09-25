# Avancement du projet — état au 25 septembre 2026

Ce document sert à partir du même point : **ce qui est fait, où sont les données, ce qui
reste et par quoi commencer.** Chaque chiffre renvoie à un document ou à un script du
dépôt. Le détail complet des tâches est dans `pilotage/feuille_de_route/TACHES.md`.

---

## 0. En une minute

**La question du projet.** Un modèle de régimes de marché appris par machine apporte-t-il
une information exploitable au-delà d'une simple mesure de volatilité ? Et si oui,
résiste-t-elle aux coûts, aux contraintes de données en temps réel et à la correction
pour tests multiples ?

**La réponse, établie.**
1. **Le classifieur marche comme classifieur.** 93,2 % d'exactitude équilibrée contre
   les récessions NBER, kappa 0,53, sur 6 377 jours hors échantillon. Cinq méthodes
   concordent. ⚠ **Mais sur deux récessions seulement.** Réestimée sur 1926-2026 avec
   les 30 variables disponibles depuis 1926, la méthode ne reconnaît que 6 récessions sur
   14 (57,5 %), pas mieux qu'une règle de volatilité (`docs/RESULTS_LONGHIST.md`).
2. **Il porte de la variance, pas de la moyenne.** Il ajoute +3,93 points de R² sur la
   volatilité future (t −3,40), contre +0,030 sur les rendements futurs (t 0,27). C'est
   donc un signal de **dimensionnement**, pas de **timing**. ⚠ **Le VIX le savait
   déjà** : au-delà du VIX, l'état n'ajoute que +0,20 point, non significatif.
3. **Aucun des sept dispositifs testés pour le monétiser ne bat une règle d'une ligne**
   (volatilité réalisée sous sa médiane). Les huit études des 23 et 24 septembre (§1)
   n'ont trouvé aucun usage utile non plus, et en expliquent la raison : le modèle entre
   en stress tard et **y reste pendant les reprises**. Le chiffre qui explique tout : l'état change
   **13 fois en 6 377 séances**, soit treize décisions en 24 ans.
4. **Quatre hypothèses dérivées ont été falsifiées** : H1, H2, H3 et la prime de
   retournement.

**Le livrable du cours.** C'est un projet de cours de M2, **pas un mémoire**. Il se
conclut par une **présentation d'une dizaine de minutes** des résultats, au format d'une
présentation en entreprise.

**L'objectif de fond, au-delà du cours.** Construire des **stratégies de trading
algorithmique dépendantes du régime**, une fois cette parenthèse macro refermée. On
cherche un résultat intéressant et prometteur. L'ambition est un Sharpe **net, en excès
du cash et hors échantillon, de l'ordre de 1 à 2**, à **coûts institutionnels**. Le seuil
de recherche du programme ne bouge pas : RESEARCH_PASS, soit un Sharpe supérieur à 0,7,
une perte maximale moins profonde que −25 % et plus de 50 % des plis de walk-forward
positifs.

**Où on en est par rapport à cet objectif.** Le meilleur livre mesuré ici fait +0,36 de
Sharpe net en excès. C'est une tendance sur 46 instruments, déjà facturée à des coûts de
futures institutionnels (environ 1 bp). Le régime n'y ajoute rien au-delà de la
volatilité. Viser 1 à 2 demande donc un autre objet que les sept dispositifs testés. Un
régime qui **arbitre entre plusieurs signaux** a été mesuré le 23/09 (plan Two Sigma) :
**il ne transfère pas**, voir §1. Reste un axe jamais testé : un régime qui entre dans la
**construction du portefeuille** (plan Bridgewater, en cours depuis le 24/09 dans la
session du compte B). **La piste la plus prometteuse du programme n'est pas un régime** :
un livre de 9 stratégies, sans filtre, fait 1,52 de Sharpe brut sur 2005-2026 et 1,91 sur
2016-2026 (`docs/RESULTS_BUDGET_RISQUE.md`). C'est un plafond : sans coût, avec des
stratégies choisies en connaissant ces années. À remesurer à coûts institutionnels et
sur les seules dates postérieures à la conception de chaque stratégie.

**L'hypothèse de coûts pour la suite est institutionnelle.** Le barème de tête du
programme, en aller-retour, est déjà de ce type : 1 bp pour les futures d'indices, de
taux et de change, 1,5 bp pour les matières premières, 5 bp pour les actions US et 7,5 bp
pour un ETF sans future équivalent. Les colonnes prudente (×2) et de stress (×4 à ×5)
restent rapportées à côté, mais ne décident pas. Les barèmes déjà verrouillés dans les
pré-enregistrements ne changent pas.

---

## 1. Ce qui est fait

### L'étude principale — racine du dépôt

| bloc | verdict | document |
|---|---|---|
| Cadrage pré-enregistré, gelé par empreinte SHA-256 et étiquette `charter-v2` | gelé | `docs/CHARTER.html`, `docs/PROTOCOL_FREEZE.md` |
| Couche 1 — l'étiquetage est fiable | ✅ 93,2 % / kappa 0,53 | `docs/RESULTS_CLASSIFIER.md` |
| Couche 2 — il sépare la variance, et seulement elle | ✅ +3,93 pts R² vol, t −3,40 | `docs/RESULTS_FINAL.md` |
| Couche 3 — règle de timing contre règle de taille | ❌ aucune règle ne bat le témoin d'une ligne | `docs/RESULTS_FINAL.md`, `RESULTS_T2.md` |
| Contrôles T1, T3, T5, les cinq plis | publiés, ils affaiblissent l'étude | `docs/RESULTS_FALSIFICATION.md`, `RESULTS_FOLDS.md` |
| Décomposition de la valeur | le ciblage de vol vaut +0,44 %, le régime −0,04 % | `docs/RESULTS_DECOMPOSITION.md` |
| Réplication Shu (2024) | partielle, une ligne retirée | `docs/REPLICATION_SHU2024.md` |
| Barrière propfirm | nul sur les nuls | `docs/RESULTS_BARRIER.md`, `NOTE_BARRIER_FR.md` |
| Véhicule de tendance, 46 instruments | **KILL** : Sharpe net en excess +0,359 | `docs/RESULTS_TREND_VEHICLE.md` |
| AHL niveau A — sélection de vitesse | **réfuté sur le signe**, Δ −0,068 contre un seuil de 2,12 | `docs/RESULTS_AHL_LEVEL_A.md` |
| AHL niveau B — rupture de corrélation | **indécidable**, non lu, aucun essai dépensé (22/09 au soir) | `docs/RESULTS_AHL_LEVEL_B.md` |

Les 12 points d'hygiène sont faits. 122 tests passent, `ruff` est propre, et le registre
d'essais compte 84 configurations distinctes (`data/trials.parquet`).

### Les chantiers dérivés — `chantiers/`

| chantier | question | verdict | document |
|---|---|---|---|
| `macro-momentum` H1 | le momentum macro absolu prédit-il la direction ? | falsifiée, Sharpe −0,38 | `chantiers/macro-momentum/docs/RESULTS_H1.md` |
| `macro-momentum` H2 | le transversal entre 19 pays ? | falsifiée, 0 cellule sur 9 ; la dispersion des taux courts de la zone euro est **nulle depuis mars 1999** | `RESULTS_H2.md`, `RESULT_DISPERSION.md`, `NOTE_DISPERSION_FR.md` |
| `macro-momentum` H3 | les surprises macro, à la Dedale ? | falsifiée, le placebo bat le signal | `RESULTS_H3.md` |
| `reversal-lab` | le retour à la moyenne court terme est-il exploitable ? | prime passée de 3,64 de Sharpe (années 90) à −0,18 (depuis 2020) | `chantiers/reversal-lab/docs/RESULTS.md` |

### Le 22 septembre au soir

- **Le niveau B d'AHL est clos.** Le critère a été commité avant le calcul du MDE
  (`9602526`). Le MDE est sorti à 49,6 % de l'erreur moyenne contre une limite de 25 %,
  donc la question est indécidable à l'échelle du livre. Le verdict ne dépend pas du
  type de rendement (52,2 % en rendements logarithmiques).
- **Les trois dépôts ne font plus qu'un**, avec leur historique : les identifiants de
  commit cités comme preuves de pré-enregistrement existent toujours. L'environnement
  Python est unique, les 122 tests passent, et la mesure du niveau B se reproduit à
  l'identique après la fusion.
- **120 fichiers de mesure ont été sauvés de `/private/tmp`**, que le redémarrage du Mac
  aurait effacés : les scripts des cinq plans de recherche, ceux des quatre
  contre-expertises A4, les données SPF et l'historique Ken French depuis 1926.

### Le 23 septembre

- **Nature du projet précisée** : un projet de cours de M2 qui se termine par une
  présentation de 10 minutes, pas un mémoire. L'objectif de fond est écrit en §0.
- **La phase 1 du plan Two Sigma est faite** par une autre session (`22abbe0`,
  `3b64ab2`, 262 tests). L'outillage est en place : 10 signaux sur les 49 secteurs US,
  le classifieur de contexte K-means, un placebo apparié exact et le protocole de test.
  **Aucun niveau de l'arbre n'a été lu et aucun essai n'a été dépensé.**
- **L'arbre Two Sigma est lu en entier et fermé** (`docs/RESULTS_TWOSIGMA.md`, commits
  `42011e9` à `e97501d`). Un classifieur de contexte (K-means sur 20 variables,
  orthogonalisé contre la volatilité, 12,5 changements d'état par an hors échantillon)
  sert à choisir entre 10 signaux sur les 49 secteurs US, sur 20 ans de plis de test
  (2006-2026) :
  - **niveau A (la moyenne) : FAIL.** Le sélecteur fait **moins bien** que le mélange
    équipondéré qu'il incline : −0,110 de Sharpe (−0,370 contre −0,260, nets de 5 bp). Le
    profil d'un état ne se transmet pas d'un pli à l'autre (R 0,047, p 0,34) ;
  - **niveau B (la variance) : sous-puissant.** La parité de risque par état gagne
    +0,064, contre une barre de 0,338. La covariance par état prévoit **moins bien** que
    la covariance unique ;
  - **niveau C (les coûts) : non montré.** La règle de cadence ne trouve aucun état où
    trader moins : économie nulle, sur la vraie partition comme sur les 1 000 placebos ;
  - Holm ne rejette rien. Les 14 sensibilités vont dans le même sens, avec une seule
    inversion de signe minuscule (A à K = 6 : +0,009). 20 essais sont journalisés, soit
    exactement les 20 déclarés ;
  - un témoin de volatilité d'une ligne fait **au moins aussi bien** que le contexte sur
    chaque canal. C'est un diagnostic, cohérent avec tout le programme : le contenu du
    régime, c'est la volatilité.
  - Deux événements sont consignés dans `docs/PROTOCOL_FREEZE.md`. Le premier est un
    amendement avant toute lecture : le placebo de C-1 valait exactement 0, donc sa
    « puissance » ne mesurait rien, et une économie minimale S\* est désormais exigée.
    Le second est un incident de registre à la lecture de A, terminée depuis son artefact
    sans aucun recalcul.
- **Le verrou Two Sigma est commité** (`docs/PRESPEC_TWOSIGMA.md`). Guillaume l'a approuvé
  le 23/09, puis deux relecteurs indépendants l'ont examiné : un second validateur sur le
  fond, un auditeur avant le commit. Leurs corrections sont les lignes 28 à 36 du §12.0 :
  entre autres, un succès est rétrogradé s'il ne tient pas sans la série NFCI, qui est
  révisée. Les deux ont confirmé le texte final.
- **Premier test « stratégie seule contre couplée au régime »** : le rebond obligataire de
  fin de mois (`docs/RESULTS_B1_COUPLAGE.md`). Il est **insensible au régime** : Sharpe 0,79
  seul, 0,82 avec réduction en stress (sous-puissant), 0,75 avec bascule vers la
  tendance (pas utile). Il est positif en 2008, 2020 et 2022. Sa pire perte (2025-2026)
  survient alors que le régime est calme depuis 2021.
- **Comparaison « stratégie seule contre couplée au régime » sur 8 stratégies, 24
  couplages** (`docs/RESULTS_COUPLAGE_STRATEGIES.md`, hypothèse zéro coût) : **aucun
  couplage n'est utile**. Deux stratégies sont pénalisées, la tendance crypto et la prime
  overnight Nasdaq, qui gagnent précisément en période de stress. C'est la matière de la
  présentation.
- **Un support de présentation en PDF est prêt** pour construire les diapositives :
  `docs/presentation/presentation_regimes.pdf`, 14 pages et 4 annexes. Il couvre d'où
  viennent les données, comment marchent les cinq modèles, pourquoi le Sparse Jump
  Model, les résultats de chaque modèle et le couplage aux stratégies. Il est régénéré par
  `scripts/build_presentation.py` : frise et états lus dans `data/`, les autres chiffres
  recopiés des `docs/RESULTS_*.md` cités en pied de page. Les Sharpe y sont **sans coût**,
  comme partout dans ce projet. Pour l'oral de 10 minutes, il reste à en tirer 8 à 10
  diapositives.
- **Remesurés à partir de code commité, plusieurs chiffres du brouillon étaient
  optimistes.** L'horloge tourne à 12,6 transitions par an hors échantillon, et non 8,3.
  Elle dépend davantage de la volatilité qu'annoncé : η² de 0,14, jusqu'à 0,6 sur un
  pli, contre 0,05 dans le brouillon. Le plafond de levier mord sur 70 % des séances. Le
  seuil de rentabilité en coûts tombe à 27-33 bp, au lieu de 93. Le test reste assez
  fin : il détecte un écart d'environ 0,26 de Sharpe.
- **Des stratégies de crise, seules et couplées au régime** (`docs/RESULTS_CRISE.md`,
  pré-enregistrement `docs/PRESPEC_CRISE.md` commité avant lecture en `7d5b95c`, une
  lecture, 9 essais `crise_coupling`, coût nul). Quatre objets dont le gain dépend de la
  volatilité ou des crises : vente de variance synthétique sur le S&P 500, vente de
  futures VIX à 1 mois, momentum UMD, indice BXM ; deux couplages (arrêt, moitié) ; plus
  un test de prédiction. **Aucun couplage n'est utile.** Le filtre **change la forme du
  risque sans changer le Sharpe** : la vente de variance évite 2008 (−20,4 % → +1,0 %)
  mais rate le rebond de 2009 (+36,6 % → +7,4 %), Sharpe 1,30 dans les deux cas ; février
  2018 et 2022 lui échappent (état calme). **Le momentum est le seul cas où il aide
  nettement** : 0,52 → 0,69, au 100ᵉ centile du placebo, mais sous le MDE (0,395), et la
  règle de volatilité au 80ᵉ centile fait autant (+0,163 contre +0,166). **Pourquoi : le
  VIX sait déjà ce que sait le modèle.** Au-delà d'un rang de volatilité, l'état ajoute
  4,48 points de R² sur la variance future (t −5,23) ; une fois le VIX dans la
  régression, 0,20 point (t −1,34, seuil 2,77). Données publiques (CBOE, CFE, Ken French)
  dans `data/raw/crisis/`, téléchargées par `scripts/fetch_crisis_data.py`.
- **Actions en calme, valeur refuge en stress** (`docs/RESULTS_REFUGE.md`, protocole dans
  `scripts/run_safe_haven_switch.py`, commité avant lecture en `5872d6b`, 3 essais
  `safe_haven_switch`, coût nul). En stress, on bascule des actions vers les obligations
  longues (TLT), l'or, ou de la volatilité achetée (un straddle synthétique). **Aucune
  bascule n'est utile.** Sharpe 0,64 seul ; 0,57 avec TLT, 0,68 avec l'or (sous-puissant,
  seuil 0,31 ; la règle médiane de volatilité fait 0,77), 0,39 avec la volatilité. La
  raison est descriptive et utile pour l'oral : ciblées en volatilité, les actions
  rapportent autant en stress qu'en calme (0,62 contre 0,64), parce que les états de
  stress contiennent aussi les rebonds (2003, 2009, 2020). TLT sauve 2008 (+0,1 % au lieu
  de −22,9 %) mais rate 2009 (−6,3 % au lieu de +20,3 %).
- **Chacune des stratégies testées, remplacée en stress par une valeur refuge**
  (`docs/RESULTS_REFUGE.md`, 2ᵉ partie ; protocole commité avant lecture dans l'étude du
  dépôt privé voisin ; 27 essais `sjm_haven_switch`, coût nul). 9 stratégies (les 8 des
  couplages + le momentum UMD) × 3 jambes (or, TLT, options = volatilité achetée).
  **Aucune bascule n'est utile.** L'or est la seule jambe qui ne détruit rien (+0,00 à
  +0,04 hors momentum) ; les options détruisent partout sauf sur le momentum (−0,15 à
  −0,43 : +29 à +38 % en 2008, puis −18 à −23 % au rebond de 2009). **Momentum + or :
  0,47 → 0,77**, t +2,68, placebo 100 %, mais sous le MDE (0,62), surtout dû à l'arrêt du
  momentum (0,70), et la règle médiane de volatilité fait autant (0,82). C'est le cas
  retenu pour la présentation.
- **Pistes d'amélioration et fil de la partie 2** (`docs/presentation/PISTES_AMELIORATION.md`,
  regard de conseiller : aucun rendement conditionné lu, aucun essai). Mesures
  descriptives dans `scripts/describe_pistes.py`. **55 % des séances de stress de A′
  tombent après le creux du S&P 500** ; la courbe des futures VIX s'était normalisée 4 à
  9 mois avant la sortie de A′. Six idées classées, dont un SJM réduit sur 1926-2026
  (14 récessions au lieu de 2, MDE du test UMD ≈ 0,15 au lieu de 0,395). Les écarts
  signalés (109 configurations au lieu de 168 ; 12 bascules sous-puissantes au lieu de
  8 ; une réponse fausse sur le VIX dans `PARTIE1_CHEMINEMENT.md` ; le 93 % et le « temps
  réel » sans leurs réserves) sont **corrigés partout** le 23/09 (`527063a`).

### Le 24 septembre : le PDF de la partie 2

- **`docs/presentation/presentation_partie2.pdf`** (16 pages), généré par
  `scripts/build_presentation_partie2.py`. Il montre le filtre appliqué à **trois
  stratégies choisies par Guillaume pour leurs profils opposés** : le momentum actions (le
  filtre aide), le rebond obligataire de fin de mois (il est neutre) et la tendance crypto
  (il pénalise). Il compare **six façons d'utiliser le filtre** : couper, réduire,
  basculer vers la tendance, or, obligations, options. Il ajoute la vérification du
  momentum sur 90 ans, le mécanisme de l'échec et la diversification. Les chiffres viennent
  des lectures uniques, copiées dans `docs/artifacts/partie2/lectures_trois_strategies.txt`.
  Les courbes sont recalculées sur données publiques et vérifiées contre ces lectures
  (le script refuse de dessiner en cas d'écart). La crypto n'apparaît qu'en chiffres
  agrégés.

- **Le deck de l'oral de 10 minutes** (14 diapositives avec la bibliographie, structure fixée
  par Guillaume : intro, données, Markov, Sparse Jump Model et ses papiers, crises, risque
  contre direction, partie 2, stratégies, cinq méthodes avec le ratio de Sharpe, limites,
  conclusion, ouverture ; notes orales minutées) existe en deux styles, comme présentations
  Claude téléchargeables en PowerPoint ou en PDF (privées, à partager depuis la page) :
  style clair https://claude.ai/artifact/Bpg2cRahCG2bsy3YAqbBh6 et style éditorial sombre
  https://claude.ai/artifact/HhFYhrryrYGZqV1s3KqoQJ. Noms : Guillaume Beaudouin et Gabriel
  Golivet. **L'oral a eu lieu le 25 septembre 2026.** Ses chiffres sont ceux des deux PDF. Les écarts de Sharpe du
  tableau des méthodes sont calculés avant arrondi, et identiques dans le deck et dans le
  PDF de la partie 2.

### Le 25 septembre : l'oral, le dossier, et neuf corrections

- **L'oral a eu lieu le 25/09.** Le **dossier remis au professeur** est
  `docs/presentation/dossier_regimes.pdf` (A4, en français), généré par
  `scripts/build_dossier.py` : résumé, démarche, données, modèles, deux résultats,
  application aux trois stratégies, mécanisme de l'échec, limites, conclusion,
  reproductibilité, bibliographie et cinq annexes. Le script recalcule le tableau de
  classification depuis `states.parquet` et refuse de construire si un chiffre publié ne
  se reproduit pas. Version de référence : l'étiquette git `dossier-2026-09-25`.
- **En préparant le dossier, neuf erreurs ont été trouvées et corrigées partout** (PDF,
  decks, note PARTIE1, résultats) :
  1. le rappel 419/435 attribué à A est celui d'A′ ; A fait 431/435 ;
  2. trois écarts au cadrage n'étaient pas déclarés : le sous-ensemble de variables commun
     jamais mis en œuvre, le Jump Model continu jamais estimé, λ recalculé toutes les
     quatre réestimations avec un repli à 20, hors grille, pour A′ depuis avril 2022
     (`docs/PROTOCOL_FREEZE.md`) ;
  3. « 93 % des récessions reconnues » est une exactitude équilibrée (rappel 96,3 %,
     spécificité 90,0 %) ;
  4. **le Sparse Jump Model ne garde pas « au plus 10 » variables** : `max_feats = 10`
     fixe à 10 le nombre *effectif* de variables, mais 17 à 24 gardent un poids non nul
     (`scripts/measure_sjm_sparsity.py`, `docs/artifacts/sjm_sparsity.txt`). La
     correction « au plus 10 » de `adef0cb`/`1fbd6b4` était elle-même fausse ;
  5. le HMM « en retard » était une hypothèse du cadrage que les données démentent : il
     entre en stress au Covid deux semaines avant A′ ; son défaut est la fausse alerte
     (35 % du temps en stress, 19 % de ses séances de stress en récession) ;
  6. « investi 77 % du temps » venait d'une ligne retirée de la réplication de Shu ;
  7. « +0,17 → +0,05 » sur 90 ans comparait deux modèles : même modèle réduit, +0,145 →
     +0,045 ;
  8. au Covid, le modèle sort une première fois du stress le 5 août 2020 (+49 %) et
     définitivement le 5 avril 2021 (+82 %) ; « 97 % » compte le jour du creux ;
  9. « 0 donnée du futur » était faux au sens strict : NFCI et inscriptions au chômage
     sont pris en valeur actuelle décalée, pas en première publication.

### Le 24 septembre : les cinq pistes du conseiller, lancées en parallèle

Guillaume a dit « lance tout ». Cinq études, chacune pré-enregistrée, une lecture chacune,
coût nul. **Aucune ne trouve un usage utile du régime.**
- **Budget de risque multi-stratégies** (`docs/RESULTS_BUDGET_RISQUE.md`, étude dans le
  dépôt privé voisin, 6 essais `sjm_risk_budget`). Le **socle** non conditionné (9
  stratégies, inverse de la volatilité, 10 % de cible) : **Sharpe brut 1,52** sur
  2005-2026 (IC 95 % [1,06 ; 1,96], perte max −24,4 %, 5 plis sur 5), **1,91** sur
  2016-2026 (perte max −10,0 %), 3,29 fois la moyenne des stratégies seules. Le régime en
  budget de risque : 3 bras, aucun utile ; diviser le risque par deux en stress ramène la
  perte max à −17,4 %, les règles de volatilité aussi.
- **Prévision de risque sur 46 marchés** (`docs/RESULTS_RISQUE.md`, 16 essais
  `risque_forecast`). L'état n'améliore la prévision d'**aucun** marché (−0,8 % de QLIKE
  sur les 43 marchés sans VIX propre) ; **le VIX améliore tous les marchés** (+5,6 %).
- **Corrélation actions-obligations** (`docs/RESULTS_COUVERTURE.md`, 6 essais
  `couverture`). Son signe **prédit le risque** d'un portefeuille à risque égal au-delà
  du VIX et de la volatilité (+3,10 points de R², t 5,50, hors échantillon confirmé ;
  placebo pile au seuil, 95,0 %) ; pas sur un 60/40 (placebo 92,8 %). Il **ne choisit pas
  la couverture** (0,78 contre 0,95 pour une poche fixe 50/50).
- **Un SJM par facteur, d'après Shu et Mulvey** (`docs/RESULTS_FACTORSJM.md`, 18 essais,
  1978-2026). Franchit les quatre axes de la règle opposable, et perd quand même : 1,39
  contre 1,51 en tenant les six facteurs. Gagne sur la fenêtre du papier (+0,29), perd
  avant 1990 (−0,63). Réplication partielle, 2 facteurs sur 4.
- **Le SJM réduit sur 1926-2026** (`docs/RESULTS_LONGHIST.md`, réserve AQR ouverte,
  4 essais `longhist_umd`). **FAIL-A** : 6 récessions sur 14, 57,5 %, κ 0,16. Le modèle a
  appris que « stress » veut dire « Grande Dépression » : aucun épisode de stress d'un
  mois hors échantillon de 1937 à 1974. Le momentum coupé en stress : +0,045 sur 90 ans
  (MDE 0,158, contre +0,166 sur 2002-2026), par le dénominateur ; la sortie asymétrique
  (idée 2) n'aide pas (−0,013) ; la règle publiée de Daniel et Moskowitz fait mieux
  (+0,106, perte max −37 % → −27 %), comme témoin sans verdict.

---

## 2. Où est quoi

```
regime-lab/                         UN SEUL dépôt, public
├── README.md                       commencer ici
├── AVANCEMENT.md                   ce fichier
├── CLAUDE.md                       consignes pour les sessions Claude Code
├── regime_lab/  scripts/  tests/   l'étude principale : code, scripts, tests
├── docs/                           cadrage gelé, résultats, registre d'amendements
├── chantiers/
│   ├── macro-momentum/             H1, H2, H3 (son propre code et ses docs)
│   └── reversal-lab/               la prime de retournement
├── pilotage/
│   ├── feuille_de_route/           TACHES, CONCLUSIONS, PISTES, synthèse A4
│   ├── plans_de_recherche/         cinq plans (AHL, AQR, Bridgewater, Rentec, Two Sigma) + ARBITRAGE
│   ├── REPRISE_PROJET.md           ancienne passation (historique ; §8bis encore utile)
│   └── mesures_brutes/             ⚠ NON versionné — voir §3.4
└── data/                           ⚠ NON versionné — voir §3
```

La littérature (4 PDF) reste **hors du dépôt**, pour des raisons de droits d'auteur :
`~/Desktop/M2/Projet_Big_Data_Regimes/3_Litterature/`.

### Mise en route sur une autre machine

```bash
git clone https://github.com/Guillaume-Beaudouin-Git/regime-lab.git
cd regime-lab
uv sync --all-packages --extra dev          # un seul .venv pour les trois études
.venv/bin/python -m pytest -q               # 849 tests au 25/09, ~15 min
```

Les tests passent sans aucune donnée. Les scripts, eux, en ont besoin (§3).

---

## 3. Les données — inventaire exact

**Aucune donnée n'est dans git.** Elles vivent uniquement sur le Mac de Guillaume. Deux
façons de les obtenir :

- **Recommandé : l'archive prête à envoyer**,
  `~/Desktop/M2/Projet_Big_Data_Regimes/donnees_regime-lab_2026-09-22.zip` (36 Mo,
  176 fichiers). Elle contient les données des trois études et celles des plans (§3.4).
  On la décompresse à la racine du dépôt. C'est le seul moyen d'obtenir *les mêmes
  chiffres* : Yahoo révise ses prix ajustés, et un nouveau téléchargement peut déplacer
  les résultats publiés.
- Sinon, les régénérer avec les scripts `fetch*.py`. Il faut alors une clé API FRED
  (gratuite, sur fred.stlouisfed.org), à placer dans `.env` sous `FRED_API_KEY`.

Toutes les données macro sont stockées en **point-in-time** : chaque ligne porte `period`
(la date décrite) et `available_at` (la date à laquelle on pouvait la connaître).

### 3.1 Étude principale — `data/`

**Brut** (`data/raw/`, produit par `scripts/fetch_data.py`, environ 3 min, clé FRED requise)

| fichier | contenu | source | période | lignes |
|---|---|---|---|---|
| `prices/cross_asset.parquet` | 8 séries : `^GSPC` `^RUT` `^IXIC` `^N225` `^TYX` `^TNX` `DX-Y.NYB` `^VIX` | Yahoo Finance, clôture ajustée | 1990-01-01 → 2026-09-08 | 73 716 |
| `prices/fred_daily.parquet` | pétrole WTI (`DCOILWTICO`) | FRED | 1990-01-02 → 2026-09-01 | 9 217 |
| `macro/*.parquet` (11 fichiers) | INDPRO, PAYEMS, CPIAUCSL, UNRATE (premières publications ALFRED) ; ICSA, NFCI (hebdo) ; BAA10Y, AAA10Y, DTB3, T10Y2Y, T10Y3M (quotidien) | FRED / ALFRED | 1990 → 2026-07/09 selon la série | 438 à 9 178 |
| `panels/factors_5.parquet` | 5 facteurs Fama-French + taux sans risque, **sans momentum** | Ken French Data Library | 1990-01-02 → 2026-07-31 | 55 272 |
| `panels/industry_49.parquet` | 49 portefeuilles sectoriels, quotidien | Ken French | 1990-01-02 → 2026-07-31 | 451 388 |
| `panels/size_bm_25.parquet` | 25 portefeuilles taille × valeur | Ken French | 1990-01-02 → 2026-07-31 | 230 300 |
| `references/ref_nber.parquet` | récessions NBER (`USREC`), **validation seulement** | FRED | 1990-01 → 2026-08 | 440 |
| `crisis/vix_futures.parquet` | règlements quotidiens de 274 contrats mensuels VIX, ramenés à une seule échelle | CBOE Futures Exchange (`scripts/fetch_crisis_data.py`) | 2004-03-26 → 2026-09-22 | 47 238 |
| `crisis/french_umd.parquet` | facteur momentum UMD quotidien | Ken French | 1926-11-03 → 2026-07-31 | 26 195 |
| `crisis/cboe_indices.parquet` | indices BXM (depuis 2002-03-22) et PUT (depuis 2007-01-03) | CBOE | → 2026-09-22 | 11 123 |

**Dérivé** (`data/cache/`)

| fichier | contenu | produit par | période |
|---|---|---|---|
| `features.parquet` | 50 variables standardisées (9 572 × 50) | `build_features.py` | 1990-01-01 → 2026-09-08 |
| `states.parquet`, `states_offline.parquet` | états des 5 familles de modèles (9 007 × 5), filtrés et avec recul | `run_phase2.py`, **15-20 min, ne pas relancer à la légère** | 1992-03-02 → 2026-09-08 |
| `trend_universe.parquet` | **46 instruments** : 13 indices actions, 13 matières premières, 11 devises (dont l'indice dollar), 9 ETF obligataires et crédit, en **prix** (6 822 × 46) | `fetch_trend_universe.py` (Yahoo, `auto_adjust=True`) | 2000-07-17 → 2026-09-10 |
| `trend_universe_m1.parquet` | le même, avec 8 devises recalées d'une séance — **panneau de référence** | `apply_m1_realignment.py` | idem |
| `trend_universe_m1_verified.parquet` | sensibilité : 3 devises recalées seulement | idem | idem |
| `shu_*.parquet` | S&P 500 depuis 1970 (données Shiller) pour la réplication Shu | `replicate_shu2024.py` | 1970 → 2026-09-09 |
| `../trials.parquet` | **le registre d'essais** : 806 lignes, 84 configurations | chaque script de décision | — |

Échantillon de référence de toutes les études sur le livre de tendance : **2003-07-17 →
2026-09-10, 6 039 séances**. C'est la première séance où au moins 30 instruments ont 252
séances d'historique.

### 3.2 `chantiers/macro-momentum/data/`

| fichier | contenu | source | période |
|---|---|---|---|
| `raw/mm_prices/assets.parquet` | 20 actifs multi-classes (indices, ETF obligataires, futures, dollar) | Yahoo | 1990-01-01 → 2026-09-10 |
| `raw/mm_macro/*.parquet` | INDPRO, PAYEMS, CPI en premières publications ; BAA10Y ; DTB3 | FRED / ALFRED | 1990 → 2026-07/09 |
| `raw/h2/markets.parquet` | 14 indices actions nationaux + 10 devises (24 séries) | Yahoo | 1990-01-02 → 2026-09-10 (les devises à partir de 1996-2006) |
| `raw/h2/rates.parquet` | taux longs et courts de **19 pays** (38 séries), mensuel | FRED / OCDE | 1990-01 → 2026-06 |
| `raw/h3_macro/*.parquet` | 13 séries US **avec tous leurs millésimes** (INDPRO, PAYEMS, CPI, core CPI, UNRATE, ICSA, TCU, DGORDER, AWHMAN, HOUST, PERMIT, RSAFS, UMCSENT) | ALFRED | 1998-01 → 2026-07/08 |
| `raw/h3_assets/prices.parquet` | 4 séries : actions, obligations, or, dollar | Yahoo | 1998-01-01 → 2026-09-10 (l'or à partir de 2000-08) |

Produits par `chantiers/macro-momentum/scripts/fetch.py`, `fetch_h2.py` et `fetch_h3.py`.

### 3.3 `chantiers/reversal-lab/data/`

| fichier | contenu | source | période |
|---|---|---|---|
| `raw/french/*.parquet` | facteurs de retournement court et long terme, momentum, 5 facteurs, déciles et portefeuilles taille × rendement passé | Ken French | 1990-01-02 → 2026-07-31 |
| `raw/stocks/us_large.parquet` | 117 grandes capitalisations US, **biais de survie déclaré** | Yahoo | 1990-01-02 → 2026-09-09 |

Produits par `chantiers/reversal-lab/scripts/fetch.py`.

### 3.4 Données des plans de recherche — sauvées, pas encore intégrées

Dans `pilotage/mesures_brutes/`, **non versionné**, avec une copie sur iCloud
(`Sauvegardes_Recherche_Quant/mesures_brutes_2026-09-22/`) :

| fichier | contenu | source | période | sert à |
|---|---|---|---|---|
| `firmes/bridgewater/probe/median_{rgdp,cpi,indprod,unemp}_level.xlsx` | médianes des prévisionnistes, 4 variables, 232 enquêtes | Fed de Philadelphie, *Survey of Professional Forecasters* | 1968-T4 → 2026-T3 | plan Bridgewater |
| `firmes/aqr/verif/49_Industry_Portfolios_Daily.csv` | 49 secteurs, **historique complet** (20 Mo) | Ken French | 1926-07-01 → 2026-07-31 | holdout scellé 1971-1989 (AQR / Two Sigma) |
| `firmes/aqr/verif/F-F_Momentum_Factor_daily.csv` | le facteur momentum qui manque à `factors_5` | Ken French | 1926-11-03 → 2026-07-31 | AQR |
| `firmes/aqr/verif/25_Portfolios_5x5.csv`, `nfci_full.csv`, `anfci_full.csv`, `baa_full.csv` | panneaux et séries complètes | Ken French, FRED | historiques complets | AQR |

⚠ Les deux fichiers `/private/tmp/spf.xlsx` et `spf_cpi.xlsx` **ne sont pas des données**.
Ils font 18 401 octets chacun : ce sont des pages HTML « 404 » servies avec un code
HTTP 200, un piège que le plan Bridgewater décrit lui-même.

⚠ **Le plan Rentec utilise des données minute qui ne sont pas dans ce dépôt** : 27
instruments, 8,8 Go, dans un dépôt privé de Guillaume. Seul lui peut exécuter ce plan.

---

## 4. Ce qui reste, précisément

Classé par ordre d'intérêt. Chaque ligne donne l'effort, les données nécessaires et le
critère de fin.

| # | tâche | effort | données | c'est fini quand |
|---|---|---|---|---|
| ~~0~~ | ~~Fixer l'objectif final~~ **FAIT le 23/09** : une présentation de 10 minutes ; objectif de fond, des stratégies dépendantes du régime (§0) | — | — | — |
| ~~1~~ | **FAIT le 25/09 : oral, deck de 14 diapositives et dossier `docs/presentation/dossier_regimes.pdf`.** Ancien énoncé : **Construire la présentation de 10 minutes**, au format entreprise, en 8 à 10 diapositives. Le fil : la question → la méthode (données point-in-time, critère écrit avant le chiffre) → ce qui marche (classifieur à 93,2 %, qui prédit la variance et pas la direction) → ce que ça implique (un outil de dimensionnement, pas de timing ; 13 changements d'état en 25 ans) → ce qu'on a testé sans succès (7 dispositifs, 4 hypothèses) → la suite, vers des stratégies dépendantes du régime. Trois figures : **(a)** R² sur la volatilité contre R² sur les rendements, un point par famille ; **(b)** la frise des 13 transitions ; **(c)** les dispositifs face à la règle d'une ligne | 1-2 j | `data/cache/*` et `docs/RESULTS_*.md`, déjà calculés | les diapositives et un texte oral de 10 minutes, répétés une fois |
| ~~2~~ | ~~**Plan Two Sigma**~~ **FAIT et FERMÉ le 23/09** : A FAIL, B sous-puissant, C non montré (`docs/RESULTS_TWOSIGMA.md`). Ce qui suit est l'ancien énoncé : ~~descendre l'arbre A → B → C** (le verrou est commité).~~ C'est la piste la plus proche de l'objectif de fond : le régime choisit entre dix signaux. Le verrou impose d'abord d'écrire et de commiter **les trois instruments** (A, B, C), leurs seuils et les empreintes des données (§13.1). Ensuite viennent les lectures dans l'ordre A, B, C, **quel que soit le résultat de chacune** ; chaque lecture compte comme un essai. Le verdict alimente la dernière diapositive | ~2 j pour les instruments, puis quelques heures par lecture | `industry_49`, `factors_5`, `features` (§3.1) | verdicts écrits dans `docs/RESULTS_TWOSIGMA*.md` |
| **3** | **Fermer l'arbre AHL** : le MDE de C1, que le pré-enregistrement déclare sous-puissant d'avance, la note de fermeture, et la décision sur la lecture A1 principale sur (126,10), qui reste due | ½-1 j | `trend_universe_m1.parquet` | `docs/RESULTS_AHL_LEVEL_C.md` et la fermeture |
| **4** | **Nettoyer et commiter `pilotage/mesures_brutes/`** : retirer les chemins absolus et privés des scripts | 1-2 h | — | les scripts dans git, les données dans `data/` |
| 5 | **Plan Bridgewater** : le régime dans la construction du portefeuille, l'autre axe jamais testé (8,5 j). Porte G0 de Rentec (1,5 j, Guillaume seul, données privées) | plusieurs jours | §3.4 | — |
| 6 | Les notes écrites sur H3 et sur la dispersion **ne sont plus prioritaires** sans mémoire. Elles restent utiles comme diapositives de réserve | ½ j | aucune | — |
| 7 | Archiver sur GitHub les anciens dépôts privés `macro-momentum` et `reversal-lab` | 5 min | — | Guillaume |

---

## 5. Prochaine séance de travail, proposition

Deux pistes en parallèle : la présentation ne demande aucun calcul, Two Sigma en demande.

| durée | ensemble | piste A · Two Sigma | piste B · présentation |
|---|---|---|---|
| 30 min | arrêter le fil de la présentation et le choix des stratégies à comparer | | |
| 2 h 30 | | instruments A, B et C, avec tests et seuils commités | figures (a) et (b), diapositives 1 à 5 |
| 30 min | point d'étape | | |
| 2 h | | lecture du niveau A, puis B et C | diapositives 6 à 10, texte oral |
| 1 h | répétition chronométrée (10 min), commits, mise à jour de ce fichier | | |

⚠ **Mis à jour le 23/09 au soir : la piste A de ce tableau est faite.** Les instruments
sont construits et commités, les trois niveaux lus et l'arbre fermé (§1). La présentation
peut en faire une diapositive : c'est le premier test de l'axe « sélection entre
signaux », et il échoue proprement, avec un placebo apparié exact et un témoin de
volatilité qui fait au moins aussi bien.

---

## 6. Règles du projet, à connaître avant de toucher au code

- Signal en T−1, trade en T. Rendements **nets, en excess** du taux cash. Erreurs HAC
  lag 6. Correction pour tests multiples dès qu'il y a plus d'un test.
- **On écrit le critère avant de produire le chiffre**, et on le commite. Un écart sous
  le MDE est « sous-puissant », **jamais** un succès.
- Tout script de décision **refuse de rendre un verdict** si une lecture n'est pas finie
  (NaN).
- Chaque lecture d'un résultat est un **essai**, journalisé dans `data/trials.parquet`.
- On ne modifie jamais un document verrouillé : toute déviation va dans
  `docs/PROTOCOL_FREEZE.md`.
- On commite tout, avec des messages qui donnent la raison. Un chiffre publié sans son
  code commité, c'est le piège n°1 de ce projet, et il s'est produit cinq fois.
