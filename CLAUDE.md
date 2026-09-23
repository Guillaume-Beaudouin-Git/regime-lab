# Projet Big Data — régimes de marché. Consignes pour les sessions Claude Code.

Tu démarres à froid. Ce fichier dit quoi faire, dans quel ordre, avec quelles commandes.
L'état d'avancement lisible par un humain est dans **`AVANCEMENT.md`**. Lis-le d'abord.

**Depuis le 22/09, tout le programme tient dans ce seul dépôt, qui est public.** Les
anciens dépôts `macro-momentum` et `reversal-lab` sont dans `chantiers/`, avec leur
historique intact : les identifiants de commit cités comme preuves de pré-enregistrement
existent toujours. Les anciens dossiers `1_Etude_principale/`, `2_Chantiers_derives/`,
`4_Feuille_de_route/` et `5_Plans_de_recherche/` n'existent plus. Les deux documents
verrouillés qui les citent encore sont couverts par la correspondance donnée dans
`docs/PROTOCOL_FREEZE.md`.

**Une deuxième session Claude Code peut tourner en parallèle** et elle ne voit que le
disque. Commite tout, mets tes raisons dans les messages de commit, et tiens ce fichier,
`AVANCEMENT.md` et `pilotage/feuille_de_route/TACHES.md` à jour. Une modification non
commitée est invisible pour l'autre session, ou pire, elle trébuche dessus.

**Le dépôt est public.** Rien sur les stratégies de trading personnelles de Guillaume,
aucun chemin absolu, aucun nom ni chemin de son dépôt privé (on écrit « le dépôt privé
voisin »). C'est la règle appliquée le 21/09 puis le 22/09.

---

## Nature du projet — précisée par Guillaume le 23/09

- **Ce n'est pas un mémoire.** C'est un projet de cours de M2, restitué par une
  **présentation d'une dizaine de minutes** au format entreprise. Tout ce qui ne sert pas
  cette présentation ou l'objectif de fond passe après.
- **L'objectif de fond**, une fois la parenthèse macro refermée, est de construire des
  **stratégies de trading algorithmique dépendantes du régime**. On cherche un résultat
  intéressant et prometteur, dont l'ambition est un Sharpe **net, en excess, hors
  échantillon, de 1 à 2**. Les règles dures ne changent pas pour autant, et un écart
  sous le MDE reste « sous-puissant ».
- **Hypothèse de coûts : institutionnels.** Le barème de tête du programme l'est déjà :
  futures 1 bp, matières premières 1,5 bp, actions US 5 bp, ETF sans future 7,5 bp. Les
  colonnes prudente et de stress restent rapportées à côté. Les barèmes déjà verrouillés
  ne changent pas.

---

## PROCHAINE ACTION

**Les priorités sont dans `AVANCEMENT.md` §4 et §5, et c'est Guillaume qui les fixe.**
Ne choisis pas à sa place. Trois chantiers sont prêts.

**1. La présentation de 10 minutes** (rien à calculer) : le fil et les trois figures
sont décrits dans `AVANCEMENT.md` §4, tâche 1.

**2. Le plan Two Sigma** : c'est la piste la plus proche de l'objectif de fond, puisque
le régime y choisit entre dix signaux. La phase 1 est faite (`22abbe0`, `3b64ab2`).
Le verrou `docs/PRESPEC_TWOSIGMA.md` est rédigé mais **pas commité** : il attend que
Guillaume relise son §12.0. Ne le commite pas sans cette relecture. Ensuite, on descend
l'arbre A → B → C, avec à chaque niveau : instrument, MDE, critère commité, puis lecture,
qui est un essai.

**3. Fermer l'arbre AHL** (priorité plus basse) :

**AHL niveau C, puis fermeture de l'arbre.** Le pré-enregistrement verrouillé
est `docs/PRESPEC_AHL.md` (commit `92e4e9e`). Ne le modifie pas.
- A est réfuté sur le signe (`docs/RESULTS_AHL_LEVEL_A.md`). B1 est indécidable
  (`docs/RESULTS_AHL_LEVEL_B.md`) : le MDE vaut 49,6 % de l'erreur moyenne contre une
  limite de 25 %, et 52,2 % en rendements log.
- C0 est passé à la lettre seulement : +0,067 de Sharpe brut, t 0,22, 2016-2026.
- ⚠ **C1 est un test de portefeuille, que le §9 déclare sous-puissant d'avance**
  (seuil 0,331). Reprends `scripts/run_ahl_level_b.py` comme modèle : instrument,
  MDE sous le nul, **critère écrit et commité avant le chiffre**, puis la lecture, qui
  est un essai à journaliser avec `regime_lab.analysis.trials.log`. Garde le garde qui
  refuse tout verdict sur une lecture non finie.
- Si C tombe, écris la fermeture (§14, niveau D). **Une mort propre sur tout l'arbre est
  un résultat valide et publiable.**

**Deux points ouverts, à ne pas trancher seul :**
1. A1 a été lu sur la jambe (63,5). Le §4 verrouillé désigne **(126,10)** comme jambe
   principale. La lecture principale reste donc due, mais c'est un essai et une décision
   sur l'arbre : demande à Guillaume.
2. `pilotage/mesures_brutes/` (non versionné) contient 120 fichiers sauvés de
   `/private/tmp` : les scripts de mesure des cinq plans et des contre-expertises A4. Ils
   portent des chemins absolus et privés. Il faut les nettoyer (chemins → configuration
   ou `Path(__file__)`) avant de les commiter.

---

## Ce que le programme a établi

**Le classifieur marche comme classifieur** : 93,2 % d'exactitude équilibrée contre les
récessions NBER, kappa 0,53, sur 6 377 jours hors échantillon.

**Il porte la variance, pas la moyenne** : +3,93 points de R² incrémental sur la
volatilité future (t −3,40), contre +0,030 sur les rendements (t 0,27).

**Sept dispositifs ont échoué à le monétiser** : T1, T3, l'atténuateur de Carver, la
barrière propfirm, la réplication Shu 2024, H-b et AHL niveau A. AHL niveau B est
indécidable, ce qui n'est pas la même chose.

⚠ **La portée exacte compte.** Ces mesures portent sur :

> Un classifieur **ordonné par la volatilité d'entraînement**, changeant d'état **0,514
> fois par an**, utilisé comme **multiplicateur de taille ou interrupteur au niveau du
> portefeuille**, sur un **livre de tendance unique de dimension effective 3,86**,
> n'apporte rien au-delà d'une volatilité réalisée sous sa médiane expansive.

Elles ne disent rien d'un état à plus d'une transition par an, d'une latente autre que la
volatilité, ni de la sélection entre signaux, de la construction de portefeuille ou de
l'exécution. Détail dans `pilotage/plans_de_recherche/ARBITRAGE.md` §6.

**Quatre hypothèses falsifiées** : H1, H2 (sa matière première a été supprimée par traité
en mars 1999), H3 et la prime de retournement.

## La règle opposable, avant toute idée neuve

Est un **septième dispositif**, et tombera sur le même placebo d'une ligne, tout objet
qui réunit les quatre propriétés suivantes : latente ordonnée par la volatilité ;
horloge sous ~2 transitions par an ; usage de dimensionnement ou d'interrupteur ; objet
conditionné de dimension effective inférieure à 4. Une piste neuve doit différer sur au
moins un de ces axes **et le chiffrer**.

## Règles dures, non négociables

- Signal en T−1, trade en T. Aucune exception.
- **Rendements nets en excess**, seule métrique de décision. HAC lag 6. Correction pour
  tests multiples dès que n_tests > 1.
- Walk-forward ≥ 5 plis, réellement évalués. Ciblage de volatilité quotidien.
- N ≥ 30 instruments pour toute affirmation transversale. 3 modifications maximum par
  hypothèse.
- **Bootstrap stationnaire par blocs**, via `regime_lab/analysis/bootstrap.py`.
- Placebo apparié obligatoire sur toute affirmation conditionnelle.
- **Un écart sous le MDE est UNDERPOWERED, JAMAIS un PASS.**
- Un critère s'écrit et se commite **avant** de toucher la donnée.
- Les documents verrouillés ne se modifient pas : les écarts vont dans
  `docs/PROTOCOL_FREEZE.md`.

## Les six pièges qui se sont réellement produits ici

1. **Un chiffre publié sans code commité, cinq fois.** La dernière : les scripts des
   mesures qui ont fondé les cinq plans vivaient dans `/private/tmp`. Relance le code
   avant d'écrire un nombre, et **n'écris jamais une mesure qui compte dans un dossier
   temporaire**.
2. **La troncature déguisée en signal, deux fois.**
3. **Un NaN qui se lit comme un verdict.** Tout script de décision refuse d'émettre un
   verdict sur une lecture non finie.
4. **Comparer des niveaux de prix à des rendements.** `trend_universe*.parquet` stocke des
   **prix**.
5. **Le gain qui passe par le dénominateur** : montre toujours qu'un gain n'est pas un
   gain d'exposition.
6. **Affirmer une cause sans l'avoir vérifiée.** Le 22/09, l'écart 7,26 / 9,10 alarmes par
   an a d'abord été attribué à une standardisation non causale. La vraie cause était le
   type de rendement (log contre simple). Corrigé dans `4dd0eb6`.

## Environnement

```bash
uv sync --all-packages --extra dev     # un seul .venv à la racine, les trois paquets en éditable
.venv/bin/python -m pytest -q          # 122 tests
.venv/bin/ruff check .                 # doit rester propre
```

Toujours `.venv/bin/python`, jamais le Python système. Si le dossier est déplacé, on
recrée le venv (`rm -rf .venv && uv sync --all-packages --extra dev`) au lieu de réparer
les chemins.

- `data/` est gitignoré : inventaire complet dans `AVANCEMENT.md` §3. Ne relance jamais
  `scripts/run_phase2.py` à la légère : il prend 15 à 20 min et réécrit
  `states.parquet`.
- `.env` (gitignoré) contient `FRED_API_KEY` et `SIMULATOR_PATH`. Seul
  `scripts/run_barrier.py` a besoin de ce dernier, un simulateur qui vit dans le dépôt
  privé voisin.
- ⚠ `~/.gitignore_global` contient `config.py` : un nouveau fichier de ce nom doit être
  ajouté avec `git add -f`.
- La littérature (PDF, sous droits) est hors du dépôt :
  `~/Desktop/M2/Projet_Big_Data_Regimes/3_Litterature/`.
