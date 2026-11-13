# Projet Big Data — régimes de marché. Consignes pour les sessions Claude Code

L'état du projet, l'inventaire des données et ce qui reste sont dans `AVANCEMENT.md` : lis-le d'abord. Ce
fichier ne garde que ce que le dépôt ne dit pas de lui-même.

## Le dépôt

- Depuis le 22/09, tout le programme tient dans ce dépôt, qui est public. Les anciens dépôts `macro-momentum` et
  `reversal-lab` sont dans `chantiers/` avec leur historique : les commits cités comme preuves de
  pré-enregistrement existent toujours. Les anciens dossiers `1_Etude_principale/`, `2_Chantiers_derives/`,
  `4_Feuille_de_route/` et `5_Plans_de_recherche/` n'existent plus ; les deux documents verrouillés qui les citent
  sont couverts par la correspondance de `docs/PROTOCOL_FREEZE.md`.
- Rien sur les stratégies de trading personnelles de Guillaume, aucun chemin absolu, aucun nom ni chemin de son
  dépôt privé : on écrit « le dépôt privé voisin » (règle appliquée le 21/09 puis le 22/09).
- Aucune ligne d'attribution dans les commits, ni `Co-Authored-By` ni `Claude-Session`. Guillaume a arrêté les
  `Co-Authored-By` partout le 13/11 (« Oui il faut arrêter les co-authored partout. ») et laissé le reste à la
  session principale (« Fait au mieux »), qui a tranché ainsi parce que l'historique de ce dépôt public a été
  nettoyé de ses signatures le 29/10 : les trailers « Co-Authored-By: Claude » en ont été retirés et `origin/main`
  a été force-poussé (voir la fin d'`AVANCEMENT.md` pour réaligner un clone).

## Deux sessions en parallèle

- Une deuxième session peut tourner en même temps et ne voit que le disque. Commite tout, mets la raison dans le
  message, et tiens à jour ce fichier, `AVANCEMENT.md` et `pilotage/feuille_de_route/TACHES.md`.
- Le dossier de travail est partagé : n'ajoute que tes fichiers, avec des chemins explicites, jamais `git add -A`.
- Plan Bridgewater (lancé le 24/09 par Guillaume) : la session du compte B (`projet-big-data-regimes-cd`) construit
  l'outillage, le verrou, les instruments et les lectures. L'autre session ne touche pas à
  `regime_lab/construction/`, `scripts/*bridgewater*`, `scripts/fetch_spf.py`, `tests/test_bridgewater_*`,
  `docs/PRESPEC_BRIDGEWATER.md`, `docs/RESULTS_BRIDGEWATER*`, `docs/artifacts/bridgewater/`, `data/raw/spf/`, ni
  aux lignes `bridgewater` de `data/trials.parquet`. Recouvrement connu, à déclarer dans le verrou : l'idée 3
  « Couverture » (`docs/RESULTS_COUVERTURE.md`) et l'idée 4 « Budget de risque » (`docs/RESULTS_BUDGET_RISQUE.md`).

## Nature du projet (Guillaume, 23/09 et 13/11)

- Ce n'est pas un mémoire : c'est un projet de cours de M2, restitué par une présentation d'une dizaine de minutes
  au format entreprise. L'oral a eu lieu le 25/09 ; le dossier remis est `docs/presentation/dossier_regimes.pdf`
  (`scripts/build_dossier.py`, étiquette `dossier-2026-09-25`). Un chiffre changé dans le dossier se change aussi
  dans les deux PDF de présentation, les deux decks et `docs/presentation/PARTIE1_CHEMINEMENT.md`.
- L'objectif (Guillaume, 13/11) : « l'objectif n'est pas les stratégies, mais bien l'identification du régime ou
  de la tendance, du sentiment. » Il remplace l'objectif de trading du 23/09 (stratégies dépendantes du régime),
  abandonné le 13/11 : un document plus ancien qui le porte encore ne le remet pas en vigueur.
- Un résultat d'identification se juge sur les mesures que le dépôt emploie déjà : exactitude équilibrée, kappa,
  rappel, spécificité et changements d'état par an contre les récessions NBER, face à une règle de volatilité d'une
  ligne (`docs/artifacts/temoin_nber.txt`) ; R² incrémental sur la volatilité future (`docs/RESULTS_CRISE.md`, test P) ou
  sur les rendements futurs (`AVANCEMENT.md` §0), avec son t. L'hygiène reste la même : critère pré-enregistré, walk-forward, placebo, MDE.
- Pour une étude qui teste un usage en trading (dimensionnement, interrupteur) : coûts de tête institutionnels,
  futures 1 bp, matières premières 1,5 bp, actions US 5 bp, ETF sans future 7,5 bp ; les colonnes prudente et de
  stress restent rapportées à côté. Les barèmes déjà verrouillés ne changent pas.

## Ce qui reste

Les priorités sont dans `AVANCEMENT.md` §4 et §5, et c'est Guillaume qui les fixe : ne choisis pas à sa place.
- Le plan Two Sigma est fermé depuis le 23/09 (`docs/RESULTS_TWOSIGMA.md`) : il n'y a plus rien à lire, et
  `scripts/run_twosigma_tree.py` refuse toute seconde lecture.
- Fermer l'arbre AHL. Le pré-enregistrement verrouillé est `docs/PRESPEC_AHL.md` (commit `92e4e9e`) ; ne le modifie
  pas. A est réfuté sur le signe, B1 indécidable (MDE à 49,6 % de l'erreur moyenne contre une limite de 25 %, 52,2 %
  en rendements log), C0 passé à la lettre seulement (+0,067 de Sharpe brut, t 0,22, 2016-2026). C1 est un test de
  portefeuille que le §9 déclare sous-puissant d'avance (seuil 0,331). Prends `scripts/run_ahl_level_b.py` pour
  modèle : instrument, MDE sous le nul, critère écrit et commité avant le chiffre, puis la lecture, qui est un essai
  à journaliser avec `regime_lab.analysis.trials.log`. Garde le garde qui refuse un verdict sur une lecture non
  finie. Si C tombe, écris la fermeture (§14, niveau D) : une mort propre sur tout l'arbre est un résultat valide.
- Deux points à ne pas trancher seul. A1 a été lu sur la jambe (63,5), alors que le §4 verrouillé désigne (126,10)
  comme jambe principale : la lecture principale reste due, c'est un essai et une décision sur l'arbre, demande à
  Guillaume. `pilotage/mesures_brutes/` (non versionné) contient les scripts de mesure des cinq plans et des
  contre-expertises A4, avec des chemins absolus et privés : les passer en configuration ou en `Path(__file__)`
  avant de les commiter.

## Citer les résultats sans les déformer

Les résultats eux-mêmes sont dans `AVANCEMENT.md` §0-1 et les `docs/RESULTS_*.md`. Ce qui suit dit comment les
citer.
- Le 93,2 % d'exactitude équilibrée contre le NBER (`docs/artifacts/temoin_nber.txt`) repose sur deux récessions
  seulement (2008-09, 2020), et la même méthode réduite à 30 variables sur 1926-2026 n'en reconnaît que 6 sur 14,
  pas mieux qu'une règle de volatilité (`docs/RESULTS_LONGHIST.md`). Ne cite pas le 93 % sans ces deux faits.
- Le +3,93 points de R² sur la volatilité future est « au-delà d'une règle de volatilité passée », jamais « au-delà
  du VIX » : au-delà du VIX il n'ajoute que +0,20 point, t −1,34 (`docs/RESULTS_CRISE.md`, test P).
- AHL niveau B est indécidable, ce qui n'est pas un échec.
- Le livre de 9 stratégies à 1,52 de Sharpe brut (2005-2026) n'est pas un résultat de régime, et c'est un plafond :
  sans coût, stratégies choisies en connaissant ces années.
- Le SJM entre en stress tard et y reste pendant les reprises, en partie à cause des réestimations : trois des six
  entrées en stress d'A′ tombent un jour de réestimation (le filtrage repart de `train.tail(252)` dans
  `regime_lab/models/base.py`).
- La portée exacte des mesures, et ce qu'elles ne disent pas, est dans `pilotage/plans_de_recherche/ARBITRAGE.md`
  §6. Pour une étude qui teste un usage en trading, une piste neuve qui réunit les quatre propriétés du « septième
  dispositif » (latente ordonnée par la
  volatilité, moins de ~2 transitions par an, usage de dimensionnement ou d'interrupteur, objet de dimension
  effective inférieure à 4) tombera sur le même placebo d'une ligne : elle doit différer sur au moins un axe et le
  chiffrer.

## Règles dures

Pour toute étude :
- Signal en T−1, trade en T, sans exception ; HAC lag 6.
- Tests multiples : Šidák sur la famille testée dans l'étude, jamais de pénalité par un compteur global
  (décision du 30/10).
- Walk-forward d'au moins 5 plis, réellement évalués.
- Au moins 30 instruments pour une affirmation transversale.
- Le nombre de modifications d'une hypothèse et le DSR se rapportent à titre informatif ; ce ne sont pas des portes
  (Guillaume, 13/11 : « On garde les 3 modifs et le DSR à titre informatif, c'est juste qu'ils ne suffisent pas en
  tant que portes. »).
- Ces deux règles (Šidák sur la famille de l'étude ; trois modifications et DSR à titre informatif) valent pour les
  études à venir : les documents verrouillés gardent leurs règles d'origine, et un écart se déclare dans
  `docs/PROTOCOL_FREEZE.md`.
- Bootstrap stationnaire par blocs, via `regime_lab/analysis/bootstrap.py`.
- Placebo apparié sur toute affirmation conditionnelle.
- Un écart sous le MDE est UNDERPOWERED, pas un PASS.
- Un critère s'écrit et se commite avant de toucher la donnée.
- Un document verrouillé ne se modifie pas : les écarts vont dans `docs/PROTOCOL_FREEZE.md`.

Pour une étude qui teste un usage en trading (dimensionnement, interrupteur), en plus :
- Rendements nets en excess, seule métrique de décision.
- Ciblage de volatilité quotidien.

## Les sept pièges qui se sont produits ici

1. Un chiffre publié sans code commité, cinq fois ; la dernière, les scripts des mesures qui ont fondé les cinq
   plans vivaient dans un dossier temporaire. Relance le code avant d'écrire un nombre, et n'écris pas une mesure qui
   compte dans un dossier temporaire.
2. La troncature déguisée en signal, deux fois.
3. Un NaN qui se lit comme un verdict : tout script de décision refuse de rendre un verdict sur une lecture non finie.
4. Comparer des niveaux de prix à des rendements : `trend_universe*.parquet` stocke des prix.
5. Le gain qui passe par le dénominateur : montre qu'un gain n'est pas un gain d'exposition.
6. Affirmer une cause sans l'avoir vérifiée. Le 22/09, l'écart de 7,26 contre 9,10 alarmes par an a d'abord été
   attribué à une standardisation non causale ; la vraie cause était le type de rendement (log contre simple),
   corrigé dans `4dd0eb6`.
7. Décrire un réglage au lieu de le mesurer. Le 25/09, « le Sparse Jump Model garde au plus 10 variables » a remplacé
   « ≈ 10 effectives » partout, alors que `max_feats = 10` borne le nombre effectif et que 17 à 24 variables gardent
   un poids non nul (`scripts/measure_sjm_sparsity.py`).

## Environnement

```bash
uv sync --all-packages --extra dev     # un seul .venv à la racine, les trois paquets en éditable
.venv/bin/python -m pytest -q          # 851 tests au 25/09, 15 à 25 min
.venv/bin/ruff check .                 # doit rester propre ; la CI (tests.yml) lance ruff sur regime_lab, scripts, tests
```

- Toujours `.venv/bin/python`, pas le Python système. Si le dossier est déplacé, recrée le venv
  (`rm -rf .venv && uv sync --all-packages --extra dev`) plutôt que de réparer les chemins.
- `data/` est gitignoré (inventaire dans `AVANCEMENT.md` §3). `scripts/run_phase2.py` prend 15 à 20 minutes et
  réécrit `states.parquet` : ne le relance pas à la légère.
- `.env` (gitignoré) contient `FRED_API_KEY` et `SIMULATOR_PATH` ; seul `scripts/run_barrier.py` a besoin du second,
  un simulateur qui vit dans le dépôt privé voisin.
- `~/.gitignore_global` contient `config.py` : un nouveau fichier de ce nom s'ajoute avec `git add -f`.
- La littérature (PDF sous droits) est hors du dépôt, dans le dossier `3_Litterature/` voisin.
