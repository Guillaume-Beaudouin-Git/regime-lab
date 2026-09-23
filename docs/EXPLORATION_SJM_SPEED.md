# Rendre le modèle à sauts parcimonieux plus réactif : une exploration

**Statut : exploratoire et descriptif (23/09/2026).** Aucun rendement, aucun Sharpe, rien
d'écrit dans `data/trials.parquet` ni dans le cache. **Aucun de ces réglages ne remplace le
classifieur figé de l'étude** (A′, `data/cache/states.parquet`). Si une variante plus
rapide doit servir dans un test de stratégie, il faut la déclarer d'avance, avec son
critère, avant de toucher la donnée.

Script : `scripts/explore_sjm_speed.py` (environ 1 min, `.venv/bin/python`).

## Ce qui a été lancé

- **Un ajustement par configuration**, sur 1992-03 → 2006-12 (3 870 séances), avec le
  wrapper de l'étude : `JumpRegimes`, `max_features=10`, états ordonnés par la volatilité
  du livre de base sur la fenêtre d'entraînement, état 0 = stress. Ensuite,
  `predict_online` tourne en avant sur tout l'échantillon, si bien que chaque étiquette de
  2007-01 à 2026-09 (5 137 séances) n'utilise que les lignes antérieures ou égales à sa
  date. **Pas de réajustement** : l'étude, elle, réajuste tous les six mois.
- **Deux jeux de variables**, tous deux pris dans `features.parquet`, qui est déjà
  standardisé en fenêtre expansive :
  - *complet* : les 50 variables, comme A′ ;
  - *rapide* (8) : `vol_rv_5`, `vol_rv_21`, `vol_term`, `vol_vix`, `vol_vrp`,
    `mom_eq_21` et `xs_dispersion_ind`, sur des fenêtres de 5 à 21 séances ou
    instantanées, plus `cre_baa_chg63` (63 séances), pour que le jeu ne se réduise pas à
    la volatilité.
- **λ ∈ {10 ; 1 ; 0,3 ; 0,1}**, soit huit ajustements. λ = 10 est ce que la calibration de
  l'étude a retenu pour la fenêtre qui finit en 2006 (journal des essais,
  `train_end` 2006-03-30). λ = 1 est le plancher de la grille déclarée, et 0,3 et 0,1 sont
  sous la grille. Attention à l'échelle : `jumpmodels` divise λ par √(nombre de
  variables). La pénalité effective vaut donc λ/7,07 pour le jeu complet et λ/2,83 pour
  le jeu rapide.
- **Mesures sur 2007-2026** :
  - transitions par an ;
  - part des séances en stress ;
  - épisodes de stress : leur nombre, dont ceux de moins de 10 séances ;
  - η² de l'état contre la volatilité réalisée sur 21 jours de `eq_us_large` ;
  - κ contre un filtre de volatilité dont la part de stress est comparable ;
  - exactitude équilibrée contre les récessions NBER
    (`reliability.external_validation`, avec le même alignement mensuel que
    `run_evaluation.py`).
- **Références**, en plus de la série A′ de l'étude :
  - le placebo de l'étude (VR 21 j sous sa médiane expansive) ;
  - un filtre causal « VR 21 j au-dessus de son 80ᵉ centile expansif ». Sa part de stress
    (23 %) est celle des modèles ajustés (20 à 22 %).

  Sans ce second filtre, l'η² ne se compare pas. Sur une volatilité asymétrique, une
  coupure à la médiane plafonne l'η² plus bas (0,34) qu'une coupure dans la queue.
  Meilleur η² possible pour une seule coupure sur la VR 21 j, en connaissant l'avenir :
  0,577.

## Résultats, 2007-01 → 2026-09, hors échantillon

| Série | Variables | λ | Transitions/an | Part stress | Épisodes (< 10 séances) | η² vs VR 21 j | κ vs filtre 80ᵉ c. | NBER, exact. équil. |
|---|---|---|---|---|---|---|---|---|
| A′ de l'étude (réajustée tous les 6 mois) | 50, parcimonieux | calibré | 0,59 | 13,8 % | 6 (1) | 0,254 | 0,40 | 95,1 % |
| Placebo médiane | VR 21 j | – | 7,75 | 51,3 % | 79 (32) | 0,342 | 0,44 | 76,5 % |
| Filtre 80ᵉ centile | VR 21 j | – | 5,30 | 22,6 % | 54 (28) | 0,504 | 1 | 83,3 % |
| Ajustement 1 | complet | 10 | 1,57 | 19,9 % | 16 (2) | 0,436 | 0,70 | 92,6 % |
| Ajustement 2 | complet | 1 | 2,85 | 20,1 % | 29 (14) | 0,443 | 0,73 | 92,9 % |
| Ajustement 3 | complet | 0,3 | 3,34 | 19,9 % | 34 (19) | 0,442 | 0,72 | 93,0 % |
| Ajustement 4 | complet | 0,1 | 3,43 | 19,8 % | 35 (20) | 0,442 | 0,72 | 93,0 % |
| Ajustement 5 | rapide | 10 | 2,01 | 21,7 % | 21 (4) | 0,436 | 0,74 | 85,4 % |
| Ajustement 6 | rapide | 1 | 3,19 | 21,6 % | 33 (12) | 0,430 | 0,74 | 85,4 % |
| Ajustement 7 | rapide | 0,3 | 4,66 | 21,2 % | 48 (27) | 0,425 | 0,73 | 85,2 % |
| Ajustement 8 | rapide | 0,1 | 5,74 | 21,1 % | 59 (37) | 0,426 | 0,74 | 84,9 % |

Les huit ajustements partagent les mêmes longs épisodes de stress : 2008-2009,
août à décembre 2011, février à août 2020, avril à décembre 2022. Ils diffèrent par les
épisodes courts accolés à ceux-ci et par quelques alertes isolées : 2015-2016 et 2018,
plus 2025-2026 pour le jeu rapide. La série A′ de l'étude n'a plus aucun épisode de stress
après le 2021-04-02 : elle ne signale pas 2022.

### Lecture

1. **Avec les 50 variables, le levier λ sature vite.** La cadence passe de 1,6 à 2,8
   transitions par an entre λ = 10 et λ = 1, puis à 3,3 et 3,4. Sous λ = 1, la partition
   ajustée ne bouge plus (1,69 transition par an en entraînement pour les trois valeurs) :
   seul le filtre en ligne change, et il plafonne vers 3,4 par an. La persistence vient
   alors des variables elles-mêmes.
2. **Le levier des variables rapides porte plus loin**, de 2,0 à 5,7 transitions par an.
   Ce n'est pas un effet de l'échelle de la pénalité : à λ nominal égal, le jeu rapide
   paie une pénalité effective 2,5 fois plus lourde, et il bascule pourtant plus souvent.
   Ses poids se concentrent sur `xs_dispersion_ind` (0,67), `vol_rv_21` (0,49), `vol_vix`
   (0,47) et `vol_rv_5` (0,28). `cre_baa_chg63` pèse 0,04, et `vol_term`, `vol_vrp` et
   `mom_eq_21` sont presque à zéro. C'est donc un modèle de dispersion et de niveau de
   volatilité.
3. **Les transitions gagnées sont surtout du clignotement.** Baisser λ ajoute surtout des
   épisodes de moins de 10 séances, en bordure des mêmes épisodes : on passe de 2 à 20
   pour le jeu complet et de 4 à 37 pour le jeu rapide. La durée médiane d'un épisode
   tombe d'environ 28 à 5 séances. Le nombre de longs épisodes ne bouge presque pas.
4. **Le prix se paie en ressemblance au filtre de volatilité, et en NBER pour le jeu
   rapide.**
   - L'η² reste vers 0,43-0,44 sur les huit ajustements, quel que soit λ, contre 0,25
     pour la série de l'étude et 0,50 pour le filtre apparié.
   - Le κ avec ce filtre vaut 0,70 à 0,74, contre 0,40 pour la série de l'étude.
   - Contre NBER, le jeu complet tient 93 % : sensibilité 98-99 %, spécificité 87 %,
     contre 96 % et 94 % pour la série de l'étude. Il alerte donc un peu plus souvent hors
     récession (2011, 2015-2016, 2018, 2022).
   - Le jeu rapide perd 8 points (85 %) et tombe au niveau du filtre apparié (83 %). La
     perte vient surtout de la sensibilité (85-86 %) : il sort du stress au printemps
     2008, en pleine récession. Sa spécificité baisse aussi un peu (84-85 %), avec des
     alertes en 2018 et en 2025-2026.
   - Réserve : sur 2007-2026, NBER ne compte que deux récessions, donc un écart d'un ou
     deux points ne veut rien dire.
5. **Une remarque sur la série de l'étude elle-même** (non vérifiée).
   - Ajusté une seule fois sur 1992-2006 au λ de l'étude (10), le modèle bascule déjà
     1,57 fois par an, soit 2,7 fois la série de l'étude (0,59).
   - L'écart vient donc du protocole de réajustement, pas de λ. Pour les fenêtres qui
     finissent en 2022-03, 2024-03 et 2026-03, le journal des essais montre la même
     cadence d'entraînement, 0,19 par an, pour **tous** les λ de 1 à 300. C'est sous la
     bande de plausibilité (0,5), et la calibration s'est rabattue sur la médiane de la
     grille (20). C'est avec ces réajustements que la série reste calme de 2021-04 à
     2026-09.
   - Hypothèse non testée : dès que la fenêtre d'entraînement contient 2008 et 2020,
     l'état de stress est réservé aux extrêmes.

## À signaler : `CALIBRATION_NOTES.md` semble périmé

La note dit que la calibration a retenu λ = 1 neuf fois sur treize, et qu'au plancher le
modèle bascule 1,3 à 2,2 fois par an. Le journal des essais du passage qui a produit
l'actuel `states.parquet` (2026-09-09) ne reproduit pas ces chiffres :

- λ retenu : 10 cinq fois, 3 deux fois, 1 deux fois et 30 une fois, plus trois replis
  sur 20 ;
- cadence d'entraînement à λ = 1 : de 0,19 à 2,40 transitions par an.

La note (`6d37912`, 08/09) est antérieure au passage à l'ordre par volatilité (`3664cc1`,
09/09), qui a changé l'objectif de la calibration. Elle décrit probablement la calibration
d'avant. Je ne l'ai pas corrigée ici : c'est à trancher.

## Conclusion

**Oui, on peut faire basculer le modèle de 3 à 6 fois par an.**

- Avec les 50 variables et λ ≤ 0,3, on atteint environ 3,4 transitions par an, et
  l'exactitude NBER se maintient à 93 %.
- Avec 8 variables rapides et λ entre 0,3 et 0,1, on atteint 4,7 à 5,7 par an, mais
  l'exactitude NBER tombe à 85 %.

**Mais ce qu'on gagne est surtout du clignotement** en bordure des mêmes épisodes de
stress. Surtout, toutes les variantes, lentes comme rapides, ressemblent bien plus à un
filtre de volatilité d'une ligne que la série de l'étude : κ de 0,70 à 0,74 contre 0,40.
Sur la cadence, la part de stress et le NBER, la plus rapide est presque le filtre
apparié lui-même : 5,7 contre 5,3 transitions par an, 85 % contre 83 %.

Au regard de la règle opposable du programme, une variante plus rapide franchit bien
l'axe « horloge au-dessus d'environ 2 transitions par an ». Mais elle reste une latente
ordonnée par la volatilité, et elle se **rapproche** du placebo au lieu de s'en
éloigner. Un test de stratégie qui l'utiliserait devrait donc être déclaré d'avance et
battre le filtre de volatilité **apparié**, pas seulement le placebo à la médiane. Il
devrait aussi différer du septième dispositif sur un autre axe, par exemple l'usage.

## Limites

- Une seule fenêtre d'entraînement et aucun réajustement : ces chiffres décrivent un
  modèle figé fin 2006, pas le protocole de l'étude.
- λ nominal identique mais pénalité effective différente entre les deux jeux (voir
  plus haut).
- L'η² est calculé sur la VR en niveau. Le κ contre le filtre apparié est la mesure de
  ressemblance la plus directe.
- Un seul jeu rapide a été essayé. Il n'y a eu ni λ = 0, ni λ entre 1 et 10 sur le jeu
  rapide.
