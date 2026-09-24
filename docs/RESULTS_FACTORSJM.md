# Résultats — un Sparse Jump Model par facteur (piste `factorsjm`, idée 5)

Pré-enregistrement : `docs/PRESPEC_FACTORSJM.md`, commité avant tout ajustement
(`1e812bf`). Instrument commité avant la lecture (`4f1555c`). Lecture unique (`3003569`),
18 essais journalisés (12 `factorsjm`, 6 `factorsjm_replication`), exactement ceux
déclarés. Tous les chiffres ci-dessous sont dans `docs/artifacts/factorsjm/instrument.txt`
et `docs/artifacts/factorsjm/reading.txt` (et `reading.json`). **Coût nul partout**, en
excès du cash.

---

## En une minute

- **L'idée** (Shu et Mulvey, arXiv 2410.14841) : au lieu d'un seul régime pour tout le
  marché, **un régime par facteur** (marché, taille, valeur, rentabilité, investissement,
  momentum), chacun estimé sur ses propres variables, et six décisions qui pilotent un
  portefeuille multi-facteurs.
- **Le dispositif franchit les quatre axes de la règle opposable** : des latentes qui ne
  sont pas des règles de volatilité, 22 changements d'état par an au total, une
  allocation entre six facteurs, une dimension effective de 4,7 à 5,3.
- **Il ne bat pas le simple fait de tenir les six facteurs.** Sur 48,6 ans (1978-2026),
  le livre piloté par les régimes fait un Sharpe de **1,39**, contre **1,51** pour les
  mêmes six facteurs tenus en permanence. Verdict du test principal : **PAS UTILE**.
- **Il sait pourtant quelque chose** : il fait mieux que 99,5 % de ses placebos (les
  mêmes décisions à des dates tirées au hasard). Mais une **règle de tendance d'une
  ligne** (« le facteur a-t-il monté sur un an ? ») en sait autant : 1,53.
- **Sur la fenêtre du papier (2007-2024), le résultat a l'air positif** (+0,29 de
  Sharpe) ; **avant 1990, il est franchement négatif** (−0,63). Le résultat publié
  dépend de sa fenêtre.
- **La réplication est partielle** : le signe publié est reproduit pour la valeur et la
  rentabilité, pas pour la taille ni le momentum (2 sur 4).

---

## 1. Ce qui a été testé

**Les données.** Les six facteurs de Ken French, quotidiens, de juillet 1963 à juillet
2026, retéléchargés en entier pour gagner en puissance : 48,6 ans hors échantillon au
lieu de 24. L'instrument a vérifié que le fichier complet coïncide exactement avec le
fichier 1990-2026 déjà présent (écart maximal 0,0000) et qu'aucune séance ne manque au
momentum.

**Le modèle, par facteur.** Un Sparse Jump Model à deux états, estimé sur les variables
du papier : tendance du facteur à 8, 21 et 63 séances, oscillateurs (RSI, %K, MACD),
déviation à la baisse, bêta sur le marché, plus quatre variables de marché. Le VIX et le
taux 2 ans, qui n'existent pas avant 1990 et 1976, sont remplacés par la volatilité
réalisée du marché et le taux 1 an. Les états sont nommés « haussier » et « baissier »
selon le rendement que le facteur y a gagné, **pas selon la volatilité**. Réestimation
tous les 6 mois, sur 8 à 12 ans d'historique ; la pénalité de saut est re-choisie tous
les 6 mois parmi six valeurs, par la règle du papier ; un état connu le jour *d* n'est
utilisé que le jour *d* + 2.

**Le portefeuille.** Chaque facteur est une « jambe » ciblée à 10 % de volatilité. Le
livre est la moyenne des six jambes.

**La famille de tests, fixée d'avance (8 tests, seuil corrigé α = 0,05 / 8)** :
- **B1, le test principal** : chaque jambe tenue quand son facteur est haussier, coupée
  quand il est baissier ;
- **B2** : l'exposition du papier, entre −1 et +1 selon le rendement moyen du facteur
  dans son état courant ;
- **F_*** : la même chose, facteur par facteur.

Chaque test est comparé au livre tenu en permanence, à trois règles d'une ligne propres
à chaque facteur (sa volatilité au-dessus de sa médiane, au-dessus de son 80ᵉ centile,
son rendement sur un an négatif), à un témoin statique pour l'exposition du papier, et à
400 placebos.

---

## 2. Le dispositif franchit les quatre axes (mesuré avant la lecture)

| axe de la règle opposable | mesure | franchi ? |
|---|---|---|
| 1. latente ordonnée par la volatilité | kappa entre l'état baissier et la règle de volatilité propre au facteur : de −0,06 à +0,14 selon le facteur | **oui** : ce ne sont pas des règles de volatilité |
| — | kappa avec la règle de tendance du facteur : de +0,19 (momentum) à +0,56 (marché) | en partie une règle de tendance |
| 2. horloge | 2,0 (marché) à 4,7 (rentabilité) changements d'état par an ; 22,4 pour le livre | **oui** |
| 3. usage | allocation entre six facteurs | **oui** |
| 4. dimension effective | 4,71 pour les six facteurs sur 1990-2026 (le chiffre du conseiller, reproduit exactement) ; 4,89 pour les jambes ; **5,30** pour les six chemins d'exposition | **oui** |

Comme le plan Two Sigma, le dispositif franchit les quatre axes ; sa latente, elle, est
ordonnée par le rendement de chaque facteur, et non par un contexte de marché.

---

## 3. Les résultats du test

48,6 ans, du 05/01/1978 au 31/07/2026, 12 243 séances. « Meilleure règle » = le delta le
plus élevé parmi les témoins d'une ligne du test.

| test | Sharpe tenu | Sharpe régime | delta | seuil (MDE) | t | placebo | meilleure règle d'une ligne | **verdict** |
|---|---|---|---|---|---|---|---|---|
| **B1 livre marche/arrêt (principal)** | 1,51 | 1,39 | **−0,126** | 0,506 | −0,97 | 99,5 % | tendance +0,020 | **PAS UTILE** |
| B2 livre, exposition du papier | 1,51 | 0,86 | −0,652 | 0,847 | −3,29 | 100 % | statique −0,206 | PAS UTILE |
| F marché | 0,58 | 0,31 | −0,270 | 0,542 | −1,81 | 57,2 % | statique −0,099 | PAS UTILE |
| F taille (SMB) | 0,06 | 0,39 | +0,324 | 0,803 | +1,56 | 98,2 % | tendance +0,315 | SOUS-PUISSANT |
| F valeur (HML) | 0,28 | 0,56 | +0,278 | 1,033 | +1,16 | 99,2 % | tendance +0,346 | SOUS-PUISSANT |
| F rentabilité (RMW) | 0,75 | 0,57 | −0,172 | 0,830 | −0,86 | 98,8 % | tendance −0,082 | PAS UTILE |
| F investissement (CMA) | 0,44 | 0,45 | +0,010 | 0,889 | +0,04 | 100 % | tendance +0,104 | SOUS-PUISSANT |
| F momentum (UMD) | 1,15 | 0,36 | −0,788 | 0,641 | −4,58 | 33,2 % | statique +0,004 | PAS UTILE (négatif au-delà du seuil, non confirmé par le placebo) |

**Sensibilités, jamais décisives :**

| sensibilité | Sharpe régime | delta | seuil | verdict |
|---|---|---|---|---|
| S1 : B1 à pénalité fixe (50, sans réglage) | 1,36 | −0,150 | 0,516 | pas utile |
| S2 : B2 à pénalité fixe | 0,79 | −0,728 | 0,865 | pas utile |
| S3 : B1 avec le marché toujours tenu (comme le papier) | 1,52 | +0,004 | 0,503 | sous-puissant ; la règle de tendance fait +0,128 |
| S4 : B2 avec le marché toujours tenu | 1,00 | −0,514 | 0,839 | pas utile |

**Aucun test n'est utile.** Aucun delta positif n'atteint son seuil de détection. Le seul
delta qui le dépasse est négatif (momentum), et le placebo ne le confirme pas. Là où le
régime aide (taille, valeur), la règle de tendance d'une ligne fait autant ou mieux.

---

## 4. Ce que la lecture montre de plus (descriptif, jamais décisif)

**Le régime change la forme du risque, pas le Sharpe.** B1 divise par deux la perte
maximale du livre : **−6,4 % au lieu de −14,1 %**. Mais la règle de tendance d'une ligne
fait la même chose (−6,9 %) avec un Sharpe de 1,53. C'est le constat du reste du
programme, retrouvé par un autre chemin.

**Le régime sait quelque chose, mais pas assez.** Le placebo garde les mêmes
expositions et les pose à des dates tirées au hasard. Il fait bien pire : −0,454 en
médiane pour B1, −1,230 pour B2. Le calage du SJM vaut donc mieux que le hasard. Il ne
vaut pas mieux que tout garder, ni qu'une règle de tendance.

**Le résultat dépend de la fenêtre.**

| delta de Sharpe | 1978-2026 | 2007-2024 (fenêtre du papier) | avant 1990 | après 1990 |
|---|---|---|---|---|
| B1 marche/arrêt | −0,126 | **+0,291** | **−0,634** | −0,007 |
| B2 exposition du papier | −0,652 | +0,134 | −1,597 | −0,393 |

Sur les cinq plis de longueur égale, B1 n'est positif que sur les deux derniers (40 %).
Le ratio d'information de B1 contre le livre tenu vaut −0,32 sur tout l'échantillon et
+0,24 sur 2007-2024 ; le papier annonce 0,40 à 0,49 contre son portefeuille équipondéré,
avec d'autres actifs.

**L'exposition signée du papier coûte cher au momentum.** Sur UMD, elle fait tomber le
Sharpe de 1,15 à 0,36. Le témoin statique et les règles de volatilité laissent UMD à
peu près intact (+0,004, −0,020, −0,048).

**Le réglage choisit souvent la pénalité la plus faible.** λ = 10, le bas de la grille,
est retenu 46 à 51 fois sur 98 pour la taille, la valeur et la rentabilité. La grille n'a
pas été élargie après coup, comme le veut la règle du programme
(`docs/CALIBRATION_NOTES.md`).

---

## 5. La réplication du papier (2007-01-03 → 2024-06-28, 17,5 ans)

La stratégie long-short mono-facteur du papier, sur le facteur brut, sans ciblage de
volatilité.

| papier | ici | Sharpe publié | **Sharpe ici** (t) | changements/an publiés | ici | verdict |
|---|---|---|---|---|---|---|
| Value | HML | 0,39 | **+0,32** (+1,30) | 3,16 | 4,81 | signe reproduit, compatible |
| Quality | RMW | 0,21 | **+0,19** (+0,80) | 0,64 | 4,12 | signe reproduit, compatible |
| Size | SMB | 0,20 | **−0,29** (−1,19) | 2,57 | 3,26 | signe **non** reproduit, **non** compatible |
| Momentum | UMD | 0,16 | **−0,21** (−0,85) | 3,66 | 2,35 | signe **non** reproduit, compatible |
| — | CMA | — | +0,50 (+1,96) | — | 2,98 | pas d'équivalent |
| — | marché | — | +0,37 (+1,51) | — | 1,26 | pas d'équivalent |

**Verdict d'ensemble : PARTIELLEMENT REPRODUIT en signe (2 sur 4).** La corrélation entre
les six stratégies est faible, comme dans le papier : +0,13 en moyenne (de +0,05 à +0,35 ;
le papier : 0,05 à 0,48). Aucun Sharpe, publié ou reproduit, n'est significatif sur 17,5
ans : le plus grand t est 1,96 (CMA), avant toute correction.

**Ce qui n'est pas répliqué** : les indices long seulement (MSCI, Russell), l'étape
Black-Litterman et ses cibles d'écart de suivi, le VIX et le taux 2 ans, le réglage de κ,
la réestimation mensuelle et les coûts de 5 pb. B2 n'est qu'un analogue en espace de
facteurs.

---

## 6. Ce qu'on peut dire en présentation

1. « On a donné à chaque facteur **son propre régime** : six modèles, 22 changements
   d'état par an, des régimes qui ne sont pas des mesures de volatilité. C'est
   exactement ce qui manquait aux sept dispositifs précédents. »
2. « Sur 48 ans, ça **ne bat pas le simple fait de tenir les six facteurs** : 1,39 contre
   1,51. Ça divise la perte maximale par deux, mais une règle d'une ligne (le facteur
   a-t-il monté sur un an ?) fait pareil. »
3. « **Sur la fenêtre du papier, ça a l'air de marcher** (+0,29). Avant 1990, ça perd
   (−0,63). Un résultat publié sur 17 ans peut n'être qu'une bonne fenêtre, et c'est
   pour ça qu'on a allongé l'échantillon avant de lire. »
4. « Ce qui rapporte, ici, c'est **la diversification entre facteurs**, pas le régime. »
   Réserve à dire en même temps : ce 1,51 est un Sharpe sans coût, sur des portefeuilles
   théoriques de Ken French (§7).

La diapositive tient en un tableau : les lignes B1 et B2 du §3, plus la ligne de la
fenêtre du §4.

---

## 7. Limites

- **Portefeuilles théoriques, coût nul.** Les facteurs de Ken French ne sont pas
  investissables tels quels (jambes courtes, petites capitalisations, rebalancements),
  et le coût est nul par décision du projet. Dans les 8 tests, la rotation annuelle des
  expositions va de 3,7 (B1) à 9,0 (HML) ; le Sharpe de 1,51 du livre tenu ne survivrait pas intact
  aux coûts, et RMW et CMA ont été découverts dans les années 2010 sur une partie de
  cet échantillon.
- **Réplication partielle par construction** : autres actifs (long-short au lieu
  d'indices long seulement), substituts pour le VIX et le taux 2 ans, κ fixé à 9,5,
  réestimation semestrielle, pas de Black-Litterman.
- **Puissance.** Même sur 48,6 ans, le seuil de détection vaut 0,506 pour B1 et 0,54 à
  1,03 pour les autres tests : un livre qui coupe la moitié de ses jambes s'éloigne
  beaucoup du livre qui les garde toutes. Les deltas positifs (taille, valeur,
  investissement) sont donc **sous-puissants**, jamais des succès.
- **Une seule graine**, une seule grille ; le bas de la grille est souvent retenu (§4).
- La causalité de l'effet de fenêtre (pourquoi 2007-2024 est favorable) **n'est pas
  vérifiée** ici. Elle est seulement constatée.

---

## 8. Écarts au protocole

1. **Aucun écart sur les tests, les seuils, les témoins ou le verdict.** Le code lu est
   celui commité avec le pré-enregistrement (`1e812bf`).
2. La première passe **synthétique** (`--synthetic --fit`) a été lancée avec une version
   du script antérieure au commit `1e812bf` : l'ajustement et le réglage y étaient une
   seule fonction. Le réglage synthétique a été rejoué avec le code commité
   (`--synthetic --select`) avant l'instrument et la lecture synthétiques. Aucune donnée
   réelle n'était en jeu.
3. Le message du commit de l'instrument affirmait à tort que le script avait changé
   depuis `1e812bf`. Il a été corrigé (le commit `56799e5` est devenu `4f1555c`,
   contenu identique) avant tout autre commit de cette piste.
4. Le PRESPEC annonçait un échantillon de test « début janvier 1978 » : c'est le
   05/01/1978.

---

## Reproduire

```bash
.venv/bin/python scripts/factorsjm_fetch.py        # Ken French 1963-2026, FRED DGS1 et DGS10
.venv/bin/python scripts/factorsjm_run.py --fit    # plusieurs heures : 36 chemins candidats, puis le réglage
.venv/bin/python scripts/factorsjm_run.py          # l'instrument, aucun rendement conditionné imprimé
.venv/bin/python scripts/factorsjm_run.py --read   # refusé : la lecture a déjà eu lieu
```

Tests : `tests/test_factorsjm.py` (18 tests, données synthétiques).
