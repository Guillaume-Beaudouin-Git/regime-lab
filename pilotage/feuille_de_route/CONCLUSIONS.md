# Ce que le programme a établi

État au 21 septembre 2026. Chaque chiffre de la section 3 provient d'une commande
exécutée pendant cette session, sur les venvs des dépôts, sans rien écrire dans
les dépôts. Les chiffres des sections 1 et 2 sont ceux des documents du programme
et sont cités comme tels.

---

## 1. Ce qui est vivant

**Le classifieur de régimes marche, comme classifieur.** A′ sparse jump : 93,2 %
d'exactitude équilibrée contre les récessions NBER, kappa 0,53, sur 6 377 jours
hors échantillon. Cinq méthodes s'accordent (kappa 0,28-0,83). Partitions stables
entre réestimations à 73-137 fois le nul. Latence réelle 0-13 jours. Un audit
indépendant a rejoué chaque script sans trouver de fuite, et `run_phase2.py`
relancé depuis un clone frais rend un `states.parquet` bit-identique.

**Ce qu'il porte est de la variance.** +3,93 points de R² incrémental sur la
volatilité future au-delà d'un quantile de volatilité (t −3,40), contre +0,030
point sur les rendements (t 0,27). Le placebo d'une ligne fait +0,004 (t 0,21).
C'est un signal de dimensionnement, pas de timing, et c'est mesuré, pas supposé.

**Les quatre falsifications tiennent.** H1 (momentum macro absolu, un pays,
données réalisées) : Sharpe −0,38, carte des signes au 7ᵉ percentile de son
placebo. H2 (transversal 19 pays) : zéro cellule sur neuf au-dessus de |t| 1,96,
et la cause est identifiée — l'écart-type des taux courts des neuf pays de la
zone euro passe de 2,35 % avant 1999 à exactement 0,00 % après. H3 (surprises
macro, noyau exponentiel) : aucun des quatre actifs n'atteint le seuil de Šidák.
Reversal : prime de 3,64 de Sharpe dans les années 90 à −0,18 depuis 2020.

Aucune de ces quatre n'est un backtest raté. Chacune est une mesure conçue pour
pouvoir dire non, qui l'a dit, sur un dispositif écrit avant la donnée. H2 est le
plus fort des quatre parce que sa cause est datée et institutionnelle : la
stratégie n'a pas décayé, sa matière première a été supprimée par traité.

**Les estimateurs de persistance sont validés.** Le DFA retrouve un H connu à
±0,01 près sur bruit gaussien fractionnaire exact ; la correction Lo-MacKinlay
efface le biais du ratio de variance naïf (0,813 → 0,999 sur marche aléatoire,
taux de rejet 5,2-7,8 % contre 5 % nominal). Ce sont des outils réutilisables.

**Le ciblage de volatilité sans paramètre gagne tous les face-à-face.** Cinq
arrivées indépendantes, cinq fois le même ordre.

---

## 2. Ce qui est mort

Nommé pour qu'on n'y revienne pas.

**Le régime comme signal de timing.** Couche 3, onze règles : au mieux 0,57 de
Sharpe contre 0,47 pour le 60/40, chaque écart sous le seuil de détection
0,271-0,399. La règle d'arrêt n° 2 du cadrage se déclenche et c'est déclaré.

**Le régime comme conditionneur de livre.** T1 : aucune famille ne bat une règle
d'une ligne (vol réalisée sous sa médiane expansive, 0,51 de Sharpe et −10,3 %
de perte maximale contre 0,47 et −35,6 %), 0 pli sur 5 contre un seuil de 3 ; il
faudrait 2 246 à 7 722 ans. T3 : le dispositif est au 89ᵉ-97ᵉ percentile en
Sharpe et alpha mais au 0ᵉ en bêta — l'alpha passe par le dénominateur.

**L'atténuateur de Carver au niveau du livre.** 0,58 contre 0,68 pour la cible de
vol sans paramètre. Il passe son placebo au 99ᵉ percentile et perd contre un
témoin. Son écart de +0,144 est sous le MDE de 0,205 : sous-puissant, pas un
succès.

**Le régime dans la géométrie de barrière propfirm.** Δp(pass) +0,28 point contre
un MDE de 6,10, au 16ᵉ percentile de son placebo, et la cible de vol sans
paramètre allonge la survie de 60 sessions contre 36 pour le régime. Le mécanisme
est arithmétique : p(pass) monte de 0,2785 par unité d'exposition, le bras régime
porte 1,0251 d'exposition moyenne, donc 0,0251 × 0,2785 = +0,70 point attendu par
l'exposition seule, pour +0,28 observé — résidu −0,42. Cause profonde : 89 % des
comptes qui échouent vivent et meurent dans un seul état. Treize transitions en
6 377 jours contre une durée de vie médiane de 90 sessions.

**Le compteur de facteurs effectifs.** Tous les |t| sous 1,6. Une version
antérieure donnait 2,18 : elle exigeait que les 46 instruments aient tous des
données à chaque date, ce qui démarrait l'échantillon en 2007. Le signal apparent
était la troncature.

**La sélection d'actifs par exposant de Hurst.** Sur 58 actifs, 3 rejets de la
marche aléatoire quand le hasard en donne 2,9. Sur 49 secteurs, trier par Hurst
donne 0,35 contre 0,33 selon le tercile.

**La prime de retournement court terme.** Kill mécanique, gross ≈ 0. Fermé.

**Le transversal macro G10 après mars 1999.** Kill mécanique. La dispersion
n'existe plus.

**Le critère Δp(pass) comme instrument d'admission.** Tué deux fois, des deux
côtés, indépendamment. Dans `regime-lab`, l'étude de barrière montre que p(pass)
s'achète avec l'exposition moyenne et rien d'autre. Dans le dépôt privé voisin, le
chantier d'admission marginale du 2 août 2026 le montre par un contrôle nul
direct : une série à moyenne exactement nulle et à persistance de 10 jours achète
**Δp(pass) = +14,95 points**. Sur 79 séries testées, un seul candidat survit à son
placebo décalé, et c'était un edge déjà trouvé par une autre voie.

---

## 3. Ce que cette session a mesuré, et qui n'était pas dans les documents

Six constats nouveaux, tous issus de commandes exécutées ici. Ils ne modifient
aucun résultat gelé : ils portent sur le **véhicule** et la **construction** du
livre de tendance des extensions, que personne n'avait audités.

### 3.1 L'univers de 46 instruments satisfait vraiment la règle N ≥ 30

`data/cache/trend_universe.parquet` : 6 822 sessions, 2000-07-17 à 2026-09-10.

| seuil | première date |
|---|---|
| ≥ 30 instruments cotés | 2002-07-30 |
| ≥ 40 instruments cotés | 2003-12-05 |
| 46 instruments cotés | 2007-12-28 |
| ≥ 30 avec 252 jours d'historique (donc tradables en tendance) | **2003-07-17** |

Soit **23,2 ans** d'échantillon conforme à la règle N ≥ 30. Composition :
13 indices actions, 13 futures de matières premières, 11 paires de change au
comptant, 9 ETF obligataires et de crédit.

La largeur est réelle et pas cosmétique. Sur l'échantillon commun 2008+ (4 612
sessions, 46 colonnes sans trou) : corrélation moyenne par paire **0,098**, ratio
de participation **11,4**, nombre effectif de paris par entropie du spectre
**20,7**, première valeur propre à 22,5 % de la variance, 24 valeurs propres pour
atteindre 90 %. C'est la structure d'un petit livre de futures gérés.

**C'est le seul endroit du programme où la règle N ≥ 30 est satisfiable, et
l'univers n'a jamais servi qu'à porter un overlay.**

### 3.2 Onze des 46 instruments portent des rendements datés d'une session trop tard

Les séries de change au comptant de yfinance sont décalées d'une session. Testé
contre les futures CME correspondants du dépôt privé voisin, sur 3 528 sessions
communes :

| série au comptant | future | corr(t, t) | corr(t+1, t) |
|---|---|---|---|
| EURUSD=X | 6E | 0,0572 | **0,8903** |
| GBPUSD=X | 6B | 0,1082 | **0,8956** |
| AUDUSD=X | 6A | 0,0637 | **0,8766** |

Vérification sur un évènement daté. Le référendum britannique a frappé le marché
le vendredi 24 juin 2016 : le future 6B inscrit −8,10 % ce jour-là, la série au
comptant inscrit −1,58 % ce jour-là et −7,91 % le lundi 27.

Les indices actions, eux, sont correctement alignés : ^GSPC contre ES donne 0,9709
en même jour et −0,1213 en décalé ; même chose pour ^NDX et ^RUT.

Deux conséquences, et la seconde est rassurante.

1. **Le signal de change est périmé de deux jours.** Un livre qui lit le prix en
   T−1 lit en réalité un évènement de T−2. Le décalage va dans le sens
   conservateur — il n'y a **aucune fuite d'information** — mais la jambe change
   du livre a travaillé sur de l'information éventée pendant vingt-cinq ans.
2. **La structure de risque n'est presque pas touchée.** Les onze séries décalent
   ensemble, donc leurs corrélations mutuelles sont préservées (−0,0328 avant
   comme après réalignement). Le nombre effectif de paris passe de 20,65 à 20,55,
   la corrélation moyenne de 0,0977 à 0,1023. La largeur mesurée en 3.1 tient.

Trois des onze sont vérifiées contre un future ; les huit autres viennent du même
point d'accès et sont **inférées, non vérifiées**.

### 3.3 Les métaux ne sont pas back-ajustés

Comparé aux séries back-ajustées quotidiennes du dépôt privé voisin
(back-ajustement proportionnel, validé contre Databento à 0,975 de corrélation
hebdomadaire) :

| regime-lab | corr. avec back-ajusté | corr. avec front-month brut |
|---|---|---|
| GC=F | 0,9843 | **1,0000** |
| SI=F | 0,9905 | **1,0000** |
| CL=F | 0,9746 | 0,9301 |

GC=F et SI=F sont des séries front-month **brutes**. Ampleur de la contamination,
mesurée sur CL entre 2012 et 2026 : sur les 134 jours de roulement, le rendement
absolu moyen est de 2,324 % en brut contre 1,648 % en back-ajusté, et la somme
des rendements de jours de roulement vaut **−22,4 % en brut contre −13,9 % en
back-ajusté** — soit 8,5 points de dérive parasite injectés sur 134 jours.

C'est la leçon n° 5 du `CLAUDE.md` du programme, retrouvée dans le dépôt.

### 3.4 Le livre de référence n'a pas de ciblage de volatilité

`regime_lab/extensions/trend.py` renormalise le brut à 1 chaque jour :

```python
weights = raw.div(raw.abs().sum(axis=1).replace(0.0, np.nan), axis=0)
```

Le risque par instrument est bien équipondéré (proportionnel à 1/σ), mais la
volatilité du **portefeuille** flotte librement : la volatilité annualisée
glissante à 63 jours va de 0,00 % à 15,14 % pour une moyenne de 4,54 %.

Le `CLAUDE.md` du programme pose « ciblage de volatilité quotidien » parmi ses
règles dures, et `EXTENSIONS.md` qualifie explicitement la renormalisation du
brut de « mauvaise construction » — mais pour l'atténuateur seulement. Le livre
de référence lui-même n'a jamais été audité sur ce point. **La règle dure n'est
pas respectée par le témoin de toutes les comparaisons du programme.**

Et le vainqueur de tous les face-à-face — la cible de vol dynamique sans
paramètre — est exactement le morceau manquant remis par-dessus.

### 3.5 Les Sharpe des extensions sont bruts, et le coût en mange la moitié

Le livre de référence, tel que publié : Sharpe **0,510**, volatilité 4,54 %,
rendement annuel brut **2,315 %**, rotation brute **14,79 fois le livre par an**.
Le point mort est à **15,7 points de base** aller-retour.

Appliqué le barème de coûts du programme lui-même, par classe d'actif :

| classe | n | rotation annuelle | bp AR | dérive %/an |
|---|---|---|---|---|
| change au comptant | 11 | 5,13 | 15,0 | **0,770** |
| ETF obligataires | 9 | 3,85 | 7,5 | 0,289 |
| futures matières premières | 13 | 2,72 | 1,5 | 0,041 |
| indices actions | 13 | 3,08 | 1,0 | 0,031 |
| **total** | 46 | 14,79 | — | **1,130** |

Le coût mange **48,8 %** du rendement brut. Le Sharpe net tombe à **0,261**. Les
deux tiers de la dérive viennent des onze paires de change facturées au tarif du
comptant de détail.

Et les Sharpe sont bruts, pas en excess. Le livre porte une exposition nette
directionnelle moyenne de **+0,268** ; le taux à trois mois moyen sur
l'échantillon est de 1,73 %. En finançant cette exposition nette, 0,520 devient
**0,413**. Si le brut entier devait être financé — ce qui est le cas pour les
neuf ETF — il tombe à **0,145**.

### 3.6 Le véhicule décide du verdict, et personne ne l'avait testé

Le seul dispositif que le programme n'ait jamais réussi à battre est la cible de
vol dynamique sans paramètre : `trend × (σ_pleine / σ_63j décalée)`. Sa
performance brute se reproduit ici au chiffre près — écart +0,244, t +2,91 contre
+0,244 et t +2,89 publiés.

Chargée des coûts, sur l'échantillon commun :

| barème de coûts | Sharpe net du livre | Sharpe net avec cible de vol | écart | t |
|---|---|---|---|---|
| mélange de détail actuel (change 15 bp, ETF 7,5 bp) | 0,163 | 0,269 | **+0,106** | **+1,23** |
| véhicule tout-futures, 1,0 bp AR | 0,410 | **0,638** | +0,228 | +2,71 |
| véhicule tout-futures, 2,0 bp AR | 0,375 | 0,586 | +0,211 | +2,51 |
| futures + change/obligataire à 5 bp | 0,316 | 0,498 | +0,182 | +2,15 |

Le MDE publié sur cette comparaison est de 0,205 ; le MDE recalculé ici par
bootstrap par blocs sur l'échantillon actif (6 569 sessions, 2001-07-05 à
2026-09-10) est de 0,273 à 21 jours de bloc, 0,264 à 63, 0,246 à 126, pour un
écart observé de +0,250.

Lecture. **Sur le véhicule réellement impliqué par l'univers tel qu'il est
stocké, le seul dispositif que le programme n'a jamais pu battre ne se distingue
plus de son témoin** : +0,106 pour un MDE de 0,205-0,273. Il ne survit que sur un
véhicule futures, où il rend +0,21 à +0,23 pour t 2,5 à 2,7.

Le ciblage de volatilité n'est donc pas un socle inconditionnel. C'est un effet
dont l'amplitude nette est du même ordre que le coût de transaction, et le choix
du véhicule le décide. À ma connaissance ce point n'est écrit nulle part dans le
programme, et il est nouveau.

---

## 4. Ce qu'il faut retenir pour décider

Trois faits commandent la suite.

**Le programme a toujours mesuré des overlays et jamais le livre.** Les cinq
réfutations portent toutes sur le conditionnement par le régime. Le livre
sous-jacent, lui, n'a jamais été audité : il est brut, sans ciblage de volatilité
de portefeuille, avec 24 % de ses instruments décalés d'une session et une partie
de ses matières premières non back-ajustées. Ce n'est pas une critique du travail
accompli — le livre était un véhicule d'essai, pas un sujet — mais cela veut dire
que la meilleure information disponible sur le livre lui-même a été produite
aujourd'hui.

**Le seuil est proche.** Sur véhicule futures à 1 bp, l'univers de 46 instruments
avec ciblage de volatilité rend 0,638 net, sur 23,2 ans conformes à N ≥ 30, sans
qu'aucune réparation ait été faite. RESEARCH_PASS demande 0,7. L'écart est de
0,06 de Sharpe. C'est la seule marge de ce genre dans tout le programme ; partout
ailleurs on est à 0,5 contre 0,47 avec un MDE de 0,3.

**La puissance suffit, tout juste.** Sur 23,2 ans, l'erreur type d'un Sharpe seul
vaut 0,232 pour un vrai Sharpe de 0,7 ; il faut 19,9 ans pour résoudre 0,7 contre
zéro à 80 % de puissance, et on en a 23,2. Un Sharpe vrai de 0,7 est donc
**détectable sur cet échantillon** — ce qui n'était vrai d'aucune des questions
posées jusqu'ici, toutes sous-puissantes d'un facteur deux à quatre. La question
change de nature parce qu'on cesse de mesurer un écart entre deux livres corrélés
pour mesurer un livre.

---

## 5. La réponse à « est-ce que ça marche »

⚠ **CORRIGÉ LE 22/09 — la phrase qui suivait généralisait au-delà des mesures.**

Ce que les six réfutations portent, à sa largeur réelle : un classifieur **ordonné par la
volatilité d'entraînement**, qui change d'état **0,514 fois par an**, utilisé comme
**multiplicateur de taille ou interrupteur au niveau du portefeuille**, sur un **livre de
tendance unique de dimension effective 3,86**, n'apporte rien au-delà d'une volatilité
réalisée sous sa médiane expansive.

Ce qu'elles ne portent pas : rien sur un état à plus d'une transition par an, rien sur une
latente qui ne soit pas la volatilité, rien sur un usage de **sélection entre signaux**, de
**construction de portefeuille** ou d'**exécution**, rien sur un objet conditionné de
dimension effective supérieure à quatre. Les cinq plans de `pilotage/plans_de_recherche/` franchissent
chacun au moins un de ces axes, mesuré et non affirmé — le détecteur de ruptures d'AHL produit
**11,04 ruptures par an contre nos 0,514**.

Le texte d'origine, conservé parce qu'il reste exact pour la construction testée :

> Le régime ne marchera pas. Cinq dispositifs, cinq réfutations, une cause commune
mesurée — treize transitions en vingt-cinq ans donnent treize décisions, ce qui
n'est pas un signal de trading. Rouvrir ce dossier serait un sixième essai qui
déflaterait les cinq premiers sans rien apporter.

Ce qui peut marcher est ce que le programme a passé deux ans à utiliser comme
témoin sans jamais l'examiner : un livre multi-actifs large, sur un véhicule
correct, dont le sujet est le dimensionnement. Les chiffres disent qu'il est à
0,06 de Sharpe du seuil, sur le seul échantillon du programme qui ait la
puissance de trancher. C'est là qu'est la masse de probabilité restante.

Le détail est dans `PISTES.md`, le pré-enregistrement dans
`PRESPEC_TREND_VEHICLE.md`.
