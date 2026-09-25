# Partie 1 de la présentation : notre cheminement, et les modèles expliqués simplement

Ce document sert à préparer l'oral. Il doit permettre de raconter le projet étape par
étape, sans jargon, à quelqu'un qui n'est pas du métier : « Xavier de la compta ». Tous
les chiffres viennent des documents du dépôt ; les exemples datés sont lus dans
`data/cache/states.parquet`.

---

## 1. L'idée en une phrase

> Les marchés alternent entre des périodes calmes et des périodes de crise. Nous avons
> cherché à **reconnaître en temps réel dans quel « mode » est le marché**, puis à savoir
> si ça permet de mieux investir.

**L'image à utiliser** : la météo. Il y a le beau temps et la tempête. Un bon modèle de
régime, c'est un baromètre qui dit « on est entré dans la tempête » *pendant* qu'elle
arrive, pas le lendemain.

---

## 2. Le cheminement, en six étapes

### Étape 1 — Point de départ : le HMM, parce que la littérature du cours y menait

Trois des quatre articles de référence du cours portent sur les *Hidden Markov Models*
(HMM) :
- *Hidden Markov Model for Stock Trading* ;
- *Hidden Markov Models in Finance* ;
- *High-order Hidden Markov Model for trend prediction in financial time series*.

Le quatrième porte sur l'exposant de Hurst, et nous l'avons gardé comme l'une de nos
variables. Le HMM était donc le point de départ naturel.

**À dire** : « La littérature du cours pointait vers un modèle classique, le HMM. On est
partis de là. »

### Étape 2 — En creusant, on a trouvé quatre faiblesses au HMM

Elles sont écrites dans notre cadrage (`docs/CHARTER.html` §02), comme des hypothèses à
vérifier :
1. **Il triche facilement.** La version standard utilise des données futures pour
   étiqueter le passé, ce qui est impossible en temps réel. En version honnête
   (« filtrée »), il est beaucoup moins tranché.
2. **Il redécouvre souvent la volatilité, rien de plus.** « Marché agité / marché calme »,
   ce qu'une règle d'une ligne sait déjà faire.
3. **Il réagirait en retard**, précisément au moment où ça compte.
4. **Il est instable** : réestimé deux fois, il peut donner deux découpages différents.

**Ce que nos données ont montré sur le point 3** : le HMM filtré n'a pas été en retard.
Au Covid, il passe en stress le 27 février 2020, deux semaines avant le Sparse Jump Model.
Son vrai défaut ici, ce sont les **fausses alertes** : il est en stress 35 % du temps,
change d'avis 49 fois en 24 ans, et seuls 19 % de ses jours de stress tombent dans une
récession officielle.

### Étape 3 — Un papier de recherche plus récent propose mieux : le Jump Model

Le *Statistical Jump Model* vient de Nystrup, Kolm et Lindström (2020-2021). Aydınhan,
Kolm, Mulvey et Shu (2024) l'étendent, et la littérature lui prête des régimes plus stables et
plus exploitables que le HMM. Son idée clé : **chaque changement d'état a un coût**,
donc le modèle ne change d'avis que si la preuve est forte.

Nous avons pris sa version *sparse* : elle **choisit elle-même au plus 10 variables**
parmi nos 50, et en ignore le reste. Nous avons aussi mis parmi les 50 des variables qui
ne sont pas de la volatilité (crédit, taux, dispersion entre secteurs…), pour qu'il ne
puisse pas se contenter de redécouvrir la volatilité.

**À dire** : « On a trouvé un modèle plus récent, le Jump Model, qui fait payer chaque
changement d'avis. Il est plus stable, et il choisit lui-même ses indicateurs. »

### Étape 4 — Pour être honnêtes, on l'a mis en concurrence

Un modèle ne vaut que comparé. Nous avons aligné cinq concurrents, avec les mêmes 50
variables (le HAR-RV utilise ses propres termes), les mêmes règles et les mêmes dates :
- le **HMM**, notre point de départ ;
- le **Jump Model** simple et le **Sparse Jump Model** ;
- deux modèles **sans état caché**, qui prédisent directement la volatilité : un modèle
  de *machine learning* (gradient boosting) et une formule classique (HAR-RV) ;
- et surtout un **témoin d'une ligne** : « stress si la volatilité récente dépasse sa
  médiane ». Si un modèle sophistiqué ne bat pas ça, il ne sert à rien.

### Étape 5 — La comparaison : le Sparse Jump Model gagne comme détecteur

| | Exactitude face aux récessions | Accord (kappa) | Changements d'état par an | Informe sur la volatilité future ? |
|---|---|---|---|---|
| **Sparse Jump Model** | **93,2 %** | **0,53** | **0,5** | **oui, +3,93 pts** |
| Jump Model | 93,3 % | 0,49 | 0,5 | oui, +3,47 pts |
| HMM | 84,7 % | 0,24 | 2,0 | oui, +2,26 pts |
| Gradient boosting | 75,0 % | 0,12 | 14,1 | oui, +2,13 pts |
| HAR-RV | 78,2 % | 0,17 | 18,4 | presque pas, +0,19 pt |

Sources : `docs/RESULTS_FINAL.md`, couches 1 et 2 ; les changements d'état sont comptés
dans `data/cache/states.parquet` sur les 6 377 séances hors échantillon (24,4 années
civiles). L'exactitude est
l'exactitude équilibrée (§4). La dernière colonne donne le R² incrémental au-delà du
témoin, sur la volatilité à 21 jours.

**À dire** : « Le Sparse Jump Model atteint 93 % d'exactitude face aux récessions
officielles, sans jamais voir le futur : il met en stress 96 % des jours de récession et
laisse en calme 90 % des jours normaux. Et il ne change d'avis qu'une fois tous les deux
ans. Les modèles plus nerveux se trompent plus souvent. »

**À dire aussi, sans quoi le chiffre trompe** :
- **Le 93 % repose sur deux récessions seulement.** La période de test (2002-2026) ne
  contient que deux récessions officielles : 2008-2009 et 2020, soit 20 mois.
- **« En temps réel » veut dire sans voir le futur, pas avant tout le monde.** Pendant le
  Covid, le modèle passe en stress le 11 mars 2020 : le VIX, l'indice de la peur, était
  déjà passé de 14 à 54.

### Étape 6 — La découverte qui compte : il prévoit le risque, pas la direction

- Sur la **volatilité** des semaines suivantes, le modèle apporte une vraie information
  au-delà du témoin (la règle d'une ligne sur la volatilité passée) : +3,93 points de R²,
  statistiquement solide (t = −3,4).
- **Mais le VIX le savait déjà.** Une fois le VIX pris en compte, le modèle n'ajoute plus
  que +0,20 point, un gain non significatif (`docs/RESULTS_CRISE.md`, test P). Le marché
  des options anticipait déjà ce que le modèle détecte.
- Sur la **direction** du marché, hausse ou baisse, il n'apporte rien : +0,03 point,
  t = 0,27.

Nous avons aussi refait à l'identique le papier de Shu et al. (2024), pour vérifier.
**Le risque se reproduit, le rendement non** : le S&P 500 conservé se reproduit au
chiffre près (Sharpe 0,48 des deux côtés) et le modèle réduit le risque comme dans le
papier, mais sans gain de rendement. Même le chiffre publié (Sharpe 0,68) ne dépasse une
simple cible de volatilité (0,61) que de 0,06, sous notre seuil de détection
(`docs/REPLICATION_SHU2024.md`).

**À dire, c'est la transition vers la partie 2** : « Notre modèle est un bon thermomètre
du risque, pas une boule de cristal. La question devient : quelles stratégies ont besoin
d'un thermomètre du risque ? »

---

## 3. Chaque modèle en une ou deux phrases, avec un exemple

### Le HMM (Hidden Markov Model)

**En une phrase** : le marché a un « état » qu'on ne voit pas directement. Le modèle le
devine à partir de ce qu'on observe, et calcule chaque jour la probabilité d'être en
stress.

**Image** : un médecin qui ne voit pas le virus, mais déduit la grippe de la fièvre et de
la toux. Il sait aussi qu'une grippe dure en général quelques jours : c'est la
« probabilité de transition ».

**En vrai** :
- il a basculé en stress le **27 février 2020**, le S&P 500 était déjà à −12 % ;
- il a passé **93 % de 2022** en stress ;
- il change d'avis environ **2 fois par an**.

### Le Jump Model

**En une phrase** : il range les jours en deux paquets, calme et stress, selon leur
ressemblance, mais il paie une **amende** à chaque changement de paquet. Il ne bascule
donc que si les indices sont nets.

**Image** : un arbitre qui ne siffle que les fautes évidentes. Ou un thermostat qui
n'allume pas le chauffage à chaque courant d'air.

**En vrai** : il change d'avis **une fois tous les deux ans** en moyenne, et un état dure
456 jours en moyenne.

### Le Sparse Jump Model (notre modèle principal)

**En une phrase** : le Jump Model, qui en plus **choisit lui-même au plus 10
indicateurs** parmi les 50, et ignore les autres.

**Image** : un analyste qui, face à un tableau de bord de 50 voyants, n'en surveille
qu'une dizaine au plus, ceux qui comptent vraiment.

**En vrai** :
- **13 changements d'état en 24 ans** ;
- en stress dès **novembre 2007 puis janvier 2008**, alors que le NBER n'a annoncé
  qu'en **décembre 2008** que la récession avait commencé en décembre 2007 ;
- en stress le **11 mars 2020**, le jour où l'OMS déclare la pandémie (S&P 500 à −19 %,
  le creux sera à −34 % le 23 mars) ;
- **calme pendant toute l'année 2022**. C'est sa limite : il n'a pas classé la baisse
  de 2022 comme une crise.

### Le gradient boosting (machine learning, LightGBM)

**En une phrase** : pas d'état caché. On **apprend directement** à prédire la volatilité
du mois suivant, avec des centaines de petits arbres de décision du type « si le VIX
dépasse tel seuil et que le crédit se tend, alors… ». Au-dessus de la médiane, on dit
« stress ».

**Image** : un comptable qui a vu des milliers de cas passés et applique des règles
empiriques.

**En vrai** : très réactif, mais nerveux. Il a changé d'avis **4 fois en février 2020**,
et environ **14 fois par an** en moyenne.

### Le HAR-RV

**En une phrase** : une **formule simple** et classique. La volatilité de demain est un
mélange de celle d'hier, de la semaine passée et du mois passé.

**Image** : prévoir la température de demain avec la moyenne d'hier, de la semaine et du
mois.

**En vrai** : **26 changements d'avis en 2008** à lui seul, 18 par an en moyenne. C'est
la référence de la littérature pour prévoir la volatilité, pas pour dater des régimes.

### Le témoin d'une ligne

**En une phrase** : « on est en stress si la volatilité des dernières semaines dépasse sa
médiane historique ». Pas de modèle, pas de paramètre.

**Pourquoi il est central** : c'est la barre à franchir. Tout ce qui ne le bat pas n'est
qu'une volatilité déguisée.

### Ce qui les distingue vraiment

| | Devine un état caché ? | Comment il change d'avis | Tempérament |
|---|---|---|---|
| HMM | oui | selon des probabilités de passage | moyen, 2 fois par an |
| Jump Model | oui | seulement si ça vaut l'amende λ | calme, 0,5 fois par an |
| Sparse Jump | oui | idem, avec au plus 10 indicateurs | calme, 0,5 fois par an |
| Gradient boosting | non, prédit la volatilité | dès que la prévision passe la médiane | nerveux, 14 fois par an |
| HAR-RV | non, prédit la volatilité | idem | très nerveux, 18 fois par an |

**Le compromis à retenir** : plus un modèle est calme, mieux il date les vraies crises
et moins il coûte en transactions. Mais il rate les épisodes moyens, comme 2022. Plus il
est nerveux, plus il voit tôt, mais plus il se trompe (fausses alertes).

---

## 4. Les mots à maîtriser, en une phrase chacun

- **Régime / état** : le « mode » du marché, ici deux modes, calme ou stress.
- **Temps réel, ou filtré** : le modèle n'utilise que ce qui était connu ce jour-là,
  jamais la suite.
- **Point-in-time** : même règle pour les données. Un chiffre macro n'est utilisé qu'à
  partir de sa date de publication, dans sa première version.
- **Hors échantillon** : on juge le modèle sur des années qu'il n'a pas vues pendant son
  apprentissage.
- **Exactitude équilibrée** : la moyenne de deux taux, la part des jours de récession
  classés en stress et la part des jours normaux classés en calme. Les deux comptent
  autant, même si les récessions sont rares. 50 % correspond au hasard. Pour le Sparse
  Jump Model : (96,3 % + 90,0 %) / 2 = 93,2 %.
- **Kappa** : l'accord avec la réalité une fois retiré ce qu'on aurait trouvé par hasard.
  0 = hasard, 1 = parfait.
- **R² incrémental** : ce qu'un indicateur ajoute à la prévision, *en plus* de ce qu'on
  savait déjà.
- **Ratio de Sharpe** : le rendement par unité de risque. Au-dessus de 1, c'est très bon.
- **Perte maximale (drawdown)** : la pire chute, d'un sommet au creux suivant.
- **Sous-puissant** : l'écart mesuré est trop petit pour être distingué du hasard sur
  cette durée. Ce n'est ni un oui ni un non.

---

## 5. Les questions probables, et la réponse courte

- **« Pourquoi ne pas juste regarder le VIX ? »** C'est la bonne question, et la réponse
  honnête est : pour prévoir la volatilité, le VIX suffit presque. Le modèle bat une règle
  simple sur la volatilité passée (+3,93 points de R²), mais au-delà du VIX il n'ajoute
  que +0,20 point, non significatif. Son intérêt est ailleurs : il date les crises de
  façon stable et lisible, et il n'a pas besoin d'un marché d'options.
  *(Corrigé le 23/09 : une version précédente disait à tort que le VIX était notre
  témoin.)*
- **« Si le modèle marche, pourquoi ne gagne-t-il pas d'argent ? »** Il prévoit
  l'**ampleur** des mouvements, pas leur **sens**. Or la plupart des stratégies gagnent
  sur le sens. Il faut donc des stratégies qui gagnent ou perdent sur l'ampleur : c'est
  la partie 2.
- **« Pourquoi 2022 n'a pas été détecté ? »** Le modèle est réglé pour ne signaler que
  les crises nettes : il a changé d'état 13 fois en 24 ans. 2022 était une baisse lente
  due aux taux, pas une panique. Les modèles plus nerveux l'ont vue, mais ils se trompent
  plus souvent ailleurs.
- **« Le modèle a-t-il triché en voyant le futur ? »** Non. Les données sont prises à
  leur date de publication, le modèle est réestimé tous les six mois sur le passé seul,
  et nos critères ont été écrits avant les résultats.
- **« Pourquoi pas plus de deux états ? »** Deux états suffisent à la question calme
  contre crise, et chaque état en plus multiplie les paramètres à estimer.

---

*Corrigé le 25/09/2026 : le « 93 % » est une exactitude équilibrée, pas une part de
récessions reconnues ; le Sparse Jump Model garde au plus 10 variables, pas exactement
10 ; le « retard » du HMM était une hypothèse du cadrage, que nos données démentent ; la
phrase « investi 77 % du temps » venait d'une ligne retirée de la réplication de Shu le
22/09 ; deux comptes de changements d'état sont recalculés sur `states.parquet`.*
