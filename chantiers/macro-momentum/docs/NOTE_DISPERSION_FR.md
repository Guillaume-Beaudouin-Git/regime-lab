# Le momentum macro des pays développés, et la matière première qu'un traité a supprimée

Note pour le mémoire. Elle se lit sans le code sous les yeux : chaque chiffre est
donné avec ce qui l'a produit, et les quatre corrections que la vérification a
imposées sont dans le texte, pas en annexe.

---

## De quoi il s'agit

Il existe une stratégie publiée qui porte un nom et un chiffre. AQR annoncent un
ratio de Sharpe de 1,2 sur 1970-2016 pour un portefeuille qui achète les pays
dont la croissance et l'inflation s'améliorent et vend ceux où elles se
dégradent. La question posée ici est simple : ce chiffre est-il atteignable avec
des données gratuites et une discipline qui interdit de regarder l'avenir.

Deux tentatives ont été construites. Elles ont échoué toutes les deux, et
l'intérêt du travail est qu'elles ont échoué pour des raisons sans rapport entre
elles. La seconde raison est datée, institutionnelle, et vérifiable dans
l'histoire économique plutôt que dans un backtest.

**La thèse, en une phrase.** Un signal macro pris en absolu sur un seul pays est
du market timing, et il ne marche pas ; en relatif entre pays développés, la
dispersion que la stratégie exploite a été supprimée par l'entrée en vigueur de
l'euro, à l'échelle où le signal la mesure.

---

## Ce qui a été gelé, et quand

C'est le point qui distingue ce travail d'un backtest. Le piège de ce type
d'étude n'est pas le signal, c'est la **table qui traduit le signal en position**.
Un scan de pratiques a reproduit la méthodologie d'un indice publié et l'a
comparée à 160 000 combinaisons de facteurs tirées au hasard : la version
officielle se plaçait au 98ᵉ percentile. C'est le percentile qu'on atteint en
essayant beaucoup de tables, pas en en comprenant une.

La table a donc été écrite d'abord, à partir d'un raisonnement de cours de macro
de première année, et jamais ajustée ensuite. Seize cellules : quatre thèmes
(croissance, inflation, politique monétaire, prime de risque de crédit) contre
quatre classes d'actifs (actions, obligations, matières premières, dollar). Deux
cellules sont signalées dans le document gelé comme discutables, et laissées
telles quelles.

L'ordre est opposable, il est inscrit dans l'historique git : la table est
commitée dans `docs/PRESPEC.md` avant tout résultat, et la pré-spécification de
la seconde hypothèse est commitée **avant même le téléchargement des données**
qu'elle utilise.

J'ai vérifié à la main que les seize signes du code correspondent aux seize
signes du document gelé. Ils correspondent. Cette vérification a dû être faite à
la main parce que le code affirme dans un commentaire qu'un test automatique s'en
charge, et que ce test n'existe pas.

---

## Première tentative : un signal absolu sur un seul pays

Quatre thèmes macro américains, chacun mesuré en **variation sur un an** et
jamais en niveau. Vingt actifs. Échantillon de décembre 1994 à septembre 2026,
soit 8 278 jours de bourse.

Résultat, net de frais : **ratio de Sharpe −0,38**, perte maximale −68,5 %. Les
quatre classes d'actifs perdent séparément : −0,16 sur les actions, −0,04 sur les
obligations, −0,27 sur les matières premières, −0,36 sur le dollar.

Un chiffre négatif ne prouve rien à lui seul. Ce qui compte est la comparaison
contre le hasard.

**La table des signes fait moins bien qu'une pièce de monnaie.** Contre 400 tables
tirées au sort, sur les mêmes données et avec les mêmes frais, la table gelée
arrive au **7ᵉ percentile** : 373 tables aléatoires sur 400 font mieux qu'elle.

Ce résultat a été audité avant d'être cru, parce qu'une comparaison de ce type
peut être faussée par les frais de transaction. Retourner des signes ne change ni
la régularité du signal ni le volume d'échanges qu'il implique : les tables
aléatoires tradent 0,065 par jour contre 0,060 pour la vraie, et payent le même
frein de coût. La table perd donc sur l'information, pas sur les frais.

Cellule par cellule, la table gelée est d'accord avec les signes constatés dans
les données sur **7 cellules sur 16**, là où une table tirée au hasard en
accorderait 8. Et les deux seules cellules dont la statistique dépasse le seuil
usuel de 2 **contredisent toutes les deux la table** : l'inflation contre le
dollar à +2,99 quand la table dit −1, l'inflation contre les matières premières à
−2,02 quand la table dit +1. Deux cellules sur seize quand le hasard en donne 0,8
reste du bruit. Mais c'est le mécanisme du 7ᵉ percentile vu de près : les seules
relations visibles pointent à l'envers du manuel.

### Le refus qui est le résultat

Une table au 7ᵉ percentile invite à retourner ses signes. Le livre aux signes
inversés, avec les mêmes frais sur exactement le même volume d'échanges, rend
**+0,32**.

Les signes n'ont pas été retournés. Choisir le sens après avoir vu lequel
rapporte est précisément la fabrication que la pré-spécification sert à empêcher.
La table est publiée telle qu'elle a été écrite, avec son résultat négatif.

Une précision, parce que le document publié du projet se trompe ici : il annonce
+0,38 pour le livre retourné. Ce chiffre est la simple négation du Sharpe *net*,
ce qui transforme les frais en gain. Les frais se soustraient dans les deux sens.
Le bon chiffre est +0,32.

---

## Ce que la vérification a invalidé

C'est la partie de cette note dont le jury doit retenir l'existence, parce
qu'elle montre comment le projet se corrige.

Le document publié `RESULTS_H1.md` affirmait que **le signal, lui, portait de
l'information** : remplacé par du bruit, il arrivait au 99ᵉ percentile. Cette
affirmation était la conclusion la plus flatteuse du chantier. Elle est fausse.

La pré-spécification gelée demandait un placebo « à cycle d'activité apparié » :
un faux signal qui bascule à la même fréquence que le vrai. Le code a implémenté
autre chose, un simple **mélange** des observations. Or les quatre thèmes sont des
variations sur un an : ils s'autocorrèlent à 0,993-0,999 d'un jour au suivant, ce
qui veut dire qu'ils bougent lentement. Mélangés, ils tombent à −0,011 et
changent de sens tous les jours.

Conséquence, mesurée : le faux signal trade **21 fois plus** que le vrai (1,269
par jour contre 0,060) et perd 0,527 de Sharpe en frais contre 0,028 pour le vrai.
Le placebo ne perdait pas parce qu'il était moins informé, mais parce qu'il était
ruiné par les frais de courtage. **Avant frais, le vrai signal est au 15ᵉ
percentile de ce placebo, pas au 99ᵉ.**

J'ai donc construit le placebo que la pré-spécification demandait, sous quatre
formes indépendantes qui conservent la lenteur du signal et donc son volume
d'échanges. Le vrai signal arrive **entre le 16ᵉ et le 23ᵉ percentile** du bruit
selon la construction, avant comme après frais.

**Le signal ne porte pas d'information directionnelle.** Le 99ᵉ percentile publié
était un artefact d'un contrôle mal implémenté.

La correction simplifie le résultat au lieu de le compliquer. La lecture
précédente était une énigme : un signal informatif branché sur une table cassée.
La lecture mesurée est que ni l'un ni l'autre ne fonctionne, ce qui est cohérent
avec le tableau des seize cellules.

---

## Seconde tentative : dix-neuf pays les uns contre les autres

Le défaut de la première tentative était structurel, la seconde change donc la
structure et non les réglages. Trois différences : on compare les pays entre eux
au lieu de parier sur la direction d'un marché ; on n'utilise que des **prix de
marché**, jamais révisés, au lieu de statistiques publiées puis corrigées ;
et dix-neuf pays au lieu d'un.

Les neuf cellules (trois thèmes contre trois classes d'actifs) ont été testées
**avant de construire le moindre portefeuille**. C'est ce qui manquait à la
première tentative, où un seul chiffre de Sharpe cachait seize relations.

**Aucune des neuf cellules n'atteint le seuil de 2**, alors que le hasard seul en
donnerait 0,5. La plus forte statistique de la table vaut 1,45. Le seuil corrigé
pour neuf tests simultanés est de 2,77.

Aucun portefeuille n'a été construit. Il n'y avait rien pour le construire, et en
construire un quand même aurait été chercher le sous-ensemble qui survit.

Un chiffre que la pré-spécification promettait et que personne n'avait calculé :
la taille d'effet minimale que cet échantillon pouvait détecter. Elle vaut 0,032
à 0,098 selon la cellule, pour une corrélation maximale observée de 0,028.
**Aucune cellule n'avait la puissance statistique de détecter un effet de la
taille de celui qu'elle a mesuré.** Il faudrait de 131 à 28 358 ans de données.

---

## Le diagnostic : neuf pays devenus un seul instrument en 1999

Une stratégie relative a besoin que les pays diffèrent. Ils ont cessé de le
faire, et la date est connue.

Dispersion des taux courts entre les dix-neuf souverains, écart-type mensuel entre
pays puis moyenne par période :

| période | taux courts | taux longs |
|---|---|---|
| 1990-1998 | **2,30 %** | 1,72 % |
| 1999-2007 | 1,37 % | 1,01 % |
| 2008-2015 | 1,08 % | 1,44 % |
| 2016-2026 | **0,89 %** | 0,91 % |

Et chez les neuf pays du panel qui ont adopté l'euro le 1ᵉʳ janvier 1999
(Allemagne, Belgique, Espagne, Finlande, France, Irlande, Italie, Pays-Bas,
Portugal), l'écart-type de leurs taux courts entre eux :

```
1990-1998 : 2,35 %
1999-2007 : 0,00 %
2016-2026 : 0,00 %
```

Vérifié mois par mois plutôt qu'arrondi : la valeur est **exactement nulle dans
309 des 330 mois** postérieurs à janvier 1999. Deux mois seulement sont
réellement non nuls, janvier 1999 à 0,158 % et février 1999 à 0,055 %, pendant la
convergence finale. **À partir de mars 1999, les neuf séries sont identiques.**

Le panel affiche dix-neuf noms. Sur le thème monétaire il en porte onze
indépendants, et sur le thème des changes, dix. La stratégie classait des pays qui
partagent une banque centrale.

**La stratégie n'a pas décliné, sa matière première a été supprimée par traité.**
C'est ce qui rend le résultat publiable : la cause est identifiée, datée, et
extérieure aux données.

### Deux morts, deux étiquettes

En coupant l'échantillon au 1ᵉʳ janvier 1999 :

| période | cellules exploitables | au-dessus du seuil | plus forte statistique |
|---|---|---|---|
| avant l'euro, 1990-1998 | 4 sur 9 | 0 | 1,74 |
| après l'euro, 1999-2026 | 9 sur 9 | 0 | 0,95 |

La distinction commande la suite du programme de recherche.

**Après 1999, la mort est mécanique.** La dispersion que la stratégie exploite est
mesurablement nulle pour neuf souverains sur dix-neuf. Aucune méthode
d'estimation ne retrouve une coupe transversale qui n'existe pas. La règle du
projet dit qu'une mort mécanique ne se rouvre pas.

**Avant 1999, c'est un défaut de puissance, pas un rejet.** Neuf ans, quatre
cellules utilisables, une statistique maximale de 1,74. La règle du projet dit
qu'un défaut de puissance se rouvre, à condition d'un échantillon qui possède
réellement la dispersion, c'est-à-dire de données payantes.

### Jusqu'où cette conclusion porte

La dispersion ci-dessus est mesurée en points de pourcentage, et sur la même
période le **niveau** des taux s'est effondré, de 7,84 % à 1,13 % en moyenne. En
mesure sans échelle, la dispersion relative **multiplie par douze** au lieu de
diviser par 2,6.

Les deux lectures sont exactes et ne répondent pas à la même question. Trois
conséquences, données pour que l'affirmation ne soit pas sur-interprétée :

* Le fait euro n'est pas affecté : neuf séries identiques le restent sous
  n'importe quelle normalisation.
* Le diagnostic vaut **pour le signal tel qu'il est construit**. Les thèmes de la
  seconde tentative sont des variations en points de pourcentage : c'est l'échelle
  absolue qui les gouverne, et c'est celle qui s'est comprimée.
* L'énoncé général « la macro des pays développés n'a plus de dispersion » est
  **faux** en mesure relative. Il doit être formulé à l'échelle du signal.

Par ailleurs, chez les dix souverains qui ont gardé leur propre taux directeur, la
dispersion absolue baisse de 1,94 % à 1,11 %, soit 43 % contre 61 % pour le panel
entier. Une partie de la compression vient donc du mouvement mondial des taux vers
zéro, et pas seulement de l'euro.

---

## Pourquoi la première tentative devait échouer

La comparaison avec le résultat publié est l'explication, et elle se décompose en
trois différences de poids inégal.

**La structure, et c'est la décisive.** AQR comparent les pays entre eux. La
première tentative construisait un signal absolu sur un seul pays, ce qui est du
market timing sous étiquette macro. La propriété transversale est justement l'endroit
d'où vient le Sharpe publié.

**Le signal.** AQR utilisent des variations de **prévisions**, donc des révisions
d'enquêtes tournées vers l'avant. La première tentative utilisait des variations
de **données réalisées publiées**, qui arrivent avec un délai. Quand une variation
sur un an de croissance réalisée devient positive, le marché a eu un an pour
l'intégrer.

**La période.** 1970-2016 contre 1994-2026. L'échantillon ancien est dominé par
l'époque dont ce travail ne voit que neuf ans : des politiques monétaires
nationales indépendantes, des inflations séparées par des points entiers entre
voisins, et des monnaies qui bougeaient seules.

La seconde tentative corrige les deux premières différences et se heurte à la
troisième.

---

## Ce que ce travail a corrigé chez lui-même

Le projet avait déjà publié deux fois des chiffres qu'une régénération ultérieure
a contredits. La vérification menée pour écrire cette note en a trouvé une
troisième forme, et il faut la nommer.

Six lignes publiées dans les documents de résultats n'étaient produites par
**aucun code commité** : la table de dispersion, l'écart-type du bloc euro, la
coupe avant/après 1999, l'accord de 7 sur 16, et le décompte des cellules
significatives. Elles avaient été calculées dans une session interactive puis
perdues.

Je les ai toutes reconstruites. **Cinq se reproduisent exactement**, jusqu'au
quatrième chiffre décimal pour les neuf cellules et au centième de point pour les
huit valeurs de dispersion. Une ne se reproduit pas : le document annonce une
seule cellule au-dessus du seuil de 2, ma reconstruction en trouve deux, dont une
juste à la frontière à 2,02. La conclusion ne change pas, puisque deux sur seize
quand le hasard en donne 0,8 reste du bruit.

Le constat porte donc deux moitiés et il faut garder les deux : **les chiffres
étaient justes, et la discipline qui permet de le savoir manquait.** Sans la
reconstruction, rien ne distinguait ces six lignes de celles qui, dans un autre
document du programme, étaient fausses.

Autres corrections apportées : l'échantillon commence en décembre 1994 et non en
1993 comme annoncé ; le livre retourné rend +0,32 et non +0,38 ; l'écart-type du
bloc euro est nul dans 309 mois sur 330 et non dans tous ; et deux commentaires du
code comptent huit membres de la zone euro là où il y en a neuf.

---

## Ce qui reste ouvert

**Fermé mécaniquement.** Le momentum macro transversal entre pays développés après
1999, sur des variables de taux. La matière première est mesurablement nulle pour
neuf souverains sur dix-neuf.

**Fermé sur preuves.** La version absolue à un seul pays. Elle perd 0,38 de
Sharpe, sa table est battue par 373 tables aléatoires sur 400, et son signal ne
bat pas un placebo correctement apparié.

**Rouvrable sous condition.** L'échantillon d'avant 1999, qui est un défaut de
puissance et non un rejet. La condition est un échantillon dont la dispersion
puisse être vérifiée avant tout calcul de rendement, ce qui implique des données
payantes.

**Rouvrable avec deux obstacles nommés.** Les marchés émergents, où la dispersion
évolue en sens inverse : le rapport entre leur dispersion et celle du G10 passe de
0,7 en 2006-2012 à 1,5 en 2020-2026. Les obstacles sont un échantillon court, des
coûts de cinq à dix fois ceux du G10, et des contrôles de capitaux qu'un backtest
sur prix comptant ne voit pas.

Enfin, deux chiffres portent toute la comparaison qui structure ce résultat et
aucun n'est vérifiable dans le dépôt : le Sharpe de 1,2 d'AQR et leur corrélation
de −0,22 aux actions. Ce sont des citations de littérature. Elles méritent une
référence paginée, qui n'existe nulle part dans le projet à ce jour.
