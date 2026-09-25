# Synthèse et Analyse Approfondie : Tactical Asset Allocation with Macroeconomic Regime Detection

## 1. Informations Générales & Fiche Synthétique

- **Titre original** : Tactical asset allocation with macroeconomic regime detection
- **Auteurs** : D. C. Oliveira, D. Sandfelder, A. Fujita, X. Dong & M. Cucuringu
- **Affiliations** : 
  - Université de São Paulo (Brésil)
  - Oxford-Man Institute of Quantitative Finance, Université d'Oxford (Royaume-Uni)
  - Medical Institute of Bioregulation, Université de Kyushu (Japon)
  - Université de Californie à Los Angeles (UCLA, États-Unis)
- **Journal** : *Quantitative Finance* (Publié le 11 juin 2026)
- **DOI** : [10.1080/14697688.2026.2659195](https://doi.org/10.1080/14697688.2026.2659195)
- **Domaine d'application** : Allocation Tactique d'Actifs (TAA), Apprentissage Automatique Non Supervisé, Séries Temporelles Macroéconomiques, Modélisation de Régimes de Marché.

---

## 2. Résumé Exécutif

L'article propose un cadre méthodologique novateur pour l'**Allocation Tactique d'Actifs (TAA)** en intégrant la détection de régimes macroéconomiques issue de techniques d'apprentissage automatique (*unsupervised machine learning*). 

Alors que la littérature traditionnelle identifie les régimes à partir des séries temporelles de rendements financiers (particulièrement bruitées), cette étude exploite la riche base de données macroéconomique **FRED-MD** (127 indicateurs mensuels de la Réserve Fédérale de St. Louis) pour extraire des états fondamentaux stables et interprétables.

L'approche se structure autour de trois étapes clés :
1. **Détection et classification probabiliste des régimes** via un algorithme *Modified K-Means* à deux niveaux (distance $L_2$ pour isoler les crises/outliers, puis distance Cosinus pour classifier les périodes typiques).
2. **Prévision conditionnelle des rendements et volatilités** via des modèles naïfs et de régression Ridge dépendant des régimes prédits et de leur matrice de transition de Markov.
3. **Optimisation et dimensionnement du portefeuille** (Long-Only, Long-Short, Mixed, Black-Litterman, Mean-Variance Optimization) appliqués sur des ETFs sectoriels américains et des contrats à terme (Futures).

Les résultats empiriques démontrent une surperformance statistique et financière majeure par rapport aux benchmarks passifs (SPY, Equal-Weight) et aux approches d'optimisation classiques (MVO).

---

## 3. Architecture Méthodologique

```
+---------------------------------------------------------------------------------+
|                                1. DONNÉES ET PCA                                |
|  Base FRED-MD (127 séries macro) -> Transformations -> PCA (61 composantes, 95% var) |
+---------------------------------------------------------------------------------+
                                        |
                                        v
+---------------------------------------------------------------------------------+
|                          2. CLASSIFICATION ÉTAGÉE (ML)                          |
|  Étape A : K-Means L2 (k=2)  --> Isolement Régime 0 (Crises / Outliers)         |
|  Étape B : K-Means Cosinus (k=5) --> Partition des 5 Régimes Typiques (1 à 5)   |
|  Conversion Fuzzy & Lissage  --> Distribution P(Régime_i) & Matrice Transition  |
+---------------------------------------------------------------------------------+
                                        |
                                        v
+---------------------------------------------------------------------------------+
|                           3. PRÉVISION CONDITIONNELLE                           |
|  Modèle Naïf (Sharpe ratio)  |  Régression Ridge Linéaire (Rendements Ponderés) |
+---------------------------------------------------------------------------------+
                                        |
                                        v
+---------------------------------------------------------------------------------+
|                        4. CONSTRUCTIONS DE PORTEFEUILLE                         |
|  Schémas de taille (lo, lns, los, mx)  |  Optimisation Paramétrique (BL, MVO)   |
+---------------------------------------------------------------------------------+
```

### 3.1 Prétraitement des Données et ACP
- **Base de données FRED-MD** : 127 séries temporelles mensuelles caractéristiques de l'économie américaine (décembre 1959 à janvier 2023).
- **Transformations** : Application stricte des 7 codes de transformation prédéfinis par McCracken & Ng (2016) pour garantir la stationnarité (différences log, taux de variation, etc.), suivies d'une standardisation.
- **Réduction de dimension (PCA)** : Extraction de 61 composantes principales permettant de capturer **95 % de la variance cumulative** de l'ensemble du panneau macroéconomique.

### 3.2 L'Algorithme Modified K-Means
L'algorithme surmonte l'instabilité du k-means classique grâce à une architecture en deux phases :

1. **Isolation des mois atypiques ($L_2$ Clustering, $k=2$)** :
   La distance $L_2$ étant très sensible à l'amplitude, elle isole efficacement les périodes de choc macroéconomique extrême (ex. Crise de 2008, choc COVID-19 de 2020) désignées sous le nom de **RÉGIME 0**.

2. **Partitionnement des mois typiques (Cosine Clustering, $k=r=5$)** :
   Sur le sous-ensemble des mois ordinaires, l'utilisation de la distance Cosinus (invariante à l'amplitude du vecteur) permet de grouper les mois selon la direction de leur vecteur d'état économique. La méthode de l'élément coudé (*elbow heuristic*) détermine la valeur optimale $r=5$.

### 3.3 Incertitude et Distributions Probabilistes (*Fuzzy Membership*)
Pour transformer l'assignation déterministe en distribution de probabilités continue :
- La probabilité d'appartenance à un cluster $C_i$ est inversement proportionnelle à la distance $d_i$ au centroïde :
  $$P(C_i) = \frac{1/d_i}{\sum_j 1/d_j}$$
- La probabilité du Régime 0 est réintégrée via une transformation logarithmique non-linéaire :
  $$P_{R0} = - P_{\max} \log_2(1 - P(\text{REGIME } 0)) \quad \text{où} \quad P_{\max} = \max_{i \in \{1..r\}} P(\text{REGIME } i)$$
- Les probabilités sont ensuite normalisées. Cela confère à l'approche la souplesse du *fuzzy c-means* tout en conservant la robustesse du K-means.

### 3.4 Matrice de Transition de Markov
La dynamique temporelle est capturée par la matrice de transition $E_t$, où chaque élément $e_{ij}$ représente la probabilité historique de passer du régime $i$ au régime $j$. La prévision de l'état au temps $t+1$ s'écrit :
$$\tilde{p}_{t+1} = \tilde{p}_t^\top E_t$$

---

## 4. Caractérisation des 6 Régimes Macroéconomiques

L'interprétation post-hoc des régimes s'appuie sur la moyenne au sein de chaque cluster de variables économiques majeures (chômage, revenu personnel réel, sentiment des consommateurs, taux des fonds fédéraux, S&P 500) :

| Indice | Intitulé du Régime | Caractéristiques Économiques Principales | Alignement Historique |
| :--- | :--- | :--- | :--- |
| **Régime 0** | **Difficulté Économique** (*Economic Difficulty*) | Chômage au plus haut, sentiment consommateur et taux Fed au plus bas. Régime hautement auto-stable (*persistent*). | Recessions NBER (1973-75, 1981-82, 2008, 2020) |
| **Régime 1** | **Reprise Économique** (*Economic Recovery*) | Inflation sous contrôle, regain de confiance des consommateurs, mais marchés actions encore timides face à l'incertitude. | Phases de post-récession immédiate |
| **Régime 2** | **Croissance Expansionniste** (*Expansionary Growth*) | Forte prospérité, performances boursières solides, sentiment élevé, inflation modérée, politique monétaire neutre. | Milieu des années 1990, expansions des années 2010 |
| **Régime 3** | **Pression Stagflationniste** (*Stagflationary Pressure*) | Inflation forte, hausse des taux d'intérêt, resserrement monétaire pesant sur les actions, croissance au ralentisseur. | Chocs pétroliers des années 1970, épisode 2021-2022 |
| **Régime 4** | **Transition Pré-Récidive** (*Pre-Recession Transition*) | Signal d'alerte. L'inflation commence à refroidir, l'activité ralentit, le chômage monte légèrement, politique monétaire restrictive. | Mois précédant les récessions majeures |
| **Régime 5** | **Boom Reflationniste** (*Reflationary Boom*) | Résurgence économique portée par une politique monétaire très accommodante (ex: Quantitative Easing), inflation élevée mais tolérée. | Périodes post-assouplissement quantitatif |

---

## 5. Stratégies de Modélisation et d'Allocation

### 5.1 Modèles Prédictifs
1. **Modèle Naïf (*Naive Sharpe*)** : Prédit le ratio de Sharpe futur de chaque actif sur la base de sa moyenne historique conditionnelle au régime prédominant à $t+1$.
2. **Régression Ridge Linéaire (*Linear Ridge*)** : Entraîne un modèle de régression Ridge par régime, puis calcule la prévision d'actif en agrégeant les prédictions pondérées par la distribution de probabilité du régime futur :
   $$\hat{y}_{t+1}^{j, \text{ridge}} = \sum_{i=0}^r \tilde{p}_{i,t+1} \cdot \hat{y}_{t+1}^{ij, \text{ridge}}$$

### 5.2 Règles de Taille de Position (*Position Sizing*)
- **Long-Only ($lo$)** : Investit uniquement dans les $l$ actifs aux meilleures prévisions positives.
- **Long-and-Short ($lns$)** : Prends des positions acheteuses sur les $l$ meilleurs et vendeuses sur les $l$ pires.
- **Long-or-Short ($los$)** : Prends des positions sur les $l$ actifs de plus forte magnitude absolue.
- **Mixte ($mx$)** : Stratégie Long-Only par défaut, autorisant les positions Short uniquement si le mois à venir est prédit en **Régime 0** (Difficulté Économique).

### 5.3 Optimisation Paramétrique
- **Mean-Variance Optimization (MVO)** : Portefeuille classique de Markowitz sans signal de régime.
- **Black-Litterman (BL)** : Injecte les rendements espérés conditionnels aux régimes comme "vues" de l'investisseur (*views*) face au marché à l'équilibre.

---

## 6. Résultats Expérimentaux Empiriques

### 6.1 Expérience 1 : ETFs Sectoriels Américains (2000–2022)
*Données : 10 ETFs sectoriels (SPY, XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY).*

#### Table de Synthèse des Performances (Fenêtre Glissante)
| Modèle / Stratégie | Ratio de Sharpe | Ratio de Sortino | Max Drawdown (%) | % Mois Positifs |
| :--- | :---: | :---: | :---: | :---: |
| **SPY (Benchmark S&P 500)** | 0.818 | 1.331 | -33.49 % | 66.2 % |
| **Equal-Weight (EW)** | 0.838 | 1.325 | -32.29 % | 66.2 % |
| **MVO Long-Only ($l=4$)** | 1.129 | 2.132 | -6.93 % | 62.9 % |
| **Naive Long-Only ($l=2$)** | 1.065 | 2.532 | -6.16 % | 54.0 % |
| **Black-Litterman Long-Only ($l=2$)** | 1.177 | 1.744 | -8.64 % | 66.5 % |
| **Ridge Long-Only ($l=2$)** | 1.134 | **4.449** | **-4.39 %** | 60.7 % |
| **Ridge Long-Only ($l=3$)** 🏆 | **1.505** | **3.170** | **-4.39 %** | 59.4 % |

#### Enseignements Clés sur les ETFs :
1. **Surperformance Majeure du Ridge** : La combinaison de la régression Ridge avec les signaux de régimes (`ridge_lo_3`) génère un ratio de Sharpe de **1.505**, surpassant largement les indices passifs et le MVO.
2. **Protection Anti-Drawdown** : La variante `ridge_lo_2` limite le drawdown maximal à **-4.39 %** tout en affichant un Sortino de **4.449**.
3. **Validation Statistique** : Le test de Nemenyi et les t-tests appariés démontrent que les modèles basés sur les régimes réels surforment les régimes générés aléatoirement de manière hautement significative ($p < 0.001$).
4. **Inconvénient du Long-Short** : Les positions courtes ajoutent du bruit et détériorent la performance globale en période de reprise boursière rapide.

---

### 6.2 Expérience 2 : Marché des Contrats à Terme / Futures (2000–2023)
*Données : 8 contrats futures représentatifs sélectionnés par ACP et K-Means ($k=4$) parmi 39 séries (Actions, Obligations, Métaux, Énergie).*

- **Actifs Sélectionnés** :
  - *Cluster 0 (Actions)* : Euro Stoxx 50 (XU), E-mini S&P 500 (ES)
  - *Cluster 1 (Obligations)* : 10-Yr US Treasury Note (TY), 5-Yr US Treasury Bond (FB)
  - *Cluster 2 (Métaux)* : Argent (ZI), Or (ZG)
  - *Cluster 3 (Énergie)* : Gaz Naturel (ZN), Pétrole Brut (ZU)

#### Table de Synthèse des Performances Futures
| Modèle / Stratégie | Ratio de Sharpe | Ratio de Sortino | Max Drawdown (%) | % Mois Positifs |
| :--- | :---: | :---: | :---: | :---: |
| **Equal-Weight (EW)** | 0.406 | 0.587 | -34.24 % | 58.2 % |
| **MVO Long-Only ($l=4$)** | 1.092 | 3.383 | -7.63 % | 61.1 % |
| **Naive Mixte ($l=3$)** | 0.954 | 2.067 | -4.94 % | 61.1 % |
| **Black-Litterman Long-Only ($l=4$)** 🏆 | **1.210** | **3.492** | **-6.29 %** | 60.7 % |
| **Ridge Long-Only ($l=3$)** | 1.051 | 3.322 | -10.98 % | 62.1 % |

---

## 7. Implications Pratiques et Limitations

### Principaux Apports
1. **Pionniers sur FRED-MD** : Première application d'un jeu de données macroéconomiques étendu (127 séries) à la détection de régimes pour l'allocation tactique.
2. **Moins de Bruit, Plus de Stabilité** : En extrayant les régimes à partir de fondamentaux économiques au lieu des cours boursiers, les signaux obtenus sont plus stables et interprétables.
3. **Modélisation Continue de l'Incertitude** : La méthode Modified K-Means évite les basculements brutaux et binaires observés dans les GMMs.

### Recommandations de Gestion
- Privilégier des modèles simples régularisés (ex. **Ridge Regression**) couplés à des contraintes **Long-Only**.
- Limiter le nombre d'actifs sélectionnés ($l=2$ ou $l=3$) pour réduire le bruit de surestimation.
- Réserver les positions vendeuses aux signaux de haute conviction de crise majeure (Régime 0).

---

## 8. Référence Bibliographique Complete

> Oliveira, D. C., Sandfelder, D., Fujita, A., Dong, X., & Cucuringu, M. (2026). **Tactical asset allocation with macroeconomic regime detection**. *Quantitative Finance*, 1–25. DOI: [10.1080/14697688.2026.2659195](https://doi.org/10.1080/14697688.2026.2659195)
