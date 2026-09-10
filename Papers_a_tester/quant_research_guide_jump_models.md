# Guide Quant Research: Downside Risk Reduction via Statistical Jump Models (JM)

> **Document de référence pour l'implémentation et le backtest d'une stratégie de changement de régime (Regime-Switching 0/1 Strategy)**
> *Basé sur le papier de recherche : Shu, Yu & Mulvey (Princeton University, 2024)*  
> *Titre original : "Downside Risk Reduction Using Regime-Switching Signals: A Statistical Jump Model Approach"*

---

## 1. Vue d'Ensemble & Objectif Recherche

L'objectif de cette recherche est d'implémenter et de tester en conditions réelles (*live-sample / walk-forward*) une stratégie d'allocation d'actifs dynamique dite **Stratégie 0/1**. 

La stratégie bascule l'exposition entre :
1. **Un actif risqué** (ex. Indice S&P 500, DAX ou Nikkei 225) avec une pondération de **100%**.
2. **Un actif sans risque** (ex. Bons du Trésor à 3 mois / T-Bills) avec une pondération de **100%**.

### Pourquoi le Statistical Jump Model (JM) plutôt qu'un HMM classique ?
- **Problème des HMM (Hidden Markov Models) :** Sensibilité extrême au bruit de marché quotidien, sur-ajustement (mis-estimation), changements de régime trop fréquents (flip-flopping), et absence de persistance temporelle explicite.
- **Solution du Jump Model (JM) :** Modèle non-paramétrique basé sur le clustering ($K$-means) pénalisé par un **Jump Penalty ($\lambda$)**. La pénalité $\lambda$ contrôle directement la persistance des régimes et réduit le turnover tout en améliorant la détection des phases baissières (*bear markets*).

---

## 2. Architecture des Données & Engineering des Features

### 2.1 Univers de Données
- **Prix / Rendements quotidiens :** S&P 500 (USA), DAX (Allemagne), Nikkei 225 (Japon) de **1970 à 2023**.
- **Taux Sans Risque ($r_{f,t}$) :** Taux T-Bill 3 mois de chaque pays correspondant.
- **Rendements en Excès ($R_t$) :** $R_t = r_{asset, t} - r_{f, t}$
- **Période d'In-Sample / Training initial :** 12 ans (3000 jours de cotation).
- **Période de Validation Cross-Validation :** 8 ans.
- **Période de Backtest Out-of-Sample (OOS) :** **1990 à 2023** (soit 34 ans de données de test).

### 2.2 Feature Set pour le Jump Model
Contrairement aux HMMs qui n'utilisent généralement que les rendements quotidiens simples $r_t$ (modélisant la volatilité conditionnelle Gaussianne), le Jump Model tire sa force d'un vecteur de features $x_t \in \mathbb{R}^D$ avec $D=3$ mesurant le risque de perte et le rendement ajusté du risque :

| Feature $d$ | Métrique | Fenêtre (Halflife $hl$) | Description Mathématique |
| :--- | :--- | :--- | :--- |
| **$x_{t,1}$** | Downside Deviation (DD) | 10 jours | $EWM_{10} \left( \sqrt{ \mathbb{E} [ R^2 \cdot \mathbb{I}_{\{R < 0\}} ] } ight)$ |
| **$x_{t,2}$** | Ratio de Sortino | 20 jours | $rac{EWM_{20}(R)}{EWM_{20}(DD)}$ |
| **$x_{t,3}$** | Ratio de Sortino | 60 jours | $rac{EWM_{60}(R)}{EWM_{60}(DD)}$ |

#### Détails d'implémentation des features :
1. **Lissage Exponentiel (EWM) :** Pour une demi-vie $hl$, le facteur d'amortissement $lpha$ est $lpha = 1 - \exp(-\ln(2)/hl)$.
2. **Downside Deviation :** Ne prend en compte que les rendements négatifs (mesure du *downside risk* pur).
3. **Normalisation / Standardisation :** À chaque fenêtre d'entraînement, le vecteur de features $X \in \mathbb{R}^{T 	imes 3}$ doit être centré et réduit (z-score scaling : moyenne 0, variance 1) pour éviter qu'une feature ne domine la distance euclidienne $L_2$.

---

## 3. Formulation Mathématique du Statistical Jump Model (JM)

### 3.1 Problème d'Optimisation
Soit une séquence d'observations de features standardisées $x_0, x_1, \dots, x_{T-1} \in \mathbb{R}^D$ sur une fenêtre de $T=3000$ jours, et $K=2$ régimes ($s_t \in \{0, 1\}$).

Le Jump Model cherche à minimiser simultanément la perte de clustering et la pénalité de transition de régime :

$$\min_{\Theta, S} \sum_{t=0}^{T-1} l(x_t, 	heta_{s_t}) + \lambda \sum_{t=1}^{T-1} \mathbb{I}_{\{s_{t-1} 
eq s_t\}}$$

Où :
- $S = \{s_0, s_1, \dots, s_{T-1}\} \in \{0, 1\}^T$ est la séquence cachée d'états/régimes.
- $\Theta = \{	heta_0, 	heta_1\}$ sont les centroïdes des $K=2$ clusters ($	heta_k \in \mathbb{R}^D$).
- $l(x_t, 	heta_{s_t}) = rac{1}{2} \|x_t - 	heta_{s_t}\|_2^2$ est la distance $L_2$ au carré ajustée.
- $\lambda \ge 0$ est le **Jump Penalty** (hyperparamètre de lissage).
- $\mathbb{I}_{\{\cdot\}}$ est la fonction indicatrice (vaut 1 si $s_{t-1} 
eq s_t$, 0 sinon).

### 3.2 Algorithme de Résolution (Coordinate Descent)
L'optimisation alternée s'effectue en deux étapes itératives jusqu'à convergence :

1. **Étape 1 : Optimisation des Centroïdes $\Theta$ (Fixer $S$)**
   Pour chaque état $k \in \{0, 1\}$ :
   $$	heta_k = rac{1}{|\{t : s_t = k\}|} \sum_{t: s_t = k} x_t$$

2. **Étape 2 : Optimisation de la Séquence d'États $S$ (Fixer $\Theta$)**
   La minimisation de la fonction coût sur les variables discrètes $S$ se résout exactement en temps linéaire $O(T K^2)$ via **Programmation Dynamique (DP)** (similaire à l'algorithme de Viterbi sans matrice de transition stochastique).

#### Récursivité de la Programmation Dynamique (DP) :
Soit $V(t, k)$ le coût minimal accumulé jusqu'au temps $t$ en se terminant dans l'état $k$ :
$$V(t, k) = l(x_t, 	heta_k) + \min_{j \in \{0, 1\}} \left( V(t-1, j) + \lambda \cdot \mathbb{I}_{\{j 
eq k\}} ight)$$
Pour $t=0$ : $V(0, k) = l(x_0, 	heta_k)$.
Une fois la matrice $V$ remplie jusqu'à $t=T-1$, on remonte le chemin optimal (*backtracking*).

3. **Re-runs contre les minima locaux :**
   Exécuter l'algorithme **10 fois** avec des initialisations différentes (type K-Means++) et conserver la solution qui minimise l'objectif global.

### 3.3 Mapping des Régimes (Bull vs Bear)
Après l'optimisation des 3000 jours :
- Calculer le rendement cumulé en excès de l'actif sous chaque état :
  - **Régime Bull ($s_t = 0$) :** État présentant le rendement cumulé le plus élevé (caractérisé par un faible Downside Deviation et un Sortino élevé).
  - **Régime Bear ($s_t = 1$) :** État présentant le rendement cumulé le plus bas / négatif (caractérisé par un Downside Deviation élevé).

---

## 4. Protocole d'Inférence en Temps Réel (Online Inference)

Pour éliminer tout biais de prospective (*look-ahead bias*), la stratégie doit générer son signal au jour $t$ en n'utilisant **strictement que les données disponibles jusqu'au jour $t$**.

### Procédure d'Inférence Online (chaque jour $t$) :
1. **Mise à jour des Centroïdes $\Theta$ :** Ré-estimés tous les 6 mois sur une fenêtre glissante de 3000 jours.
2. **Fenêtre Glissante de Lookback ($l = 3000$ jours) :**
   Prendre les features $x_{t-l+1}, \dots, x_t$.
3. **Exécution du DP à Centroïdes Fixes :**
   Résoudre l'étape DP de la section 3.2 sur cette fenêtre de 3000 jours en conservant les centroïdes $\Theta$ fixés à leurs dernières valeurs optimales.
4. **Extraction du Signal :**
   Extraire uniquement la dernière valeur $\hat{s}_t$ de la séquence d'états optimale. C'est le régime déduit au jour $t$.

---

## 5. Walk-Forward Cross-Validation : Sélection Dynamique du Jump Penalty ($\lambda$)

La valeur optimale du Jump Penalty $\lambda$ varie selon la volatilité globale et les structures de marché. Le papier propose d'optimiser $\lambda$ dynamiquement par **Time-Series Cross-Validation**.

### Protocole de Calibration Mensuelle :
Au premier jour de chaque mois $m$ :
1. **Fenêtre de Validation :** Considérer les **8 années précédentes** de données historiques (mode simulation live).
2. **Grille de Candidats $\lambda$ :** Tester une liste de valeurs (ex. $\lambda \in \{0.0, 5.0, 15.0, 35.0, 50.0, 70.0, 100.0, 150.0\}$).
3. **Backtest Fictif sur la Fenêtre de Validation :** Pour chaque candidat $\lambda$, générer la séquence d'inférence online $(\hat{s}_	au)$ sur les 8 ans, exécuter la Stratégie 0/1 avec le délai de trading (1 jour) et les frais de transaction (10 bps).
4. **Critère de Sélection :** Sélectionner le $\hat{\lambda}$ qui maximise le **Ratio de Sharpe** de la stratégie sur la période de validation.
5. **Application OOS :** Utiliser ce $\hat{\lambda}$ optimal pour générer les signaux de trading quotidiens pendant tout le mois $m$.

---

## 6. Référentiel Benchmark : Hidden Markov Model (HMM)

Pour benchmarker le Jump Model, le papier utilise un **HMM Gaussien à 2 états** :
- **Input :** Rendements logarithmiques quotidiens $r_t$.
- **Fenêtre d'entraînement :** 3000 jours glissants (mis à jour quotidiennement).
- **Décodage :** Algorithme de Viterbi pour trouver la séquence d'états.
- **Lissage du HMM (Filtre Médian / Rolling Mean) :**
  Pour compenser l'instabilité du HMM aux extrémités de la fenêtre, on applique une moyenne glissante sur les états inférés en ligne $(\hat{s}_t)$ avec une fenêtre $k$ (optimisée par cross-validation de la même façon que $\lambda$, $k \in \{0, 2, 4, 8, 20\}$).
  Si $	ext{RollingMean}_k(\hat{s}_t) > 0.5$, le signal est Bear ($1$), sinon Bull ($0$).

---

## 7. Règles d'Exécution & Backtest Quant

### 7.1 Stratégie de Trading 0/1
$$	ext{Pondération Actif Risqué } w_t = egin{cases} 1.0 & 	ext{si } \hat{s}_{	ext{effective}, t} = 0 	ext{ (Bull)} \ 0.0 & 	ext{si } \hat{s}_{	ext{effective}, t} = 1 	ext{ (Bear)} \end{cases}$$
$$	ext{Pondération Actif Sans Risque } w_{rf, t} = 1.0 - w_t$$

### 7.2 Délai de Trading (*Trading Delay*)
- **Baseline (1 jour de délai) :** Le signal calculé à la clôture du jour $t$ ($\hat{s}_t$) est exécuté à la clôture du jour $t+1$. La position prend effet sur le rendement du jour $t+2$.
- **Tests de Robustesse :** Tester avec des délais de 5 jours ($t+6$) et 10 jours ($t+11$).

### 7.3 Frictions & Frais de Transaction
- **Coût de Transaction Unilatéral (One-Way) :** **10 bps (0.10%)** par transaction.
- **Frais au jour $t$ :** $	ext{Cost}_t = 0.0010 	imes |w_t - w_{t-1}|$.

---

## 8. Résultats du Papier (Benchmarks pour Validation de votre Code)

Votre implémentation doit reproduire des performances similaires aux chiffres officiels du papier sur la période OOS 1990–2023 (Délai = 1 jour, Frais = 10 bps) :

### Tableau Comparatif des Performances (OOS 1990–2023)

| Métrique | S&P 500 (B&H) | S&P 500 (HMM) | S&P 500 (JM) | DAX (B&H) | DAX (JM) | Nikkei 225 (B&H) | Nikkei 225 (JM) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rendement Annuel (CAGR)** | 10.2% | 8.5% | **11.2%** | 6.8% | **8.6%** | 0.8% | **4.7%** |
| **Volatilité Annuelle** | 18.2% | 11.3% | **13.1%** | 22.1% | **16.4%** | 23.4% | **17.1%** |
| **Ratio de Sharpe** | 0.48 | 0.54 | **0.68** | 0.30 | **0.44** | 0.12 | **0.31** |
| **Max Drawdown (MDD)** | -55.2% | -28.9% | **-26.6%** | -72.7% | **-39.4%** | -79.1% | **-45.3%** |
| **Ratio de Calmar** | 0.16 | 0.21 | **0.33** | 0.09 | **0.18** | 0.04 | **0.12** |
| **Expected Shortfall ($ES_{5\%}$)**| -2.7% | -1.8% | **-2.0%** | -3.3% | **-2.5%** | -3.4% | **-2.6%** |
| **Turnover Annuel** | 0% | 141% | **44%** | 0% | **170%** | 0% | **72%** |
| **Levier Moyen (Exposition)** | 100% | 72% | **80%** | 100% | **84%** | 100% | **75%** |

### Insights Clefs à Vérifier :
1. **Réduction massive du Max Drawdown :** Le JM divise le MDD du S&P 500 par 2 (-55.2% $ightarrow$ -26.6%) et celui du DAX par 2 (-72.7% $ightarrow$ -39.4%).
2. **Turnover contenu :** Le Jump Model génère un turnover environ 3 fois inférieur au HMM sur le S&P 500 (44% vs 141%), prouvant l'effet stabilisateur du Jump Penalty $\lambda$.
3. **Surperformance en Rendement & Sharpe :** Même avec un levier moyen réduit à 80%, le JM bat le Buy & Hold en rendement brut (+1.0% sur S&P, +1.8% sur DAX, +3.9% sur Nikkei).

---

## 9. Plan d'Implémentation Pas-à-Pas pour Quant Researcher

Pour coder ce système proprement en Python, suivez l'architecture modulaire ci-dessous :

```
project_root/
│
├── data/
│   ├── sp500_daily.csv       # Date, Close, TBill_3M
│   ├── dax_daily.csv
│   └── nikkei_daily.csv
│
├── src/
│   ├── features.py           # Calcul EWM Downside Deviation & Sortino
│   ├── jump_model.py         # DP Viterbi, Coordinate Descent, Fit & Online Predict
│   ├── hmm_model.py          # Baseline HMM avec Median Filter
│   ├── cross_validation.py   # Walk-forward optimization de lambda et k
│   └── backtester.py         # Moteur de simulation avec délais et frais (10 bps)
│
└── main_backtest.py          # Script maître d'exécution OOS 1990-2023
```

### 9.1 Algorithme Python Résumé du Jump Model (Cœur Algorithmique)

```python
import numpy as np

def fit_jump_model(X, n_states=2, jump_penalty=50.0, max_iter=20, n_init=10):
    """
    X: shape (T, D) - Features standardisées
    jump_penalty: lambda
    """
    T, D = X.shape
    best_obj = float('inf')
    best_centroids = None
    best_states = None
    
    for init in range(n_init):
        # Initialisation aléatoire ou k-means++
        idx = np.random.choice(T, n_states, replace=False)
        centroids = X[idx].copy()
        states = np.zeros(T, dtype=int)
        
        for iteration in range(max_iter):
            # Étape 1: Programmation Dynamique (Fixer Centroïdes, Trouver États)
            V = np.zeros((T, n_states))
            backtrack = np.zeros((T, n_states), dtype=int)
            
            # Coût initial t=0
            for k in range(n_states):
                V[0, k] = 0.5 * np.sum((X[0] - centroids[k])**2)
                
            for t in range(1, T):
                dist = 0.5 * np.sum((X[t] - centroids)**2, axis=1) # (n_states,)
                for k in range(n_states):
                    costs = V[t-1, :] + jump_penalty * (np.arange(n_states) != k)
                    best_prev = np.argmin(costs)
                    V[t, k] = dist[k] + costs[best_prev]
                    backtrack[t, k] = best_prev
                    
            # Backtracking
            states[T-1] = np.argmin(V[T-1, :])
            for t in range(T-2, -1, -1):
                states[t] = backtrack[t+1, states[t+1]]
                
            # Étape 2: Mise à jour des centroïdes
            new_centroids = np.zeros_like(centroids)
            for k in range(n_states):
                mask = (states == k)
                if np.sum(mask) > 0:
                    new_centroids[k] = np.mean(X[mask], axis=0)
                else:
                    new_centroids[k] = centroids[k]
                    
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
            
        obj_val = np.min(V[T-1, :])
        if obj_val < best_obj:
            best_obj = obj_val
            best_centroids = centroids
            best_states = states
            
    return best_centroids, best_states
```

---

## 10. Checklist de Validation du Quant Researcher

Avant de valider vos résultats de backtest, vérifiez les points suivants :
- [ ] **Biais de prospective (Look-ahead bias) :** Les features $x_t$ au jour $t$ sont-elles calculées sans aucune donnée du jour $t+1$ ?
- [ ] **Standardisation :** La moyenne et l'écart-type de z-score scaling sont-ils calculés **uniquement sur la fenêtre glissante d'entraînement** (sans utiliser le futur OOS) ?
- [ ] **Délai d'exécution :** Le signal $\hat{s}_t$ calculé à la clôture de $t$ s'applique-t-il bien sur les rendements de $t+2$ ?
- [ ] **Frais de transaction :** Les 10 bps sont-ils prélevés à chaque changement de position ($0 ightarrow 1$ et $1 ightarrow 0$) ?
- [ ] **Taux sans risque :** Les rendements du $0/1$ en période Baissière accumulent-ils correctement le rendement du T-Bill quotidien ?

---
*Ce document résume l'intégralité du papier "Downside Risk Reduction Using Regime-Switching Signals: A Statistical Jump Model Approach" (Shu et al., 2024) et constitue le cahier des charges complet pour votre implémentation.*
