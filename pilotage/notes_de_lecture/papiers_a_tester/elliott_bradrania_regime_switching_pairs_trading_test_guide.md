# Elliott & Bradrania (2018) — Regime-Switching Pairs Trading
## Guide pratique pour reproduire et tester le modèle

**Paper :** Robert J. Elliott & Reza Bradrania, *Estimating A Regime Switching Pairs Trading Model*, Quantitative Finance, 2018.  
**DOI :** 10.1080/14697688.2017.1403035

---

## 1. Objectif du papier

Le papier propose un modèle de **pairs trading dont la dynamique du spread change selon un régime de marché latent**.

L'idée centrale :

> Un spread de pairs trading n'a pas forcément la même vitesse de mean reversion, le même niveau d'équilibre ou la même volatilité dans toutes les périodes de marché.

Le marché est représenté par une chaîne de Markov à un nombre fini d'états :

- `Z_t = 1` : régime 1
- `Z_t = 2` : régime 2
- ...
- `Z_t = N` : régime N

Le régime `Z_t` est **caché** : on ne l'observe pas directement. On l'infère à partir du spread observé.

Le papier développe ensuite :
1. un modèle mean-reverting du spread ;
2. un modèle de régimes de Markov ;
3. un filtre permettant d'estimer la probabilité de chaque régime ;
4. une estimation des paramètres propres à chaque régime ;
5. une estimation de la matrice de transition ;
6. des formules récursives permettant de mettre ces estimations à jour au fur et à mesure que de nouvelles observations arrivent.

**Point important pour le test :** le papier développe surtout le **modèle statistique et son estimation**. Il ne constitue pas, à lui seul, une preuve que la stratégie de trading résultante produit un Sharpe élevé après coûts dans un environnement moderne.

---

# 2. Architecture générale

Le pipeline à reproduire est :

```text
Prix actif 1 ─┐
              ├──> Spread x_t ──> modèle mean-reverting
Prix actif 2 ─┘                         │
                                        ▼
                              Régime latent Z_t
                                        │
                       ┌────────────────┴───────────────┐
                       ▼                                ▼
              Paramètres régime i              Probabilités P(Z_t=i)
              a_i, b_i, σ_i                    conditionnelles au spread
                       │                                │
                       └────────────────┬───────────────┘
                                        ▼
                              Signal de trading
                                        │
                                        ▼
                              Backtest OOS
```

Pour un premier test, il est recommandé de séparer deux questions :

### Test A — Est-ce que le modèle détecte réellement des changements de dynamique ?

Comparer les paramètres et probabilités de régime dans le temps :

- vitesse de mean reversion ;
- half-life ;
- volatilité ;
- niveau d'équilibre ;
- durée des régimes ;
- fréquence des transitions.

### Test B — Est-ce que cette information améliore réellement le trading ?

Comparer :

1. pairs trading classique ;
2. pairs trading avec paramètres dynamiques ;
3. pairs trading conditionné par le régime.

Le deuxième test est le vrai test économique.

---

# 3. Construction du spread

Le papier définit simplement :

\[
x_t = S_1(t) - S_2(t)
\]

où `S1` et `S2` sont les prix des deux actifs.

Le papier donne notamment comme exemples :

- Coca-Cola / Pepsi
- Ford / General Motors

L'objectif est d'utiliser deux actifs économiquement liés.

## Attention

Le papier utilise un **spread de différence de prix** dans sa formulation :

\[
x_t = S_{1,t} - S_{2,t}
\]

Il ne formule pas ici le spread classique :

\[
\log(S_{1,t}) - \beta \log(S_{2,t})
\]

avec un hedge ratio estimé par régression.

Pour une reproduction fidèle, commencer par la définition du papier.

Pour une version de recherche plus réaliste, tester ensuite plusieurs définitions du spread :

- différence brute ;
- log-prix ;
- spread avec hedge ratio ;
- hedge ratio rolling ;
- hedge ratio estimé uniquement sur le train.

Ne mélange pas ces variantes dans le premier test : commence par reproduire le modèle le plus fidèlement possible.

---

# 4. Modèle de mean reversion

Le modèle de base du papier est :

\[
x_{k+1}-x_k = (a-bx_k)\tau + \sigma\sqrt{\tau}\epsilon_{k+1}
\]

avec :

- \(a \geq 0\)
- \(b > 0\)
- \(\sigma \geq 0\)
- \(\epsilon_{k+1}\sim N(0,1)\)
- \(\tau\) = intervalle temporel.

On peut le réécrire :

\[
x_{k+1}=a\tau +(1-b\tau)x_k+\sigma\sqrt{\tau}\epsilon_{k+1}
\]

Le papier définit :

\[
A=a\tau
\]

\[
B=1-b\tau
\]

\[
C=\sigma\sqrt{\tau}
\]

donc :

\[
x_{k+1}=A+Bx_k+C\epsilon_{k+1}
\]

C'est cette forme qui est utilisée dans le filtre.

---

# 5. Interprétation intuitive des paramètres

## `a`

Terme de drift/intercept.

## `b`

**Vitesse de mean reversion.**

Plus `b` est élevé, plus le spread revient rapidement vers son niveau d'équilibre.

## `σ`

**Volatilité du spread.**

## Niveau d'équilibre

À partir de :

\[
a-bx=0
\]

on obtient :

\[
\mu=\frac{a}{b}
\]

Donc le régime `i` possède son propre niveau d'équilibre :

\[
\mu_i=\frac{a_i}{b_i}
\]

## Half-life

Le papier ne présente pas explicitement une formule de trading basée sur la half-life, mais pour l'analyse pratique on peut dériver une approximation :

\[
HL \approx \frac{\ln(2)}{b}
\]

Pour un modèle discret, une approximation plus cohérente avec le coefficient AR(1) est :

\[
HL \approx \frac{\ln(0.5)}{\ln(|B|)}
\]

avec :

\[
B=1-b\tau
\]

**À utiliser comme métrique diagnostique**, pas comme signal officiellement proposé par le papier.

---

# 6. Introduction du régime de marché

Le point fondamental du papier est que les paramètres deviennent dépendants du régime :

\[
a \rightarrow a_i
\]

\[
b \rightarrow b_i
\]

\[
\sigma \rightarrow \sigma_i
\]

si le régime courant est `i`.

Le modèle devient :

\[
x_{k+1}
=
a_i\tau
+
(1-b_i\tau)x_k
+
\sigma_i\sqrt{\tau}\epsilon_{k+1}
\]

ou :

\[
x_{k+1}=A_i+B_ix_k+C_i\epsilon_{k+1}
\]

Ainsi chaque régime possède :

| Paramètre | Signification |
|---|---|
| `a_i` | drift/intercept du régime |
| `b_i` | vitesse de mean reversion |
| `σ_i` | volatilité |
| `μ_i = a_i/b_i` | niveau d'équilibre |
| `B_i = 1-b_iτ` | coefficient AR |
| `C_i = σ_i√τ` | écart-type de l'innovation |

---

# 7. Régimes = chaîne de Markov

Le régime latent est une chaîne de Markov finie :

\[
Z_k\in\{e_1,e_2,\ldots,e_N\}
\]

Le papier suggère qu'en pratique `N` est petit, avec notamment le cas :

\[
N=2
\]

par exemple :

- régime 1 = marché « good »
- régime 2 = marché « bad »

Mais **les noms good/bad ne sont pas imposés statistiquement**.

Un régime doit être interprété à partir des paramètres estimés.

---

# 8. Matrice de transition

Le papier définit :

\[
\pi_{ji}
=
P(Z_{k+1}=e_j|Z_k=e_i)
\]

La matrice :

\[
\Pi=(\pi_{ji})
\]

contient donc les probabilités de transition.

Pour deux régimes :

```text
             régime futur
             1       2
régime 1   p11     p21
régime 2   p12     p22
```

Attention à la convention d'indexation du papier :

\[
\pi_{ji}=P(Z_{k+1}=e_j|Z_k=e_i)
\]

Donc la colonne `i` correspond au régime actuel et la ligne `j` au régime futur.

Chaque colonne doit sommer à 1 :

\[
\sum_j \pi_{ji}=1
\]

---

# 9. Pourquoi une chaîne de Markov ?

La chaîne de Markov introduit une **persistance des régimes**.

Exemple :

```text
Régime 1 ──> Régime 1
   │
   └───────> Régime 2

Régime 2 ──> Régime 2
   │
   └───────> Régime 1
```

Si :

\[
P(Z_{t+1}=1|Z_t=1)=0.97
\]

le régime 1 est très persistant.

Une première approximation de la durée attendue d'un régime `i` est :

\[
E[D_i]\approx \frac{1}{1-p_{ii}}
\]

Cette formule est une interprétation pratique de la matrice de transition ; elle n'est pas une métrique de trading explicitement proposée dans le papier.

---

# 10. Le régime est caché

C'est extrêmement important.

On ne fait PAS :

```text
if VIX > 30:
    regime = 2
```

Le papier ne définit pas le régime avec une variable observable externe.

Le régime est **latent** :

```text
spread
   ↓
probabilité régime 1
probabilité régime 2
...
```

L'information utilisée pour inférer `Z_t` vient du spread observé.

---

# 11. Le filtre de régime

Le papier construit un filtre récursif.

Il définit :

\[
\hat Z_k
=
E[Z_k|\mathcal F_k^x]
\]

où :

\[
\mathcal F_k^x
=
\sigma\{x_l,l\leq k\}
\]

Autrement dit :

> estimation du régime courant conditionnellement à toutes les observations du spread disponibles jusqu'à `k`.

Le vecteur :

\[
\hat Z_k
\]

contient les probabilités conditionnelles des différents régimes.

Pour deux régimes :

```text
P(regime 1 | data jusqu'à t) = 0.82
P(regime 2 | data jusqu'à t) = 0.18
```

On peut alors dire que le modèle estime :

```text
regime 1 avec probabilité 82 %
regime 2 avec probabilité 18 %
```

---

# 12. Formule centrale du filtre

Le papier définit :

\[
q_k=E[\Lambda_k Z_k|\mathcal F_k^x]
\]

puis :

\[
\hat Z_k
=
\frac{q_k}{\langle q_k,\mathbf 1\rangle}
\]

La récursion principale est :

\[
q_k=\Pi D(k)q_{k-1}
\]

où `D(k)` est une matrice diagonale contenant les densités associées à chaque régime.

---

# 13. Densité associée à chaque régime

Pour chaque régime `i` :

\[
\phi_i(k)
=
\frac{
\phi\left(
\frac{x_k-A_i-B_ix_{k-1}}{C_i}
\right)
}{
C_i\phi(x_k)
}
\]

avec :

\[
\phi(z)=\frac{1}{\sqrt{2\pi}}e^{-z^2/2}
\]

et :

\[
D(k)=diag(\phi_1(k),...,\phi_N(k))
\]

En pratique, l'idée importante est beaucoup plus simple :

> Pour chaque nouveau spread `x_k`, calculer à quel point l'observation est compatible avec chacun des régimes.

Le régime dont la dynamique explique le mieux l'observation reçoit davantage de poids.

---

# 14. Version intuitive du filtre

Pour chaque nouvelle observation :

### Étape 1 — prédiction

À partir du régime précédent :

\[
P(Z_k|x_{1:k-1})
\]

utiliser la matrice de transition `Π`.

### Étape 2 — calcul de la vraisemblance

Pour chaque régime :

\[
x_k \sim N(A_i+B_ix_{k-1},C_i^2)
\]

calculer :

\[
L_i=P(x_k|Z_k=i,x_{k-1})
\]

### Étape 3 — mise à jour

Combiner :

```text
probabilité précédente
×
probabilité de transition
×
vraisemblance de l'observation
```

### Étape 4 — normalisation

Obtenir :

```text
P(Z_t=1 | x_1,...,x_t)
...
P(Z_t=N | x_1,...,x_t)
```

C'est cette probabilité qui est utile pour le trading.

---

# 15. Estimation des paramètres

Les paramètres inconnus sont :

\[
A,B,C,\Pi
\]

ou de manière équivalente :

\[
a_i,b_i,\sigma_i,\Pi
\]

Le papier utilise l'algorithme **Expectation-Maximization (EM)**.

Le principe :

```text
Initialisation des paramètres
          ↓
      E-step
          ↓
Estimation probabiliste des états cachés
          ↓
      M-step
          ↓
Ré-estimation des paramètres
          ↓
     convergence ?
      ↙       ↘
    non       oui
     ↓         ↓
  répéter    modèle final
```

---

# 16. E-step : idée pratique

Comme `Z_t` n'est pas observé, on ne sait pas directement quelles observations appartiennent à quel régime.

On calcule donc des **poids probabilistes**.

Par exemple :

```text
observation t

régime 1 : 0.90
régime 2 : 0.10
```

Cette observation contribue donc principalement à l'estimation du régime 1.

Pour chaque régime, le papier utilise des sommes conditionnelles du type :

\[
G_k^r(f)
=
\sum_{\ell=1}^{k}
\langle Z_{\ell-1},e_r\rangle f(\ell)
\]

et :

\[
J_k^r
=
\sum_{\ell=1}^{k}
\langle Z_{\ell-1},e_r\rangle
\]

Comme `Z` est caché, ces quantités sont remplacées par leurs espérances conditionnelles.

---

# 17. Statistiques nécessaires pour ré-estimer `A`, `B`, `C`

Le papier utilise essentiellement les moments :

\[
1
\]

\[
x_t
\]

\[
x_{t-1}
\]

\[
x_t^2
\]

\[
x_{t-1}^2
\]

\[
x_tx_{t-1}
\]

pondérés par la probabilité d'appartenance au régime.

Cela ressemble donc fortement à une **régression AR(1) pondérée par les probabilités de régime**.

---

# 18. M-step : mise à jour de A

Pour le régime `i` :

\[
\hat A_i
=
\frac{
\hat G_i(x_t)-B_i\hat G_i(x_{t-1})
}{
\hat J_i
}
\]

Interprétation :

> moyenne pondérée de `x_t - B_i x_{t-1}` sur les observations attribuées probabilistiquement au régime `i`.

---

# 19. M-step : mise à jour de B

\[
\hat B_i
=
\frac{
\hat G_i(x_tx_{t-1})
-
A_i\hat G_i(x_{t-1})
}{
\hat G_i(x_{t-1}^2)
}
\]

C'est l'équivalent d'une estimation de coefficient AR pondérée par la probabilité d'être dans le régime.

Ensuite :

\[
b_i=\frac{1-B_i}{\tau}
\]

---

# 20. M-step : mise à jour de C

Le papier définit une quantité intermédiaire :

\[
H_i
=
G_i(x_t^2)
+
J_iA_i^2
+
G_i(x_{t-1}^2)B_i^2
-
2A_iG_i(x_t)
-
2B_iG_i(x_tx_{t-1})
+
2A_iB_iG_i(x_{t-1})
\]

puis :

\[
\hat C_i
=
\sqrt{\frac{H_i}{J_i}}
\]

et donc :

\[
\sigma_i=\frac{C_i}{\sqrt{\tau}}
\]

---

# 21. M-step : matrice de transition

Le papier estime :

\[
\hat\pi_{ji}
=
\frac{\hat N_{ij}}{\hat J_i}
\]

où :

\[
N_{ij}
=
\sum_{\ell=1}^{k}
\langle Z_{\ell-1},e_i\rangle
\langle Z_\ell,e_j\rangle
\]

représente le nombre de transitions du régime `i` vers le régime `j`.

Comme le régime est caché, on utilise son espérance conditionnelle :

\[
\hat N_{ij}
=
E[N_{ij}|\mathcal F_k^x]
\]

Donc :

```text
nombre attendu de transitions i → j
------------------------------------
nombre attendu de périodes dans i
```

---

# 22. Résumé de l'EM à coder

Pour `N=2` :

```python
initialize A[2], B[2], C[2], Pi[2,2]

repeat until convergence:

    # E-step
    # ---------------------------------
    # Calculer les probabilités de régime
    # à partir du spread et des paramètres actuels.

    regime_prob = filter(x, A, B, C, Pi)

    # Calculer les probabilités attendues
    # des transitions i -> j.

    transition_prob = expected_transitions(...)

    # M-step
    # ---------------------------------
    for i in regimes:

        update A[i]
        update B[i]
        update C[i]

    update Pi

    check log_likelihood improvement
```

**Pour une première implémentation, il est raisonnable d'utiliser un HMM Gaussian/AR(1) standard comme vérification numérique de l'implémentation. Mais si le but est une réplication stricte, les équations du papier doivent rester la référence.**

---

# 23. Paramétrisation directement exploitable

Pour coder plus facilement, tu peux travailler avec :

\[
x_t = \alpha_i+\beta_i x_{t-1}+\epsilon_t
\]

où :

\[
\alpha_i=A_i
\]

\[
\beta_i=B_i
\]

\[
Var(\epsilon_t)=C_i^2
\]

Puis récupérer :

\[
b_i=\frac{1-\beta_i}{\tau}
\]

\[
a_i=\frac{\alpha_i}{\tau}
\]

\[
\sigma_i=\frac{C_i}{\sqrt{\tau}}
\]

et :

\[
\mu_i=\frac{a_i}{b_i}
\]

Cette représentation est beaucoup plus simple à manipuler en Python.

---

# 24. Ce que le papier permet d'obtenir

À chaque date `t`, le modèle peut fournir :

```text
spread
prob_regime_1
prob_regime_2
...
A_1, B_1, C_1
A_2, B_2, C_2
...
transition_matrix
```

À partir de `B_i` et `C_i`, tu peux calculer :

### Mean reversion

\[
b_i=\frac{1-B_i}{\tau}
\]

### Half-life

\[
HL_i\approx\frac{\ln(0.5)}{\ln(|B_i|)}
\]

### Niveau d'équilibre

\[
\mu_i=\frac{a_i}{b_i}
\]

### Volatilité

\[
\sigma_i=\frac{C_i}{\sqrt{\tau}}
\]

---

# 25. Comment transformer le modèle en stratégie de trading

**C'est ici qu'il faut faire attention : le papier décrit le modèle de dynamique et son estimation, mais ne donne pas une stratégie complète avec des seuils de z-score modernes, stop-loss, sizing, coûts, etc.**

Il faut donc définir explicitement une règle de trading.

Une première traduction naturelle :

## Étape 1

Calculer le régime probable :

\[
p_i(t)=P(Z_t=i|x_{1:t})
\]

## Étape 2

Pour chaque régime, calculer son équilibre :

\[
\mu_i=\frac{a_i}{b_i}
\]

## Étape 3

Calculer l'écart au régime :

\[
d_i(t)=x_t-\mu_i
\]

## Étape 4

Standardiser par la volatilité du régime.

Une version simple :

\[
z_i(t)=
\frac{x_t-\mu_i}{\sigma_{x,i}}
\]

où `σ_x,i` est l'écart-type du spread associé au régime.

## Étape 5

Combiner les régimes.

Par exemple :

\[
z_t^{mix}
=
\sum_i p_i(t)z_i(t)
\]

**Attention : cette combinaison est une extension pratique proposée pour le test, pas une équation de trading du papier.**

---

# 26. Variante plus propre pour le premier test

Ne mélange pas immédiatement les régimes.

Utilise :

```text
si P(regime 1) > 0.70 :
    utiliser les paramètres du régime 1

si P(regime 2) > 0.70 :
    utiliser les paramètres du régime 2

sinon :
    pas de nouveau trade
```

Cela permet de tester clairement :

> Est-ce que les paramètres spécifiques au régime améliorent le signal ?

---

# 27. Signal de pairs trading

Une règle de départ simple :

```text
z > +entry_threshold
    → short spread

z < -entry_threshold
    → long spread

|z| < exit_threshold
    → sortie
```

Exemple :

```text
entrée : |z| > 2
sortie : |z| < 0.5
```

Mais **2 / 0.5 ne viennent pas du papier**.

Ce sont des paramètres expérimentaux à tester.

Il faut idéalement tester plusieurs seuils sans choisir celui qui maximise l'OOS.

---

# 28. Comment traduire le signal en positions

Si :

\[
x_t=S_{1,t}-S_{2,t}
\]

alors :

### Spread positif / trop élevé

```text
short S1
long S2
```

### Spread négatif / trop faible

```text
long S1
short S2
```

Si tu utilises ensuite un hedge ratio `β`, la construction de position doit être adaptée.

---

# 29. Test minimal recommandé

Commence par :

```text
2 actifs
daily data
N = 2 régimes
spread = S1 - S2
τ = 1 jour
```

Puis :

```text
train
    ↓
EM
    ↓
A, B, C, Pi
    ↓
filtre
    ↓
probabilités de régime
    ↓
signal
    ↓
OOS
```

Ne commence pas directement avec 50 paires et 10 régimes.

---

# 30. Dataset idéal pour le premier test

Utiliser :

- deux actions liquides ;
- historique long ;
- prix ajustés ;
- fréquence quotidienne ;
- données sans trous importants.

Exemples de paires économiquement liées :

```text
KO / PEP
F / GM
XOM / CVX
JPM / BAC
```

Ces tickers sont uniquement des exemples ; il faut vérifier empiriquement qu'ils constituent une paire appropriée sur la période étudiée.

---

# 31. Pair selection

Le papier explique que le pairs trading commence par l'identification de deux instruments similaires, mais son modèle ne fournit pas un protocole moderne complet de sélection de pairs.

Pour un test sérieux :

### Option 1 — reproduction simple

Choisir quelques paires économiquement évidentes.

### Option 2 — sélection quantitative

À chaque date de formation :

1. sélectionner l'univers ;
2. calculer corrélation ;
3. tester cointegration ;
4. sélectionner les paires ;
5. figer les paires ;
6. tester uniquement sur la période suivante.

**Ne sélectionne jamais les paires en utilisant toute la période historique avant le backtest.**

---

# 32. Walk-forward recommandé

Architecture :

```text
TRAIN
████████████████████

TEST
                    ████████
```

Puis on avance :

```text
TRAIN
        ████████████████████

TEST
                            ████████
```

Puis :

```text
TRAIN
                ████████████████████

TEST
                                    ████████
```

À chaque nouvelle fenêtre :

1. ré-estimer les paramètres ;
2. ré-estimer les régimes ;
3. générer les signaux ;
4. trader uniquement après la date de calibration.

---

# 33. Point critique : look-ahead bias

Il ne faut surtout pas faire :

```python
fit_model(all_data)
regime = model.predict(all_data)
backtest(all_data)
```

C'est insuffisant pour démontrer une performance OOS.

Il faut :

```python
for each walk_forward_window:

    train = past_data
    test = future_data

    fit(train)

    for t in test:
        update_filter_using_only_information_available_at_t
        generate_signal
```

Le filtre doit être causal.

---

# 34. Filtrage vs smoothing

Pour le trading live, utiliser :

\[
P(Z_t|x_{1:t})
\]

et non une probabilité utilisant des observations futures :

\[
P(Z_t|x_{1:T}),\quad T>t
\]

La deuxième est un **smoother** et introduit de l'information future si elle est utilisée directement pour générer les trades historiques.

Le papier développe un **filtre récursif**, ce qui est particulièrement intéressant pour une utilisation online.

---

# 35. Ce qu'il faut mesurer statistiquement

Pour chaque modèle :

### Rendement

- CAGR
- rendement annuel
- rendement mensuel
- rendement par année

### Risque

- volatilité annualisée
- Max Drawdown
- VaR
- CVaR
- downside deviation

### Ratios

- Sharpe
- Sortino
- Calmar
- Profit Factor

### Trading

- nombre de trades
- turnover
- durée moyenne
- exposition moyenne
- win rate
- average win
- average loss
- payoff ratio

### Robustesse

- bootstrap
- Monte Carlo
- perturbation des paramètres
- variation des coûts
- variation des seuils
- variation des fenêtres
- variation de l'univers
- périodes de crise

---

# 36. Les tests les plus importants

Le résultat principal à chercher n'est pas :

> « Le modèle a un Sharpe de 2.1 ».

Le test important est :

\[
Performance_{RegimeSwitching}
>
Performance_{Baseline}
\]

sur des périodes **OOS**.

---

# 37. Les quatre modèles à comparer

## Model A — Baseline

Pairs trading classique :

```text
spread
→ z-score rolling
→ entrée/sortie
```

## Model B — Mean reversion dynamique

```text
spread
→ estimation rolling de AR(1)/OU
→ z-score dynamique
```

## Model C — Elliott & Bradrania

```text
spread
→ 2-state Markov regime switching
→ paramètres a_i, b_i, σ_i
→ probabilités de régime
→ signal conditionnel
```

## Model D — autre détecteur de régime

Par exemple :

```text
SJM
HMM standard
ou autre modèle
```

Le but est de savoir si le gain vient réellement du regime switching ou simplement du fait d'avoir un modèle plus sophistiqué.

---

# 38. Expérience idéale

Faire exactement :

```text
                 ┌── Baseline
                 │
same pair ───────┼── Rolling OU
                 │
                 ├── Elliott-Bradrania
                 │
                 └── SJM/HMM
```

Même :

- période ;
- paire ;
- coûts ;
- règles d'entrée ;
- règles de sortie ;
- sizing ;
- fréquence.

Sinon la comparaison n'est pas propre.

---

# 39. Test des régimes eux-mêmes

Avant même de trader, regarder :

## 1. `b_i`

Les régimes ont-ils vraiment des vitesses de mean reversion différentes ?

Exemple :

```text
Regime 1 : b = 0.08
Regime 2 : b = 0.015
```

C'est potentiellement économiquement intéressant.

---

## 2. Half-life

```text
Regime 1 : 5 jours
Regime 2 : 45 jours
```

Cela indique une modification importante de la dynamique.

---

## 3. σ_i

```text
Regime 1 : faible volatilité
Regime 2 : forte volatilité
```

---

## 4. μ_i

Le niveau d'équilibre change-t-il ?

```text
Regime 1 : μ = 10
Regime 2 : μ = 13
```

Si oui, un z-score basé sur une moyenne fixe pourrait être mal spécifié.

---

# 40. Test très important : séparation des régimes

Tracer :

```text
date
│
│ regime 1 probability
│ ███████████
│
│ regime 2 probability
│       █████████████
│
└────────────────────── time
```

Puis superposer :

- spread ;
- volatilité ;
- drawdown du spread ;
- signal ;
- P&L.

Le but est de vérifier si les régimes correspondent à de véritables changements de comportement du spread.

---

# 41. Ce qui serait un résultat convaincant

Exemple hypothétique :

```text
                    Sharpe    MaxDD    PF
Baseline             0.82     -18%   1.18
Rolling OU            1.03     -14%   1.27
Regime Switching      1.42      -9%   1.43
```

Puis vérifier :

```text
OOS Sharpe
OOS MaxDD
stabilité par sous-période
stabilité par paire
stabilité des paramètres
```

Un résultat comme celui-ci serait beaucoup plus intéressant qu'un backtest unique optimisé.

---

# 42. Ce qui ne serait PAS convaincant

```text
Sharpe = 2.8
```

mais :

- une seule paire ;
- 12 trades ;
- paramètres optimisés sur tout l'historique ;
- pas de coûts ;
- pas de slippage ;
- régime estimé avec données futures ;
- sélection de paire faite après observation des résultats ;
- énorme drawdown avant la fin ;
- performance concentrée sur 2008.

Cela ne permettrait pas de conclure que le modèle fonctionne.

---

# 43. Initialisation des paramètres

L'EM peut dépendre de l'initialisation.

Tester plusieurs initialisations :

```text
seed 1
seed 2
seed 3
...
seed 20
```

Puis comparer :

- log-likelihood finale ;
- paramètres ;
- matrice de transition ;
- classification des régimes ;
- performance OOS.

Si le modèle donne des résultats très différents selon l'initialisation, c'est un point important de fragilité.

---

# 44. Label switching

Avec deux régimes :

```text
Regime 1
Regime 2
```

le modèle peut parfaitement décider :

```text
ancien regime 1 = nouveau regime 2
ancien regime 2 = nouveau regime 1
```

Ce n'est pas une différence économique.

Il faut donc identifier les régimes par leurs paramètres, par exemple :

```text
Regime FAST mean-reversion
Regime SLOW mean-reversion
```

plutôt que par leur numéro.

---

# 45. Contraintes à imposer au modèle

Le modèle suppose :

\[
b_i>0
\]

donc :

\[
B_i<1
\]

Pour une dynamique mean-reverting stable, il faut également surveiller :

\[
|B_i|<1
\]

En pratique, vérifier après chaque estimation :

```python
assert sigma_i > 0
assert b_i > 0
assert abs(B_i) < 1
```

Si l'EM produit un régime explosif, il faut traiter cela comme un problème de calibration/modèle, pas simplement supprimer le résultat.

---

# 46. Stabilité du modèle

Tester :

### Nombre de régimes

```text
N = 2
N = 3
```

Mais ne pas multiplier les régimes inutilement.

### Fenêtre de calibration

```text
1 an
2 ans
3 ans
5 ans
```

### Fréquence

```text
daily
weekly
```

### Paires

Plusieurs paires indépendantes.

---

# 47. Coûts de transaction

Le papier n'est pas un backtest moderne détaillant tous les coûts.

Pour ton test, intégrer :

```text
commission
bid/ask spread
slippage
financing / borrow si nécessaire
```

Pour une stratégie long-short actions, le coût de short/borrow peut devenir important.

Tester au minimum :

```text
0 bps
5 bps
10 bps
20 bps
50 bps
```

par transaction, en explicitant exactement ce que représente le coût.

---

# 48. Sizing

Pour un premier test :

```text
fixed notional
```

puis :

```text
volatility targeting
```

Par exemple :

\[
w_t \propto \frac{1}{\hat\sigma_t}
\]

Mais le sizing dynamique est une **extension expérimentale**, pas une composante spécifiée dans le papier.

Pour isoler l'effet du modèle, comparer d'abord avec le même sizing pour tous les modèles.

---

# 49. Test de robustesse des seuils

Si :

```text
entry = 2
exit = 0.5
```

donne un excellent résultat, tester :

```text
entry = 1.5 / 1.75 / 2 / 2.25 / 2.5
exit  = 0.25 / 0.5 / 0.75 / 1
```

Mais surtout :

> Ne sélectionner pas le meilleur couple de paramètres sur l'OOS final.

Utiliser le train/validation pour choisir les paramètres et réserver l'OOS final pour la véritable évaluation.

---

# 50. Test de falsification

Un excellent test consiste à demander :

> Est-ce que le regime switching apporte réellement quelque chose ?

Faire plusieurs comparaisons :

### Test 1

Même spread, même signal, mais :

```text
paramètres fixes
vs
paramètres conditionnels au régime
```

### Test 2

Même paramètres, mais :

```text
régime réel
vs
régime randomisé
```

Si le régime randomisé produit presque la même performance, l'information de régime est probablement peu utile.

### Test 3

Permuter les labels des régimes.

La performance ne devrait pas changer.

---

# 51. Test de permutation particulièrement intéressant

Conserver :

- mêmes prix ;
- mêmes trades ;
- même nombre de régimes.

Mais randomiser les régimes.

Puis comparer :

\[
Sharpe_{real}
\]

à une distribution :

\[
Sharpe_{randomized}
\]

Si le Sharpe réel est nettement supérieur à la distribution randomisée, cela apporte une preuve supplémentaire que la structure des régimes contient de l'information.

---

# 52. Test de persistance

À partir de `Π` :

\[
p_{ii}
\]

mesurer la persistance de chaque régime.

Vérifier :

```text
p11
p22
```

et la durée attendue.

Un modèle avec :

```text
p11 = 0.999
p22 = 0.999
```

peut être trop persistant.

Un modèle avec :

```text
p11 = 0.51
p22 = 0.52
```

ne décrit pratiquement pas de régimes persistants.

Ce n'est pas nécessairement mauvais, mais il faut comprendre ce que le modèle apprend.

---

# 53. Test économique par régime

Calculer le P&L séparément lorsque :

```text
P(regime 1) > 0.7
```

et :

```text
P(regime 2) > 0.7
```

Puis :

```text
Sharpe regime 1
Sharpe regime 2
MaxDD regime 1
MaxDD regime 2
PF regime 1
PF regime 2
```

C'est probablement l'une des analyses les plus utiles pour comprendre le modèle.

---

# 54. Test : le régime change-t-il vraiment le signal ?

Comparer :

```text
z-score classique
```

à :

```text
z-score basé sur μ_i et σ_i
```

Si les signaux sont quasiment identiques, le régime switching n'apporte peut-être pas beaucoup d'information.

Regarder :

\[
corr(signal_{baseline}, signal_{regime})
\]

ainsi que :

- nombre de trades différents ;
- dates d'entrée différentes ;
- dates de sortie différentes.

---

# 55. Test : le régime améliore-t-il surtout le risk management ?

Il est possible que le régime ne donne pas beaucoup plus de rendement mais réduise le risque.

Donc regarder séparément :

```text
return
volatility
MaxDD
Sortino
Calmar
tail losses
```

Un modèle qui produit :

```text
même CAGR
mais
50 % moins de MaxDD
```

peut être économiquement très intéressant.

---

# 56. Implémentation Python recommandée

Structure :

```text
project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data.ipynb
│   ├── 02_spread.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_regime_model.ipynb
│   ├── 05_backtest.ipynb
│   └── 06_robustness.ipynb
│
├── src/
│   ├── data.py
│   ├── spread.py
│   ├── model.py
│   ├── filter.py
│   ├── em.py
│   ├── signal.py
│   ├── backtest.py
│   └── metrics.py
│
└── results/
```

---

# 57. Ordre de développement recommandé

## Phase 1 — Spread

```text
load prices
→ align dates
→ calculate spread
→ plot spread
```

## Phase 2 — modèle sans régime

Estimer :

\[
x_t=A+Bx_{t-1}+\epsilon_t
\]

Vérifier :

- `B`
- `b`
- half-life
- residuals
- normality approximative.

## Phase 3 — modèle 2 régimes

Implémenter :

```text
A1 B1 C1
A2 B2 C2
Pi
```

## Phase 4 — filtre

Obtenir :

```text
P1(t)
P2(t)
```

## Phase 5 — EM

Faire converger :

```text
parameters
→ probabilities
→ parameters
→ probabilities
...
```

## Phase 6 — stratégie

Ajouter :

```text
entry
exit
positions
costs
```

## Phase 7 — OOS

Faire le walk-forward.

## Phase 8 — robustesse

Seulement après.

---

# 58. Pseudo-code complet

```python
# ==========================================
# 1. DATA
# ==========================================

prices = load_prices()

S1 = prices["asset_1"]
S2 = prices["asset_2"]

x = S1 - S2


# ==========================================
# 2. WALK FORWARD
# ==========================================

for train_start, train_end, test_start, test_end in windows:

    x_train = x.loc[train_start:train_end]
    x_test  = x.loc[test_start:test_end]


    # ======================================
    # 3. INITIALIZE
    # ======================================

    params = initialize_two_regime_model(x_train)


    # ======================================
    # 4. EM
    # ======================================

    for iteration in range(max_iter):

        probs = filter_model(
            x_train,
            params
        )

        expected_transitions = estimate_transitions(
            x_train,
            probs
        )

        params_new = M_step(
            x_train,
            probs,
            expected_transitions
        )

        if converged(params, params_new):
            break

        params = params_new


    # ======================================
    # 5. OOS FILTER
    # ======================================

    probs_test = filter_online(
        x_test,
        params,
        initial_state=last_train_state
    )


    # ======================================
    # 6. SIGNAL
    # ======================================

    signal = generate_regime_signal(
        x_test,
        probs_test,
        params
    )


    # ======================================
    # 7. BACKTEST
    # ======================================

    pnl = backtest(
        signal,
        prices.loc[test_start:test_end],
        costs=True
    )


    save_results(pnl, params, probs_test)
```

---

# 59. Résultats à sauvegarder à chaque fenêtre

Pour chaque fenêtre :

```text
train_start
train_end
test_start
test_end

A1
B1
C1
a1
b1
sigma1
mu1
half_life1

A2
B2
C2
a2
b2
sigma2
mu2
half_life2

Pi11
Pi12
Pi21
Pi22

log_likelihood
iterations

OOS return
OOS Sharpe
OOS MaxDD
OOS trades
OOS turnover
```

Cela permettra ensuite d'étudier la stabilité du modèle.

---

# 60. Diagnostics indispensables

## Plot 1 — Spread

```text
spread
```

## Plot 2 — Probabilités

```text
P(regime 1)
P(regime 2)
```

## Plot 3 — paramètres

```text
b_1
b_2
σ_1
σ_2
μ_1
μ_2
```

## Plot 4 — trades

Sur le spread :

```text
long entries
short entries
exits
```

## Plot 5 — equity curve

Comparer :

```text
baseline
regime switching
```

---

# 61. Questions auxquelles ton test doit répondre

### Question 1

Les données montrent-elles réellement plusieurs dynamiques de mean reversion ?

### Question 2

Les régimes sont-ils stables ?

### Question 3

Les régimes ont-ils des half-lives différentes ?

### Question 4

Les régimes ont-ils des volatilités différentes ?

### Question 5

La matrice de transition montre-t-elle une vraie persistance ?

### Question 6

Les probabilités de régime sont-elles suffisamment confiantes pour être utilisées ?

### Question 7

Le regime switching améliore-t-il le signal ?

### Question 8

Améliore-t-il le Sharpe ?

### Question 9

Réduit-il le MaxDD ?

### Question 10

Le gain survit-il aux coûts ?

### Question 11

Le gain survit-il OOS ?

### Question 12

Le gain existe-t-il sur plusieurs paires ?

### Question 13

Le gain existe-t-il sur plusieurs périodes ?

### Question 14

Le gain survit-il à une variation raisonnable des paramètres ?

---

# 62. Ce que le papier ne fournit PAS

Il est important de ne pas attribuer au papier des éléments qu'il ne développe pas.

Le papier ne fournit pas une recette complète pour :

- sélectionner automatiquement les meilleures paires ;
- choisir un seuil de z-score ;
- définir un stop-loss ;
- définir un take-profit ;
- faire du position sizing moderne ;
- intégrer les coûts de transaction détaillés ;
- faire du portefeuille multi-paires ;
- optimiser le turnover ;
- faire une analyse Deflated Sharpe ;
- faire du White's Reality Check ;
- faire du SPA test ;
- faire du Monte Carlo de stratégie ;
- fournir une preuve moderne de rentabilité OOS.

Ces éléments doivent être ajoutés pour transformer le modèle statistique en véritable recherche quantitative.

---

# 63. Ce que le modèle est réellement

La meilleure façon de voir le papier :

```text
Ce n'est PAS :
"une stratégie de trading complète"

C'est :
"un modèle probabiliste permettant de représenter
une dynamique de spread mean-reverting dont les
paramètres changent selon un régime latent."
```

La stratégie vient ensuite.

---

# 64. Première expérience recommandée

Pour ne pas complexifier inutilement :

### Univers

Une seule paire économiquement cohérente.

### Fréquence

Daily.

### Régimes

`N = 2`.

### Spread

\[
x_t=S_{1,t}-S_{2,t}
\]

### Modèle

\[
x_t=A_i+B_ix_{t-1}+C_i\epsilon_t
\]

### Estimation

EM.

### Filtre

Probabilités online.

### Signal

Régime dominant si :

\[
P_i>0.70
\]

puis mean-reversion conditionnelle.

### Comparaison

```text
Baseline z-score
vs
Regime switching
```

### Validation

Walk-forward.

### Coûts

Inclus.

---

# 65. Deuxième expérience

Après avoir validé le modèle sur une paire :

```text
10–50 paires
```

avec sélection hors échantillon.

Pour chaque paire :

```text
fit
→ regime detection
→ signal
→ OOS P&L
```

Puis portefeuille :

```text
equal weight
```

puis éventuellement :

```text
volatility weighting
```

---

# 66. Troisième expérience : régime global vs régime du spread

Une extension intéressante est de comparer :

### Modèle du papier

Le régime est inféré du spread :

```text
spread → regime
```

### Modèle alternatif

Le régime est inféré de variables de marché :

```text
VIX
SPX
rates
credit spreads
volatility
market trend
→ regime
```

Puis :

```text
market regime
→ pair trading parameters
```

Cela permet de tester si le régime pertinent est :

- un régime **spécifique au spread** ;
- ou un régime **global de marché**.

Cette expérience dépasse le papier.

---

# 67. Quatrième expérience : comparaison avec HMM standard

Le modèle peut être comparé à un HMM Gaussian/AR(1).

Même :

- données ;
- nombre de régimes ;
- période ;
- stratégie ;
- coûts.

Comparer :

```text
Elliott-Bradrania
vs
HMM
```

sur :

- log-likelihood ;
- stabilité ;
- probabilités ;
- performance OOS.

---

# 68. Cinquième expérience : comparaison avec Statistical Jump Model

Pour ton programme de recherche, c'est une comparaison particulièrement intéressante :

```text
Spread
 │
 ├── Elliott & Bradrania
 │
 ├── HMM
 │
 └── Statistical Jump Model
```

Puis utiliser exactement la même couche de trading au-dessus.

Le but est de comparer **les détecteurs de régime**, pas de comparer trois stratégies différentes.

---

# 69. Critère de succès recommandé

Ne définir pas le succès uniquement comme :

\[
Sharpe>1
\]

Utiliser une grille :

```text
Sharpe > 1
Sortino > 1
MaxDD < 15 %
Calmar acceptable
Profit Factor > 1.2
nombre de trades suffisant
OOS positif
robuste aux coûts
robuste aux paramètres
robuste aux sous-périodes
```

Pour un modèle destiné à être réellement exploitable, rechercher idéalement une amélioration **stable** plutôt qu'un Sharpe maximal.

---

# 70. Verdict attendu du test

À la fin du projet, pouvoir répondre clairement :

```text
1. Le modèle identifie-t-il des régimes statistiquement distincts ?
2. Ces régimes ont-ils des dynamiques de mean reversion différentes ?
3. Les régimes sont-ils persistants ?
4. Les probabilités sont-elles suffisamment stables ?
5. Le regime switching améliore-t-il le trading ?
6. L'amélioration existe-t-elle OOS ?
7. L'amélioration survit-elle aux coûts ?
8. L'amélioration existe-t-elle sur plusieurs paires ?
9. L'amélioration existe-t-elle sur plusieurs périodes ?
10. Le gain est-il supérieur à celui d'un HMM/SJM plus simple ou plus moderne ?
```

---

# 71. Résumé en une seule formule

Le cœur du papier peut être résumé comme :

\[
\boxed{
x_t=A_{Z_{t-1}}+B_{Z_{t-1}}x_{t-1}
+C_{Z_{t-1}}\epsilon_t
}
\]

avec :

\[
Z_t\sim Markov(\Pi)
\]

et :

\[
P(Z_t=i|x_{1:t})
\]

estimé récursivement.

Puis :

\[
A_i,\ B_i,\ C_i,\ \Pi
\]

sont estimés par EM.

---

# 72. Checklist avant de considérer le test comme valide

```text
[ ] Données propres
[ ] Prix ajustés
[ ] Dates correctement alignées
[ ] Spread défini avant le test
[ ] Pas de look-ahead
[ ] Pair selection uniquement avec données disponibles
[ ] EM uniquement sur train
[ ] Filtre online en OOS
[ ] Pas de smoothing futur
[ ] Coûts inclus
[ ] Slippage inclus
[ ] Short costs pris en compte si nécessaire
[ ] Plusieurs initialisations EM
[ ] N=2 testé
[ ] Paramètres économiquement interprétés
[ ] Baseline identique
[ ] Walk-forward
[ ] Plusieurs sous-périodes
[ ] Plusieurs paires
[ ] Sensibilité aux seuils
[ ] Sensibilité aux coûts
[ ] Monte Carlo / bootstrap
[ ] Analyse par régime
[ ] Comparaison baseline vs regime switching
```

---

# 73. Référence

Elliott, R. J., & Bradrania, R. (2018). **Estimating A Regime Switching Pairs Trading Model.** *Quantitative Finance*, 18(5), 877–883. DOI: 10.1080/14697688.2017.1403035.

Le document source fourni a été utilisé comme référence principale pour les équations, la structure du modèle, le filtre, l'algorithme EM et les estimations récursives.

