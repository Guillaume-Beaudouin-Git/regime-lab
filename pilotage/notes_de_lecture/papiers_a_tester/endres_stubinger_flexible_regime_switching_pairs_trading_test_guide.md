# Endres & Stübinger (2018) — Flexible Regime Switching Model + Pairs Trading
## Guide pratique pour reproduire et tester le papier

**Paper :** Sylvia Endres & Johannes Stübinger, *A flexible regime switching model with pairs trading application to the S&P 500 high-frequency stock returns*  
**Version :** FAU Discussion Papers in Economics No. 07/2018, 16 mai 2018.  
**Données originales :** S&P 500, minute par minute, 1998–2015.

---

# 1. Verdict : est-ce pertinent ?

## Oui — très pertinent pour ton programme de recherche.

Je considère ce papier **plus directement testable et plus intéressant économiquement que le papier Elliott & Bradrania (2018)** que tu viens de lire.

Pourquoi ?

Le papier ne s'arrête pas à :

```text
spread
→ régime latent
→ paramètres
```

Il construit une chaîne complète :

```text
prix
 ↓
spread
 ↓
détection automatique du nombre de régimes
 ↓
modèle Lévy-driven OU dans chaque régime
 ↓
sélection des meilleures paires
 ↓
Bollinger bands
 ↓
trading OOS
 ↓
transaction costs
 ↓
analyse des facteurs
 ↓
robustesse / bootstrap
```

Les auteurs utilisent en particulier un algorithme qui **choisit automatiquement le nombre de régimes**, au lieu d'imposer `N=2`. Ils l'appliquent ensuite à une stratégie de pairs trading haute fréquence. Le papier rapporte 93,73 % de rendement annualisé et un Sharpe de 3,92 après coûts sur leur échantillon 1998–2015. Ces chiffres sont évidemment à **répliquer**, pas à considérer comme acquis aujourd'hui. fileciteturn5file0L13-L40

### Mon classement pour toi

| Critère | Elliott & Bradrania | Endres & Stübinger |
|---|---:|---:|
| Régime switching | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Pairs trading | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Détection automatique du nombre de régimes | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Jumps / fat tails | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Application trading complète | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Backtest empirique | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Robustesse | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Facilité de réplication | ⭐⭐⭐⭐ | ⭐⭐ |
| Pertinence pour SJM/HMM research | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**Le gros défaut :** le papier est conçu pour du **high-frequency minute data**, avec une modélisation des jumps assez lourde. Tu ne dois donc pas commencer par reproduire l'intégralité du papier au niveau microstructure.

Le bon plan est de faire une **réplication simplifiée**, puis d'ajouter progressivement les composants sophistiqués.

---

# 2. L'idée fondamentale

Le papier part de l'idée suivante :

> Une paire d'actions peut avoir une dynamique de spread différente selon les périodes.

Le spread est défini :

\[
X_t =
\ln\left(\frac{S_A(t)}{S_A(0)}\right)
-
\ln\left(\frac{S_B(t)}{S_B(0)}\right)
\]

Ce qui revient à :

\[
X_t =
\ln S_A(t)-\ln S_B(t)
-
[\ln S_A(0)-\ln S_B(0)]
\]

Le terme constant n'affecte pas les variations du spread.

Donc, pour la réplication, tu peux essentiellement travailler avec :

\[
X_t=\ln S_A(t)-\ln S_B(t)
\]

Le spread change entre plusieurs régimes :

\[
Z_t\in\{1,\dots,r\}
\]

et `Z_t` est un processus de Markov continu.

Dans chaque régime :

\[
dX_t =
\theta_i(\mu_i(t)-X_t)dt+dL_{i,t}
\]

où :

- \(\theta_i\) = vitesse de mean reversion ;
- \(\mu_i(t)\) = niveau d'équilibre ;
- \(L_{i,t}\) = processus de Lévy ;
- `Z_t=i` = régime courant.

fileciteturn5file0L47-L73

---

# 3. Pourquoi Lévy plutôt que OU classique ?

C'est l'un des éléments les plus intéressants du papier.

Un OU classique suppose essentiellement :

\[
dX_t=\theta(\mu-X_t)dt+\sigma dW_t
\]

Donc les innovations sont gaussiennes.

Mais les spreads financiers peuvent présenter :

- queues épaisses ;
- outliers ;
- jumps ;
- changements brutaux.

Le papier ajoute donc un processus de Lévy.

Ils décomposent le bruit en :

```text
Brownian diffusion
+
jumps
```

avec :

\[
L_{i,t}
=
\sigma_i W_t
+
\sum_{j=1}^{N_{i,t}}\xi_{i,j}
\]

où :

- `σ_i` = diffusion ;
- `N_i,t` = processus de Poisson ;
- `ξ_i,j` = taille des jumps.

fileciteturn5file0L72-L114

---

# 4. Ce que cela change pour le trading

Avec un OU classique :

```text
écart extrême
→ probablement mean reversion
```

Mais avec des jumps :

```text
écart extrême
→ soit mean reversion normale
→ soit jump
→ soit changement de régime
```

Donc un modèle qui ignore les jumps peut confondre :

```text
JUMP
```

avec :

```text
CHANGEMENT DE RÉGIME
```

C'est précisément un des arguments empiriques du papier.

Dans leur simulation, le modèle Lévy-driven OU identifie mieux le nombre de régimes que le modèle OU classique lorsque des jumps existent. fileciteturn5file0L273-L328

---

# 5. Le concept le plus important du papier : nombre de régimes automatique

C'est probablement la partie la plus intéressante pour ton travail sur les régimes.

Contrairement à un HMM classique où tu fixes :

```python
n_regimes = 2
```

le papier commence par :

\[
r=1
\]

puis teste :

\[
r=2
\]

puis :

\[
r=3
\]

etc.

À chaque étape, il compare la qualité statistique du modèle avec le **BIC**.

Le nombre de régimes augmente tant que :

\[
BIC_r < BIC_{r-1}
\]

Sinon :

```text
r - 1 = nombre optimal de régimes
```

fileciteturn5file0L154-L210

---

# 6. BIC utilisé dans le papier

Le critère utilisé est :

\[
BIC_r
=
n\ln\left(\frac{CLS_r}{n}\right)
+
m\ln(n)
\]

où :

- `n` = nombre d'observations ;
- `m` = nombre de paramètres ;
- `CLS_r` = erreur conditionnelle des prévisions à un pas.

Le CLS est :

\[
CLS_r
=
\sum_{i=1}^{n-1}
\left(
X_{i+1}
-
E_p[X_{i+1}|X_i,\dots,X_1]
\right)^2
\]

Plus le BIC est faible, meilleur est le compromis fit / complexité.

fileciteturn5file0L154-L180

---

# 7. Comment les régimes sont initialement construits

C'est une subtilité très importante.

Le papier ne fait pas directement :

```text
HMM → régimes
```

Il utilise d'abord la **volatilité rolling du spread** :

\[
v_t = rolling\ volatility(X_t)
\]

Puis il cherche des seuils :

\[
c_1<c_2<...<c_{r-1}
\]

pour séparer les données.

Pour `r=2` :

```text
v < c1       → régime 1
v >= c1      → régime 2
```

Pour `r=3` :

```text
v < c1       → régime 1

c1 <= v < c2  → régime 2

v >= c2      → régime 3
```

Le choix des seuils est optimisé avec le BIC.

Le papier impose aussi qu'un régime représente au moins **15 % des observations**. fileciteturn5file0L181-L210

---

# 8. Smart grid search

Tester toutes les combinaisons de seuils serait coûteux.

Le papier utilise donc :

```text
start.grid
      ↓
évaluation BIC
      ↓
smart.grid
      ↓
nouveaux seuils
      ↓
évaluation
      ↓
...
```

Le principe :

> améliorer progressivement les seuils plutôt que faire une énorme recherche exhaustive.

L'algorithme est décrit explicitement dans leur Algorithm 1. fileciteturn5file0L211-L270

---

# 9. Important : ce n'est pas vraiment un HMM standard

Pour ton travail, il faut bien distinguer :

### HMM

```text
observations
→ probabilités d'états cachés
→ transition matrix
```

### Elliott & Bradrania

```text
spread
→ modèle regime-switching
→ état latent
```

### Endres & Stübinger

```text
rolling volatility
→ seuils
→ régimes
→ modèle Lévy-OU par régime
```

Leur régime est donc initialement **volatility-driven**, plutôt qu'un état latent totalement libre.

C'est une différence fondamentale.

---

# 10. Modèle dans chaque régime

Une fois les observations séparées :

```text
S1
S2
...
Sr
```

les auteurs estiment un modèle Lévy-driven OU dans chaque sous-échantillon.

Pour chaque régime `i` :

\[
\Theta_i=(\mu_i,\theta_i,\sigma_i)
\]

avec en plus les paramètres de jumps.

Le papier utilise les estimateurs de Mai (2012, 2014).

fileciteturn5file0L273-L328

---

# 11. Détection des jumps

C'est un autre morceau essentiel.

Le papier utilise un seuil :

\[
\nu_n=\Delta_n^\beta
\]

avec :

\[
\beta\in(0,1/2)
\]

Dans leur simulation :

\[
\beta=0.3
\]

Les incréments suffisamment grands sont considérés comme jumps et exclus de l'estimation de la partie continue.

fileciteturn5file0L115-L142

Pour leurs données minute :

\[
\Delta_n=
\frac{1}{250\times391}
\]

et ils choisissent la borne supérieure de `β`, conformément aux références citées. fileciteturn5file0L435-L444

---

# 12. Ce que tu dois retenir du jump filtering

Le raisonnement est :

```text
petit mouvement
→ diffusion normale

énorme mouvement
→ potentiellement jump
```

Cela évite d'estimer la vitesse de mean reversion à partir de mouvements extrêmes qui ne correspondent pas à la dynamique normale du spread.

Pour ton premier test daily, **ne reproduis pas immédiatement toute la théorie du jump filtering**.

Fais :

```text
Version 1 :
OU classique

Version 2 :
OU + jump filter

Version 3 :
Lévy / jump model complet
```

Tu pourras mesurer si les jumps apportent réellement quelque chose.

---

# 13. Architecture complète du papier

Le pipeline exact est approximativement :

```text
                PRICES
                   │
          ┌────────┴────────┐
          │                 │
        Stock A           Stock B
          │                 │
          └────────┬────────┘
                   ▼
                SPREAD
                   │
                   ▼
       Rolling spread volatility
                   │
                   ▼
      Regime Classification Algorithm
                   │
          ┌────────┴─────────┐
          ▼                  ▼
     number regimes      thresholds
          │                  │
          └────────┬─────────┘
                   ▼
       Lévy-driven OU per regime
                   │
                   ▼
       μ, θ, σ + jump characteristics
                   │
                   ▼
             PAIR RANKING
                   │
                   ▼
             TOP 10 PAIRS
                   │
                   ▼
          Bollinger bands
                   │
                   ▼
              TRADES
                   │
                   ▼
          transaction costs
                   │
                   ▼
               P&L
```

---

# 14. Sélection des paires

Le papier utilise une fenêtre de formation de :

\[
30\text{ jours}
\]

puis une fenêtre de trading de :

\[
5\text{ jours}
\]

Le tout est déplacé d'un jour :

```text
30d formation + 5d trading

→ shift 1 day

30d formation + 5d trading

→ shift 1 day
...
```

Ils obtiennent 4494 fenêtres sur 1998–2015. fileciteturn5file0L388-L401

---

# 15. Univers de données

Le papier utilise :

- constituants du S&P 500 ;
- minute data ;
- 1998–2015 ;
- 4529 jours ;
- 391 observations par jour ;
- 09:30–16:00 ;
- données ajustées splits / corporate events / dividendes.

Surtout :

> ils reconstruisent les constituants historiques de l'indice afin d'éviter le survivorship bias.

C'est extrêmement important.

fileciteturn5file0L404-L423

---

# 16. Pour ta réplication : ne commence pas avec ça

Tu n'as probablement pas besoin de :

```text
500 actions
×
391 bars/day
×
18 years
```

pour comprendre si le modèle fonctionne.

Commence par :

```text
5–20 paires
daily
10–20 ans
```

puis augmente.

---

# 17. Sélection basée sur les paramètres du modèle

Le papier classe les paires en utilisant :

- vitesse de mean reversion ;
- volatilité ;
- nombre de jumps ;
- taille des jumps.

Ils exigent également que les deux actions soient dans la même industrie.

Les paramètres sont pondérés par la taille du régime. fileciteturn5file0L435-L452

L'idée économique est :

```text
bonne paire =
mean reversion rapide
+
activité suffisante
+
volatilité exploitable
+
jumps correctement modélisés
```

---

# 18. Ranking à reproduire

Construire pour chaque paire :

```text
mean_reversion_speed
volatility
jump_frequency
jump_size
```

Puis classer les paires.

Le papier sélectionne les **10 meilleures**.

fileciteturn5file0L445-L452

---

# 19. Trading rule

C'est ici que le papier devient directement exploitable.

Ils utilisent des **Bollinger Bands** autour du niveau de mean reversion :

\[
\mu(t)\pm k\sigma(t)
\]

Si :

\[
X_t>\mu(t)+k\sigma(t)
\]

alors :

```text
SHORT A
LONG B
```

Si :

\[
X_t<\mu(t)-k\sigma(t)
\]

alors :

```text
LONG A
SHORT B
```

fileciteturn5file0L453-L474

---

# 20. Paramètres exacts du papier

Pour leur expérience :

```text
formation = 30 jours
trading = 5 jours
top pairs = 10
k = 0.5
running volatility = 1955 minutes
transaction costs = 20 bps
```

Les positions sont fermées au retour du spread de l'autre côté de la bande correspondant à l'ouverture, ou forcées à la fin des 5 jours. fileciteturn5file0L453-L481

---

# 21. Attention au `k = 0.5`

C'est extrêmement agressif.

À :

\[
k=0.5
\]

on entre dès que le spread est à seulement 0.5 écart-type du niveau d'équilibre.

Cela génère beaucoup de trades.

Les auteurs testent :

```text
k = 0.3
k = 0.5
k = 0.7
```

et trouvent que des `k` faibles donnent généralement davantage de profits mais davantage de coûts. fileciteturn6file0L268-L307

**Ne prends donc surtout pas `k=0.5` comme vérité universelle.**

---

# 22. Le résultat empirique principal

Sur leur période :

```text
1998–2015
top 10 pairs
```

après transaction costs :

| Stratégie | Return annuel | Sharpe |
|---|---:|---:|
| CCM | 2.15 % | -0.07 |
| BBM | 5.52 % | 0.47 |
| OUM | 47.24 % | 1.77 |
| **LDM** | **93.73 %** | **3.92** |
| S&P 500 | 3.93 % | 0.09 |

Le LDM est donc très nettement supérieur dans leur expérience. fileciteturn6file0L79-L103

---

# 23. Mais attention : le drawdown est loin d'être faible

Le LDM présente :

\[
MaxDD=29.70\%
\]

après coûts.

Donc :

```text
Sharpe 3.92
```

ne signifie absolument pas :

```text
low risk
```

Pour tes critères personnels, ce résultat n'est **pas suffisamment low-DD**.

Il faut donc tester si une variante du modèle peut obtenir :

```text
Sharpe > 1
MaxDD < 10–15 %
```

Ce serait beaucoup plus intéressant pour ton objectif.

---

# 24. Distribution des rendements

Après coûts, le LDM présente :

```text
skewness ≈ +1.22
kurtosis ≈ 25.67
VaR 1% ≈ -3.78%
CVaR 1% ≈ -5.19%
```

Donc le rendement est très non-gaussien.

Cela justifie précisément l'intérêt du modèle jump-based.

fileciteturn6file0L37-L58

---

# 25. Activité de trading

Le LDM effectue environ :

\[
6.60
\]

round-trip trades par paire sur chaque fenêtre de 5 jours.

Le temps moyen en position est environ :

\[
1.04\text{ jour}
\]

C'est donc une stratégie **très active**.

fileciteturn6file0L60-L78

---

# 26. Pourquoi le LDM gagne selon les auteurs

Le mécanisme est :

```text
régime
+
mean reversion speed
+
volatility
+
jumps
```

Le modèle sélectionne des paires ayant :

```text
mean reversion rapide
```

et :

```text
volatilité élevée
```

tout en traitant les jumps explicitement.

La vitesse de mean reversion élevée réduit la durée pendant laquelle une position est exposée. fileciteturn6file0L65-L71

---

# 27. Très important : le papier est surtout un modèle de sélection + trading

Il ne faut pas retenir uniquement :

```text
Lévy OU
```

Le vrai avantage proposé est :

```text
détection flexible des régimes
+
modélisation jump
+
sélection des paires
+
règle de trading
```

Donc si tu testes seulement :

```text
Lévy OU sur une paire
```

tu ne reproduis pas le cœur de leur contribution.

---

# 28. Comparaisons utilisées par le papier

Ils comparent leur LDM à :

### CCM

Sélection par corrélation.

Entrée :

\[
|spread|>2\sigma
\]

Sortie au retour vers l'équilibre.

### BBM

Corrélation + Bollinger Bands dynamiques.

### OUM

Même framework de régime mais OU classique, sans jumps.

### MKT

S&P 500 buy & hold.

fileciteturn5file0L495-L527

---

# 29. Le benchmark le plus important pour toi

À mon avis :

\[
\boxed{LDM\quad vs\quad OUM}
\]

C'est le test scientifique le plus intéressant.

Pourquoi ?

Parce que :

```text
OUM = regime switching + OU
LDM = regime switching + Lévy OU
```

Donc tu peux isoler :

> Est-ce que la modélisation explicite des jumps apporte quelque chose ?

Ensuite :

```text
OUM vs fixed OU
```

teste :

> Est-ce que les régimes apportent quelque chose ?

Et enfin :

```text
LDM vs SJM
```

teste :

> Est-ce que cette manière particulière de détecter les régimes est supérieure à une autre ?

---

# 30. Les régimes réellement observés

Sur toutes les paires :

### LDM

```text
1 régime : 53 %
2 régimes : 39 %
3 régimes : 7 %
4 régimes : 1 %
```

### OUM

```text
1 régime : 66 %
2 régimes : 31 %
3 régimes : 3 %
4 régimes : 0 %
```

Les auteurs concluent que le nombre de régimes est généralement faible, mais qu'imposer le même nombre à toutes les paires serait une mauvaise spécification. fileciteturn6file0L164-L201

**C'est une information très utile pour ton travail sur les modèles de régimes.**

---

# 31. Le résultat le plus intéressant pour ton research

Le papier suggère quelque chose d'important :

> le nombre optimal de régimes peut être **pair-dependent**.

Donc :

```text
Pair A → 1 régime
Pair B → 2 régimes
Pair C → 3 régimes
```

plutôt que :

```text
tout l'univers → 2 régimes
```

C'est une idée que je testerais absolument contre :

```text
HMM(2)
HMM(3)
SJM(2)
SJM(3)
```

---

# 32. Performance dans le temps

Le LDM reste positif dans leurs trois grandes sous-périodes :

```text
1998–2006 → 154.04 %
2007–2009 → 105.14 %
2010–2015 → 26.20 %
```

après coûts.

La performance baisse donc fortement après 2010, mais reste positive dans leur échantillon. fileciteturn6file0L104-L128

**Pour une réplication moderne, c'est justement un point à examiner : le signal survit-il après 2015 ?**

---

# 33. Exposition aux facteurs

Ils testent :

- Fama-French 3 factors ;
- Fama-French 3+2 ;
- Fama-French 5.

Leur alpha reste autour de :

\[
0.26-0.27\%
\]

par jour après coûts.

L'exposition marché est faible/non significative, ce qui est cohérent avec le caractère dollar-neutral. fileciteturn6file0L219-L267

Ils trouvent cependant une exposition positive significative au facteur reversal.

Donc :

> leur alpha n'est pas totalement mystérieux ; une partie du mécanisme ressemble à du short-term reversal.

---

# 34. Robustesse aux paramètres

Ils testent :

```text
top 5 / 10 / 20 pairs
k = 0.3 / 0.5 / 0.7
cost = 0 / 10 / 20 / 30 / 40 bps
```

Leurs résultats restent favorables sur une plage assez large.

Le point de break-even pour leur paramétrage standard est environ :

\[
c\approx40\text{ bps}
\]

fileciteturn6file0L268-L307

**C'est une excellente expérience à reproduire.**

---

# 35. Bootstrap

Ils réalisent également un bootstrap en conservant les signaux d'entrée/sortie mais en remplaçant les véritables titres par des titres aléatoires.

Ils répètent l'expérience :

\[
200
\]

fois.

Les paires randomisées donnent environ :

\[
-0.008\%
\]

par jour avant coûts, contre :

\[
0.27\%
\]

pour LDM après coûts.

fileciteturn6file0L308-L321

C'est un test particulièrement intéressant pour ton protocole.

---

# 36. Ce que tu dois absolument améliorer par rapport au papier

Même si le papier est solide pour son époque, ton protocole doit être plus strict.

Je recommande :

```text
walk-forward
+
OOS pur
+
transaction costs réalistes
+
slippage
+
short borrow
+
multiple univers
+
multiple periods
+
Deflated Sharpe
+
bootstrap
+
parameter perturbation
+
regime permutation
```

Et surtout :

> ne pas utiliser le même historique pour sélectionner les paires et évaluer la stratégie.

---

# 37. Attention au look-ahead

Le protocole du papier utilise des fenêtres :

```text
30d formation
5d trading
```

C'est bien adapté à une architecture OOS.

Pour ta réplication :

```text
TRAIN = 30 jours
TEST = 5 jours
```

puis :

```text
TRAIN ← shift 1 day
TEST  ← shift 1 day
```

Toutes les décisions du trading period doivent être basées uniquement sur le train et les données disponibles jusque-là.

---

# 38. Première réplication que je te recommande

Ne fais PAS directement le modèle complet.

Fais quatre versions.

## V0 — baseline

```text
log(A) - log(B)
→ z-score
→ mean reversion
```

## V1 — OU

```text
spread
→ OU
→ mean reversion
```

## V2 — regime switching OU

```text
spread
→ volatility regimes
→ OU par régime
→ Bollinger
```

## V3 — LDM

```text
spread
→ volatility regime classification
→ jump filter
→ Lévy OU
→ pair ranking
→ Bollinger
```

Puis :

## V4 — SJM

```text
spread
→ Statistical Jump Model
→ même couche de trading
```

---

# 39. Architecture expérimentale idéale

```text
                         SAME DATA
                             │
               ┌─────────────┴─────────────┐
               │                           │
             PAIRS                       PERIOD
               │
       ┌───────┴────────┐
       │                │
     TRAIN              TEST
       │                │
       ▼                ▼
┌──────────────────────────────┐
│      REGIME DETECTORS        │
├──────────────────────────────┤
│ Fixed OU                     │
│ HMM                          │
│ Elliott-Bradrania            │
│ Endres-Stübinger             │
│ SJM                          │
└──────────────────────────────┘
               │
               ▼
       SAME TRADING RULE
               │
               ▼
          SAME COSTS
               │
               ▼
          SAME SIZING
               │
               ▼
          OOS RESULTS
```

C'est cette architecture qui te permettra de dire :

> « le modèle de régime apporte réellement X ».

---

# 40. Test SJM particulièrement intéressant

Puisque tu travailles actuellement sur SJM, je ferais :

### Model A

```text
Endres-Stübinger
```

### Model B

```text
SJM
```

Mais surtout :

**même couche de trading.**

Par exemple :

```text
regime detector
      ↓
μ_regime
σ_regime
θ_regime
      ↓
Bollinger signal
```

Sinon tu compares à la fois :

- deux détecteurs ;
- deux modèles de spread ;
- deux règles de trading ;

et tu ne sais plus d'où vient la performance.

---

# 41. Expérience très importante : fixed number vs automatic number

Tester :

```text
HMM 2
HMM 3
HMM 4
```

contre :

```text
Endres-Stübinger automatic r
```

contre :

```text
SJM 2
SJM 3
```

Question :

\[
\boxed{
\text{Le choix automatique du nombre de régimes apporte-t-il réellement quelque chose ?}
}
\]

C'est une question centrale du papier.

---

# 42. Expérience jump vs no-jump

Tester :

```text
OU
vs
jump-filtered OU
vs
Lévy OU
```

avec :

```text
same regimes
same pair selection
same signal
same costs
```

Résultat attendu si l'hypothèse du papier est correcte :

```text
Lévy OU
>
jump-filtered OU
>
OU
```

mais c'est précisément ce que ton backtest doit vérifier.

---

# 43. Expérience regime vs no-regime

Faire :

```text
Fixed OU
        vs
Regime OU
```

Même règle de trading.

Si :

```text
Regime OU >> Fixed OU
```

alors les régimes apportent quelque chose.

Si :

```text
Regime OU ≈ Fixed OU
```

alors la sophistication est probablement inutile.

---

# 44. Expérience pair selection

Tester séparément :

```text
random pairs
correlation pairs
cointegration pairs
OU-ranked pairs
LDM-ranked pairs
```

C'est important car le Sharpe du papier peut venir en partie de la **sélection des paires**, et pas uniquement du modèle de régime.

---

# 45. Ce qu'il faut enregistrer

Pour chaque paire et chaque fenêtre :

```text
pair
train_start
train_end
test_start
test_end

n_regimes

regime_thresholds

theta_i
mu_i
sigma_i

jump_frequency_i
jump_size_i

BIC

ranking_score

trades
turnover

gross_return
net_return

Sharpe
Sortino
Calmar
MaxDD
PF
VaR
CVaR
```

Puis conserver les résultats au niveau :

```text
trade
pair
window
year
regime
portfolio
```

---

# 46. Diagnostics à produire

### A. Nombre de régimes

Histogramme :

```text
1 regime █████████
2 regimes ██████
3 regimes ██
4 regimes ▏
```

### B. Régime vs volatilité

Tracer :

```text
rolling vol
+
regime
```

### C. Régime vs spread

```text
spread
+
regime
```

### D. Mean reversion speed

```text
theta_1
theta_2
theta_3
```

### E. Jumps

```text
jump frequency
jump size
```

### F. P&L

```text
LDM
OUM
baseline
```

---

# 47. Critères de validation que je te recommande

Pour toi :

### Minimum

\[
Sharpe>1
\]

### Préférable

\[
Sharpe>1.5
\]

### Risque

\[
MaxDD<15\%
\]

idéalement :

\[
MaxDD<10\%
\]

### En plus

```text
OOS positive
PF > 1.2
Sortino > 1
Calmar solide
suffisamment de trades
robustesse aux coûts
robustesse aux seuils
plusieurs paires
plusieurs périodes
```

---

# 48. Le piège principal du papier

Le résultat :

\[
Sharpe=3.92
\]

est extrêmement élevé.

Il faut donc chercher agressivement les raisons possibles :

```text
high-frequency edge
+
very favorable historical period
+
pair selection
+
high turnover
+
20 bps assumed costs
+
specific universe
+
parameter selection
+
data availability
```

Le papier prend des mesures contre le data snooping, notamment en variant `p`, `k`, les coûts et via bootstrap, mais cela ne garantit évidemment pas que le même Sharpe existe sur les marchés post-2015. fileciteturn6file0L268-L321

---

# 49. Le test moderne le plus important

Prendre :

```text
2010–2015
```

puis tester :

```text
2016–2020
2021–2025
2026
```

sans modifier le modèle.

Encore mieux :

```text
rolling training
→ rolling OOS
```

et mesurer l'évolution du Sharpe.

Question :

\[
Sharpe_{1998-2015}
\stackrel{?}{>}
Sharpe_{2016-2026}
\]

Si le signal disparaît après 2015, il faut le savoir.

---

# 50. Variante daily recommandée

Comme tu travailles beaucoup avec des données accessibles gratuitement :

```text
daily adjusted close
```

Utiliser :

\[
X_t=\ln P_{A,t}-\ln P_{B,t}
\]

Puis :

```text
rolling volatility
→ regime classification
→ OU
→ Bollinger
```

Tu peux ensuite ajouter les jumps.

Ce ne sera **pas une réplication exacte du papier**, mais ce sera une excellente expérience de robustesse.

---

# 51. Variante intraday ensuite

Si la version daily fonctionne :

```text
5 min
15 min
30 min
1 h
```

Puis comparer :

```text
frequency
Sharpe
MaxDD
turnover
cost sensitivity
```

Le papier est spécifiquement conçu pour le high-frequency, donc une reproduction intraday est conceptuellement plus proche. fileciteturn5file0L404-L423

---

# 52. Plan de code

```text
project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data.ipynb
│   ├── 02_pairs.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_regime_classifier.ipynb
│   ├── 05_ou.ipynb
│   ├── 06_jump_filter.ipynb
│   ├── 07_ldm.ipynb
│   ├── 08_backtest.ipynb
│   └── 09_robustness.ipynb
│
├── src/
│   ├── data.py
│   ├── pairs.py
│   ├── spread.py
│   ├── regimes.py
│   ├── ou.py
│   ├── jumps.py
│   ├── ranking.py
│   ├── signals.py
│   ├── backtest.py
│   └── metrics.py
│
└── results/
```

---

# 53. Pseudo-code simplifié

```python
for window in walk_forward_windows:

    train = data.loc[window.train]
    test  = data.loc[window.test]

    # 1. Construct spreads
    spreads = build_spreads(train)

    # 2. For every pair
    for pair in spreads:

        x = pair.spread

        # 3. Estimate rolling volatility
        vol = rolling_volatility(x)

        # 4. Find optimal number of regimes
        r, thresholds = regime_classifier(
            x,
            vol,
            max_regimes=4
        )

        # 5. Classify observations
        regimes = classify(x, vol, thresholds)

        # 6. Estimate OU/Lévy-OU per regime
        params = fit_regime_models(
            x,
            regimes
        )

        # 7. Calculate pair ranking
        score = rank_pair(params)

    # 8. Select top pairs
    selected = select_top_pairs(scores, n=10)

    # 9. Trade OOS
    for pair in selected:

        signal = bollinger_signal(
            test[pair],
            params[pair]
        )

        pnl = backtest(
            signal,
            costs=True
        )

    save_results(...)
```

---

# 54. Version simplifiée que je te conseille réellement de coder

Ne commence pas par le Lévy process complet.

### V1

```text
log spread
↓
rolling volatility
↓
BIC regime classifier
↓
OU par régime
↓
Bollinger
↓
backtest
```

### V2

Ajouter :

```text
jump filter
```

### V3

Ajouter :

```text
Lévy OU
```

### V4

Ajouter :

```text
automatic pair ranking
```

### V5

Comparer :

```text
HMM
SJM
Endres-Stübinger
```

---

# 55. Ce que je chercherais personnellement dans les résultats

Pas :

> « Est-ce que je peux obtenir 93 % comme le papier ? »

Mais :

### Question 1

Le régime switching améliore-t-il vraiment le pairs trading ?

### Question 2

Le nombre de régimes optimal varie-t-il selon la paire ?

### Question 3

La volatilité est-elle suffisante pour détecter les régimes ?

### Question 4

Les jumps améliorent-ils la détection ?

### Question 5

Les jumps améliorent-ils le P&L ?

### Question 6

La sélection par mean-reversion speed fonctionne-t-elle OOS ?

### Question 7

L'edge survit-il à des coûts réalistes ?

### Question 8

L'edge survit-il après 2015 ?

### Question 9

SJM fait-il mieux ?

### Question 10

HMM fait-il mieux ?

---

# 56. Résultat scientifique idéal

Tu pourrais finalement obtenir quelque chose comme :

```text
                         OOS Sharpe    MaxDD
Baseline                  0.71        -18%
OU                        0.92        -16%
Regime OU                 1.18        -12%
Regime Lévy-OU            1.31         -9%
SJM + OU                  1.47         -8%
```

Même si les chiffres sont beaucoup plus faibles que ceux du papier, ce serait **un résultat extrêmement intéressant** si la hiérarchie est stable.

---

# 57. Conclusion du papier à retenir

Le papier montre trois idées essentielles :

### 1.

Les spreads peuvent présenter plusieurs régimes de volatilité.

### 2.

Les jumps et queues épaisses peuvent être importants pour identifier correctement ces régimes.

### 3.

Une stratégie de pairs trading qui exploite :

```text
regime
+
mean reversion
+
volatility
+
jumps
```

peut obtenir des résultats très élevés dans leur historique.

Les auteurs rapportent une performance très supérieure aux benchmarks et une robustesse à plusieurs paramètres et coûts. fileciteturn6file0L322-L354

---

# 58. Mais ton objectif doit être différent

Le papier cherche :

> une stratégie HF très performante.

Toi, tu cherches plutôt :

> **un détecteur de régime robuste que tu peux brancher à différentes stratégies.**

Donc utilise ce papier comme **brique de recherche**, pas comme stratégie à copier aveuglément.

La structure la plus intéressante pour ton programme est :

```text
                    REGIME DETECTOR
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
      HMM                 SJM          Endres-Stübinger
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                           ▼
                  REGIME CHARACTERISTICS
                           │
              ┌────────────┼────────────┐
              │            │            │
          mean-rev       vol          jumps
              │            │            │
              └────────────┼────────────┘
                           ▼
                    TRADING STRATEGY
```

C'est, à mon avis, la partie la plus exploitable du papier pour ton projet de recherche.

---

# 59. Checklist de réplication

```text
[ ] Construire log-spread
[ ] Vérifier stationnarité / mean reversion
[ ] Rolling volatility
[ ] Implémenter BIC
[ ] Implémenter r = 1
[ ] Implémenter r = 2
[ ] Implémenter r = 3
[ ] Implémenter r = 4
[ ] Implémenter contrainte 15 %
[ ] Implémenter smart grid
[ ] OU par régime
[ ] Jump filter
[ ] Lévy OU
[ ] Pair ranking
[ ] Top 5 / 10 / 20
[ ] Bollinger k = 0.3 / 0.5 / 0.7
[ ] 20 bps benchmark
[ ] Cost sensitivity
[ ] Walk-forward
[ ] OOS
[ ] Bootstrap
[ ] Random pairs
[ ] Analyse facteurs
[ ] Analyse par période
[ ] Post-2015 test
[ ] Comparaison fixed OU
[ ] Comparaison HMM
[ ] Comparaison SJM
[ ] Deflated Sharpe
[ ] Monte Carlo
[ ] Analyse MaxDD
[ ] Analyse turnover
```

---

# 60. Référence

Endres, S., & Stübinger, J. (2018). **A flexible regime switching model with pairs trading application to the S&P 500 high-frequency stock returns.** FAU Discussion Papers in Economics, No. 07/2018.

Le papier source fourni est utilisé comme référence principale pour le modèle, l'algorithme de classification des régimes, l'étude Monte Carlo, le protocole de formation/trading, les règles de Bollinger, les benchmarks, les résultats et les tests de robustesse.
