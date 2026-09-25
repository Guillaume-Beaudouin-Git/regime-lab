# Guide de Révision et Banques de Tests : Attention Is All You Need & BERT

**Objectif :** Préparation aux examens, auto-évaluation et tests de connaissances sur les articles *Attention Is All You Need* (Vaswani et al., 2017) et *BERT* (Devlin et al., 2018).  
**Format :** Markdown (.md)  

---

## Table des Matières
1. [Fiche de Synthèse Rapide (Aide-Mémoire)](#1-fiche-de-synthèse-rapide-aide-mémoire)
2. [Article 1 : *Attention Is All You Need* (Vaswani et al., 2017)](#2-article-1--attention-is-all-you-need-vaswani-et-al-2017)
3. [Article 2 : *BERT* (Devlin et al., 2018)](#3-article-2--bert-devlin-et-al-2018)
4. [Analyse Comparative et Tableau Récapitulatif](#4-analyse-comparative-et-tableau-récapitulatif)
5. [Banque de Tests - Partie 1 : QCM (Questions à Choix Multiples)](#5-banque-de-tests---partie-1--qcm-questions-à-choix-multiples)
6. [Banque de Tests - Partie 2 : Questions de Cours & Exercices Calculatoires](#6-banque-de-tests---partie-2--questions-de-cours--exercices-calculatoires)
7. [Banque de Tests - Partie 3 : Flashcards Conceptuelles & Pièges d'Examen](#7-banque-de-tests---partie-3--flashcards-conceptuelles--pièges-dexamen)
8. [Corrigés Détaillés des Tests](#8-corrigés-détaillés-des-tests)

---

## 1. Fiche de Synthèse Rapide (Aide-Mémoire)

* **Vaswani et al. (2017) - Transformer :**
  * Architecture : **Encodeur - Décodeur** pour la traduction automatique (Seq2Seq).
  * Innovation clé : Suppression de la récurrence (RNN) et des convolutions (CNN). Utilisation exclusive du **Self-Attention**.
  * Formule Attention : $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$.
  * Multi-Head : $h = 8$ têtes, $d_{\text{model}} = 512$, $d_k = d_v = 64$.
  * Encodage Positionnel : Fonctions sinusoïdales et cosinusoïdales fixes.

* **Devlin et al. (2018) - BERT :**
  * Architecture : **Encodeur Transformer uniquement** (BERT-Base: 12 couches, BERT-Large: 24 couches).
  * Innovation clé : Représentation **profondément bidirectionnelle** via le pré-entraînement auto-supervisé.
  * Tâches de pré-entraînement :
    1. **MLM (Masked Language Model) :** Masquage de 15% des tokens (80% `[MASK]`, 10% aléatoire, 10% inchangé).
    2. **NSP (Next Sentence Prediction) :** Prédiction binaire (`IsNext` vs `NotNext`).
  * Représentations : Token Embeddings + Segment Embeddings + Position Embeddings.

---

## 2. Article 1 : *Attention Is All You Need* (Vaswani et al., 2017)

### Concept & Motivations
Les RNNs calculent séquentiellement ($\mathcal{O}(n)$ étapes temporelles), ce qui empêche la parallélisation sur GPU. Le Transformer permet un calcul parallèle simultané sur toute la séquence.

### Le Mécanisme d'Attention
* **$Q$ (Query), $K$ (Key), $V$ (Value) :** Obtenus par projection linéaire des embeddings d'entrée : $Q = X W^Q$, $K = X W^K$, $V = X W^V$.
* **Facteur d'échelle $\sqrt{d_k}$ :** Évite que le produit scalaire $QK^T$ prenne de trop grandes valeurs pour de grandes dimensions, ce qui saturerait la fonction Softmax (gradients quasi-nuls).
* **Attention Masquée (Décodeur) :** Empêche le décodeur de regarder les tokens futurs lors de la génération autorégressive.

---

## 3. Article 2 : *BERT* (Devlin et al., 2018)

### Concept & Motivations
Les modèles antérieurs (comme GPT-1) étaient unidirectionnels (gauche-droite). BERT démontre qu'un contexte bi-directionnel profond à toutes les couches améliore drastiquement la compréhension de texte.

### Structure d'Entrée
* `[CLS]` : Premier token de chaque séquence. Sa représentation finale est utilisée pour les tâches de classification globale.
* `[SEP]` : Token de séparation entre deux phrases.

---

## 4. Analyse Comparative et Tableau Récapitulatif

| Critère d'Évaluation | Transformer (Vaswani 2017) | BERT (Devlin 2018) |
| :--- | :--- | :--- |
| **Composant employé** | Encodeur + Décodeur | Encodeur uniquement |
| **Objectif premier** | Traduction / Génération (Seq2Seq) | Représentation & Compréhension |
| **Directionnalité** | Bi (Encodeur) / Auto-régressif (Décodeur) | Profondément Bi-directionnel |
| **Pré-entraînement** | Supervisé (Paires de phrases) | Auto-supervisé sur texte brut |
| **Génération de texte** | Oui (Native) | Non (Inadapté) |

---

## 5. Banque de Tests - Partie 1 : QCM (Questions à Choix Multiples)

#### Q1. Pourquoi Vaswani et al. divisent-ils le produit scalaire $QK^T$ par $\sqrt{d_k}$ dans la formule d'attention ?
- A) Pour réduire le nombre de paramètres du modèle.
- B) Pour éviter que le gradient ne disparaisse à cause de la saturation du softmax.
- C) Pour rendre le modèle autorégressif.
- D) Pour convertir les vecteurs en probabilités.

#### Q2. Dans BERT, parmi les 15% de tokens sélectionnés pour le Masked Language Model (MLM), quelle est la proportion exacte remplacée par le token `[MASK]` ?
- A) 100%
- B) 50%
- C) 80%
- D) 15%

#### Q3. Quelle est la complexité temporelle par couche d'une couche de Self-Attention pour une séquence de longueur $n$ et une dimension $d$ ?
- A) $\mathcal{O}(n \cdot d^2)$
- B) $\mathcal{O}(n^2 \cdot d)$
- C) $\mathcal{O}(n \cdot d)$
- D) $\mathcal{O}(n^3)$

#### Q4. Quel est le rôle du token `[CLS]` dans BERT ?
- A) Séparer deux phrases distinctes.
- B) Masquer les mots inconnus.
- C) Servir de représentation globale de la séquence pour les tâches de classification.
- D) Indiquer la fin d'un paragraphe.

#### Q5. Quelle est la différence majeure entre OpenAI GPT (v1) et BERT (2018) ?
- A) GPT utilise des convolutions, BERT utilise le Transformer.
- B) GPT est un modèle unidirectionnel (gauche-droite), alors que BERT est profondement bidirectionnel.
- C) BERT contient plus de 100 milliards de paramètres, contrairement à GPT.
- D) GPT ne peut pas faire de fine-tuning.

---

## 6. Banque de Tests - Partie 2 : Questions de Cours & Exercices Calculatoires

### Exercice 1 : Calcul des dimensions des matrices d'attention
Soit une séquence d'entrée composée de $n = 50$ tokens. La dimension cachée du modèle est $d_{\text{model}} = 512$. On utilise $h = 8$ têtes d'attention.
1. Quelle est la dimension $d_k$ de chaque tête d'attention ?
2. Quelles sont les dimensions des matrices de projections $W_i^Q, W_i^K, W_i^V$ pour une tête $i$ ?
3. Quelle est la dimension de la matrice d'attention issue de $QK^T$ avant multiplication par $V$ ?

### Question de Cours 1 : L'encodage positionnel
Pourquoi le Transformer original a-t-il besoin d'un encodage positionnel (*Positional Encoding*), alors que les RNN n'en ont pas besoin ? Pourquoi les auteurs ont-ils choisi des fonctions sinusoïdales ?

### Question de Cours 2 : L'objectif Next Sentence Prediction (NSP)
Expliquez le fonctionnement de la tâche NSP lors du pré-entraînement de BERT. Pourquoi a-t-elle été introduite et comment les données d'entraînement sont-elles construites pour cette tâche ?

---

## 7. Banque de Tests - Partie 3 : Flashcards Conceptuelles & Pièges d'Examen

### Carte 1 : Cross-Attention vs Self-Attention
* **Question :** Dans l'architecture du Transformer (Vaswani 2017), quelle est la différence entre le bloc *Self-Attention* et le bloc *Cross-Attention* (ou Encoder-Decoder Attention) ?
* **Piège classique d'examen :** Confondre les matrices $Q, K, V$ envoyées au décodeur. Dans le Cross-Attention, d'où viennent $Q$, $K$ et $V$ ?

### Carte 2 : Le décalage Pré-entraînement / Fine-tuning de BERT
* **Question :** Pourquoi les auteurs de BERT n'ont-ils pas remplacé 100% des 15% de tokens sélectionnés par le token `[MASK]` ? Quel problème cela aurait-il posé ?

### Carte 3 : Parallélisme d'entraînement
* **Question :** Pourquoi l'entraînement de l'encodeur Transformer est-il entièrement parallélisable, alors que le décodeur ne l'est pas complètement lors de l'inférence ?

---

## 8. Corrigés Détaillés des Tests

### Réponses QCM
* **Q1 : Réponse B.** Lorsque $d_k$ est grand, le produit scalaire croît en magnitude, poussant la fonction Softmax dans des zones où les dérivées sont extrêmement faibles (problème du gradient vanishing). Diviser par $\sqrt{d_k}$ maintient la variance à 1.
* **Q2 : Réponse C.** 80% des tokens sélectionnés sont remplacés par `[MASK]`, 10% par un token aléatoire, et 10% restent inchangés.
* **Q3 : Réponse B.** La matrice d'attention $QK^T$ a une taille $(n 	imes n)$, d'où une complexité spatiale et temporelle en $\mathcal{O}(n^2 \cdot d)$.
* **Q4 : Réponse C.** L'état caché final correspondant à `[CLS]` agrège l'information de toute la séquence via le mécanisme d'auto-attention.
* **Q5 : Réponse B.** GPT utilise un décodeur causale (masquage à droite), empêchant chaque token de voir le contexte futur. BERT utilise un encodeur bidirectionnel.

### Correction Exercice 1 (Dimensions)
1. $d_k = d_{\text{model}} / h = 512 / 8 = 64$.
2. Chaque matrice de projection $W_i^Q, W_i^K, W_i^V$ a pour dimension $(d_{\text{model}} 	imes d_k) = (512 	imes 64)$.
3. Pour une séquence de $n = 50$ tokens, la matrice $Q$ a une dimension $(50 	imes 64)$ et $K^T$ a une dimension $(64 	imes 50)$. Le produit $Q K^T$ donne une matrice de dimension **$(50 	imes 50)$**, représentant les scores d'attention entre chaque paire de tokens de la séquence.

### Correction Question de Cours 1 (Encodage Positionnel)
* Les RNN traitent les mots séquentiellement ($t=1, t=2, \dots$), donc l'ordre est implicite. Le Transformer traite tous les tokens simultanément et est **invariant par permutation**.
* Sans encodage positionnel, la phrase "Le chat mange la souris" et "La souris mange le chat" produiraient exactement les mêmes représentations vectorielles.
* Les fonctions sinusoïdales ont été choisies car elles permettent au modèle d'extrapoler à des longueurs de séquence supérieures à celles vues pendant l'entraînement et permettent de calculer facilement les positions relatives ($PE_{pos+k}$ est une combinaison linéaire de $PE_{pos}$).

### Correction Question de Cours 2 (NSP)
* **Objectif :** Entraîner le modèle à comprendre les relations entre phrases (nécessaire pour QA, NLI).
* **Construction :** Pour chaque exemple, on choisit deux phrases A et B.
  * 50% du temps, B est la phrase séquentielle suivante dans le texte brut (label `IsNext`).
  * 50% du temps, B est une phrase aléatoire tirée du corpus (label `NotNext`).
* **Prédiction :** La représentation du token `[CLS]` passe par une couche linéaire pour classifier `IsNext` ou `NotNext`.

### Correction Flashcard 1 (Cross-Attention)
* Dans le **Self-Attention** (Encodeur ou Décodeur), $Q, K, V$ proviennent tous de la même couche précédente.
* Dans le **Cross-Attention** (Décodeur) :
  * Les Requêtes $Q$ proviennent du décodeur (couche précédente du décodeur).
  * Les Clés $K$ et Valeurs $V$ proviennent de la **sortie finale de l'encodeur**.

### Correction Flashcard 2 (Décalage MASK)
* Le token `[MASK]` n'apparaît **jamais** lors de la phase de Fine-Tuning ou d'inférence réelle.
* Si 100% des mots cibles étaient remplacés par `[MASK]` pendant le pré-entraînement, le modèle créerait une dépendance excessive à ce token artificiel, créant un décalage entre pré-entraînement et utilisation réelle. D'où le compromis 80/10/10.

---
*Document créé à des fins de révision, de préparation d'examens et d'auto-évaluation.*
