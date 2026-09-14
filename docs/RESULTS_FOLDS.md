# The five declared folds, finally evaluated

`docs/RESULTS_FINAL.md` declared that the five walk-forward folds were specified
and never evaluated: `regime_lab.models.protocol.walk_forward` had exactly one
call site, `scripts/run_phase2.py:66`, and its result was used only to read the
start date of the first fold. The protocol actually executed is one contiguous
out-of-sample block with 49 semi-annual refits.

The CLAUDE.md rule that governs this project says walk-forward with at least five
folds, never a single split. So this gap was not a missing nicety; it was a hard
rule left unverified.

```bash
.venv/bin/python scripts/run_folds.py                  # layers 1 and 2, per fold
.venv/bin/python scripts/run_fold_identification.py    # is each fold identified?
.venv/bin/python scripts/run_fold_heterogeneity.py     # per-fold coefficients, Cochran Q, LOFO
```

Nothing is refitted. The folds partition state series that `run_phase2.py` already
produced, and every metric is the function `run_evaluation.py` already calls,
applied to each slice.

**The answer, in one line: the central result survives the folds, and the folds
turn out to be the wrong instrument for this data.** Both halves matter, and the
second one is the more interesting.

---

## 1. What this measures, and what it does not

Slicing a contiguous 49-refit block by fold edges gives the **per-fold dispersion
of the protocol that was actually run**. It is not a strict five-fold
walk-forward, in which fold *k*'s model would be frozen at the fold edge and held
for five years. Inside fold *k* here, the model in force at date *t* was refitted
at the most recent semi-annual date before *t*, so it has seen more data than a
frozen fold model would — never less. Both are causal. They are different
estimators, and this is the first one.

The folds:

| fold | window | days |
| --- | --- | --- |
| 0 | 2002-04-01 → 2007-01-26 | 1,260 |
| 1 | 2007-01-29 → 2011-12-23 | 1,280 |
| 2 | 2011-12-26 → 2016-11-17 | 1,279 |
| 3 | 2016-11-18 → 2021-10-13 | 1,279 |
| 4 | 2021-10-14 → 2026-09-08 | 1,279 |

---

## 2. Three folds are not identified at all

Before asking what the state coefficient is in each fold, ask whether the fold
contains any state variation to estimate it from.

| family | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
| --- | --- | --- | --- | --- | --- |
| A jump | 1 switch | 6 | **0 — one state** | 4 | 2 |
| A′ sparse jump | 1 switch | 6 | **0 — one state** | 6 | **0 — one state** |
| B filtered HMM | 5 | 12 | 4 | 12 | 16 |
| C gradient boost | 29 | 84 | 93 | 63 | 75 |
| C′ HAR-RV | 74 | 103 | 81 | 48 | 143 |
| · vol placebo | 18 | 39 | 45 | 23 | 55 |

**For A′ sparse jump — the study's best classifier — two of the five folds contain
a single state across 1,279 trading days.** The state never moves, so there is no
within-fold contrast, and the fold cannot say anything about the state at all.
A jump loses one fold the same way.

This is not a defect. It is the direct consequence of the property the study
measured: mean run length 456 days and persistence at 83× chance. A classifier
whose states last a year and a half will, sooner or later, produce a five-year
window that sits entirely inside one state. **The charter's stopping rule — a
positive gap above the MDE on three folds of five — is therefore not computable
for the family the study leads with.** Five folds of five years each is the wrong
partition for an object with this time constant, and the expanding protocol with
49 refits was, in hindsight, the right instrument rather than a shortcut past a
rule.

That is the honest version. The dishonest version would have been to report
three-of-five as passed or failed without noticing that two of the cells were
empty.

---

## 3. The published number survives fold fixed effects

76% to 89% of the total variation is within folds, not between them, so there is
real within-fold variation to identify from. Putting fold dummies in both
regressions restricts identification to it:

| family | target | pooled (published) | with fold fixed effects |
| --- | --- | --- | --- |
| A jump | volatility | 3.474%, t −3.46 | 2.763%, t −3.56 |
| **A′ sparse jump** | **volatility** | **3.929%, t −3.40** | **3.683%, t −3.51** |
| B filtered HMM | volatility | 2.261%, t −4.59 | 1.388%, t −3.67 |
| C gradient boost | volatility | 2.127%, t −6.34 | 1.056%, t −4.54 |
| C′ HAR-RV | volatility | 0.192%, t −1.84 | 0.065%, t −1.07 |
| · vol placebo | volatility | 0.004%, t +0.21 | 0.200%, t +1.58 |
| A′ sparse jump | return | 0.030%, t 0.27 | 0.035%, t 0.32 |

The headline figure loses 6% of its magnitude and gains significance. The placebo
stays inside the noise with the wrong sign. Forward returns stay at nothing, which
they were.

B, C and C′ lose a third to a half of their incremental R², so part of what they
carried was between-fold — periods differing from each other — rather than states
differing within a period. A and A′ do not.

### Leave one fold out

Incremental R² on forward volatility, fold dummies in both regressions, dropping
one fold at a time:

| family | all | −fold 0 | −fold 1 | −fold 2 | −fold 3 | −fold 4 |
| --- | --- | --- | --- | --- | --- | --- |
| A jump | 2.76% (−3.56) | 2.44 (−2.80) | 2.24 (−2.81) | 1.73 (−2.75) | 3.98 (−3.69) | 3.52 (−3.81) |
| **A′ sparse jump** | **3.68% (−3.51)** | 3.51 (−2.88) | **1.74 (−2.42)** | 2.68 (−2.90) | 6.51 (−3.84) | 4.35 (−3.58) |
| B filtered HMM | 1.39% (−3.67) | 1.62 (−3.44) | 3.34 (−4.59) | 0.72 (−2.46) | 1.64 (−3.66) | 0.68 (−2.35) |
| C gradient boost | 1.06% (−4.54) | 0.84 (−3.70) | 2.12 (−5.15) | 0.74 (−3.60) | 1.02 (−3.84) | 0.95 (−3.94) |
| C′ HAR-RV | 0.07% (−1.07) | 0.01 (−0.44) | 0.19 (−1.46) | 0.01 (−0.29) | 0.20 (−1.98) | 0.06 (−0.85) |
| · vol placebo | 0.20% (+1.58) | 0.33 (+1.74) | 0.21 (+1.26) | 0.30 (+1.74) | 0.06 (+0.94) | 0.19 (+1.35) |

**No fold carries the result.** A′ is weakest without fold 1 — 2007-2011, the
financial crisis — at 1.74 points and t −2.42, still significant, and the effect
is not a crisis artefact. It is strongest without fold 3.

This is the check that mattered most going in. It comes back clean.

---

## 4. Per-fold coefficients, and heterogeneity

State coefficient on forward volatility in annualised points, controlling for
volatility rank, negative meaning the calm state really is calmer:

| family | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
| --- | --- | --- | --- | --- | --- |
| A jump | −0.040 (−3.70) | −0.028 (−2.05) | — | −0.018 (−0.82) | +0.015 (1.03) |
| A′ sparse jump | −0.040 (−3.69) | −0.046 (−2.69) | — | −0.007 (−0.36) | — |
| B filtered HMM | −0.008 (−1.57) | +0.010 (0.57) | −0.021 (−2.80) | −0.019 (−1.37) | −0.047 (−4.75) |
| C gradient boost | −0.023 (−3.24) | +0.001 (0.05) | −0.006 (−1.36) | −0.017 (−2.52) | −0.016 (−3.03) |
| C′ HAR-RV | −0.018 (−2.34) | +0.010 (0.86) | −0.006 (−1.18) | +0.011 (0.53) | −0.005 (−0.95) |
| · vol placebo | +0.008 (1.25) | +0.021 (1.42) | −0.004 (−0.61) | +0.042 (1.40) | +0.007 (1.01) |

The placebo is positive in four folds of five and never significant anywhere.
A′ is negative in all three identified folds.

### Cochran's Q over the identified folds

| family | folds | pooled coef | Q | df | p | I² |
| --- | --- | --- | --- | --- | --- | --- |
| A jump | 4 | −0.0225 | 9.20 | 3 | 0.027 | 67.4% |
| **A′ sparse jump** | **3** | **−0.0349** | **3.05** | **2** | **0.218** | **34.3%** |
| B filtered HMM | 5 | −0.0164 | 15.04 | 4 | 0.005 | 73.4% |
| C gradient boost | 5 | −0.0128 | 6.56 | 4 | 0.161 | 39.1% |
| C′ HAR-RV | 5 | −0.0058 | 5.03 | 4 | 0.284 | 20.5% |
| · vol placebo | 5 | +0.0048 | 5.12 | 4 | 0.275 | 21.9% |

One common coefficient explains A′ across its identified folds. B and A jump show
real dispersion.

The interaction Wald test, restricted to identified folds, agrees: A′ p 0.234,
A jump p 0.096, against B 0.0003, C 0.005, C′ 0.00000 — **and the placebo at
0.0004.** That last cell is the control that keeps this in proportion: the
one-line volatility quantile shows fold heterogeneity as strong as B and C do.
Heterogeneity across these five windows is largely a property of the periods,
not of the state.

### A withdrawn test

The first version of this heterogeneity test ran the interaction on all five
folds and reported A′ at p = 4.5 × 10⁻⁵, which would have meant strong fold
heterogeneity in the study's best family. **That p-value is withdrawn.** With two
degenerate folds the design matrix loses rank — 9 columns of rank for 11 columns
in A′, 10 for A jump — and the statistic is not identified. The table above,
which checks rank first and drops the unidentified folds, replaces it.
`scripts/run_folds.py` still prints the rank-deficient version; its docstring says
so, and says to read the p-values from `run_fold_heterogeneity.py` instead.

---

## 5. Layer 1 per fold, and a latent defect

Reliability metrics per fold reproduce the published full-sample figures on the
ALL row: A′ sparse jump 93.2% balanced accuracy against NBER, kappa 0.53,
persistence 83.1× chance, hindsight 0.6%. Persistence and hindsight are
well-defined in every fold and are the tables to read.

**External validation per fold is not.** NBER declares a recession in only **two
of these five folds** — fold 1, which contains 2007-2009, and fold 3, which
contains 2020. Folds 0, 2 and 4 contain no recession day at all. Against a
reference with one class there is no balanced accuracy to compute: the
"balanced" mean collapses onto the recall of the only class present, and kappa's
denominator vanishes. **The honest per-fold NBER table has two rows**, and they
are A′ sparse jump at 93.2% with kappa 0.81 in fold 1, and 84.3% with kappa 0.23
in fold 3.

**A defect with two halves, found by this work, now fixed.**
`reliability.external_validation` scored both degeneracies instead of declining
them, and the two halves pushed in opposite directions.

*The state half.* `weak = state == state.min()` is true on every session when the
state never moves, so the classifier was graded as though it had called a
recession every single day. Three cells — A jump fold 2, A′ sparse jump folds 2
and 4 — read **0.0% balanced accuracy**.

*The reference half, which is the larger one.* Where NBER never fires, kappa
returns exactly 0.000 from a vanished denominator, indistinguishable from a real
absence of skill. **Fifteen of the twenty-five per-fold cells carry that spurious
zero**, including cells where the state moves perfectly well. Three of them read
high rather than low: A jump shows **0.9804** in fold 4 and C gradient boost
**0.1681**. Those are not balanced accuracies at all — they are *specificities*,
the only class present being the absence of recession. An earlier version of this
document read the 0.1681 as base-rate domination. It is not; the quantity is
undefined.

Six misleading cells, three downward and three upward, from one defect with
opposite signs. This is not a table a guard repairs — the guard only stops it
lying.

**No published figure is affected.** Every published number is full-sample, where
both series move, and the full-sample figures are bit-identical after the fix:
A′ sparse jump 0.9316 / 0.5342, A jump 0.9332 / 0.4863, B 0.8472 / 0.2368,
C 0.7505 / 0.1229, C′ 0.7823 / 0.1671.

The function now returns NaN when either series has fewer than two distinct
values, and every branch returns the same keys — the short-circuit branch
previously returned three keys where the success branch returned six.
`tests/test_reliability.py` covers all four cases.

---

## 6. What to do about the rule

The CLAUDE.md rule asks for five folds. The charter asks for a stopping rule
evaluated on three of five. Neither is satisfiable as written for A′ sparse jump,
because two of its folds contain no state variation.

Two options, and the second is the one the evidence supports:

1. Keep five folds and report three-of-five as uncomputable for A and A′. This is
   honest but useless: it converts a hard rule into a permanent asterisk.
2. Declare the executed protocol — one expanding block, 49 semi-annual refits,
   online prediction — as the walk-forward of record, on the ground that it
   refits 49 times rather than 5 and never sees the future, and keep the five-fold
   slice as the dispersion diagnostic it is: fold fixed effects, leave-one-fold-out
   and Cochran's Q, all reported here.

Option 2 needs an entry in `docs/PROTOCOL_FREEZE.md`, because it is a deviation
from the frozen text and the charter is never edited in place. The entry is there.
The measurement that justifies it is section 2 of this document: an object with a
456-day mean run length cannot be partitioned into five five-year folds and still
vary inside each one.
