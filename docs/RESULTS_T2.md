# T2 — the decisive test, and the stopping rule it fires

Out of sample, April 2002 to September 2026, 6,377 sessions. Every figure is on
returns in excess of the three-month bill. No transaction costs yet: turnover is
reported but not charged.

## No family beats a moving volatility target

The benchmark is the same 60/40 book sized to an ex-ante volatility target, with
the target set at each refit to the volatility the overlay realised on that
refit's training window. It uses no regimes at all.

| family | Sharpe overlay | Sharpe benchmark | ΔSharpe | CEQ overlay | CEQ benchmark |
| --- | --- | --- | --- | --- | --- |
| A jump model | 0.18 | 0.56 | −0.39 | **−0.06%** | 2.86% |
| A′ sparse jump | 0.12 | 0.56 | −0.44 | **−0.36%** | 2.78% |
| B filtered HMM | 0.47 | 0.57 | −0.10 | 1.87% | 2.42% |
| C gradient boosting | 0.48 | 0.55 | −0.07 | 1.67% | 2.08% |
| C′ HAR-RV | 0.59 | 0.56 | **+0.04** | 2.37% | 2.21% |

Every volatility gap is under 10%, so the comparisons are like-for-like and the
test is admissible rather than merely reported.

**The certainty equivalent says something the Sharpe ratio hides.** Both jump
models have a *negative* certainty equivalent under constant relative risk
aversion: an investor with γ = 5 would rather hold nothing than hold them, while
the same investor values the benchmark at nearly three percent a year. A
scale-invariant metric could not have shown that, which is exactly why the
charter required metrics that respond to leverage.

## The three-parameter regression from 2003 wins

HAR-RV — daily, weekly and monthly realised volatility, ordinary least squares —
is the only specification that edges past the benchmark, and it beats the
gradient boosting model trained on all fifty features. This is the standard
result in the realised-volatility literature and it survived being tested here
rather than being cited.

Its margin is +0.04 of Sharpe, and it has the highest turnover in the table
(0.071 a session against 0.008 for the HMM). Charging any realistic cost removes
it. The finding is that machine learning did not beat the linear baseline, not
that the linear baseline works.

## Stop rule 3 fires on every family

The charter stops the study if the minimum detectable effect exceeds 0.30 of
Sharpe. Phase 0 measured 0.174 and flagged it explicitly as a lower bound,
because it compared two versions of one book correlated at 0.91. Against a real
overlay:

| family | correlation with benchmark | MDE | stop rule | years needed for an MDE of 0.20 |
| --- | --- | --- | --- | --- |
| A jump model | 0.38 | 0.545 | **fires** | 188 |
| A′ sparse jump | 0.38 | 0.524 | **fires** | 174 |
| B filtered HMM | 0.78 | 0.328 | **fires** | 68 |
| C gradient boosting | 0.66 | 0.399 | **fires** | 101 |
| C′ HAR-RV | 0.73 | 0.384 | **fires** | 93 |

We have 25 years. Settling whether regime conditioning beats volatility
targeting, at this effect size, needs **68 to 188**.

That is the result. Not "regimes do not work" — the data cannot say. The
honest statement is that the question is underpowered by a factor of three to
seven in sample length, and now we know by how much.

## What this does not say

* It does not say the overlays are harmless. Both jump models destroy value
  outright on the certainty equivalent, and that gap (−0.06% against +2.86%) is
  large relative to the noise band in a way the Sharpe difference is not.
* It does not say volatility targeting is good. It says it is the thing to beat,
  and that nothing here beat it.
* It says nothing about costs, which are not charged. The only family with a
  positive margin is the one that trades most.
