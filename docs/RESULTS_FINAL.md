# Final results

Out of sample April 2002 – September 2026, 6,377 sessions, returns in excess of
the three-month bill, states ordered by training volatility, costs at two basis
points round trip.

**Revised 2026-09-10 after an independent audit.** Three of the figures below are
lower than the version published two days earlier. What changed and why is at
the bottom of this file.

A one-line causal rule — realised volatility below its own expanding median —
appears in every table as `· vol placebo`. It costs nothing and it decides which
of these results mean anything.

## Layer 1 — the labelling is reliable

| family | mean run | vs chance | intra-block instability | NBER bal. acc | kappa |
| --- | --- | --- | --- | --- | --- |
| A jump model | 456 d | 97× | 0.9% | 93.3% | 0.49 |
| A′ sparse jump | 456 d | 83× | 0.6% | **93.2%** | **0.53** |
| B filtered HMM | 128 d | 54× | 0.7% | 84.7% | 0.24 |
| C gradient boosting | 18 d | 9× | 0.0% | 75.0% | 0.12 |
| C′ HAR-RV | 14 d | 7× | 0.0% | 78.2% | 0.17 |
| · vol placebo | 34 d | 16× | 0.0% | — | — |

The project's own pre-registered drawdown rule scores 91.1% / kappa 0.44 against
NBER. The models are better, by four points of balanced accuracy and nine of
kappa — a real margin, and a smaller one than "the best stress detector of the
five" suggested.

Chance-corrected agreement between methods runs 0.28 to 0.83, all positive.
Before the ordering was fixed the two camps sat at −0.39, which was one sign
error in one wrapper and nothing else.

**On the third column.** It measures disagreement between the online and offline
labels *within each six-month block*. It cannot detect a uniform lag: a model
three months late in both passes scores 1.00. For families C and C′ it is 0.00
by construction, since their offline prediction returns the online one. Measured
separately against NBER peaks, real detection latency is **0 to 13 calendar
days** — which supports the conclusion better than the number that was published
as evidence for it.

## Layer 2 — it separates variance, and only variance

| family | mean spread | MDE | detected | vol spread | MDE | detected |
| --- | --- | --- | --- | --- | --- | --- |
| A jump model | −1.02% | 19.5% | no | −6.51% | 4.5% | yes |
| A′ sparse jump | −0.46% | 21.9% | no | −6.59% | 5.0% | yes |
| B filtered HMM | −2.58% | 11.3% | no | −5.42% | 2.9% | yes |
| C gradient boosting | −0.84% | 8.0% | no | −4.54% | 2.2% | yes |
| C′ HAR-RV | −3.12% | 9.2% | no | −4.99% | 2.4% | yes |
| **· vol placebo** | −2.22% | 10.1% | no | **−3.52%** | 2.4% | **yes** |

Five out of five detect the variance separation and zero out of five detect the
mean separation — but so does the placebo, and the states are *ordered by
training volatility*, so "they separate variance" is close to a tautology on the
persistence of that order. The spread alone does not establish anything.

**What does establish it** is the marginal contribution over a volatility
quantile, and it has to be read on both targets:

| | forward RETURNS | | forward VOLATILITY | |
| --- | --- | --- | --- | --- |
| | incremental R² | t | incremental R² | t |
| A jump model | +0.022 pt | 0.24 | **+3.47 pt** | −3.46 |
| A′ sparse jump | +0.030 pt | 0.27 | **+3.93 pt** | −3.40 |
| B filtered HMM | +0.007 pt | −0.22 | +2.26 pt | −4.59 |
| C gradient boosting | +0.023 pt | 0.52 | +2.13 pt | −6.34 |
| C′ HAR-RV | +0.008 pt | −0.30 | +0.19 pt | −1.84 |
| · vol placebo | +0.000 pt | −0.03 | **+0.004 pt** | 0.21 |

**This is the study's strongest positive result and it was missing from the
first write-up.** The fitted models add two to four points of R² about forward
volatility beyond what a volatility quantile already knows; the one-line placebo
adds nothing at all. About forward returns, everything — models and placebo
alike — adds nothing.

## Layer 3 — timing rule against sizing rule, all eleven rows

| family | rule | Sharpe | net | vol | maxDD | CEQ | turnover |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base 60/40 | — | 0.47 | 0.47 | 10.5% | −35.6% | 2.16% | — |
| A jump | on/off | 0.54 | 0.54 | 7.5% | −22.3% | 2.66% | 0.0020 |
| A jump | sized | 0.50 | 0.50 | 10.6% | −29.3% | 2.45% | 0.0014 |
| A′ sparse | on/off | 0.55 | 0.55 | 7.9% | −22.3% | 2.80% | 0.0020 |
| A′ sparse | sized | 0.49 | 0.49 | 10.5% | −31.2% | 2.40% | 0.0013 |
| B HMM | on/off | 0.47 | 0.46 | 5.7% | −13.7% | 1.83% | 0.0077 |
| B HMM | sized | 0.50 | 0.49 | 10.6% | −30.4% | 2.42% | 0.0057 |
| C gradient boosting | on/off | 0.48 | 0.42 | 4.5% | −10.2% | 1.39% | 0.0540 |
| C gradient boosting | sized | 0.52 | 0.50 | 10.0% | −32.5% | 2.49% | 0.0393 |
| C′ HAR-RV | on/off | 0.57 | 0.50 | 5.1% | −13.4% | 1.87% | 0.0707 |
| C′ HAR-RV | sized | 0.56 | 0.53 | 11.4% | −33.5% | 2.76% | 0.0749 |

The earlier version showed seven of these eleven rows, keeping the better rule
per family on gross Sharpe — a selection on the reported metric, unannounced.
The omitted rows are not all unfavourable: C′ sized has the best net Sharpe in
the table (0.53) and the second-best certainty equivalent.

Every improvement over the base book sits inside the 0.271–0.399 detection
threshold measured in T2. **At the portfolio layer nothing here is decidable.**

## What the project establishes

1. **The classification is reliable and measurable.** 93% balanced accuracy
   against NBER, kappa 0.53, persistence 83–97× chance, five methods in
   agreement, real-time latency of 0–13 days.
2. **What it carries is variance, not mean** — and it carries variance *beyond a
   volatility quantile*, which the placebo does not. Two to four points of
   incremental R² on forward volatility, t between −3.4 and −6.3.
3. **It is a sizing signal, not a timing signal.** An on/off rule is a timing
   rule.
4. **At the portfolio layer the question is underpowered** by a factor of two to
   four in sample length — except for A′ sparse jump, where the sample resolves
   0.27 and the measured difference is −0.03. There, the benchmark is not beaten
   and that is a measurement.

## What was corrected, and what remains open

Corrections after the audit, all lowering a published figure:

| | published | corrected |
| --- | --- | --- |
| NBER balanced accuracy, A′ | 95.1% / kappa 0.56 | 93.2% / kappa 0.53 |
| recession recall, A | 434 / 434 | 419 / 435 |
| detection threshold range | 0.33 – 0.55 | 0.271 – 0.399 |
| sample shortfall | factor 3 to 7 | factor 2 to 4 |
| years to settle | 68 – 188 | 46 – 101 |

The NBER label had been routed through the lagged macro path, giving the
classifier up to 45 days of hindsight at every regime boundary. It is now scored
against the calendar month it describes.

Open, and declared rather than quietly omitted:

- **Stop rule 2 fires and was never announced.** The charter reduces phases 4
  and 5 to a minimal demonstration if incremental R² on forward returns is below
  0.2 point for every family. Measured: 0.007 to 0.030 point. It fires. The
  finding on forward *volatility* is what makes the study worth continuing, and
  that is a departure from the frozen rule, not a satisfaction of it.
- **Three of the six falsification tests were never implemented.** T1 (the
  volatility-quantile placebo at strategy level), T3 (matched-exposure placebo)
  and T5 (adjusted Rand index between successive refits, at paired cadence) do
  not exist in this repository. T4 was described in the code as "the placebo
  test of the charter", which substituted one test for another.
- **The five folds are declared and never used.** `walk_forward` is called once,
  to take the first fold's start date. No per-fold result exists. The actual
  protocol is one contiguous out-of-sample block with 49 semi-annual refits,
  which is defensible but is not what the charter specifies.
- **The second frozen base strategy was never evaluated.** `equal_risk_momentum`
  has no call site. The charter says both are reported whatever happens.
- **Revisions are not stored.** The loader requests initial releases only, so
  `realtime_trace` is exercised by tests and not by the data. The panel serves
  the first print of a period forever, which is conservative rather than exact —
  the README claim that any past panel is rebuilt "exactly as it stood" is
  wrong, and is corrected there.
- **The trials log holds 83 distinct configurations**, not the 805 evaluations
  its row count suggests; repeated runs stack duplicates.
