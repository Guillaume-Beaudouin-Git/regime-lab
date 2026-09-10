# The classifier works. The mapping was inverted.

Measured on the label sequence, with no portfolio anywhere. 6,377 out-of-sample
sessions, April 2002 to September 2026.

## The jump model detects every NBER recession, and was scored as its own opposite

|  | balanced accuracy | with labels flipped | kappa flipped |
| --- | --- | --- | --- |
| A jump model | **6.0%** | **94.0%** | 0.50 |
| A′ sparse jump | 4.9% | 95.1% | 0.56 |
| B filtered HMM | 84.7% | 15.3% | −0.14 |
| C gradient boosting | 75.0% | 25.0% | −0.13 |
| C′ HAR-RV | 77.4% | 22.6% | −0.12 |

Confusion matrix for the jump model, states as produced:

```
                    called WEAK   called STRONG
expansion                 5,228             715
NBER recession                0             434
```

**Every one of the 434 recession days sits in the state the model called
strong.** Perfect recall, a 12% false-positive rate on expansions, and by any
reading the best stress detector of the five — scored as the worst because the
ordering convention put its labels the wrong way round.

The convention is the reference implementation's `sort_by="cumret"`: states are
ranked by the cumulative return earned in each on the training window. The
volatile state apparently earned more cumulatively in training, which is
plausible when violent recoveries fall inside it, and the whole signal flipped.

**The portfolio layer could not have revealed this.** There it looked like a
model with a Sharpe of 0.16 and 18% exposure — a bad strategy, not an inverted
one. Only scoring the labels against a reference defined outside the project
exposed it. That is the argument for evaluating a classifier as a classifier,
made concrete.

Caveat on what the 94% means: NBER dates are published six to eighteen months
after the fact, so this is retrospective validation. The *states* are real-time
— filtered, never smoothed — so the claim is that the model flags, as it
happens, the periods later dated as recessions. Only the ordering of the two
labels is retrospective.

## The labels are stable, and recognisable in real time

| family | mean run | vs chance | hindsight changes | kappa online/offline |
| --- | --- | --- | --- | --- |
| A jump model | 245 d | 51× | 0.9% | 0.97 |
| A′ sparse jump | 199 d | 36× | 0.3% | 0.99 |
| B filtered HMM | 128 d | 54× | 0.7% | 0.99 |
| C gradient boosting | 18 d | 9× | 0.0% | 1.00 |
| C′ HAR-RV | 14 d | 7× | 0.0% | 1.00 |

Detection latency, which was expected to be a major cost, is **almost nil**.
Seeing the rest of the block would change under one percent of labels. Regimes
here are not recognised late; they are recognised as they happen.

Method agreement, chance-corrected, splits into two camps that disagree because
one of them is inverted: A with A′ at 0.83, B with C and C′ at 0.51-0.55, and
**−0.39 between the two camps**.

## It separates variance, not mean

| horizon 21 sessions | forward mean spread | forward vol spread | t (HAC) | MDE |
| --- | --- | --- | --- | --- |
| A jump model | +1.53% | +6.70% | 0.27 | 19.6% |
| A′ sparse jump | −0.03% | +6.77% | −0.00 | 21.2% |
| B filtered HMM | −2.58% | −5.42% | −0.73 | 11.3% |
| C gradient boosting | −0.84% | −4.54% | −0.32 | 8.0% |
| C′ HAR-RV | −3.12% | −4.99% | −1.04 | 9.2% |

The volatility spread is large, consistently signed, and correctly oriented for
every family once the inversion is accounted for. The mean spread is small,
inconsistently signed, and nowhere near its detection threshold.

For the correctly-oriented families the mean spread is **negative**: the calm
state earns *less*. That is not a defect of the model, it is the risk premium —
the compensation for holding through turbulence is collected in the turbulent
state.

And once a plain volatility quantile is in the regression, the state adds
between 0.007% and 0.071% of incremental R² on forward returns. Essentially
nothing.

## What this means

**The regime classification is real, reliable and recognisable in real time. It
carries information about forward variance and none about forward mean.**

It is therefore a **sizing** signal, not a **timing** signal — and the on/off
rule the study applied is a timing rule. The portfolio result was not evidence
that regimes fail; it was evidence that this signal was being asked the wrong
question, on top of an inverted label.

That is the difference from the literature. Papers report classification quality
and volatility separation, both of which are strong here, and apply the state to
scale exposure. This study converted the state into one bit of direction, which
the signal does not support, and then measured the outcome through a Sharpe
difference — the noisiest statistic available.

---

## Amendment, 2026-09-10 — the figures above are pre-audit

Every accuracy figure on this page was measured **before** the NBER audit and is
superseded by `RESULTS_FINAL.md`. The NBER label had been routed through the
lagged macro path, which handed the classifier up to 45 days of hindsight at
every regime boundary.

| | on this page | corrected |
| --- | --- | --- |
| NBER balanced accuracy, A′ | 95.1% / kappa 0.56 | **93.2% / kappa 0.53** |
| recession recall, A | 434 / 434 | **419 / 435** |

Perfect recall was an artefact of that hindsight. The ranking of the five
families, the inversion finding, the persistence table and the sizing-not-timing
conclusion are unaffected — the correction lowers a level, not an ordering.

`RESULTS_FINAL.md` is the citable source for any accuracy number.
