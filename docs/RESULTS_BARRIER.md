# The regime signal against a drawdown barrier

The last application of the central result that had not been refuted, and the
fifth independent arrival at the same conclusion.

```bash
.venv/bin/python scripts/run_barrier.py
```

The script needs the audited propfirm simulator, which lives in a separate
repository and is not a declared dependency of this package; it exits with an
explanation rather than an `ImportError` if that checkout is absent. The French
write-up, with the full parameter citations and the sensitivity tables, is
`docs/NOTE_BARRIER_FR.md`.

## Why the thesis was worth testing

The study established that regime classification carries **variance, not mean**.
Nothing monetises that in an ordinary book: T1, T3 and the Carver attenuator all
land inside the noise.

A propfirm account is not a Sharpe problem. It is a survival problem against a
drawdown barrier, where variance decides the probability of touching it directly.
So a signal that predicts only variance should be worth something *here* even
though it is worth nothing there. That was the argument, and it was well posed.

It is wrong, for a reason it did not anticipate.

## The result

Primary: A′ sparse jump, exposure-matched, Tradeify Select Flex 25K trailing
geometry, account volatility anchored to barrier depth. 6,128 sessions,
2003-03-12 to 2026-09-08. 4,000 pseudo-accounts, nested Politis-Romano bootstrap
at 300 × 300, mean block 63 with sensitivities at 21 and 126.

| | |
| --- | --- |
| Δp(pass) | **+0.28 points** (10.9% → 11.2%) |
| history standard error | 2.18 points |
| 95% CI | [−3.5, +5.0] |
| minimum detectable effect | **6.10 points** |
| verdict on magnitude | **UNDERPOWERED** — the effect is twenty times smaller than this sample resolves |

Durations, which the brief asked for ahead of any Sharpe:

| | constant | matched regime | Δ | se | MDE |
| --- | --- | --- | --- | --- | --- |
| sessions to target | 128 | 126 | −2 | — | — |
| sessions to failure | 90 | 90 | **+0.0** | 7.4 | 20.7 |
| longest spell underwater | 186 | 186 | −0.5 | 9.5 | 26.6 |

The controls do better on the quantity that matters: the one-line volatility
placebo takes 6 sessions off the longest spell underwater, and a parameter-free
volatility target takes off 11 while adding 16 sessions of survival.

## Why it is a non-effect and not merely an underpowered one

The real effect sits **inside** all three null distributions and **below the
mean** of two of them.

| null | draws | real | null mean | percentile |
| --- | --- | --- | --- | --- |
| realised-volatility quantile sizing | 200 | +0.3 pt | **+2.6 pt** | **16th** |
| the same, on Δ sessions to failure | 200 | +0.0 | +8.3 | **4th** |
| states permuted, run lengths preserved | 400 | +0.3 pt | −0.1 pt | 57th |
| circular rotation of the leverage path | 400 | +0.3 pt | +0.7 pt | 30th |
| familial, max over five families | 150 | +1.7 pt | +3.0 pt | 17th |

Head to head, the arm that knows about regimes loses to both arms that do not:
**−2.6 points** against the volatility placebo and **−2.3 points** against the
parameter-free volatility target, at all three block lengths. It beats both at
one of twelve rungs of the risk scale.

An effect at the 16th percentile of its own placebo is not a small effect. It is
the absence of one.

## The mechanism is arithmetic

Measured on the constant-size arm alone, across a forced grid of average exposures
from 0.90 to 1.12, p(pass) rises with a slope of **0.2785 per unit of exposure**.
The regime arm carries an average exposure of 1.0251, so the 0.0251 it holds above
the constant arm buys `0.0251 × 0.2785` = **+0.70 points** of p(pass) by exposure
alone. The observed Δp(pass) is **+0.28 points**.

**Residual: −0.42 points.** The arm delivers less than a constant book sized at its
own average exposure would have.

> Two different quantities here happen to share the digits 0.28, and conflating
> them is how a factor of 100 gets into a result. The **slope** is 0.2785 of
> p(pass) per *unit* of exposure — doubling average exposure would add 27.85
> points. The **Δp(pass)** is 0.28 of a *percentage point*. They are unrelated;
> the collision is a coincidence of this dataset.

The arithmetic closes to within one point on all four devices tested — residuals
−0.2, −0.4, +0.0, −0.6. And the volatility-quantile null's own +2.6-point gain is
likewise all exposure: 1.088 × 0.2785 = +2.44 expected. At this geometry, nothing
buys anything beyond its own average exposure.

Mean exposures: constant 1.000, regime with gross free 1.113, regime matched
1.025, volatility placebo 1.103, volatility target 1.116.

This is **T3's denominator illusion, found again at the barrier level**. There the
apparent alpha passed through beta; here p(pass) is bought with exposure, not with
variance timing.

## What the thesis was missing: a time constant

| quantity | value |
| --- | --- |
| median life of an account that fails | **90 sessions** |
| median sessions to target when reached | 128 |
| state transitions in the whole sample (A′ sparse jump) | **13** in 6,377 sessions |
| mean transitions inside a 90-session window | **0.18** |
| share of 90-session windows with **no** transition | **89%** |
| share of 360-session windows with no transition | 71% |

**Eighty-nine per cent of the accounts that fail live and die inside a single
regime state.** The signal has nothing to say during the lifetime of a propfirm
challenge. Its time constant is a year; the barrier's is a quarter.

### The bind this exposes

The mismatch is measurable per family, and it scales with run length:

| family | transitions | mean per 90-session window | 90-session windows with none |
| --- | --- | --- | --- |
| A jump | 11 | 0.16 | 87.3% |
| A′ sparse jump | 13 | 0.19 | 89.3% |
| B filtered HMM | 41 | 0.59 | 68.2% |
| C gradient boost | 344 | 4.90 | 30.1% |
| C′ HAR-RV | 450 | 6.25 | 17.5% |

So the fast families do have the right time constant. They are also the families
with the least to say: C′ HAR-RV is the weakest row of the central table at 0.192
points and t −1.84, and C gradient boost is the family T5 found unstable across
refits, at a traded-block ARI of 0.255 with 19% of blocks relabelled.

**The families that carry the information are too slow, and the families with the
right speed do not carry the information.** That is consistent with the familial
null: the best of five still sits at the 17th percentile.

## A standalone figure, and the one lead left open

This TSMOM book does not pass a Tradeify Select Flex 25K evaluation more than
**38%** of the time at any leverage. That is a property of the book, not of the
regime overlay, and it bounds what any overlay on it could achieve.

On **FTMO static-floor** geometry the matched regime arm extends median survival
by **+36 sessions**, which is consistent with AlphaSimplex — on a fixed floor,
surviving long *is* the objective. Two reasons it is not claimed as a result: the
parameter-free volatility target extends it by **60**, and that geometry runs
through a local adapter rather than the audited simulator. It is the only
direction this work leaves open, and it would need a real FTMO simulator.

## Errors found in this work, and named

**One: the leverage cap would have handicapped the two controls, and only them.**
The first version sized the controls at `VOL_TARGET / sigma_book` with
`VOL_TARGET = 0.10` on a book realising 4.54% volatility, so `0.10/sigma` had a
median of 2.96 and the cap of 2.0 **bit on 76.8% of sessions**, cutting the
controls' leverage standard deviation from 0.446 to 0.271 — 39% of the contrast
under test. The regime arm, which divides by per-state volatility, capped at 1.32
to 1.60 for four of five families. **The bug pushed in the direction that
favoured the thesis.** Corrected to a causal expanding mean of the book's own
volatility; the cap now bites on 0.00% of sessions.

**Two:** a stray factor of 100 in the exposure decomposition, which reported
+17.7 points explained for an arm at 1.113 exposure — larger than the entire
p(pass).

**Three:** the common sample was built as a suffix, so a contiguity assertion
fired on the first pass (two gaps; states ending 09-08 against the book ending
09-10).

Two mid-course amendments, disclosed with what they produced before the change.
The declared risk scale did not bracket 0.50; extended, the "maximise p(pass)"
calibration rule pointed at 25% account volatility, where the median account
resolves in **10 sessions** — a rung that destroys the mechanism under test. The
rule was replaced by a purely geometric anchor, account volatility set to barrier
depth. **The rung the abandoned rule pointed at gives Δp(pass) = −1.1 points**, so
the replacement did not favour the result, and the full scale is published so that
this can be checked.

## Verdict

**NULL on the nulls, UNDERPOWERED on the magnitude — both, and not one standing
in for the other.**

Fifth independent arrival, and the last the programme had in reserve. The
classifier measured it directly; the Shu et al. (2024) replication found it; the
Carver attenuator found it; T1, T3 and T5 found it; and the barrier geometry — the
only application still standing — finds it too. Regimes carry variance rather than
mean, and **no device monetises it, not even where variance is literally the
objective.**

What survives every comparison in this document is the thing the practitioner scan
named first: parameter-free volatility targeting, which knows nothing about regimes
and does better.
