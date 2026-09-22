# AHL level A — the state does not move the speed ranking

`docs/PRESPEC_AHL.md`, locked 2026-09-23 before any statistic of the tree existed.
A1 spent one trial; the register now holds 84 distinct configurations.

```bash
.venv/bin/python scripts/run_ahl_level_a.py          # the instrument, reads nothing
.venv/bin/python scripts/run_ahl_level_a.py --read   # A1, a trial
```

**A1 is refuted, on the sign.** Δ = **−0.068 points** of hit rate against a
tree-corrected decision threshold of **2.119**, and the stated direction was positive.

## Why this refutation is not like the six before it

Every earlier device in this programme died underpowered: the effect was smaller than
the instrument could resolve, so the honest verdict was "this sample cannot tell". Here
the instrument resolves 2.12 points and the effect is **0.068** — three per cent of the
resolution, with the wrong sign. This is not a failure to see. It is seeing nothing.

## The mechanism, measured rather than inferred

| | fast F (63, 5) | slow S (252, 21) | F − S |
|---|---|---|---|
| consolidating, 2,607 sessions | 49.331% | 49.805% | **−0.474%** |
| trending, 2,362 sessions | 49.041% | 49.447% | **−0.407%** |

The slow signal beats the fast one **in both states, by very nearly the same margin**.
Δ is the difference between those two differences, and it is −0.068 points. The state
does not modulate the speed ranking; it leaves it intact.

That is a stronger statement than "the switch does not pay". Man AHL's thesis is that
directionless markets are destructive for slow trend following and that a detector
should shorten the signal there. On this universe the detector fires 12.81 times a year,
separates the sample almost evenly — 47.5% trending — and the ranking of the two speeds
is the same on both sides of it.

Worth stating alongside: all four hit rates sit **below 50%**. Daily directional
accuracy is not how trend following earns, so this is not itself a defect; but a study
proposing to select between speeds on directional accuracy should say that the quantity
it selects on is at coin-flip in every cell.

## What was fixed before the lock, and what it cost

The day-one instrument split ER63 at its expanding median. §5 specifies BOCPD with a
named conjugate prior, the posterior-mean ER63 since the last changepoint, and a
21-session dwell. Building the frozen version instead costs 1,070 sessions of burn-in —
4,969 usable from 2007-08-23, against 6,039 from 2003-07-17 — and moves the decision
threshold from 1.33 to 2.12 points. Both are below the 2.7-5.4 the pre-registration
expected, which is why A1 was posable at all.

The rotation null sits at **+0.027 points**, so the statistic is not biased by its own
construction rather than by the state.

## Where the tree goes

To level B, as the pre-registration says it does, and B does not inherit A's failure
reason. A conditions the *signal* on a scalar path-shape latent, and the measurement
above says that latent does not separate the speeds. B leaves the signal alone and
changes **how risk is spread across the 46 instruments** — an allocation that does not
exist today, because `extensions/trend.py` scales each instrument by its own σ and then
normalises by gross, so **the book carries no covariance term at all**. B adds an absent
module rather than tuning a present one, and neither the thinness of the speed contrast
nor the 23-session episode length bears on it.

B1 is a **variance** claim — does a correlation break predict a rise in the gap between
the book's predicted and realised volatility — and variance is the one channel where this
programme already holds positive evidence.
