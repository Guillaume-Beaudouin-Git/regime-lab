# AHL level A — the state does not move the speed ranking, on the leg that was read

`docs/PRESPEC_AHL.md`, locked 2026-09-23 before any statistic of the tree existed.
A1 spent one trial; the register now holds 84 distinct configurations.

```bash
.venv/bin/python scripts/run_ahl_level_a.py          # the instrument, reads nothing
.venv/bin/python scripts/run_ahl_level_a.py --read   # A1, a trial
```

> ## ⚠ Correction, 2026-09-22 — this is the sensitivity leg, not the headline
>
> **What follows was read on F′ = (63, 5). The locked §4 names F = (126, 10) as the
> headline fast leg** and (63, 5) as "a declared sensitivity, not a separate trial".
> I fixed `FAST = (63, 5)` in both `run_ahl_panel_power.py` and `run_ahl_level_a.py`,
> so the 2.12-point threshold was measured on the sensitivity leg as well.
>
> It is worse than a mix-up of two arms. §6 of the same locked text had already
> declared F′ **dead on cost** before any return was seen — it dies at 7.9 bp blended
> round trip, against F which survives to the conservative column. The pre-registration
> chose (126, 10) as the headline *for that reason*. I measured the leg it had already
> discarded.
>
> **So the locked A1 has not been read, and the refutation below is a refutation of
> F′.** What it establishes about F′ stands: on that leg, on an instrument resolving
> 2.12 points, the state moves the speed ranking by −0.068. What it establishes about
> the headline is nothing.
>
> I am not inferring the headline result from this one. The two legs are not
> interchangeable — F agrees with S on 74.4% of instrument-sessions against 64.1% for
> F′, so F is the *thinner* contrast — but "thinner, therefore also null" is an
> argument, not a measurement, and this programme does not publish arguments as
> measurements. The headline reading is owed, its own MDE must be measured under the
> null first, and whether to spend that trial is a decision about the tree rather than
> a repair. It is logged as open in `docs/PROTOCOL_FREEZE.md` and in `AVANCEMENT.md`.
>
> One consequence for the register: the trial logged as `ahl_level_a` carries
> `fast=[63, 5]`. §11 says a declared sensitivity is not a separate trial, so that
> entry is an over-count against the headline and an under-description of what was
> measured. It is left in place and annotated rather than deleted — removing a logged
> trial is exactly what the register exists to prevent.

**A1 was read on F′ and is refuted there, on the sign.** Δ = **−0.068 points** of hit
rate against a tree-corrected decision threshold of **2.119** measured on the same leg,
and the stated direction was positive.

## Why this refutation is not like the six before it

Every earlier device in this programme died underpowered: the effect was smaller than
the instrument could resolve, so the honest verdict was "this sample cannot tell". Here
the instrument resolves 2.12 points and the effect is **0.068** — three per cent of the
resolution, with the wrong sign. This is not a failure to see. It is seeing nothing.

## The mechanism, measured rather than inferred

| | fast **F′** (63, 5) — the sensitivity | slow S (252, 21) | F′ − S |
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
