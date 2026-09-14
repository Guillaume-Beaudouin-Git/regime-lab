# The three falsification controls the charter promised

`docs/RESULTS_FINAL.md` declared, in its limitations section, that three of the
charter's six falsification controls had never been implemented: T1, the
volatility-quantile placebo at strategy level; T3, the matched-exposure placebo;
T5, the adjusted Rand index between successive refits at paired cadence.

They are implemented now. Nothing was refitted for T1 and T3; T5 needed the
frozen protocol re-run to keep the fitted models, and that re-run reproduces
`data/cache/states.parquet` bit-for-bit before any T5 number is read.

```bash
.venv/bin/python scripts/run_t1_control.py
.venv/bin/python scripts/run_t3_control.py
.venv/bin/python scripts/run_t5_refit.py      # writes data/cache/t5_refits/
.venv/bin/python scripts/run_t5_control.py
```

All three are deterministic: re-running them reproduces these figures to the
character.

The three arrive at one conclusion, from three unrelated directions. **The
classifier is a stable, reproducible object, and the position rule built on it
carries no information a one-line volatility quantile does not already carry.**
That is the study's central finding, reached here for the fourth time.

---

## T1 — the volatility-quantile placebo, at strategy level

The placebo already existed one level down, on conditional moments. T1 is the
same comparison on the traded book.

| | |
| --- | --- |
| book | 60/40 in excess of cash, 9,570 sessions, 1990-01 to 2026-09 |
| evaluation | 6,377 out-of-sample sessions, 2002-04-01 to 2026-09-08 |
| placebo | 1 when 20-session realised volatility is below its expanding median, 252-session burn-in, 0 NaN inside the window |
| rule | `mapping.state_to_position`, identical on both legs, signal t-1 traded t |
| inference | stationary block bootstrap, mean block 63, 2,000 draws, paired index path; HAC lag 6 |

### The legs

| leg | Sharpe | vol | maxDD | CEQ | exposed | turnover |
| --- | --- | --- | --- | --- | --- | --- |
| base 60/40, hold | 0.47 | 10.5% | −35.6% | 2.16% | | |
| **placebo, volatility quantile** | **0.51** | 5.2% | −10.3% | 1.95% | 58.8% | 0.0315 |
| A jump | 0.54 | 7.5% | −22.3% | 2.67% | 81.7% | 0.0020 |
| A′ sparse jump | 0.55 | 7.9% | −22.3% | 2.81% | 84.1% | 0.0020 |
| B filtered HMM | 0.47 | 5.7% | −13.7% | 1.87% | 64.7% | 0.0077 |
| C gradient boost | 0.48 | 4.5% | −10.2% | 1.67% | 47.8% | 0.0540 |
| C′ HAR-RV | 0.57 | 5.1% | −13.4% | 2.23% | 58.1% | 0.0707 |

The placebo alone lifts the book from 0.47 to 0.51 and cuts the drawdown from
−35.6% to −10.3%. Every regime family is then measured against *that*, not
against the unconditional book.

### The gaps

| family | ΔSharpe vs placebo | MDE | 95% CI | t HAC6 | verdict |
| --- | --- | --- | --- | --- | --- |
| A jump | +0.04 | 0.44 | [−0.28, +0.35] | 1.34 | inside the noise |
| A′ sparse jump | +0.05 | 0.43 | [−0.26, +0.35] | 1.56 | inside the noise |
| B filtered HMM | −0.04 | 0.41 | [−0.33, +0.24] | 0.10 | inside the noise |
| C gradient boost | −0.02 | 0.43 | [−0.32, +0.30] | −0.61 | inside the noise |
| C′ HAR-RV | +0.06 | 0.22 | [−0.10, +0.21] | 0.58 | inside the noise |

A gap smaller than the minimum detectable effect is underpowered. It is not a
null, and it is never a pass.

**Stopping rule 1 asked for a positive gap above the MDE on three folds of five.
The best any family achieves is zero of five.** Per-fold gaps swing from −0.71
to +1.01 against per-fold MDEs of 0.35 to 1.36 — the fold-level test has no
resolution at all.

### What it would take

| family | \|ΔSharpe\| | MDE | shortfall | years needed against 25 available |
| --- | --- | --- | --- | --- |
| A jump | 0.036 | 0.443 | ×154.7 | 3,914 |
| A′ sparse jump | 0.046 | 0.435 | ×88.8 | 2,246 |
| B filtered HMM | 0.039 | 0.408 | ×112.2 | 2,840 |
| C gradient boost | 0.025 | 0.435 | ×305.1 | 7,722 |
| C′ HAR-RV | 0.059 | 0.222 | ×14.1 | 358 |

`docs/RESULTS_FINAL.md` declared the portfolio question underpowered by a factor
of two to four, from 46 to 101 years needed. That estimate was for the *effect
the study hoped to find*. T1 measures the *effect actually observed*, and it is
one to two orders of magnitude smaller again. The honest reading is not "the
regime overlay is no better than a volatility quantile" but "this sample cannot
tell the two apart, and the observed gap is nowhere near the size that would let
it."

### Sensitivities, declared before the primary was read

- **Placebo window 21 sessions instead of 20.** Gaps fall to +0.01 to +0.03. The
  headline result is not robust to a one-session change in the placebo's
  definition, which is itself a statement about how little separates the legs.
- **States with detection latency** (`states_offline.parquet`). Gaps *rise* to
  +0.06 to +0.08, A′ reaching t 1.81. Lagging the signal does not hurt it, which
  is what a variance signal with 0-13 day latency should do and a timing signal
  should not.
- **Placebo matched on realised exposure** rather than on the median. A′ falls to
  −0.02, A to −0.11. This construction reads the quantile from the family's own
  realised exposure and therefore uses hindsight; it is deliberately generous to
  the placebo and it overlaps T3.
- **The truncation trap, checked on purpose.** Building the placebo on the
  evaluation window only starts it at 2003-04-14, and the common sample is 6,377
  against 6,377 — zero sessions lost. Gaps then fall to −0.01 and −0.00. This
  project has twice manufactured a signal out of a truncated sample; here the
  truncation costs nothing and the gap still disappears.
- **Bootstrap block length** 21 / 63 / 126 / 252: MDEs move from 0.43 to 0.46 for
  A′. The conclusion does not depend on the block.

### Costs and multiplicity

Both legs trade, so both are charged. At 5 bp round-trip, C gradient boost goes
to −0.10 and C′ HAR-RV to −0.04; A and A′ improve, because the placebo turns over
0.0315 against their 0.0020. Šidák for five families gives α 0.0102, z 3.410
against 2.802, which raises every MDE by about 23%. Nothing changes category.

---

## T3 — the matched-exposure placebo

T3 answers the question T1 cannot: when a regime overlay beats a benchmark, is it
because the signal is informative, or because the overlay is simply *less
exposed*? A defensive rule compared with a fully exposed baseline wins
mechanically, and that win is beta, not signal.

Two placebo constructions, both matched on mean exposure and turnover:

- **markov** — preserves the marginal and the transition probabilities of the
  real position path. Acceptance 11-18%, so 9,400 to 15,800 draws were rejected
  to keep 2,000.
- **rotation** — preserves the entire leverage distribution exactly, by rotating
  the realised position path. Acceptance 100%.

Matching is verified rather than assumed: exposure and turnover agree with the
real path to four decimals, maximum gap 0.0000 to 0.0105.

### The result, and the mechanism

| signal | placebo | pct Sharpe | pct mean | pct alpha | pct beta |
| --- | --- | --- | --- | --- | --- |
| A jump, on/off | rotation | 93rd | 61st | 96th | **0th** |
| A′ sparse jump, on/off | rotation | 94th | 76th | 95th | **0th** |
| A′ sparse jump, sized | rotation | 89th | 15th | 89th | **0th** |
| B filtered HMM, on/off | rotation | 85th | 28th | 92nd | **0th** |
| C gradient boost, on/off | rotation | 87th | 45th | 89th | **0th** |
| C′ HAR-RV, on/off | rotation | 97th | 53rd | 97th | **0th** |
| **· one-line vol quantile, on/off** | rotation | **96th** | **49th** | **96th** | **0th** |

Read the columns, not the rows. Every family sits high on Sharpe and on alpha,
at or below the median on **mean return**, and at the **0th percentile on beta** —
lower beta than all 2,000 matched paths. The apparent alpha arrives entirely
through the denominator. Nothing is added to the numerator.

And the last row settles it: **the one-line volatility quantile produces the same
profile**, 96th percentile on Sharpe and alpha, 49th on mean, 0th on beta. A high
percentile here is a property of any de-risking overlay, not of regime detection.

All seventeen paired block bootstraps against the best-matched placebo path land
inside the noise. A′ sparse jump: +0.196 against MDE 0.421 (markov), +0.093
against 0.357 (rotation), +0.042 against 0.152 (sized). The largest observed gap
anywhere is C gradient boost at +0.339 against MDE 0.563. The one-line volatility
placebo gets +0.092 against 0.543 — the same verdict by the same margin.

T3 also puts a number on the effective sample. The position paths carry 209 to
496 independent decisions, not 6,377 daily observations. That is what the placebo
distributions are wide about, and it is why the MDEs in T1 are so large.

---

## T5 — adjusted Rand index between successive refits

T5 asks whether the states are a property of the model or an artefact of the
protocol. If the partition is rebuilt from scratch every six months and lands
somewhere else each time, the "regimes" are estimation noise with a name.

The charter names the sources T5 is meant to catch, and they are not label
switching: EM local optima under different initialisations, parameter drift in an
expanding window, and the choice of the number of states.

Cadence pairing is checked rather than assumed: all five families refit on the
identical 49 dates. Every model is scored on one common 6,377-session grid, so
ARI(i, i+k) is comparable across every i and every k with no window-length
confound.

**Label-convention invariance, verified:** max |ARI(x,y) − ARI(x, 1−y)| over
families is 0.000e+00. ARI cannot be fooled by the inversion that cost this
project three days (failure 7). That is a guard, not a finding, and it is reported
as the verification it is.

### Adjacent refits against three nulls

| family | mean ARI | median | p10 | min | circular null | Markov null |
| --- | --- | --- | --- | --- | --- | --- |
| A jump | 0.957 | 0.993 | 0.851 | 0.663 | 0.015 | 0.006 |
| A′ sparse jump | 0.958 | 0.997 | 0.851 | 0.656 | 0.013 | 0.007 |
| B filtered HMM | 0.940 | 0.976 | 0.873 | 0.397 | 0.037 | 0.007 |
| C gradient boost | 0.780 | 0.784 | 0.676 | 0.623 | 0.023 | 0.001 |
| C′ HAR-RV | 0.969 | 0.974 | 0.938 | 0.897 | 0.009 | 0.001 |

Adjacent partitions agree at 73 to 137 times the matched null. **All 48 adjacent
pairs, in all five families, sit above the 95th percentile of both nulls.** The
circular-shift null preserves each sequence's exact run-length structure and
marginal and destroys only the alignment, so this is not a persistence artefact.

The lag profile decays monotonically — A′ from 0.958 at lag 1 to 0.533 at lag 48,
24 years apart — which is parameter drift in an expanding window, the thing the
charter said to look for, behaving as it should.

### Where T5 finds a problem

| family | ARI, traded block | relabelled | worst block | date |
| --- | --- | --- | --- | --- |
| A jump | 0.938 | 1.0% | 0.000 | 2007-10-01 |
| A′ sparse jump | 0.906 | 1.6% | 0.000 | 2009-04-01 |
| B filtered HMM | 0.864 | 2.5% | 0.000 | 2005-09-30 |
| **C gradient boost** | **0.255** | **19.0%** | **−0.100** | 2021-04-01 |
| C′ HAR-RV | 0.949 | 0.7% | 0.548 | 2005-04-01 |

On the semi-annual block each model actually traded, C gradient boost is the
outlier by an order of magnitude: ARI 0.255, one block in five relabelled, and a
worst block *below zero* — worse than a random partition. C gradient boost
appears in the published central table at +2.13 points on forward volatility,
t −6.34. That number stands, because it is measured on the labels the study
traded; but T5 shows those labels are not a stable object across refits, and the
family should not be read as evidence that a regime was *detected*.

The zero worst-block values for A, A′ and B are the degenerate blocks — a block
in which the state never moves has no partition to compare, and ARI is 0 by
construction rather than by disagreement.

**Stress.** Using the repository's own mechanical dating rule
(`reliability.drawdown_reference`, >10% below the running 12-month high), no
family degrades around stress: differences −0.057 to +0.073, permutation p 0.34
to 0.93. Whatever else is true, the partition does not come apart in the periods
that matter most.

---

## What these three do and do not establish

They establish that the classifier is real: stable across refits at 73-137× the
null, invariant to the labelling convention, and bit-reproducible from a clean
re-run of the frozen protocol.

They establish that the position rule built on it is not distinguishable from a
one-line volatility quantile, and that its apparent alpha passes through beta
rather than through mean return.

They do not establish that the regime overlay is *worthless*. T1's gaps are 14 to
305 times smaller than what this sample can resolve. The correct verdict is
**underpowered**, and the project's own vocabulary keeps that distinct from null.

One family, C gradient boost, comes out worse than it went in.
