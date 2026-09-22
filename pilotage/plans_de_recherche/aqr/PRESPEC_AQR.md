# PRESPEC — Conditional factor timing on a cross-sectional equity signal library

**Status: DRAFT.** Not binding. Becomes binding only when the status line reads LOCKED,
the file is committed, and the lock is recorded in `docs/PROTOCOL_FREEZE.md` with a date.
Any deviation after the lock is an amendment in that register, never an edit in place.

**Drafted:** 2026-09-22. **Author:** research session, AQR angle.
**Scope:** the whole escalation tree — levels A, B, C and the sealed holdout. The
multiple-testing correction below is computed over the tree's total test count, fixed
here, before any of it runs.

---

## 0. One paragraph

The regime classifier built in `regime-lab` separates forward variance and not forward
mean, and six devices that used it to size or switch a single directional book have
failed, each losing to a one-line volatility quantile. This pre-specification tests a
different object with a different usage: a **library of three cross-sectional signal
books on 49 industry portfolios**, whose **relative weights** are tilted by a
**financial-conditions impulse** at **constant gross exposure**. The primary statistic is
a conditional information coefficient on a 49 × 439 panel, not a portfolio Sharpe on 439
months. The valuation-dispersion state that the reference firm actually uses is
**excluded in advance** on measured grounds: 0.12 sign changes per year and an AR(1)
half-life of 456 sessions cannot be resolved by 36 years.

---

## 1. What differs from the six refuted devices

Declared against the programme's four axes. Every figure is from a command run before
this file was written; the scripts sit next to it.

| axis | six refutations | here | figure |
|---|---|---|---|
| 1. latent variable | volatility-ordered partition | NFCI 13-week change | ρ with 63-day realised vol: **+0.323** vs +1.000 for the placebo, +0.639 for the NFCI level, +0.687/+0.733 for the cross-sectional dispersion features |
| 2. time constant | A′ sparse jump, **13 transitions in 6 377 sessions** | **2.42 transitions/year**, unchanged under a 21- or 63-session minimum dwell | **43.9 complete episodes over 36.3 years** vs 6.5 over 24.4 |
| 3. usage | portfolio sizing, ON/OFF switch | **relative selection between signals at constant gross** | Σ w = 1 at every date; gross and market beta matched by construction |
| 4. conditioned object | one TSMOM book (46 instruments) or a 60/40 | **three cross-sectional books over N = 49** | two of the three have opposite regime behaviour in the literature |

Axis 1 is a partial difference, not orthogonality: ρ = +0.323 is roughly 10 % shared
variance with realised volatility. That is why placebo P3 exists.

---

## 2. Data, fixed before the lock

### On disk, opened and counted

| file | content | rows / sessions | span |
|---|---|---|---|
| `data/raw/panels/industry_49.parquet` | 49 industries, daily percent returns | 451 388 / 9 212 | 1990-01-02 → 2026-07-31 |
| `data/raw/panels/size_bm_25.parquet` | 25 size-BM portfolios | 230 300 / 9 212 | same |
| `data/raw/panels/factors_5.parquet` | `ff_mkt-rf, smb, hml, rmw, cma, rf` — **no momentum factor** | 55 272 / 9 212 | same |
| `data/raw/macro/fin_nfci.parquet` | NFCI, weekly, 7-day publication lag encoded | 1 913 | 1990-01-05 → 2026-08-28 |
| `data/raw/macro/fin_{aaa,baa}_spread.parquet` | daily, 1-day lag | 9 171 each | 1990-01-02 → 2026-09-08 |
| `data/cache/features.parquet` | 50 features, expanding z-scores clipped at ±5 | 9 572 × 50 | 1990-01-01 → 2026-09-08 |

All 49 industries are populated on all 9 212 sessions of 1990-2026.

### To be fetched, availability tested and not assumed

| source | URL | verified |
|---|---|---|
| 49 industries daily, full history | `mba.tuck.dartmouth.edu/…/49_Industry_Portfolios_daily_CSV.zip` | downloaded, 4 186 243 bytes, **26 296 sessions, 1926-07-01 → 2026-07-31** |
| NFCI, full history | `fred.stlouisfed.org/graph/fredgraph.csv?id=NFCI` | downloaded, no API key, **2 906 weekly points from 1971-01-08** |
| ANFCI (cycle-adjusted), robustness variant | `…fredgraph.csv?id=ANFCI` | downloaded, same depth |
| FF momentum factor, daily (sensitivity only) | `…/F-F_Momentum_Factor_daily_CSV.zip` | downloaded, 85 202 bytes, 26 211 lines, last date **20260731** |
| 25-portfolio BE/ME (descriptive only) | `…/25_Portfolios_5x5_CSV.zip` | downloaded, 548 334 bytes, **1 201 monthly rows × 25** |

Nothing is written into any repository. Fetched files land outside the repositories until
the lock; on the lock they are stored through `regime_lab.data.store` with a sha256
manifest, under new names, never overwriting a stored panel.

### Samples, fixed here

| window | sessions | months | industries populated (mean / min) | role |
|---|---|---|---|---|
| 1985-01-01 → 1989-12-31 | — | — | 49.00 / 49 | formation buffer only, never evaluated |
| **1990-01-02 → 2026-07-31** | **9 212** | **439** | 49.00 / 49 | **headline walk-forward, 5 expanding folds** |
| 1966-01-01 → 1971-04-07 | — | — | 48.86 / 48 | formation buffer for the holdout |
| **1971-04-08 → 1989-12-31** | **4 733** | **225** | 49.00 / 49 | **sealed holdout, opened at most once** |

The holdout is sealed on the lock date. It is opened only if a level passes on the
headline window, only once, and only for the directional criterion in §7.

### What the panels are not

Ken French portfolio returns are total returns on paper portfolios with no transaction
cost embedded and no shorting constraint. Each book here is dollar-neutral and gross-one,
so its return is an excess return by construction; the cash balance is **not** credited at
the risk-free rate, which is the conservative direction.

---

## 3. Construction, fixed before the lock

**State.** `S_t = NFCI_t − NFCI_{t−13w}`, z-scored on an expanding window with
`min_periods = 504`, clipped to ±3, read at T−1 using only what the parquet's
`available_at` says was published by T−1 (7-day lag, already encoded). Traded at T.

**Books**, each cross-sectional over the 49 industries, rank-based, dollar-neutral, gross
one, daily volatility targeted to 10 %, monthly rebalance staggered over 5 business days:

- **B1** momentum, 12-month formation skipping the last month;
- **B2** low beta, 252-session beta against the equal-weight industry index;
- **B3** long-horizon reversal, 60-to-12 months — the value proxy available when BE/ME is
  not. Short-term reversal is excluded: the programme has already closed it (gross ≈ 0,
  premium down from 3.64 Sharpe in the 1990s to −0.18 since 2020).

**Tilt.** `w_k,t = 1/3 + δ · s_k · clip(z_{t−1}, −2, +2)/2`, with `Σ s_k = 0`,
`Σ|s_k| = 2`, `δ = 0.25`. Weights stay in [0.083, 0.583] and `Σ_k w_k = 1` at every date.

**Declared sign map, level A**, from the literature and not from our data:
`s = (B1 −1, B2 +1, B3 0)`. Momentum down when conditions tighten (momentum crashes occur
at rebounds from stress, Daniel & Moskowitz 2016); low beta up when leverage is withdrawn
(Frazzini & Pedersen 2014); the value proxy flat, because we cannot justify a sign and
declaring an unjustifiable sign is precisely how H1 failed.

**Hard rules inherited, no exceptions.** Signal at T−1, trade at T. Excess returns. HAC
lag 6. Walk-forward ≥ 5 folds. Daily volatility targeting. N ≥ 30 for anything
cross-sectional. Politis–Romano stationary block bootstrap, never iid. Matched placebo
mandatory. **A gap below the MDE is UNDERPOWERED and never a PASS.**

---

## 4. Power, computed before the lock

Calibration: Lo's standard error `SE = sqrt((1 + S²/2)/T)` with `(z_{α/2} + z_β) = 2.8016`
reproduces the programme's published reference exactly — 23.2 years → **0.638** (published
0.639), 19.9 years → **0.701** (published 0.70).

**Tree-wide correction.** Confirmatory tests: A, B, C-variance, C-Sharpe (gated on
C-variance), holdout. **N = 5.** Šidák: `α₁ = 1 − 0.95^(1/5) = 0.010206`,
`(z_{α₁/2} + z_β) = 3.4104`, an inflation of **+21.7 %** over the uncorrected threshold.

Within-level multiplicity across the three books is absorbed by taking the maximum
statistic against the P1 null distribution, not by a further Šidák term. This is declared
because counting the same multiplicity twice is as dishonest as not counting it.

**Primary channel — conditional IC.** With N = 49, the per-month standard error of one
cross-sectional rank IC is `1/√48 = 0.1443`.

| window | months | SE of the IC difference (HAC ×1.3) | MDE, Šidák n = 5 |
|---|---|---|---|
| headline, balanced states | 439 | 0.0179 | 0.0611 |
| **headline, 30/70 state split — the threshold that binds** | **439** | **0.0195** | **0.0666** |
| sealed holdout | 225 | 0.0250 | 0.0853 |
| both windows joined | 664 | 0.0146 | 0.0497 |

Target effect: unconditional cross-sectional momentum IC on industries runs 0.03–0.05; the
hypothesis implies roughly +0.06 in calm and −0.02 under tightening, so ΔIC ≈ 0.08. That
is above 0.0666 by a factor of 1.2 and no more. **If the true effect is 0.03 we will not
see it and we will write UNDERPOWERED.**

**Secondary channel — Sharpe — declared too coarse in advance.** The smallest standalone
Sharpe the timing overlay must carry to be resolvable is **0.615** on the headline window
(0.485 if the window were extended to 1971). The requirement is scale-invariant: δ
multiplies the effect and the MDE in the same ratio, so no choice of tilt size buys power.
Bootstrap cross-check on synthetic paired series with `regime_lab.analysis.power`, mean
block 63: paired Sharpe-difference MDE of 0.181 at ρ = 0.95 and **0.114 at ρ = 0.98**,
after the n = 5 inflation.

**Rejected in advance as underpowered.** A sign test over complete state episodes would
need a win rate of **0.757 over 44 episodes** (0.708 over 67). It is reported descriptively
and does not enter the test count.

**Excluded in advance on measured grounds.** The price-updated value spread —
9 065 daily observations, 1990-08-01 → 2026-07-31, built from the June BE/ME reformation
dragged forward by the legs' cumulative returns — has an AR(1) half-life of **456 sessions
(1.81 years)** and **0.12 sign changes per year** in a ±0.5 z band, i.e. **4.3 transitions
in 36 years**. Its correlation with realised volatility is the lowest of all candidates
(+0.106), so it passes axis 1 and fails axis 2. It is **not** in the confirmatory tree.
The raw Ken French BE/ME is worse still: mean |Δlog BE/ME| of **0.1939 in the July
reformation month against 0.0037 in the other eleven**, a ratio of **52.4×**.

---

## 5. Costs, and the level at which the hypothesis dies — written before testing

Programme schedule, round trip: futures ~0.01 %, **equities 0.05–0.10 %**, FX 0.15 %,
crypto 0.17 %.

Measured tilt turnover of the NFCI 13-week-change state: **1.84 units of |Δw| per year**.
At δ = 0.25 and Σ|s_k| = 2, incremental gross traded = **0.92 per year**.

| round trip | incremental drag | in Sharpe at a 10 % vol target |
|---|---|---|
| 0.05 % | 0.046 %/yr | 0.0046 |
| 0.10 % | 0.092 %/yr | 0.0092 |
| 0.20 % | 0.184 %/yr | 0.0184 |

**Death threshold.** The conditional branch is dead when the incremental drag equals the
paired MDE of 0.114: a round trip of **1.24 %** at δ = 0.25, **0.62 %** at δ = 0.50. The
equity schedule leaves a margin of 12× to 25×. **Costs are not the binding constraint
here; power is.** Writing this now forbids attributing a later failure to fees.

The books' own turnover is a separate matter. It cancels in the paired contrast but not in
the level. Levels will be published at 0.05 %, 0.10 % and 0.20 %. If the static composite
does not clear RESEARCH_PASS (Sharpe > 0.7, MaxDD > −25 %, WF > 50 %), the increment
remains a valid scientific result and is **explicitly not a tradeable one**.

---

## 6. Placebos

| | placebo | matched on | rejection rule |
|---|---|---|---|
| P1 | matched-transition null state: same transition count and same dwell-time law, random phase, 1 000 draws by stationary block bootstrap of the state sequence | the time constant | the realised ΔIC must exceed P1's 95th percentile |
| P2 | static tilt whose weight vector equals the time average of the conditional tilt | the average factor exposure | failing to beat P2 means the result is a static bet in disguise — the Asness critique of valuation timing, applied here unchanged |
| P3 | state = realised volatility of the equal-weight industry index below its expanding median, same usage, same δ | the usage | the one-line rule has beaten five devices; it must be beaten, not merely zero |
| P4 | exposure control: realised market beta and gross of both legs | exposure | \|Δβ\| ≤ 0.02 and identical gross to 1e−9, else the level is void |
| P5 | sign-map permutation over the 3! = 6 orderings | the sign map | descriptive only; with three books this test is weak and saying so is better than presenting it as strong |

---

## 7. The tree, with every falsification criterion fixed here

### Level A — declared sign map

Primary statistic: **ΔIC**, the difference between a signal's monthly cross-sectional rank
ICs in the two states, HAC lag 6, forward one-month non-overlapping returns.

**A falls if any one of these holds:**

1. no book reaches |ΔIC| ≥ **0.0666** with the declared sign;
2. the maximum ΔIC over the three books does not exceed P1's **95th percentile**;
3. fewer than **4 of 5** walk-forward folds carry the same sign;
4. P3's ΔIC is **greater than or equal to** the real state's;
5. leave-one-year-out: dropping any single calendar year takes ΔIC below 0.0666 — the
   result is then declared **fragile** and is not a PASS.

Secondary, reported but not confirmatory: the paired Sharpe increment of the tilt against
the static equal-weight composite, with its MDE. **Declared UNDERPOWERED in advance unless
it exceeds 0.114.**

*Prior: P(A passes) ≈ 12 %.*

### Level B — estimated sign map, walk-forward

**Why B escapes A's cause.** A can fall for three reasons: (i) the declared sign map is
wrong; (ii) the state carries no selection information; (iii) the tilt is too small.
Reason (iii) is excluded by construction, since the power requirement is scale-invariant.
B addresses (i): it asserts no sign, it **estimates** `s_k` from the conditional IC
**inside the training fold only** and applies it to the test fold. If B also falls, the
conclusion is no longer "we guessed the signs wrong" but "**no stable conditional map
exists**", which is strictly stronger.

**B falls if** criteria 1–5 of level A fail on the estimated map, **or** if the sign of the
estimated `s_k` fails to agree across ≥ 4 of 5 training folds — in which case we declare
"no stable conditional map exists" and move to C without running the test.

*Prior: P(B passes | A falls) ≈ 10 %.*

### Level C — the state conditions the between-signal covariance

**Why C escapes B's cause.** If A and B fall, the residual reason is (ii): the state does
not carry the mean. That agrees with everything the programme has measured — +0.030 point
of R² on forward returns, t 0.27. But the same programme measured that the state **does**
carry variance — +3.93 points of incremental R² on forward volatility, t −3.40. C asks the
only question the acquired evidence supports, at an aggregation level never tested: not the
size of the portfolio, which is what all six refutations tested, but the **covariance matrix
between the three books**.

**Primary criterion, on variance and not on Sharpe.** The out-of-sample realised variance
of the equal-risk composite under the state-conditional covariance must beat the same
composite under a static Ledoit–Wolf shrunk covariance, on **≥ 4 of 5 folds**, by more than
the MDE computed on log realised variance with the same block bootstrap.

**Gated second test.** If and only if the variance criterion passes, we ask whether the
variance gain converts into a resolvable Sharpe at constant target risk. If it does not, we
publish a variance result and **make no Sharpe claim**.

*Priors: P(C-variance) ≈ 35 %; P(C-variance and conversion) ≈ 10.5 %.*

### Sealed holdout — opened at most once

Opened only if a level passes, on **1971-04-08 → 1989-12-31, 225 months, 4 733 sessions**,
formation from 1966, 49 industries populated throughout. Criterion: **sign agreement** of
ΔIC and |ΔIC| ≥ half the headline threshold. The holdout resolves only 0.0853; this is a
directional replication on 18.7 years the programme has never touched, not a second hunt
for a p-value.

### Closure, if everything falls

Declared statement, with its ground: **"the regime carries neither the mean nor the
between-signal covariance of a cross-sectional equity signal library; the variance channel
is exhausted at every aggregation level this sample can resolve."** This is strictly
stronger than the programme's current statement, which covers only a trend book and a
60/40.

---

## 8. Sensitivities, declared non-confirmatory

They are reported and they cannot turn a KILL into a PASS.

1. **FF-5 + momentum as the library (N = 6).** Fails the N ≥ 30 rule outright. Reported
   because it is the literal AQR object, labelled as failing the rule.
2. **The 25 size-BM panel as a second test bed (N = 25).** Also fails N ≥ 30, and it
   overlaps the 49 industries on the same underlying stocks, so it is not independent
   evidence.
3. **ANFCI in place of NFCI**, the cycle-adjusted variant, verified available from 1971.
4. **δ = 0.50 in place of 0.25.**
5. **Cost schedule at 0.05 %, 0.10 %, 0.20 % round trip.**
6. **The price-updated value spread**, descriptive: its level, half-life and transition
   count, published as the measured reason not to time factors by valuation on a 36-year
   sample.

---

## 9. Known killers and the direction they push

| | killer | direction | how it is handled |
|---|---|---|---|
| 1 | the practitioner scan lists valuation-based factor timing among the things to avoid | toward the null | adopted, not escaped: measured at 0.12 sign changes/year and a 1.81-year half-life, and excluded from the tree |
| 2 | lesson 6, N ≥ 30 | toward the null | satisfied at N = 49 on the headline; the FF-5 library (N = 6) is non-confirmatory |
| 3 | the volatility quantile has beaten five devices | toward the null | ρ = +0.323, about 10 % shared variance; P3 is the judge |
| 4 | momentum crashes concentrate in very few months | toward a false positive | leave-one-year-out, criterion 5 of level A |
| 5 | paper portfolios, no costs, no shorting constraint, 1971 shorting was not free | toward an optimistic level | levels at three cost schedules; the holdout is read for sign only |
| 6 | 49 industries and 25 size-BM portfolios share the same stocks | toward overstated confirmation | declared non-independent |
| 7 | B3 is a value proxy, not value | interpretive | if B3 carries nothing we cannot separate "value does not respond" from "our proxy is not value"; written in advance |
| 8 | NFCI is built from ~105 series including volatility measures | toward the P3 channel | ANFCI as the pre-declared robustness variant |
| 9 | timing luck, > 100 bp annualised for fixed-date rebalancing | toward noise | 5-business-day staggering in the construction |
| 10 | NFCI is published with a 7-day lag | toward look-ahead | already encoded in the parquet; the state at T−1 uses only what was published by T−1 |

---

## 10. What this yields even if the whole tree falls

1. An audited cross-sectional equity reference book, in excess, correctly costed, over
   36.6 years and N = 49 — the equity counterpart of the trend book the programme has just
   built, and a denominator it has never had on this side.
2. A measured rather than folkloric reason not to time factors by valuation: half-life
   1.81 years, 0.12 sign changes per year, 4.3 transitions in 36 years, against a
   resolvable floor of 0.615 standalone Sharpe.
3. The measured superiority of macro **changes** over state **classifications** as
   conditioners: 2.42 vs 1.27 transitions per year, ρ +0.323 vs +0.639 with realised
   volatility. Transferable to the other firm angles.
4. The first test of the regime outside the trend / 60-40 world. A closure at level C
   would generalise the programme's central result by a full step.
5. If it is never opened, the 1971–1989 holdout stays sealed: 225 months, 49 industries
   populated throughout, available for a future hypothesis.

---

## 11. Reproducibility

Measurement scripts written before this file and kept beside it:
`measure_states.py` (first pass, contains the error named in §12),
`measure_states2.py` (the pass all quoted state figures come from),
`power_aqr.py`, `design_numbers.py`, `ic_power.py`, `holdout_power.py`.
Verification downloads in `verif/`.

**Not done, deliberately.** No book is built, no IC is realised, no Sharpe is computed, no
factor mean is conditioned on any state. The ΔIC whose threshold this document fixes has
never been calculated. The programme's rule is that the angle is written before the data is
touched, and the value of the four acquired falsifications depends on it.

## 12. Errors made while drafting, named

1. The cached features are expanding z-scores clipped at ±5, not levels. The first
   orthogonality pass took their logarithm, produced NaN for every negative value, and ran
   on 3 720 observations instead of ~9 100. It read ρ = −0.118 for the value spread where
   the correct figure is **+0.106**, and it read ρ = +0.785 for `vol_rv_63` **against
   itself**, which had to be 1.000 and should have stopped the pass immediately. Corrected
   in `measure_states2.py`; every state figure quoted here is from the second pass.
2. The Ken French BE/ME section was first taken to be annual. It is monthly, 1 201 rows —
   but the monthly variation is reweighting alone, 52.4× smaller outside the reformation
   month. The conclusion holds for a different reason than assumed, and the measurement,
   not the intuition, settled it.
3. A script named `numbers.py` shadowed the standard library's `numbers` module and broke
   the numpy import. Renamed `design_numbers.py`.
