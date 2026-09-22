# Pre-registration — growth × inflation quadrants as a portfolio-construction input

**Status: DRAFT.** Not locked. Nothing in this document may be changed once it
is locked; deviations go to `docs/PROTOCOL_FREEZE.md` as amendments and are
never edited in place.

**Written: 2026-09-22, before any return was computed.** Every count, date and
transition rate quoted below comes from a script that reads parquet and
spreadsheet files and computes no performance whatsoever. Those scripts sit
beside this file.

---

## 0. Admission test

Six devices have been refuted in this programme. All six share the same
narrowness: the latent variable is realised volatility, the object switches
0.51 times a year, the use is sizing or timing, and the conditioned object is a
single 12-1 trend book over 46 instruments. This design must differ on at least
one axis. It differs on four.

| axis | the six refutations | this design | measured |
|---|---|---|---|
| 1. latent variable | realised volatility | growth and inflation **surprise against published professional consensus** | 180 usable quarters, 1981Q3–2026Q2 |
| 2. time constant | 0.51 transitions/year | **3.11 transitions/year** | 6.1× |
| 3. use | sizing, timing | **portfolio construction** (the objective function, not a dated decision) | never tested before |
| 4. conditioned object | one trend book, 46 instruments | **five risk sleeves, 30 instruments, 23.6 years** | 2003-12-05 → 2026-09-10 |

Admissible.

### 0.1 The measurement that shapes the whole design

Persistence of the quadrant, asset sample, quarterly grid:

```
P(same quadrant as last quarter)                0.220
P(same) under independent draws at this occupancy  0.266
```

The consensus-surprise quadrant is **not a persistent state**. It is
indistinguishable from a sequence of independent quarterly shocks, and is in
fact marginally anti-persistent. That is what an efficient consensus must
produce.

This forbids every ex-ante use of the label. It does not forbid the use
documented for Bridgewater: All Weather does not predict the quadrant, it
balances risk across all four so that no single one ruins the portfolio. The
label therefore enters the **objective function**, estimated on training folds
only, and is never read at execution time. No variant in this tree is a timing
bet, and none may be added later.

---

## 1. Data

### 1.1 The quadrant axes

Philadelphia Fed Survey of Professional Forecasters, free, downloaded and
opened on 2026-09-22 from
`https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/survey-of-professional-forecasters/data-files/files` (lowercase path):

| file | shape | coverage |
| --- | --- | --- |
| `median_rgdp_level.xlsx` | 232 × 12 | 1968Q4 – 2026Q3 |
| `median_cpi_level.xlsx` | 232 × 11 | 1968Q4 – 2026Q3 |
| `median_unemp_level.xlsx` | 232 × 12 | 1968Q4 – 2026Q3 |
| `median_indprod_level.xlsx` | 232 × 10 | 1968Q4 – 2026Q3 |

**Two filenames I guessed were wrong and the server did not say so.**
`median_ngdp_level.xlsx` and `median_cpi_growth.xlsx` both return **HTTP 200
with an 18,401-byte HTML page titled `Error - 404`**, byte-identical to each
other. Every file used here was verified with `file` (ZIP archive) and then
opened with pandas. A 200 is not a file. This is the Stooq trap and the OECD
country-code trap, and it is recorded here so the next reader does not repeat it.

**Surprise construction, fixed here.** For survey quarter *q*:

```
growth surprise_q    = (RGDP1_{q+1} − RGDP2_q) / RGDP1_q × 400      annualised points
inflation surprise_q =  CPI1_{q+1}  − CPI2_q                         percentage points
```

`RGDP2_q` is the consensus nowcast of quarter *q*'s level, published in survey
*q*; `RGDP1_{q+1}` is that same quarter's level as it was known one survey
later. Both sides were public when the later survey landed. `CPI1`/`CPI2` are
already annualised rates. Nothing revised later enters either side.

Measured: 180 usable quarters, growth surprise sd **35.481** annualised points,
inflation surprise sd **1.333** points.

**Availability stamp.** The surprise for quarter *q* is dated at the 14th day of
the second month of *q*+1. **The published SPF release-date table has not been
tested and must be used at implementation.** A declared sensitivity adds one
further month of lag; if the verdict moves between the two, it is undecided.

**Quadrant.** `label = 2·1[growth surprise > 0] + 1[inflation surprise > 0]`,
held from its availability stamp until the next. Signal at T−1, trade at T.

### 1.2 What the quadrant looks like on the asset sample

```
91 quarters, 22.5 years, 5,938 sessions from 2003-12-05

(growth−, inflation−)   841 sessions  14.2%   13 quarters
(growth−, inflation+)  1792 sessions  30.2%   28 quarters
(growth+, inflation−)  1630 sessions  27.5%   25 quarters
(growth+, inflation+)  1675 sessions  28.2%   25 quarters

transitions 70  ->  3.11 per year
median spell 1.0 quarter, maximum 3 quarters
```

Comparison objects, all measured:

| object | transitions/year |
| --- | --- |
| A′ sparse jump, the refuted classifier | 0.51 |
| SPF quadrant, full history 44.7 years | 2.95 |
| SPF quadrant, asset sample | **3.11** |
| H3 AR-proxy quadrant, 63-day kernel, quarterly grid | 2.27 |
| H3 AR-proxy quadrant, 63-day kernel, daily grid | 7.08 |
| level quadrant, yoy vs expanding median, quarterly | 1.20 |

### 1.3 The sleeve panel

`regime-lab/data/cache/trend_universe.parquet`, 6,822 sessions × 46 columns,
2000-07-17 → 2026-09-10. The headline takes five sleeves, **30 instruments**,
from **2003-12-05** (TIP is the binding start) to 2026-09-10 — **23.6 years**.

```
equity            12   ^GSPC ^NDX ^RUT ^GDAXI ^FTSE ^N225 ^AXJO ^GSPTSE ^HSI ^AEX ^FCHI ^IBEX
duration           4   SHY IEF TLT AGG
inflation-linked   1   TIP
commodity         11   CL=F HO=F NG=F HG=F PL=F ZC=F ZS=F ZW=F KC=F SB=F CT=F
precious metals    2   GC=F SI=F
```

Smallest Sharpe separable from zero, by configuration:

| configuration | start | years | n | resolvable SR |
| --- | --- | --- | --- | --- |
| **5 sleeves — headline** | **2003-12-05** | **23.6** | **30** | **0.632** |
| 6 sleeves, with credit | 2007-12-19 | 19.4 | 33 | 0.712 |
| 7 sleeves, with FX | 2007-12-19 | 19.4 | 44 | 0.712 |

Adding credit or FX costs 4.2 years and pushes the resolvable Sharpe from 0.632
to 0.712, above this programme's 0.70 gate. **Credit and FX are sensitivities
over 19.4 years and can never be the headline**, for the reason the M2 audit
recorded on 2026-09-21: the binding constraint is temporal depth, not width.

**N = 30 exactly, with no margin.** The N ≥ 30 rule is met at the instrument
level. At the decision level the breadth is **five sleeves**, and that is
declared here rather than discovered later; it is why the walk-forward folds are
thin (5 folds = 18.2 quarters and 1,187 sessions each).

### 1.4 On-disk series used, and one that cannot be used

Point-in-time frames, `series_id / period / available_at / value`, validated
through `regime_lab/data/pit.py`:

| file | rows | periods | span |
| --- | --- | --- | --- |
| `data/raw/macro/rate_cash_3m.parquet` | 9,177 | daily | 1990-01-02 → 2026-09-08 |
| `data/raw/macro/fin_baa_spread.parquet` | 9,171 | daily | 1990-01-02 → 2026-09-04 |
| `data/raw/macro/fin_aaa_spread.parquet` | 9,171 | daily | 1990-01-02 → 2026-09-04 |
| `data/raw/macro/macro_cpi.parquet` | 438 | 438 | 1990-01-01 → 2026-07-01 |
| `data/raw/macro/macro_indpro.parquet` | 439 | 439 | 1990-01-01 → 2026-07-01 |

⚠ **The eleven `data/raw/macro/` series carry one value per period and no stored
revisions** — 438 rows for 438 CPI periods. They are a publication lag applied
to a revised series, not a vintage history; `macro_indpro` drops from 141.8 to
108.8 between February and March 1990, which is a rebasing. **They cannot serve
as a surprise axis.** They are used for level variants, for the credit-spread
axis of B1, and for cash funding. Genuine vintages, revisions included, exist in
`macro-momentum/data/raw/h3_macro/` (13 series, e.g. `payems` 4,560 rows for 344
periods) and are the fallback daily axis.

External sources tested rather than assumed, all on 2026-09-22:

| source | status |
| --- | --- |
| FRED `T10YIE` | HTTP 200, real CSV, **6,188 rows, 2003-01-02 → 2026-09-21** |
| FRED `DFII10` | HTTP 200, real CSV, 6,188 rows, 2003-01-02 → 2026-09-18 |
| FRED `DTB3` | HTTP 200, real CSV, 18,971 rows from 1954-01-04 |
| FRED `GDPC1` | HTTP 200, real CSV, 319 rows from 1947-01-01 |
| total-return equity indices | **not available free; see killer 5** |
| SPF release-date table | **not tested; see killer 8** |

⚠ `openpyxl` is **absent from the regime-lab venv** (`ImportError`) and present
in the macro-momentum venv. Hygiene item B5. SPF reads go through the
macro-momentum venv, or the dependency is declared first.

---

## 2. Construction

Identical on both legs of every comparison, so that only the conditioning
differs.

* **Sleeve return.** Excess of the cash rate throughout. The nine bond and
  credit ETFs are downloaded with `auto_adjust=True` and are therefore total
  return: every one of them is funded at `rate_cash_3m`. This is the defect A4
  found and repaired, worth 0.0414 of Sharpe; it is not repeated here.
* **Sleeve aggregation.** Within a sleeve, instruments at equal risk.
* **Portfolio.** Daily volatility targeting at 10% annualised, using
  `regime_lab/extensions/vehicle.py`, whose target held at 10.84%–10.90% across
  three panels. The multiplier is lagged one session behind its own estimator.
* **Leverage cap.** 3.0 on the per-instrument multiplier, the reading logged on
  2026-09-21. Under that reading the cap is **inert**, which is the absence of a
  constraint and not a guarantee; gross exposure is reported at median, p95 and
  maximum.
* **Rebalance.** Quarterly, on the availability stamp of the quadrant, identical
  on both legs at levels A and B.
* **Timing.** Signal at T−1, trade at T, without exception.
* **Costs.** `vehicle.py` `COST_SCHEDULE`, round trip: equity 1.0 bp, FX 1.0,
  fixed income 1.0, commodity 1.5, cash vehicle 7.5. The 30-instrument blend is
  **1.217 bp**. Three columns (`headline`, `conservative`, `stress`) plus the
  literal `all_cash` reading are reported side by side.
* **Inference.** HAC Newey-West lag 6. Stationary block bootstrap
  (Politis–Romano), 2,000 draws, mean block 63 sessions. Never iid.
* **Walk-forward.** 5 folds minimum. Every cell-conditional quantity is
  estimated on the training fold only.

---

## 3. The escalation tree

Declared in full before any measurement. **A declared tree is not p-hacking; an
undeclared one is.** No branch may be added after locking without an amendment
in `PROTOCOL_FREEZE.md`, which raises the correction below.

### Level A — the quadrant as an evaluation partition and an objective, nothing more

**Legs.** (i) *Blind*: unconditional risk parity across the five sleeves.
(ii) *Balanced*: weights chosen so the four cells contribute equal shares of
portfolio variance, cell-conditional covariance estimated on the training fold
only. The current quadrant is never read in operation.

**Primary statistic.**

```
D = log(variance of the worst cell) − log(variance of the best cell)
```

A dispersion of variance, not of mean. This follows what the programme has
already measured about this class of object — it carries **variance** (+3.93
points of incremental R² on forward volatility, t −3.40) and **not mean**
(+0.030 point, t 0.27) — and it is also what All Weather actually claims.

**PASS requires all four:**

1. reduction in D of at least **0.204**, the measured MDE;
2. percentile ≥ 95 against **P1**, 2,000 random partitions preserving occupancy
   (0.143 / 0.308 / 0.275 / 0.275) and the spell distribution (median 1 quarter,
   maximum 3);
3. survival of **P2**, matching on mean gross exposure and ex-ante volatility
   session by session;
4. sign stable between the `headline` and `conservative` cost columns.

**Secondary statistic, reported and declared incapable of a PASS.** Net excess
Sharpe within each cell, each with its MDE beside it. The (growth−, inflation−)
cell holds 841 sessions = 3.34 years, below the 3.92 years at which the
resolvable-Sharpe inversion has a positive denominator: **its resolvable Sharpe
is infinite and it is declared undecidable here, in advance.**

**FAIL if** the reduction in D is below 0.204 (→ UNDERPOWERED, never a PASS) or
P1 matches it (→ the label carries nothing this partition-shape does not).

### Level B — chosen by *why* A failed

**B1, if A failed because P1 matched it.** The defect is then the **axis**, not
the use: a consensus surprise is unforecastable by construction (persistence
0.220 against 0.266) and therefore cuts the sample into near-random blocks,
which is exactly why a random partition did as well. B1 replaces the axis with a
**persistent, market-implied** one: inflation = change in the 10-year breakeven
`T10YIE` (6,188 daily observations from 2003-01-02, verified); growth = change
in `fin_baa_spread` (9,171 observations on disk). Same construction, same
criterion. **B1 escapes A's failure because its axis is persistent, and because
it is the expectation the portfolio is actually exposed to rather than the one
economists wrote down.**

**B2, if A failed because the gap was below the MDE.** The defect is then
**estimation**: equalising four variance contributions from 13 to 28 quarters
per cell is too noisy. B2 removes all conditional estimation and replaces it
with an **ex-ante sleeve → quadrant map**, fixed here and never adjusted:

```
equity            -> growth+
duration          -> growth−, inflation−
inflation-linked  -> inflation+
commodity         -> inflation+, growth+
precious metals   -> inflation+, growth−
```

Weights follow from a sign constraint, not from a covariance. **B2 escapes A's
failure because it estimates nothing on the thin cells.** **P3 is mandatory**:
2,000 random sleeve → quadrant maps. This is precisely the test H1 failed, where
its argued sign map landed at the **7th percentile** of random maps. The map is
worth nothing unless it beats its own permutations at the 95th percentile.

### Level C — if B falls, the one use left

Rebalance timing. The quadrant need not be predicted for a portfolio to
rebalance **on the day it changes** rather than on a fixed date. Compare the
same construction rebalanced on transition dates (3.11 per year, measured)
against a quarterly calendar (4.0 per year) — near-matched in frequency.
Placebo: random dates drawn from the same spacing law. This is an **execution**
question, the fifth use, and neither A, nor B, nor any of the six refutations
touched it.

### Closure

If C fails, the programme declares closure with the motive written here in
advance: *growth × inflation against expectations adds nothing to a sleeve set
already chosen to span it — tested as an evaluation label, as an axis, as an
ex-ante map, and as an execution grid.* That would be the programme's **first
closure on the construction axis**, covering four uses rather than one.

### Multiple testing, over the whole tree

Five tests are declared: A primary, A secondary, B1, B2, C. At most three will
ever run, but the correction covers all five, because the choice of branch
depends on a result.

| declared tests | per-test α (Šidák, family 0.05) | two-sided z |
| --- | --- | --- |
| 1 | 0.05000 | 1.960 |
| 3 | 0.01695 | 2.388 |
| **5 — this tree** | **0.01021** | **2.569** |
| 6 | 0.00851 | 2.631 |

**Threshold: |t| ≥ 2.569.** A sixth test added later moves it to 2.631 and is
logged as an amendment.

---

## 4. Power, stated before any coefficient

**Standalone, smallest Sharpe separable from zero, 80% power, two-sided 5%:**

| sample | years | resolvable SR |
| --- | --- | --- |
| 5-sleeve panel | 23.60 | **0.632** |
| cell (growth−, inflation+), 1,792 sessions | 7.11 | 1.569 |
| cell (growth+, inflation+), 1,675 sessions | 6.65 | 1.698 |
| cell (growth+, inflation−), 1,630 sessions | 6.47 | 1.757 |
| **cell (growth−, inflation−), 841 sessions** | **3.34** | **infinite** |

**Paired**, `MDE = 2.8016 · √(2(1−ρ)) · SE`:

| sample | SE | ρ=0.90 | ρ=0.95 | ρ=0.98 | ρ=0.99 |
| --- | --- | --- | --- | --- | --- |
| full panel, 23.6 yr | 0.2183 | 0.274 | **0.193** | 0.122 | 0.087 |
| one fold of five, 4.7 yr | 0.4882 | 0.612 | 0.433 | 0.274 | 0.193 |
| largest cell, 7.1 yr | 0.3977 | 0.498 | 0.352 | 0.223 | 0.158 |
| smallest cell, 3.3 yr | 0.5806 | 0.727 | 0.514 | 0.325 | 0.230 |

Programme reference points: K3 sizing MDE 0.246 against a gap of +0.068
(UNDERPOWERED); K4 regime resolution 0.043 against +0.014 (UNDERPOWERED).

**Variance statistic**, SE of log σ̂ ≈ 1/√(2n), blocked at 63 sessions because a
quarter is the independent unit:

| sample | n_eff | MDE on the log-ratio |
| --- | --- | --- |
| full panel, 5,938 sessions | 94.3 | **0.204** |
| largest cell, 1,792 | 28.4 | 0.371 |
| smallest cell, 841 | 13.3 | 0.542 |

**Verdict, written before measurement.** The question is **decidable on the
variance** for a reduction in extreme-variance dispersion of 22.6% or more — a
move from 3:1 to 2:1 is log(1.5) = 0.405, twice the MDE; a move from 3:1 to
2.5:1 is log(1.2) = 0.182, below it and declared UNDERPOWERED in advance. The
question is **not decidable on the mean** in the smallest cell at any effect
size. **A gap below the MDE is UNDERPOWERED and never a PASS.**

---

## 5. Cost gate, written before the test

Blended round trip over the 30 instruments at the `headline` schedule: **1.217
bp**.

| extra round trips/year on the conditioned leg | annual drag | Sharpe at a 10% vol target |
| --- | --- | --- |
| +2 | 0.024% | 0.0024 |
| +4 | 0.049% | 0.0049 |
| +8 | 0.097% | 0.0097 |
| +16 | 0.195% | 0.0195 |

**The hypothesis is declared dead on cost if the measured incremental drag
exceeds 0.10 of Sharpe.** At the `headline` schedule that needs **82 extra round
trips a year**; the conditioned leg will produce at most 4 (calendar) + 3.11
(transitions) ≈ 8. **This hypothesis will not die of friction.** It will die of
power or of placebo, and saying so in advance prevents a later failure from
being blamed on fees.

The one schedule that can kill it is the literal `all_cash` reading at 7.5 bp on
all 30, where 13.3 extra round trips a year suffice. Per the vehicle rule, a
result whose sign moves between `headline` and `conservative` is **undecided,
never a pass**.

---

## 6. Matched placebos

**P1 — partition placebo.** Random quadrant labels preserving the measured
occupancy and spell distribution, 2,000 stationary block-bootstrap draws over
quarters. Answers the question that kills: does the reduction come from *this*
partition, or would any partition of the same shape do as well? Severe precisely
because measured persistence is 0.220.

**P2 — exposure placebo, the T3 lesson.** Both legs matched session by session
on mean gross exposure and ex-ante volatility. T3 showed an apparent alpha
passing **entirely through beta** — 0th percentile on beta, median on mean. The
barrier study showed p(pass) buys with exposure at **0.2785 per unit**. No
unmatched difference is read.

**P3 — map placebo, level B2.** 2,000 random sleeve → quadrant maps. Exactly the
test H1 failed at the 7th percentile.

---

## 7. Known killers, and the direction each pushes

1. **The smallest cell is undecidable on the mean** — 841 sessions, 3.34 years,
   resolvable SR infinite. → variance statistic primary; per-cell Sharpe
   reported, never able to pass.
2. **Risk parity may already be quadrant-balanced.** The sleeves were chosen
   *because* they span growth and inflation; equalising cell variance may
   recover almost the same weights. → the most likely failure of level A, toward
   a null difference.
3. **The quadrant is near-independent quarter to quarter** (0.220 vs 0.266). →
   P1 becomes a near-perfect placebo; pushes against this partition carrying
   anything specific.
4. **H1 died on a sign map.** → pushes against B2; P3 is not optional.
5. **Equity sleeves are price indices, not total return**, while the nine bond
   and credit ETFs are total return and must be funded. Roughly 2 points of
   dividend a year. → mechanically tilts risk-parity weights **toward bonds**.
   Remedy: fund the ETFs at `rate_cash_3m`, and add a dividend yield back to the
   equity sleeve as a declared sensitivity.
6. **23.6 years starting in December 2003** contain one inflation regime change
   and one secular bond bull market. The (growth−, inflation−) cell holds 13
   quarters, concentrated around 2008–2015. → the cell may be **a date wearing
   the costume of a state**. Remedy: report each cell's composition by decade
   *before* the statistic.
7. **Decision breadth is 5, not 30.** → thin walk-forward folds: 18.2 quarters
   and 1,187 sessions per fold at 5 folds.
8. **The SPF availability stamp is approximated** at the 14th of the second
   month of *q*+1. Real dates vary; an error in the wrong direction is a leak. →
   use the published release-date table, **untested here**, and carry a declared
   sensitivity adding one further month of lag.

---

## 8. What is learned if the whole tree falls

1. The programme's **first measured macro time constant**: 3.11 transitions a
   year against A′ sparse jump's 0.51, and persistence **0.220 against 0.266 for
   independence**. A macroeconomic quadrant is **not a regime**; it is a
   sequence of near-independent shocks. Every future attempt to predict it is
   closed in advance by a number.
2. **Real consensus is free and verified**: 232 SPF surveys, 1968Q4–2026Q3, four
   files opened. H3 had to model the surprise and validate it at r = +0.667; we
   now know it did not have to. This feeds **C2**, the write-up of H3, which must
   state what was out of reach and what was not.
3. A **quadrant risk decomposition of a five-sleeve diversified book over 23.6
   years**, which the programme does not possess.
4. **The construction axis would be closed** — the one axis of the four that the
   six refutations never touched. The programme's narrow sentence could then
   widen honestly from sizing and timing to construction as well.
5. The measured fact that **this is not a cost question**: 0.0195 of Sharpe for
   sixteen extra round trips a year.

---

## 9. Hard rules inherited

Signal at T−1, trade at T · excess returns throughout · HAC lag 6 · walk-forward
≥ 5 folds · daily volatility targeting · N ≥ 30 instruments · stationary block
bootstrap (Politis–Romano), never iid · matched placebo mandatory · **a gap
below the MDE is UNDERPOWERED and never a PASS** · at most three modifications
per hypothesis.

## 10. Amendment log

| date | change | reason |
| --- | --- | --- |
| — | — | — |
