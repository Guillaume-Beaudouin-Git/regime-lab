# PRESPEC — Change-point regimes as a signal SELECTOR on the 46-instrument library

**Status: LOCKED 2026-09-23**, by this commit, before any statistic of the escalation
tree was read. Any edit from here is an amendment in `docs/PROTOCOL_FREEZE.md`.

**What the instrument measured before the lock, which §9 deferred to measurement rather
than guessing.** The sample start is 2003-07-17 on the strict reading; the state A
pipeline — ER63, expanding standardisation, BOCPD, a 756-session expanding median and a
21-session dwell — consumes burn-in and leaves **4,969 usable sessions, 2007-08-23 to
2026-09-10**, on 46 instruments. The detector fires **244 changepoints, 12.81 a year**,
against the programme's 0.53. The state is balanced: trending on 47.5% of sessions.

The decision MDE for A1, by the procedure §9 fixes — stationary block bootstrap over the
date dimension, whole cross-section preserved per draw, under the null — is **1.42 / 1.43
/ 1.57 points raw at blocks 21 / 63 / 126, and 1.91 / 1.92 / 2.12 after the ×1.346 tree
correction**. The rotation null gives 1.92 corrected and is centred at +0.027 points, so
it is not biased by the statistic's own construction. **The decision threshold is the
widest reading, 2.12 points.** §9 declared an expectation of 2.7-5.4 corrected; the
measurement lands below it, which is favourable and is why A1 is posable at all.

No statistic of the tree has been read at the time of this lock.

**Programme:** `regime-lab`, post-closure line.
**Parent documents:** `docs/CHARTER.html` (frozen), `docs/PRESPEC_TREND_VEHICLE.md`
(locked 2026-09-21), `docs/RESULTS_TREND_VEHICLE.md`, `docs/EXTENSIONS.md`,
`../../2_Chantiers_derives/reversal-lab/docs/RESULTS.md`.
**Source firm approach:** Man AHL — Bayesian change-point detection separating trending
from consolidating phases; correlation-breakdown monitoring used to recalibrate position
sizing.

---

## 0. One sentence

The programme has refuted six devices that all condition a single trend book, by size,
on a volatility-ordered state that transitions 0.53 times a year; this study asks a
different question — whether a change-point state on a **path-shape** latent, at **11
transitions a year**, **selects** which member of a signal library runs on each
instrument, **without touching exposure**.

---

## 1. Prior measurements, disclosed

The design was not blind. Everything below was measured before this document was
written, on prices and latent variables only. **No return, Sharpe, drawdown or turnover
of any construction proposed here has been computed.** Scripts:
`measure_timeconstant.py`, `measure_two.py`.

**Q1 — the candidate latents, against realised volatility.** Active sample, 46
instruments, 2003-07-16 to 2026-09-10.

| latent | Pearson vs VOL63 | Spearman | n |
|---|---|---|---|
| ER63, book-median directional efficiency | **+0.096** | **−0.044** | 6,040 |
| ‖C₆₃ − C₂₅₂‖_F, correlation-matrix distance | **+0.070** | **+0.118** | 5,725 |
| ‖C₆₃ − C₂₅₂‖_F residualised on volatility | +0.009 | +0.075 | 4,780 |
| mean pairwise correlation, in level | +0.580 | +0.559 | 5,914 |
| first-eigenvalue share | +0.594 | +0.633 | 5,914 |

**Q2 — transition counts.** Re-measured on `data/cache/states.parquet`, 6,377 common
sessions, 24.4 years: A jump 13 (0.53/yr), A′ sparse jump 13 (0.53/yr), B filtered HMM
49 (2.01/yr), C gradient boost 344 (14.08/yr), C′ HAR-RV 450 (18.41/yr).

On the trend universe: BOCPD (Adams & MacKay 2007, Normal-inverse-gamma, hazard 1/250)
on ER63 gives **11.04 changepoints/year**; at hazard 1/60, 11.94. Page CUSUM on ER63:
13.92 (k=0.5, h=5), 4.73 (k=1.0, h=5). CUSUM on the residualised matrix distance: 7.26
(k=1.0, h=5), 15.66 (k=0.5, h=5). A plain expanding-median threshold on ER63 gives
**28.58/year**, mean state duration nine sessions.

**Q3 — signal-library geometry.** Sign agreement with sign(252,21): 74.4% for
sign(126,10), 64.1% for sign(63,5), 52.8% for sign(21,1). Sign flips per instrument per
year (median): 6.88 / 10.05 / 15.12 / 17.94 / 24.83 for (252,21) / (126,10) / (63,5) /
(42,1) / (21,1).

**Q4 — state occupancy.** ER63 below its expanding median: 51.4% of sessions; below the
expanding 33rd percentile: 34.0%; below the 25th: 25.0%.

**Consequences for blindness, stated plainly.** Three design choices were made *after*
seeing the table above and are therefore not out of sample. (i) ER63 was chosen as the
headline latent because it is orthogonal to volatility. (ii) The correlation limb uses
a matrix **distance** and not a **level**, because the level is 0.58 correlated with
volatility. (iii) The headline fast leg is (126,10) rather than (63,5), on the flip
rates of Q3 and the cost arithmetic of §6 — a cost decision, not a performance
decision, since no return has been seen. All three are disclosed so a reader can
discount them, in the same way and for the same reason that
`PRESPEC_TREND_VEHICLE.md` §1 disclosed the vehicle choice.

---

## 2. Admission test

The device must differ from the six refutations on at least one of four axes. It
differs on four.

| axis | the six refutations | this study | evidence |
|---|---|---|---|
| **1. latent** | volatility (states ordered by training volatility, `mapping.py`) | path directional efficiency; correlation-matrix distance | r = +0.096 / −0.044 and +0.070 / +0.118 with VOL63 |
| **2. time constant** | 0.53 transitions/yr | **11.04/yr**, 21× | BOCPD hazard 1/250, active sample |
| **3. usage** | portfolio sizing, or an ON/OFF switch | **selection between signals**, per instrument, exposure-neutral | §5 construction, §8 assertion P-E |
| **4. object** | one book, TSMOM 12-1 on 46 | a library, 3 speeds × 46 = 138 signal legs, plus one reversal leg | §4 |

Axes 3 and 4 carry the weight. Axis 3 is the one that escapes the family T1, T3 and the
Carver attenuator killed, because that family is defined by cutting risk and this device
cannot cut risk. Axis 2 is what makes axis 3 operable: a state with 0.53 transitions a
year cannot select anything 13 times in 25 years.

---

## 3. Sample

- **Universe:** the 46 instruments of `data/cache/trend_universe_m1.parquet`. No
  instrument is added, removed or substituted for any reason, including performance.
- **Start:** the first session on which at least 30 instruments have 252 sessions of
  price history. Measured on all three stored panels: **2003-07-16**.
  ⚠ `PRESPEC_TREND_VEHICLE.md` §3 states 2003-07-17 and `RESULTS_TREND_VEHICLE.md`
  reports 6,039 sessions. One session of disagreement. **This must be resolved and
  logged before this document is locked**, not afterwards.
- **End:** 2026-09-10. **Length:** 6,040 sessions, 23.15 years.
- **Correlation block:** the 28 instruments complete on every session from the start
  date. 28 < 30: the N ≥ 30 rule binds on cross-sectional *claims*, and the correlation
  latent is a market-state input rather than a cross-sectional claim. The conditioned
  book keeps all 46. This reading is declared here so that it can be contested rather
  than discovered.
- **No data is downloaded.** The study needs none. Back-adjusted commodity futures would
  be desirable and are not available at usable depth: `scripts/audit_vehicle_coverage.py`
  (2026-09-21) measured the substitutable intersection at 8.9 years, which lifts the
  smallest resolvable Sharpe from 0.639 to 1.253 and makes substitution undecidable.
  That decision is inherited, not re-litigated.
- Declared breadth, for any appeal to the fundamental law: mean pairwise correlation
  0.098 (0.130 on the 28-instrument block measured here), participation ratio 11.4,
  entropy-effective bets 20.7. **The breadth of this book is 11 to 21, not 46.**

---

## 4. The object: a signal library, and it is capped

Three trend speeds and one reversal leg. Nothing else. The cap replaces the "three
modifications" rule, which is written for a single hypothesis and does not fit a tree.

| leg | definition | role |
|---|---|---|
| S | `sign(P[t−21] / P[t−252] − 1)` | the base, unchanged from `extensions/trend.py` |
| **F** | `sign(P[t−10] / P[t−126] − 1)` | **headline** fast leg, chosen on cost |
| F′ | `sign(P[t−5] / P[t−63] − 1)` | declared sensitivity, not a separate trial |
| R | `−sign(P[t] / P[t−5] − 1)`, cross-sectionally demeaned | level C only, behind gate C0 |

No other lookback, no cross-sectional rank, no instrument selection, no stop, no
pyramiding. Adding one is an amendment.

---

## 5. Frozen construction

Every value is inherited from committed code, and the source is named.

| parameter | value | source |
|---|---|---|
| per-instrument risk scaling | `0.10 / σ_63d`, capped at 3.0 | `extensions/trend.py`, unchanged |
| portfolio volatility estimator | 63-day rolling sd of book returns, lagged one session | `extensions/vehicle.py`, unchanged |
| portfolio volatility target | 10% annualised | `VOL_TARGET`, unchanged |
| portfolio leverage cap | 3.0, on the multiplier | reading (a), settled 2026-09-21 |
| signal lag | one session | hard rule |
| cost charged | on the absolute change in each instrument's exposure | `strategies/costs.py`, unchanged |
| latent A | ER63 = `|log P_t − log P_{t−63}| / Σ|Δ log P|`, cross-sectional median | new, specified here |
| detector A | BOCPD, Normal-inverse-gamma (μ₀=0, κ₀=1, α₀=1, β₀=1), hazard 1/250, run-length truncation 600, input standardised on an expanding window with 252 minimum | new, specified here |
| state A | "consolidating" when the posterior-mean ER63 since the last changepoint is below its expanding median (756-session minimum) | new, specified here |
| latent B | `‖C₆₃ − C₂₅₂‖_F / sqrt(k(k−1))` on the 28-instrument block, residualised on VOL63 by causal expanding OLS (756 minimum) | new, specified here |
| detector B | two-sided Page CUSUM, k = 1.0, h = 5.0, reset at alarm | new, specified here |
| minimum dwell | no state change within 21 sessions of the previous one | new, cost control |

**The cap must not bind.** A cap binding on more than 5% of sessions is a design
failure, not a result — the barrier study found one biting on 76.8% of sessions and
cutting 39% of the contrast under test in the direction that favoured the thesis. The
binding rate is reported in every table. It is also recorded that under reading (a) the
cap is currently **inert** (0.00%), which is the absence of a constraint and not a
guarantee: gross runs 3.08 median, 5.30 at p95 and 43.8 at maximum on 2004-07-07.

---

## 6. Cost model, and the death threshold, declared before any result

Schedule inherited from `PRESPEC_TREND_VEHICLE.md` §6 and `extensions/vehicle.py`:
equity index / currency / fixed income futures 1.0 bp round trip, commodities 1.5,
any instrument on a non-futures vehicle 7.5. Conservative and stress columns at 2× and
5×. Class counts over the 46, measured: 13 commodity, 13 equity index, 9 currency,
8 cash vehicle, 3 fixed income.

- Equal-weight blended headline rate: **2.272 bp**; conservative **4.543 bp**.
- Turnover-weighted effective rate: **≈ 2.9 bp**, derived from the published measurement
  that eight instruments of 46 carry 69% of the cost.
- Base M3 book turnover: **58.92×/year** → 1.71%/year drag → **0.171 of Sharpe** at a
  10% target.

**Incremental turnover of the switch**, derived from measured flip rates (§1 Q3), 34%
consolidating occupancy (Q4), median gross 3.08 and 11.04 switches/year:

| fast leg | extra flips/yr | switch turnover/yr | total increment | drag at 2.9 bp | in Sharpe |
|---|---|---|---|---|---|
| **F = (126,10)** | 53.9 | 17.4 | **+24.6× (+42%)** | 0.71%/yr | **0.071** |
| F′ = (63,5) | 130.0 | 24.4 | **+41.8× (+71%)** | 1.21%/yr | **0.121** |

**Death thresholds, written before the test.** The hypothesis is dead on cost when the
incremental drag reaches the tree-corrected MDE of 0.331 Sharpe, i.e. 3.31%/year:

- **F dies at 13.5 bp** blended round trip;
- **F′ dies at 7.9 bp** blended round trip.

Under the headline schedule (2.9 bp effective) neither dies. Under the literal `ALL_CASH`
reading of §6 that `vehicle.py` deliberately retains — 7.5 bp headline, 15 bp
conservative — **F′ is dead outright and F dies in the conservative column.** A result
whose sign moves between the headline and conservative columns is reported as undecided,
never as a pass.

---

## 7. Returns convention

Net excess returns are the only decision metric. Futures returns are excess by
construction. Any instrument on a cash vehicle is funded at the 3-month cash rate from
`data/raw/macro/rate_cash_3m.parquet`, at the `available_at` date and never the `period`
date. The nine bond and credit ETFs are downloaded with `auto_adjust=True` and are
therefore total-return series that must be funded: `FUNDED_ON_CASH` and
`NO_FUTURES_VEHICLE` are different questions and are kept as different objects. That
conflation cost 0.0414 of Sharpe on the previous headline and the test asserting the two
tuples answer different questions is inherited.

---

## 8. Placebos, specified before any result

**P-E — exposure neutrality. Blocking, checked to machine precision.** The device
changes only the sign pattern. Assertion:
`|mean gross(switched) − mean gross(base)| / mean gross(base) < 0.02`, and the same on
the median and the p95. **A violation aborts the measurement and is logged as a
construction defect**; it is not a result to be discussed. The barrier study measured
p(pass) rising at 0.2785 per unit of exposure. While exposure does not move, that channel
is closed by construction rather than by argument.

**P-B — beta on the base book. Mandatory.** T3 showed an apparent alpha travelling
entirely through beta (0th percentile in beta, median in mean). Every conditioned arm is
regressed on the base book with HAC lag 6 and reports α, β and residual alpha. **A gain
whose β departs from 1 by more than 0.05 is treated as a disguised exposure gain**,
whatever P-E says.

**P1 — circular rotation of the state, matched on episode duration.** The observation is
the episode sequence, not the session. Each of 400 draws rotates the state series
circularly against the dates: the episode-duration law and the 11.04/year rate are
preserved exactly and only calendar alignment is destroyed. Unbiasedness is checked by
confirming the placebo mean lands on the unconditional statistic, as it did at +0.428
against 0.44. **Burn-in is purged before any percentile is read** — P1's published
"88.5th percentile" was a statistic of the first 63 sessions and reads 99.8% once purged.

**P2 — signals re-derived on block-shuffled prices.** Never a permutation of the P&L
column. The change-point detector is re-estimated on the shuffled prices, so the placebo
inherits the detector's own estimation noise.

**Parameter-free control, mandatory at every level.** The dynamic volatility target that
beat every device this programme has built (+0.244 against the attenuator's +0.144) runs
alongside each arm. **An arm that clears its placebo but loses to the one-line control
is refused.** That is the failure mode `EXTENSIONS.md` caught and it is written into the
protocol rather than left to judgement.

---

## 9. Power, computed before the model

**Portfolio level — declared underpowered in advance.** Via
`regime_lab.analysis.power.minimum_detectable_sharpe_difference`, stationary block
bootstrap, mean block 126, 600 draws, on paired synthetic 10%-volatility legs:

| sample | leg corr 0.98 | 0.95 | 0.90 | 0.80 |
|---|---|---|---|---|
| 23.2 yr | 0.108 | 0.148 | 0.307 | 0.384 |
| 20.0 yr | 0.122 | 0.206 | 0.276 | 0.378 |
| 11.6 yr | 0.143 | 0.302 | 0.289 | 0.409 |

The programme's own real-data anchor for this comparison shape (6,569 sessions) is
**0.246 / 0.264 / 0.273** at blocks 126 / 63 / 21, which the synthetic grid places near a
leg correlation of 0.92. A switched book differs from the base on 25.6% (F) to 35.9% (F′)
of instrument-sessions, so its leg correlation with the base will be roughly 0.88-0.93
and its MDE **0.28-0.31 before the tree correction, 0.331 after it**.

**Stated as a zero-cost result, before measuring:** a plausible conditional speed switch
is worth 0.10-0.30 of Sharpe in the published CTA literature; the decision threshold is
0.331. **This sample cannot resolve the portfolio question.** A2, B2 and C1 are therefore
reported but are not the gate. Claiming otherwise would repeat the Carver attenuator
(+0.144 under an MDE of 0.205) and H-b (+0.014 under a resolution of 0.043).

**Panel level — where the power is.** The six refutations collapsed the question to one
portfolio Sharpe difference, discarding the cross-section. The programme has already
shown on itself that a panel test resolves what a portfolio test cannot: +3.93 points of
incremental R² on forward volatility at t −3.40.

First-rank statistic:
`Δ = [hit(F | consolidating) − hit(S | consolidating)] − [hit(F | trending) − hit(S | trending)]`,
weighted by risk-scaled exposure so it maps to the book, over 46 × 6,040
instrument-sessions.

Analytic bracket, given declared breadth of 11.4 to 20.7 effective bets:

| clustering | effective n | MDE on the hit-rate gap |
|---|---|---|
| by date (signal × return is near-white) | 68,900 – 125,000 | ≈ 0.8 pp |
| by 63-session episode (the programme's block) | 1,090 – 1,990 | ≈ 4.5 – 6.0 pp |

The truth lies between and is not guessed here. **The pre-registration fixes the
procedure, not the number:** the decision MDE is measured by stationary block bootstrap
over the date dimension in blocks of 63, the whole cross-section preserved within each
draw, **under the null and before any interaction coefficient is read**. Declared
expectation 2-4 pp, 2.7-5.4 pp after the tree correction. Blocks 21 / 63 / 126 all
reported.

---

## 10. The escalation tree

A tree declared in advance is not p-hacking; an undeclared tree is. The three levels,
their falsification criteria and the correction are fixed here.

### Level A — speed selection on a trend/chop change point

Construction: in the trending state each instrument carries S; in the consolidating
state it carries F. Risk scales untouched, minimum dwell 21 sessions, signal at T−1.

- **A1 (decides).** Panel: does the state shift the relative directional accuracy of F
  over S, in the stated direction?
  **Falsification:** |Δ| below the null-bootstrap MDE, or the sign is negative ⇒ A1
  dead, go to B.
- **A2 (reported, not decisive).** Portfolio: net excess Sharpe gap ≥ **0.331**.
  **Falsification:** below 0.331 ⇒ UNDERPOWERED, never a pass.
- **Refusal condition at every level:** losing to the parameter-free volatility target.

*Why A may fail, named in advance:* F and S agree on 74.4% of instrument-sessions, so
the contrast is thin; and at 11.04 changepoints a year an episode lasts about 23
sessions, which may be too short to amortise a 1.46× flip rate. Detector latency —
0-13 days measured on the programme's own states — would eat a further large fraction
of a 23-session episode, and is reported explicitly rather than discovered.

### Level B — correlation break as a CONSTRUCTION input

**Why B escapes A's failure reason.** A conditions the *signal* on a scalar path-shape
latent. B leaves the signal alone and changes *how risk is spread across the 46*. That
allocation does not exist today: `extensions/trend.py` scales by σ_i alone and then
normalises by gross, so **there is no covariance term in the book at all**. B adds an
absent module; it does not tune a present one. Neither the thinness of the speed
contrast nor the 23-session episode length bears on it.

Construction: the covariance estimator's window resets at each detected break and
lengthens between breaks; that covariance feeds a risk allocation with the trend sign
imposed, renormalised to the same gross so P-E holds.

- **B1 (decides).** Does a break predict a rise in risk-model error — the gap between
  the book's predicted and realised volatility? This is a **variance** claim, the one
  channel where the programme already has positive evidence.
  **Falsification:** HAC lag-6 t below the tree-corrected threshold, or wrong sign ⇒
  B1 dead, go to C.
- **B2.** Portfolio: net excess Sharpe gap ≥ 0.331, plus a mandatory secondary — does
  the reset covariance hold the 10% target tighter than the fixed one?

*Known killer, not minimised:* the correlation **level** is 0.58-0.59 correlated with
volatility, and that object already failed inside `concentration.py` at all |t| < 1.6.
B is admissible only in its residualised matrix-distance form, and the residualisation
is locked here, before measurement.

### Level C — switch to mean reversion, behind an unconditional gate

**Why C escapes B's failure reason.** B still asks the trend signal to be right and the
risk structure to be improvable. C stops asking: in a detected consolidation, the
instrument's own 5-day move is faded rather than followed.

- **C0 (unconditional gate, half a day).** Is the 5-day cross-sectional reversal premium
  on the 46 instruments **gross of costs** positive over 2016-2026?
  **Falsification:** ≤ 0 ⇒ **C is dead on the mechanical-kill rule the programme already
  applied to US equities, and the tree closes.** Conditioning a premium that no longer
  exists is the search for the subset where a dead effect still lives, and `reversal-lab`
  explicitly refused to run it.
- **C1.** Only if C0 passes: does the consolidating state improve the reversal leg by at
  least the MDE? Same placebos, same correction.

*Known killer, not minimised:* the short-term reversal premium fell from 3.64 Sharpe in
1990-1999 to **−0.18 in 2020-2026** on the CRSP universe, a mechanical kill. That
measurement is on US-listed equities and its own document says "nothing here speaks to
reversal in other markets"; 46 cross-asset futures are not that object. The prior stays
low and C0 exists precisely so that the low prior is tested cheaply rather than argued.

### Level D — the closure, and what it states

If A, B and C fall, what is written is wider than anything the programme can write
today: the failure is **not** the volatility ordering of the state (crossed at
r = +0.096 / −0.044), **not** the 0.53/year time constant (crossed at 11.04/year),
**not** the sizing usage (replaced by exposure-neutral selection), and **not** the single
book (replaced by a 138-leg library). The four escapes a reader would raise against
"regimes do not monetise here" are closed one at a time, each with a number.

---

## 11. Multiple testing, over the whole tree

Primary tests: **A1, A2, B1, B2, C0, C1 = 6.** Holm-Bonferroni at family-wise
α = 0.05: rank thresholds 0.00833 / 0.0100 / 0.0125 / 0.0167 / 0.0250 / 0.0500. The
strictest two-sided z is **2.638** against 1.960, a factor of **1.346** on every MDE.

| quantity | uncorrected | **tree-corrected** |
|---|---|---|
| portfolio MDE, block 126 | 0.246 | **0.331** |
| portfolio MDE, block 63 | 0.264 | 0.355 |
| portfolio MDE, block 21 | 0.273 | 0.367 |
| panel MDE, declared expectation | 2 – 4 pp | **2.7 – 5.4 pp** |

**The count of six cannot rise without a logged amendment**, and an amendment recomputes
the correction and states what the previous count produced before the change.

Sensitivities that are reported but **not chosen between** are not separate trials, by
the rule already in force: the three cost columns, the three block lengths, the
`trend_universe_m1_verified` panel, the F′ = (63,5) leg. If a choice is ever made among
them, each becomes a trial retroactively and the deflation is recomputed.

**Deflated Sharpe, and the estimator that is fixed here.** The base is the 83 distinct
configuration hashes already in `data/trials.parquet` (805 rows), plus every
configuration of this tree, logged through `regime_lab.analysis.trials` including any
abandoned after one look. **The variance estimator is: the variance of Sharpe across
distinct `config_hash`, taking the last row per hash.** It is fixed here because its
absence is exactly what collapsed K5 last time, where the deflated excess ran from
−0.159 to +0.286 across four defensible collapses of the same file and the
pre-registration had fixed only the trial *count*.

---

## 12. Kill criteria, locked at lock time

- **KA1** — panel gap below the null-bootstrap MDE or wrong sign ⇒ A1 refuted, proceed
  to B.
- **KA2** — portfolio gap below 0.331 ⇒ UNDERPOWERED, reported as such, never a pass.
- **KB1** — break does not predict risk-model error at the corrected threshold ⇒ B1
  refuted, proceed to C.
- **KB2** — as KA2.
- **KC0** — the unconditional reversal premium is ≤ 0 gross over 2016-2026 ⇒ level C
  dead and the tree closes.
- **KC1** — as KA2.
- **KX — exposure.** P-E violated, or |β − 1| > 0.05 against the base book ⇒ the arm is
  void, whatever it reads.
- **KY — control.** Any arm that loses to the parameter-free volatility target is
  refused, whatever its placebo says.
- **KZ — cap.** The leverage cap binding on more than 5% of sessions is a design failure;
  the construction is rebuilt and the pre-cap result published alongside.

A clean kill on all of these is a valid and reportable outcome, and §14 says what gets
written in that case.

---

## 13. Hard rules inherited

Signal at T−1, trade at T. Excess returns, HAC lag 6, multiple-testing correction
whenever n_tests > 1. Walk-forward with at least 5 folds, genuinely evaluated, folds
partitioned before any fold result is read. Daily volatility targeting. N ≥ 30
instruments for any cross-sectional claim. Stationary block bootstrap (Politis-Romano),
never iid. Matched placebo on any conditional claim. **A gap below the MDE is
UNDERPOWERED and never a PASS.**

Operational: never re-run `scripts/run_phase2.py` (15-20 minutes, overwrites
`states.parquet`). Re-download nothing. Use the repository's own venv. Deviations go to
`docs/PROTOCOL_FREEZE.md`; the charter is never edited in place.

---

## 14. What gets published, in every outcome

| outcome | what is written |
|---|---|
| A1 passes | a measured mechanism: a change-point state on a volatility-orthogonal latent selects between signal speeds — the first positive conditional result in the programme |
| A1 fails, B1 passes | the correlation break is a risk-model input and not a return input, which is the variance-not-mean result arriving a second time through a different door |
| B1 fails, C0 fails | the reversal kill extends from US equities to cross-asset futures, scoped and dated, and `reversal-lab`'s open question is closed |
| any portfolio arm clears 0.331 | a deployable conditional book, and the programme's first |
| everything falls | the four qualifiers drop off the programme's narrow sentence, each with a number; `EXTENSIONS.md` §1 gets the measured cause of its own failure (+0.594 / +0.633); a pre-declared cost frontier at 13.5 bp for the whole switching family; a signal-level denominator of 138 audited legs; and the methodological result that a panel test with ~50× the effective sample still sees nothing — meaning the effect is **absent**, not hidden |

Every row is a result. That is the point of writing this before the measurement.
