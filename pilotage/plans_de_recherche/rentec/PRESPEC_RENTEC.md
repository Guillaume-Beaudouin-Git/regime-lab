# PRE-REGISTRATION — Microstructure HMM as an execution-cost object
### Status: DRAFT · Version 0.1 · 2026-09-22 · Author: Claude Code (Rentec branch)
### Not frozen. Freezing requires §11 to be completed and the document committed unedited thereafter.

---

## 1. Question

Renaissance Technologies is documented as estimating hidden Markov states over
micro-regimes — liquidity shocks, volatility bursts, order-book condition — and using the
**transition probabilities** between them to reweight signals and to steer **short-horizon
execution**. This pre-registration tests the execution half of that claim on the data this
programme owns, and nothing else.

**It does not test a directional hypothesis.** No level of this tree produces a trading signal,
a position, or a return series. The outcome variable at every level is a **cost in basis points**
or a **turnover ratio**. This is a deliberate design choice justified in §5.

---

## 2. Admission test against the six prior refutations

The programme has falsified six devices (T1, T3, the Carver attenuator, the propfirm barrier
geometry, the Shu et al. 2024 replication, and H-b). All six share four narrownesses. This
design must differ on at least one axis, stated and quantified.

| Axis | The six | This design | Measured |
|---|---|---|---|
| 1. Latent variable | volatility (states ordered by training vol in `mapping.py`) | microstructure state over 6 features, 3 of them scale-free (variance ratio, return autocorrelation, relative volume) | state-vs-vol-tercile agreement 0.570–0.816 (chance 0.333) — **partial**; but a same-frequency realized-vol quintile explains **−0.016 to +0.186** of the log-spread variance, so the *target* is not volatility-driven |
| 2. Time constant | 13 transitions in 6,377 sessions = **0.514/year** | 1-minute HMM, K=3 | **4,879–21,258/year** on the FX/index/metal estate; **3,353–36,208/year** on crypto; **9,492× to 70,443×** the reference. Even the daily modal aggregate (level C) gives **86–147/year = 167–286×** |
| 3. Use | portfolio sizing, ON/OFF switch | **execution** (A, B) then **portfolio construction** (C) | untested by any of the six |
| 4. Conditioned object | one book, TSMOM 12-1 on 46 instruments | the mandatory order list of an already-frozen book, and its cost | by construction |

**Verdict: ADMISSIBLE on all four axes**, unreservedly on 2 and 3, on the target but not fully
on the state for axis 1, by construction on 4.

**Declared against the design:** axis 1 is the weak one. The state is not orthogonal to
volatility. What makes the design non-redundant is not the state but the target, and this is
tested directly by placebo P3 (§7), whose defeat will be recorded rather than assumed.

---

## 3. Frozen sample

**Source.** One-minute bar files held in a private sibling repository, outside this one.
27 instruments, 129,237,891 one-minute bars, 2003-08-08 → 2026-06-07, columns
`timestamp_utc, open, high, low, close, tick_volume, spread, real_volume`, timestamps in true
UTC. Integrity already audited on disk (that repository's manifest: bar-density audit, classified anomaly
windows, overlap correlations 0.99997–1.0, seam gate PASS on US500/US100/GER40).

**The `spread` column is a real floating spread** (89–1,017 distinct values per instrument) and
is **the target variable**, not a proxy. It is flagged as an **LP-class proxy**: it carries no
commission and no slippage, and comes from one counterparty class.

**Inclusion rule, fixed now, applied before any arm runs.**
- median 2021+ quoted spread ≤ **5 bps**;
- `spread > 0` coverage ≥ 0.75 over the instrument's own window;
- ≥ 1,000,000 bars.

On the measurements already taken this admits GBPUSD (0.36), USDJPY (0.26), XAUUSD (0.63),
US100 (0.73), GER40 (0.74), US500 (0.87) and excludes USOIL (6.53), COPPER (10.19),
NGAS (203.76). EURUSD (0.19 bps) fails the coverage rule at 0.61 and enters as a **separate
stratum**, never pooled. The remaining instruments (AUDJPY, AUDUSD, EURGBP, EURJPY, NZDUSD,
USDCAD, USDCHF, XAGUSD, UKOIL, US30, and the seven softs) are measured at freeze time and
admitted or excluded by the same rule, with the list written into §11 before any arm runs.

**Strata (fixed):** S1 = FX majors and crosses; S2 = indices; S3 = metals. Pooling is
equal-risk across strata.

**Walk-forward:** 5 non-overlapping contiguous folds over the common window, expanding
estimation, out-of-sample evaluation only. Signal state filtered at t−1, execution at t, no
exception.

**Excluded by pre-registration:** the crypto estate
(same private repository, 17 instruments, 2018-01-01 → 2026-06-01). Its
`spread` column is **identically zero** (max 0, non-zero fraction 0.0000 over 4,418,408 BTC
rows, verified). It cannot host a cost study. It may be used for state-estimation robustness
only, never for an outcome.

---

## 4. The state

**Model.** Gaussian HMM, K = 3, diagonal covariance, fitted per instrument on standardised
features, `hmmlearn` 0.3.3, `random_state = 0`, `n_iter = 100`, `tol = 1e-4`.
**K is fixed at 3 by this document.** The K ∈ {2,3,4} sweep already run (`timeconstant_btc.csv`)
counted transitions only, produced no outcome, and is logged as pre-data calibration.

**Features, all strictly backward-looking, 60-minute window:**
1. `park` — Parkinson range, rolling RMS of log(high/low). *The volatility axis, kept as the
   control channel, not removed.*
2. `vr` — log variance ratio, Var(5-bar sums)/(5·Var(1-bar)). *Scale-free; trend vs
   consolidation.*
3. `ac` — rolling first-order autocorrelation of 1-minute log returns. *Scale-free.*
4. `lvol` — log tick volume minus its rolling 24-hour median. *Scale-free.*
5. `spr` — log quoted spread in bps minus its rolling 24-hour median.
6. `amihud` — log(1 + |ret|/tick_volume) minus its rolling 24-hour median.

Instruments with `tick_volume = 0` (USOIL, COPPER, NGAS are already excluded on cost) drop
features 4 and 6; this is recorded per instrument at freeze time.

**Filtering rule.** At minute t the state used is the forward-filtered posterior computed on
information through t−1 only. Smoothed (Viterbi) states are forbidden in every arm and are used
only in §5's transition counts, which are descriptive.

---

## 5. Why the outcome is measured in basis points and not in Sharpe

Computed with `regime_lab/analysis/bootstrap.stationary_indices`, mean block 63, α 0.05,
power 0.80. The published programme reference is reproduced (0.637 against a published 0.639),
which validates the machinery:

| sample | MDE on a Sharpe difference |
|---|--:|
| 23.2 years (8,468 sessions) | 0.637 |
| 8.4 years (3,066 sessions, 24/7) | 0.960 |
| 5.6 years (2,043 sessions, 24/7) | **1.252** |

On the intraday window nothing below Sharpe ≈ 1.0 is decidable. **A return-space study of this
angle would be underpowered before it began.** That is recorded here as a zero-cost result.

In cost space, on 1,978 daily aggregates, mean block 21 days:

| | days | prize (bps) | bootstrap SE | **MDE** | prize/MDE |
|---|--:|--:|--:|--:|--:|
| BTC | 1,978 | 1.024 | 0.0826 | **0.231** | **4.4×** |
| ETH | 1,978 | 1.336 | 0.1011 | **0.283** | **4.7×** |
| SOL | 1,978 | 2.282 | 0.1423 | **0.399** | **5.7×** |
| DOGE | 1,978 | 2.194 | 0.1996 | **0.559** | **3.9×** |

**Correction recorded against this document's own first measurement.** Those prizes were first
computed from a Corwin-Schultz spread estimator, which is **non-positive on 42.4 % of 1-minute
bars**; clipped at zero its within-hour p25 is 0, so "mean − p25" and "mean − min" came out
identical (1.02 and 1.02 on BTC) — the tell of a degenerate measure. Re-measured with the
strictly positive Parkinson range: within-hour mean − p25 = **2.68 bps on BTC (41 % of the
6.95 bps level)** and **5.02 bps on SOL (34 % of 14.83)**. The order of magnitude holds. The
estate's real `spread` column makes the proxy unnecessary, which is the second reason the
venue is the estate and not crypto.

**Operative MDE.** Recomputed on the frozen estate sample before the first arm and written into
§11. Expected magnitude **0.2–0.6 bps per side**. Any gap below it is UNDERPOWERED and **never**
a PASS.

**Sharpe translation, written now so nobody re-frames it later.** The trend book's measured
rotation is 58.92×/year. A 0.5 bps/side saving is 0.5 × 2 × 58.92 = **59 bps/year**, i.e.
**+0.059 of Sharpe** on a book earning 3.59 %/year in excess at 10 % vol. That is above the K4
pair's own resolution (0.043) and far below the K3 sizing MDE (0.246). **Visible in basis
points, invisible in Sharpe.** This is the frame, not a shortfall of it.

---

## 6. Gate G0 — partition stability, before any test is spent

The daily classifier earned its standing partly through partition stability across
re-estimations at 73–137× the null. This state has **not** shown that. Measured instability,
same specification and same seed: BTC 2020 and 2022 give 3,721 and 3,353 transitions/year,
BTC 2024 and 2025 give 17,340 and 18,960 — a factor of five. Two fits emitted a
non-convergence warning.

**G0.** Re-estimate the HMM on the 5 walk-forward folds. Measure cross-fold partition
agreement (adjusted Rand on the aligned label sequence over the overlap) against a
block-permutation null, mean block 21 days, 1,000 permutations.

**Pass:** agreement ≥ **10× the null** on ≥ 4 of the 5 fold pairs. The bar is deliberately far
below the daily classifier's 73–137×, and is declared low.

**Fail:** the tree stops. **Zero tests are spent**, the multiple-testing ledger is incremented
by **0**, and the published result is: *a microstructure state at 5,000–21,000 transitions per
year is not a stable partition on this estate.*

---

## 7. Placebos — mandatory, three of them, matched on three different dimensions

- **P1 — random minute.** Matched on *opportunity*: same instrument, same window, same number
  of executions, minute drawn uniformly, 200 draws per window.
- **P2 — hour-of-day × slow level.** Matched on *trivial information*. This is the one-line
  control, the analogue of the expanding-median volatility quantile that beat all six prior
  devices. Measured share of log-spread variance: hour-of-day alone 0.007 (XAUUSD) to 0.625
  (GER40); hour-of-day plus a slow rolling level 0.407 (US500) to 0.802 (XAUUSD). The slow level
  absorbs the spread's own autocorrelation (lag-1 0.800–0.947, lag-1440 0.426–0.879), so no
  separate AR placebo is added and no extra test is spent. **P2 is the real opponent.**
- **P3 — realized-volatility quintile.** Matched on *the previous winner*. Measured at −0.016 to
  +0.186 of log-spread variance, i.e. expected inert. Included so that its defeat is recorded.

**Structural control at level B, not optional.** The arm's volume-weighted average execution
time must stay within **±2 minutes** of TWAP's. This is the direct analogue of T3's matched
exposure test, applied to time instead of beta: it forecloses mechanically the channel by which
an apparent cost gain would in fact be drift exposure.

---

## 8. The tree

### Level A — which minute · 4 tests

The book must execute once inside an imposed 60-minute window. The arm selects the minute whose
t−1-filtered state carries the lowest state-conditional expected cost.

**Outcome:** net implementation shortfall in bps against the arrival price — spread paid
**minus** drift incurred. The spread-only figure is reported but is never the criterion.

**Tests:** A1 pooled vs P1 · A2 pooled vs P2 · A3 pooled vs P3 · A4 stratum split S1 vs S2∪S3.

**PASS requires, simultaneously:** net saving over **P1 and P2** above the frozen MDE;
block-bootstrap p below the Holm threshold; ≥ 4 of 5 folds positive.

**FAIL modes, named in advance:**
- saving over P2 below the MDE → *redundant with trivial information*;
- spread saving positive but net shortfall ≤ 0 → *adverse selection*.

### Level B — what rate · 4 tests

**Why B escapes A's specific killer.** If A falls to adverse selection, it falls because waiting
exposes the order to drift. B never waits. The order is split into M = 12 five-minute slices,
**all of which execute**; the state modulates only the *weight* of each slice, weights summing
to 1 with a floor of 1/(3M). Drift exposure equals TWAP's **by construction**, and this is
checked, not hoped: §7's ±2-minute control disqualifies the arm outright if it drifts.

**Outcome:** volume-weighted net shortfall in bps against arrival price.

**Tests:** B1 vs uniform TWAP · B2 vs the P2-derived schedule · B3 vs the P3-derived schedule ·
B4 stratum split.

**PASS:** same MDE and Holm bar, ≥ 4 of 5 folds.
**FAIL mode, named in advance:** *the cost forecast is too noisy at the 5-minute slice level to
beat a flat schedule once the floor is imposed.*

### Level C — what band · 4 tests

**Why C escapes B's specific killer.** If B falls to slice-level forecast noise, C stops
forecasting at five minutes. It aggregates the state to the day (modal state; measured at 86–147
transitions/year, still 167–286× the A′ reference) and changes the **use** from execution to
**portfolio construction**: a no-trade band of width k × E[cost | state] on the existing trend
book, which rebalances only when the weight gap exceeds the day's state-conditional expected
cost.

**Outcome:** **annual turnover at constant tracking error**, not a Sharpe. Reference rotation
58.92×/year. Tracking error to the unbanded book must stay within **±0.5 %** annualised;
breaching it disqualifies the arm whatever its turnover.

**Tests:** C1 vs a static band · C2 vs a P2-derived band · C3 vs a P3-derived band · C4 stratum
split.

**PASS:** turnover reduction above the MDE at held tracking error, Holm-adjusted, ≥ 4 of 5 folds.
**FAIL mode, named in advance:** *the daily aggregate is a slow level in disguise*, which P2
will say.

### Level D — closure

If G0 passes and A, B and C all fall, the declaration is:

> A latent state estimated from OHLCV and a quoted spread, running at 5,000–21,000 transitions
> per year, adds nothing to execution cost beyond an hour-of-day table and a slow rolling level,
> under three uses and three time constants.

Narrow, measured, and symmetric to the statement the programme already holds at the other end of
the time-constant axis.

**Named but NOT in this tree's α budget:** Binance `bookDepth` (tested by HTTP header, 200,
337–521 KB/day, present 2023-06 through 2026-06, absent 2022-06 and earlier — five dates tested,
the exact boundary was not located) gives a genuine depth state for crypto perpetuals at roughly
170 MB per instrument-year. It covers the venue that has **no** cost column on disk, so it
requires its own pre-registration and its own cost measurement. It is not a continuation of this
tree.

---

## 9. Multiple-testing correction — computed in advance over the whole tree

**Total tests in the tree: 12** (4 at A, 4 at B, 4 at C; G0 spends 0).

- **Holm-Bonferroni** across all 12 at family-wise α = 0.05. The smallest p must clear
  **0.05/12 = 0.00417**, i.e. **z = 2.87**.
- **Noise ceiling:** E[max of 12 independent standard normals] ≈ **2.16 σ**. Any apparent effect
  below 2.16 σ is at the ceiling and is read as such, whatever its nominal p.
- All p-values come from a **stationary block bootstrap** (Politis–Romano, mean block 21 days,
  2,000 draws). **Never iid.**
- The programme's global test ledger is incremented by **12** whatever the outcome.

**Process rule inherited from the neighbouring repository's postmortem, accepted here
explicitly:** it is forbidden to re-pre-register a fresh null in order to reset the trial
counter on a space already opened. If the tree falls, it falls. A fourth idea on this estate
inherits the 12.

**Scope firewall.** The BTC price/regime space is declared exhausted in that postmortem:
n_trials ≈ 50, E[maxSR] ≈ 1.8–2.3, DSR 0.00 and 0.02–0.075 on the two promoted specs. **No
result from this tree may be converted into a Sharpe claim on BTC without inheriting those 50
trials.** This tree opens a distinct null — an execution-cost null — and says so.

---

## 10. Hard rules inherited

Signal at t−1, execute at t. Excess returns wherever a return appears at all. HAC lag 6.
Walk-forward ≥ 5 folds, never a single split. Daily vol targeting on any book touched.
N ≥ 30 for cross-sectional strategies — **N = 27 here; the rule binds only on cross-sectional
strategies and this is not one, but it is declared, and level C may not be converted into a
cross-sectional claim.** Stationary block bootstrap, never iid. Matched placebo mandatory.
A gap below the MDE is UNDERPOWERED and **never** a PASS.

---

## 11. To complete before freezing

- [ ] Final instrument list after applying §3's inclusion rule to all 27, with the measured
      median spread of each, including the seven softs not yet measured.
- [ ] Per-instrument feature availability (`tick_volume = 0` drops features 4 and 6).
- [ ] The operative MDE in bps per side, recomputed on the frozen estate sample.
- [ ] The five walk-forward fold boundaries, as dates.
- [ ] The band coefficient k and the reference weight series for level C.
- [ ] Sign-off that no arm has been run.

**This document is DRAFT. Once §11 is complete it is frozen and amended only through
`docs/PROTOCOL_FREEZE.md`, never edited in place.**

---

## Appendix — every number in this document, and the command that produced it

All scripts sit beside this file. The interpreter is
`.../.venv/bin/python`.

| number | script |
|---|---|
| transitions/year, dwell times, vol-tercile agreement | `measure_timeconstant.py`, `measure_stability_cost.py` |
| cross-year and cross-instrument instability | `measure_stability_cost.py` → `stability.csv` |
| MDE in cost space and Sharpe space | `measure_power.py` |
| spread distributions, corr(spread,\|ret\|), hour-of-day and slow-level R², spread autocorrelation, per-year `spread>0` coverage | inline heredocs, reproduced in `PLAN.md` §b and §d |
| free-source availability | nine `curl -sI` header requests, results in `PLAN.md` §b |

No market data was downloaded. `scripts/run_phase2.py` was not run. Nothing was written to any
repository. **No backtest return was computed at any point**, by design: the angle is written
before the data is asked about the hypothesis.
