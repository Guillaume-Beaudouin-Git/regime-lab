# Two Sigma level A — the first-moment channel (§6, §12.5-§12.7)

`docs/PRESPEC_TWOSIGMA.md`, LOCKED at `30f7d69`. Written by `scripts/run_twosigma_tree.py`; the numbers below come from its output and from `docs/artifacts/twosigma/`, never typed by hand.

<!-- INSTRUMENT:BEGIN -->
## Instrument (§13.1 steps 1-3)

**Nothing of the tree was read.** The instrument printed no mean, Sharpe, IC or per-state statistic of the real partition or of any real arm, and no Sharpe difference or beta of any placebo arm (§13.1 step 2). It computed none of the reading's statistics: no Δ_A1 or Δ_B2, no p-value, no `R`, `G` or `S_C`, no interval `h_fk`, no gate. It computed `m` and the state second-moment matrices, which the arms' demeaned legs need, and printed neither. What it printed is what §13.1 step 2 allows: counts, the turnover of the arms it builds with the kill line, realised sd and cap share, the MDEs of demeaned legs with T_A1, T_B2, MDE_C and S*, the null percentiles of A-2, B-1 and C-1 (real returns under content-free partitions, never the real labels), and the NFCI diagnostic (labels only).

Computed on 2026-09-23 at code commit `42011e9`. Every threshold is stored bitwise in `docs/artifacts/twosigma/thresholds.json`, sections `A`, `A:K=3`, `A:K=5`, `A:K=6`, `A:smooth21`, `A:d=0.25`, `A:d=1.00`; the reading refuses to run unless that file is committed, unmodified, and equal bitwise to its own recomputation, on inputs with the same SHA-256 and packages at the versions `uv.lock` pins (§13.1 step 4). The tables round what the file stores.

Packages pinned by `uv.lock`: numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, statsmodels 0.15.0, threadpoolctl 3.7.0.

| input file | SHA-256 |
|---|---|
| `data/raw/panels/industry_49.parquet` | `505e212b65c4e5df5cafe1a7df5c954bb8dfaf06d27a73e8645ba1090efc436a` |
| `data/raw/panels/factors_5.parquet` | `690046e548a434c2218769a8cb3c0701ab6075fae1c5cc29bc1a6f077e925159` |
| `data/cache/features.parquet` | `f531881f1da80d03f12ce4af961ddb8665f013af2e75312ab4f52f3722011a08` |
| `data/raw/prices/cross_asset.parquet` | `570bb4c9f31732e0a063e2e8198720872256e0d6c6fca3924a77e47c96b4da4c` |

### What was measured

- The partition of each variant refitted walk-forward (§12.2), its cells (§12.4) and its 1,000 uniform placebo draws, seed 0, `method="uniform"` (§12.11).
- `m` fitted per fold on the training folds (§12.5), the selector and the equal-weight control on the traded path (§12.3), both targeted and netted of 5 bp; the blinded bootstrap of their **demeaned** 5 bp legs on the 5,031 test sessions (2,000 draws, seed 0, blocks 21/63/126) giving the MDEs, T_A1 and SE* (§12.6 steps 1-2); their held turnover and the kill line (step 3); realised sd and cap share (step 5).
- A-2's null: `R^(j)` on each placebo draw, profiles recomputed with the draw's labels (§12.7). The real `R` was not computed.
- The NFCI diagnostic on the 18-feature partition (§12.6 step 6), labels only.

### Counts (§12.4, §12.6 step 5, §13.4)

| section | K | d | training cells | A-2 cells (per fold) | floor | abstaining (fold, state) rows | fallback test sessions | folds at the control | placebo exact |
|---|---|---|---|---|---|---|---|---|---|
| `A` | 4 | 0.50 | 19 of 20 | 14 (2/3/4/3/2) | 10.0 | 1 | 372 of 5,031 (7.4%) | 0 | yes |
| `A:K=3` | 3 | 0.50 | 15 of 15 | 11 (2/2/3/3/1) | 7.5 | 0 | 0 of 5,031 (0.0%) | 0 | yes |
| `A:K=5` | 5 | 0.50 | 24 of 25 | 15 (2/3/5/3/2) | 12.5 | 1 | 34 of 5,031 (0.7%) | 0 | yes |
| `A:K=6` | 6 | 0.50 | 28 of 30 | 17 (3/4/4/4/2) | 15.0 | 2 | 30 of 5,031 (0.6%) | 0 | yes |
| `A:smooth21` | 4 | 0.50 | 19 of 20 | 13 (2/3/4/2/2) | 10.0 | 1 | 372 of 5,031 (7.4%) | 0 | yes |
| `A:d=0.25` | 4 | 0.25 | 19 of 20 | 14 (2/3/4/3/2) | 10.0 | 1 | 372 of 5,031 (7.4%) | 0 | yes |
| `A:d=1.00` | 4 | 1.00 | 19 of 20 | 14 (2/3/4/3/2) | 10.0 | 1 | 372 of 5,031 (7.4%) | 0 | yes |

### A-1: resolution and threshold (§12.6 steps 1-2, §12.12)

Blinded bootstrap SE of the paired Sharpe difference on demeaned legs, its MDE at α 0.05 (block 63, for reference) and at α 0.05/6 per block; `T_A1 = max(0.338, max_b MDE(0.05/6, b))`; Lo's standalone threshold at 0.05/6 is printed beside it and decides nothing (§12.12).

| section | SE 21 | SE 63 | SE 126 | MDE(0.05) 63 | MDE(0.05/6) 21 | MDE(0.05/6) 63 | MDE(0.05/6) 126 | T_A1 | block* | Lo standalone |
|---|---|---|---|---|---|---|---|---|---|---|
| `A` | 0.0654 | 0.0624 | 0.0587 | 0.1747 | 0.2277 | 0.2170 | 0.2042 | **0.3380** | 21 | 0.932 |
| `A:K=3` | 0.0831 | 0.0795 | 0.0808 | 0.2227 | 0.2891 | 0.2766 | 0.2810 | **0.3380** | 21 | 0.932 |
| `A:K=5` | 0.0604 | 0.0564 | 0.0540 | 0.1580 | 0.2101 | 0.1963 | 0.1879 | **0.3380** | 21 | 0.932 |
| `A:K=6` | 0.0765 | 0.0734 | 0.0700 | 0.2056 | 0.2662 | 0.2553 | 0.2438 | **0.3380** | 21 | 0.932 |
| `A:smooth21` | 0.0741 | 0.0718 | 0.0710 | 0.2010 | 0.2578 | 0.2497 | 0.2469 | **0.3380** | 21 | 0.932 |
| `A:d=0.25` | 0.0350 | 0.0337 | 0.0317 | 0.0945 | 0.1217 | 0.1174 | 0.1103 | **0.3380** | 21 | 0.932 |
| `A:d=1.00` | 0.1065 | 0.1018 | 0.0977 | 0.2852 | 0.3706 | 0.3542 | 0.3401 | **0.3706** | 21 | 0.932 |

### A-1: turnover and the kill line (§12.6 step 3), sd and cap share (step 5)

Held turnover ×/yr on the test sessions, restricted first; `K_kill = min(12, T_A1 × σ_sel × 10,000 / 20)`; the kill fires if ΔT > K_kill.

| section | selector ×/yr | control ×/yr | ΔT | K_kill | kill | σ selector | σ control | cap share selector | cap share control | missing test sessions |
|---|---|---|---|---|---|---|---|---|---|---|
| `A` | 43.81 | 36.66 | 7.14 | 12.00 | no | 8.13% | 7.88% | 66.9% | 70.5% | 0 / 0 |
| `A:K=3` | 43.96 | 36.66 | 7.30 | 12.00 | no | 8.03% | 7.88% | 72.2% | 70.5% | 0 / 0 |
| `A:K=5` | 45.80 | 36.66 | 9.14 | 12.00 | no | 8.01% | 7.88% | 69.7% | 70.5% | 0 / 0 |
| `A:K=6` | 45.82 | 36.66 | 9.16 | 12.00 | no | 7.94% | 7.88% | 72.0% | 70.5% | 0 / 0 |
| `A:smooth21` | 40.58 | 36.66 | 3.92 | 12.00 | no | 8.15% | 7.88% | 68.9% | 70.5% | 0 / 0 |
| `A:d=0.25` | 39.53 | 36.66 | 2.86 | 12.00 | no | 7.97% | 7.88% | 70.0% | 70.5% | 0 / 0 |
| `A:d=1.00` | 50.86 | 36.66 | 14.20 | 12.00 | **fires** | 8.39% | 7.88% | 62.6% | 70.5% | 0 / 0 |

### A-2: the null (§12.7)

Percentiles of `R^(·)` over the 1,000 draws. `q99 − q50` is the resolution quoted in a NOT SHOWN (§13.6).

| section | cells | q50 | q95 | q99 | q99 − q50 |
|---|---|---|---|---|---|
| `A` | 14 | 0.0030 | 0.1862 | 0.2624 | 0.2593 |
| `A:K=3` | 11 | 0.0077 | 0.2106 | 0.2744 | 0.2667 |
| `A:K=5` | 15 | -0.0028 | 0.1621 | 0.2412 | 0.2440 |
| `A:K=6` | 17 | 0.0004 | 0.1466 | 0.2328 | 0.2325 |
| `A:smooth21` | 13 | -0.0023 | 0.1907 | 0.2374 | 0.2397 |

### NFCI diagnostic (§12.6 step 6, §3)

The partition refitted on the 18 features without `fin_nfci` and `fin_nfci_chg13w` (same K, `n_init`, seed, folds, training start), against the declared one, labels only:

- out-of-sample label agreement after `context.align_labels`: 4,037 of 5,031 test sessions (80.2%); fold 1 73.7%, fold 2 72.8%, fold 3 70.8%, fold 4 90.1%, fold 5 93.9%;
- out-of-sample transitions per year: 15.60 (declared partition: 12.55);
- pooled η² against log rv: 0.058 (declared: 0.142).

### UNDECIDABLE before the reading (§13.4)

None: no lock of this level is UNDECIDABLE before its reading, at the primary or at any sensitivity.

### The criteria, copied from the lock

Verbatim from `docs/PRESPEC_TWOSIGMA.md` (LOCKED, `30f7d69`). They were fixed before any statistic existed; this file only restates them beside the thresholds.

**A-1, §12.6:**

> **Statistic.** `Δ_A1 = SR(selector) − SR(control)` over the 5,031 pooled test sessions.
>
> 2. **Threshold:** `T_A1 = max(0.338, max_b MDE(0.05/6, b))`.
>
> 3. **Kill turnover** *(lock, forced by measurement, §5)*:
>    `K_kill = min(12.0, T_A1 × σ_sel × 10,000 / 20)`, where `σ_sel` is the selector's
>    realised annualised sd on the test sessions (Inputs), and
>    `ΔT = annual_turnover(selector held weights on the test sessions) −
>    annual_turnover(control held weights on the test sessions)`.
>    - **The kill fires if `ΔT > K_kill`.**
>
> - `p_boot = 2(1 − Φ(|Δ_A1| / SE*))`.
> - `p_HAC = 2(1 − Φ(|t|))`, with `t = protocol.paired_hac_t(selector, control, lags=6)`
>   on the daily net differences (Inputs). *(lock)* If the sign of *t* differs from the
>   sign of `Δ_A1`, `p_HAC := 1`: the HAC test is on the mean difference of two arms that
>   run at different realised sd, and a *t* of the wrong sign is no evidence for Δ.
> - **`p_A1 = max(p_boot, p_HAC)`.** The two coincide only at equal realised volatility,
>   which the cap breaks, and requiring both removes the choice between them. If
>   `Δ_A1 ≤ 0`, `p_A1 := 1` in Holm (§13.3).
>
> **Verdict — the first line that applies:**
> 1. **UNDECIDABLE**: any instrument or reading value is non-finite, including any one of
>    the 1,000 placebo draws of Δ or β; a paired leg is missing on a test session; or the
>    selector is at the fallback on more than half of the test sessions.
> 2. **FAIL (cost)**: the kill fired. The reading is still made and reported; it cannot
>    pass.
> 3. **FAIL**: `Δ_A1 ≤ 0` at 5 bp.
> 4. **UNDECIDED**: `Δ_A1 > 0` at 5 bp and `≤ 0` at 10 bp.
> 5. **UNDERPOWERED**: `0 < Δ_A1 < T_A1`, or `Δ_A1 ≥ T_A1` with `p_A1` not rejected by
>    Holm (§13.3). Never a PASS.
> 6. **DOWNGRADED (beta)**: the beta percentile is ≥ 0.95.
> 7. **DOWNGRADED (not conditional)**: the difference percentile among the placebo arms is
>    < 0.95. The gain is then no larger than what a content-free partition with the same
>    clock produces.
> 8. **DOMINATED (volatility)**: `Δ_W ≥ Δ_A1`. The one-line volatility partition does at
>    least as well, so the gain cannot be credited to the context.
> 9. **DOWNGRADED (PIT)** *(lock, second validator)*: the same lock rebuilt on the
>    18-feature partition without the NFCI (§8, control 5) does not meet this lock's
>    criterion (lines 1-8 applied to the rebuild give anything but PASS).
> 10. **PASS.**

**A-2, §12.7:**

> - **Statistic:** `R = plain arithmetic mean of ρ_fk` over the qualifying cells.
>
> - **p-value:** `p_A2 = protocol.placebo_p_value(R, R^(·)) = (1 + #{j : R^(j) ≥ R}) /
>   1,001`. Ties count against the real partition.
> - **The lock holds** if `p_A2 ≤ 0.01` (at most 9 of 1,000 draws at or above R) **and**
>   Holm rejects (§13.3). If A-2 ranks first in Holm, its bar is `p ≤ 0.00833`, at most 7
>   draws.
>
> - **Verdict:**
>   - **UNDECIDABLE**: `R` or any `R^(j)` is not finite, the placebo check is not exact, or
>     fewer than 10 cells qualify (half of the 5K cells at a K sensitivity);
>   - **PASS**: the lock holds and `R_W < R`, and the lock rebuilt on the 18-feature
>     partition (§8, control 5) holds with its own witness;
>   - **DOWNGRADED (PIT)**: the lock holds, `R_W < R`, and the lock rebuilt on the
>     18-feature partition does not hold with its own witness;
>   - **DOMINATED**: the lock holds and `R_W ≥ R`;
>   - **NOT SHOWN**, otherwise. No power was measured for A-2 against any declared
>     alternative, so under this lock A-2 **never reads FAIL**. A FAIL would need, before
>     the reading, a power of at least 0.80 measured blind against an effect planted
>     into demeaned legs. That needs a blindness ruling this lock does not give. Any such
>     instrument would be an amendment in `docs/PROTOCOL_FREEZE.md`, committed before
>     any reading.

**level verdict, §13.2:**

> **The cost column that decides is 5 bp.** 10 bp can only turn a positive reading into
> UNDECIDED, and 20 bp enters only through the kill rule. A level's verdict is PASS only
> if its locks pass: both at A and at B, and C-1 at C. Otherwise the level takes the
> first status among its locks in this order: FAIL (FAIL (cost) included), then
> UNDECIDABLE, then UNDECIDED, then UNDERPOWERED, then NOT SHOWN, then DOWNGRADED, then
> DOMINATED. Following ARBITRAGE §5(b), no
> PASS in one channel is traded for a FAIL in another: a turnover saving, a QLIKE gain
> and a Sharpe difference are not commensurable.

**Holm, §13.3:**

> Order the six p-values `p_(1) ≤ … ≤ p_(6)`. Find the largest *j* such that
> `p_(i) ≤ 0.05 / (6 − i + 1)` for every `i ≤ j`. The primaries ranked 1 to *j* are
> rejected. Rejection is **necessary, never sufficient**: each lock must also meet its own
> criterion in §12. The final Holm step runs after C is read. Each level's results file
> states its provisional verdicts at the Bonferroni level, 0.05/6, which implies Holm
> rejection.

**UNDECIDABLE, §13.4:**

> - **At every level:**
>   - a non-finite instrument reading, or a non-finite real statistic;
>   - *(lock, from an audit)* **any one** of the 1,000 draws of any placebo statistic a
>     lock uses (`R`, `G`, `S_C`, `Δ^(j)`, `β^(j)`) is not finite. No draw is dropped,
>     replaced or redrawn. `protocol.placebo_p_value` and
>     `protocol.placebo_percentile` return NaN in that case (commit `3b64ab2`);
>   - a paired leg missing on any test session;
>   - a placebo check that is not exact.
> - **At A and B, in addition:**
>   - fewer than 10 qualifying cells (half of the 5K cells at a K sensitivity). At A the
>     cells counted are the (fold, state) pairs that enter A-2 (§12.4); at B they are the
>     qualifying training cells;
>   - more than half of the test sessions at the fallback (§12.6 at A, §12.9 at B);
>   - at B, a pooled training matrix that is not positive definite.
<!-- INSTRUMENT:END -->

<!-- READING:BEGIN -->
## Reading (§13.1 step 4)

Not read. The reading of this level runs only after the instruments of A, B and C and their thresholds are committed, in the order A, B, C (§13.1).
<!-- READING:END -->

<!-- SENSITIVITIES:BEGIN -->
## Sensitivities (§12.13)

Not read. They are read after the six primaries and the Holm step.
<!-- SENSITIVITIES:END -->
