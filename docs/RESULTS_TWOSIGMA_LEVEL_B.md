# Two Sigma level B — the second-moment channel (§6, §12.8-§12.9)

`docs/PRESPEC_TWOSIGMA.md`, LOCKED at `30f7d69`. Written by `scripts/run_twosigma_tree.py`; the numbers below come from its output and from `docs/artifacts/twosigma/`, never typed by hand.

<!-- INSTRUMENT:BEGIN -->
## Instrument (§13.1 steps 1-3)

**Nothing of the tree was read.** The instrument printed no mean, Sharpe, IC or per-state statistic of the real partition or of any real arm, and no Sharpe difference or beta of any placebo arm (§13.1 step 2). It computed none of the reading's statistics: no Δ_A1 or Δ_B2, no p-value, no `R`, `G` or `S_C`, no interval `h_fk`, no gate. It computed `m` and the state second-moment matrices, which the arms' demeaned legs need, and printed neither. What it printed is what §13.1 step 2 allows: counts, the turnover of the arms it builds with the kill line, realised sd and cap share, the MDEs of demeaned legs with T_A1, T_B2, MDE_C and S*, the null percentiles of A-2, B-1 and C-1 (real returns under content-free partitions, never the real labels), and the NFCI diagnostic (labels only).

Computed on 2026-09-23 at code commit `42011e9`. Every threshold is stored bitwise in `docs/artifacts/twosigma/thresholds.json`, sections `B`, `B:K=3`, `B:K=5`, `B:K=6`, `B:smooth21`; the reading refuses to run unless that file is committed, unmodified, and equal bitwise to its own recomputation, on inputs with the same SHA-256 and packages at the versions `uv.lock` pins (§13.1 step 4). The tables round what the file stores.

Packages pinned by `uv.lock`: numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, statsmodels 0.15.0, threadpoolctl 3.7.0.

| input file | SHA-256 |
|---|---|
| `data/raw/panels/industry_49.parquet` | `505e212b65c4e5df5cafe1a7df5c954bb8dfaf06d27a73e8645ba1090efc436a` |
| `data/raw/panels/factors_5.parquet` | `690046e548a434c2218769a8cb3c0701ab6075fae1c5cc29bc1a6f077e925159` |
| `data/cache/features.parquet` | `f531881f1da80d03f12ce4af961ddb8665f013af2e75312ab4f52f3722011a08` |
| `data/raw/prices/cross_asset.parquet` | `570bb4c9f31732e0a063e2e8198720872256e0d6c6fca3924a77e47c96b4da4c` |

### What was measured

- The partition of each variant (§12.2), its cells (§12.4) and its 1,000 uniform placebo draws, seed 0 (§12.11).
- The second-moment matrices about zero per fold and qualifying cell, on the training folds (§12.8), with the positive-definite check; the pooled and state-conditional risk-parity arms on the traded path, gross-matched, targeted and netted of 5 bp (§12.9); the blinded bootstrap of their **demeaned** 5 bp legs on the 5,031 test sessions (2,000 draws, seed 0, blocks 21/63/126) giving the MDEs and T_B2; their turnover, the kill line, realised sd and cap share.
- B-1's null: `G^(j)` on each placebo draw, every matrix re-estimated on the draw's training labels (§12.8). The real `G`, the witnesses and the placebo arms were not computed.
- The return-space share of blend variance through the correlations, from fold 5's pooled training matrix (§6 Level B's formula, §12.8 "Instrument").

### Counts (§12.4, §12.8, §12.9, §13.4)

| section | K | qualifying training cells (per fold) | floor | pooled matrices PD (per fold) | cells not PD | fallback test sessions | abstaining test sessions | placebo exact |
|---|---|---|---|---|---|---|---|---|
| `B` | 4 | 19 of 20 (3/4/4/4/4) | 10.0 | [yes; yes; yes; yes; yes] | 0 | 372 (7.4%) | 372 | yes |
| `B:K=3` | 3 | 15 of 15 (3/3/3/3/3) | 7.5 | [yes; yes; yes; yes; yes] | 0 | 0 (0.0%) | 0 | yes |
| `B:K=5` | 5 | 24 of 25 (5/5/5/4/5) | 12.5 | [yes; yes; yes; yes; yes] | 0 | 34 (0.7%) | 34 | yes |
| `B:K=6` | 6 | 28 of 30 (6/6/5/5/6) | 15.0 | [yes; yes; yes; yes; yes] | 0 | 30 (0.6%) | 30 | yes |
| `B:smooth21` | 4 | 19 of 20 (3/4/4/4/4) | 10.0 | [yes; yes; yes; yes; yes] | 0 | 372 (7.4%) | 372 | yes |

### B-2: resolution and threshold (§12.9)

`T_B2 = max(0.338, max_b MDE_B2(0.05/6, b))` on the pair's demeaned 5 bp legs.

| section | SE 21 | SE 63 | SE 126 | MDE(0.05/6) 21 | MDE(0.05/6) 63 | MDE(0.05/6) 126 | T_B2 | block* |
|---|---|---|---|---|---|---|---|---|
| `B` | 0.0427 | 0.0425 | 0.0393 | 0.1484 | 0.1479 | 0.1369 | **0.3380** | 21 |
| `B:K=3` | 0.0401 | 0.0390 | 0.0388 | 0.1397 | 0.1356 | 0.1350 | **0.3380** | 21 |
| `B:K=5` | 0.0483 | 0.0438 | 0.0410 | 0.1680 | 0.1525 | 0.1428 | **0.3380** | 21 |
| `B:K=6` | 0.0449 | 0.0445 | 0.0425 | 0.1562 | 0.1547 | 0.1479 | **0.3380** | 21 |
| `B:smooth21` | 0.0413 | 0.0373 | 0.0340 | 0.1436 | 0.1297 | 0.1183 | **0.3380** | 21 |

### B-2: turnover and the kill line, sd and cap share (§12.9)

Held turnover ×/yr on the test sessions, restricted first; `K_kill = min(12, T_B2 × σ_state × 10,000 / 20)`; the kill fires if ΔT > K_kill.

| section | state arm ×/yr | pooled arm ×/yr | ΔT | K_kill | kill | σ state | σ pooled | cap share state | cap share pooled | missing test sessions |
|---|---|---|---|---|---|---|---|---|---|---|
| `B` | 51.44 | 49.81 | 1.63 | 12.00 | no | 7.21% | 7.15% | 81.8% | 82.3% | 0 / 0 |
| `B:K=3` | 50.03 | 49.81 | 0.22 | 12.00 | no | 7.18% | 7.15% | 81.1% | 82.3% | 0 / 0 |
| `B:K=5` | 51.50 | 49.81 | 1.69 | 12.00 | no | 7.25% | 7.15% | 80.9% | 82.3% | 0 / 0 |
| `B:K=6` | 51.52 | 49.81 | 1.71 | 12.00 | no | 7.24% | 7.15% | 81.0% | 82.3% | 0 / 0 |
| `B:smooth21` | 49.53 | 49.82 | -0.29 | 12.00 | no | 7.19% | 7.15% | 82.0% | 82.3% | 0 / 0 |

### B-1: the null and the correlation share (§12.8)

Percentiles of `G^(·)` over the 1,000 draws; `q99 − q50` is the resolution quoted in a NOT SHOWN (§13.6). The correlation share is the return-space counterpart of §6's 23.9% in position space.

| section | q50 | q95 | q99 | q99 − q50 | non-finite draws | correlation share (fold 5) |
|---|---|---|---|---|---|---|
| `B` | -1.9624 | -1.0087 | -0.7260 | 1.2363 | 0 | 47.4% |
| `B:K=3` | -2.4327 | -0.8176 | -0.2560 | 2.1766 | 0 | 47.4% |
| `B:K=5` | -3.8256 | -1.6586 | -1.0251 | 2.8005 | 0 | 47.4% |
| `B:K=6` | -5.3114 | -2.1546 | -1.4067 | 3.9047 | 0 | 47.4% |
| `B:smooth21` | -2.1881 | -1.2340 | -0.8407 | 1.3473 | 0 | 47.4% |

### UNDECIDABLE before the reading (§13.4)

None: no lock of this level is UNDECIDABLE before its reading, at the primary or at any sensitivity.

### The criteria, copied from the lock

Verbatim from `docs/PRESPEC_TWOSIGMA.md` (LOCKED, `30f7d69`). They were fixed before any statistic existed; this file only restates them beside the thresholds.

**B-1, §12.8:**

> - **Statistic:** `G = mean over the 5,031 test sessions of [ℓ_t(Σ_f) − ℓ_t(Σ_f,k(t))]`,
>   where *k(t)* is the lagged state along fold *f*'s path. G is positive when the
>   state-conditional forecast is better.
> - **Null and p-value:** G on the same 1,000 placebo draws, with every matrix
>   re-estimated on the draw's training labels. `p_B1 = protocol.placebo_p_value(G,
>   G^(·)) = (1 + #{G^(j) ≥ G}) / 1,001`. The lock holds if `p_B1 ≤ 0.01` and Holm
>   rejects.
>
>   - **A B-1 that holds is reported DOMINATED unless `p_W1 ≤ 0.05` and `p_W2 ≤ 0.05`.**
>     **HARDER.**
>
> - **Verdict:**
>   - **UNDECIDABLE**: `G` or any `G^(j)` is not finite, the placebo check is not exact,
>     or a pooled training matrix is not positive definite (§13.4);
>   - **PASS**: the lock holds and neither witness dominates, and the lock rebuilt on the
>     18-feature partition (§8, control 5) holds with its own witnesses;
>   - **DOWNGRADED (PIT)**: the lock holds, neither witness dominates, and the lock rebuilt
>     on the 18-feature partition does not hold with its own witnesses;
>   - **DOMINATED**: the lock holds and a witness dominates;
>   - **NOT SHOWN**, otherwise. As at A-2, no power was measured for B-1, so under this
>     lock B-1 **never reads FAIL** (§12.7).

**B-2, §12.9:**

> - **Statistic:** `Δ_B2 = SR(state) − SR(pooled)` over the pooled test sessions.
> - **Threshold:** `T_B2 = max(0.338, max_b MDE_B2(0.05/6, b))` on the pair's demeaned
>   legs, measured by the instrument; SE\* at the block that gives the largest
>   MDE_B2(0.05/6), as in §12.6.
> - **p-value:** `p_B2 = max(p_boot, p_HAC)` as in §12.6, with `p_HAC := 1` when the sign
>   of *t* differs from that of `Δ_B2`, and `p_B2 := 1` in Holm when `Δ_B2 ≤ 0`.
> - **Kill rule** *(lock, from an audit: the comparator is B-2's own pair)*:
>   - `ΔT = protocol.annual_turnover(state arm held weights restricted to the 5,031 test
>     sessions) − protocol.annual_turnover(pooled arm held weights restricted to the same
>     sessions)`. It is **not** measured against the equal-weight control: pooled risk
>     parity overweights the slow legs and turns over far less than the control.
>   - `K_kill = min(12.0, T_B2 × σ_state × 10,000 / 20)`, where σ_state is √252 × sd
>     (ddof 1) of the state arm's 5 bp net daily excess returns on the test sessions.
>     Anchored to the decision bar, as at A-1 (§12.6), and not to the pair's own
>     uncorrected MDE. The state and pooled legs are nearly identical, so that MDE is
>     small, and a kill scaled to it would fire almost automatically.
>   - The kill fires if `ΔT > K_kill`.
> - **Fallback at B-2:** a test session on which the state arm holds
>   `risk_parity_mix(Σ_f)` because its lagged state is missing, its lagged state's
>   training cell does not qualify, or its `Σ_fk` failed the positive-definite check.
>   The >50% UNDECIDABLE line of §12.6 applies to that count.
>
> - **Verdict lines:** those of §12.6, in the same order, with `Δ_B2`, `T_B2`, `p_B2`,
>   this kill and `Δ_W,B2` in place of A-1's.

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

| lock | provisional verdict (Bonferroni 0.05/6) | p entering Holm |
|---|---|---|
| B-1 | **NOT SHOWN** | 0.9151 |
| B-2 | **UNDERPOWERED** | 0.1441 |

Level B, provisional: **UNDERPOWERED**. The final verdict follows the Holm step, after C (`docs/RESULTS_TWOSIGMA.md`).

### B-1 (§12.8)

- G -3.3603 (diagonal diagnostic G_diag -2.3740); **p_B1 0.9151**, percentile 0.0850, null q99 -0.7260.
- Witness W1: mean d -3.7105, p_W1 1.0000; witness W2: mean d -4.4261, p_W2 1.0000.

### B-2 (§12.9)

|  | 5 bp | 10 bp | 20 bp |
|---|---|---|---|
| Δ_B2 | 0.0641 | 0.0553 | 0.0373 |

- SR state -0.4918, SR pooled -0.5560 (5 bp); T_B2 0.3380; HAC-6 t 1.4607; p_boot 0.1326, p_HAC 0.1441, **p_B2 0.1441**.
- β state -0.1216; beta percentile 0.8140, difference percentile 0.8040; witness W1: Δ_W,B2 0.1450.
- Kill: ΔT 1.63 against K_kill 12.00, fires no.

§8 control 5 was not due: no lock would otherwise PASS, at the provisional bar or under a Holm rejection.

### Trial rows (§13.5)

| test | sharpe | delta | threshold | p | placebo_pct | verdict | note |
|---|---|---|---|---|---|---|---|
| B-1 | n/a | -3.3603 | -0.7260 | 0.9151 | 0.0850 | NOT SHOWN | delta = G, threshold = the null's q99, p = p_B1 (§12.8) |
| B-2 | -0.4918 | 0.0641 | 0.3380 | 0.1441 | 0.8040 | UNDERPOWERED |  |

The logged `sharpe` is that of a near-zero-net long-short book of industry portfolios with no borrow cost. It is not a measure of progress toward the programme's ambition of a net Sharpe of 1 to 2 (§13.5).
<!-- READING:END -->

<!-- SENSITIVITIES:BEGIN -->
## Sensitivities (§12.13)

Not read. They are read after the six primaries and the Holm step.
<!-- SENSITIVITIES:END -->
