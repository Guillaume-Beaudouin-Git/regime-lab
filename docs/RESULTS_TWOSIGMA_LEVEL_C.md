# Two Sigma level C — the cost channel (§6, §12.10)

`docs/PRESPEC_TWOSIGMA.md`, LOCKED at `30f7d69`. Written by `scripts/run_twosigma_tree.py`; the numbers below come from its output and from `docs/artifacts/twosigma/`, never typed by hand.

<!-- INSTRUMENT:BEGIN -->
## Instrument (§13.1 steps 1-3)

**Nothing of the tree was read.** The instrument printed no mean, Sharpe, IC or per-state statistic of the real partition or of any real arm, and no Sharpe difference or beta of any placebo arm (§13.1 step 2). It computed none of the reading's statistics: no Δ_A1 or Δ_B2, no p-value, no `R`, `G` or `S_C`, no interval `h_fk`, no gate. It computed `m` and the state second-moment matrices, which the arms' demeaned legs need, and printed neither. What it printed is what §13.1 step 2 allows: counts, the turnover of the arms it builds with the kill line, realised sd and cap share, the MDEs of demeaned legs with T_A1, T_B2, MDE_C and S*, the null percentiles of A-2, B-1 and C-1 (real returns under content-free partitions, never the real labels), and the NFCI diagnostic (labels only).

Computed on 2026-09-23 at code commit `42011e9`. Every threshold is stored bitwise in `docs/artifacts/twosigma/thresholds.json`, sections `C`, `C:K=3`, `C:K=5`, `C:K=6`, `C:smooth21`; the reading refuses to run unless that file is committed, unmodified, and equal bitwise to its own recomputation, on inputs with the same SHA-256 and packages at the versions `uv.lock` pins (§13.1 step 4). The tables round what the file stores.

Packages pinned by `uv.lock`: numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, statsmodels 0.15.0, threadpoolctl 3.7.0.

| input file | SHA-256 |
|---|---|
| `data/raw/panels/industry_49.parquet` | `505e212b65c4e5df5cafe1a7df5c954bb8dfaf06d27a73e8645ba1090efc436a` |
| `data/raw/panels/factors_5.parquet` | `690046e548a434c2218769a8cb3c0701ab6075fae1c5cc29bc1a6f077e925159` |
| `data/cache/features.parquet` | `f531881f1da80d03f12ce4af961ddb8665f013af2e75312ab4f52f3722011a08` |
| `data/raw/prices/cross_asset.parquet` | `570bb4c9f31732e0a063e2e8198720872256e0d6c6fca3924a77e47c96b4da4c` |

### What was measured

- The target `w*`, the control's held weights after the multiplier, and the state-blind cadence books of the menu H = {1, 2, 3, 5, 8, 13, 21}, among them the weekly twin (h = 5) (§12.10). They read no label.
- C-1's null: on each of the 1,000 uniform placebo draws (seed 0), the whole rule (λ, the intervals, the training check) re-run on the draw's training labels and the arm run on its test labels, giving `S^(j)` (§12.10). **The conditional arm was not built on the real partition**: no `h_fk`, no `S_C`, no gate, no turnover or δ of it (§12.10 "Instrument").
- The cells of each partition, from its labels only (§12.4).

### Counts, the twin, sd and cap share (§12.10)

Held turnover ×/yr and mean δ on the test sessions; σ_twin is the realised sd of the twin's 5 bp net daily excess returns there.

| section | K | training cells (per fold) | abstaining test sessions | placebo exact | twin ×/yr | twin mean δ | σ_twin | twin missing | control cap share |
|---|---|---|---|---|---|---|---|---|---|
| `C` | 4 | 19 (3/4/4/4/4) | 372 | yes | 17.95 | 0.1890 | 7.84% | 0 | 70.5% |
| `C:K=3` | 3 | 15 (3/3/3/3/3) | 0 | yes | 17.95 | 0.1890 | 7.84% | 0 | 70.5% |
| `C:K=5` | 5 | 24 (5/5/5/4/5) | 34 | yes | 17.95 | 0.1890 | 7.84% | 0 | 70.5% |
| `C:K=6` | 6 | 28 (6/6/5/5/6) | 30 | yes | 17.95 | 0.1890 | 7.84% | 0 | 70.5% |
| `C:smooth21` | 4 | 19 (3/4/4/4/4) | 372 | yes | 17.95 | 0.1890 | 7.84% | 0 | 70.5% |

### The state-blind menu (§12.10)

Held turnover ×/yr on the test sessions of each state-blind book.

| section | h = 1 | h = 2 | h = 3 | h = 5 | h = 8 | h = 13 | h = 21 |
|---|---|---|---|---|---|---|---|
| `C` | 36.66 | 27.04 | 22.57 | 17.95 | 14.49 | 11.47 | 9.24 |
| `C:K=3` | 36.66 | 27.04 | 22.57 | 17.95 | 14.49 | 11.47 | 9.24 |
| `C:K=5` | 36.66 | 27.04 | 22.57 | 17.95 | 14.49 | 11.47 | 9.24 |
| `C:K=6` | 36.66 | 27.04 | 22.57 | 17.95 | 14.49 | 11.47 | 9.24 |
| `C:smooth21` | 36.66 | 27.04 | 22.57 | 17.95 | 14.49 | 11.47 | 9.24 |

### C-1: the null, MDE_C and S* (§12.10 "Resolution and power")

`MDE_C = q99 − q20`; `S* = 0.05 × σ_twin × 10,000 / 5`. **Amendment of 2026-09-23** (`docs/PROTOCOL_FREEZE.md`, before any reading): the power claim holds ("powered") only if `0 < MDE_C ≤ S*` **and** no single value takes 20% or more of the draws ("atom share"); a C-1 that does not hold then reads FAIL, otherwise NOT SHOWN. An atom makes `q99 − q20` measure no shift. C-1 holds only if `p ≤ 0.01`, Holm rejects, the gate holds **and** `S_C ≥ S*`: with an atom null the placebo bar alone would let any positive saving through. The draws column counts the finite `S^(j)` below, at and above zero.

| section | q20 | q50 | q95 | q99 | MDE_C | S* | powered | draws < 0 / = 0 / > 0 | atom share | non-finite draws |
|---|---|---|---|---|---|---|---|---|---|---|
| `C` | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** | **7.843** | no | 0 / 1,000 / 0 | 1.000 | 0 |
| `C:K=3` | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** | **7.843** | no | 0 / 1,000 / 0 | 1.000 | 0 |
| `C:K=5` | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** | **7.843** | no | 0 / 1,000 / 0 | 1.000 | 0 |
| `C:K=6` | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** | **7.843** | no | 0 / 1,000 / 0 | 1.000 | 0 |
| `C:smooth21` | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** | **7.843** | no | 0 / 1,000 / 0 | 1.000 | 0 |

### UNDECIDABLE before the reading (§13.4)

None: no lock of this level is UNDECIDABLE before its reading, at the primary or at any sensitivity.

### The criteria, copied from the lock

Verbatim from `docs/PRESPEC_TWOSIGMA.md` (LOCKED, `30f7d69`). They were fixed before any statistic existed; this file only restates them beside the thresholds.

**C-1, §12.10:**

> - **Statistic:** `S_C = annual_turnover(twin held weights on the test sessions) −
>   annual_turnover(conditional held weights on the test sessions)`, each restricted to
>   the test sessions **first**, in ×/yr.
> - **Gate:** the conditional arm's mean `δ_t` over the test sessions must not exceed the
>   twin's.
> - **Null:** `S_C` on the 1,000 placebo draws. On each draw the whole rule (λ, the
>   intervals, the training check) is re-run on the draw's training labels, and the arm
>   is run on the draw's test labels. `p_C1 = protocol.placebo_p_value(S_C, S^(·))`. The
>   re-optimisation on every draw absorbs the rule's in-sample optimisation bias.
> - **C-1 holds** if `p_C1 ≤ 0.01`, Holm rejects, and the gate holds.
> - **Resolution and power, before the reading** *(lock, from an audit)*. From the null,
>   the instrument prints q50, q95, q99 and **`MDE_C = q99 − q20`**: the shift of the null
>   that puts 80% of it above its 99th percentile, under a location shift. The saving
>   worth 0.05 Sharpe at 5 bp is **`S* = 0.05 × σ_twin × 10,000 / 5`**, where σ_twin is
>   the twin's realised annualised sd of its 5 bp net daily excess returns on the test
>   sessions (about 7.9×/yr at the control's 7.88%). **If `MDE_C ≤ S*`**, the instrument
>   has a power of at least 0.80 against a saving of S\*, and a C-1 that does not hold
>   reads FAIL; otherwise it reads NOT SHOWN.
>
> - **Verdict:**
>   - **UNDECIDABLE**: `S_C` or any `S^(j)` is not finite, the placebo check is not
>     exact, or a leg is missing on a test session;
>   - **FAIL**: the gate fails; or the lock does not hold and `MDE_C ≤ S*`;
>   - **NOT SHOWN**: the lock does not hold and `MDE_C > S*`;
>   - **DOMINATED**: the lock holds and `S_W ≥ S_C`;
>   - **PASS**: the lock holds and `S_W < S_C`, and the lock rebuilt on the 18-feature
>     partition (§8, control 5) holds with its own witness and gate;
>   - **DOWNGRADED (PIT)**: the lock holds, `S_W < S_C`, and the lock rebuilt on the
>     18-feature partition does not hold with its own witness and gate.

**C-2, §12.10:**

> - **C-2:** the bound `S_C × bps / 10,000 / σ_twin` at 5, 10 and 20 bp. It is reported
>   beside the control's whole cost, never a PASS, and **`p_C2 := 1`** in Holm.
> - **Level C's verdict is C-1's.**

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
