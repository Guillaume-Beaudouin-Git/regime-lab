# Two Sigma tree — the Holm step and the level verdicts

`docs/PRESPEC_TWOSIGMA.md`, LOCKED at `30f7d69`. Written by `scripts/run_twosigma_tree.py holm` from the committed reading artifacts in `docs/artifacts/twosigma/`.

## Holm–Bonferroni over the six primaries (§13.3)

| rank | primary | p entering Holm | bar 0.05/(6 − i + 1) | Holm |
|---|---|---|---|---|
| 1 | B-2 | 0.1441 | 0.00833 | not rejected |
| 2 | A-2 | 0.3397 | 0.01000 | not rejected |
| 3 | B-1 | 0.9151 | 0.01250 | not rejected |
| 4 | A-1 | 1.0000 | 0.01667 | not rejected |
| 5 | C-1 | 1.0000 | 0.02500 | not rejected |
| 6 | C-2 | 1.0000 | 0.05000 | not rejected |

A lock UNDECIDABLE, an A-1 or B-2 with Δ ≤ 0, a missing p and C-2 enter with p := 1. Rejection is necessary, never sufficient (§13.3).

## Lock verdicts

| lock | provisional (0.05/6) | final, after Holm |
|---|---|---|
| A-1 | FAIL | **FAIL** |
| A-2 | NOT SHOWN | **NOT SHOWN** |
| B-1 | NOT SHOWN | **NOT SHOWN** |
| B-2 | UNDERPOWERED | **UNDERPOWERED** |
| C-1 | NOT SHOWN | **NOT SHOWN** |
| C-2 | BOUND | **BOUND** |

## Level verdicts (§13.2) and their wording (§13.6)

| level | verdict | §13.6 wording |
|---|---|---|
| A | **FAIL** | does not transfer |
| B | **UNDERPOWERED** | is not shown to transfer |
| C | **NOT SHOWN** | is not shown to transfer |

Resolutions for the wording of §13.6: T_A1 0.3380, T_B2 0.3380, MDE_C 0.000; the nulls' q99 − q50 are in each level's results file.

### The criteria, copied from the lock

Verbatim from `docs/PRESPEC_TWOSIGMA.md` (LOCKED, `30f7d69`). They were fixed before any statistic existed; this file only restates them beside the thresholds.

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

<!-- SENSITIVITIES:BEGIN -->
## Sensitivities (§12.13)

Not read yet.
<!-- SENSITIVITIES:END -->
