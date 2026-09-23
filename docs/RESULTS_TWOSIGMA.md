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

| level | final verdict | with the §12.13 label |
|---|---|---|
| A | FAIL | **FAIL** |
| B | UNDERPOWERED | **UNDERPOWERED** |
| C | NOT SHOWN | **NOT SHOWN** |

14 sensitivity rows logged; details in each level's results file.
<!-- SENSITIVITIES:END -->

---

## Closure of the tree (§6, §13.6) — written after the sensitivities

No level passes, so the closure statement of §6 is written, with each level in the
wording §13.6 gives its status and with the applied numbers:

> A context partition refitted walk-forward on twenty macro-financial and
> cross-sectional features, orthogonalised on log realised volatility (pooled
> out-of-sample η² 0.14 against it) and switching about 12.5 times a year out of sample,
> **does not transfer** out of sample as a selector over a ten-signal US industry
> library on the first moment: the state-tilted book earns 0.110 Sharpe *less* than the
> equal-weight blend it tilts (−0.370 against −0.260, net of 5 bp, in excess), and the
> training-fold state profiles do not carry to the test fold (R 0.047, p 0.34). On the
> second moment it **is not shown to transfer**: risk parity on its state-conditional
> second moments gains +0.064 over the pooled matrix, against a decision bar of 0.338
> after correction for six primary tests (the fitted pair itself resolves 0.15 at
> α 0.05/6, so the bar is the lock's floor), and its state-conditional matrices
> forecast the ten legs *worse* than a single pooled one (QLIKE gain −3.36, 8.5th percentile of the matched placebo). Nor **is it
> shown** to let the equal-weight book trade less than a state-blind weekly cadence at
> the same tracking distance: the pooled-budget rule keeps the weekly interval in every
> state of every fold, on the real partition as on all 1,000 content-free ones
> (S_C = 0; power unmeasured). All over 20.0 years of paired out-of-sample sessions,
> August 2006 to July 2026. The book's entire cost at 5 bp, 0.233 Sharpe, is below the
> 0.338 bar, so no Sharpe gain from cost alone is decidable on this sample.

**What this is not.** It is not "regimes do not monetise". It is one partition (K-means
on a volatility-orthogonalised context), one use (selection among signals), one library
(ten US industry signals), 20 years. It says nothing of the classifier's established
content, which is variance (`AVANCEMENT.md` §0).

**Robustness (§12.13, never deciding).** The 14 sensitivity rows agree with the
primaries. Level A reads FAIL at K = 3 (Δ −0.147), K = 5 (−0.094), the 21-session
smoothing (−0.128) and d = 0.25 (−0.049), and FAIL (cost) at d = 1.00 (−0.212; its kill
was known to fire before the reading). **One sign reversal, written here as §12.13
requires:** at K = 6, Δ_A1 = **+0.009**, positive at 5 bp and not at 10 bp, so
UNDECIDED. It is 3% of the bar. Level B is UNDERPOWERED at every variant (Δ_B2 +0.030
to +0.080). Level C reads S_C = 0 at every variant.

**What the volatility witnesses show (diagnostics, no trial, deciding nothing).** On
every channel, the one-line volatility partition does at least as well as the context
orthogonal to it: Δ_W +0.201 at A-1, R_W 0.283 at A-2, both witnesses beat the context
on the B-1 loss, Δ_W,B2 +0.145 at B-2. This agrees with what the programme had already
measured: the regime classifier's content is volatility. The witnesses were not tested
against a placebo here, so these figures are an indication and not a finding.

**The book, for scale.** The equal-weight control of the ten industry signals nets a
Sharpe of −0.26 at 5 bp over the test years. The weekly twin, which is state-blind, nets
−0.12. The logged `sharpe` values are those of near-zero-net long-short industry books
with no borrow cost. None measures progress toward the programme's ambition of a net
Sharpe of 1 to 2 (§13.5).

**The record.**

| step | commit |
|---|---|
| lock | `30f7d69` |
| build | `22abbe0`, `3b64ab2`, `540a778`, `9905eb7` |
| C-1 amendment, before any reading: the atom null, S_C ≥ S*, C-2 BOUND, the guards | `42011e9` |
| instruments and thresholds committed before any reading | `d29de38` |
| reading A: computed at `d29de38`, rows logged at `2d1315f` after the register incident, nothing recomputed | `27d2cae` |
| readings B and C | `1fc44fe`, `90938fe` |
| Holm, then the sensitivities | `0ba0775`, `74dae25` |

The register holds 20 `twosigma` rows, the 20 declared evaluations of §7, out of 890
rows and 168 distinct configurations in total.

**What stays learned whatever the verdict (§10).**
- The library's dimension: 7.97 effective dimensions of 10 on the industry panel,
  against 3.86 of 11 on the 46-instrument universe, which cannot host a selection
  study.
- An exact matched placebo for walk-forward partitions, construction 4. It is uniform
  over join-free orders, and it is the construction that works where the swap repair
  completes 4 draws in 1,000.
- The cost geometry of a selection overlay on this book: breakeven 27 to 33 bp, and the
  book's whole cost 0.23 Sharpe at 5 bp.
- A cadence rule that finds nothing to save on this book, on real and content-free
  partitions alike.
