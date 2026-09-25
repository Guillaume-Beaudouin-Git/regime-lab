# Replicating Shu, Yu and Mulvey (2024) — S&P 500

Every earlier hypothesis in this programme was built rather than replicated, and
each null was therefore ambiguous. This one follows a published protocol to the
letter and stops before improving on it.

Protocol as published: three features (downside deviation at a ten-day
half-life, Sortino ratios at twenty and sixty), a three-thousand-day estimation
window, centroids refitted every six months, online inference in between, the
jump penalty selected monthly by maximising Sharpe over the preceding eight
years, one-day execution delay, ten basis points one way. Out of sample
**2 January 1990 to 29 December 2023**.

## The benchmark reproduces exactly, which validates the harness

| | CAGR | volatility | Sharpe | worst drawdown |
| --- | --- | --- | --- | --- |
| buy and hold, ours | **10.2%** | **18.2%** | **0.48** | −55.3% |
| buy and hold, published | **10.2%** | **18.2%** | **0.48** | −55.2% |

Four figures out of four. That settles three things at once — the Sharpe
convention is theirs, the drawdown definition is theirs, the sample window is
theirs — before any judgement is passed on the strategy.

**It also found the one data error.** The first attempt used `^GSPC`, a price
index, and returned a buy-and-hold Sharpe of 0.37 with a CAGR of 8.0%. The paper
uses total return. The missing 2.25% a year is dividends.

The total-return index only starts in 1988, too late to leave twelve years of
training before a 1990 start, so the series was rebuilt as price return plus a
daily accrual of Shiller's monthly dividend yield, and **checked against the real
index** over the 9,743 days where both exist: daily correlation **0.999626**, and
10.16% against 10.21% annualised over 1990-2023. Accurate to about five basis
points a year.

## The model's risk reproduces; its return does not

| | CAGR | volatility | Sharpe | worst drawdown | exposure | turnover |
| --- | --- | --- | --- | --- | --- | --- |
| jump model, ours | **8.6%** | **13.0%** | **0.50** | −32.0% | 76.6% | 112% |
| jump model, published | **11.2%** | **13.1%** | **0.68** | −26.6% | 80.0% | 44% |

Volatility lands on 13.0% against 13.1%. Exposure lands on 77% against 80%. The
model de-risks by the same amount, at the same frequency, in the same places.

**The return does not follow.** Ours is 8.6% where theirs is 11.2%, and the gap
is not noise — it is the whole result.

The arithmetic says what it would take. At 76.6% exposure, holding the index
unconditionally and the bill otherwise gives 0.766 × 10.2% + 0.234 × 2.8% =
**8.5%**. Ours is 8.6%: **the timing contributes about a tenth of a point a
year.** Theirs, at 80% exposure, requires the invested stretches to return
**13.3% annualised against an unconditional 10.2%** — a conditional selection
effect of three points.

Two possibilities cannot be separated here: an implementation detail we have
wrong, or a published figure that is optimistic. What can be said is that the
harness reproduces their benchmark to four figures and their model's entire risk
profile, and still finds no return advantage.

The turnover gap points at one candidate. Selecting the penalty monthly makes
ours trade 112% a year against their 44%, because a penalty that changes month
to month rewrites the state path. Held fixed, our turnover falls to 50-76% at
the higher penalties, in their range — and the Sharpe does not improve.

## Changing one thing: the benchmark the paper does not test

The paper measures against buy and hold. A companion study in this repository
found that no regime overlay it tested beat a **dynamic volatility target** — a
rule with no regimes, no estimation and no parameters beyond a target.

Same data, same harness, same period:

| | CAGR | volatility | Sharpe | worst drawdown |
| --- | --- | --- | --- | --- |
| jump model (penalty by cross-validation) | ~~8.6%~~ | ~~13.0%~~ | ~~**0.50**~~ | ~~−32.0%~~ |
| volatility target, matched to the model's volatility | 10.3% | 13.4% | **0.61** | −35.4% |
| volatility target, at the training-period volatility | 11.9% | 15.9% | **0.62** | −41.8% |


> ### ⚠ Correction, 2026-09-22 — the cross-validation row is withdrawn
>
> **No code ever produced it.** `scripts/replicate_shu2024.py` swept eight fixed
> penalties and none of its rows reads 8.6% / 13.0% / 0.50 / −32.0% / 76.6% / 112%.
> This is the fourth time this programme has published a figure whose generating code
> was not committed, and the remedy it used the first time is to write the missing
> analysis rather than retract unexamined. So it was written: `select_penalty` now
> re-chooses the penalty on each refit's own training window, scoring candidates on a
> held-out block by the excess-return separation the model exists to produce.
>
> **It does not reproduce the row either.**
>
> | | CAGR | vol | Sharpe | maxDD | exposure | turnover |
> |---|---|---|---|---|---|---|
> | cross-validation, as now written | 7.8% | 10.2% | 0.54 | −25.0% | 63.6% | **762%** |
> | the withdrawn published row | 8.6% | 13.0% | 0.50 | −32.0% | 76.6% | **112%** |
>
> Turnover is the tell: 762% against 112%. The selection lands on λ = 0 on **44 of 90
> refits**, and λ = 0 is the unpenalised model, which turns over 1448% a year. A
> criterion that scores state separation on a held-out block is indifferent to how much
> the resulting state flips, so it buys separation with turnover.
>
> **Why this stops here rather than continuing.** "Penalty by cross-validation" does not
> say cross-validated *on what*, and the criterion is therefore underdetermined by the
> document. Scoring on validation Sharpe instead, or on any measure that charges
> turnover, would select differently and would land somewhere else. Trying criteria
> until one returns 112% would be fitting the procedure to a remembered answer, which is
> the failure this repository exists to make visible. The row is withdrawn, what a
> defensible arm actually produces is published beside it, and the gap is left open.
>
> Nothing else in this document moves: the buy-and-hold reproduces to the decimal, the
> eight-penalty sweep is unchanged, and the conclusion — the risk reproduces, the return
> does not — rests on the sweep and not on the withdrawn row.

**At the same volatility, the free rule returns 1.7 points more a year and
scores 0.11 higher.** The jump model's advantage over buy and hold is real and
is almost entirely the de-risking; a volatility target does the same de-risking
without a model.

The model keeps one genuine edge: the drawdown. −32.0% against −35.4% at matched
volatility, and −26.6% in the published version against −41.8% for the
training-volatility target. Cutting exposure *because the market has turned* is
not the same as cutting it *because it has become volatile*, and the difference
shows up in the tail rather than in the mean.

> **Note, 2026-09-25.** The paragraph above ("At the same volatility…") and points 2 to 4
> below still quote the withdrawn cross-validation row (8.6%, 13.0%, 0.50, −32.0%, 76.6%).
> They are kept as written, and should be read with the correction above. What stands
> without that row: the benchmark reproduces to the decimal; the published model (0.68)
> sits about 0.06-0.07 above a volatility target at matched volatility (0.61), inside the
> detection threshold; and the conclusion rests on the eight-penalty sweep.

## What this establishes

1. **The benchmark reproduces exactly.** The harness is validated against an
   external reference, which no earlier hypothesis in this programme had.
2. **The model's risk profile reproduces; its return does not.** Volatility to
   within a tenth of a point, exposure to within three, and a return 2.6 points
   short.
3. **Our replication's return is what mechanical de-exposure predicts.** The
   timing adds a tenth of a point a year, not three.
4. **Against a dynamic volatility target the model loses**, 0.50 to 0.61 at
   matched volatility. Even the published 0.68 sits only 0.06 above it, inside
   the 0.27 to 0.55 detection threshold this repository measured for exactly
   this comparison.
5. **The drawdown advantage survives** and is the part worth keeping.

That is the same shape as this repository's central result, reached from a
different direction: the regime signal is real, what it carries is risk rather
than return, and the free alternative captures most of what it delivers.
