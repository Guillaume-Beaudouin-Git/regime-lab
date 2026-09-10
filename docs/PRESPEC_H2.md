# Hypothesis 2 — pre-specified before any data was touched

Hypothesis 1 built a single-country, absolute macro signal and traded it
directionally. That is market timing wearing a macro label, and it failed. This
hypothesis changes the structure rather than the parameters.

## The claim

**Countries whose monetary and growth conditions are improving relative to other
countries outperform those where they are deteriorating, within each asset
class.** Long the improving, short the deteriorating. The bet is relative and
never directional, so the book carries no view on whether markets go up.

## Why market-based state variables rather than macro releases

Every input is a price, quoted daily, **never revised**:

| theme | measured as | why it is not a macro release |
| --- | --- | --- |
| Policy | one-year change in the 2-year government yield | the rates market prices the next two years of policy; a published rate prints what already happened |
| Growth expectations | one-year change in the 10-year minus 2-year slope | a steepening curve is the market's own growth forecast, available the same day |
| Financial conditions | one-year change in the real exchange rate against the dollar | capital flows respond to conditions before statistics record them |

This removes the two defects of hypothesis 1 at once. The signal becomes
forward-looking, because rates markets anticipate; and the vintage problem
disappears, because a quoted yield is never restated.

## Universe

Government bond yields, equity indices and currencies for the countries with
usable free daily history: United States, Germany, United Kingdom, Japan,
Canada, Australia, Switzerland, France, Italy, Spain. Ten is thin — the frozen
methodology of the companion project requires thirty instruments for a
cross-sectional book — so **this study reports its own minimum detectable effect
first and treats a positive result as provisional until the count is met**.

## Construction, fixed here

* Signals cross-sectionally ranked, then demeaned, so the book is neutral by
  construction rather than by fitting.
* Equal weight across the three themes. Weighting them would be three parameters
  the pre-specification does not grant.
* Each leg volatility-targeted before aggregation.
* Signal lagged one session.
* Costs charged at the levels already declared in `book.py`.

## What a result has to beat

* A duty-cycle-matched placebo.
* The same book with country labels shuffled — which destroys the cross-section
  while keeping every other property.
* The minimum detectable effect, reported before the Sharpe.

## Frozen

| date | change | reason |
| --- | --- | --- |
| — | — | — |
