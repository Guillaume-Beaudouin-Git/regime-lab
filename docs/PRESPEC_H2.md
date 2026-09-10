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
| 2026-09-10 | universe widened from 10 countries to 19 | The pre-specification assumed ten countries had usable free history. Nineteen do: the OECD sovereign yield series are keyed on two-letter country codes, and an initial availability check used three-letter codes and returned nothing. Corrected **before any return was computed**, and it moves the study toward the thirty-instrument minimum rather than away from it. |
| 2026-09-10 | the currency theme uses the **nominal** exchange rate, not the real one | A real rate needs a CPI ratio, and CPI is revised. Building it from current-vintage foreign CPI would reintroduce exactly the leak that choosing market-based state variables was meant to remove. At a one-year horizon inflation differentials among these countries are small against nominal moves, so the simplification costs little and protects the point-in-time claim. |
| 2026-09-10 | cell-level statistics are reported **before** any book is built | Hypothesis 1 failed because a sign map was fixed and a book assembled on top of it, which buried nine separate questions inside one number. Each theme-by-asset-class pair is now tested on its own, with a multiple-testing correction, and the book is only assembled from the pre-specified signs afterwards. Both are reported. |
| 2026-09-10 | frequency declared monthly | The sovereign yield series are published monthly. The signal is a one-year change, so monthly sampling loses nothing, and the alternative — interpolating to daily — would invent observations. |
