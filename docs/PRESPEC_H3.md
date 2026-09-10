# Hypothesis 3 — pre-specified before any data was touched

Hypotheses 1 and 2 tested macro **states** and macro **changes**, and both
failed. This one tests macro **surprises**: the gap between what printed and
what was expected. That is an event, not a state, and the distinction is the
reason it is worth a third attempt rather than a variation on the first two.

The design follows a model built by another practitioner (Dedale,
`brieuctrader/Dedale`, `Python/macro_score_overview.ipynb`) whose published
result is one currency pair with an information coefficient of 0.103 and an
information ratio of 0.867. This study cannot reproduce that result and does not
try to; what it tests is the **mechanism**, on data where a mechanism can be
tested honestly.

## What is being tested, and what is not

**Not testable here.** The published result is on AUD/NZD and needs roughly
seventy macro releases a year per currency pair, sourced from a paid consensus
provider. Free vintage history for Australia begins in 2010 and carries four to
twelve releases a year. **The specific pair claim is out of reach and is not
addressed.**

**Testable here.** The United States has twelve monthly indicators with genuine
vintage history reaching 1999 or earlier, which is about a hundred and forty
releases a year — twice the density the published model works with. The
mechanism can therefore be tested more strictly than it was built.

## The claim

**An exponentially decayed score built from macro surprises predicts returns
over the following one to two weeks.**

## Construction, fixed here

**The surprise.** No free source of historical market consensus exists. The
surprise is therefore modelled: for each release, the value **as first
published**, minus a forecast built from an autoregression on the series' own
prior **first releases** only. Nothing revised, nothing published later, enters
either side.

This is not the market's surprise, and the difference is a limitation rather
than a detail: the market trades the deviation from what economists said, not
from what a time-series model said. The proxy is therefore **validated against
the Philadelphia Fed's Survey of Professional Forecasters**, which is free,
quarterly and carries real consensus, on the series where both exist. The
correlation between the two is reported before any return is examined, and a
weak correlation invalidates the proxy rather than being worked around.

**The kernel.** `S(t) = Σ w ξ exp(−λ (t − t_k))`, with the half-life fixed at
**10 days** — the midpoint of the seven-to-fourteen range the source model
states. Not tuned.

**The weights.** Equal across indicators. The source model weights inflation and
employment above output; adopting a weighting scheme would add parameters this
pre-specification does not grant, and equal weight is the honest default when
the alternative is unmeasured.

**The horizon.** Ten trading days, matching the half-life.

## The three structural questions, which matter more than the headline

1. **Does the kernel earn its place?** The decayed score is compared against the
   raw surprise on its release day, with no decay at all. If the kernel adds
   nothing, the model's central mechanism is decoration.
2. **What is the effective sample size?** Overlapping ten-day forward returns
   sampled daily share their future. The number reported is the corrected one,
   and every t-statistic uses it.
3. **Does anything survive multiple testing?** Every asset tested is reported,
   with the Šidák threshold for the number tested. **The distribution of the
   statistics is the result; its maximum is not.**

## What a result has to beat

* A **sign-shuffled placebo**: the same release dates and the same kernel, with
  the sign of each surprise randomised. This destroys the information while
  keeping the timing and the autocorrelation intact.
* The **minimum detectable effect**, reported before the coefficient.
* A cost gate before anything else: gross per trade divided by round-trip cost
  must exceed 1.5. The author's own registry records this ratio never exceeding
  0.36 across three independent measurements in retail foreign exchange.

## Assets

Dollar index, S&P 500, ten-year Treasury, gold. Four, declared here, and all
four reported whatever they show.

## Frozen

| date | change | reason |
| --- | --- | --- |
| — | — | — |
