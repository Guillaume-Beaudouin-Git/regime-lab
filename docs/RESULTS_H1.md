# Hypothesis 1 — falsified, and the diagnosis says why

The frozen sign map, applied to four macro themes measured as one-year changes,
run on twenty cross-asset instruments from 1993 to 2026.

| | result |
| --- | --- |
| book, net of costs | **Sharpe −0.38**, worst drawdown −68.5% |
| placebo on the **map** — 400 random sign maps | mean −0.03 → real map at the **7th percentile** |
| placebo on the **signal** — themes shuffled | mean −0.72 → real signal at the **99th percentile** |

Those two placebos together are the finding. **The signal carries information —
it beats noise decisively. The sign map does worse than a coin flip.**

The cell-by-cell diagnosis confirms it: the frozen map agrees with the empirical
signs on **7 of 16 cells**, where a random map would agree on 8. And only **one
cell of sixteen** reaches a Newey-West |t| above 2, against 0.8 expected by
chance. There is no reliable directional relationship here for any map to
exploit.

**The signs are not flipped.** Reversing them after seeing this would produce a
Sharpe of +0.38 and would be exactly the fabrication the pre-specification
exists to prevent.

## Why this differs from the published result it was modelled on

AQR report a Sharpe of 1.2 over 1970-2016 for macro momentum. Three differences,
and the second is almost certainly the one that matters.

**The signal is not the same.** They use changes in *forecasts* — forward-looking
survey revisions. This study uses changes in *published realised data*, which is
backward-looking and arrives with a publication lag. By the time a one-year
change in realised growth is positive, the market has had a year to price it.

**The structure is not the same, and this is the decisive one.** Their macro
momentum is **cross-sectional across countries**: long the country whose growth
is improving, short the one whose growth is deteriorating. That is where their
−0.22 correlation to equities comes from. This study built a single-country
absolute signal, which is market timing. **A timing version of a cross-sectional
strategy was built, and market timing is the thing that never works.**

**The period is not the same.** 1970-2016 against 1993-2026, and the authors
write that their backtests "are likely overstated".

## Classification

A **mechanical kill for this specification**, and an argument for a different
one. Hypothesis 2 is pre-specified separately and does not inherit anything from
this file except the infrastructure.
