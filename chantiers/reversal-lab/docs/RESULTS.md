# The short-term reversal premium has been arbitraged away

Measured on Ken French's CRSP-universe portfolios, which include every company
that later delisted. That matters more here than in almost any other study: the
names absent from a retail listing are the ones that fell and never recovered,
which is exactly what a reversal strategy buys.

## The effect was enormous, and it is gone

Long the worst decile of prior-month returns, short the best, gross of costs:

| period | Sharpe | annual return | daily premium |
| --- | --- | --- | --- |
| 1990-1999 | **3.64** | +61.2% | +24.3 bp |
| 2000-2009 | 1.23 | +42.5% | +16.9 bp |
| 2010-2019 | 0.54 | +8.9% | +3.5 bp |
| **2020-2026** | **−0.18** | **−6.9%** | **−2.7 bp** |

The gross daily premium fell from 24 basis points to minus three. Since 2010 the
spread has been positive in **9 years out of 17** — a coin flip gives 8.5.

The 1990s figure was never harvestable at anything like that level: the quoted
spread was an eighth of a dollar, and the strategy rebalances monthly across the
whole listed universe including microcaps.

## What is left is exactly what Nagel (2012) predicted, and it is not enough

Reversal is a payment for providing liquidity, so it should pay when liquidity is
scarce. Split by an expanding rank of the VIX:

| VIX tercile | 1990-2009 | 2010-2026 | 2020-2026 |
| --- | --- | --- | --- |
| low | 3.69 | −0.34 | −1.44 |
| middle | 2.25 | 0.11 | −0.27 |
| **high** | 1.64 | **0.39** | **0.18** |

Two things at once. The shape flipped: in the first two decades reversal paid
*more* in calm markets, which contradicts the liquidity story; since 2010 it pays
only in the top volatility tercile, which confirms it. And the surviving figure
is 0.18 to 0.39 **gross**, in the regime where spreads are widest and where a
liquidity provider is by definition the one who cannot get out.

## Classification

Under the reopening policy this is a **mechanical kill**: the gross premium is
approximately zero and turns negative in the current decade, before any cost is
charged. That is the category that stays closed. It is not a vehicle problem, not
a cost problem, not a power problem — there is nothing left to harvest.

The conditioning layer built for this study — selecting stocks by a
persistence estimator — was never run against the recent sample, because
conditioning a premium that no longer exists is a search for the subset where a
dead effect still lives.

## What this closes

* Short-term reversal on US equities, at any horizon from one day to one month.
* By extension, the persistence-estimator conditioner that motivated the study:
  a companion measurement had already shown it selects no better than a coin
  flip on sector portfolios, and there is now no premium for it to select from.

## What it does not close

The measurement is US-listed equities. Nothing here speaks to reversal in other
markets, at intraday horizons inside a single session, or to the liquidity
provision business as run by someone who is actually a market maker rather than
a taker.
