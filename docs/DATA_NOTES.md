# Data notes

Findings that cost time to discover and would silently corrupt results.

## FRED truncates ICE BofA series to two years

`BAMLH0A0HYM2` and `BAMLC0A0CM` return only the trailing ~24 months from the
public CSV endpoint, with no error and no warning — a licence restriction. A
credit feature built on them exists only after 2023 and is either dropped by the
panel or forward-filled into nonsense. Replaced by `BAA10Y` and `AAA10Y`
(Moody's over the 10-year Treasury), daily from 1986.

The fetch script now prints `SHORT HISTORY` whenever a series starts more than
two years after the requested sample start. Any new series must be checked
against that flag before it is used.

## Short coverage on the current price catalogue

Measured on business days 1990–2026:

| series | coverage | cause |
| --- | --- | --- |
| `eq_eu` (`^STOXX50E`) | 53% | Yahoo history incomplete before the 2000s |
| `cmd_oil` (`CL=F`) | 71% | continuous futures start ~2000 |
| `cmd_gold` (`GC=F`) | 71% | continuous futures start ~2000 |
| `fin_stlfsi` (`STLFSI4`) | 89% | index begins 1993-12-31 |

To be resolved in phase 1 before these feed any feature: WTI has a full daily
history on FRED (`DCOILWTICO`, 1986–), and a European equity index with usable
Yahoo depth (`^GDAXI`, 1987–) can stand in for the Euro Stoxx. A series below
~90% coverage must not enter the panel without an explicit note here.

## ALFRED vintages need the API, not the CSV endpoint

The public `fredgraph.csv` endpoint has no vintage parameter that works; the
documented URL forms all return an HTML error page. Real-time histories come
from `api.stlouisfed.org/fred/series/observations` with
`realtime_start=1776-07-04&realtime_end=9999-12-31`, which requires a free key.

## Repeated levels on the yield series are granularity, not staleness

The quality screen flags `bond_us_10y` and `bond_us_long` at 13-16% repeated
levels in 1991-1993, above the 10% threshold. This is not a stale feed: Treasury
yields are quoted to two decimal places, so an unchanged print is a genuine
observation whenever the day's move is under a basis point, and the longest run
of repeats is two or three sessions rather than the long flat stretches a dead
feed produces.

The threshold is calibrated for price series. Yields are read against the run
length instead. The flag stays in the report rather than being silenced, because
suppressing a warning is how the next real one gets missed.

## The power calculation is a lower bound

`scripts/phase0_report.py` compares a 60/40 book with a volatility-targeted
version of itself. The two correlate at 0.91, so the common market move largely cancels in
the paired difference and its standard error sits near its floor — which is also
why the block length barely changes it. A regime overlay tracks the benchmark
less closely and will have a larger minimum detectable effect. The number that
decides stop rule 3 is the one recomputed against the actual overlay in phase 3.

## Two Yahoo commodity series cost ten years of sample

Yahoo's continuous futures for gold (`GC=F`) and WTI (`CL=F`) begin in late
2000. Because the feature matrix is used on its common sample, those two columns
alone pushed the study's start from March 1992 to November 2001 — losing four of
the fourteen stress episodes, to buy two features out of forty-four.

**Oil** is substituted with `DCOILWTICO` from FRED: full daily history from
1990, never revised, so the publication-lag path is exact rather than
approximate.

**Gold is dropped rather than proxied.** There is no free long series: the LBMA
fixing was withdrawn from FRED, the bullion ETF starts in 2004, and `^XAU` is an
index of gold *miners* — an equity sector that fell with the market in 2008.
Using it would have injected equity beta into a feature labelled as a safe
haven, which is worse than not having the feature.

`STLFSI4` is dropped on the same logic: it measures the same construct as the
Chicago Fed's NFCI but begins in 1994, and keeping both would have cost four
years including the 1990 recession for no new information.

The general rule this establishes: **a feature is worth its coverage cost only
if no existing feature carries the same construct over a longer sample.**
