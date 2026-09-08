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
