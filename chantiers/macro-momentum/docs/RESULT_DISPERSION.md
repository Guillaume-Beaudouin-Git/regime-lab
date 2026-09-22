# Macro momentum in developed markets: the cross-section was legislated away

This document supersedes the conclusions of `docs/RESULTS_H1.md` on one point,
stated in §7 rather than buried. The French write-up for the dissertation is
`docs/NOTE_DISPERSION_FR.md`.

Every figure below carries the command that produced it. Commands marked
**[reconstructed]** had no committed code when this document was written; the
scripts that produce them now live beside it and are listed in §8.

---

## 1. The question

Macro momentum is a published strategy with a published number: AQR report a
Sharpe of 1.2 over 1970-2016 for a book that goes long the countries whose
growth and inflation conditions are improving and short those where they are
deteriorating.¹ Two hypotheses were built to ask whether that number is
reachable with free data and a point-in-time contract.

They are different strategies, and the difference turned out to be the result.
Hypothesis 1 traded a **single country on an absolute signal** built from
**realised, revised** macro releases. Hypothesis 2 traded **nineteen countries
against each other** on **market-priced state variables that are never revised**.

Both failed. The interesting part is that they failed for unrelated reasons, and
the second reason is dated, institutional, and checkable against the historical
record rather than against a backtest.

¹ External citation, not verified in this repository. See §9.

---

## 2. What was pre-registered, and when

The order matters, so it is given as a sequence with commit evidence.

| | artefact | contents | commit |
|---|---|---|---|
| 1 | `docs/PRESPEC.md` | the sixteen-cell sign map, written from textbook macroeconomics, with the reasoning for each row and two cells flagged as contestable | `39df8d4` |
| 2 | `docs/RESULTS_H1.md` | H1 falsified | `39df8d4` |
| 3 | `docs/PRESPEC_H2.md` | H2's construction, fixed **before any cross-country data was downloaded** | `39df8d4`, amended `5c86f28` |
| 4 | `docs/RESULTS_H2.md` | H2 falsified | `5c86f28` |

The sign map is the object at risk in this kind of study. A practitioner scan
found that reproducing a published index methodology and comparing it against
160,000 random factor combinations placed the official version at the 98th
percentile, which is where a map lands when many were tried. So the map was
written first, argued from a first-year macroeconomics course, and never fitted.

**The map in the code matches the map in the pre-specification on all sixteen
cells.** Checked cell by cell, by hand, because the docstring in
`macro_momentum/exposures.py` claims a test suite enforces this and the `tests/`
directory is empty (§7).

H2's pre-specification carries four dated amendments, all made before any return
was computed. The substantive one widened the universe from ten countries to
nineteen after an availability check had failed on three-letter OECD country
codes where the series use two letters. It moves the study toward the house
minimum of thirty instruments rather than away from it, and still does not reach
it (§6).

---

## 3. Hypothesis 1: an absolute signal, one country

Four themes, each a one-year change and never a level, standardised on an
expanding window and clipped at three standard deviations. Twenty cross-asset
instruments. The sample runs **December 1994 to September 2026**, 8,278 trading
days; the two nested one-year differences and the 756-day standardisation burn-in
consume the first five years.

```
$ .venv/bin/python scripts/run.py
```

| | Sharpe | volatility | worst drawdown |
|---|---|---|---|
| gross | −0.35 | 6.9% | −67.7% |
| net of costs | **−0.38** | 6.9% | **−68.5%** |

Turnover is 0.0598 per day. Every asset class loses on its own signal: equities
−0.16 over eight instruments, bonds −0.04 over five, commodities −0.27 over six,
the dollar −0.36 on one.

### The map is worse than a coin flip

400 randomly signed maps, the same themes, the same instruments, the same costs:

```
placebo mean −0.03, sd 0.23, p95 +0.37
real map    −0.38  ->  7th percentile
373 of 400 random maps beat the frozen one
```

This comparison is clean, which had to be checked rather than assumed. Flipping
signs changes neither the signal's smoothness nor the trading it implies: placebo
turnover averages 0.0646 per day against the real book's 0.0598, spread 0.0502 to
0.0794, and the cost drag is 0.029 of Sharpe against 0.028. **[reconstructed]**
The map therefore loses on information, not on frictions.

### Sixteen cells, and the two that reach significance point the wrong way

Each theme regressed on each asset class composite, one-session lag, Newey-West
covariance. **[reconstructed]**

```
agreement with the frozen map : 7 of 16   (chance: 8.0)
cells with |t| > 2            : 2 of 16   (chance: 0.8)
largest |t|                   : 2.99  (inflation -> dollar)
```

| cell | slope | t (HAC 6) | empirical sign | frozen map |
|---|---|---|---|---|
| inflation → dollar | +1.31e-04 | +2.99 | +1 | −1 |
| inflation → commodities | −2.21e-04 | −2.02 | −1 | +1 |

Two of sixteen against 0.8 expected is not a finding: under Poisson(0.8),
P(X ≥ 2) is about 19%. What the table shows is the mechanism behind the 7th
percentile. **Both cells that clear the threshold contradict the frozen map.**
The textbook signs are not merely uninformative on this sample; on the two
relationships strong enough to see, they are backwards.

`RESULTS_H1.md` reports one such cell rather than two. The borderline cell sits
at 2.02 and a different compositing choice would move it below 2, so this is
plausibly a construction difference. It cannot be settled, because the code
behind the published figure was never committed.

### Not flipping the signs is the result

A map at the 7th percentile invites its own reversal. The reversed book, charged
the same costs on identical turnover, returns a net Sharpe of **+0.32**.
**[reconstructed]**

It was not reversed. Choosing the signs after seeing which direction paid is the
fabrication the pre-specification exists to prevent, and the frozen map is
reported as written.

`RESULTS_H1.md` gives +0.38 for the reversed book. That figure negates the *net*
Sharpe, which silently turns the cost drag into a gain; costs subtract in both
directions. The correct figure is +0.32.

---

## 4. Hypothesis 2: nineteen countries against each other

H1's defect was structural, so H2 changed the structure rather than the
parameters. Every input is a price, quoted monthly, never revised: the one-year
change in the 2-year yield, in the 10s-2s slope, and in the nominal exchange
rate against the dollar. Nineteen sovereigns, January 1990 to June 2026.

The nine theme-by-asset-class cells were tested **before any book was
assembled**, which is what H1 lacked: H1 buried sixteen relationships inside one
Sharpe.

```
$ .venv/bin/python scripts/run_h2.py
```

| theme | bonds | currencies | equities |
|---|---|---|---|
| policy | IC −0.0229, t −1.13 | IC −0.0156, t −0.68 | IC −0.0167, **t −1.45** |
| slope | IC −0.0088, t −0.38 | IC +0.0227, t +0.95 | IC +0.0014, t +0.10 |
| currency | IC +0.0278, t +0.80 | IC −0.0131, t −0.56 | IC −0.0096, t −0.36 |

Zero of nine cells reaches |t| = 1.96, where chance gives 0.5. The largest
statistic is 1.45. The Šidák threshold for nine tests is 2.77.

**No book was built.** There was nothing to build one from, and assembling one
anyway would have been the search for the subset that survives.

### The sample could not have detected the effect it observed

`PRESPEC_H2.md` promised the minimum detectable effect before any Sharpe. It was
never computed, so it is computed here, at 80% power and 5% two-sided.
**[reconstructed]**

```
MDE per cell        : 0.0322 to 0.0975 of IC   (median 0.0644)
largest |IC| observed :        0.0278
years of data required : 131 to 28,358
```

**No cell had the power to detect an effect the size of the one it measured.**
The median MDE is 2.3 times the largest information coefficient in the table.
This is a measurement of the question's difficulty, and it applies to all nine
cells rather than only to the pre-euro subsample that `RESULTS_H2.md` labels
underpowered.

---

## 5. The diagnosis: nine sovereigns became one instrument in 1999

A relative strategy needs countries to differ. The following is measured from
`data/raw/h2/rates.parquet`, per month across countries, then averaged over each
period. **[reconstructed]**

| period | dispersion of short rates | of long rates |
|---|---|---|
| 1990-1998 | **2.30%** | 1.72% |
| 1999-2007 | 1.37% | 1.01% |
| 2008-2015 | 1.08% | 1.44% |
| 2016-2026 | **0.89%** | 0.91% |

Among the nine sovereigns of this panel that adopted the euro on 1 January 1999
(Belgium, Germany, Spain, Finland, France, Ireland, Italy, Netherlands,
Portugal), the standard deviation of their own short rates against each other:

```
1990-1998 : 2.35%
1999-2007 : 0.00%
2008-2015 : 0.00%
2016-2026 : 0.00%
```

Checked month by month rather than rounded: the figure is **exactly zero in 309
of the 330 months after January 1999**. Nineteen of the remaining months carry
floating-point residue at 1e-16. Two are genuinely non-zero: **January 1999 at
0.158% and February 1999 at 0.055%**, the final convergence. From March 1999
onward the nine series are identical.

The panel reads nineteen names. On the policy theme it carries eleven
independent ones, and on the currency theme ten. The strategy was ranking
countries that share a central bank.

### The split, and two kills that need different labels

**[reconstructed]**

| period | usable cells | above \|t\| 1.96 | largest \|t\| |
|---|---|---|---|
| before the euro, 1990-1998 | 4 of 9 | 0 | 1.74 (policy → bonds) |
| after the euro, 1999-2026 | 9 of 9 | 0 | 0.95 (slope → currencies) |

**After 1999: a mechanical kill, closed.** The dispersion the strategy trades is
measurably zero for nine of nineteen sovereigns. No estimator recovers a
cross-section that does not exist, and under the reopening policy a mechanical
kill stays closed.

**Before 1999: a power failure, reopenable.** Nine years, four usable cells, a
largest statistic of 1.74. That is an underpowered sample rather than a
rejection, and the reopening policy treats the two differently. Reopening
requires a sample that has cross-sectional dispersion, which means paid vintage
data.

### How far this generalises, measured rather than asserted

The dispersion above is in percentage points, and rate *levels* collapsed over
the same period. A scale-free measure moves the opposite way.
**[reconstructed]**

| period | sd of short rates | mean level | coefficient of variation |
|---|---|---|---|
| 1990-1998 | 2.30% | 7.84% | **0.299** |
| 2016-2026 | 0.89% | 1.13% | **3.664** |

Absolute dispersion falls by a factor of 2.6 while relative dispersion rises by a
factor of 12. Both readings are correct and they answer different questions.
Among the ten non-euro sovereigns that kept their own policy rate, absolute
dispersion fell from 1.94% to 1.11%, a 43% decline against 61% for the full
panel, so part of the compression is a global move toward the zero bound rather
than the euro.

Three consequences, stated so the claim cannot be overread:

* The euro fact is unaffected. Nine identical series are identical under any
  normalisation.
* The diagnosis holds **for the signal as built**. H2's themes are one-year
  changes in percentage points, so the absolute scale is the one that governs
  them, and it is the one that compressed.
* The general proposition *developed macro no longer has enough cross-sectional
  dispersion to trade* is **false in scale-free terms** and must be stated at the
  scale of the signal.

---

## 6. Why hypothesis 1 had to fail

The comparison to the published result is the explanation, and it separates into
three differences of unequal weight.

**Structure, which is decisive.** AQR's macro momentum is cross-sectional across
countries. H1 built a single-country absolute signal, which is market timing
under a macro label. A timing version of a cross-sectional strategy was
constructed, and the cross-sectional property is where the published Sharpe comes
from.

**The signal.** AQR use changes in *forecasts*, which are forward-looking survey
revisions. H1 used changes in *published realised data*, which arrive with a
publication lag. By the time a one-year change in realised growth turns positive,
the market has had a year to price it.

**The period.** 1970-2016 against 1994-2026. The earlier sample is dominated by
the era this study sees nine years of: independent national monetary policies,
inflation rates differing by whole percentage points between neighbours, and
currencies moving on their own.

H2 fixed the first two and ran into the third.

A further constraint that neither hypothesis satisfied: the house methodology
requires thirty instruments for a cross-sectional book. H2's three cross-sections
are nineteen bonds, fourteen equity indices and ten currencies. **The minimum is
met in none of them.**

---

## 7. Corrections to the published documents

Named here rather than silently fixed, because the project has twice published
figures that a later regeneration contradicted.

**The central claim of `RESULTS_H1.md` does not hold.** It reads: *the signal
carries information — it beats noise decisively*, on the strength of the real
signal reaching the 99th percentile of a placebo. `PRESPEC.md` specified that
placebo as *a duty-cycle-matched random series*. `scripts/run.py` implements a
**shuffle** of each theme's observations instead. The four themes are one-year
changes of macro series and autocorrelate at 0.993 to 0.999 at one day; a
shuffled theme autocorrelates at −0.011. The consequence is arithmetic:

```
real book       : turnover 0.0598/day, cost drag 0.028 of Sharpe
shuffled placebo: turnover 1.2692/day, cost drag 0.527 of Sharpe   (21.2x)

gross of costs, the real signal sits at the 15th percentile of that placebo
```

The shuffled placebo loses because it trades twenty times more, not because it
knows less. Under four nulls that preserve the duty cycle, and therefore the
turnover: **[reconstructed]**

| null | turnover vs real | percentile of real, gross | net |
|---|---|---|---|
| random circular rotation, 200 draws | 1.1x | **22** | **22** |
| stationary block bootstrap, mean block 63d | 1.6x | 20 | 23 |
| stationary block bootstrap, mean block 126d | 1.4x | 18 | 20 |
| stationary block bootstrap, mean block 252d | 1.2x | 16 | 16 |

The real signal sits between the 16th and 23rd percentile of noise, gross and
net, across four independent constructions. **It does not carry directional
information.** The 99th percentile was an artefact of the substituted placebo.

The correction simplifies the H1 result rather than complicating it. The earlier
reading was a puzzle: an informative signal wired to a broken map. The measured
reading is that neither half works, which is consistent with the sixteen-cell
table, where the only two visible relationships run against the map.

Smaller corrections:

| published | measured | where |
|---|---|---|
| sample "from 1993 to 2026" | 1994-12 to 2026-09 | `RESULTS_H1.md` |
| reversed book "+0.38" | +0.32 net | `RESULTS_H1.md` |
| "one cell of sixteen" above \|t\| 2 | two, under the construction in §8 | `RESULTS_H1.md` |
| "exactly zero" | exact in 309 of 330 months; non-zero in Jan and Feb 1999 only | `RESULTS_H2.md` |
| "eight of them share the euro" | nine | `countries.py:4`, `run_h2.py:82` |

Unfulfilled pre-registration commitments, found by reading each
"what a result has to beat" section against the code:

* `PRESPEC.md` promised a duty-cycle-matched placebo (substituted) and an
  unconditional volatility-targeted buy-and-hold benchmark per asset class
  (absent).
* `PRESPEC_H2.md` promised a duty-cycle-matched placebo, a country-label-shuffled
  book, and the minimum detectable effect reported first. None was implemented.
  No book was built, so there was no Sharpe to frame, but the MDE was promised
  unconditionally and is supplied in §4.
* `exposures.py` states that the test suite checks the sign map against
  `PRESPEC.md`. `tests/` is empty. The check was done by hand for this document
  and passes on all sixteen cells.

---

## 8. Reproducing this document

The six figures marked **[reconstructed]** had no committed code. The scripts
that produce them are the deliverable that closes that gap:

| script | produces |
|---|---|
| `verify_h2_diagnostic.py` | the dispersion table, the euro bloc, the pre/post-1999 split |
| `verify_h1_cells.py` | the sixteen-cell table and the agreement count |
| `verify_h1_placebo.py` | the duty-cycle-matched nulls of §7 |

Construction of the sixteen cells, fixed before the output was seen: each asset
class becomes one equal-weight composite of its instruments, each instrument
inverse-volatility scaled exactly as `book.py` does it; each cell regresses the
composite's next-day return on the standardised theme lagged one session, with a
Newey-West covariance at lag 6, the house rule, and lag 21 reported as a
sensitivity. The agreement count and the |t| > 2 count are identical at both
lags.

The duty-cycle-matched nulls use the companion project's own
`regime_lab.analysis.bootstrap.stationary_indices` for the block variants. The
rotation variant shifts each theme circularly by a random offset, which preserves
the entire autocorrelation function and therefore matches turnover by
construction.

---

## 9. What remains open, and under what condition

**Closed, mechanically.** Post-1999 developed-market cross-country macro
momentum on rate-based state variables. The raw material is measurably zero for
nine of nineteen sovereigns.

**Closed, on the evidence.** The single-country absolute version. It loses 0.38 of
Sharpe net, its map is beaten by 373 of 400 random maps, and its signal does not
beat a duty-cycle-matched null.

**Reopenable, with a stated condition.** The pre-1999 sample is a power failure
rather than a rejection: nine years and four usable cells, largest statistic 1.74.
Reopening requires a sample with cross-sectional dispersion that the measurement
in §5 can confirm before any return is computed, which means paid vintage data.

**Reopenable, with two named obstacles.** Emerging markets, where dispersion runs
the other way. The ratio of emerging to G10 dispersion is reported as moving from
0.7x in 2006-2012 to 1.5x in 2020-2026, with free coverage of twelve currencies
from 2006, twelve equity indices from 2012 and six local bond ETFs from 2013.
**Neither figure is verified here**: no emerging-market code or data exists in
`macro-momentum`, and both are carried over from a project handoff note. They are
a lead to be checked, not a measurement, and the check belongs before any
pre-registration rather than after. The obstacles are a short sample, costs five
to ten times the G10, and capital controls that a spot backtest cannot see.

**Not established.** That developed macro lacks cross-sectional dispersion in
general. The claim holds at the absolute scale the signal uses and reverses under
a coefficient of variation (§5).

Two citations carry the comparison that structures this result and neither is
verifiable from this repository: AQR's Sharpe of 1.2 over 1970-2016 and their
−0.22 correlation to equities. A paginated reference belongs in this document and
does not currently exist anywhere in `macro-momentum`.
