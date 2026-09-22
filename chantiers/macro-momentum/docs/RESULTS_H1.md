# Hypothesis 1 — falsified, and the diagnosis says why

The frozen sign map, applied to four macro themes measured as one-year changes,
run on twenty cross-asset instruments from 1993 to 2026.

| | result |
| --- | --- |
| book, net of costs | **Sharpe −0.38**, worst drawdown −68.5% |
| placebo on the **map** — 400 random sign maps | mean −0.03 → real map at the **7th percentile** |
| placebo on the **signal** — duty-cycle-matched, as pre-specified | real signal at the **16th–23rd percentile** |

> **Correction, 2026-09-14.** The row above previously read "themes shuffled →
> real signal at the 99th percentile", and this document concluded from it that
> *the signal carries information — it beats noise decisively*. **That conclusion
> was false, and it came from a placebo the pre-specification did not ask for.**
>
> `PRESPEC.md` specifies "the same map with the signal replaced by a
> **duty-cycle-matched** random series". `scripts/run.py` instead shuffled each
> theme's observations independently. The four themes are one-year changes and
> autocorrelate at 0.993 to 0.999 at lag one; shuffled, they autocorrelate at
> −0.011. The shuffled book therefore turns over about twenty times more than the
> real one and pays 0.527 of Sharpe in transaction costs against the real book's
> 0.028. **The real signal won that comparison on costs, not on information.**
>
> Under four nulls that preserve the duty cycle and therefore the turnover — a
> circular rotation, which preserves the entire autocorrelation function exactly,
> and a stationary block bootstrap at mean blocks of 63, 126 and 252 days — the
> real signal sits at the **16th to 23rd percentile, gross and net**. It is worse
> than a random series with the same trading rhythm.
>
> Reproduce with `scripts/verify_h1_placebo.py`.

So only one of the two placebos says what it was thought to say. The map result
stands and is clean: turnover is matched between the real map and the random ones
(0.0646 against 0.0598), and 373 of 400 random maps beat the frozen one. **The
sign map does worse than a coin flip, and the signal it is applied to does not
carry information either.**

The cell-by-cell diagnosis agrees: the frozen map matches the empirical signs on
**7 of 16 cells**, where a random map would match 8. Two cells of sixteen reach a
Newey-West |t| above 2 — not one, as this document previously stated — and both
*contradict* the frozen map (inflation → dollar +2.99, inflation → commodities
−2.02). Two of sixteen against 0.8 expected is still noise, P ≈ 19%. There is no
reliable directional relationship here for any map to exploit.

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
