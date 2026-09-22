# AHL level B — undecidable at book level, and therefore not read

`docs/PRESPEC_AHL.md`, locked at `92e4e9e`. The instrument and its decision criterion
were committed at `9602526`, **before** the MDE existed.

```bash
.venv/bin/python scripts/run_ahl_level_b.py          # the instrument; stops at the criterion
```

**B1 is UNDECIDABLE on this sample.** The tree-corrected decision threshold is
**0.1265**, which is **49.6%** of the book's mean volatility miss. The criterion written
beforehand required it to be at most 25%. B1 was therefore **not read** and **no trial
was spent**, so the register stays at 84 distinct configurations. The tree goes to
level C, as the pre-registration says it does.

## The question, and how it was made measurable

§10 asks whether a correlation break predicts a rise in **risk-model error**, which it
defines as the gap between the book's predicted and realised volatility. It names the
question but not the statistic, so the statistic was fixed in the script's docstring and
committed before any number existed:

| piece | definition |
|---|---|
| the book | the M3 headline book, 46 instruments, net of headline costs, in excess of cash |
| its risk model | the incumbent model named in §5: the 63-session rolling sd of book returns, lagged one session. The multiplier is set so that this model predicts exactly 10%; the cap binds on 0.00% of sessions |
| the error | `e_t = |log(RV_{t+1..t+21} / 10%)|`: how far the book misses its own target over the next month |
| the break | an accepted CUSUM alarm (k 1, h 5, reset, 21-session dwell) on the residualised matrix distance, within the last 21 sessions |
| B1 | the difference in mean error between post-break sessions and all other sessions; stated sign positive |
| decision t | the smaller of HAC lag 6 and HAC lag 20, because the forward windows overlap |
| KY at B1 | a matched placebo: the same detector run on VOL63. B1 would also have had to survive it as a covariate |

## The instrument

4,760 usable sessions, 2008-05-13 to 2026-08-12. On the 28-instrument block the
residualised latent is orthogonal to volatility: Pearson +0.004, Spearman +0.123.

| reading | se | MDE raw | MDE tree-corrected |
|---|---|---|---|
| bootstrap, block 21 | 0.02837 | 0.0795 | 0.1070 |
| bootstrap, block 63 | 0.03066 | 0.0859 | 0.1156 |
| bootstrap, block 126 | 0.03354 | 0.0940 | **0.1265** |
| rotation null, 400 draws | 0.03029 | — | 0.1142 |

The rotation null is centred at −0.0015, so the statistic carries no bias from its own
construction. The unconditional mean error is 0.2550 (sd 0.1956), which puts the
criterion at 0.0637.

## Why the question cannot be decided here

Two things together. First, the error is mostly **sampling noise that no risk model can
remove**. Even if the model were perfect, the log of a realised volatility measured over
21 sessions has an sd of about 1/√42 ≈ 0.15. That noise accounts for most of the 0.255
mean miss. Second, B1 is a **single time series at book level**. It has 48 break
episodes and cannot use the cross-section, which is the lever that let the panel test of
level A resolve 2.12 points. The instrument can see a break that raises the average
miss by half, but nothing smaller. The expectation declared before the measurement was
60-75% and the MDE came in at 50%, better than expected but still twice the limit.

## The instrument did not stand where the pre-registration assumed

This was written into the docstring before the MDE was run, and is repeated here:

- The CUSUM crosses its threshold **166 times, 9.10 a year**, but the crossings cluster.
  After the 21-session dwell, **48 alarms, 2.63 a year**, are accepted. §1 had disclosed
  7.26 a year, a figure that reproduces only when the input is standardised on the full
  sample (6.91). That standardisation is not causal. 2.63 a year still clears the
  ~2-a-year threshold of the rule on a seventh device, but by a quarter rather than by
  the factor of 3.6 that §2 implied for this latent.
- B is on for 21.2% of sessions rather than about half.
- The matched volatility detector fires 12 times (0.66 a year), and its overlap with B
  is +0.055. The two detectors fire on different sessions, so volatility would not
  have been the reason for a pass or a failure here.

## What this does and does not say

It does **not** say that correlation breaks carry no information about risk. It says the
pre-registered book-level test cannot see any effect below roughly half the average
miss, and that no plausible effect is that large. The **panel** version of the same
question, which asks whether pairwise covariance forecasts fail after a break across 378
pairs, would have the power that level A had. It was **not** pre-registered. Running it
now would be an amendment and a new trial, chosen after this number was seen, and this
file does not run it.

**B2 is not built.** The prespec makes it reportable but never decisive. With B1
undecidable, building the reset-covariance book would test a construction whose premise
the sample cannot check, and the tree does not ask for it.

## Where the tree goes

To **level C**: in a detected consolidation, the instrument's own 5-day move is faded
instead of followed. Gate C0 passed **to the letter only**: +0.067 of gross Sharpe,
t 0.22, 2016-2026. This is the weakest branch of the tree and is flagged as such. C1
comes next, following the same order: instrument, MDE, criterion written in advance,
then the reading.
