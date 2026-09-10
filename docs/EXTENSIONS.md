# Extensions — work that follows the frozen study, and does not touch it

The charter is frozen and its results are published. Nothing in
`regime_lab/extensions/` alters a number in `RESULTS_FINAL.md`,
`RESULTS_CLASSIFIER.md` or `RESULTS_T2.md`; the features, the states and the
protocol of the study itself are untouched.

What is here comes from a scan of what quantitative macro funds actually deploy,
run after the study closed. That scan produced three leads the study had opened
and never executed, and each is written up with what practice claims and what
this repository measures.

| extension | what practice claims | status |
| --- | --- | --- |
| Effective number of risk factors | a CTA publishes a linear inverse relation with trend performance — eight factors in 2022 was its best year, thirty-plus in 2023 its worst | their measure is **contemporaneous**; this tests whether it *predicts* |
| Continuous volatility attenuator | a practitioner running public code with his own money measures Sharpe 0.49 in calm markets against 0.003 in turbulent ones, and converts it into a smoothed multiplier worth 0.05 to 0.09 | tested here on a trend book |
| Regime into barrier geometry | the one large quantitative fund that documents a deployment activates its regime model only for shocks beyond three standard deviations | needs the barrier simulator, which lives in another repository |

The first is the one worth running first: it costs a single lag and it decides
whether the study's best non-volatility feature is a signal or a dashboard.

---

# Results

Universe: 46 cross-asset instruments, July 2000 to September 2026, 6,822
sessions. Reference book: twelve-minus-one time-series momentum, each instrument
at a common ex-ante volatility, equal risk, signal lagged one session. **Sharpe
0.55.** Nothing below alters a frozen result.

## 1. The concentration measure neither predicts nor describes, here

Effective number of factors, from the perplexity of the eigenvalue spectrum:
mean 12.8 across 46 instruments, ranging from 2.3 to 16.9. Regressed against the
book's forward performance:

| lag | effective factors, t | absorption ratio, t |
| --- | --- | --- |
| contemporaneous | −0.79 | +1.05 |
| one month ahead | −1.06 | +1.53 |
| three months ahead | −0.40 | +0.65 |
| six months ahead | −1.24 | +1.27 |

Every sign points the published way — fewer independent factors, better trend —
and nothing reaches two. The contemporaneous relation the CTA publishes does not
even replicate as a description on this universe.

**An earlier run of this table reached t = 2.18** at one month. It used a panel
built by requiring all instruments to have data on every date, which started the
sample in December 2007 and discarded eight years. Relaxing that — letting
instruments enter as they list — dropped the statistic to 1.53. The apparent
signal was the truncation.

This is a non-replication rather than a refutation: the published relation is
measured on a hundred-plus futures universe with that firm's own trend system.

## 2. The volatility regime is real and larger here than published

Split by an **expanding** percentile of average cross-instrument volatility, so
the ranking never sees the future:

| volatility tercile | book Sharpe | share of time |
| --- | --- | --- |
| low | **0.82** | 48% |
| middle | 0.67 | 27% |
| **high** | **−0.32** | 13% |
| unclassified (756-day burn-in) | — | 12% |

The published figures on 37 futures over 36 years are 0.49, 0.40 and 0.003. The
conditional effect replicates, keeps its sign, and is **wider** here.

## 3. The attenuator passes its placebo — and is beaten by a rule with no parameters

The published device is a continuous multiplier on each instrument's own
volatility percentile, smoothed over ten days. Applied per instrument:

| construction | Sharpe | gap | t |
| --- | --- | --- | --- |
| reference book, full sample | 0.51 | — | — |
| per-instrument, gross renormalised to 1 | 0.51 | −0.003 | −0.35 |
| per-instrument, gross free to move | 0.55 | +0.037 | +0.18 |

The first row is kept as a control because the mistake is worth recording: with
gross exposure renormalised to one, multiplying by the attenuator **reallocates
between instruments instead of reducing risk** — not the published intervention
at all. With gross free it becomes the right operation, and a t of +0.18 says it
adds nothing measurable.

The device acts per instrument; the effect in section 2 is at market level. So
the same multiplier was applied to the book on the market's own volatility
percentile. **Every comparison below is on the common sample** — the 5,994
sessions from September 2003 where the expanding percentile exists. This matters:
against the full-sample book the same gaps read +0.07 instead of +0.14, and the
difference is the burn-in period, not the conditioning.

| construction | Sharpe | gap | t |
| --- | --- | --- | --- |
| reference book, common sample | 0.44 | — | — |
| book-level, L = 2 − 1.5Q | **0.58** | **+0.144** | **+2.60** |
| gentler, L = 1.5 − 0.75Q | 0.53 | +0.091 | +2.60 |
| binary, flat in the top tercile | 0.64 | +0.199 | −0.47 |

The binary row is the tell: the **largest** Sharpe gain of the three carries a
**negative** t on the mean daily difference. The gain is arriving in the
denominator, not the numerator — less volatility, not more return.

Then the matched placebo. The leverage profile is the observation, not the day,
so each draw rotates that profile circularly against the dates: autocorrelation
preserved exactly, alignment destroyed.

```
placebo (leverage profile rotated, 400 draws)   : mean +0.428, sd 0.073, p95 +0.552
real                                            : +0.583  ->  99th percentile
minimum detectable effect at 80% power          : 0.205
observed gap                                    : +0.144
```

The placebo mean of +0.428 sits on the unconditional book Sharpe of 0.44, which
is the check that the placebo itself is unbiased rather than the signal.

**So the device clears its placebo.** It does not clear the second test, and the
second test is the one that decides:

| control, no parameters | Sharpe | gap | t |
| --- | --- | --- | --- |
| **dynamic volatility target** | **0.68** | **+0.244** | **+2.89** |
| regime attenuator | 0.58 | +0.144 | +2.60 |

Scaling the book by the inverse of its own trailing volatility — a rule that
knows nothing about regimes, has no percentile, no threshold and no fitted
constant — beats the regime device by a full tenth of a Sharpe. And the
attenuator's own gap of +0.144 is below the 0.205 this sample can resolve at
80% power, so it is **underpowered, not a pass**.

One further caveat, stated because it would otherwise be buried: the choice to
condition on **market** volatility came from measuring the tercile spread on the
same sample. The percentile itself is expanding and causal, but the decision to
look there is not out of sample.

## Verdict

The volatility regime is real, replicates, and is larger on this universe than
in the published source. **No device tested beats a parameter-free volatility
target.** That is the same shape as the study's own central result — regimes
carry variance rather than mean, so the honest use of one is sizing, and plain
inverse-volatility sizing already collects it.

This is now the third independent arrival at that conclusion in this repository:
the classifier study measured it directly (+2 to +4 points of incremental R² on
forward volatility, ~0.02 on returns); the Shu 2024 replication found the
published jump-model overlay at 0.50 against 0.61 for the free rule at matched
volatility; and this extension finds the practitioner attenuator at 0.58 against
0.68. Three different devices, three different sources, same ordering.

---

## Amendment, 2026-09-10

An earlier version of this file reported a tercile split of 0.88 / 0.89 / −0.45,
a book-level gap of **+0.053 at t 2.59**, a reference book Sharpe of **0.55**,
and a placebo verdict of **"94th percentile, does not clear"**. Those figures were
produced in a working session and the code that produced them was never
committed: `run_extensions.py` as committed computed only the per-instrument
attenuator, and section 1 alone reproduced from it.

The analysis has been rewritten into the script and re-run. Section 1 reproduced
to the digit, confirming the data are unchanged. The t-statistics reproduced
(+2.60 against +2.59, −0.47 against −0.41). The Sharpe levels did not, because
the earlier run compared constructions measured on **different samples** — the
attenuated book starts after a 756-day burn-in, the reference book did not.

The verdict survives, and the reason for it does not. The device does **not**
fail its placebo; it passes at the 99th percentile. It fails against a control
the earlier version never ran. This is the same failure mode already caught once
on `RESULTS_T2.md`: a document regenerated from a session rather than from the
committed script.
