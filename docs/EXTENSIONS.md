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

Split by an expanding percentile of average cross-instrument volatility:

| volatility tercile | book Sharpe | share of time |
| --- | --- | --- |
| low | **0.88** | 38% |
| middle | 0.89 | 29% |
| **high** | **−0.45** | 20% |

The published figures on 37 futures over 36 years are 0.49, 0.40 and 0.003. The
conditional effect replicates and is **wider** here.

## 3. The attenuator: right phenomenon, and the placebo still refuses it

The published device is a continuous multiplier on each instrument's own
volatility percentile, smoothed over ten days. Applied as written:

| construction | Sharpe | gap | t |
| --- | --- | --- | --- |
| reference book | 0.55 | — | — |
| per-instrument attenuator | 0.56 | +0.004 | +1.38 |

Nothing. The first attempt was worse and for a mechanical reason worth
recording: gross exposure was normalised to one, so multiplying by the
attenuator **reallocated between instruments instead of reducing risk** — not
the published intervention at all. With gross free to move it becomes the right
operation, and still adds nothing.

The device acts per instrument; the effect measured above is at market level. So
the same multiplier was applied to the book on the market's own volatility
percentile:

| construction | Sharpe | gap | t |
| --- | --- | --- | --- |
| book-level, L = 2 − 1.5Q | **0.61** | **+0.053** | **+2.59** |
| gentler, L = 1.5 − 0.75Q | 0.58 | +0.028 | +2.59 |
| binary, flat in the top tercile | 0.60 | +0.047 | −0.41 |

**+0.053 with a t of 2.59, inside the published range of +0.05 to +0.09.** And
then the matched placebo:

```
placebo (same leverage profile, dates shuffled) : mean +0.506, sd 0.068, p95 +0.609
real                                            : +0.608  ->  94th percentile
minimum detectable effect at 80% power          : 0.190
observed gap                                    : +0.077
```

**Two independent reasons to refuse it.** It does not clear its own placebo, and
the observed gap is less than half the effect this sample could resolve.

The disagreement between t = 2.59 and the 94th percentile is the finding, not a
contradiction. The t-statistic treats each day as an observation; the placebo
treats the leverage *profile* as the observation, and that profile is heavily
autocorrelated. This is a direct replication of a methodology lesson already in
the author's own registry: an autocorrelation-matched placebo catches what
neither the t-statistic, the bootstrap, nor the walk-forward sees.

One further caveat, stated because it would otherwise be buried: the choice to
condition on **market** volatility came from measuring the tercile spread on the
same sample. The percentile itself is expanding and causal, but the decision to
look there is not out of sample.

## Verdict

The volatility regime is real, replicates, and is larger on this universe than
in the published source. **No device tested converts it into a detectable gain.**
That is the same shape as the study's own central result — a conditional effect
that is genuine and does not survive the trip to a portfolio — arrived at from a
different direction.
