# Hypothesis 3 — falsified, and the placebo says how

Modelled macro surprises from 4,819 United States releases across thirteen
indicators, 1998 to 2026 — about **169 releases a year**, more than twice the
density the source model works with. Exponential kernel at a ten-day half-life,
ten-day holding horizon, four assets.

## The proxy passed its gate first

No free source of historical market consensus exists. Trading Economics
discontinued its guest account; ForexFactory publishes a forecast field but only
for the current week. So the surprise is modelled: the value as first published,
minus an autoregression refitted at every release on that series' own earlier
first releases.

Scored against the Philadelphia Fed's Survey of Professional Forecasters, which
is free and carries real consensus, on 95 common quarters of unemployment:

```
Pearson correlation   +0.667
rank correlation      +0.657
sign agreement         65.3%
dispersion             0.514 pt against the SPF's 0.354
```

Good enough to use, and wider than the real thing — an autoregression is
surprised more often than economists are, which is the expected direction.

## The kernel does not beat a surprise with no decay, and neither predicts

Slope in basis points of ten-day return per unit of score, Newey-West t at the
horizon lag:

| asset | decayed score, t | raw surprise, t |
| --- | --- | --- |
| dollar | −0.02 | −0.30 |
| equities | +0.06 | −0.36 |
| ten-year | +0.18 | +0.72 |
| gold | −0.82 | −1.49 |

Every statistic below 1.5. The Šidák threshold for four assets is 2.49 and
**none of the four reaches it**.

## The effective sample, measured rather than assumed

The score's autocorrelation is **0.9498 at one day** and 0.6585 at ten. That is
the kernel doing exactly what it is designed to do, and it is also why a
t-statistic computed on daily observations of it means very little.

```
daily observations                                7,476
after removing 10-day forward-return overlap        748
number of macro releases                          4,819
```

## The placebo is the result

Same release dates, same kernel, only the **sign** of each surprise randomised —
which destroys the information while leaving the timing and the autocorrelation
untouched.

| asset | real \|t\| | placebo mean \|t\| | placebo p95 | percentile of the real |
| --- | --- | --- | --- | --- |
| dollar | 0.02 | 0.87 | 2.20 | **2%** |
| equities | 0.06 | 0.97 | 2.32 | **2%** |
| ten-year | 0.18 | 0.91 | 2.14 | 10% |
| gold | 0.82 | 0.97 | 2.40 | 51% |

**The real signal is weaker than a sign-randomised version of itself.**
Destroying the information makes the statistic larger.

And the number that transfers: **a sign-randomised score reaches \|t\| of 2.1 to
2.4 at the 95th percentile, purely from the kernel's autocorrelation.** That,
not 1.96, is the threshold a claim built on this kind of score has to clear.

## Indicator by indicator, the distribution matches the null

Thirteen indicators against four assets, 52 tests:

```
cells above the Sidak threshold of 3.29     : 0
cells above a naive 1.96                    : 2   (chance gives 2.6)
largest |t| observed                        : 2.48

distribution of |t| : median 0.66, p90 1.49
expected under the null : median 0.67, p90 1.64
```

The distribution of the 52 statistics is **indistinguishable from noise**. Not
"nothing reached significance" — the whole distribution matches the null.

## What this says about the model it was modelled on

The published result is an information coefficient of 0.103 on AUD/NZD with an
information ratio of 0.867, drawn from a table of pairs.

**That specific claim was not tested and cannot be**: it needs about seventy
releases a year per currency pair from a paid consensus provider, and free
vintage history for Australia begins in 2010 with four to twelve a year.

What can be said is arithmetic on the published numbers, and it is not
favourable:

* The reported information ratio implies a breadth of 70 opportunities a year,
  consistent with two countries' release calendars. Fine.
* The t-statistic behind an IC of 0.103 depends entirely on how many independent
  observations there are. On 1,755 daily observations it is 4.36. Corrected for
  the seven-to-fourteen-day overlap the model itself specifies, it is **1.16 to
  1.64**. Counted as macro events, **2.30**. The Šidák threshold for the 28 pairs
  an eight-currency table contains is **3.12**. No defensible correction clears
  it.
* With the most generous assumption — 490 independent events — the expected
  largest \|IC\| across 28 pairs drawn from pure noise is **0.1039**. The observed
  0.1035 sits at the **53rd percentile** of that distribution. The best pair in
  the table is exactly where the best pair of a noise table would be.
* The other pair shown, EUR/GBP, has an IC of **−0.032**. One works, one is
  backwards, which is what noise looks like.
* The stated reason AUD/NZD works — both currencies depend on China, local rates
  and the global cycle — is an argument that their surprises are **correlated**,
  which lowers effective breadth and the information ratio with it: from 0.87 to
  0.61 at a correlation of 0.5.

## Classification

**Mechanical kill.** The gross statistic is indistinguishable from noise before
any cost is charged, on a denser release calendar than the source model uses,
with a validated surprise proxy. Under the reopening policy this stays closed.

What remains open is narrow and should be stated as such: this is the United
States against four assets. A real consensus surprise, on a currency pair whose
two economies publish independently, is a different measurement — and it is
behind a paywall.
