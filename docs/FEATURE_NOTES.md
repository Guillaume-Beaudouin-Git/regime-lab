# Feature notes

## The Hurst exponent carries nothing here, but the idea behind it does

The asymmetry, drawdown and long-memory family was promised in the frozen
charter and was missing from the first build of the feature matrix. That was an
omission against the protocol rather than a decision, and it is recorded as one.
Adding it brought the count from 44 features to 50.

The Hurst exponent belongs to that family, and it is prominent in practitioner
writing on market regimes. Measured rather than assumed, on a rolling 252-day
window estimated by detrended fluctuation analysis:

| | R² against realised volatility | IC against the next 21 days | t |
| --- | --- | --- | --- |
| `asy_hurst` | 18.7% | −0.006 | −0.13 |
| `asy_vr_20` (variance ratio) | 0.8% | **+0.124** | **+2.61** |
| `asy_vr_5` | 10.3% | +0.040 | +0.84 |
| `asy_skew_63` | 0.0% | +0.002 | +0.04 |
| `vol_rv_21` (control) | 99.4% | +0.082 | +1.74 |

And the estimator's own distribution:

```
mean 0.473   sd 0.092   min -0.205   p5 0.327   p95 0.615   max 0.781
```

**A Hurst exponent below zero is not a number the quantity can take.** The
estimator is producing values outside its admissible range, which is direct
evidence that on a window this short it is dominated by its own noise rather
than by the market. Its information about forward returns is nil.

The idea survives its estimator. The variance ratio asks the same question —
does this series trend or revert — with a sampling distribution that is known,
and it is the strongest single predictor in the table, ahead of realised
volatility, while sharing almost nothing with it (R² 0.8%).

**The exponent is kept in the feature set anyway.** It was declared before the
measurement, and removing a feature because it has just been seen to fail is the
mirror image of adding one because it has just been seen to work. The sparse
model will drop it out of sample, or it will not, and either way the decision is
made by the protocol rather than by hindsight.

Caveat on the table: the t-statistics use overlapping 21-day forward returns with
a crude effective-sample correction (n/21), not a proper HAC estimator, and each
is a single full-sample test with no multiple-testing correction. They are
suggestive, not results.

## The return convention was inconsistent between phases

Phase 0 measured power on returns in excess of the three-month bill; the first
phase 2 run did not, and reported Sharpe ratios gross of the risk-free rate. Two
numbers that should have been comparable were not, and one was wrong.

Both now come from `regime_lab/strategies/book.py`, which is the only place the
base book is defined. Excess is correct twice over: it is the project's stated
rule, and it is what makes the on/off rule mean what it says. Gating an excess
series to zero represents holding cash, which is what a flat book does. Gating a
gross series to zero represents holding nothing at all, which no one does — and
which quietly penalises the overlay rather than flattering it.
