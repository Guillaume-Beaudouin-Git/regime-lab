# Calibration notes

## The jump penalty selects the floor of its own grid

Over thirteen in-training calibrations, the declared grid
`(1, 3, 10, 30, 100, 300)` returns `lambda = 1` — its minimum — nine times, and
never returns anything above 10. That is a **boundary solution**, and boundary
solutions usually mean the grid is centred in the wrong place.

**The grid is not extended.** Widening it after seeing which end wins is an
amendment made in the light of results, which is the exact move the protocol
freeze exists to prevent. It is declined and recorded here instead.

Two facts make the decision easy rather than merely principled:

* The objective is nearly flat across the live part of the grid. Training Sharpe
  at `lambda = 1` versus `lambda = 3`: 0.19 vs 0.19 (2014), 0.22 vs 0.21 (2018),
  0.21 vs 0.18 (2022). Nothing meaningful sits between them, so a lower penalty
  would not change a conclusion.
* Even at the floor the model switches only 1.3 to 2.2 times a year. Persistence
  here is a property of the feature space, not of the penalty: the states are
  slow because the features are slow.

The plausibility band is doing its job at the other end. At `lambda >= 100` the
switching rate falls below one change every two years, the candidate is rejected
before it is scored, and its Sharpe is never recorded — which is why those cells
are empty in the trials log rather than merely poor.

## In-training Sharpe of the gated book falls by two thirds after 2010

| training window ends | 2002 | 2006 | 2010 | 2018 | 2026 |
| --- | --- | --- | --- | --- | --- |
| best training Sharpe | 0.56 | 0.42 | 0.11 | 0.22 | 0.22 |

These are **in-sample** figures on an expanding window, so they are the most
favourable numbers the model will ever produce. A gate that cannot reach 0.25
in-sample after 2010, against a base book at 0.63 out of sample, is not being
let down by the evaluation.

This is worth stating before any out-of-sample table is read, because it means a
weak out-of-sample result is not evidence of overfitting — there is nothing
fitted well enough to overfit.
