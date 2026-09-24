"""Portfolio construction under a growth x inflation partition (the Bridgewater study).

The pre-registration is `docs/PRESPEC_BRIDGEWATER.md` (draft until locked). Modules:

``quadrant``
    The partition itself: consensus surprises from the Philadelphia Fed Survey of
    Professional Forecasters, their point-in-time availability stamps, the daily
    label series, label-only statistics, and the market-implied candidates of B1.
``sleeves``
    The five risk sleeves over 30 instruments and the state-blind book: excess
    returns with the ETFs funded, within-sleeve weights, a daily volatility target,
    costs on held weights. It reads no label.
``evaluate``
    The tree's statistics and placebos: D and the balanced leg (level A), P1, P2, the
    volatility witness, B2's map and P3, level C's cadence books, and the null
    distributions from which the power is measured before the lock.
"""
