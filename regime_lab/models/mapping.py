"""The single rule that turns a state into a position, fixed before any fit.

Every family produces a different object — a partition, a filtered probability,
a predicted volatility — and they are only comparable if one declared rule
converts each into the same thing. Choosing that rule after seeing the states
would select on the sign of the mean/variance relation per state, which is the
quantity the study is testing.

**The rule.** Hold the base strategy in full while the model is in its stronger
state, and hold nothing otherwise. Which state is the stronger one is decided by
the ordering imposed at fit time, on the training window alone.

That is the plain reading of the "on/off" question, and it is deliberately the
crudest defensible choice: a rule with dials would put the study's degrees of
freedom in the place hardest to audit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def state_to_position(states: pd.Series, *, n_states: int = 2) -> pd.Series:
    """One when the model sits in its strongest state, zero otherwise.

    States arrive ordered by the return earned in them on training data, so the
    strongest is the last index. The signal is lagged one session: a state read
    from today's close is traded tomorrow.
    """
    on = (states == float(n_states - 1)).astype(float)
    return on.shift(1).rename("position")


def apply_overlay(returns: pd.Series, position: pd.Series) -> pd.Series:
    """Return the base strategy gated by ``position``."""
    aligned = position.reindex(returns.index).fillna(0.0)
    return (returns * aligned).rename("overlay")


def turnover(position: pd.Series) -> float:
    """Average absolute change in exposure per session."""
    return float(position.diff().abs().mean())


def summary(returns: pd.Series, *, periods: int = 252) -> dict[str, float]:
    """Annualised Sharpe, volatility, and the worst drawdown."""
    clean = returns.dropna()
    if clean.std(ddof=1) == 0:
        return {"sharpe": np.nan, "vol": 0.0, "max_drawdown": 0.0, "exposure": np.nan}
    curve = (1 + clean).cumprod()
    drawdown = (curve / curve.cummax() - 1).min()
    return {
        "sharpe": float(clean.mean() / clean.std(ddof=1) * np.sqrt(periods)),
        "vol": float(clean.std(ddof=1) * np.sqrt(periods)),
        "max_drawdown": float(drawdown),
        "exposure": float((clean != 0).mean()),
    }
