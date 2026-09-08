"""Momentum and volatility features from the cross-asset price panel."""

from __future__ import annotations

import numpy as np
import pandas as pd

EQUITY = "eq_us_large"


def _returns(panel: pd.DataFrame, column: str) -> pd.Series:
    return np.log(panel[column]).diff()


def momentum(panel: pd.DataFrame) -> pd.DataFrame:
    """Trend at several horizons, plus its breadth and its acceleration."""
    out = pd.DataFrame(index=panel.index)
    price = panel[EQUITY]

    for window in (21, 63, 252):
        out[f"mom_eq_{window}"] = price.pct_change(window)

    # Acceleration: a fast trend running ahead of a slow one is a different
    # state from the same fast trend inside a slow uptrend.
    out["mom_eq_accel"] = out["mom_eq_21"] - out["mom_eq_252"] / 12.0

    for column, tag in [("cmd_oil", "oil"), ("fx_dollar", "usd")]:
        if column in panel:
            out[f"mom_{tag}_63"] = panel[column].pct_change(63)

    trend_assets = [c for c in (EQUITY, "eq_us_small", "eq_jp", "cmd_oil") if c in panel]
    above = pd.concat(
        [(panel[c] > panel[c].rolling(200).mean()).astype(float) for c in trend_assets], axis=1
    )
    out["mom_breadth_200d"] = above.mean(axis=1)

    return out


def volatility(panel: pd.DataFrame) -> pd.DataFrame:
    """Realised volatility at several scales, and its shape.

    Includes the jump share and the downside/upside variance ratio, so that the
    volatility block carries information about the *character* of the variance
    and not only its level.
    """
    out = pd.DataFrame(index=panel.index)
    r = _returns(panel, EQUITY)

    for window in (5, 21, 63):
        out[f"vol_rv_{window}"] = r.rolling(window).std() * np.sqrt(252)

    out["vol_term"] = out["vol_rv_5"] / out["vol_rv_63"]
    out["vol_of_vol"] = out["vol_rv_21"].rolling(63).std()

    down = r.where(r < 0, 0.0)
    up = r.where(r > 0, 0.0)
    out["vol_semi_ratio"] = down.rolling(63).apply(lambda x: np.sum(x**2), raw=True) / up.rolling(
        63
    ).apply(lambda x: np.sum(x**2), raw=True).replace(0.0, np.nan)

    # Bipower variation is robust to jumps; the gap against realised variance is
    # therefore an estimate of the jump contribution (Barndorff-Nielsen-Shephard).
    bpv = (np.pi / 2) * (r.abs() * r.abs().shift(1)).rolling(21).sum()
    rvar = (r**2).rolling(21).sum()
    out["vol_jump_share"] = ((rvar - bpv) / rvar.replace(0.0, np.nan)).clip(lower=0.0)

    if "vol_vix" in panel:
        out["vol_vix"] = panel["vol_vix"]
        # Implied against realised: the premium paid for variance, not its level.
        out["vol_vrp"] = panel["vol_vix"] / 100.0 - out["vol_rv_21"]

    return out
