"""Layer 2: does the state predict anything, and which moment?

Still no portfolio. The question here is whether the label carries information
about what happens next, measured directly on forward returns rather than
through a trading rule that would throw away everything except one bit.

The distinction that matters most is between the two moments. A state that
predicts forward **variance** but not forward **mean** is not useless — it is
useful for sizing and useless for timing, and saying so is a different
conclusion from "regimes do not work". Conflating them is how a real result gets
reported as a failure, and how a failure gets reported as a result.

Statistical power is the other reason to work at this layer. A difference in
conditional mean over several thousand sessions is estimated far more precisely
than a difference in portfolio Sharpe over a handful of independent episodes, so
a question that is hopeless at the portfolio layer can be answerable here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from regime_lab.analysis.bootstrap import stationary_indices


def forward_return(returns: pd.Series, horizon: int) -> pd.Series:
    """Sum of the next ``horizon`` returns, aligned to the decision date."""
    return returns.shift(-1).rolling(horizon).sum().shift(-(horizon - 1))


def forward_volatility(returns: pd.Series, horizon: int) -> pd.Series:
    """Realised volatility over the next ``horizon`` sessions, annualised."""
    forward = returns.shift(-1).rolling(horizon).std().shift(-(horizon - 1))
    return forward * np.sqrt(252)


def conditional_moments(
    states: pd.Series, returns: pd.Series, *, horizon: int = 21
) -> pd.DataFrame:
    """Forward mean, volatility and skew in each state."""
    frame = pd.concat(
        {
            "state": states,
            "fwd_ret": forward_return(returns, horizon),
            "fwd_vol": forward_volatility(returns, horizon),
        },
        axis=1,
    ).dropna()

    grouped = frame.groupby("state")["fwd_ret"]
    out = pd.DataFrame(
        {
            "n": grouped.size(),
            "mean_annual": grouped.mean() * 252 / horizon,
            "skew": grouped.skew(),
            "fwd_vol": frame.groupby("state")["fwd_vol"].mean(),
        }
    )
    return out


def mean_difference_test(
    states: pd.Series, returns: pd.Series, *, horizon: int = 21, draws: int = 2_000
) -> dict[str, float]:
    """Test whether the strong state's forward mean exceeds the weak state's.

    The t-statistic uses Newey-West standard errors with a lag matching the
    horizon, because overlapping forward windows are mechanically autocorrelated
    and an ordinary standard error would be far too small. The confidence
    interval is a stationary block bootstrap, for the same reason.
    """
    frame = pd.concat({"state": states, "fwd": forward_return(returns, horizon)}, axis=1).dropna()
    if frame["state"].nunique() < 2:
        return {"difference": np.nan, "t_hac": np.nan, "n": len(frame)}

    strong = (frame["state"] == frame["state"].max()).astype(float)
    design = sm.add_constant(strong.to_numpy())
    model = sm.OLS(frame["fwd"].to_numpy(), design).fit(
        cov_type="HAC", cov_kwds={"maxlags": horizon}
    )

    values = frame["fwd"].to_numpy()
    flags = strong.to_numpy().astype(bool)
    rng = np.random.default_rng(0)
    differences = np.empty(draws)
    for d in range(draws):
        idx = stationary_indices(len(values), horizon * 3, rng)
        v, f = values[idx], flags[idx]
        differences[d] = v[f].mean() - v[~f].mean() if f.any() and (~f).any() else np.nan

    scale = 252 / horizon
    return {
        "difference": float(model.params[1] * scale),
        "t_hac": float(model.tvalues[1]),
        "ci_low": float(np.nanpercentile(differences, 2.5) * scale),
        "ci_high": float(np.nanpercentile(differences, 97.5) * scale),
        "mde": float(2.802 * np.nanstd(differences, ddof=1) * scale),
        "n": int(len(frame)),
    }


def variance_versus_mean(
    states: pd.Series, returns: pd.Series, *, horizon: int = 21
) -> dict[str, float]:
    """Does the state predict the same direction in both moments?

    Reports the spread in forward mean and in forward volatility between the
    strong and weak state. A state that separates volatility but not mean is a
    sizing signal, not a timing signal, and the two conclusions are different.
    """
    moments = conditional_moments(states, returns, horizon=horizon)
    if len(moments) < 2:
        return {"mean_spread": np.nan, "vol_spread": np.nan, "aligned": np.nan}

    strong, weak = moments.index.max(), moments.index.min()
    mean_spread = float(moments.loc[strong, "mean_annual"] - moments.loc[weak, "mean_annual"])
    vol_spread = float(moments.loc[strong, "fwd_vol"] - moments.loc[weak, "fwd_vol"])
    return {
        "mean_spread": mean_spread,
        "vol_spread": vol_spread,
        # Useful for timing only if the state the model calls strong really does
        # earn more, not merely shake less.
        "aligned": bool(mean_spread > 0 and vol_spread < 0),
    }


def incremental_information(
    states: pd.Series, returns: pd.Series, realised_vol: pd.Series, *, horizon: int = 21
) -> dict[str, float]:
    """R-squared of the state over and above a volatility quantile.

    This is the placebo test of the charter, run at the level of prediction
    instead of the level of a portfolio. If the state adds nothing once a plain
    volatility measure is in the regression, then whatever it found was already
    in the volatility.
    """
    frame = pd.concat(
        {
            "state": states,
            "vol_rank": realised_vol.rank(pct=True),
            "fwd": forward_return(returns, horizon),
        },
        axis=1,
    ).dropna()
    if len(frame) < 500 or frame["state"].nunique() < 2:
        return {"r2_vol": np.nan, "r2_both": np.nan, "incremental": np.nan}

    y = frame["fwd"].to_numpy()
    vol_only = sm.OLS(y, sm.add_constant(frame[["vol_rank"]].to_numpy())).fit()
    both = sm.OLS(y, sm.add_constant(frame[["vol_rank", "state"]].to_numpy())).fit(
        cov_type="HAC", cov_kwds={"maxlags": horizon}
    )
    return {
        "r2_vol": float(vol_only.rsquared),
        "r2_both": float(both.rsquared),
        "incremental": float(both.rsquared - vol_only.rsquared),
        "t_state": float(both.tvalues[2]),
        "n": int(len(frame)),
    }


def volatility_difference_test(
    states: pd.Series, returns: pd.Series, *, horizon: int = 21, draws: int = 2_000
) -> dict[str, float]:
    """Test the state's separation of forward *variance*, with its own power.

    The mean test is underpowered here and the variance test need not be: a
    volatility spread is estimated far more precisely than a mean spread on the
    same data, which is the statistical reason a regime signal can be usable for
    sizing while being useless for timing. Reporting only the mean test would
    have hidden that.
    """
    frame = pd.concat(
        {"state": states, "fwd_vol": forward_volatility(returns, horizon)}, axis=1
    ).dropna()
    if frame["state"].nunique() < 2:
        return {"difference": np.nan, "t_hac": np.nan, "mde": np.nan, "n": len(frame)}

    calm = (frame["state"] == frame["state"].max()).astype(float)
    design = sm.add_constant(calm.to_numpy())
    model = sm.OLS(frame["fwd_vol"].to_numpy(), design).fit(
        cov_type="HAC", cov_kwds={"maxlags": horizon}
    )

    values = frame["fwd_vol"].to_numpy()
    flags = calm.to_numpy().astype(bool)
    rng = np.random.default_rng(0)
    differences = np.empty(draws)
    for d in range(draws):
        idx = stationary_indices(len(values), horizon * 3, rng)
        v, f = values[idx], flags[idx]
        differences[d] = v[f].mean() - v[~f].mean() if f.any() and (~f).any() else np.nan

    return {
        "difference": float(model.params[1]),
        "t_hac": float(model.tvalues[1]),
        "mde": float(2.802 * np.nanstd(differences, ddof=1)),
        "detected": bool(abs(model.params[1]) > 2.802 * np.nanstd(differences, ddof=1)),
        "n": int(len(frame)),
    }
