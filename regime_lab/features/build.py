"""Assemble the feature matrix from the point-in-time store."""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_lab.data import build_panel, store
from regime_lab.data.universe import MACRO_LAGGED, MACRO_VINTAGED
from regime_lab.features import crosssection, macro, market
from regime_lab.features.standardise import expanding_zscore, winsorise

#: Which module produced a feature, read off its name prefix.
FAMILIES = {
    "mom_": "momentum",
    "vol_": "volatility",
    "xs_": "cross-section",
    "mac_": "macro",
    "cre_": "credit",
    "fin_": "conditions",
    "rat_": "rates",
}


def family_of(name: str) -> str:
    for prefix, family in FAMILIES.items():
        if name.startswith(prefix):
            return family
    return "other"


def wide(source: str, name: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Load one stored frame as a wide, point-in-time correct panel."""
    return build_panel(store.read(source, name), dates)


def raw_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build every feature, before standardisation."""
    price_panel = pd.concat(
        [wide("prices", "cross_asset", dates), wide("prices", "fred_daily", dates)], axis=1
    )
    macro_frame = store.load_many([("macro", n) for n in list(MACRO_VINTAGED) + list(MACRO_LAGGED)])
    macro_panel = build_panel(macro_frame, dates, max_staleness=pd.Timedelta(days=120))

    industries = wide("panels", "industry_49", dates)
    size_bm = wide("panels", "size_bm_25", dates)
    factors = wide("panels", "factors_5", dates)

    blocks = [
        market.momentum(price_panel),
        market.volatility(price_panel),
        crosssection.build(industries, size_bm, factors),
        macro.build(macro_panel),
    ]
    return pd.concat(blocks, axis=1)


def build(dates: pd.DatetimeIndex, *, min_periods: int = 252) -> pd.DataFrame:
    """Build the standardised feature matrix a model may consume.

    Standardisation is expanding, so a value is scaled by the distribution known
    when it arrived, and winsorisation is applied after scaling rather than
    inside any feature's constructor.
    """
    features = raw_features(dates)
    return winsorise(expanding_zscore(features, min_periods=min_periods))


def volatility_r2(features: pd.DataFrame, panel: pd.DataFrame) -> pd.Series:
    """R-squared of each feature against contemporaneous realised volatility.

    The charter makes this the admissibility test for the placebo comparison: if
    every feature is a restatement of volatility, then beating a volatility
    quantile was never possible and the comparison is empty. Reported per
    feature so the claim can be checked rather than asserted.
    """
    returns = np.log(panel["eq_us_large"]).diff()
    realised = (returns.rolling(20).std() * np.sqrt(252)).reindex(features.index)

    out = {}
    for name in features.columns:
        pair = pd.concat([features[name], realised], axis=1).dropna()
        if len(pair) < 500:
            continue
        out[name] = float(np.corrcoef(pair.iloc[:, 0], pair.iloc[:, 1])[0, 1] ** 2)
    return pd.Series(out).sort_values(ascending=False)
