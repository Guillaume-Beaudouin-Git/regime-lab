"""Family C: predict the quantity directly, with no latent state at all.

This is the family that answers the objection the whole project starts from. If
a gradient boosting model trained on the same features predicts forward
volatility well enough to drive the same on/off decision, then the latent regime
layer is an expensive intermediary and should be reported as one.

The comparison is deliberately awkward for family C in one respect and generous
in another, and both are stated rather than hidden. Generous: it sees labels,
which families A and B never do, so on prediction quality it should win. Awkward:
its advantage only counts if it survives the conversion into the *same*
position rule, because a better forecast that does not change the decision is
not worth anything.

HAR-RV is included because it is the baseline that machine learning models of
realised volatility usually fail to beat, and omitting it would flatter the
gradient boosting result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression


def forward_volatility(returns: pd.Series, *, horizon: int = 21) -> pd.Series:
    """Realised volatility over the ``horizon`` sessions *after* each date."""
    forward = returns.shift(-1).rolling(horizon).std().shift(-(horizon - 1))
    return (forward * np.sqrt(252)).rename(f"fwd_vol_{horizon}")


def har_features(returns: pd.Series) -> pd.DataFrame:
    """The three HAR components: daily, weekly and monthly realised volatility."""
    rv = returns.rolling(1).std() * np.sqrt(252)
    return pd.DataFrame(
        {
            "har_d": (returns.abs() * np.sqrt(252)).rolling(1).mean(),
            "har_w": (returns.rolling(5).std() * np.sqrt(252)),
            "har_m": (returns.rolling(21).std() * np.sqrt(252)),
        },
        index=rv.index,
    )


class ForwardVolModel:
    """Predict forward volatility, then threshold it into the same on/off state.

    The threshold is the median predicted volatility **on the training window**,
    so the decision boundary never sees the evaluation sample.
    """

    def __init__(self, *, horizon: int = 21, linear: bool = False, seed: int = 0) -> None:
        self.horizon = horizon
        self.linear = linear
        self.seed = seed
        self.model_: object | None = None
        self.features_: list[str] = []
        self.threshold_: float | None = None
        self.state_vol_: dict[int, float] = {}

    def fit(self, features: pd.DataFrame, returns: pd.Series) -> None:
        target = forward_volatility(returns, horizon=self.horizon)
        frame = pd.concat([features, target], axis=1).dropna()
        if frame.empty:
            raise RuntimeError("no overlapping training rows")

        x = frame.iloc[:, :-1]
        y = frame.iloc[:, -1]
        self.features_ = list(x.columns)

        if self.linear:
            self.model_ = LinearRegression().fit(x, y)
        else:
            self.model_ = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=50,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.seed,
                verbose=-1,
            ).fit(x, y)

        self.threshold_ = float(np.median(self.model_.predict(x)))

        # Realised volatility per state on the training window, so this family
        # can feed the sizing rule like the state-space ones. Without it the
        # supervised rows came out empty, which read as a missing result rather
        # than as a missing attribute.
        state = (self.model_.predict(x) < self.threshold_).astype(int)
        aligned = returns.reindex(x.index)
        self.state_vol_ = {
            int(k): float(v) for k, v in pd.Series(aligned.to_numpy()).groupby(state).std().items()
        }

    def predict_volatility(self, features: pd.DataFrame) -> pd.Series:
        """Predicted forward volatility, for scoring against the realised value."""
        if self.model_ is None:
            raise RuntimeError("fit before predicting")
        frame = features.loc[:, self.features_].ffill().dropna()
        return pd.Series(self.model_.predict(frame), index=frame.index)

    def predict_offline(self, features: pd.DataFrame) -> pd.Series:
        """Identical to the online state, by construction.

        A pointwise regression has no sequential smoothing to exploit, so this
        family pays no latency penalty at all. That is a real advantage over the
        state-space families and it is worth showing rather than assuming.
        """
        return self.predict_online(features)

    def predict_online(self, features: pd.DataFrame) -> pd.Series:
        """State 1 when predicted volatility is below the training median.

        Low predicted volatility is mapped to the strong state so that the
        output carries the same meaning as families A and B, where state 1 is
        the one with the better training return. No look-ahead is involved: the
        threshold comes from training data and the features from the past.
        """
        if self.model_ is None or self.threshold_ is None:
            raise RuntimeError("fit before predicting")
        frame = features.loc[:, self.features_].ffill().dropna()
        predicted = pd.Series(self.model_.predict(frame), index=frame.index)
        return (predicted < self.threshold_).astype(float).reindex(features.index)
