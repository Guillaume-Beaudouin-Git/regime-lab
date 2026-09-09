"""Family B: a Gaussian hidden Markov model, filtered rather than smoothed.

This is the reference the subject asks for, and it is treated fairly: the same
features, the same refit schedule, the same number of states as family A.

The one thing not taken from the library is the state probability. ``hmmlearn``
returns the *smoothed* posterior, conditioned on the whole sequence including
observations after the date in question. Trading on that is a look-ahead, and it
is the single most common defect in published regime-switching backtests. The
forward recursion below conditions on the past only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


class FilteredHMM:
    """Gaussian HMM whose state estimate uses no future information."""

    def __init__(
        self,
        *,
        n_states: int = 2,
        covariance_type: str = "diag",
        n_iter: int = 200,
        ordering: str = "volatility",
        random_state: int = 0,
    ) -> None:
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.ordering = ordering
        self.random_state = random_state
        self.model_: GaussianHMM | None = None
        self.features_: list[str] = []
        self.order_: np.ndarray | None = None
        self.state_vol_: dict[int, float] = {}

    def fit(self, features: pd.DataFrame, returns: pd.Series) -> None:
        frame = features.dropna()
        self.features_ = list(frame.columns)

        self.model_ = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.model_.fit(frame.to_numpy())

        # Order states so that 0 is the turbulent one after every refit. Ranking
        # by volatility rather than by return is the same convention used for the
        # jump model, and for the same reason: a return-based ranking inverted
        # that model's labels against every external reference.
        smoothed = self.model_.predict(frame.to_numpy())
        aligned = pd.Series(returns.reindex(frame.index).to_numpy())
        if self.ordering == "cumret":
            score = aligned.groupby(smoothed).mean().reindex(range(self.n_states)).fillna(0.0)
            self.order_ = np.argsort(score.to_numpy())
        else:
            vols = aligned.groupby(smoothed).std().reindex(range(self.n_states))
            self.order_ = np.argsort(-vols.fillna(vols.max()).to_numpy())
        rank = {int(s): int(r) for r, s in enumerate(self.order_)}
        self.state_vol_ = {
            rank[int(k)]: float(v) for k, v in aligned.groupby(smoothed).std().items()
        }

    def filtered_probabilities(self, features: pd.DataFrame) -> pd.DataFrame:
        """Forward recursion: ``P(state_t | observations up to t)``."""
        if self.model_ is None or self.order_ is None:
            raise RuntimeError("fit before predicting")

        frame = features.loc[:, self.features_].ffill().dropna()
        emission = self.model_._compute_log_likelihood(frame.to_numpy())
        emission = np.exp(emission - emission.max(axis=1, keepdims=True))

        transition = self.model_.transmat_
        alpha = self.model_.startprob_ * emission[0]
        alpha /= alpha.sum()

        out = np.empty_like(emission)
        out[0] = alpha
        for t in range(1, len(emission)):
            alpha = (alpha @ transition) * emission[t]
            total = alpha.sum()
            alpha = alpha / total if total > 0 else np.full(self.n_states, 1 / self.n_states)
            out[t] = alpha

        ranked = out[:, self.order_]
        return pd.DataFrame(ranked, index=frame.index).reindex(features.index)

    def predict_offline(self, features: pd.DataFrame) -> pd.Series:
        """Viterbi path over the whole block — the smoothed state, not tradable.

        This is what the library returns by default, and what most published
        regime backtests trade on. Keeping it here, clearly labelled and used
        only to price latency, is the honest way to show what it is worth.
        """
        if self.model_ is None or self.order_ is None:
            raise RuntimeError("fit before predicting")
        frame = features.loc[:, self.features_].ffill().dropna()
        path = self.model_.predict(frame.to_numpy())
        rank = {int(s): int(r) for r, s in enumerate(self.order_)}
        mapped = np.array([rank[int(v)] for v in path], dtype=float)
        return pd.Series(mapped, index=frame.index).reindex(features.index)

    def predict_online(self, features: pd.DataFrame) -> pd.Series:
        """Most likely state under the filtered distribution."""
        return self.filtered_probabilities(features).idxmax(axis=1).astype(float)
