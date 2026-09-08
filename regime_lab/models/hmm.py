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
        random_state: int = 0,
    ) -> None:
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.model_: GaussianHMM | None = None
        self.features_: list[str] = []
        self.order_: np.ndarray | None = None

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

        # Order states by the mean return earned in them on the training window,
        # so that state 0 means the same thing after every refit. Ordering is the
        # actual fix for label switching; the deeper instabilities of the EM
        # algorithm — local optima, drifting parameters — are what the stability
        # test measures.
        smoothed = self.model_.predict(frame.to_numpy())
        means = pd.Series(returns.reindex(frame.index).to_numpy()).groupby(smoothed).mean()
        self.order_ = np.argsort(means.reindex(range(self.n_states)).fillna(0.0).to_numpy())

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

    def predict_online(self, features: pd.DataFrame) -> pd.Series:
        """Most likely state under the filtered distribution."""
        return self.filtered_probabilities(features).idxmax(axis=1).astype(float)
