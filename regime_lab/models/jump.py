"""Family A: the statistical jump model.

A k-means style partition of standardised features, with a penalty ``lambda``
charged for every change of state. The penalty is what separates it from
clustering: it buys persistence directly, and persistence is turnover, and
turnover is cost.

One caveat is worth stating rather than glossing. The quadratic loss the model
minimises is the log-likelihood of a spherical Gaussian with a common variance
across states. The model is distribution-free in the sense that it maximises no
likelihood, not in the sense that it assumes nothing.

Wrapper around the authors' reference implementation (Nystrup, Kolm and
Lindström; Aydınhan, Kolm, Mulvey and Shu), so the results are comparable with
the published ones rather than with a private reimplementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
from jumpmodels.sparse_jump import SparseJumpModel


class JumpRegimes:
    """Discrete or continuous jump model under the common protocol."""

    def __init__(
        self,
        *,
        n_states: int = 2,
        jump_penalty: float = 50.0,
        continuous: bool = False,
        max_features: float | None = None,
        ordering: str = "volatility",
        random_state: int = 0,
    ) -> None:
        self.n_states = n_states
        self.jump_penalty = jump_penalty
        self.continuous = continuous
        self.max_features = max_features
        self.ordering = ordering
        self.random_state = random_state
        self.order_: np.ndarray | None = None
        self.state_vol_: dict[int, float] = {}
        self.model_: JumpModel | SparseJumpModel | None = None
        self.features_: list[str] = []

    def fit(self, features: pd.DataFrame, returns: pd.Series) -> None:
        frame = features.dropna()
        self.features_ = list(frame.columns)

        if self.max_features is not None:
            self.model_ = SparseJumpModel(
                n_components=self.n_states,
                max_feats=self.max_features,
                jump_penalty=self.jump_penalty,
                cont=self.continuous,
                random_state=self.random_state,
            )
        else:
            self.model_ = JumpModel(
                n_components=self.n_states,
                jump_penalty=self.jump_penalty,
                cont=self.continuous,
                random_state=self.random_state,
            )

        # Label switching has to be resolved somehow, and the choice matters more
        # than it looks. The reference implementation's "cumret" ranks states by
        # the cumulative return earned in each on the training window; measured
        # against NBER dates that convention put this model's labels exactly
        # backwards, because violent recoveries fall inside the volatile state and
        # can make it the higher-earning one. Ranking by volatility instead never
        # touches returns and cannot flip.
        sort_by = "cumret" if self.ordering == "cumret" else None
        self.model_.fit(frame, ret_ser=returns.reindex(frame.index), sort_by=sort_by)

        raw = np.asarray(self.model_.predict(frame), dtype=int)
        aligned = returns.reindex(frame.index)
        if self.ordering == "cumret":
            self.order_ = np.arange(self.n_states)
        else:
            vols = pd.Series(aligned.to_numpy()).groupby(raw).std()
            vols = vols.reindex(range(self.n_states)).fillna(vols.max())
            # Descending volatility: state 0 is the turbulent one, the last is calm.
            self.order_ = np.argsort(-vols.to_numpy())
        rank = {int(state): int(r) for r, state in enumerate(self.order_)}
        self.state_vol_ = {
            rank[int(k)]: float(v)
            for k, v in pd.Series(aligned.to_numpy()).groupby(raw).std().items()
        }

    def predict_offline(self, features: pd.DataFrame) -> pd.Series:
        """Assign states using the whole block, future included.

        Not tradable, and not meant to be: the gap against ``predict_online`` is
        the price of having to recognise a regime as it happens rather than
        afterwards.
        """
        if self.model_ is None:
            raise RuntimeError("fit before predicting")
        frame = features.loc[:, self.features_].ffill().dropna()
        predicted = self._rank(self.model_.predict(frame))
        return pd.Series(predicted, index=frame.index).reindex(features.index)

    def predict_online(self, features: pd.DataFrame) -> pd.Series:
        """Filter forward: each row is assigned using only rows up to it."""
        if self.model_ is None:
            raise RuntimeError("fit before predicting")
        frame = features.loc[:, self.features_].ffill().dropna()
        predicted = self._rank(self.model_.predict_online(frame))
        return pd.Series(predicted, index=frame.index).reindex(features.index)

    def _rank(self, raw) -> np.ndarray:
        """Apply the fitted state ordering to a raw label sequence."""
        if self.order_ is None:
            raise RuntimeError("fit before predicting")
        rank = {int(state): int(r) for r, state in enumerate(self.order_)}
        return np.array([rank[int(v)] for v in np.asarray(raw)], dtype=float)
