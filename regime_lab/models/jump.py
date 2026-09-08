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
        random_state: int = 0,
    ) -> None:
        self.n_states = n_states
        self.jump_penalty = jump_penalty
        self.continuous = continuous
        self.max_features = max_features
        self.random_state = random_state
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

        # sort_by="cumret" orders states by the cumulative return earned in them,
        # on the training window only. This fixes label switching deterministically:
        # state 0 is always the weaker one, so states are comparable across refits.
        self.model_.fit(frame, ret_ser=returns.reindex(frame.index), sort_by="cumret")

    def predict_online(self, features: pd.DataFrame) -> pd.Series:
        """Filter forward: each row is assigned using only rows up to it."""
        if self.model_ is None:
            raise RuntimeError("fit before predicting")
        frame = features.loc[:, self.features_].ffill().dropna()
        predicted = self.model_.predict_online(frame)
        return pd.Series(np.asarray(predicted, dtype=float), index=frame.index).reindex(
            features.index
        )
