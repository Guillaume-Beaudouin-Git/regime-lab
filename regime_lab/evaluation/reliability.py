"""Layer 1: is the classification itself reliable?

No portfolio appears in this module, deliberately. Converting a state into a
position and reading the result through a Sharpe ratio is the noisiest possible
lens on a classifier, and it answers a different question. Here the object of
study is the label sequence.

Every metric below is chance-corrected or compared against a null, because the
usual regime-classification metrics have well-known ways of looking good for
free:

* accuracy against NBER recessions is 92% for a model that always says
  "expansion", since recessions are 8% of months — so balanced accuracy and
  Cohen's kappa are used instead of accuracy;
* persistence is bought directly by the jump penalty and by the transition
  matrix, so observed durations are compared with what an independent draw of
  the same marginal frequency would produce;
* agreement between two models that both track volatility is not evidence that
  either found a regime, so agreement is read alongside what each shares with a
  volatility quantile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, cohen_kappa_score


def durations(states: pd.Series) -> pd.DataFrame:
    """Length of every uninterrupted run, by state."""
    clean = states.dropna()
    if clean.empty:
        return pd.DataFrame(columns=["state", "length"])
    change = clean.ne(clean.shift()).cumsum()
    runs = clean.groupby(change).agg(state=("first"), length=("size"))
    return runs.reset_index(drop=True)


def persistence_vs_chance(states: pd.Series) -> dict[str, float]:
    """Observed mean run length against an independent draw of the same marginal.

    An independent sequence with probability ``p`` of a state produces runs of
    mean ``1 / (1 - p)``. The ratio says how much of the persistence is more
    than the marginal frequency already implies.
    """
    clean = states.dropna()
    runs = durations(clean)
    if runs.empty:
        return {"observed": np.nan, "chance": np.nan, "ratio": np.nan}

    observed = float(runs["length"].mean())
    frequencies = clean.value_counts(normalize=True)
    chance = float(sum(freq * (1 / (1 - freq)) for freq in frequencies if freq < 1))
    return {
        "observed": observed,
        "chance": chance,
        "ratio": observed / chance if chance > 0 else np.nan,
    }


def agreement(a: pd.Series, b: pd.Series) -> dict[str, float]:
    """How far two label sequences agree, beyond chance."""
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 100:
        return {"raw": np.nan, "kappa": np.nan, "ari": np.nan, "n": len(pair)}
    x = pair.iloc[:, 0].astype(int)
    y = pair.iloc[:, 1].astype(int)
    return {
        "raw": float((x == y).mean()),
        "kappa": float(cohen_kappa_score(x, y)),
        "ari": float(adjusted_rand_score(x, y)),
        "n": int(len(pair)),
    }


def external_validation(states: pd.Series, reference: pd.Series) -> dict[str, float]:
    """Score the weak state against an externally defined stress label.

    ``reference`` must come from outside this project — NBER dates, or a
    drawdown rule written before the states were seen. Scoring a classifier
    against labels derived the way the classifier works is circular, and it is
    how this kind of result is usually made to look good.
    """
    pair = pd.concat([states.rename("state"), reference.rename("ref")], axis=1).dropna()
    if len(pair) < 100:
        return {"balanced_accuracy": np.nan, "kappa": np.nan, "majority": np.nan}

    weak = (pair["state"] == pair["state"].min()).astype(int)
    truth = pair["ref"].astype(int)
    majority = float(max(truth.mean(), 1 - truth.mean()))

    return {
        "balanced_accuracy": float(balanced_accuracy_score(truth, weak)),
        "kappa": float(cohen_kappa_score(truth, weak)),
        "majority": majority,
        "raw_accuracy": float((truth == weak).mean()),
        "base_rate": float(truth.mean()),
        "n": int(len(pair)),
    }


def drawdown_reference(prices: pd.Series, *, threshold: float = 0.10) -> pd.Series:
    """A stress label written as a rule, not drawn by hand.

    One whenever the market sits more than ``threshold`` below its running
    twelve-month high. Pre-specified, mechanical, and independent of any model
    in this repository — unlike the hand-drawn episode bands used to illustrate
    the charter, which are not admissible as a scoring target.
    """
    drawdown = prices / prices.rolling(252).max() - 1.0
    return (drawdown < -threshold).astype(float).rename("stress")


def hindsight_gap(online: pd.Series, offline: pd.Series) -> dict[str, float]:
    """How much of the labelling only becomes clear afterwards.

    The share of sessions on which knowing the rest of the block would have
    changed the label. A classifier whose labels move a lot under hindsight is
    not unreliable in itself — it is telling you that regimes are recognisable
    late, which is the thing a real-time user cannot have.
    """
    pair = pd.concat([online.rename("on"), offline.rename("off")], axis=1).dropna()
    if pair.empty:
        return {"changed_share": np.nan, "kappa": np.nan}
    return {
        "changed_share": float((pair["on"] != pair["off"]).mean()),
        "kappa": float(cohen_kappa_score(pair["on"].astype(int), pair["off"].astype(int))),
        "n": int(len(pair)),
    }
