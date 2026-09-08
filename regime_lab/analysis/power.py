"""How large an effect this sample could detect, computed before any model.

Thirty-six years of daily data look like nine thousand observations, but the
number of *independent stress episodes* is closer to fourteen. That is the
sample size that governs any claim about conditioning on regimes. If the
minimum detectable effect exceeds any plausible effect, the honest result is
that the question is underpowered — which is a finding, not a failure, and one
that has to be established before modelling rather than after.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from regime_lab.analysis.bootstrap import paired_sharpe_difference


@dataclass(frozen=True)
class PowerResult:
    """Outcome of a minimum-detectable-effect calculation."""

    se: float
    mde: float
    observed: float
    mean_block: int
    draws: int
    alpha: float
    power: float

    def verdict(self, plausible: float) -> str:
        """State whether an effect of size ``plausible`` is detectable here."""
        if self.mde > plausible:
            return (
                f"UNDERPOWERED: detecting a Sharpe difference needs {self.mde:.2f}, "
                f"above the {plausible:.2f} considered plausible"
            )
        return f"powered: {self.mde:.2f} detectable, below the {plausible:.2f} plausible effect"


def minimum_detectable_sharpe_difference(
    a: np.ndarray,
    b: np.ndarray,
    *,
    mean_block: int = 63,
    draws: int = 2_000,
    alpha: float = 0.05,
    power: float = 0.80,
    seed: int = 0,
) -> PowerResult:
    """Minimum detectable difference in annualised Sharpe between two legs.

    ``mean_block`` should approximate the length of a stress episode, since that
    is the unit of independent information. Sixty-three sessions is one quarter.
    """
    distribution = paired_sharpe_difference(a, b, mean_block=mean_block, draws=draws, seed=seed)
    se = float(distribution.std(ddof=1))
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)

    scale = np.sqrt(252)
    observed = float(a.mean() / a.std(ddof=1) * scale - b.mean() / b.std(ddof=1) * scale)

    return PowerResult(
        se=se,
        mde=float(z * se),
        observed=observed,
        mean_block=mean_block,
        draws=draws,
        alpha=alpha,
        power=power,
    )
