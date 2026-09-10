"""The frozen sign map, transcribed from docs/PRESPEC.md and never fitted.

This table is the study's most dangerous object. A practitioner reproduced a
published index methodology of exactly this shape and compared it against
160,000 randomly drawn alternatives: the official map sat at the 98th percentile
of that distribution, which is where you land by trying many maps, not by
understanding one.

So the map is written from textbook reasoning, frozen with a content hash before
any return was computed, and tested against random maps rather than against
zero. If the real map cannot beat a few hundred random ones, it was a fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: ``{theme: {asset class: sign}}``. Read as: when the theme rises, hold this
#: sign. Identical to the table in docs/PRESPEC.md; the test suite checks that.
SIGN_MAP: dict[str, dict[str, int]] = {
    "growth":    {"equities": +1, "bonds": -1, "commodities": +1, "dollar": +1},
    "inflation": {"equities": -1, "bonds": -1, "commodities": +1, "dollar": -1},
    "policy":    {"equities": -1, "bonds": -1, "commodities": -1, "dollar": +1},
    "risk":      {"equities": -1, "bonds": +1, "commodities": -1, "dollar": +1},
}

#: Cells the pre-specification flags as genuinely contestable. They are left as
#: written; naming them is what stops a later reader from assuming every sign
#: was equally obvious.
CONTESTED = {("growth", "dollar"), ("policy", "commodities")}


def asset_signal(themes: pd.DataFrame, asset_class: str) -> pd.Series:
    """Combine the four themes into one signal for an asset class.

    Equal weight across themes. Weighting them would be four more parameters
    fitted on the same sample, and the pre-specification does not grant them.
    """
    if asset_class not in next(iter(SIGN_MAP.values())):
        raise KeyError(f"unknown asset class {asset_class!r}")
    parts = [themes[t] * SIGN_MAP[t][asset_class] for t in SIGN_MAP if t in themes]
    return pd.concat(parts, axis=1).mean(axis=1).rename(asset_class)


def random_map(rng: np.random.Generator) -> dict[str, dict[str, int]]:
    """One randomly signed map, for the placebo that the real map must beat."""
    classes = list(next(iter(SIGN_MAP.values())))
    return {t: {c: int(rng.choice([-1, 1])) for c in classes} for t in SIGN_MAP}


def apply_map(
    themes: pd.DataFrame, mapping: dict[str, dict[str, int]], asset_class: str
) -> pd.Series:
    """``asset_signal`` against an arbitrary map, used by the placebo."""
    parts = [themes[t] * mapping[t][asset_class] for t in mapping if t in themes]
    return pd.concat(parts, axis=1).mean(axis=1).rename(asset_class)
