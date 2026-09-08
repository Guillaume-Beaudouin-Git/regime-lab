"""Macro, credit and rate features.

Every macro feature is a **rate of change**, never a level. Industrial
production was published at 114.23 for January 2008 and reads 102.24 today, and
almost all of that gap is index rebasing rather than revision: a level is not
comparable across vintages, while a growth rate is. Measured on the growth rate,
the true revision for that month is 0.13 of a percentage point.
"""

from __future__ import annotations

import pandas as pd


def build(panel: pd.DataFrame) -> pd.DataFrame:
    """Assemble the macro, credit and rate blocks from the point-in-time panel."""
    out = pd.DataFrame(index=panel.index)

    # --- activity and prices, as growth rates -------------------------------
    for column, tag in [
        ("macro_indpro", "indpro"),
        ("macro_payems", "payems"),
        ("macro_cpi", "cpi"),
    ]:
        if column in panel:
            out[f"mac_{tag}_yoy"] = panel[column].pct_change(252)
            out[f"mac_{tag}_3m"] = panel[column].pct_change(63)

    if "macro_unrate" in panel:
        unemployment = panel["macro_unrate"]
        out["mac_unrate_chg12m"] = unemployment.diff(252)
        # Sahm-style: the three-month average against its own twelve-month low.
        out["mac_unrate_sahm"] = unemployment.rolling(63).mean() - unemployment.rolling(252).min()

    if "macro_claims" in panel:
        claims = panel["macro_claims"].rolling(20).mean()
        out["mac_claims_yoy"] = claims.pct_change(252)

    # --- credit -------------------------------------------------------------
    if "fin_baa_spread" in panel:
        out["cre_baa"] = panel["fin_baa_spread"]
        out["cre_baa_chg63"] = panel["fin_baa_spread"].diff(63)
    if {"fin_baa_spread", "fin_aaa_spread"} <= set(panel.columns):
        # Baa minus Aaa isolates default compensation from the level of yields.
        out["cre_quality"] = panel["fin_baa_spread"] - panel["fin_aaa_spread"]
        out["cre_quality_chg63"] = out["cre_quality"].diff(63)

    # --- financial conditions ----------------------------------------------
    for column, tag in [("fin_nfci", "nfci"), ("fin_stlfsi", "stlfsi")]:
        if column in panel:
            out[f"fin_{tag}"] = panel[column]
            out[f"fin_{tag}_chg13w"] = panel[column].diff(65)

    # --- rates --------------------------------------------------------------
    for column, tag in [("rate_curve_10y2y", "slope_10y2y"), ("rate_curve_10y3m", "slope_10y3m")]:
        if column in panel:
            out[f"rat_{tag}"] = panel[column]
            out[f"rat_{tag}_chg63"] = panel[column].diff(63)

    if "rate_cash_3m" in panel:
        out["rat_cash_chg12m"] = panel["rate_cash_3m"].diff(252)

    return out
