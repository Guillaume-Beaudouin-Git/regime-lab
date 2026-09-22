"""Cross-country macro momentum: nine questions before one number.

Hypothesis 1 fixed a sign map, built a book on it, and produced a single Sharpe
that turned out to hide sixteen separate relationships, only one of which was
statistically visible. This file inverts the order. Each theme is tested against
each asset class on its own, with a correction for the nine tests being run, and
a book is assembled from the pre-specified signs only afterwards.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from regime_lab.data import build_panel

from macro_momentum import config
from macro_momentum.countries import COUNTRIES, DURATION_10Y, FX_INVERTED, effective_breadth

warnings.filterwarnings("ignore")
RULE = "=" * 78
LOOKBACK = 12          # months: every theme is a one-year change
HORIZON = 1            # months held

#: Signs fixed from asset pricing, not fitted. Zero means the pre-specification
#: makes no claim for that pair and it is reported but not traded.
SIGNS = {
    "policy": {"bonds": -1, "fx": +1, "equities": -1},
    "slope":  {"bonds": -1, "fx":  0, "equities": +1},
    "currency": {"bonds": 0, "fx": +1, "equities": 0},
}


def monthly_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    rates = config.read("h2", "rates")
    markets = config.read("h2", "markets")
    end = min(rates["period"].max(), markets["period"].max())
    grid = pd.date_range("1990-01-01", end, freq="MS")
    return build_panel(rates, grid, max_staleness=pd.Timedelta(days=120)), build_panel(
        markets, grid, max_staleness=pd.Timedelta(days=45)
    )


def build_signals(rates: pd.DataFrame, markets: pd.DataFrame) -> dict[str, pd.DataFrame]:
    codes = [c for c in COUNTRIES if f"long_{c}" in rates and f"short_{c}" in rates]
    short = pd.DataFrame({c: rates[f"short_{c}"] for c in codes})
    long = pd.DataFrame({c: rates[f"long_{c}"] for c in codes})
    slope = long - short

    fx_level, fx_by_ccy = {}, {}
    for code in codes:
        ccy = COUNTRIES[code][1]
        if ccy == "USD":
            level = pd.Series(1.0, index=rates.index)
        elif f"fx_{ccy}" in markets:
            raw = markets[f"fx_{ccy}"]
            level = 1.0 / raw if ccy in FX_INVERTED else raw
        else:
            continue
        fx_level[code] = level
        fx_by_ccy.setdefault(ccy, level)
    fx = pd.DataFrame(fx_level)

    return {
        "policy": short.diff(LOOKBACK),
        "slope": slope.diff(LOOKBACK),
        "currency": np.log(fx).diff(LOOKBACK),
        "currency_by_ccy": np.log(pd.DataFrame(fx_by_ccy)).diff(LOOKBACK),
    }


def build_returns(rates: pd.DataFrame, markets: pd.DataFrame) -> dict[str, pd.DataFrame]:
    codes = [c for c in COUNTRIES if f"long_{c}" in rates]
    # Duration approximation: carry over the month, less the price move implied
    # by the yield change. A proxy, and the only free way to get nineteen
    # sovereign bond returns.
    y = pd.DataFrame({c: rates[f"long_{c}"] for c in codes}) / 100.0
    bonds = y.shift(1) / 12.0 - DURATION_10Y * y.diff()

    # One instrument per currency, not per country. Eight euro members share the
    # euro, so keying on the country code would put the same return series in the
    # cross-section eight times: the ranks would be identical, the apparent
    # breadth would read eighteen, and the effective breadth would be eleven.
    fx_cols = {}
    for code in codes:
        ccy = COUNTRIES[code][1]
        if ccy != "USD" and ccy not in fx_cols and f"fx_{ccy}" in markets:
            s = markets[f"fx_{ccy}"]
            fx_cols[ccy] = (1.0 / s if ccy in FX_INVERTED else s).pct_change()
    eq_cols = {c: markets[f"eq_{c}"].pct_change() for c in codes if f"eq_{c}" in markets}
    return {"bonds": bonds, "fx": pd.DataFrame(fx_cols), "equities": pd.DataFrame(eq_cols)}


def cross_sectional_ic(signal: pd.DataFrame, forward: pd.DataFrame) -> dict[str, float]:
    """Rank correlation between a demeaned signal and next month's return."""
    common = signal.columns.intersection(forward.columns)
    if len(common) < 4:
        return {"ic": np.nan, "t": np.nan, "n": 0, "breadth": len(common)}
    s = signal[common].rank(axis=1, pct=True)
    s = s.sub(s.mean(axis=1), axis=0)
    f = forward[common]
    per_date = s.corrwith(f, axis=1, method="spearman").dropna()
    if len(per_date) < 60:
        return {"ic": np.nan, "t": np.nan, "n": len(per_date), "breadth": len(common)}
    model = sm.OLS(per_date.to_numpy(), np.ones(len(per_date))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 12}
    )
    return {
        "ic": float(per_date.mean()),
        "t": float(model.tvalues[0]),
        "n": int(len(per_date)),
        "breadth": int(len(common)),
    }


def main() -> None:
    rates, markets = monthly_panel()
    signals = build_signals(rates, markets)
    returns = build_returns(rates, markets)
    forward = {k: v.shift(-HORIZON) for k, v in returns.items()}

    codes = [c for c in COUNTRIES if f"long_{c}" in rates]
    breadth = effective_breadth(codes)

    print(RULE)
    print(f"MACRO MOMENTUM TRANSVERSAL  {rates.index.min():%Y-%m} a {rates.index.max():%Y-%m}, "
          f"pas mensuel")
    print(RULE)
    print(f"\n   {breadth['sovereigns']} souverains, mais {breadth['currencies']} devises "
          f"distinctes : la zone euro compte pour une seule politique monetaire.")
    print(f"   instruments : {returns['bonds'].shape[1]} obligations, {returns['fx'].shape[1]} "
          f"devises, {returns['equities'].shape[1]} indices actions\n")

    # The currency cross-section is keyed by currency, so the signal facing it
    # must be too; every other pair is keyed by country.
    def signal_for(theme: str, cls: str) -> pd.DataFrame:
        if cls == "fx" and theme == "currency":
            return signals["currency_by_ccy"]
        if cls == "fx":
            by_ccy = {}
            for code in codes:
                ccy = COUNTRIES[code][1]
                if ccy != "USD" and ccy not in by_ccy and code in signals[theme]:
                    by_ccy[ccy] = signals[theme][code]
            return pd.DataFrame(by_ccy)
        return signals[theme]

    print("LES NEUF CELLULES, TESTEES SEPAREMENT\n")
    print(f"   {'theme':<11}" + "".join(f"{c:>22}" for c in ("bonds", "fx", "equities")))
    cells = {}
    for theme in ("policy", "slope", "currency"):
        row = f"   {theme:<11}"
        for cls, fwd in forward.items():
            r = cross_sectional_ic(signal_for(theme, cls), fwd)
            cells[(theme, cls)] = r
            if np.isfinite(r["t"]):
                row += f"{f'IC {r[chr(39)+chr(39)] if False else r["ic"]:+.4f}  t {r["t"]:+.2f}':>22}"
            else:
                row += f"{'—':>22}"
        print(row)

    live = {k: v for k, v in cells.items() if np.isfinite(v["t"])}
    strong = [k for k, v in live.items() if abs(v["t"]) > 1.96]
    print(f"\n   {len(strong)} cellules sur {len(live)} depassent |t| = 1.96 ; "
          f"le hasard en donne {0.05*len(live):.1f}")
    if live:
        sidak = 1 - (1 - 0.05) ** (1 / len(live))
        from scipy import stats as st
        crit = st.norm.ppf(1 - sidak / 2)
        survivors = [k for k, v in live.items() if abs(v["t"]) > crit]
        print(f"   seuil de Sidak pour {len(live)} tests : |t| > {crit:.2f}  ->  "
              f"{len(survivors)} survivant(s) {survivors if survivors else ''}")
    print("\n" + RULE)


if __name__ == "__main__":
    main()
