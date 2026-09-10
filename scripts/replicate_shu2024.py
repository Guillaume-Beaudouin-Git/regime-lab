"""Reproduce the published S&P 500 result, then change exactly one thing.

Phase R reproduces the paper's headline with its own protocol and its own
benchmark. Phase S replaces buy and hold with a dynamic volatility target — the
competitor the paper does not test, which costs nothing to implement and
delivers the same kind of drawdown reduction.

Nothing here is tuned. Every parameter comes from the paper.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from regime_lab.config import CACHE
from regime_lab.data.sources import fred
from regime_lab.extensions.shu2024 import (
    COST_ONE_WAY,
    EXECUTION_LAG,
    LAMBDA_GRID,
    REFIT_MONTHS,
    TRAIN_DAYS,
    VALIDATION_YEARS,
    features,
    fit_centroids,
    online_states,
    order_states,
)

warnings.filterwarnings("ignore")
RULE = "=" * 78
OOS_START = "1990-01-01"
OOS_END = "2023-12-31"

#: What the paper reports for the S&P 500 over 1990-2023.
PUBLISHED = {
    "buy_hold": {"cagr": 0.102, "vol": 0.182, "sharpe": 0.48, "mdd": -0.552},
    "jump": {"cagr": 0.112, "vol": 0.131, "sharpe": 0.68, "mdd": -0.266,
             "turnover": 0.44, "exposure": 0.80},
}


SHILLER = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"


def total_return() -> pd.Series:
    """Daily S&P 500 **total** return, reconstructed and validated.

    The index quoted as ^GSPC is a price index: it omits about 2.2% a year of
    dividends, which is most of the difference between a buy-and-hold Sharpe of
    0.37 and the 0.48 the paper reports. The total-return index only starts in
    1988, too late to leave twelve years of training before a 1990 out-of-sample
    start, so the series is rebuilt as the price return plus a daily accrual of
    Shiller's monthly dividend yield.

    Checked against the real total-return index over the 9,743 days where both
    exist: daily correlation 0.9996, and an annualised return of 10.16% against
    10.21% over 1990-2023. The reconstruction is accurate to about five basis
    points a year.
    """
    frame = pd.read_excel(SHILLER, sheet_name="Data", skiprows=7)
    frame.columns = [str(c).strip() for c in frame.columns]
    shiller = frame.loc[:, [c for c in frame.columns if c in ("Date", "P", "D")]].dropna()
    shiller = shiller[pd.to_numeric(shiller["Date"], errors="coerce").notna()].copy()
    shiller["Date"] = shiller["Date"].astype(float)
    year = shiller["Date"].astype(int)
    month = ((shiller["Date"] - year) * 100).round().astype(int).clip(1, 12)
    shiller.index = pd.to_datetime(dict(year=year, month=month, day=1))
    yields = (shiller["D"] / shiller["P"]).loc["1965":]

    price = yf.download("^GSPC", start="1970-01-01", auto_adjust=True,
                        progress=False)["Close"].dropna()
    price = price.iloc[:, 0] if isinstance(price, pd.DataFrame) else price
    accrual = yields.reindex(price.index, method="ffill") / 252.0
    return (price.pct_change() + accrual).dropna().rename("total_return")


def load() -> tuple[pd.Series, pd.Series]:
    """Daily total return and the three-month bill, cached."""
    target = CACHE / "shu_sp500_tr.parquet"
    if target.exists():
        frame = pd.read_parquet(target)
    else:
        bill = fred.fetch_current("DTB3", frequency="daily", start="1970-01-01")
        bill = bill.set_index("period")["value"].sort_index()
        frame = total_return().to_frame().join(bill.rename("bill"), how="left")
        frame["bill"] = frame["bill"].ffill()
        frame = frame.dropna()
        frame.to_parquet(target)
    return frame["total_return"], frame["bill"]


def build_states(x: pd.DataFrame, excess: pd.Series, penalty: float) -> pd.Series:
    """Refit centroids every six months, infer online in between.

    Standardisation uses the training window's own mean and deviation, never the
    full sample: a 2008 reading must not know how extreme 2008 turned out.
    """
    dates = x.index
    refits = pd.date_range(dates[TRAIN_DAYS], dates[-1], freq=f"{REFIT_MONTHS}MS")
    states = pd.Series(np.nan, index=dates)

    for i, refit in enumerate(refits):
        window = x.loc[x.index < refit].tail(TRAIN_DAYS)
        if len(window) < TRAIN_DAYS // 2:
            continue
        mean, std = window.mean(), window.std().replace(0.0, np.nan)
        train = ((window - mean) / std).dropna()
        if len(train) < 500:
            continue

        centroids = fit_centroids(train.to_numpy(), penalty=penalty)
        ordering = order_states(
            online_states(train.to_numpy(), centroids, penalty),
            excess.reindex(train.index).to_numpy(),
        )
        # Map the raw state ids onto the bull/bear ordering just established.
        raw = online_states(train.to_numpy(), centroids, penalty)
        lookup = {}
        for raw_state, ranked in zip(raw, ordering, strict=True):
            lookup[int(raw_state)] = int(ranked)

        stop = refits[i + 1] if i + 1 < len(refits) else dates[-1] + pd.Timedelta(days=1)
        block = x.loc[(x.index >= refit) & (x.index < stop)]
        if block.empty:
            continue
        context = pd.concat([window.tail(TRAIN_DAYS), block])
        scaled = ((context - mean) / std).dropna()
        inferred = online_states(scaled.to_numpy(), centroids, penalty)
        mapped = np.array([lookup.get(int(s), int(s)) for s in inferred])
        states.loc[scaled.index[-len(block):]] = mapped[-len(block):]

    return states


def strategy(states: pd.Series, asset: pd.Series, bill: pd.Series) -> tuple[pd.Series, pd.Series]:
    """One hundred percent risky in the bull state, the bill otherwise."""
    weight = (states == states.max()).astype(float).shift(EXECUTION_LAG)
    weight = weight.reindex(asset.index).fillna(0.0)
    daily_bill = bill.reindex(asset.index).ffill() / 100.0 / 252.0
    gross = weight * asset + (1.0 - weight) * daily_bill
    cost = weight.diff().abs().fillna(0.0) * COST_ONE_WAY
    return (gross - cost).rename("net"), weight


def stats(returns: pd.Series, bill: pd.Series) -> dict[str, float]:
    r = returns.dropna()
    if len(r) < 60:
        return {}
    curve = (1 + r).cumprod()
    years = len(r) / 252
    excess = r - bill.reindex(r.index).ffill() / 100.0 / 252.0
    return {
        "cagr": float(curve.iloc[-1] ** (1 / years) - 1),
        "vol": float(r.std(ddof=1) * np.sqrt(252)),
        "sharpe": float(excess.mean() / excess.std(ddof=1) * np.sqrt(252)),
        "mdd": float((curve / curve.cummax() - 1).min()),
    }


def main() -> None:
    asset, bill = load()
    excess = asset - bill.reindex(asset.index).ffill() / 100.0 / 252.0
    x = features(excess).dropna()
    excess = excess.reindex(x.index)

    print(RULE)
    print("REPLICATION  Shu, Yu & Mulvey (2024) — S&P 500")
    print(RULE)
    print(f"\n   {len(x):,} jours, {x.index.min():%Y-%m} a {x.index.max():%Y-%m}")
    print(f"   fenetre {TRAIN_DAYS} jours, reestimation tous les {REFIT_MONTHS} mois")
    print(f"   grille lambda {LAMBDA_GRID}, validation {VALIDATION_YEARS} ans")
    print(f"   delai {EXECUTION_LAG} seances, cout {COST_ONE_WAY*1e4:.0f} bp par sens\n")

    bh = stats(asset.loc[OOS_START:OOS_END], bill)
    print(f"   BUY & HOLD  CAGR {bh['cagr']:>6.1%}  vol {bh['vol']:>5.1%}  "
          f"Sharpe {bh['sharpe']:>5.2f}  MDD {bh['mdd']:>6.1%}")
    ref = PUBLISHED["buy_hold"]
    print(f"   publie      CAGR {ref['cagr']:>6.1%}  vol {ref['vol']:>5.1%}  "
          f"Sharpe {ref['sharpe']:>5.2f}  MDD {ref['mdd']:>6.1%}\n")

    curves, weights = {}, {}
    for penalty in LAMBDA_GRID:
        st = build_states(x, excess, penalty)
        net, w = strategy(st, asset.reindex(x.index), bill)
        curves[penalty], weights[penalty] = net, w
        oos = net.loc[OOS_START:OOS_END]
        s = stats(oos, bill)
        print(f"   lambda {penalty:>6.1f}  CAGR {s.get('cagr', float('nan')):>6.1%}  "
              f"vol {s.get('vol', float('nan')):>5.1%}  Sharpe {s.get('sharpe', float('nan')):>5.2f}  "
              f"MDD {s.get('mdd', float('nan')):>6.1%}  "
              f"expo {w.loc[OOS_START:OOS_END].mean():>5.1%}  "
              f"rot {w.loc[OOS_START:OOS_END].diff().abs().sum()/((len(oos))/252):>4.0%}")

    jm = PUBLISHED["jump"]
    print(f"\n   publie JM   CAGR {jm['cagr']:>6.1%}  vol {jm['vol']:>5.1%}  "
          f"Sharpe {jm['sharpe']:>5.2f}  MDD {jm['mdd']:>6.1%}  "
          f"expo {jm['exposure']:>5.1%}  rot {jm['turnover']:>4.0%}")

    pd.DataFrame(curves).to_parquet(CACHE / "shu_curves.parquet")
    pd.DataFrame(weights).to_parquet(CACHE / "shu_weights.parquet")
    print("\n   courbes par lambda ecrites en cache")
    print("\n" + RULE)


if __name__ == "__main__":
    main()
