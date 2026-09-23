"""Descriptive facts behind docs/presentation/PISTES_AMELIORATION.md.

Nothing here reads a strategy return. No position is taken, no Sharpe is computed,
nothing is written to the trials log or to the cache. What is measured:

1. the sparse jump model's (A') stress episodes, and where the S&P 500 and the VIX
   stood when it entered and left them (the trough is located with hindsight: this
   describes the episodes, it is not a rule);
2. the VIX futures curve (front over second contract) inside and outside the episodes;
3. the structure of candidate labels for the ideas in the note: transitions per year,
   stress share, agreement with the one-line volatility rule at the 80th percentile.
   These are properties of labels, never of what trading on them would earn;
4. the stock-bond correlation regime, as a latent: clock and link with volatility;
5. the effective dimension of the volatility panel of the 46 trend instruments;
6. the arithmetic of the minimum detectable effect on a 1927-2026 momentum sample;
7. the NBER recession count, from FRED (public CSV, no key; skipped if offline);
8. a three-state sparse jump model, one fit on 1992-2006 and filtered online after,
   exactly as `explore_sjm_speed.py` does for two states: which third state appears;
9. the effective dimension of the six Fama-French daily factors (unconditional).

Usage: .venv/bin/python scripts/describe_pistes.py
Output: printed; the committed copy is docs/artifacts/pistes/description.txt.
"""

from __future__ import annotations

import io
import urllib.request

import numpy as np
import pandas as pd
from scipy import stats

from regime_lab.config import CACHE, RAW
from regime_lab.extensions.crisis import volatility_tail_rule
from regime_lab.models.jump import JumpRegimes
from regime_lab.strategies.book import base_book

STATE = "A' sparse jump"
PERIODS = 252
RULE = "-" * 78


def series(path, series_id: str) -> pd.Series:
    frame = pd.read_parquet(path)
    one = frame[frame["series_id"] == series_id].set_index("period")["value"]
    one.index = pd.DatetimeIndex(one.index)
    return one.sort_index().astype(float)


def transitions_per_year(label: pd.Series) -> float:
    label = label.dropna()
    changes = int((label != label.shift()).sum()) - 1
    years = (label.index[-1] - label.index[0]).days / 365.25
    return changes / years


def kappa(a: pd.Series, b: pd.Series) -> float:
    both = pd.concat([a, b], axis=1).dropna().to_numpy(float)
    x, y = both[:, 0], both[:, 1]
    po = float((x == y).mean())
    pe = float(x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean()))
    return (po - pe) / (1 - pe)


def stress_episodes(state: pd.Series, merge_gap: int = 60) -> list[tuple]:
    """Stress runs (label 0) merged when separated by at most ``merge_gap`` sessions."""
    seg = (state != state.shift()).cumsum()
    out: list[tuple] = []
    for _, g in state.groupby(seg):
        if g.iloc[0] != 0:
            continue
        if out and len(state.loc[out[-1][1]:g.index[0]]) - 2 <= merge_gap:
            out[-1] = (out[-1][0], g.index[-1])
        else:
            out.append((g.index[0], g.index[-1]))
    return out


def confirmed(flag: pd.Series, k: int) -> pd.Series:
    """True once ``flag`` has held for each of the last ``k`` sessions."""
    return flag.astype(float).rolling(k).min().eq(1.0)


def hysteresis(flag: pd.Series, k: int) -> pd.Series:
    """0/1 label that flips only after ``k`` consecutive sessions on the other side."""
    values = flag.astype(int).to_numpy()
    out = np.empty(len(values), dtype=float)
    current, run = values[0], 0
    for i, v in enumerate(values):
        run = run + 1 if v != current else 0
        if run >= k:
            current, run = v, 0
        out[i] = current
    return pd.Series(out, index=flag.index)


def vix_curve(vix: pd.Series) -> pd.DataFrame:
    """Front and second VIX futures settlements; contracts expiring after the date."""
    frame = pd.read_parquet(RAW / "crisis" / "vix_futures.parquet")
    wide = frame.pivot(index="period", columns="series_id", values="value")
    wide.index = pd.DatetimeIndex(wide.index)
    wide.columns = pd.DatetimeIndex([c[3:] for c in wide.columns])
    wide = wide.sort_index(axis=1)
    rows = {}
    for day in wide.index:
        row = wide.loc[day]
        live = row[(row.index > day) & row.notna()]
        if len(live) >= 2:
            rows[day] = (live.iloc[0], live.iloc[1])
    curve = pd.DataFrame(rows, index=["vx1", "vx2"]).T.join(vix.rename("vix")).dropna()
    # the first date after which both front contracts always settle (run_crisis_coupling)
    curve = curve.loc["2006-09-01":]
    curve["inverted"] = curve["vx1"] > curve["vx2"]
    return curve


def participation_ratio(corr: np.ndarray) -> float:
    eig = np.linalg.eigvalsh(corr)
    return float(eig.sum() ** 2 / (eig**2).sum())


def main() -> None:
    prices = RAW / "prices" / "cross_asset.parquet"
    spx = series(prices, "eq_us_large")
    vix = series(prices, "vol_vix").reindex(spx.index)
    y10 = series(prices, "bond_us_10y").reindex(spx.index)
    state = pd.read_parquet(CACHE / "states.parquet")[STATE].dropna()
    log_ret = np.log(spx).diff()
    rv21 = log_ret.rolling(21).std() * np.sqrt(PERIODS)
    rv63 = log_ret.rolling(63).std() * np.sqrt(PERIODS)
    tail = volatility_tail_rule(spx.pct_change())  # 1 = calm, 0 = stress

    # ------------------------------------------------------------------ 1
    print(RULE, "\n1. A' stress episodes (label 0), S&P 500 and VIX around them\n", RULE, sep="")
    print(f"sample {state.index[0].date()} -> {state.index[-1].date()}, {len(state)} labels, "
          f"{int((state == 0).sum())} stress, {transitions_per_year(state):.3f} transitions/yr")
    peak = spx.rolling(252, min_periods=1).max()
    total_after = 0
    for start, end in stress_episodes(state):
        window = spx.loc[start:end]
        trough = window.idxmin()
        stress = state.loc[start:end]
        stress = stress[stress == 0]
        after = int((stress.index >= trough).sum())
        total_after += after
        before = vix.loc[:start].iloc[-21:]
        rebound = window.loc[trough:]
        plus20 = rebound[rebound >= window.min() * 1.2]
        print(f"\n{start.date()} -> {end.date()}: {len(stress)} stress labels")
        print(f"  entry: S&P {spx.asof(start) / peak.asof(start) - 1:+.1%} from its 252-session "
              f"high; VIX {vix.asof(start):.1f} (range over the 21 sessions before: "
              f"{before.min():.1f}-{before.max():.1f})")
        fall = window.min() / spx.asof(start) - 1
        print(f"  S&P trough in the episode {trough.date()} ({fall:+.1%} from entry); "
              f"{after} stress labels on or after it ({after / len(stress):.0%})")
        print(f"  exit: S&P {spx.asof(end) / window.min() - 1:+.1%} above the trough; VIX "
              f"{vix.asof(end):.1f}; first close 20% above the trough "
              f"{plus20.index[0].date() if len(plus20) else 'none'}")
    print(f"\nall episodes: {total_after} of {int((state == 0).sum())} stress labels "
          f"({total_after / int((state == 0).sum()):.0%}) fall on or after their episode's trough")

    # ------------------------------------------------------------------ 2
    print("\n", RULE, "\n2. VIX futures curve: front above second contract (inverted)\n",
          RULE, sep="")
    curve = vix_curve(vix)
    print(f"sample {curve.index[0].date()} -> {curve.index[-1].date()}, {len(curve)} sessions; "
          f"inverted {curve['inverted'].mean():.1%}")
    lab = state.reindex(curve.index, method="ffill")
    for a, b in (("2008-09", "2009-12"), ("2020-02", "2021-05")):
        m = pd.DataFrame({
            "inverted share": curve.loc[a:b, "inverted"].resample("ME").mean(),
            "VIX mean": curve.loc[a:b, "vix"].resample("ME").mean(),
            "A' stress share": (lab.loc[a:b] == 0).resample("ME").mean(),
        })
        m.index = m.index.strftime("%Y-%m")
        print(m.round(2).to_string(), "\n")
    for year in ("2011", "2015", "2018", "2022", "2025"):
        c = curve.loc[year]
        print(f"{year}: {int(c['inverted'].sum())} of {len(c)} sessions inverted; "
              f"A' stress labels {int((state.loc[year] == 0).sum())}")

    # ------------------------------------------------------------------ 3
    print("\n", RULE, "\n3. Candidate labels: structure only (0 = stress)\n", RULE, sep="")
    a_calm = state.reindex(spx.index, method="ffill")
    rows = []

    def describe(name: str, label: pd.Series, on: pd.DatetimeIndex) -> None:
        lab_ = label.reindex(on).dropna()
        ref = tail.reindex(lab_.index)
        rows.append((name, lab_.index[0].date(), transitions_per_year(lab_),
                     float((lab_ == 0).mean()), kappa(lab_, ref)))

    oos = spx.index[spx.index >= state.index[0]]
    describe("A' as published", a_calm, oos)
    describe("vol rule, 80th percentile", tail, oos)
    for k in (5, 10):
        rv_exit = confirmed(rv21 < rv63, k)
        describe(f"A' stress unless RV21<RV63 for {k} sessions",
                 a_calm.where(~((a_calm == 0) & rv_exit), 1.0), oos)
    ts = curve.index
    inverted = curve["inverted"]
    for k in (5, 10):
        flat = confirmed(~inverted, k)
        a_ts = a_calm.reindex(ts)
        describe(f"A' stress unless curve upright for {k} sessions",
                 a_ts.where(~((a_ts == 0) & flat), 1.0), ts)
    describe("curve inverted, 5-session hysteresis", 1 - hysteresis(inverted, 5), ts)
    bear = (spx < spx.shift(504)).astype(float).where(spx.shift(504).notna())
    describe("S&P below its level 504 sessions earlier", 1 - bear, oos)
    print(f"{'label':52s} {'from':>10s} {'tr/yr':>6s} {'stress':>7s} {'kappa vs vol80':>15s}")
    for name, start, tr, share, kap in rows:
        print(f"{name:52s} {start!s:>10s} {tr:6.2f} {share:7.1%} {kap:15.2f}")
    for k in (5, 10):
        flat = confirmed(~inverted, k)
        a_ts = a_calm.reindex(ts)
        comb = a_ts.where(~((a_ts == 0) & flat), 1.0)
        for start, end in stress_episodes(state):
            if end < ts[0]:
                continue
            seg = comb.loc[start:end]
            first_calm = seg[seg == 1].index.min()
            print(f"  k={k}: A' episode {start.date()} -> {end.date()}, combined label first "
                  f"calm on {first_calm.date() if pd.notna(first_calm) else 'never'}")

    # ------------------------------------------------------------------ 4
    print("\n", RULE, "\n4. Stock-bond correlation (63 sessions), bond proxy = -d(10y yield)\n",
          RULE, sep="")
    corr = log_ret.rolling(63).corr(-y10.diff())
    ok = corr.dropna()
    pos = (ok > 0).astype(float)
    print(f"sample {ok.index[0].date()} -> {ok.index[-1].date()}; positive "
          f"{pos.mean():.1%}; sign changes/yr raw {transitions_per_year(pos):.2f}, with a "
          f"21-session hysteresis {transitions_per_year(hysteresis(ok > 0, 21)):.2f}")
    j = pd.concat([ok.rename("c"), rv21.rename("rv")], axis=1, sort=True).dropna()
    print(f"Spearman with RV21: correlation level {stats.spearmanr(j['c'], j['rv'])[0]:+.2f}, "
          f"positive-sign indicator {stats.spearmanr(j['c'] > 0, j['rv'])[0]:+.2f}")
    by_year = (ok > 0).groupby(ok.index.year).mean()
    print("share of sessions with a positive correlation, by year:")
    print(" ".join(f"{y}:{v:.0%}" for y, v in by_year.items()))

    # ------------------------------------------------------------------ 5
    print("\n", RULE, "\n5. Effective dimension of the volatility panel, 46 instruments\n",
          RULE, sep="")
    px = pd.read_parquet(CACHE / "trend_universe_m1.parquet").loc["2003-07-17":]
    # CL=F settled below zero in April 2020: no log price that day
    rv = np.log(px.where(px > 0)).diff().rolling(21, min_periods=15).std()
    monthly = np.log(rv.resample("ME").last()).dropna(axis=0, how="any")
    pr_level = participation_ratio(monthly.corr().to_numpy())
    pr_change = participation_ratio(monthly.diff().dropna().corr().to_numpy())
    eig = np.sort(np.linalg.eigvalsh(monthly.corr().to_numpy()))[::-1]
    print(f"{monthly.shape[1]} instruments x {monthly.shape[0]} month-ends; participation "
          f"ratio of log RV21: levels {pr_level:.2f}, monthly changes {pr_change:.2f}; "
          f"first eigenvalue {eig[0]:.1f} / {len(eig)}")

    # ------------------------------------------------------------------ 6
    print("\n", RULE, "\n6. MDE arithmetic for a momentum test on 1927-2026\n", RULE, sep="")
    z_beta = stats.norm.ppf(0.80)
    mde_now, tests_now, years_now = 0.395, 9, (pd.Timestamp("2026-07-31")
                                               - pd.Timestamp("2002-04-02")).days / 365.25
    years_long = (pd.Timestamp("2026-07-31") - pd.Timestamp("1927-11-01")).days / 365.25
    for tests in (1, 3):
        scale = ((stats.norm.ppf(1 - 0.025 / tests) + z_beta)
                 / (stats.norm.ppf(1 - 0.025 / tests_now) + z_beta))
        print(f"{tests} test(s): {mde_now} x {scale:.3f} x sqrt({years_now:.1f}/{years_long:.1f}) "
              f"= {mde_now * scale * np.sqrt(years_now / years_long):.3f}")
    print("assumes the paired standard error per year is the same before 2002; it is not "
          "(1930s momentum was far more volatile), so this is an order of magnitude only")

    # ------------------------------------------------------------------ 7
    print("\n", RULE, "\n7. NBER recessions (FRED USREC, monthly)\n", RULE, sep="")
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC"
        with urllib.request.urlopen(url, timeout=20) as resp:
            usrec = pd.read_csv(io.BytesIO(resp.read()), index_col=0, parse_dates=True).iloc[:, 0]
        for a, b in (("1926-07", "2026-09"), ("1937-01", "2026-09"), ("1990-01", "2026-09"),
                     ("2002-04", "2026-09")):
            w = usrec.loc[a:b]
            starts = int(((w == 1) & (w.shift(fill_value=0) == 0)).sum())
            print(f"{a} -> {b}: {starts} recessions, {int(w.sum())} recession months")
    except Exception as exc:  # offline: the rest of the file does not depend on it
        print(f"skipped ({type(exc).__name__})")

    # ------------------------------------------------------------------ 8
    print("\n", RULE, "\n8. Three-state sparse jump model: one fit 1992-2006, online after\n",
          RULE, sep="")
    features = pd.read_parquet(CACHE / "features.parquet").dropna()
    book = base_book().reindex(features.index).dropna()
    features = features.loc[book.index]
    model = JumpRegimes(n_states=3, jump_penalty=10.0, max_features=10.0)
    model.fit(features.loc[:"2006-12-31"], book.loc[:"2006-12-31"])
    three = model.predict_online(features).loc["2007-01-01":]
    print("training volatility of the base book by ranked state (0 = highest): "
          + ", ".join(f"{k}: {v:.5f}" for k, v in sorted(model.state_vol_.items())))
    print(f"shares {three.value_counts(normalize=True).sort_index().round(3).to_dict()}; "
          f"{transitions_per_year(three):.2f} transitions/yr (2007-2026)")
    seg = (three != three.shift()).cumsum()
    for _, g in three.groupby(seg):
        if g.iloc[0] != 2 and len(g) >= 15:
            print(f"  state {int(g.iloc[0])}: {g.index[0].date()} -> {g.index[-1].date()} "
                  f"({len(g)} sessions)")
    cols = ["mom_eq_21", "mom_eq_63", "vol_rv_21", "vol_vix", "asy_drawdown_252", "cre_baa"]
    print("mean standardised feature by state, 2007-2026:")
    print(features.loc["2007-01-01":, cols].groupby(three).mean().round(2).to_string())

    # ------------------------------------------------------------------ 9
    print("\n", RULE, "\n9. Effective dimension of the six daily Fama-French factors\n",
          RULE, sep="")
    ff = pd.read_parquet(RAW / "panels" / "factors_5.parquet")
    ff = ff.pivot(index="period", columns="series_id", values="value")
    ff.index = pd.DatetimeIndex(ff.index)
    umd = series(RAW / "crisis" / "french_umd.parquet", "ff_umd")
    six = ff.drop(columns=["ff_rf"]).join(umd.rename("ff_umd"), how="inner").dropna()
    print(f"{list(six.columns)}, {six.index[0].date()} -> {six.index[-1].date()}: participation "
          f"ratio {participation_ratio(six.corr().to_numpy()):.2f} of {six.shape[1]}")


if __name__ == "__main__":
    main()
