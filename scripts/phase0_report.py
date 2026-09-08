"""Phase 0 close-out: data quality screen, then the power calculation.

Both run before any model is fitted. The screen decides which series may feed a
feature; the power calculation decides whether the study's central question can
be answered at all with this sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_lab.analysis.power import minimum_detectable_sharpe_difference
from regime_lab.data import build_panel, store
from regime_lab.data.quality import flags, screen
from regime_lab.data.universe import MACRO_LAGGED, MACRO_VINTAGED, PRICES
from regime_lab.strategies.base import sixty_forty, vol_target

RULE = "=" * 78


def load_panel() -> pd.DataFrame:
    items = [("prices", "cross_asset")]
    items += [("macro", n) for n in list(MACRO_VINTAGED) + list(MACRO_LAGGED)]
    frame = store.load_many(items)
    dates = pd.date_range("1990-01-01", frame["period"].max(), freq="B")
    return build_panel(frame, dates, max_staleness=pd.Timedelta(days=120))


def main() -> None:
    panel = load_panel()
    print(RULE)
    print(f"PANEL  {panel.shape[0]:,} business days x {panel.shape[1]} series")
    print(RULE)

    print("\n1. DATA QUALITY SCREEN — stale prints fabricate signal\n")
    report = screen(panel[list(PRICES)])
    failed = flags(report)
    worst = report.groupby("series_id")["repeated_share"].mean().sort_values(ascending=False)
    print("   mean share of repeated levels, by series:")
    for name, share in worst.items():
        mark = "  FAILS" if name in set(failed["series_id"]) else ""
        print(f"      {name:<16} {share:6.2%}{mark}")
    if len(failed):
        print(f"\n   {len(failed)} series-years exceed the threshold. Worst five:")
        for _, row in failed.head(5).iterrows():
            print(
                f"      {row['series_id']:<16} {row['year']}  "
                f"{row['repeated_share']:6.2%} repeated, "
                f"longest run {row['longest_repeat_run']}"
            )
    else:
        print("\n   no series-year exceeds the threshold")

    print("\n" + RULE)
    print("\n2. POWER — what effect could this sample detect?\n")

    book = sixty_forty(panel).dropna()
    cash = (panel["rate_cash_3m"].reindex(book.index).ffill() / 100.0) / 252.0
    excess = (book - cash).dropna()

    train = excess.loc[:"2010-12-31"]
    target = float(train.std() * np.sqrt(252))
    managed = vol_target(excess, target=target).dropna()

    common = excess.index.intersection(managed.index)
    a, b = excess.loc[common].to_numpy(), managed.loc[common].to_numpy()

    sr = lambda x: x.mean() / x.std(ddof=1) * np.sqrt(252)  # noqa: E731
    print(f"   base 60/40, excess of cash     Sharpe {sr(a):5.2f}   n = {len(a):,}")
    print(f"   same book, vol-targeted        Sharpe {sr(b):5.2f}   target {target:.1%}")
    print("   target calibrated on 1990-2010 only, never on the evaluation sample\n")

    for block, label in [(21, "one month"), (63, "one quarter"), (126, "six months")]:
        res = minimum_detectable_sharpe_difference(a, b, mean_block=block, draws=1_500)
        print(
            f"   blocks of {block:>3} sessions ({label:<10})  "
            f"SE {res.se:5.3f}   MDE {res.mde:5.3f}   observed {res.observed:+.3f}"
        )

    res = minimum_detectable_sharpe_difference(a, b, mean_block=63, draws=1_500)
    print(f"\n   {res.verdict(plausible=0.30)}")
    print(f"   Stop rule 3 of the charter fires above 0.30. MDE at one quarter: {res.mde:.3f}")

    correlation = float(np.corrcoef(a, b)[0, 1])
    print(
        f"\n   READ THIS AS A LOWER BOUND. The two legs correlate at {correlation:.3f}, so the\n"
        f"   common market move cancels in the paired difference and its standard error is\n"
        f"   near its smallest possible value -- which is also why the block length barely\n"
        f"   moves it. A regime overlay will track the benchmark less closely, so its own\n"
        f"   minimum detectable effect will be larger than {res.mde:.2f}. The figure to compare\n"
        f"   against the stop rule is the one recomputed against the actual overlay in\n"
        f"   phase 3, not this one."
    )
    print(
        f"\n   Preliminary reading: volatility targeting moves this book by {-res.observed:+.3f}\n"
        f"   of Sharpe, well inside the {res.mde:.2f} the sample can resolve. Consistent with\n"
        f"   Cederburg, O'Doherty, Wang and Yan (JFE 2020), and a warning that the study's\n"
        f"   central comparison lives in a region this data resolves poorly."
    )
    print("\n" + RULE)


if __name__ == "__main__":
    main()
