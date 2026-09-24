"""Phase A of `docs/PRESPEC_LONGHIST.md`: the reduced SJM as a classifier, no return read.

Scores the out-of-sample states of `scripts/longhist_fit.py` against the NBER
recessions of 1937-2026 (fourteen of them), next to the one-line controls, and
measures the axis-1 gate that must be settled before Phase B is read. No strategy
return appears in this file: the market series is used only to build the controls
(realised volatility, the 24-month bear market), exactly as the classifier's own
features use it.

Sections
    0  inputs (SHA-256)
    1  every label: balanced accuracy, kappa, recall, precision, recessions detected,
       median entry latency, transitions per year, stress share        (§5.1)
    2  PASS-A for the long SJM                                          (§5.2)
    3  A2: kappa gap against the one-line rules, stationary bootstrap   (§5.2)
    4  recession by recession                                           (§5.1)
    5  five consecutive folds, descriptive                              (§5.2)
    6  agreement between labels, the axis-1 gate and axis 2             (§5.4)
    7  the 50-feature A' on the common period 2002-2026                 (§5.3)
    8  the model itself: penalty path and feature weights
    9  the INDPRO sensitivity, if its states exist                      (§3.4)
   10  descriptive, added after 1-9 were seen: where the stress state lives

Usage:
    .venv/bin/python scripts/longhist_validate.py > docs/artifacts/longhist/validation.txt
"""

from __future__ import annotations

import hashlib
import json
import warnings

import numpy as np
import pandas as pd

from regime_lab.analysis import trials
from regime_lab.config import CACHE, RAW
from regime_lab.data import store
from regime_lab.evaluation.predictive import volatility_quantile_placebo
from regime_lab.evaluation.reliability import external_validation
from regime_lab.extensions import crisis
from regime_lab.extensions import longhist as lh
from regime_lab.models.jump import JumpRegimes

warnings.filterwarnings("ignore")
RULE = "=" * 78
OOS_START = pd.Timestamp("1937-01-01")
END = pd.Timestamp("2026-07-31")
A_PRIME_START = pd.Timestamp("2002-04-01")
BA_MIN, KAPPA_MIN, DETECTED_MIN, RECESSIONS = 0.85, 0.40, 12, 14
GATE = 0.10
DRAWS, BLOCK = 2000, 252
INPUTS = (
    RAW / "longhist" / "ff3_daily.parquet",
    RAW / "longhist" / "usrec.parquet",
    CACHE / "longhist_states.parquet",
    CACHE / "longhist_refits.parquet",
    CACHE / "states.parquet",
)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def usrec() -> pd.Series:
    frame = store.read(lh.SOURCE, "usrec")
    return pd.Series(frame["value"].to_numpy(float), index=pd.DatetimeIndex(frame["period"]))


def asof(label: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """The last value stamped at or before each session of ``index`` (no lag)."""
    known = label.dropna().sort_index()
    return known.reindex(known.index.union(index)).ffill().reindex(index)


def labels(states: pd.DataFrame, oos: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Every label of §4 on the out-of-sample sessions, 1 = calm, 0 = stress."""
    market = lh.market_returns(lh.load_wide("ff3_daily"))
    simple = market["total"]
    price = (1.0 + simple).cumprod()
    sjm = states["state"]
    bear = lh.bear_market(price)
    out = {
        "SJM long": sjm,
        "asymmetric exit k=10": lh.asymmetric_exit(sjm, simple, k=10),
        "asymmetric exit k=5": lh.asymmetric_exit(sjm, simple, k=5),
        "median rule": volatility_quantile_placebo(simple),
        "80th rule": crisis.volatility_tail_rule(simple),
        "DM panic": lh.panic_state(price, simple),
        "bear 24m": (1.0 - bear).where(bear.notna()),
    }
    return {k: v.reindex(oos) for k, v in out.items()}


def score(label: pd.Series, ref: pd.Series, episodes) -> dict:
    ext = external_validation(label, ref)
    pair = pd.concat([label.rename("s"), ref.rename("r")], axis=1).dropna()
    stress, rec = pair["s"].eq(lh.STRESS), pair["r"].eq(1.0)
    table = lh.detection(label, episodes)
    detected = table[table["detected"]]
    return {
        "ba": ext["balanced_accuracy"], "kappa": ext["kappa"],
        "recall": float(stress[rec].mean()), "precision": float(rec[stress].mean()),
        "detected": int(table["detected"].sum()), "episodes": len(table),
        "latency": float(detected["latency_days"].median()) if len(detected) else float("nan"),
        "transitions": lh.transitions_per_year(label), "stress": float(stress.mean()),
        "n": ext["n"], "table": table,
    }


def print_scores(rows: dict[str, dict]) -> None:
    print(f"   {'label':<22} {'bal.acc':>8} {'kappa':>6} {'recall':>7} {'precis.':>8} "
          f"{'detected':>9} {'latency':>8} {'tr/yr':>6} {'stress':>7}")
    for name, s in rows.items():
        print(f"   {name:<22} {s['ba']:>8.1%} {s['kappa']:>6.2f} {s['recall']:>7.1%} "
              f"{s['precision']:>8.1%} {s['detected']:>4}/{s['episodes']:<4} "
              f"{s['latency']:>7.0f}d {s['transitions']:>6.2f} {s['stress']:>7.1%}")


def main() -> None:
    states = pd.read_parquet(CACHE / "longhist_states.parquet")
    oos = states["state"].dropna().index
    oos = oos[(oos >= OOS_START) & (oos <= END)]
    rec = usrec()
    episodes = [e for e in lh.recessions(rec) if e[0] >= oos.min() and e[1] <= END]
    ref = lh.daily_reference(rec, oos)
    labs = labels(states, oos)

    print(RULE)
    print("LONGHIST PHASE A — the reduced SJM as a classifier, 1937-2026, no return read")
    print(RULE)
    print("\n0.  INPUTS (SHA-256)\n")
    for path in INPUTS:
        print(f"   {sha256(path)[:16]}  {path.relative_to(RAW.parent)}")
    nan = {k: int(v.isna().sum()) for k, v in labs.items()}
    print(f"\n   out of sample {oos.min():%Y-%m-%d} -> {oos.max():%Y-%m-%d}, {len(oos):,} "
          f"sessions; {len(episodes)} NBER recessions, {int(ref.sum()):,} recession sessions "
          f"({ref.mean():.1%})")
    print(f"   missing values per label: {nan}")

    print(f"\n{RULE}\n1.  EVERY LABEL AGAINST NBER (stress = recession), whole window\n")
    rows = {name: score(lab, ref, episodes) for name, lab in labs.items()}
    print_scores(rows)
    print("\n   latency: median over detected recessions of (first stress session in")
    print("   [first recession month - 183 days, trough]) - first day of the first")
    print("   recession month, calendar days; negative = ahead of the NBER month")

    s = rows["SJM long"]
    print(f"\n{RULE}\n2.  PASS-A (§5.2) — the long SJM\n")
    checks = {
        f"balanced accuracy >= {BA_MIN:.0%}": (s["ba"], s["ba"] >= BA_MIN),
        f"kappa >= {KAPPA_MIN:.2f}": (s["kappa"], s["kappa"] >= KAPPA_MIN),
        f"recessions detected >= {DETECTED_MIN} of {RECESSIONS}":
            (s["detected"], s["detected"] >= DETECTED_MIN),
    }
    finite = all(np.isfinite(v) for v, _ in checks.values()) and len(episodes) == RECESSIONS
    for label, (value, ok) in checks.items():
        shown = f"{value:.3f}" if isinstance(value, float) else str(value)
        print(f"   {label:<36} {shown:>8}   {'yes' if ok else 'NO'}")
    if not finite:
        verdict_a = "NOT FINITE, or not 14 recessions — no verdict"
    elif all(ok for _, ok in checks.values()):
        verdict_a = "PASS-A"
    else:
        missing = [k for k, (_, ok) in checks.items() if not ok]
        verdict_a = "FAIL-A — missing: " + "; ".join(missing)
    print(f"\n   => {verdict_a}")

    print(f"\n{RULE}\n3.  A2 — kappa against NBER, long SJM minus a one-line rule, paired")
    print(f"    stationary block bootstrap, mean block {BLOCK}, {DRAWS} draws, 95% interval\n")
    for other, deciding in (("80th rule", True), ("median rule", False), ("DM panic", False),
                            ("asymmetric exit k=10", False)):
        ci = lh.kappa_difference_ci(labs["SJM long"], labs[other], ref, block=BLOCK,
                                    draws=DRAWS, seed=0)
        if not all(np.isfinite([ci["delta"], ci["low"], ci["high"]])):
            word = "NOT FINITE — no verdict"
        elif ci["low"] > 0:
            word = "BETTER"
        elif ci["high"] < 0:
            word = "WORSE"
        else:
            word = "INDISTINGUISHABLE"
        tag = "decides A2" if deciding else "reported"
        print(f"   vs {other:<22} delta kappa {ci['delta']:+.3f}  [{ci['low']:+.3f}, "
              f"{ci['high']:+.3f}]  => {word}  ({tag})")

    print(f"\n{RULE}\n4.  RECESSION BY RECESSION — stress sessions inside / sessions, latency\n")
    names = ("SJM long", "asymmetric exit k=10", "80th rule", "DM panic")
    print(f"   {'first month':<12} {'last month':<11}" +
          "".join(f"{n[:18]:>22}" for n in names))
    tables = {n: rows[n]["table"] for n in names}
    for i, (peak, trough) in enumerate(episodes):
        line = f"   {peak:%Y-%m}      {trough:%Y-%m}     "
        for n in names:
            r = tables[n].iloc[i]
            lat = "   -" if not np.isfinite(r["latency_days"]) else f"{r['latency_days']:+5.0f}d"
            mark = "*" if r["detected"] else " "
            line += f"{r['stress_sessions']:>6}/{r['sessions']:<5}{mark}{lat:>8}  "
        print(line)
    print("\n   * = detected (at least 21 stress sessions inside the NBER months)")

    print(f"\n{RULE}\n5.  FIVE CONSECUTIVE FOLDS, descriptive (NBER balanced accuracy / kappa)\n")
    edges = np.linspace(0, len(oos), 6).round().astype(int)
    print(f"   {'fold':<25} {'rec. share':>10}" + "".join(f"{n[:18]:>22}" for n in names))
    for f, (a, b) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        part = oos[a:b]
        line = (f"   {f + 1}: {part.min():%Y-%m} -> {part.max():%Y-%m}      "
                f"{ref.loc[part].mean():>8.1%}")
        for n in names:
            e = external_validation(labs[n].loc[part], ref.loc[part])
            line += f"      {e['balanced_accuracy']:>6.1%} / {e['kappa']:>5.2f}"
        print(line)

    print(f"\n{RULE}\n6.  AGREEMENT BETWEEN LABELS (kappa), THE AXIS-1 GATE, AXIS 2\n")
    order = ["SJM long", "asymmetric exit k=10", "asymmetric exit k=5", "median rule",
             "80th rule", "DM panic", "bear 24m"]
    print(f"   {'':<22}" + "".join(f"{n[:11]:>12}" for n in order))
    for a in order:
        print(f"   {a:<22}" + "".join(
            f"{'-' if a == b else format(lh.kappa(labs[a], labs[b]), '+.2f'):>12}" for b in order))
    k_sjm = lh.kappa(labs["SJM long"], labs["bear 24m"])
    k_tail = lh.kappa(labs["80th rule"], labs["bear 24m"])
    gap = k_sjm - k_tail
    print(f"\n   axis 1: kappa(SJM long, bear) {k_sjm:+.3f} - kappa(80th rule, bear) "
          f"{k_tail:+.3f} = {gap:+.3f}  (gate {GATE:.2f})")
    if not np.isfinite(gap):
        gate = "NOT FINITE — no verdict on the gate"
    elif gap < GATE:
        gate = ("BELOW THE GATE — Phase B is declared a seventh device before it is read; "
                "the written prediction for B3 becomes: not useful")
    else:
        gate = "ABOVE THE GATE — the long SJM differs from the volatility rule on axis 1"
    print(f"   => {gate}")
    k_asym = lh.kappa(labs["asymmetric exit k=10"], labs["bear 24m"])
    print(f"   same measure for the asymmetric exit k=10: {k_asym - k_tail:+.3f}")
    tr = rows["SJM long"]["transitions"]
    print(f"   axis 2: the long SJM changes state {tr:.2f} times a year "
          f"({'above' if tr > 2 else 'below'} ~2); the asymmetric exit k=10 "
          f"{rows['asymmetric exit k=10']['transitions']:.2f}, k=5 "
          f"{rows['asymmetric exit k=5']['transitions']:.2f}")

    print(f"\n{RULE}\n7.  THE 50-FEATURE A' ON THE COMMON PERIOD, {A_PRIME_START:%Y-%m-%d} -> "
          f"{END:%Y-%m-%d} (§5.3)\n")
    a_prime = pd.read_parquet(CACHE / "states.parquet")["A' sparse jump"]
    common = oos[oos >= A_PRIME_START]
    a_on = asof(a_prime, common)
    ref_c = ref.loc[common]
    eps_c = [e for e in episodes if e[0] >= A_PRIME_START]
    rows_c = {"A' (50 features)": score(a_on, ref_c, eps_c),
              "SJM long (30)": score(labs["SJM long"].loc[common], ref_c, eps_c),
              "asymmetric exit k=10": score(labs["asymmetric exit k=10"].loc[common], ref_c,
                                            eps_c),
              "80th rule": score(labs["80th rule"].loc[common], ref_c, eps_c)}
    print_scores(rows_c)
    print(f"\n   kappa(A', SJM long) on the common period: "
          f"{lh.kappa(a_on, labs['SJM long'].loc[common]):+.3f}; "
          f"{len(common):,} sessions, {len(eps_c)} recessions — descriptive, decides nothing")
    print("   A' is taken as known at each session: its last state stamped at or before it")

    print(f"\n{RULE}\n8.  THE MODEL — penalty path and feature weights (training windows only)\n")
    refits = pd.read_parquet(CACHE / "longhist_refits.parquet")
    pen = refits["penalty"]
    changes = pen[pen.ne(pen.shift())]
    print(f"   {len(refits)} refits; penalty at each change: " + ", ".join(
        f"{d:%Y-%m} {v:g}" for d, v in changes.items()))
    print("   penalty counts: " + ", ".join(f"{k:g}: {v}" for k, v in
                                           pen.value_counts().sort_index().items()))
    log = trials.read()
    cal = log[log["family"] == "longhist_calibration"].copy()
    if not cal.empty:
        configs = cal["config"].map(json.loads)
        cal["train_end"] = configs.map(lambda c: c["train_end"])
        valid = cal.groupby("train_end")["m_sharpe"].apply(lambda s: s.notna().any())
        fallback = valid[~valid].index
        rates = cal.loc[cal["train_end"].isin(fallback), "m_switches_per_year"]
        print(f"   calibrations logged: {valid.size} ({len(cal)} candidates); with no admissible "
              f"candidate, hence the grid-median fallback lambda 20: {len(fallback)} "
              f"(train ends {', '.join(sorted(fallback))})")
        if len(fallback):
            print(f"   their candidates switch {rates.min():.3f} to {rates.max():.3f} times a year "
                  "in training, under the 0.5 floor of the admissible band")
    weights = refits.filter(like="w_")
    weights.columns = [c[2:] for c in weights.columns]
    mean_w = weights.mean().sort_values(ascending=False)
    active = (weights > 1e-6).mean()
    print(f"\n   {'feature':<24} {'mean weight':>12} {'share of refits > 0':>20}")
    for name, w in mean_w.items():
        print(f"   {name:<24} {w:>12.3f} {active[name]:>20.0%}")

    print(f"\n{RULE}\n9.  SENSITIVITY — the same model with INDPRO (revised, partly circular)\n")
    path = CACHE / "longhist_states_indpro.parquet"
    if path.exists():
        ind = pd.read_parquet(path)["state"].reindex(oos)
        rows_i = {"SJM long": rows["SJM long"], "SJM long + INDPRO": score(ind, ref, episodes)}
        print_scores(rows_i)
        print(f"\n   kappa(SJM long, SJM long + INDPRO) {lh.kappa(labs['SJM long'], ind):+.3f}"
              f"; {sha256(path)[:16]} {path.relative_to(RAW.parent)}")
        print("   never deciding: INDPRO is the current vintage and feeds the NBER dating")
    else:
        print("   not run (no longhist_states_indpro.parquet)")

    print(f"\n{RULE}\n10. DESCRIPTIVE, ADDED AFTER SECTIONS 1-9 WERE SEEN — where the stress state")
    print("    lives. No verdict changes; this checks a cause instead of asserting it.\n")
    s_on = states["state"].reindex(oos)
    s_off = states["state_offline"].reindex(oos)
    decade = (oos.year // 10) * 10
    share = pd.DataFrame({"online": s_on.eq(lh.STRESS), "offline": s_off.eq(lh.STRESS)},
                         index=oos).groupby(decade).mean()
    print("   out-of-sample stress share by decade:")
    print("   " + "  ".join(f"{d}s {r['online']:>5.1%}" for d, r in share.iterrows()))
    runs = s_on.ne(s_on.shift()).cumsum()
    first = next((g.index[0] for _, g in s_on.groupby(runs)
                  if g.iloc[0] == lh.STRESS and len(g) >= 21), None)
    print(f"   first stress run of at least 21 sessions starts {first:%Y-%m-%d}")
    lh.use_fast_dp()
    features = pd.read_parquet(CACHE / "longhist_features.parquet")
    market = lh.market_returns(lh.load_wide("ff3_daily"))["excess"]
    for refit in (pd.Timestamp("1936-12-31"), pd.Timestamp("1970-06-30")):
        at = refits.index[refits.index <= refit].max()
        train = features.loc[features.index < at]
        model = JumpRegimes(jump_penalty=float(refits.loc[at, "penalty"]), max_features=10.0)
        model.fit(train, market.reindex(train.index))
        inside = model.predict_online(train).eq(lh.STRESS)
        by_year = inside.groupby(inside.index.year).mean()
        top = by_year[by_year > 0.05]
        vols = {k: v * np.sqrt(252) for k, v in model.state_vol_.items()}
        print(f"\n   refit {at:%Y-%m-%d} (penalty {refits.loc[at, 'penalty']:g}), training "
              f"{train.index.min():%Y} -> {train.index.max():%Y}: stress on "
              f"{inside.mean():.1%} of training sessions; market vol in stress "
              f"{vols.get(0, float('nan')):.1%}, in calm {vols.get(1, float('nan')):.1%}")
        print("   training years with more than 5% stress sessions: " + ", ".join(
            f"{y} {v:.0%}" for y, v in top.items()))
    print(f"\n{RULE}")


if __name__ == "__main__":
    main()
