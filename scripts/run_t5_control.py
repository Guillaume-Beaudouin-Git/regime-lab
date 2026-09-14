"""T5 — adjusted Rand index between successive refits, at paired cadence.

Charter, section 05, control 5: "Indice de Rand ajusté entre partitions
successives, à cadence de réestimation appariée entre familles — sans quoi on
republie un artefact de protocole plutôt qu'une propriété de modèle."

Charter, section 02: the sources T5 is meant to catch are named there and are
NOT label switching — "les optima locaux de l'algorithme EM selon
l'initialisation, la dérive des paramètres en fenêtre expansive, et le choix du
nombre d'états".

Design.

* Cadence pairing between families is a property of the protocol and is checked,
  not assumed: all five families must refit on the identical 49 dates.
* Every model is scored on ONE common date grid (the 6 377 out-of-sample
  sessions), so ARI(i, i+k) is comparable across every i and every k with no
  window-length confound. A second, operational reading uses the semi-annual
  block each model actually traded.
* Three nulls, because an ARI without one proves nothing:
    - label permutation      -> shown to be degenerate (ARI is invariant), kept
                                only as the verification it actually is;
    - matched circular shift -> preserves each sequence's exact run-length
                                structure and marginal, destroys the alignment;
    - matched Markov draw    -> preserves marginal and transition probabilities;
    - distant refits         -> the lag-k profile, k up to 48 (24 years apart).
* Stress dating is the repository's own mechanical rule
  (`reliability.drawdown_reference`), never hand-drawn bands: the charter says
  those are not admissible as a scoring target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from regime_lab.config import CACHE
from regime_lab.evaluation.reliability import drawdown_reference
from regime_lab.strategies.book import panels

OUT = CACHE / "t5_refits"
FAMILIES = {
    "A  jump": "A__jump",
    "A' sparse jump": "Ap_sparse_jump",
    "B  filtered HMM": "B__filtered_HMM",
    "C  gradient boost": "C__gradient_boost",
    "C' HAR-RV": "Cp_HAR-RV",
}
B_NULL = 200
RNG = np.random.default_rng(20260910)
RULE = "=" * 78


def ari(x: np.ndarray, y: np.ndarray) -> float:
    return float(adjusted_rand_score(x, y))


def best_match_disagreement(x: np.ndarray, y: np.ndarray) -> float:
    """Share of dates relabelled, after the best of the two label conventions."""
    d = float((x != y).mean())
    return min(d, 1.0 - d)


def transition_matrix(x: np.ndarray) -> tuple[float, float, float]:
    """(P(stay|0), P(stay|1), P(start=1)) of a binary sequence."""
    a, b = x[:-1], x[1:]
    p0 = float((b[a == 0] == 0).mean()) if (a == 0).any() else 0.5
    p1 = float((b[a == 1] == 1).mean()) if (a == 1).any() else 0.5
    return p0, p1, float(x.mean())


def simulate_markov(n: int, p0: float, p1: float, pi: float, reps: int) -> np.ndarray:
    """`reps` independent binary chains of length `n`, vectorised over reps."""
    out = np.empty((reps, n), dtype=np.int8)
    cur = (RNG.random(reps) < pi).astype(np.int8)
    out[:, 0] = cur
    stay = np.array([p0, p1])
    for t in range(1, n):
        u = RNG.random(reps)
        keep = u < stay[cur]
        cur = np.where(keep, cur, 1 - cur).astype(np.int8)
        out[:, t] = cur
    return out


def null_circular(x: np.ndarray, y: np.ndarray, reps: int) -> np.ndarray:
    """ARI against circularly shifted copies of y: exact structure, no alignment."""
    n = len(y)
    shifts = RNG.integers(1, n, size=reps)
    return np.array([ari(x, np.roll(y, int(s))) for s in shifts])


def null_markov(x: np.ndarray, y: np.ndarray, reps: int) -> np.ndarray:
    """ARI between two independent chains matched on marginal and persistence."""
    n = len(x)
    xs = simulate_markov(n, *transition_matrix(x), reps)
    ys = simulate_markov(n, *transition_matrix(y), reps)
    return np.array([ari(xs[b], ys[b]) for b in range(reps)])


def main() -> None:
    published = pd.read_parquet(CACHE / "states.parquet")
    oos = published.dropna(how="all").index
    price_panel, _ = panels()
    stress = drawdown_reference(price_panel["eq_us_large"]).reindex(oos).fillna(0.0)

    print(RULE)
    print("T5  ADJUSTED RAND INDEX BETWEEN SUCCESSIVE REFITS")
    print(RULE)

    # ---------------------------------------------------------------- cadence
    diag = pd.read_parquet(CACHE / "refit_diagnostics.parquet")
    schedules = {f: tuple(g["refit"].sort_values()) for f, g in diag.groupby("family")}
    identical = len(set(schedules.values())) == 1
    print("\n[0] cadence pairing between families")
    print(f"    5 families, {len(next(iter(schedules.values())))} refits each, "
          f"schedules identical across families: {identical}")
    print(f"    common OOS grid: {len(oos):,} sessions "
          f"{oos.min():%Y-%m-%d} to {oos.max():%Y-%m-%d}")

    rows, lag_rows, pair_rows = [], [], []

    for name, tag in FAMILIES.items():
        frame = pd.read_parquet(OUT / f"labels_online_{tag}.parquet")
        cols = sorted(frame.columns)
        grid = frame.reindex(oos)
        missing = int(grid.isna().sum().sum())
        L = {c: grid[c].to_numpy(dtype=int) for c in cols}
        refits = [pd.Timestamp(c) for c in cols]

        # ------------------------------------------------- verification n.7
        y0 = L[cols[1]]
        flip_delta = abs(ari(L[cols[0]], y0) - ari(L[cols[0]], 1 - y0))

        # ------------------------------------------------------ lag profile
        by_lag = {}
        for k in range(1, len(cols)):
            vals = [ari(L[cols[i]], L[cols[i + k]]) for i in range(len(cols) - k)]
            by_lag[k] = np.array(vals)
            lag_rows.append({"family": name, "lag": k, "n_pairs": len(vals),
                             "mean_ari": float(np.mean(vals)),
                             "median_ari": float(np.median(vals)),
                             "min_ari": float(np.min(vals))})

        adjacent = by_lag[1]
        distant = np.concatenate([by_lag[k] for k in by_lag if k >= 20])

        # ------------------------------------------------------------ nulls
        circ, mark, pct_circ, pct_mark = [], [], [], []
        for i in range(len(cols) - 1):
            x, y = L[cols[i]], L[cols[i + 1]]
            obs = adjacent[i]
            nc = null_circular(x, y, B_NULL)
            nm = null_markov(x, y, B_NULL)
            circ.append(nc.mean())
            mark.append(nm.mean())
            pct_circ.append(float((nc < obs).mean()))
            pct_mark.append(float((nm < obs).mean()))

            block = oos[(oos >= refits[i]) & (oos < refits[i + 1])]
            # Third window: the 504 sessions ending at the EARLIER refit. Both
            # models were trained on every one of them, so this window carries no
            # extrapolation at all and isolates pure estimation stability.
            hist = frame.index[frame.index < refits[i]][-504:]
            if len(block) >= 30:
                xb = grid.loc[block, cols[i]].to_numpy(dtype=int)
                yb = grid.loc[block, cols[i + 1]].to_numpy(dtype=int)
                if len(hist) >= 252:
                    xh = frame.loc[hist, cols[i]].dropna()
                    yh = frame.loc[hist, cols[i + 1]].reindex(xh.index).dropna()
                    xh = xh.reindex(yh.index)
                    ari_hist = ari(xh.to_numpy(dtype=int), yh.to_numpy(dtype=int))
                else:
                    ari_hist = np.nan
                pair_rows.append({
                    "family": name,
                    "refit_from": refits[i].date(),
                    "refit_to": refits[i + 1].date(),
                    "n_block": len(block),
                    "ari_grid": obs,
                    "ari_block": ari(xb, yb),
                    "ari_shared_train_504": ari_hist,
                    "relabelled_share": best_match_disagreement(xb, yb),
                    "stress_share": float(stress.loc[block].mean()),
                })

        rows.append({
            "family": name,
            "missing_labels": missing,
            "flip_delta": flip_delta,
            "ari_adj_mean": float(adjacent.mean()),
            "ari_adj_median": float(np.median(adjacent)),
            "ari_adj_min": float(adjacent.min()),
            "ari_adj_p10": float(np.percentile(adjacent, 10)),
            "null_circ_mean": float(np.mean(circ)),
            "null_mark_mean": float(np.mean(mark)),
            "ari_distant_mean": float(distant.mean()),
            "pairs_above_circ_95": float(np.mean(np.array(pct_circ) >= 0.95)),
            "pairs_above_mark_95": float(np.mean(np.array(pct_mark) >= 0.95)),
        })
        print(f"    {name:<18} done ({len(cols)} refits, {missing} missing labels)")

    summary = pd.DataFrame(rows).set_index("family")
    lags = pd.DataFrame(lag_rows)
    pairs = pd.DataFrame(pair_rows)
    summary.to_parquet(OUT / "t5_summary.parquet")
    lags.to_parquet(OUT / "t5_lag_profile.parquet")
    pairs.to_parquet(OUT / "t5_pairs.parquet")

    pd.set_option("display.width", 200)
    print("\n[1] label-convention invariance (failure n.7 guard)")
    print("    max |ARI(x,y) - ARI(x,1-y)| over families: "
          f"{summary['flip_delta'].max():.3e}")

    print(f"\n[2] ARI between adjacent refits, common {len(oos):,}-session grid")
    print(f"    {'family':<18} {'mean':>7} {'median':>7} {'p10':>7} {'min':>7} "
          f"{'null circ':>10} {'null Mkv':>9} {'lag>=20':>8}")
    for f, r in summary.iterrows():
        print(f"    {f:<18} {r['ari_adj_mean']:>7.3f} {r['ari_adj_median']:>7.3f} "
              f"{r['ari_adj_p10']:>7.3f} {r['ari_adj_min']:>7.3f} "
              f"{r['null_circ_mean']:>10.3f} {r['null_mark_mean']:>9.3f} "
              f"{r['ari_distant_mean']:>8.3f}")

    print("\n[3] share of the 48 adjacent pairs above the 95th percentile of their own null")
    print(f"    {'family':<18} {'circular':>9} {'Markov':>9}")
    for f, r in summary.iterrows():
        print(f"    {f:<18} {r['pairs_above_circ_95']:>8.1%} {r['pairs_above_mark_95']:>8.1%}")

    print("\n[4] lag profile, mean ARI (common grid)")
    piv = lags.pivot(index="lag", columns="family", values="mean_ari")
    print(piv.loc[[1, 2, 4, 8, 16, 24, 32, 40, 48]].round(3).to_string())

    print("\n[5] operational reading: the semi-annual block each model actually traded,")
    print("    and the 504 shared training sessions before the earlier refit")
    print(f"    {'family':<18} {'ARI blk':>8} {'ARI train':>10} {'relabelled':>11} "
          f"{'worst blk':>10} {'date':>12}")
    for f, g in pairs.groupby("family"):
        worst = g.loc[g["ari_block"].idxmin()]
        print(f"    {f:<18} {g['ari_block'].mean():>8.3f} "
              f"{g['ari_shared_train_504'].mean():>10.3f} "
              f"{g['relabelled_share'].mean():>10.1%} {worst['ari_block']:>10.3f} "
              f"{str(worst['refit_from']):>12}")

    # -------------------------------------------------------------- stress
    print("\n[6] does the ARI degrade around stress? "
          "(stress = >10% below the running 12-month high, reliability.drawdown_reference)")
    print(f"    {'family':<18} {'n stress':>8} {'ARI stress':>11} {'ARI calm':>9} "
          f"{'diff':>7} {'p (perm)':>9}")
    stress_rows = []
    for f, g in pairs.groupby("family"):
        flag = (g["stress_share"] >= 0.5).to_numpy()
        vals = g["ari_block"].to_numpy()
        if flag.sum() == 0 or flag.sum() == len(flag):
            print(f"    {f:<18} degenerate split")
            continue
        obs = vals[flag].mean() - vals[~flag].mean()
        draws = np.array([
            (lambda p: vals[p].mean() - vals[~p].mean())(
                RNG.permutation(flag).astype(bool)
            )
            for _ in range(10000)
        ])
        p = float((np.abs(draws) >= abs(obs)).mean())
        stress_rows.append({"family": f, "n_stress": int(flag.sum()),
                            "ari_stress": vals[flag].mean(),
                            "ari_calm": vals[~flag].mean(), "diff": obs, "p": p})
        print(f"    {f:<18} {flag.sum():>8} {vals[flag].mean():>11.3f} "
              f"{vals[~flag].mean():>9.3f} {obs:>7.3f} {p:>9.4f}")
    pd.DataFrame(stress_rows).to_parquet(OUT / "t5_stress.parquet")

    print("\n" + RULE)


if __name__ == "__main__":
    main()
