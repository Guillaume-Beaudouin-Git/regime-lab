"""T3 — the matched-exposure placebo the charter froze and the study never ran.

The charter (docs/CHARTER.html, section 05, third test) specifies:

    "Placebo a taux d'exposition apparie -- Signal aleatoire expose le meme
     pourcentage du temps. Largement subsume par T2 en version dynamique :
     une execution, une ligne de resultat."

Two readings of that sentence and one deliberate addition, all stated before any
number was produced:

1.  "Expose le meme pourcentage du temps" is read as: the placebo's mean
    exposure equals the real overlay's mean exposure, on the same out-of-sample
    index.  The real overlay position is binary, so mean exposure and share of
    time exposed are the same quantity.

2.  A *constant* exposure benchmark is arithmetically empty on the Sharpe ratio.
    Scaling a return series by a constant leaves its Sharpe to the decimal --
    the exact defect the charter itself corrected in T2 v1.  Matching exposure
    therefore has to be done with a *random path*, not with a constant.

3.  ADDITION, declared: the charter matches exposure only.  A random 0/1 signal
    with the right unconditional exposure but drawn i.i.d. would flip state on
    roughly half the sessions, against 0.2% to 7% for the real overlays.  It
    would be a different object -- differently autocorrelated, differently
    priced once costs are charged, and holding its exposure in single days
    rather than in blocks.  This control therefore matches the *switch rate*
    (mean absolute change in position, i.e. turnover) as well.  Matching more
    than the charter asks makes the placebo harder to beat, never easier.

Two independent placebo constructions are run, because a single one is a single
modelling choice:

*   ``markov``   -- a two-state Markov chain calibrated so that its stationary
                    exposure and its switch rate equal the real ones in
                    expectation, then accepted only if the realised values fall
                    inside a declared tolerance.
*   ``rotation`` -- the real position path rotated circularly by a random
                    offset.  Exposure, turnover, and the whole holding-time
                    distribution are matched *exactly* by construction; only
                    the alignment with the market is destroyed.  This is the
                    "same profile, dates shuffled" placebo the post-study
                    extensions used to refuse the book-level attenuator.

Everything is in excess of the three-month bill (``strategies.book.base_book``).
Reported: matched exposures on both sides, the percentile of the real strategy
in each placebo distribution, a Newey-West lag-6 test of the daily difference
against the deterministic matched-exposure book, and a Politis-Romano
stationary block bootstrap of the paired Sharpe difference.

Read-only: this script writes nothing inside any repository.
"""

from __future__ import annotations

import argparse
import warnings
import zlib

import numpy as np
import pandas as pd
import statsmodels.api as sm

from regime_lab.analysis.bootstrap import stationary_indices
from regime_lab.analysis.power import minimum_detectable_sharpe_difference
from regime_lab.config import CACHE
from regime_lab.evaluation.predictive import volatility_quantile_placebo
from regime_lab.models.mapping import apply_overlay, state_to_position, state_to_size
from regime_lab.strategies.book import base_book
from regime_lab.strategies.costs import charge

warnings.filterwarnings("ignore")

RULE = "=" * 84
PERIODS = 252
HAC_LAG = 6
#: Declared before any draw: a placebo path is kept only if its realised
#: exposure is within 0.5 point of the target and its realised turnover within
#: 15% of it.  Rejected draws are re-drawn, and the acceptance rate is reported.
EXPOSURE_TOL = 0.005
TURNOVER_TOL = 0.15


def sharpe(x: np.ndarray) -> float:
    s = x.std(ddof=1)
    return float(x.mean() / s * np.sqrt(PERIODS)) if s > 0 else np.nan


def alpha_beta(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """Annualised alpha, beta and R-squared of ``y`` on ``x``, closed form."""
    vx = x.var(ddof=1)
    if vx <= 0:
        return np.nan, np.nan, np.nan
    beta = float(np.cov(y, x, ddof=1)[0, 1] / vx)
    alpha = float((y.mean() - beta * x.mean()) * PERIODS)
    r2 = float(np.corrcoef(y, x)[0, 1] ** 2)
    return alpha, beta, r2


def hac_mean_t(diff: np.ndarray, lags: int = HAC_LAG) -> tuple[float, float]:
    """Annualised mean of ``diff`` and its Newey-West t-statistic."""
    model = sm.OLS(diff, np.ones(len(diff))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(model.params[0] * PERIODS), float(model.tvalues[0])


def hac_alpha(y: np.ndarray, x: np.ndarray, lags: int = HAC_LAG) -> tuple[float, float, float]:
    """Annualised alpha, its Newey-West t-statistic, and beta."""
    model = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(model.params[0] * PERIODS), float(model.tvalues[0]), float(model.params[1])


def markov_path(
    n: int, exposure: float, switch: float, rng: np.random.Generator
) -> np.ndarray:
    """One two-state path with stationary exposure ``exposure`` and switch rate ``switch``.

    For a binary chain the mean absolute change per session is
    ``2 p01 p10 / (p01 + p10)`` and the stationary probability of state one is
    ``p01 / (p01 + p10)``.  Inverting the pair gives ``p01 = s / (2 (1 - e))``
    and ``p10 = s / (2 e)``, which is what is drawn here.

    Built from alternating geometric run lengths rather than session by session.
    The geometric distribution is memoryless, so the residual length of the run
    in progress at the start of the sample is geometric with the same parameter:
    starting mid-run introduces no bias, and the draw costs one call instead of
    ``n``.
    """
    p01 = min(switch / (2.0 * (1.0 - exposure)), 1.0)
    p10 = min(switch / (2.0 * exposure), 1.0)

    state = 1 if rng.random() < exposure else 0
    runs: list[np.ndarray] = []
    total = 0
    # Expected number of runs is switch * n; draw them in generous batches.
    batch = max(int(switch * n * 1.5) + 8, 16)
    while total < n:
        lengths_0 = rng.geometric(p01, size=batch)
        lengths_1 = rng.geometric(p10, size=batch)
        for k in range(batch):
            first, second = (
                (lengths_1[k], lengths_0[k]) if state == 1 else (lengths_0[k], lengths_1[k])
            )
            runs.append(np.full(first, float(state)))
            runs.append(np.full(second, 1.0 - float(state)))
            total += int(first) + int(second)
            if total >= n:
                break
    return np.concatenate(runs)[:n]


def rotation_path(
    position: np.ndarray, rng: np.random.Generator, *, margin: int = 252
) -> np.ndarray:
    """The real path rotated circularly by a random offset of at least ``margin`` days."""
    n = len(position)
    offset = int(rng.integers(margin, n - margin))
    return np.roll(position, offset)


def draw_placebos(
    position: np.ndarray,
    *,
    kind: str,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Return ``draws`` matched placebo paths and the number of rejected attempts."""
    n = len(position)
    exposure = float(position.mean())
    switch = float(np.abs(np.diff(position)).mean())
    rng = np.random.default_rng(seed)

    paths = np.empty((draws, n), dtype=np.float64)
    rejected = 0
    kept = 0
    while kept < draws:
        if kind == "markov":
            candidate = markov_path(n, exposure, switch, rng)
        elif kind == "rotation":
            candidate = rotation_path(position, rng)
        else:
            raise ValueError(f"unknown placebo kind {kind!r}")

        e_hat = float(candidate.mean())
        s_hat = float(np.abs(np.diff(candidate)).mean())
        ok_e = abs(e_hat - exposure) <= EXPOSURE_TOL
        ok_s = abs(s_hat - switch) <= max(TURNOVER_TOL * switch, 1.0 / n)
        if ok_e and ok_s:
            paths[kept] = candidate
            kept += 1
        else:
            rejected += 1
            if rejected > 200 * draws:
                raise RuntimeError(f"{kind}: matching tolerance unreachable")
    return paths, rejected


def percentile_of(value: float, distribution: np.ndarray) -> float:
    """Share of the placebo distribution strictly below ``value``, in percent."""
    finite = distribution[np.isfinite(distribution)]
    return float(100.0 * np.mean(finite < value))


def block_bootstrap_sharpe_ci(
    x: np.ndarray, *, mean_block: int = 63, draws: int = 2_000, seed: int = 0
) -> tuple[float, float]:
    """Politis-Romano stationary block bootstrap CI for one leg's Sharpe."""
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for d in range(draws):
        idx = stationary_indices(len(x), mean_block, rng)
        out[d] = sharpe(x[idx])
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    args = parser.parse_args()

    states = pd.read_parquet(CACHE / "states.parquet")
    diagnostics = pd.read_parquet(CACHE / "refit_diagnostics.parquet")
    book = base_book()

    oos = states.dropna(how="all").index
    base = book.reindex(oos).dropna()
    oos = base.index
    base_np = base.to_numpy()

    print(RULE)
    print(
        f"T3  matched-exposure placebo, {len(oos):,} out-of-sample sessions "
        f"({oos.min():%Y-%m} to {oos.max():%Y-%m})"
    )
    print(RULE)
    print(
        f"\n   excess of the 3m bill | {args.draws:,} placebo paths per construction | "
        f"HAC lag {HAC_LAG} | seed {args.seed}"
    )
    print(f"   base book, held in full: Sharpe {sharpe(base_np):.2f}, "
          f"vol {base.std(ddof=1) * np.sqrt(PERIODS):.1%}\n")

    # Positions, not states: the on/off rule of the charter, the sizing rule the
    # study's own conclusion points to, and the one-line volatility-quantile rule
    # from evaluation/predictive.py.  The one-liner is not a family; it is the
    # reference that says how much of a T3 pass is available with no model at
    # all.  Its expanding median is computed on the full book, so it sees the
    # same pre-2002 history the models trained on, and stays causal.
    train_vol = float(book.loc[: oos.min()].std() * np.sqrt(PERIODS))
    signals: dict[str, pd.Series] = {}
    for column in states.columns:
        signals[f"{column} | on/off"] = state_to_position(states[column]).reindex(oos).fillna(0.0)
        diag = diagnostics[diagnostics["family"] == column].set_index("refit").sort_index()
        vol_columns = [c for c in diag.columns if c.startswith("state_vol_")]
        if vol_columns:
            signals[f"{column} | sized"] = (
                state_to_size(states[column], diag[vol_columns], target=train_vol)
                .reindex(oos)
                .fillna(0.0)
            )
    signals["- one-line vol quantile | on/off"] = (
        state_to_position(volatility_quantile_placebo(book)).reindex(oos).fillna(0.0)
    )

    summary_rows: list[dict[str, object]] = []

    for family, position in signals.items():
        overlay = apply_overlay(base, position).dropna()
        common = overlay.index
        pos_np = position.reindex(common).to_numpy()
        ovl_np = overlay.to_numpy()
        bse_np = base.reindex(common).to_numpy()

        e_real = float(pos_np.mean())
        s_real = float(np.abs(np.diff(pos_np)).mean())
        ac1_real = float(pd.Series(pos_np).autocorr(1))

        sr_real = sharpe(ovl_np)
        mean_real = float(ovl_np.mean() * PERIODS)
        a_real, ta_real, b_real = hac_alpha(ovl_np, bse_np)
        r2_real = float(np.corrcoef(ovl_np, bse_np)[0, 1] ** 2)

        net_real = charge(overlay, position.reindex(common), bps=args.cost_bps).to_numpy()
        sr_real_net = sharpe(net_real)

        print(RULE)
        print(f"\n{family}\n")
        changes = int((np.diff(pos_np) != 0).sum())
        print(f"   real overlay   exposure {e_real:.4f}   turnover {s_real:.4f}   "
              f"lag-1 autocorr {ac1_real:.4f}   position changes {changes:,}")
        print(f"   the profile carries {changes:,} independent decisions, not "
              f"{len(pos_np):,} -- that is what the placebo distribution is wide about")
        print(f"   real overlay   Sharpe {sr_real:.3f}   mean {mean_real:+.2%}/yr   "
              f"alpha vs base {a_real:+.2%}/yr (t {ta_real:+.2f})  beta {b_real:.2f}  "
              f"R2 {r2_real:.0%}")

        # The deterministic matched-exposure benchmark, kept for the record: it
        # is beta-matched but Sharpe-identical to the base book by construction.
        constant = e_real * bse_np
        d_mean, d_t = hac_mean_t(ovl_np - constant)
        print(f"\n   constant-exposure book at {e_real:.4f} of the base: "
              f"Sharpe {sharpe(constant):.3f} (= base book, scale invariance)")
        print(f"   daily difference overlay - constant-exposure: "
              f"{d_mean:+.2%}/yr, Newey-West lag-{HAC_LAG} t {d_t:+.2f}")

        binary = bool(np.isin(pos_np, (0.0, 1.0)).all())
        kinds = ("markov", "rotation") if binary else ("rotation",)
        if not binary:
            print(f"\n   continuous profile (mean leverage {e_real:.4f}, "
                  f"min {pos_np.min():.2f}, max {pos_np.max():.2f}): the two-state Markov\n"
                  "   construction does not apply; the rotation placebo matches mean leverage,\n"
                  "   turnover and the whole leverage distribution exactly.")
        for kind in kinds:
            # Deterministic per-family seed: ``hash`` on str is salted per process.
            family_seed = args.seed + zlib.crc32(family.encode()) % 100_000
            paths, rejected = draw_placebos(
                pos_np, kind=kind, draws=args.draws, seed=family_seed
            )
            e_hat = paths.mean(axis=1)
            s_hat = np.abs(np.diff(paths, axis=1)).mean(axis=1)

            returns = paths * bse_np[None, :]
            sr = np.array([sharpe(r) for r in returns])
            mn = returns.mean(axis=1) * PERIODS
            alphas = np.array([alpha_beta(r, bse_np)[0] for r in returns])
            betas = np.array([alpha_beta(r, bse_np)[1] for r in returns])

            traded = np.abs(np.diff(paths, axis=1, prepend=paths[:, :1]))
            net = returns - traded * args.cost_bps / 10_000.0
            sr_net = np.array([sharpe(r) for r in net])

            print(f"\n   placebo [{kind}]  accepted {args.draws:,} paths, "
                  f"rejected {rejected:,} "
                  f"({100 * args.draws / (args.draws + rejected):.0f}% acceptance)")
            print(f"      exposure  real {e_real:.4f}   placebo {e_hat.mean():.4f} "
                  f"+/- {e_hat.std(ddof=1):.4f}   max gap {np.abs(e_hat - e_real).max():.4f}")
            print(f"      turnover  real {s_real:.4f}   placebo {s_hat.mean():.4f} "
                  f"+/- {s_hat.std(ddof=1):.4f}   max gap {np.abs(s_hat - s_real).max():.4f}")
            print(f"      Sharpe    real {sr_real:.3f}   placebo {sr.mean():.3f} "
                  f"+/- {sr.std(ddof=1):.3f}   p95 {np.percentile(sr, 95):.3f}"
                  f"   -> real at the {percentile_of(sr_real, sr):.0f}th percentile")
            print(f"      mean      real {mean_real:+.2%}   placebo {mn.mean():+.2%} "
                  f"+/- {mn.std(ddof=1):.2%}"
                  f"   -> real at the {percentile_of(mean_real, mn):.0f}th percentile")
            print(f"      alpha     real {a_real:+.2%}   placebo {alphas.mean():+.2%} "
                  f"+/- {alphas.std(ddof=1):.2%}   beta {betas.mean():.2f}"
                  f"   -> real at the {percentile_of(a_real, alphas):.0f}th percentile")
            print(f"      Sharpe net {args.cost_bps:.0f}bp  real {sr_real_net:.3f}   "
                  f"placebo {sr_net.mean():.3f}"
                  f"   -> real at the {percentile_of(sr_real_net, sr_net):.0f}th percentile")
            print(f"      beta      real {b_real:.3f}   placebo {betas.mean():.3f} "
                  f"+/- {betas.std(ddof=1):.3f}"
                  f"   -> real at the {percentile_of(b_real, betas):.0f}th percentile")

            # Average placebo path: the matched-exposure benchmark the draws
            # actually define, rather than the constant-exposure idealisation.
            pool = returns.mean(axis=0)
            pool_mean, pool_t = hac_mean_t(ovl_np - pool)
            print(f"      daily difference overlay - mean placebo: {pool_mean:+.2%}/yr, "
                  f"Newey-West lag-{HAC_LAG} t {pool_t:+.2f}")

            # Representative path picked on matching quality, never on outcome.
            distance = np.abs(e_hat - e_real) / max(e_real, 1e-9) + np.abs(s_hat - s_real) / max(
                s_real, 1e-9
            )
            rep = int(np.argmin(distance))
            power = minimum_detectable_sharpe_difference(
                ovl_np, returns[rep], mean_block=63, draws=1_000, seed=args.seed
            )
            print(f"      paired block bootstrap vs the best-matched path: "
                  f"observed {power.observed:+.3f}  MDE {power.mde:.3f}  "
                  f"-> {'DETECTED' if abs(power.observed) > power.mde else 'inside the noise'}")

            summary_rows.append(
                {
                    "family": family,
                    "placebo": kind,
                    "exposure_real": e_real,
                    "exposure_placebo": float(e_hat.mean()),
                    "turnover_real": s_real,
                    "turnover_placebo": float(s_hat.mean()),
                    "sharpe_real": sr_real,
                    "sharpe_placebo_mean": float(sr.mean()),
                    "sharpe_pct": percentile_of(sr_real, sr),
                    "alpha_real": a_real,
                    "alpha_t_hac6": ta_real,
                    "alpha_pct": percentile_of(a_real, alphas),
                    "beta_real": b_real,
                    "r2_real": r2_real,
                    "sharpe_net_pct": percentile_of(sr_real_net, sr_net),
                    "mean_pct": percentile_of(mean_real, mn),
                    "beta_placebo": float(betas.mean()),
                    "beta_pct": percentile_of(b_real, betas),
                    "diff_vs_pool_annual": pool_mean,
                    "diff_vs_pool_t_hac6": pool_t,
                    "mde_paired": power.mde,
                    "observed_paired": power.observed,
                }
            )

        lo, hi = block_bootstrap_sharpe_ci(ovl_np, draws=1_000, seed=args.seed)
        print(f"\n   block bootstrap CI on the real overlay's own Sharpe: "
              f"[{lo:.2f}, {hi:.2f}]")

    print("\n" + RULE)
    print("\nSUMMARY  percentile of the real strategy in its own matched placebo\n")
    print(f"   {'signal':<34} {'placebo':<10} {'SR real':>8} {'SR plac':>8} "
          f"{'pct SR':>7} {'pct mean':>9} {'pct alpha':>10} {'pct net':>8} {'t pool':>8}")
    for row in summary_rows:
        print(f"   {row['family']:<34} {row['placebo']:<10} {row['sharpe_real']:>8.3f} "
              f"{row['sharpe_placebo_mean']:>8.3f} {row['sharpe_pct']:>6.0f}th "
              f"{row['mean_pct']:>8.0f}th {row['alpha_pct']:>9.0f}th "
              f"{row['sharpe_net_pct']:>7.0f}th {row['diff_vs_pool_t_hac6']:>+8.2f}")

    frame = pd.DataFrame(summary_rows)
    out = CACHE / "t3_summary.csv"
    frame.to_csv(out, index=False)
    print(f"\n   written: {out}")
    print("\n" + RULE)


if __name__ == "__main__":
    main()
