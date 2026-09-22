"""A matched placebo for any partition: the same clock, the same occupancy, no content.

A conditional claim — "state k does X" — needs a null that shares everything with
the real partition except its content. T3 showed what happens otherwise: a gain can
travel entirely through a dimension the null left unmatched. The two dimensions every
partition in this programme has, and that any placebo must reproduce, are the **clock**
(how many transitions) and the **occupancy** (what share of sessions each state holds).

Every construction kept here matches both **exactly**, and more: it keeps every
episode as a ``(length, state)`` pair, so each state keeps its own dwell-time
distribution too. What it deliberately does not keep is *when* the episodes happen
and *which state follows which* — the calendar placement and the transition matrix
are the content the null removes.

The Two Sigma plan (`PRESPEC_TWOSIGMA.md` §8, P6) took three attempts. The two
rejected ones are written down so they are not rebuilt; both were measured by the
pre-lock script on the toy library's sample of 7,952 sessions (1994-12-23 to
2026-07-31), where the real K=4 context partition has 214 transitions, 6.77 a year —
not on the 34.52-year context sample whose 6.46 the draft's §8 table sets beside them.

1. **Permute the ``(length, state)`` pairs, nothing else** — rejected. Whenever two
   neighbours in the shuffled order share a state they merge into one episode, and
   the transition is lost. Measured: 4.79 transitions a year against 6.77, **29% of
   the clock gone**. Occupancy exact (0.0000), since the pairs are intact.
2. **Permute lengths and states independently, then forbid a repeat across a join**
   (swap in a later state, or draw one at random when none is left) — rejected. The
   clock is right (6.77 against 6.77) but a state no longer keeps its own lengths — a
   rare state can inherit the longest episodes — and the random fallback changes the
   state multiset. Occupancy broken by up to **0.3043**.
3. **Permute the pairs, then repair each same-state join by swapping whole pairs**
   (``method="swap"``) — adopted by the draft. Pairs are never split, so the
   ``(length, state)`` multiset and therefore the occupancy are exact by construction;
   a completed repair leaves no two neighbours in the same state, so there are exactly
   ``episodes - 1`` transitions. On the disclosed partition (223 transitions over
   34.52 years, one block): 300 draws of 300, 223 transitions in each.

**The swap repair is greedy, and it fails where the draft never looked.** For a join
at position ``i`` it takes the first later pair whose state differs from both
neighbours of ``i`` and whose own new neighbours differ from the displaced state.
That works when no state holds much more than a third of the episodes. It fails when
one state holds close to half of them — the partition then has almost no freedom (at
exactly half, a single state must occupy every other slot) and a swap-by-swap repair
of a random permutation rarely reaches it. With two states, strict alternation is the
only join-free order, and the failure rate grows with the block: measured over 200
draws of 60 permutations each, 0 failures at 8 episodes, 23 at 16, 132 at 24, 183 at
30, 199 at 60. The declared object is in that regime: once the walk-forward labels
are cut into (fold, segment) blocks, a "normal" state that sits between every
excursion holds 45-50% of a block's episodes. At K=4 the swap repair completes 4
draws in 1,000 (36 when the context training starts in 1992), at K=3 none
(`scripts/measure_twosigma_context.py`, section D). Retrying a fresh permutation
already conditions the null on the permutations the greedy can repair; dropping a
draw after ``max_tries`` failures, as the pre-lock script did, conditions it further
and silently. This module **raises** :class:`PlaceboError` instead, and
:func:`placebo_feasibility` measures how often that happens.

4. **Draw the episode order uniformly among the join-free ones** (``method="uniform"``)
   — **proposed, not adopted**: the lock must choose. The state sequence is drawn
   uniformly among all sequences with the real per-state episode counts and no two
   equal neighbours, then each state's own lengths are permuted uniformly among its
   slots. The result is *exactly* a uniform permutation of the ``(length, state)``
   pairs conditioned on having no join — attempt 1 conditioned on the clock, which is
   the null the swap repair approximates. It never fails when a join-free order
   exists (the real partition is one), needs no retry, and conditions on nothing
   else. The count of join-free sequences comes from an insertion recursion: insert
   the states one at a time into the sequence of the previous ones, tracking how many
   same-state neighbours remain to be broken; the draw walks that recursion backwards.
   Clock, occupancy and episode multiset are exact by construction and verified on
   every draw.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.special import gammaln

#: Fresh permutations tried per draw before giving up, the pre-lock script's value.
MAX_TRIES = 60

Method = Literal["swap", "uniform"]


class PlaceboError(RuntimeError):
    """No join-free arrangement was produced: the swap repair gave up, or none exists."""


def _codes(labels: pd.Series) -> tuple[np.ndarray, pd.Index]:
    if labels.isna().any():
        raise ValueError("labels contain missing values; a partition must label every session")
    codes, uniques = pd.factorize(labels, sort=True)
    return codes.astype(np.int64), pd.Index(uniques)


def run_lengths(labels: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Episode lengths and episode states, in order.

    An episode is a maximal run of consecutive sessions in the same state.
    """
    values = np.asarray(labels)
    if values.size == 0:
        return np.empty(0, dtype=np.int64), values[:0]
    starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    lengths = np.diff(np.r_[starts, values.size])
    return lengths.astype(np.int64), values[starts]


def transitions(labels: pd.Series | np.ndarray) -> int:
    """Number of sessions whose state differs from the previous session's."""
    values = np.asarray(labels)
    return int((values[1:] != values[:-1]).sum()) if values.size else 0


def episodes(labels: pd.Series) -> pd.DataFrame:
    """One row per episode: its state, first and last session, and length."""
    lengths, states = run_lengths(labels)
    starts = np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)
    ends = starts + lengths - 1
    return pd.DataFrame({
        "state": states,
        "start": labels.index[starts],
        "end": labels.index[ends],
        "length": lengths,
    })


def occupancy(labels: pd.Series | np.ndarray) -> pd.Series:
    """Share of sessions spent in each state, sorted by state."""
    return pd.Series(np.asarray(labels)).value_counts(normalize=True).sort_index()


def span_years(index: pd.DatetimeIndex) -> float:
    """Calendar length of an index in years of 365.25 days, first to last session."""
    return float((index.max() - index.min()).days / 365.25) if len(index) else float("nan")


# ------------------------------------------------------------- construction 3: swap


def _arrange(
    lengths: np.ndarray, states: np.ndarray, rng: np.random.Generator, max_tries: int
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """The draft's construction: permute pairs, repair joins by swapping whole pairs.

    A line-for-line port of the pre-lock script: the same generator calls, so the
    same seed gives the same arrangement. Returns the arranged lengths, states and
    the number of permutations used, or ``None`` if no permutation was repaired.
    """
    n = len(states)
    for attempt in range(1, max_tries + 1):
        order = rng.permutation(n)
        arranged_lengths, arranged = lengths[order].copy(), states[order].copy()
        for i in range(1, n):
            if arranged[i] != arranged[i - 1]:
                continue
            after = arranged[i + 1] if i + 1 < n else -1
            j = next(
                (
                    k for k in range(i + 1, n)
                    if arranged[k] != arranged[i - 1]
                    and arranged[k] != after
                    and (k + 1 >= n or arranged[k + 1] != arranged[i])
                    and arranged[k - 1] != arranged[i]
                ),
                None,
            )
            if j is None:
                break
            arranged_lengths[[i, j]] = arranged_lengths[[j, i]]
            arranged[[i, j]] = arranged[[j, i]]
        if not (arranged[1:] == arranged[:-1]).any():
            return arranged_lengths, arranged, attempt
    return None


# ---------------------------------------------------------- construction 4: uniform


def _log_comb(n: np.ndarray | int, k: np.ndarray | int) -> np.ndarray:
    """``log C(n, k)``, ``-inf`` wherever the binomial is zero."""
    n, k = np.broadcast_arrays(np.asarray(n, dtype=float), np.asarray(k, dtype=float))
    out = np.full(n.shape, -np.inf)
    ok = (k >= 0) & (k <= n)
    out[ok] = gammaln(n[ok] + 1.0) - gammaln(k[ok] + 1.0) - gammaln(n[ok] - k[ok] + 1.0)
    return out


def _grouped_logsumexp(values: np.ndarray, groups: np.ndarray, size: int) -> np.ndarray:
    out = np.full(size, -np.inf)
    finite = np.isfinite(values)
    values, groups = values[finite], groups[finite]
    if values.size == 0:
        return out
    top = np.full(size, -np.inf)
    np.maximum.at(top, groups, values)
    sums = np.bincount(groups, weights=np.exp(values - top[groups]), minlength=size)
    hit = sums > 0
    out[hit] = top[hit] + np.log(sums[hit])
    return out


class JoinFreeSequences:
    """Uniform sampler of sequences with fixed letter counts and no two equal neighbours.

    Letters are inserted one kind at a time into the sequence of the kinds before
    them. A sequence of the first kinds may hold ``b`` "bad" slots — two equal
    neighbours — which later kinds must break. Inserting ``c`` copies of a new
    letter as ``k`` runs, ``j`` of them into bad slots and ``k - j`` into the other
    ``L + 1 - b`` slots, can be done in ``C(c-1, k-1) C(b, j) C(L+1-b, k-j)`` ways
    and leaves ``b - j + c - k`` bad slots. Every final sequence arises from exactly
    one chain of such choices, so the forward recursion counts the join-free
    sequences (``b = 0`` at the end) and a backward walk through it samples one
    uniformly. Weights are kept in logs: the counts reach 10^240 on this
    programme's partitions.
    """

    def __init__(self, counts: Sequence[int]):
        counts = np.asarray(counts, dtype=np.int64)
        if (counts < 0).any():
            raise ValueError("letter counts must be non-negative")
        self.letters = np.flatnonzero(counts > 0)
        self.counts = counts[self.letters]
        self.offsets = np.r_[0, np.cumsum(self.counts)[:-1]].astype(np.int64)
        log_f = [np.zeros(1)]
        for c, length in zip(self.counts, self.offsets, strict=True):
            previous = log_f[-1]
            b = np.arange(len(previous))[:, None, None]
            k = np.arange(1, c + 1)[None, :, None]
            j = np.arange(0, c + 1)[None, None, :]
            weight = (
                previous[:, None, None]
                + _log_comb(c - 1, k - 1)
                + _log_comb(b, j)
                + _log_comb(length + 1 - b, k - j)
            )
            after = np.broadcast_to(b - j + c - k, weight.shape)
            size = int(length + c)
            valid = np.isfinite(weight)
            log_f.append(_grouped_logsumexp(weight[valid], after[valid], size))
        self._log_f = log_f

    @property
    def log_count(self) -> float:
        """Natural log of the number of join-free sequences; ``-inf`` if there is none."""
        return float(self._log_f[-1][0]) if self.counts.size else 0.0

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """One join-free sequence of letters, uniformly among all of them."""
        if not np.isfinite(self.log_count):
            counts = dict(zip(self.letters.tolist(), self.counts.tolist(), strict=True))
            raise PlaceboError(f"no sequence with counts {counts} avoids two equal neighbours")
        plan: list[tuple[int, int]] = []
        target = 0
        for i in range(len(self.counts) - 1, -1, -1):
            c, length, previous = int(self.counts[i]), int(self.offsets[i]), self._log_f[i]
            k = np.arange(1, c + 1)[:, None]
            j = np.arange(0, c + 1)[None, :]
            b = target + j + k - c
            inside = (b >= 0) & (b < len(previous))
            safe = np.where(inside, b, 0)
            weight = np.where(
                inside,
                previous[safe]
                + _log_comb(c - 1, k - 1)
                + _log_comb(safe, j)
                + _log_comb(length + 1 - safe, k - j),
                -np.inf,
            )
            p = np.exp(weight - weight.max()).ravel()
            pick = int(rng.choice(p.size, p=p / p.sum()))
            ki, ji = np.unravel_index(pick, weight.shape)
            plan.append((int(ki) + 1, int(ji)))
            target = int(b[ki, ji])
        plan.reverse()

        word = np.empty(0, dtype=np.int64)
        for letter, c, (k, j) in zip(self.letters, self.counts, plan, strict=True):
            bad = np.flatnonzero(word[1:] == word[:-1]) + 1
            good = np.setdiff1d(np.arange(word.size + 1), bad)
            slots = np.sort(np.r_[
                rng.choice(bad, j, replace=False), rng.choice(good, k - j, replace=False)
            ]).astype(np.int64)
            cuts = np.sort(rng.choice(np.arange(1, c), k - 1, replace=False))
            parts = np.diff(np.r_[0, cuts, c]).astype(np.int64)
            word = np.insert(word, np.repeat(slots, parts), letter)
        return word


def _uniform_arrange(
    lengths: np.ndarray,
    states: np.ndarray,
    sampler: JoinFreeSequences,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    order = sampler.sample(rng)
    arranged_lengths = np.empty_like(lengths)
    for state in np.unique(states):
        own = lengths[states == state]
        arranged_lengths[order == state] = own[rng.permutation(own.size)]
    return arranged_lengths, order


# ------------------------------------------------------------------------- drawing


@dataclass(frozen=True)
class MatchedPlacebo:
    """Placebo partitions, one column per draw, and how hard each was to build.

    ``draws`` has the input's index and label values. ``tries`` counts the fresh
    permutations each draw needed (1 = the first repair succeeded; always 1 for
    ``method="uniform"``); with groups, it is the largest count over the groups.
    """

    draws: pd.DataFrame
    tries: np.ndarray
    method: str = "swap"


def _contiguous_groups(groups: pd.Series, index: pd.Index) -> list[np.ndarray]:
    aligned = groups.reindex(index)
    if aligned.isna().any():
        raise ValueError("groups must label every session of the partition")
    values = aligned.to_numpy()
    starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    if len(starts) != len(pd.unique(values)):
        raise ValueError("each group must be one contiguous block of sessions")
    bounds = np.r_[starts, len(values)]
    return [np.arange(bounds[g], bounds[g + 1]) for g in range(len(starts))]


def _blocks(labels: pd.Series, groups: pd.Series | None, n: int) -> list[np.ndarray]:
    return [np.arange(n)] if groups is None else _contiguous_groups(groups, labels.index)


def matched_placebos(
    labels: pd.Series,
    n_draws: int,
    *,
    seed: int = 0,
    max_tries: int = MAX_TRIES,
    groups: pd.Series | None = None,
    method: Method = "swap",
) -> MatchedPlacebo:
    """Draw partitions with the real clock and occupancy and no calendar content.

    Every draw keeps the multiset of ``(length, state)`` episodes of ``labels``, so
    the occupancy and each state's dwell-time distribution are exact, and has no two
    consecutive episodes in the same state, so the transition count is exact.

    ``method="swap"`` is the draft's construction 3; ``method="uniform"`` is the
    proposed construction 4 (module docstring). The lock chooses.

    ``groups`` — for example the (fold, segment) of each session — makes the placebo
    match **within** each contiguous group: episodes are permuted inside their group
    only, which is what labels refitted per fold need, since K-means numbers its
    clusters arbitrarily at each refit. Transitions and occupancy are then exact
    group by group; a transition across a group boundary is not controlled.

    Draw ``d`` depends only on ``seed`` and ``d``, not on ``n_draws``.

    Raises :class:`PlaceboError` when a swap draw cannot be repaired within
    ``max_tries`` permutations — :func:`placebo_feasibility` measures how often.
    """
    if n_draws < 1:
        raise ValueError("n_draws must be at least 1")
    if max_tries < 1:
        raise ValueError("max_tries must be at least 1")
    if method not in ("swap", "uniform"):
        raise ValueError(f"method must be 'swap' or 'uniform', not {method!r}")
    codes, uniques = _codes(labels)
    blocks = _blocks(labels, groups, len(codes))
    pieces = [run_lengths(codes[block]) for block in blocks]
    samplers = (
        [JoinFreeSequences(np.bincount(s, minlength=len(uniques))) for _, s in pieces]
        if method == "uniform" else None
    )

    out = np.empty((len(codes), n_draws), dtype=np.int64)
    tries = np.empty(n_draws, dtype=np.int64)
    for d, child in enumerate(np.random.SeedSequence(seed).spawn(n_draws)):
        rng = np.random.default_rng(child)
        worst = 0
        for b, (block, (lengths, states)) in enumerate(zip(blocks, pieces, strict=True)):
            if samplers is not None:
                arranged_lengths, arranged_states = _uniform_arrange(
                    lengths, states, samplers[b], rng
                )
                used = 1
            else:
                arranged = _arrange(lengths, states, rng, max_tries)
                if arranged is None:
                    raise PlaceboError(
                        f"draw {d}: no join-free arrangement of {len(states)} episodes over "
                        f"{len(np.unique(states))} states within {max_tries} permutations"
                    )
                arranged_lengths, arranged_states, used = arranged
            path = np.repeat(arranged_states, arranged_lengths)
            if (
                path.size != block.size
                or (arranged_states[1:] == arranged_states[:-1]).any()
                or not np.array_equal(np.bincount(path, minlength=len(uniques)),
                                      np.bincount(codes[block], minlength=len(uniques)))
            ):
                raise RuntimeError(f"draw {d}: the construction broke the clock or occupancy")
            out[block, d] = path
            worst = max(worst, used)
        tries[d] = worst

    values = uniques.to_numpy()[out]
    draws = pd.DataFrame(values, index=labels.index, columns=pd.RangeIndex(n_draws, name="draw"))
    return MatchedPlacebo(draws=draws, tries=tries, method=method)


def placebo_feasibility(
    labels: pd.Series,
    n_draws: int,
    *,
    seed: int = 0,
    max_tries: int = MAX_TRIES,
    groups: pd.Series | None = None,
) -> np.ndarray:
    """How often the swap repair succeeds: permutations used per draw and block.

    Returns an ``(n_draws, n_blocks)`` array, 0 where the block could not be
    repaired within ``max_tries``. It walks the same random streams as
    :func:`matched_placebos` with ``method="swap"``, so every draw with no zero here
    is the draw that function returns; unlike it, this one does not stop at the
    first failure.
    """
    if n_draws < 1:
        raise ValueError("n_draws must be at least 1")
    codes, _ = _codes(labels)
    blocks = _blocks(labels, groups, len(codes))
    pieces = [run_lengths(codes[block]) for block in blocks]
    used = np.zeros((n_draws, len(blocks)), dtype=np.int64)
    for d, child in enumerate(np.random.SeedSequence(seed).spawn(n_draws)):
        rng = np.random.default_rng(child)
        for b, (lengths, states) in enumerate(pieces):
            arranged = _arrange(lengths, states, rng, max_tries)
            used[d, b] = 0 if arranged is None else arranged[2]
    return used


# ------------------------------------------------------------------------ checking


@dataclass(frozen=True)
class PlaceboCheck:
    """How closely a set of placebo draws matches the real partition."""

    real_transitions: int
    transitions: np.ndarray
    max_occupancy_deviation: float
    episodes_preserved: bool

    @property
    def transitions_sd(self) -> float:
        """Standard deviation of the transition count across draws; 0 when exact."""
        return float(self.transitions.std())

    @property
    def exact(self) -> bool:
        """Clock, occupancy and episode multiset all reproduced in every draw."""
        return bool(
            (self.transitions == self.real_transitions).all()
            and self.max_occupancy_deviation == 0.0
            and self.episodes_preserved
        )


def _multiset(lengths: np.ndarray, states: np.ndarray) -> list[tuple[object, int]]:
    return sorted(zip(states.tolist(), lengths.tolist(), strict=True), key=repr)


def check_placebos(
    labels: pd.Series, draws: pd.DataFrame, *, groups: pd.Series | None = None
) -> PlaceboCheck:
    """Compare every draw to the real partition on clock, occupancy and episodes.

    With ``groups``, transitions are counted within groups only and occupancy and the
    episode multiset are compared group by group — the quantities
    :func:`matched_placebos` holds exact when it is given the same groups.
    """
    blocks = _blocks(labels, groups, len(labels))
    real = labels.to_numpy()

    def clock(values: np.ndarray) -> int:
        return sum(transitions(values[block]) for block in blocks)

    counts = np.array([clock(draws[c].to_numpy()) for c in draws.columns], dtype=np.int64)
    deviation = 0.0
    preserved = True
    for block in blocks:
        truth = occupancy(real[block])
        truth_runs = _multiset(*run_lengths(real[block]))
        for column in draws.columns:
            values = draws[column].to_numpy()[block]
            drawn = occupancy(values).reindex(truth.index, fill_value=0.0)
            extra = set(pd.unique(values)) - set(truth.index)
            deviation = max(deviation, float((drawn - truth).abs().max()), float(bool(extra)))
            preserved = preserved and _multiset(*run_lengths(values)) == truth_runs
    return PlaceboCheck(
        real_transitions=clock(real),
        transitions=counts,
        max_occupancy_deviation=deviation,
        episodes_preserved=preserved,
    )
