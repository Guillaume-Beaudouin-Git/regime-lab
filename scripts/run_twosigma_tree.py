"""The Two Sigma tree: instruments, readings, Holm and sensitivities — one CLI (§13.1).

The specification is `docs/PRESPEC_TWOSIGMA.md`, **LOCKED at commit 30f7d69**. Its §13.1
fixes the order this script enforces:

    instrument       §13.1 steps 1-3. Runs the instruments of A, B and C for the primary
                     and for every §12.13 sensitivity (K=3, K=5, K=6 and smooth21 at the
                     three levels; d=0.25 and d=1.00 at A-1), prints only what §13.1
                     step 2 allows, writes every threshold to
                     docs/artifacts/twosigma/thresholds.json through the core
                     (tree.write_thresholds: floats bitwise, input SHA-256, uv.lock
                     versions) and the INSTRUMENT section of
                     docs/RESULTS_TWOSIGMA_LEVEL_{A,B,C}.md. Nothing of the tree is read.
                     Refuses to overwrite a threshold file that git already tracks.
    read A|B|C       §13.1 step 4, A then B then C. Each level's module recomputes its
                     instrument and refuses (tree.ReadingRefused) unless the threshold
                     file is tracked and unmodified at HEAD, the inputs' SHA-256 and the
                     package versions match, and every threshold is bitwise equal. Then
                     the reading, the trial rows of §13.5, the reading artifact
                     docs/artifacts/twosigma/reading_<level>.json and the READING section
                     of the level's results file. B needs A's reading committed, C
                     needs B's; a level is read once.
    holm             §13.1 step 5, §13.3: after C, the Holm step over the six primaries
                     from the three committed reading artifacts, the final lock and level
                     verdicts, docs/RESULTS_TWOSIGMA.md and docs/artifacts/twosigma/
                     holm.json. Logs no row (§13.5).
    sensitivities    §13.1 step 5, §12.13: after the committed Holm step, the 14
                     sensitivity readings, their trial rows (with the PASS (not robust)
                     note against the final primary level verdicts) and the SENSITIVITIES
                     sections. Resumable: a sensitivity already read is never re-read.

The placebo loops run in the core's deterministic process pool (``tree.draw_map``,
``--workers``, default the CPU count); every result is bitwise the same whatever the
worker count, which the bitwise check of the reading relies on.

Blindness (§11, §13.1 step 2). ``instrument`` prints and writes the level instruments'
printouts and nothing else: counts, turnover of the arms each instrument builds, the kill
line, realised sd and cap share, demeaned-leg MDEs, T_A1, T_B2, MDE_C, S*, the null
percentiles of A-2, B-1 and C-1, and the NFCI diagnostic. It never calls a reading.

    .venv/bin/python scripts/run_twosigma_tree.py instrument
    .venv/bin/python scripts/run_twosigma_tree.py read A      # a trial: the lead only
    .venv/bin/python scripts/run_twosigma_tree.py read B
    .venv/bin/python scripts/run_twosigma_tree.py read C
    .venv/bin/python scripts/run_twosigma_tree.py holm
    .venv/bin/python scripts/run_twosigma_tree.py sensitivities
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from regime_lab.analysis import trials
from regime_lab.config import ROOT
from regime_lab.selection import level_a, level_b, level_c, tree
from regime_lab.selection.tree import (
    BOOT_DRAWS,
    D025,
    D100,
    INPUT_FILES,
    K3,
    K5,
    K6,
    N_DRAWS,
    PASS,
    PLACEBO_SEED,
    PRIMARIES,
    PRIMARY,
    SMOOTH21,
    THRESHOLDS_FILE,
    UNDECIDABLE,
    ReadingRefused,
    TreeData,
    Variant,
)

Emit = Callable[[str], None]

LEVELS: tuple[str, ...] = ("A", "B", "C")
#: The primary rows each level's reading logs (§13.5).
LEVEL_TESTS: dict[str, tuple[str, ...]] = {"A": ("A-1", "A-2"), "B": ("B-1", "B-2"),
                                           "C": ("C-1", "C-2")}
MODULES: dict[str, Any] = {"A": level_a, "B": level_b, "C": level_c}
TITLES = {
    "A": "level A — the first-moment channel (§6, §12.5-§12.7)",
    "B": "level B — the second-moment channel (§6, §12.8-§12.9)",
    "C": "level C — the cost channel (§6, §12.10)",
}
#: §7, §12.13: the variants whose instruments are committed before A is read, in the
#: order they run (the d rows share the primary's partition and placebo).
INSTRUMENT_VARIANTS: tuple[Variant, ...] = (PRIMARY, K3, K5, K6, SMOOTH21, D025, D100)
SENSITIVITIES: tuple[Variant, ...] = (K3, K5, K6, SMOOTH21, D025, D100)
ARTIFACTS = "docs/artifacts/twosigma"
RESULTS_FILE = "docs/RESULTS_TWOSIGMA_LEVEL_{level}.md"
SUMMARY_FILE = "docs/RESULTS_TWOSIGMA.md"
READING_FILE = f"{ARTIFACTS}/reading_{{level}}.json"
HOLM_FILE = f"{ARTIFACTS}/holm.json"
SENSITIVITY_FILE = f"{ARTIFACTS}/sensitivities.json"
BLOCKS = ("INSTRUMENT", "READING", "SENSITIVITIES")
#: §13.5 (second validator): every results document that quotes the logged Sharpe says so.
SHARPE_CAVEAT = (
    "The logged `sharpe` is that of a near-zero-net long-short book of industry "
    "portfolios with no borrow cost. It is not a measure of progress toward the "
    "programme's ambition of a net Sharpe of 1 to 2 (§13.5)."
)
NOTHING_READ = (
    "**Nothing of the tree was read.** The instrument printed no mean, Sharpe, IC or "
    "per-state statistic of the real partition or of any real arm, and no Sharpe "
    "difference or beta of any placebo arm (§13.1 step 2). It computed none of the "
    "reading's statistics: no Δ_A1 or Δ_B2, no p-value, no `R`, `G` or `S_C`, no interval "
    "`h_fk`, no gate. It computed `m` and the state second-moment matrices, which the "
    "arms' demeaned legs need, and printed neither. What it printed is what §13.1 step 2 "
    "allows: counts, the turnover of the arms it builds with the kill line, realised sd "
    "and cap share, the MDEs of demeaned legs with T_A1, T_B2, MDE_C and S*, the null "
    "percentiles of A-2, B-1 and C-1 (real returns under content-free partitions, never "
    "the real labels), and the NFCI diagnostic (labels only)."
)


class Criterion(NamedTuple):
    """A lock's criterion, copied verbatim from the lock (tested against its text)."""

    level: str
    lock: str
    section: str
    text: tuple[str, ...]


#: The criteria of every lock and of §13.2-§13.4, copied verbatim from the lock's lines
#: (tests/test_twosigma_script.py checks each against docs/PRESPEC_TWOSIGMA.md).
CRITERIA: tuple[Criterion, ...] = (
    Criterion('A', 'A-1', '§12.6', (
r"""**Statistic.** `Δ_A1 = SR(selector) − SR(control)` over the 5,031 pooled test sessions.""",
r"""2. **Threshold:** `T_A1 = max(0.338, max_b MDE(0.05/6, b))`.""",
r"""3. **Kill turnover** *(lock, forced by measurement, §5)*:
   `K_kill = min(12.0, T_A1 × σ_sel × 10,000 / 20)`, where `σ_sel` is the selector's
   realised annualised sd on the test sessions (Inputs), and
   `ΔT = annual_turnover(selector held weights on the test sessions) −
   annual_turnover(control held weights on the test sessions)`.
   - **The kill fires if `ΔT > K_kill`.**""",
r"""- `p_boot = 2(1 − Φ(|Δ_A1| / SE*))`.
- `p_HAC = 2(1 − Φ(|t|))`, with `t = protocol.paired_hac_t(selector, control, lags=6)`
  on the daily net differences (Inputs). *(lock)* If the sign of *t* differs from the
  sign of `Δ_A1`, `p_HAC := 1`: the HAC test is on the mean difference of two arms that
  run at different realised sd, and a *t* of the wrong sign is no evidence for Δ.
- **`p_A1 = max(p_boot, p_HAC)`.** The two coincide only at equal realised volatility,
  which the cap breaks, and requiring both removes the choice between them. If
  `Δ_A1 ≤ 0`, `p_A1 := 1` in Holm (§13.3).""",
r"""**Verdict — the first line that applies:**
1. **UNDECIDABLE**: any instrument or reading value is non-finite, including any one of
   the 1,000 placebo draws of Δ or β; a paired leg is missing on a test session; or the
   selector is at the fallback on more than half of the test sessions.
2. **FAIL (cost)**: the kill fired. The reading is still made and reported; it cannot
   pass.
3. **FAIL**: `Δ_A1 ≤ 0` at 5 bp.
4. **UNDECIDED**: `Δ_A1 > 0` at 5 bp and `≤ 0` at 10 bp.
5. **UNDERPOWERED**: `0 < Δ_A1 < T_A1`, or `Δ_A1 ≥ T_A1` with `p_A1` not rejected by
   Holm (§13.3). Never a PASS.
6. **DOWNGRADED (beta)**: the beta percentile is ≥ 0.95.
7. **DOWNGRADED (not conditional)**: the difference percentile among the placebo arms is
   < 0.95. The gain is then no larger than what a content-free partition with the same
   clock produces.
8. **DOMINATED (volatility)**: `Δ_W ≥ Δ_A1`. The one-line volatility partition does at
   least as well, so the gain cannot be credited to the context.
9. **DOWNGRADED (PIT)** *(lock, second validator)*: the same lock rebuilt on the
   18-feature partition without the NFCI (§8, control 5) does not meet this lock's
   criterion (lines 1-8 applied to the rebuild give anything but PASS).
10. **PASS.**""",
    )),
    Criterion('A', 'A-2', '§12.7', (
r"""- **Statistic:** `R = plain arithmetic mean of ρ_fk` over the qualifying cells.""",
r"""- **p-value:** `p_A2 = protocol.placebo_p_value(R, R^(·)) = (1 + #{j : R^(j) ≥ R}) /
  1,001`. Ties count against the real partition.
- **The lock holds** if `p_A2 ≤ 0.01` (at most 9 of 1,000 draws at or above R) **and**
  Holm rejects (§13.3). If A-2 ranks first in Holm, its bar is `p ≤ 0.00833`, at most 7
  draws.""",
r"""- **Verdict:**
  - **UNDECIDABLE**: `R` or any `R^(j)` is not finite, the placebo check is not exact, or
    fewer than 10 cells qualify (half of the 5K cells at a K sensitivity);
  - **PASS**: the lock holds and `R_W < R`, and the lock rebuilt on the 18-feature
    partition (§8, control 5) holds with its own witness;
  - **DOWNGRADED (PIT)**: the lock holds, `R_W < R`, and the lock rebuilt on the
    18-feature partition does not hold with its own witness;
  - **DOMINATED**: the lock holds and `R_W ≥ R`;
  - **NOT SHOWN**, otherwise. No power was measured for A-2 against any declared
    alternative, so under this lock A-2 **never reads FAIL**. A FAIL would need, before
    the reading, a power of at least 0.80 measured blind against an effect planted
    into demeaned legs. That needs a blindness ruling this lock does not give. Any such
    instrument would be an amendment in `docs/PROTOCOL_FREEZE.md`, committed before
    any reading.""",
    )),
    Criterion('B', 'B-1', '§12.8', (
r"""- **Statistic:** `G = mean over the 5,031 test sessions of [ℓ_t(Σ_f) − ℓ_t(Σ_f,k(t))]`,
  where *k(t)* is the lagged state along fold *f*'s path. G is positive when the
  state-conditional forecast is better.
- **Null and p-value:** G on the same 1,000 placebo draws, with every matrix
  re-estimated on the draw's training labels. `p_B1 = protocol.placebo_p_value(G,
  G^(·)) = (1 + #{G^(j) ≥ G}) / 1,001`. The lock holds if `p_B1 ≤ 0.01` and Holm
  rejects.""",
r"""  - **A B-1 that holds is reported DOMINATED unless `p_W1 ≤ 0.05` and `p_W2 ≤ 0.05`.**
    **HARDER.**""",
r"""- **Verdict:**
  - **UNDECIDABLE**: `G` or any `G^(j)` is not finite, the placebo check is not exact,
    or a pooled training matrix is not positive definite (§13.4);
  - **PASS**: the lock holds and neither witness dominates, and the lock rebuilt on the
    18-feature partition (§8, control 5) holds with its own witnesses;
  - **DOWNGRADED (PIT)**: the lock holds, neither witness dominates, and the lock rebuilt
    on the 18-feature partition does not hold with its own witnesses;
  - **DOMINATED**: the lock holds and a witness dominates;
  - **NOT SHOWN**, otherwise. As at A-2, no power was measured for B-1, so under this
    lock B-1 **never reads FAIL** (§12.7).""",
    )),
    Criterion('B', 'B-2', '§12.9', (
r"""- **Statistic:** `Δ_B2 = SR(state) − SR(pooled)` over the pooled test sessions.
- **Threshold:** `T_B2 = max(0.338, max_b MDE_B2(0.05/6, b))` on the pair's demeaned
  legs, measured by the instrument; SE\* at the block that gives the largest
  MDE_B2(0.05/6), as in §12.6.
- **p-value:** `p_B2 = max(p_boot, p_HAC)` as in §12.6, with `p_HAC := 1` when the sign
  of *t* differs from that of `Δ_B2`, and `p_B2 := 1` in Holm when `Δ_B2 ≤ 0`.
- **Kill rule** *(lock, from an audit: the comparator is B-2's own pair)*:
  - `ΔT = protocol.annual_turnover(state arm held weights restricted to the 5,031 test
    sessions) − protocol.annual_turnover(pooled arm held weights restricted to the same
    sessions)`. It is **not** measured against the equal-weight control: pooled risk
    parity overweights the slow legs and turns over far less than the control.
  - `K_kill = min(12.0, T_B2 × σ_state × 10,000 / 20)`, where σ_state is √252 × sd
    (ddof 1) of the state arm's 5 bp net daily excess returns on the test sessions.
    Anchored to the decision bar, as at A-1 (§12.6), and not to the pair's own
    uncorrected MDE. The state and pooled legs are nearly identical, so that MDE is
    small, and a kill scaled to it would fire almost automatically.
  - The kill fires if `ΔT > K_kill`.
- **Fallback at B-2:** a test session on which the state arm holds
  `risk_parity_mix(Σ_f)` because its lagged state is missing, its lagged state's
  training cell does not qualify, or its `Σ_fk` failed the positive-definite check.
  The >50% UNDECIDABLE line of §12.6 applies to that count.""",
r"""- **Verdict lines:** those of §12.6, in the same order, with `Δ_B2`, `T_B2`, `p_B2`,
  this kill and `Δ_W,B2` in place of A-1's.""",
    )),
    Criterion('C', 'C-1', '§12.10', (
r"""- **Statistic:** `S_C = annual_turnover(twin held weights on the test sessions) −
  annual_turnover(conditional held weights on the test sessions)`, each restricted to
  the test sessions **first**, in ×/yr.
- **Gate:** the conditional arm's mean `δ_t` over the test sessions must not exceed the
  twin's.
- **Null:** `S_C` on the 1,000 placebo draws. On each draw the whole rule (λ, the
  intervals, the training check) is re-run on the draw's training labels, and the arm
  is run on the draw's test labels. `p_C1 = protocol.placebo_p_value(S_C, S^(·))`. The
  re-optimisation on every draw absorbs the rule's in-sample optimisation bias.
- **C-1 holds** if `p_C1 ≤ 0.01`, Holm rejects, and the gate holds.
- **Resolution and power, before the reading** *(lock, from an audit)*. From the null,
  the instrument prints q50, q95, q99 and **`MDE_C = q99 − q20`**: the shift of the null
  that puts 80% of it above its 99th percentile, under a location shift. The saving
  worth 0.05 Sharpe at 5 bp is **`S* = 0.05 × σ_twin × 10,000 / 5`**, where σ_twin is
  the twin's realised annualised sd of its 5 bp net daily excess returns on the test
  sessions (about 7.9×/yr at the control's 7.88%). **If `MDE_C ≤ S*`**, the instrument
  has a power of at least 0.80 against a saving of S\*, and a C-1 that does not hold
  reads FAIL; otherwise it reads NOT SHOWN.""",
r"""- **Verdict:**
  - **UNDECIDABLE**: `S_C` or any `S^(j)` is not finite, the placebo check is not
    exact, or a leg is missing on a test session;
  - **FAIL**: the gate fails; or the lock does not hold and `MDE_C ≤ S*`;
  - **NOT SHOWN**: the lock does not hold and `MDE_C > S*`;
  - **DOMINATED**: the lock holds and `S_W ≥ S_C`;
  - **PASS**: the lock holds and `S_W < S_C`, and the lock rebuilt on the 18-feature
    partition (§8, control 5) holds with its own witness and gate;
  - **DOWNGRADED (PIT)**: the lock holds, `S_W < S_C`, and the lock rebuilt on the
    18-feature partition does not hold with its own witness and gate.""",
    )),
    Criterion('C', 'C-2', '§12.10', (
r"""- **C-2:** the bound `S_C × bps / 10,000 / σ_twin` at 5, 10 and 20 bp. It is reported
  beside the control's whole cost, never a PASS, and **`p_C2 := 1`** in Holm.
- **Level C's verdict is C-1's.**""",
    )),
    Criterion('*', 'level verdict', '§13.2', (
r"""**The cost column that decides is 5 bp.** 10 bp can only turn a positive reading into
UNDECIDED, and 20 bp enters only through the kill rule. A level's verdict is PASS only
if its locks pass: both at A and at B, and C-1 at C. Otherwise the level takes the
first status among its locks in this order: FAIL (FAIL (cost) included), then
UNDECIDABLE, then UNDECIDED, then UNDERPOWERED, then NOT SHOWN, then DOWNGRADED, then
DOMINATED. Following ARBITRAGE §5(b), no
PASS in one channel is traded for a FAIL in another: a turnover saving, a QLIKE gain
and a Sharpe difference are not commensurable.""",
    )),
    Criterion('*', 'Holm', '§13.3', (
r"""Order the six p-values `p_(1) ≤ … ≤ p_(6)`. Find the largest *j* such that
`p_(i) ≤ 0.05 / (6 − i + 1)` for every `i ≤ j`. The primaries ranked 1 to *j* are
rejected. Rejection is **necessary, never sufficient**: each lock must also meet its own
criterion in §12. The final Holm step runs after C is read. Each level's results file
states its provisional verdicts at the Bonferroni level, 0.05/6, which implies Holm
rejection.""",
    )),
    Criterion('*', 'UNDECIDABLE', '§13.4', (
r"""- **At every level:**
  - a non-finite instrument reading, or a non-finite real statistic;
  - *(lock, from an audit)* **any one** of the 1,000 draws of any placebo statistic a
    lock uses (`R`, `G`, `S_C`, `Δ^(j)`, `β^(j)`) is not finite. No draw is dropped,
    replaced or redrawn. `protocol.placebo_p_value` and
    `protocol.placebo_percentile` return NaN in that case (commit `3b64ab2`);
  - a paired leg missing on any test session;
  - a placebo check that is not exact.
- **At A and B, in addition:**
  - fewer than 10 qualifying cells (half of the 5K cells at a K sensitivity). At A the
    cells counted are the (fold, state) pairs that enter A-2 (§12.4); at B they are the
    qualifying training cells;
  - more than half of the test sessions at the fallback (§12.6 at A, §12.9 at B);
  - at B, a pooled training matrix that is not positive definite.""",
    )),
)


# ---------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------


#: Every committed file a reading executes or depends on. Amendment of 2026-09-23
#: (``docs/PROTOCOL_FREEZE.md``): two Claude sessions share this working tree, and the
#: bitwise check of §13.1 step 4 covers the instrument's outputs only, so an uncommitted
#: edit to reading-only code (verdict lines, Holm, rows) would otherwise pass unseen. A
#: reading refuses unless each of these is tracked and identical to HEAD.
CODE_FILES: tuple[str, ...] = (
    "pyproject.toml",
    "uv.lock",
    "regime_lab/config.py",
    "regime_lab/data/pit.py",
    "regime_lab/data/sources/kenfrench.py",
    "regime_lab/analysis/bootstrap.py",
    "regime_lab/analysis/placebo.py",
    "regime_lab/analysis/power.py",
    "regime_lab/analysis/trials.py",
    "regime_lab/extensions/trend.py",
    "regime_lab/extensions/vehicle.py",
    "regime_lab/selection/__init__.py",
    "regime_lab/selection/context.py",
    "regime_lab/selection/folds.py",
    "regime_lab/selection/library.py",
    "regime_lab/selection/protocol.py",
    "regime_lab/selection/tree.py",
    "regime_lab/selection/level_a.py",
    "regime_lab/selection/level_b.py",
    "regime_lab/selection/level_c.py",
    "scripts/run_twosigma_tree.py",
)


@dataclass(frozen=True)
class Setup:
    """Where the tree reads and writes, and the constants it runs at.

    The CLI always runs at the lock's constants (1,000 placebo draws, 2,000 bootstrap
    draws, §12.11, §12.12); other values exist for synthetic tests, and trial rows are
    written to the real register only at the lock's constants.
    """

    root: Path = ROOT
    inputs: tuple[str, ...] = INPUT_FILES
    draws: int = N_DRAWS
    boot_draws: int = BOOT_DRAWS
    workers: int | None = None
    trials_path: Path = field(default=trials.TRIALS)
    code_files: tuple[str, ...] = CODE_FILES

    @property
    def thresholds(self) -> Path:
        return self.root / THRESHOLDS_FILE

    def path(self, relative: str) -> Path:
        return self.root / relative

    @property
    def lock_constants(self) -> bool:
        return (self.draws, self.boot_draws) == (N_DRAWS, BOOT_DRAWS)

    def require_code(self) -> str:
        """Refuse unless every :data:`CODE_FILES` entry is tracked and identical to HEAD;
        return HEAD, which the reading artifact records (amendment of 2026-09-23)."""
        for relative in self.code_files:
            tree.require_committed(self.path(relative), self.root,
                                   what="the code a reading runs (amendment of 2026-09-23)")
        return tree.git_head(self.root)

    def refuse_logged(self, tests: Sequence[str], variant: str) -> None:
        """Refuse up front, before anything is computed, if the register already holds a
        ``twosigma`` row for one of ``tests`` at ``variant``: a second reading is a
        protocol breach (§13.5; amendment of 2026-09-23 — ``tree.log_trial`` checks the
        same thing, but only after the statistics are computed)."""
        frame = trials.read(self.trials_path)
        if frame.empty or "family" not in frame:
            return
        for raw in frame.loc[frame["family"] == tree.TRIAL_FAMILY, "config"]:
            row = json.loads(raw)
            if row.get("test") in tests and row.get("variant") == variant:
                raise ReadingRefused(f"({row.get('test')}, {variant}) is already in the "
                                     "register: a second reading is a protocol breach")

    def check_register(self) -> None:
        """Refuse to log to the real register at anything but the lock's constants."""
        if Path(self.trials_path).resolve() == trials.TRIALS.resolve() and not self.lock_constants:
            raise ValueError(f"trial rows go to {trials.TRIALS.name} only at the lock's "
                             f"constants ({N_DRAWS} draws, {BOOT_DRAWS} bootstrap draws)")


def levels_of(variant: Variant) -> tuple[str, ...]:
    """The levels a variant is evaluated at (§7): A only for the d rows (A-1)."""
    return tuple(level for level in LEVELS
                 if level in variant.levels or (level == "A" and "A-1" in variant.levels))


# ---------------------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------------------


def fmt(value: Any, digits: int = 4) -> str:
    """A value for print or markdown: ``non-finite`` strings and NaN are shown as such."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return tree.NON_FINITE
        return f"{value:.{digits}f}"
    if isinstance(value, list | tuple):
        return "[" + "; ".join(fmt(v, digits) for v in value) + "]"
    return str(value)


def na(value: Any, digits: int = 4) -> str:
    """A trial-row field: NaN is "n/a" there (no arm, no threshold, no p2), never a number."""
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return fmt(value, digits)


def pct(value: Any, digits: int = 2) -> str:
    """A share as a percentage."""
    if isinstance(value, float) and math.isfinite(value):
        return f"{100.0 * value:.{digits}f}%"
    return fmt(value)


def _leaves(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    if isinstance(value, Mapping):
        out = []
        for key, item in value.items():
            out.extend(_leaves(item, (*path, str(key))))
        return out
    return [(path, value)]


def format_printout(section: str, printout: Mapping[str, Any], seconds: float) -> str:
    """An instrument printout for the terminal: every leaf, grouped by its parent key."""
    lines = [f"=== section {section} ({seconds:.0f} s) ==="]
    groups: dict[tuple[str, ...], list[str]] = {}
    for path, value in _leaves(printout):
        text = repr(value) if isinstance(value, float) else fmt(value)
        groups.setdefault(path[:-1], []).append(f"{path[-1]}={text}")
    for parent, items in groups.items():
        lines.append(f"  {'.'.join(parent) or 'top'}: " + "  ".join(items))
    return "\n".join(lines)


def quote(text: str) -> str:
    """A markdown blockquote of the lock's text, line for line."""
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def criteria_markdown(level: str) -> str:
    """The criteria of a level's locks and of §13.2-§13.4, copied from the lock."""
    parts = ["### The criteria, copied from the lock",
             "",
             "Verbatim from `docs/PRESPEC_TWOSIGMA.md` (LOCKED, `30f7d69`). They were fixed "
             "before any statistic existed; this file only restates them beside the "
             "thresholds."]
    for criterion in CRITERIA:
        if criterion.level not in (level, "*"):
            continue
        parts += ["", f"**{criterion.lock}, {criterion.section}:**", ""]
        parts.append("\n>\n".join(quote(text) for text in criterion.text))
    return "\n".join(parts)


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _get(mapping: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(mapping, Mapping) or key not in mapping:
            return None
        mapping = mapping[key]
    return mapping


def _cost(mapping: Any, bps: float) -> Any:
    """A cost column however the level keyed it (``5``, ``5.0``)."""
    if not isinstance(mapping, Mapping):
        return None
    for key in (f"{bps:g}", str(float(bps)), bps, float(bps)):
        if key in mapping:
            return mapping[key]
    return None


def _reasons(printout: Mapping[str, Any]) -> list[str]:
    undecidable = printout.get("undecidable")
    if isinstance(undecidable, Mapping):
        return [f"{lock}: {reason}" for lock, reasons in undecidable.items()
                for reason in reasons]
    return [f"C-1: {reason}" for reason in (undecidable or [])]


# ---------------------------------------------------------------------------------------
# results files
# ---------------------------------------------------------------------------------------


def _block(name: str, body: str) -> str:
    return f"<!-- {name}:BEGIN -->\n{body.rstrip()}\n<!-- {name}:END -->"


def replace_block(path: Path, name: str, body: str) -> None:
    """Replace one marked block of a results file (the instrument writes the file)."""
    text = path.read_text()
    pattern = re.compile(rf"<!-- {name}:BEGIN -->.*?<!-- {name}:END -->", re.S)
    if not pattern.search(text):
        raise ValueError(f"{path.name} has no {name} block: run the instrument first")
    path.write_text(pattern.sub(lambda _: _block(name, body), text, count=1))


def results_file(level: str, instrument_body: str) -> str:
    head = (f"# Two Sigma {TITLES[level]}\n\n"
            "`docs/PRESPEC_TWOSIGMA.md`, LOCKED at `30f7d69`. Written by "
            "`scripts/run_twosigma_tree.py`; the numbers below come from its output and "
            "from `docs/artifacts/twosigma/`, never typed by hand.\n")
    reading = ("## Reading (§13.1 step 4)\n\nNot read. The reading of this level runs only "
               "after the instruments of A, B and C and their thresholds are committed, "
               "in the order A, B, C (§13.1).")
    sensitivities = ("## Sensitivities (§12.13)\n\nNot read. They are read after the six "
                     "primaries and the Holm step.")
    return "\n".join([head, _block("INSTRUMENT", instrument_body), "",
                      _block("READING", reading), "", _block("SENSITIVITIES", sensitivities),
                      ""])


def _header(payload: Mapping[str, Any], sections: Sequence[str]) -> str:
    inputs = table(("input file", "SHA-256"),
                   [(f"`{k}`", f"`{v}`") for k, v in payload["inputs"].items()])
    packages = ", ".join(f"{k} {v}" for k, v in payload["packages"].items())
    heads = sorted(set(payload["git_head"].values()))
    return "\n".join([
        "## Instrument (§13.1 steps 1-3)",
        "",
        NOTHING_READ,
        "",
        f"Computed on {dt.date.today().isoformat()} at code commit "
        f"`{', '.join(h[:7] for h in heads)}`. Every threshold is stored bitwise in "
        f"`{THRESHOLDS_FILE}`, sections {', '.join(f'`{s}`' for s in sections)}; the "
        "reading refuses to run unless that file is committed, unmodified, and equal "
        "bitwise to its own recomputation, on inputs with the same SHA-256 and packages "
        "at the versions `uv.lock` pins (§13.1 step 4). The tables round what the file "
        "stores.",
        "",
        f"Packages pinned by `uv.lock`: {packages}.",
        "",
        inputs,
    ])


def _undecidable_md(printouts: Mapping[str, Mapping[str, Any]]) -> str:
    lines = ["### UNDECIDABLE before the reading (§13.4)", ""]
    found = [(section, reason) for section, p in printouts.items() for reason in _reasons(p)]
    if not found:
        lines.append("None: no lock of this level is UNDECIDABLE before its reading, at the "
                     "primary or at any sensitivity.")
    else:
        lines.append("A lock listed here is not read and spends no trial row (§13.4).")
        lines.append("")
        lines += [f"- `{section}` — {reason}" for section, reason in found]
    return "\n".join(lines)


def _run_sizes(printouts: Mapping[str, Mapping[str, Any]]) -> tuple[str, str, str]:
    """Placebo draws, bootstrap draws and test sessions of the instruments, as printed."""
    first = next(iter(printouts.values()))
    sessions = (_get(first, "counts", "test_sessions") or _get(first, "test_sessions")
                or _get(first, "cells", "test_sessions"))
    return (fmt(_get(first, "placebo", "draws")), fmt(_get(first, "bootstrap", "draws")),
            fmt(sessions))


def instrument_markdown_a(printouts: Mapping[str, Mapping[str, Any]]) -> str:
    counts, mdes, arms, nulls = [], [], [], []
    for section, p in printouts.items():
        c, a1 = p["counts"], p["A-1"]
        counts.append((f"`{section}`", fmt(p["K"]), fmt(p["d"], 2),
                       f"{fmt(c['train_cells'])} of {fmt(c['cells_possible'])}",
                       f"{fmt(c['a2_cells'])} ({c['a2_cells_per_fold']})",
                       fmt(c["cell_floor"], 1), fmt(c["abstaining_rows"]),
                       f"{fmt(c['fallback_sessions'])} of {fmt(c['test_sessions'])} "
                       f"({pct(c['fallback_share'], 1)})",
                       fmt(c["control_folds"]), fmt(p["placebo"]["exact"])))
        mde = a1["mde_0.05/6"]
        mdes.append((f"`{section}`", *(fmt(a1["se"][b]) for b in ("21", "63", "126")),
                     fmt(_get(a1, "mde_0.05", "63")),
                     *(fmt(mde[b]) for b in ("21", "63", "126")),
                     f"**{fmt(a1['T_A1'])}**", fmt(a1["block_star"]),
                     fmt(a1["standalone_threshold"], 3)))
        turnover, kill = a1["turnover"], a1["kill"]
        arms.append((f"`{section}`", fmt(turnover["selector"], 2), fmt(turnover["control"], 2),
                     fmt(turnover["delta"], 2), fmt(kill["k_kill"], 2),
                     "**fires**" if kill["fires"] is True else fmt(kill["fires"]),
                     pct(a1["sigma"]["selector"]), pct(a1["sigma"]["control"]),
                     pct(a1["cap_share"]["selector"], 1), pct(a1["cap_share"]["control"], 1),
                     f"{fmt(a1['missing_sessions']['selector'])} / "
                     f"{fmt(a1['missing_sessions']['control'])}"))
        if "A-2" in p:
            n = p["A-2"]
            nulls.append((f"`{section}`", fmt(n["cells"]), fmt(n["q50"]), fmt(n["q95"]),
                          fmt(n["q99"]), fmt(n["q99-q50"])))
    draws, boot, sessions = _run_sizes(printouts)
    parts = [
        "### What was measured",
        "",
        "- The partition of each variant refitted walk-forward (§12.2), its cells (§12.4) "
        f"and its {draws} uniform placebo draws, seed 0, `method=\"uniform\"` (§12.11).",
        "- `m` fitted per fold on the training folds (§12.5), the selector and the "
        "equal-weight control on the traded path (§12.3), both targeted and netted of 5 bp; "
        f"the blinded bootstrap of their **demeaned** 5 bp legs on the {sessions} test "
        f"sessions ({boot} draws, seed 0, blocks 21/63/126) giving the MDEs, T_A1 and SE* "
        "(§12.6 steps 1-2); their held turnover and the kill line (step 3); realised sd "
        "and cap share (step 5).",
        "- A-2's null: `R^(j)` on each placebo draw, profiles recomputed with the draw's "
        "labels (§12.7). The real `R` was not computed.",
        "- The NFCI diagnostic on the 18-feature partition (§12.6 step 6), labels only.",
        "",
        "### Counts (§12.4, §12.6 step 5, §13.4)",
        "",
        table(("section", "K", "d", "training cells", "A-2 cells (per fold)", "floor",
               "abstaining (fold, state) rows", "fallback test sessions",
               "folds at the control", "placebo exact"), counts),
        "",
        "### A-1: resolution and threshold (§12.6 steps 1-2, §12.12)",
        "",
        "Blinded bootstrap SE of the paired Sharpe difference on demeaned legs, its MDE at "
        "α 0.05 (block 63, for reference) and at α 0.05/6 per block; "
        "`T_A1 = max(0.338, max_b MDE(0.05/6, b))`; Lo's standalone threshold at 0.05/6 is "
        "printed beside it and decides nothing (§12.12).",
        "",
        table(("section", "SE 21", "SE 63", "SE 126", "MDE(0.05) 63", "MDE(0.05/6) 21",
               "MDE(0.05/6) 63", "MDE(0.05/6) 126", "T_A1", "block*", "Lo standalone"),
              mdes),
        "",
        "### A-1: turnover and the kill line (§12.6 step 3), sd and cap share (step 5)",
        "",
        "Held turnover ×/yr on the test sessions, restricted first; "
        "`K_kill = min(12, T_A1 × σ_sel × 10,000 / 20)`; the kill fires if ΔT > K_kill.",
        "",
        table(("section", "selector ×/yr", "control ×/yr", "ΔT", "K_kill", "kill",
               "σ selector", "σ control", "cap share selector", "cap share control",
               "missing test sessions"), arms),
        "",
        "### A-2: the null (§12.7)",
        "",
        f"Percentiles of `R^(·)` over the {draws} draws. `q99 − q50` is the resolution "
        "quoted in a NOT SHOWN (§13.6).",
        "",
        table(("section", "cells", "q50", "q95", "q99", "q99 − q50"), nulls),
    ]
    nfci = _get(printouts.get("A", {}), "nfci")
    if isinstance(nfci, Mapping):
        per_fold = ", ".join(f"fold {k} {pct(v, 1)}"
                             for k, v in nfci["agreement_per_fold"].items())
        parts += [
            "",
            "### NFCI diagnostic (§12.6 step 6, §3)",
            "",
            f"The partition refitted on the {nfci['features']} features without `fin_nfci` "
            "and `fin_nfci_chg13w` (same K, `n_init`, seed, folds, training start), against "
            "the declared one, labels only:",
            "",
            f"- out-of-sample label agreement after `context.align_labels`: "
            f"{fmt(nfci['agree_sessions'])} of {fmt(nfci['test_sessions'])} test sessions "
            f"({pct(nfci['agreement'], 1)}); {per_fold};",
            f"- out-of-sample transitions per year: {fmt(nfci['transitions_per_year'], 2)} "
            "(declared partition: 12.55);",
            f"- pooled η² against log rv: {fmt(nfci['eta2_log_rv'], 3)} (declared: 0.142).",
        ]
    return "\n".join(parts)


def instrument_markdown_b(printouts: Mapping[str, Mapping[str, Any]]) -> str:
    counts, mdes, arms, nulls = [], [], [], []
    for section, p in printouts.items():
        c, b2 = p["cells"], p["B-2"]
        counts.append((f"`{section}`", fmt(p["K"]),
                       f"{fmt(c['qualifying_training'])} of {fmt(c['training'])} "
                       f"({c['per_fold']})", fmt(c["floor"], 1),
                       fmt(p["matrices"]["pooled_positive_definite"]),
                       fmt(p["matrices"]["cells_not_positive_definite"]),
                       f"{fmt(p['fallback']['test_sessions'])} "
                       f"({pct(p['fallback']['share'], 1)})",
                       fmt(c["abstaining_test_sessions"]), fmt(p["placebo"]["exact"])))
        mde = _get(b2, "mde") or {}
        mdes.append((f"`{section}`", *(fmt(_get(mde, "se", b)) for b in ("21", "63", "126")),
                     *(fmt(_get(mde, "mde_0.05/6", b)) for b in ("21", "63", "126")),
                     f"**{fmt(b2['T_B2'])}**", fmt(_get(mde, "block_star"))))
        turnover, kill = b2["turnover"], b2["kill"]
        arms.append((f"`{section}`", fmt(turnover["state"], 2), fmt(turnover["pooled"], 2),
                     fmt(turnover["delta"], 2), fmt(kill["k_kill"], 2),
                     "**fires**" if kill["fires"] is True else fmt(kill["fires"]),
                     pct(b2["sigma"]["state"]), pct(b2["sigma"]["pooled"]),
                     pct(b2["cap_share"]["state"], 1), pct(b2["cap_share"]["pooled"], 1),
                     f"{fmt(b2['missing_test_sessions']['state'])} / "
                     f"{fmt(b2['missing_test_sessions']['pooled'])}"))
        b1 = p["B-1"]
        nulls.append((f"`{section}`", fmt(b1["null"]["q50"]), fmt(b1["null"]["q95"]),
                      fmt(b1["null"]["q99"]), fmt(b1["null"]["q99-q50"]),
                      fmt(b1["non_finite_draws"]), pct(b1["correlation_share_fold5"], 1)))
    draws, boot, sessions = _run_sizes(printouts)
    return "\n".join([
        "### What was measured",
        "",
        f"- The partition of each variant (§12.2), its cells (§12.4) and its {draws} uniform "
        "placebo draws, seed 0 (§12.11).",
        "- The second-moment matrices about zero per fold and qualifying cell, on the "
        "training folds (§12.8), with the positive-definite check; the pooled and "
        "state-conditional risk-parity arms on the traded path, gross-matched, targeted and "
        "netted of 5 bp (§12.9); the blinded bootstrap of their **demeaned** 5 bp legs on "
        f"the {sessions} test sessions ({boot} draws, seed 0, blocks 21/63/126) giving the "
        "MDEs and T_B2; their turnover, the kill line, realised sd and cap share.",
        "- B-1's null: `G^(j)` on each placebo draw, every matrix re-estimated on the draw's "
        "training labels (§12.8). The real `G`, the witnesses and the placebo arms were not "
        "computed.",
        "- The return-space share of blend variance through the correlations, from fold 5's "
        "pooled training matrix (§6 Level B's formula, §12.8 \"Instrument\").",
        "",
        "### Counts (§12.4, §12.8, §12.9, §13.4)",
        "",
        table(("section", "K", "qualifying training cells (per fold)", "floor",
               "pooled matrices PD (per fold)", "cells not PD", "fallback test sessions",
               "abstaining test sessions", "placebo exact"), counts),
        "",
        "### B-2: resolution and threshold (§12.9)",
        "",
        "`T_B2 = max(0.338, max_b MDE_B2(0.05/6, b))` on the pair's demeaned 5 bp legs.",
        "",
        table(("section", "SE 21", "SE 63", "SE 126", "MDE(0.05/6) 21", "MDE(0.05/6) 63",
               "MDE(0.05/6) 126", "T_B2", "block*"), mdes),
        "",
        "### B-2: turnover and the kill line, sd and cap share (§12.9)",
        "",
        "Held turnover ×/yr on the test sessions, restricted first; "
        "`K_kill = min(12, T_B2 × σ_state × 10,000 / 20)`; the kill fires if ΔT > K_kill.",
        "",
        table(("section", "state arm ×/yr", "pooled arm ×/yr", "ΔT", "K_kill", "kill",
               "σ state", "σ pooled", "cap share state", "cap share pooled",
               "missing test sessions"), arms),
        "",
        "### B-1: the null and the correlation share (§12.8)",
        "",
        f"Percentiles of `G^(·)` over the {draws} draws; `q99 − q50` is the resolution "
        "quoted in a NOT SHOWN (§13.6). The correlation share is the return-space "
        "counterpart of §6's 23.9% in position space.",
        "",
        table(("section", "q50", "q95", "q99", "q99 − q50", "non-finite draws",
               "correlation share (fold 5)"), nulls),
    ])


def instrument_markdown_c(printouts: Mapping[str, Mapping[str, Any]]) -> str:
    rows, menu, nulls = [], [], []
    for section, p in printouts.items():
        c, twin, null = p["cells"], p["twin"], p["null"]
        rows.append((f"`{section}`", fmt(p["K"]), f"{fmt(c['train'])} ({c['train_per_fold']})",
                     fmt(c["abstaining"]), fmt(p["placebo"]["exact"]),
                     fmt(twin["turnover"], 2), fmt(twin["mean_delta"]), pct(twin["sigma"]),
                     fmt(twin["missing"]), pct(p["control_cap_share"], 1)))
        menu.append((f"`{section}`", *(fmt(v, 2) for v in p["menu_turnover"].values())))
        nulls.append((f"`{section}`", fmt(null["q20"], 3), fmt(null["q50"], 3),
                      fmt(null["q95"], 3), fmt(null["q99"], 3), f"**{fmt(p['MDE_C'], 3)}**",
                      f"**{fmt(p['S_star'], 3)}**", fmt(p["powered"]),
                      " / ".join(fmt(null[k]) for k in ("below_zero", "at_zero",
                                                        "above_zero")),
                      fmt(null["atom_share"], 3), fmt(null["non_finite"])))
    first = next(iter(printouts.values()))
    draws, _, _ = _run_sizes(printouts)
    return "\n".join([
        "### What was measured",
        "",
        "- The target `w*`, the control's held weights after the multiplier, and the "
        "state-blind cadence books of the menu H = {1, 2, 3, 5, 8, 13, 21}, among them the "
        "weekly twin (h = 5) (§12.10). They read no label.",
        f"- C-1's null: on each of the {draws} uniform placebo draws (seed 0), the whole rule "
        "(λ, the intervals, the training check) re-run on the draw's training labels and "
        "the arm run on its test labels, giving `S^(j)` (§12.10). **The conditional arm was "
        "not built on the real partition**: no `h_fk`, no `S_C`, no gate, no turnover or δ "
        "of it (§12.10 \"Instrument\").",
        "- The cells of each partition, from its labels only (§12.4).",
        "",
        "### Counts, the twin, sd and cap share (§12.10)",
        "",
        "Held turnover ×/yr and mean δ on the test sessions; σ_twin is the realised sd of "
        "the twin's 5 bp net daily excess returns there.",
        "",
        table(("section", "K", "training cells (per fold)", "abstaining test sessions",
               "placebo exact", "twin ×/yr", "twin mean δ", "σ_twin", "twin missing",
               "control cap share"), rows),
        "",
        "### The state-blind menu (§12.10)",
        "",
        "Held turnover ×/yr on the test sessions of each state-blind book.",
        "",
        table(("section", *(f"h = {h}" for h in first["menu_turnover"])), menu),
        "",
        "### C-1: the null, MDE_C and S* (§12.10 \"Resolution and power\")",
        "",
        "`MDE_C = q99 − q20`; `S* = 0.05 × σ_twin × 10,000 / 5`. **Amendment of "
        "2026-09-23** (`docs/PROTOCOL_FREEZE.md`, before any reading): the power claim "
        "holds (\"powered\") only if `0 < MDE_C ≤ S*` **and** no single value takes 20% "
        "or more of the draws (\"atom share\"); a C-1 that does not hold then reads "
        "FAIL, otherwise NOT SHOWN. An atom makes `q99 − q20` measure no shift. C-1 holds "
        "only if `p ≤ 0.01`, Holm rejects, the gate holds **and** `S_C ≥ S*`: with an atom "
        "null the placebo bar alone would let any positive saving through. The draws "
        "column counts the finite `S^(j)` below, at and above zero.",
        "",
        table(("section", "q20", "q50", "q95", "q99", "MDE_C", "S*", "powered",
               "draws < 0 / = 0 / > 0", "atom share", "non-finite draws"), nulls),
    ])


INSTRUMENT_MARKDOWN = {"A": instrument_markdown_a, "B": instrument_markdown_b,
                       "C": instrument_markdown_c}


# ---------------------------------------------------------------------------------------
# instrument — §13.1 steps 1-3
# ---------------------------------------------------------------------------------------


def instrument_one(
    level: str, data: TreeData, variant: Variant, partition: tree.Partition,
    draws: tree.PlaceboDraws, setup: Setup,
) -> dict[str, Any]:
    """One level's instrument at one variant, on a shared partition and placebo."""
    if level == "A":
        return level_a.instrument(data, variant, draws, boot_draws=setup.boot_draws,
                                  partition=partition, workers=setup.workers)
    if level == "B":
        return level_b.instrument(data, variant, draws, partition=partition,
                                  boot_draws=setup.boot_draws, workers=setup.workers)
    return level_c.instrument(data, variant, draws, partition=partition, workers=setup.workers)


def run_instruments(
    data: TreeData,
    setup: Setup,
    *,
    variants: Sequence[Variant] = INSTRUMENT_VARIANTS,
    emit: Emit = print,
) -> dict[str, dict[str, Any]]:
    """Every level's instrument at every variant (§13.1 step 1), in :data:`LEVELS` order
    within each variant; returns ``{section: printout}``. One partition and one placebo
    (§12.11: 1,000 draws, seed 0, uniform) per partition key, shared by the levels and by
    the d rows, which use the primary's partition (§12.13). Prints each printout."""
    sections: dict[str, dict[str, Any]] = {}
    shared: dict[Any, tuple[tree.Partition, tree.PlaceboDraws]] = {}
    for variant in variants:
        if variant.partition_key not in shared:
            start = time.perf_counter()
            partition = tree.partition_paths(data, variant)
            draws = tree.placebo_draws(partition.paths, setup.draws, PLACEBO_SEED)
            shared[variant.partition_key] = (partition, draws)
            emit(f"--- partition K={variant.K} smoothing={variant.smoothing or 0} "
                 f"features={len(variant.features)}: {draws!r} "
                 f"({time.perf_counter() - start:.0f} s)")
        partition, draws = shared[variant.partition_key]
        for level in levels_of(variant):
            start = time.perf_counter()
            section = tree.section_name(level, variant)
            sections[section] = instrument_one(level, data, variant, partition, draws, setup)
            emit(format_printout(section, sections[section], time.perf_counter() - start))
    return sections


def write_instruments(sections: Mapping[str, Mapping[str, Any]], setup: Setup) -> dict[str, Any]:
    """The threshold file (§13.1 step 3, through ``tree.write_thresholds``) and the
    INSTRUMENT sections of the three results files. Refuses to rewrite a threshold file
    that git tracks: an instrument changed after its commit is an amendment (§13.1)."""
    path = setup.thresholds
    if path.exists() and tree.is_tracked(path, setup.root):
        raise RuntimeError(f"{THRESHOLDS_FILE} is committed: the instruments are fixed, and "
                           "changing one is an amendment in docs/PROTOCOL_FREEZE.md (§13.1)")
    payload = tree.write_thresholds(dict(sections), path=path, root=setup.root,
                                    inputs=setup.inputs)
    for level in LEVELS:
        mine = {s: p for s, p in sections.items() if s.split(":")[0] == level}
        if not mine:
            continue
        body = "\n\n".join([
            _header(payload, list(mine)),
            INSTRUMENT_MARKDOWN[level](mine),
            _undecidable_md(mine),
            criteria_markdown(level),
        ])
        target = setup.path(RESULTS_FILE.format(level=level))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(results_file(level, body))
    return payload


# ---------------------------------------------------------------------------------------
# readings — §13.1 step 4
# ---------------------------------------------------------------------------------------


def read_level(level: str, data: TreeData, variant: Variant, setup: Setup) -> dict[str, Any]:
    """One level's reading at one variant through its module, which recomputes the
    instrument and calls ``tree.verify_for_reading`` before any statistic (§13.1 step 4)."""
    where = dict(root=setup.root, path=setup.thresholds, inputs=setup.inputs)
    if level == "A":
        return level_a.read(data, variant, setup.draws, boot_draws=setup.boot_draws,
                            workers=setup.workers, **where)
    if level == "B":
        return level_b.read(data, variant, setup.draws, boot_draws=setup.boot_draws,
                            workers=setup.workers, pit_draws=setup.draws, **where)
    return level_c.read(data, variant, setup.draws, workers=setup.workers, **where)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


def _load_committed(path: Path, root: Path, what: str) -> dict[str, Any]:
    tree.require_committed(path, root, what=what)
    return json.loads(path.read_text())


def _log_rows(rows: Mapping[str, Mapping[str, Any]], variant: Variant, setup: Setup) -> None:
    for test, row in rows.items():
        tree.log_trial(test, variant, **row, path=setup.trials_path)


def run_read(level: str, data: TreeData, setup: Setup, *, emit: Emit = print) -> dict[str, Any]:
    """§13.1 step 4 for one level's primary: the order A, B, C, one reading per level.

    Refuses before anything is computed if the level was already read (its artifact
    exists), its results file with the instrument section is not committed (§13.1 step
    3), or the previous level's reading artifact is not committed; the level module then
    refuses unless its thresholds verify. Writes the artifact, logs the §13.5 rows
    (provisional Bonferroni verdicts), marks the artifact logged, and writes the READING
    section of the level's results file.
    """
    if level not in LEVELS:
        raise ValueError(f"not a level: {level!r}")
    artifact = setup.path(READING_FILE.format(level=level))
    if artifact.exists():
        raise ReadingRefused(f"level {level} has been read already ({artifact.name}): a "
                             "second reading is a protocol breach (§13.5)")
    tree.require_committed(setup.path(RESULTS_FILE.format(level=level)), setup.root,
                           what="the instrument section of the results file (§13.1 step 3)")
    previous = {"B": "A", "C": "B"}.get(level)
    if previous is not None:
        tree.require_committed(setup.path(READING_FILE.format(level=previous)), setup.root,
                               what=f"the reading of level {previous}")
    head = setup.require_code()
    setup.refuse_logged(LEVEL_TESTS[level], PRIMARY.name)
    setup.check_register()
    emit(f"code at HEAD {head}; packages pinned by uv.lock: {tree.locked_versions(setup.root)}")
    reading = read_level(level, data, PRIMARY, setup)
    stored = {"reading": tree.plain(reading), "rows_logged": False, "git_head": head}
    _write_json(artifact, stored)
    _finish_reading(level, stored, setup, emit=emit)
    return reading


def _finish_reading(
    level: str, stored: dict[str, Any], setup: Setup, *, emit: Emit = print
) -> None:
    """The end of a reading, shared by :func:`run_read` and :func:`resume_logging`: log the
    artifact's §13.5 rows, mark them logged, write the READING section."""
    artifact = setup.path(READING_FILE.format(level=level))
    _log_rows(stored["reading"]["trial_rows"], PRIMARY, setup)
    stored["rows_logged"] = True
    _write_json(artifact, stored)
    body = READING_MARKDOWN[level](stored["reading"])
    replace_block(setup.path(RESULTS_FILE.format(level=level)), "READING", body)
    emit(body)


def resume_logging(level: str, setup: Setup, *, emit: Emit = print) -> dict[str, Any]:
    """Finish a reading whose statistics were computed and stored but whose trial rows
    were not logged (the artifact reads ``rows_logged: false``), without recomputing
    anything: the rows, the verdicts and the READING section come from the artifact.

    Refuses unless that artifact exists and is not marked logged, the results file is
    committed, the code a reading runs is committed and unmodified, and the register
    holds no row of this level. Records the HEAD at which the rows were logged beside the
    HEAD at which the reading was computed.
    """
    if level not in LEVELS:
        raise ValueError(f"not a level: {level!r}")
    artifact = setup.path(READING_FILE.format(level=level))
    if not artifact.exists():
        raise ReadingRefused(f"no reading artifact for level {level}: nothing to resume")
    stored = json.loads(artifact.read_text())
    if stored.get("rows_logged"):
        raise ReadingRefused(f"level {level}'s rows are already logged: nothing to resume")
    tree.require_committed(setup.path(RESULTS_FILE.format(level=level)), setup.root,
                           what="the instrument section of the results file (§13.1 step 3)")
    stored["logging_head"] = setup.require_code()
    setup.refuse_logged(LEVEL_TESTS[level], PRIMARY.name)
    setup.check_register()
    emit(f"resuming level {level}: computed at {stored.get('git_head')}, rows logged at "
         f"{stored['logging_head']}; nothing is recomputed")
    _finish_reading(level, stored, setup, emit=emit)
    return stored


def _lock_lines(reading: Mapping[str, Any], locks: Sequence[str]) -> str:
    verdicts = reading.get("verdicts", {})
    rows = [(lock, f"**{verdicts.get(lock, '—')}**", fmt(_get(reading, "holm_p", lock)))
            for lock in locks]
    return table(("lock", "provisional verdict (Bonferroni 0.05/6)", "p entering Holm"), rows)


def _rows_md(reading: Mapping[str, Any]) -> str:
    rows = reading.get("trial_rows") or {}
    if not rows:
        return "No trial row: every lock was UNDECIDABLE before its reading (§13.4)."
    out = [(test, na(r.get("sharpe")), na(r.get("delta")), na(r.get("threshold")),
            na(r.get("p")), na(r.get("placebo_pct")), str(r.get("verdict")),
            str(r.get("note") or "")) for test, r in rows.items()]
    return "\n\n".join([
        table(("test", "sharpe", "delta", "threshold", "p", "placebo_pct", "verdict", "note"),
              out),
        SHARPE_CAVEAT,
    ])


def _pit_md(pit: Any) -> str:
    if not pit:
        return ("§8 control 5 was not due: no lock would otherwise PASS, at the provisional "
                "bar or under a Holm rejection.")
    return ("§8 control 5, the 18-feature rebuild (downgrade only, no trial row): "
            f"`{json.dumps(pit.get('verdicts', pit.get('verdict')), ensure_ascii=False)}`.")


def reading_markdown_a(reading: Mapping[str, Any]) -> str:
    a1, a2 = reading.get("A-1", {}), reading.get("A-2", {})
    parts = ["## Reading (§13.1 step 4)", "", _lock_lines(reading, ("A-1", "A-2")), "",
             f"Level A, provisional: **{_get(reading, 'verdicts', 'A')}**. The final verdict "
             "follows the Holm step, after C (`docs/RESULTS_TWOSIGMA.md`).", ""]
    if a1.get("read"):
        parts += [
            "### A-1 (§12.6)",
            "",
            table(("", "5 bp", "10 bp", "20 bp"), [
                ("SR selector", *(fmt(_cost(_get(a1, "sharpe", "selector"), b)) for b in
                                  (5, 10, 20))),
                ("SR control", *(fmt(_cost(_get(a1, "sharpe", "control"), b)) for b in
                                 (5, 10, 20))),
                ("Δ_A1", *(fmt(_cost(a1.get("delta"), b)) for b in (5, 10, 20))),
            ]),
            "",
            f"- T_A1 {fmt(a1.get('T_A1'))}, SE* {fmt(a1.get('se_star'))} (block "
            f"{fmt(a1.get('block_star'))}); HAC-6 t {fmt(a1.get('t_hac'))}; p_boot "
            f"{fmt(a1.get('p_boot'))}, p_HAC {fmt(a1.get('p_hac'))}, **p_A1 "
            f"{fmt(a1.get('p'))}**.",
            f"- β selector {fmt(_get(a1, 'beta', 'selector'))} (control "
            f"{fmt(_get(a1, 'beta', 'control'))}); beta percentile "
            f"{fmt(a1.get('beta_pct'))}, difference percentile {fmt(a1.get('diff_pct'))} "
            f"among {fmt(a1.get('placebo_arms'))} placebo arms.",
            f"- Volatility witness W1: Δ_W {fmt(a1.get('delta_w'))}.",
            f"- ΔT {fmt(a1.get('turnover_delta'), 2)} ×/yr against K_kill "
            f"{fmt(a1.get('k_kill'), 2)}: kill {fmt(a1.get('kill_fires'))}; fallback "
            f"{pct(a1.get('fallback_share'), 1)} of the test sessions.",
            "- Per-fold components of Δ_A1 (diagnostic, §12.14): "
            + ", ".join(f"fold {k} {fmt(v)}" for k, v in (a1.get("per_fold") or {}).items())
            + ".",
            "",
        ]
    else:
        parts += ["### A-1 (§12.6)", "", "UNDECIDABLE before its reading, not read: "
                  + "; ".join(a1.get("undecidable_reasons") or []), ""]
    if a2.get("read"):
        parts += [
            "### A-2 (§12.7)",
            "",
            f"- R {fmt(a2.get('R'))} over {fmt(a2.get('cells'))} cells; **p_A2 "
            f"{fmt(a2.get('p'))}**, percentile {fmt(a2.get('placebo_pct'))}; null q50 "
            f"{fmt(a2.get('q50'))}, q95 {fmt(a2.get('q95'))}, q99 {fmt(a2.get('q99'))}.",
            f"- Volatility witness W1: R_W {fmt(a2.get('R_W'))}.",
            "- Per-fold means (diagnostic, §12.14): "
            + ", ".join(f"fold {k} {fmt(v)}" for k, v in (a2.get("per_fold") or {}).items())
            + ".",
            "",
        ]
    elif a2:
        parts += ["### A-2 (§12.7)", "", "UNDECIDABLE before its reading, not read: "
                  + "; ".join(a2.get("undecidable_reasons") or []), ""]
    pit = {k: _get(reading, k, "pit", "verdict_lines") for k in ("A-1", "A-2")
           if _get(reading, k, "pit")}
    parts += [_pit_md({"verdicts": pit} if pit else None), "", "### Trial rows (§13.5)", "",
              _rows_md(reading)]
    return "\n".join(parts)


def reading_markdown_b(reading: Mapping[str, Any]) -> str:
    b1, b2 = reading.get("B-1", {}), reading.get("B-2", {})
    w1, w2 = _get(b1, "witnesses", "W1") or {}, _get(b1, "witnesses", "W2") or {}
    pair = b2.get("reading") or {}
    parts = ["## Reading (§13.1 step 4)", "", _lock_lines(reading, ("B-1", "B-2")), "",
             f"Level B, provisional: **{_get(reading, 'verdicts', 'B')}**. The final verdict "
             "follows the Holm step, after C (`docs/RESULTS_TWOSIGMA.md`).", "",
             "### B-1 (§12.8)", ""]
    if b1.get("read"):
        parts += [
            f"- G {fmt(b1.get('G'))} (diagonal diagnostic G_diag {fmt(b1.get('G_diag'))}); "
            f"**p_B1 {fmt(b1.get('p'))}**, percentile {fmt(b1.get('placebo_pct'))}, null q99 "
            f"{fmt(b1.get('threshold'))}.",
            f"- Witness W1: mean d {fmt(w1.get('mean'))}, p_W1 {fmt(w1.get('p'))}; witness "
            f"W2: mean d {fmt(w2.get('mean'))}, p_W2 {fmt(w2.get('p'))}.",
            "",
        ]
    else:
        parts += ["UNDECIDABLE before its reading, not read: "
                  + "; ".join(_get(reading, "instrument", "undecidable", "B-1") or []), ""]
    parts += ["### B-2 (§12.9)", ""]
    if b2.get("read"):
        parts += [
            table(("", "5 bp", "10 bp", "20 bp"),
                  [("Δ_B2", *(fmt(_cost(b2.get("delta"), b)) for b in (5, 10, 20)))]),
            "",
            f"- SR state {fmt(b2.get('sharpe_state'))}, SR pooled "
            f"{fmt(b2.get('sharpe_pooled'))} (5 bp); T_B2 {fmt(b2.get('threshold'))}; HAC-6 t "
            f"{fmt(pair.get('t_hac'))}; p_boot {fmt(pair.get('p_boot'))}, p_HAC "
            f"{fmt(pair.get('p_hac'))}, **p_B2 {fmt(pair.get('p'))}**.",
            f"- β state {fmt(b2.get('beta_state'))}; beta percentile "
            f"{fmt(b2.get('beta_pct'))}, difference percentile {fmt(b2.get('diff_pct'))}; "
            f"witness W1: Δ_W,B2 {fmt(b2.get('delta_w'))}.",
            f"- Kill: ΔT {fmt(_get(b2, 'kill', 'delta_turnover'), 2)} against K_kill "
            f"{fmt(_get(b2, 'kill', 'k_kill'), 2)}, fires {fmt(_get(b2, 'kill', 'fires'))}.",
            "",
        ]
    else:
        parts += ["UNDECIDABLE before its reading, not read: "
                  + "; ".join(_get(reading, "instrument", "undecidable", "B-2") or []), ""]
    parts += [_pit_md(reading.get("pit")), "", "### Trial rows (§13.5)", "", _rows_md(reading)]
    return "\n".join(parts)


def reading_markdown_c(reading: Mapping[str, Any]) -> str:
    c1, c2 = reading.get("C-1", {}), reading.get("C-2", {})
    parts = ["## Reading (§13.1 step 4)", "", _lock_lines(reading, ("C-1", "C-2")), "",
             f"Level C, provisional: **{_get(reading, 'verdicts', 'C')}** (C-1's, §12.10). "
             "The final verdict follows the Holm step (`docs/RESULTS_TWOSIGMA.md`).", "",
             "### C-1 (§12.10)", ""]
    if c1.get("read"):
        rules = c1.get("rules") or {}
        parts += [
            f"- **S_C {fmt(c1.get('S_C'), 3)} ×/yr**: twin "
            f"{fmt(_get(c1, 'turnover', 'twin'), 2)}, conditional "
            f"{fmt(_get(c1, 'turnover', 'conditional'), 2)}; **p_C1 {fmt(c1.get('p'))}**, "
            f"percentile {fmt(c1.get('placebo_pct'))}, null q99 {fmt(c1.get('q99'), 3)}.",
            f"- Gate: mean δ conditional {fmt(_get(c1, 'mean_delta', 'conditional'))} against "
            f"twin {fmt(_get(c1, 'mean_delta', 'twin'))}: holds {fmt(c1.get('gate'))}.",
            f"- MDE_C {fmt(c1.get('MDE_C'), 3)} against S* {fmt(c1.get('S_star'), 3)}; "
            f"witness W1: S_W {fmt(c1.get('S_W'), 3)}.",
            f"- Conditional arm: σ {pct(_get(c1, 'sigma', 'conditional'))}, 5 bp net Sharpe "
            f"{fmt(c1.get('sharpe'))}; fallback test sessions "
            f"{fmt(c1.get('fallback_sessions'))}, test sessions of folds holding the twin "
            f"{fmt(c1.get('twin_sessions'))}.",
            "",
            table(("fold", "states", "h_fk", "λ", "holds the twin", "reason"),
                  [(k, fmt(v.get("states")), json.dumps(v.get("h")), fmt(v.get("lambda")),
                    fmt(v.get("holds_twin")), str(v.get("reason"))) for k, v in rules.items()]),
            "",
            "### C-2 (§12.10)",
            "",
            table(("", "5 bp", "10 bp", "20 bp"), [
                ("bound S_C × bps / 10,000 / σ_twin",
                 *(fmt(_cost(c2.get("bound"), b)) for b in (5, 10, 20))),
                ("the control's whole cost (Sharpe)",
                 *(fmt(_cost(c2.get("control_cost"), b)) for b in (5, 10, 20))),
            ]),
            "",
            "C-2 is a bound, never a PASS; `p_C2 := 1` in Holm.",
            "",
        ]
    else:
        parts += ["UNDECIDABLE before its reading, not read: "
                  + "; ".join(c1.get("undecidable") or []), ""]
    parts += [_pit_md(reading.get("pit")), "", "### Trial rows (§13.5)", "", _rows_md(reading)]
    return "\n".join(parts)


READING_MARKDOWN = {"A": reading_markdown_a, "B": reading_markdown_b, "C": reading_markdown_c}


# ---------------------------------------------------------------------------------------
# Holm — §13.3, §13.6
# ---------------------------------------------------------------------------------------

#: §13.6: how each status is worded; the closure statement is written only if no level
#: passes, with T_A1 at its applied value.
WORDING = {
    "FAIL": "does not transfer",
    "FAIL (cost)": "does not survive its own turnover",
    "UNDERPOWERED": "is not shown to transfer",
    "NOT SHOWN": "is not shown to transfer",
    UNDECIDABLE: "is not shown to transfer",
    "UNDECIDED": "is not shown to survive a conservative cost schedule",
    "DOMINATED": "passes its test, but a one-line volatility partition does at least as "
                 "well; it is not credited to the context",
    "DOWNGRADED": "passes its test, but the gain is market exposure (beta), or is no larger "
                  "than a content-free partition with the same clock produces (not "
                  "conditional), or does not hold without the revised NFCI (PIT); it is not "
                  "credited to the context",
    PASS: "PASS",
}


def wording(verdict: str) -> str:
    """§13.6's words for a level status."""
    if verdict in WORDING:
        return WORDING[verdict]
    return WORDING[tree.status_class(verdict)]


def run_holm(setup: Setup, *, emit: Emit = print) -> dict[str, Any]:
    """§13.3 after C: Holm over the six primaries from the committed reading artifacts
    (p := 1 where §13.3 sets it), then each level's final verdicts through its module's
    ``lock_verdicts`` with the rejected set (§13.2). Writes ``holm.json`` and
    ``docs/RESULTS_TWOSIGMA.md``; logs nothing (§13.5)."""
    setup.require_code()
    readings = {}
    for level in LEVELS:
        stored = _load_committed(setup.path(READING_FILE.format(level=level)), setup.root,
                                 f"the reading of level {level}")
        if not stored.get("rows_logged"):
            raise ReadingRefused(f"level {level}'s trial rows were not all logged")
        readings[level] = stored["reading"]
    p: dict[str, float] = {}
    provisional: dict[str, str] = {}
    for reading in readings.values():
        p.update({k: v for k, v in reading["holm_p"].items() if k in PRIMARIES})
        provisional.update({k: v for k, v in reading["verdicts"].items() if k in PRIMARIES})
    result = tree.holm(p, verdicts=provisional)
    final = {level: MODULES[level].lock_verdicts(readings[level], result.rejected)
             for level in LEVELS}
    payload = {
        "p_entered": result.p,
        "table": [{"rank": int(rank), **{k: tree.plain(v) for k, v in row.items()}}
                  for rank, row in result.table.iterrows()],
        "rejected": sorted(result.rejected),
        "provisional": provisional,
        "final": final,
        "levels": {level: final[level][level] for level in LEVELS},
    }
    _write_json(setup.path(HOLM_FILE), payload)
    body = holm_markdown(payload, readings)
    summary = setup.path(SUMMARY_FILE)
    summary.write_text("\n".join([
        "# Two Sigma tree — the Holm step and the level verdicts",
        "",
        "`docs/PRESPEC_TWOSIGMA.md`, LOCKED at `30f7d69`. Written by "
        "`scripts/run_twosigma_tree.py holm` from the committed reading artifacts in "
        "`docs/artifacts/twosigma/`.",
        "",
        body,
        "",
        _block("SENSITIVITIES", "## Sensitivities (§12.13)\n\nNot read yet."),
        "",
    ]))
    emit(body)
    return payload


def holm_markdown(payload: Mapping[str, Any], readings: Mapping[str, Any]) -> str:
    rows = [(str(r["rank"]), r["test"], fmt(r["p"]), fmt(r["bar"], 5),
             "**rejected**" if r["rejected"] else "not rejected") for r in payload["table"]]
    locks = [(test, payload["provisional"].get(test, "—"),
              f"**{payload['final'][test[0]][test]}**") for test in PRIMARIES]
    levels = [(level, f"**{verdict}**", wording(verdict))
              for level, verdict in payload["levels"].items()]
    thresholds = {
        "A": _get(readings["A"], "A-1", "T_A1"),
        "B": _get(readings["B"], "B-2", "threshold"),
        "C": _get(readings["C"], "C-1", "MDE_C"),
    }
    return "\n".join([
        "## Holm–Bonferroni over the six primaries (§13.3)",
        "",
        table(("rank", "primary", "p entering Holm", "bar 0.05/(6 − i + 1)", "Holm"), rows),
        "",
        "A lock UNDECIDABLE, an A-1 or B-2 with Δ ≤ 0, a missing p and C-2 enter with "
        "p := 1. Rejection is necessary, never sufficient (§13.3).",
        "",
        "## Lock verdicts",
        "",
        table(("lock", "provisional (0.05/6)", "final, after Holm"), locks),
        "",
        "## Level verdicts (§13.2) and their wording (§13.6)",
        "",
        table(("level", "verdict", "§13.6 wording"), levels),
        "",
        f"Resolutions for the wording of §13.6: T_A1 {fmt(thresholds['A'])}, T_B2 "
        f"{fmt(thresholds['B'])}, MDE_C {fmt(thresholds['C'], 3)}; the nulls' q99 − q50 are "
        "in each level's results file.",
        "",
        criteria_markdown("*"),
    ])


# ---------------------------------------------------------------------------------------
# sensitivities — §12.13
# ---------------------------------------------------------------------------------------


def run_sensitivities(
    data: TreeData,
    setup: Setup,
    *,
    variants: Sequence[Variant] = SENSITIVITIES,
    emit: Emit = print,
) -> dict[str, Any]:
    """§12.13 after the committed Holm step: each sensitivity's full evaluation at each of
    its levels (its own partition, placebo and committed thresholds), its §13.5 row with
    the PASS (not robust) note against the final primary level verdict, and the robust
    labels. Resumable: sections already in ``sensitivities.json`` are not read again."""
    holm = _load_committed(setup.path(HOLM_FILE), setup.root, "the Holm step")
    primary_levels: dict[str, str] = holm["levels"]
    head = setup.require_code()
    setup.check_register()
    artifact = setup.path(SENSITIVITY_FILE)
    done: dict[str, Any] = json.loads(artifact.read_text()) if artifact.exists() else {}
    for variant in variants:
        for level in levels_of(variant):
            section = tree.section_name(level, variant)
            if done.get(section, {}).get("rows_logged"):
                continue
            if section in done:
                raise ReadingRefused(f"{section} was read but its rows were not all logged: "
                                     "resolve by hand before going on")
            setup.refuse_logged(("A-1",) if variant.levels == ("A-1",) else (level,),
                                variant.name)
            reading = read_level(level, data, variant, setup)
            rows = MODULES[level].trial_rows(reading, variant,
                                             primary_level=primary_levels[level])
            done[section] = {"variant": variant.name, "level": level,
                             "reading": tree.plain(reading), "rows": tree.plain(rows),
                             "rows_logged": False, "git_head": head}
            _write_json(artifact, done)
            _log_rows(rows, variant, setup)
            done[section]["rows_logged"] = True
            _write_json(artifact, done)
            emit(f"{section}: {json.dumps(tree.plain(rows), ensure_ascii=False)}")
    labels = robust_labels(done, primary_levels)
    for level in LEVELS:
        replace_block(setup.path(RESULTS_FILE.format(level=level)), "SENSITIVITIES",
                      sensitivities_markdown(level, done, labels))
    replace_block(setup.path(SUMMARY_FILE), "SENSITIVITIES",
                  sensitivities_summary(done, labels, primary_levels))
    return {"sections": done, "labels": labels}


def robust_labels(done: Mapping[str, Any], primary_levels: Mapping[str, str]) -> dict[str, str]:
    """§12.13: a level PASS reads PASS (not robust) when any K or smoothing row at that
    level has ``delta ≤ 0``; the d rows do not enter."""
    labels = {}
    for level in LEVELS:
        deltas = [row.get("delta") for entry in done.values() if entry["level"] == level
                  and entry["variant"] not in (D025.name, D100.name)
                  for row in entry["rows"].values()]
        finite = [float(d) for d in deltas if isinstance(d, int | float)]
        labels[level] = tree.robust_label(primary_levels[level], finite)
    return labels


def sensitivities_markdown(level: str, done: Mapping[str, Any], labels: Mapping[str, str]) -> str:
    rows = []
    for section, entry in done.items():
        if entry["level"] != level:
            continue
        for test, row in entry["rows"].items():
            rows.append((f"`{section}`", test, na(row.get("sharpe")), na(row.get("delta")),
                         na(row.get("threshold")), na(row.get("p")), na(row.get("p2")),
                         str(row.get("verdict")), str(row.get("note") or "")))
    return "\n".join([
        "## Sensitivities (§12.13)",
        "",
        "They never found a PASS and never revoke one; a sign reversal is written here.",
        "",
        table(("section", "test", "sharpe", "delta", "threshold", "p", "p2", "verdict",
               "note"), rows) if rows else "No sensitivity row was spent at this level.",
        "",
        f"Level {level} label (§12.13): **{labels[level]}**.",
        "",
        SHARPE_CAVEAT,
    ])


def sensitivities_summary(
    done: Mapping[str, Any], labels: Mapping[str, str], primary_levels: Mapping[str, str]
) -> str:
    rows = [(level, primary_levels[level], f"**{labels[level]}**") for level in LEVELS]
    spent = sum(len(entry["rows"]) for entry in done.values())
    return "\n".join([
        "## Sensitivities (§12.13)",
        "",
        table(("level", "final verdict", "with the §12.13 label"), rows),
        "",
        f"{spent} sensitivity rows logged; details in each level's results file.",
    ])


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=("instrument", "read", "resume-log", "holm",
                                            "sensitivities"))
    parser.add_argument("level", nargs="?", choices=LEVELS)
    parser.add_argument("--workers", type=int, default=None,
                        help="processes for the placebo loops (default: the CPU count)")
    args = parser.parse_args(argv)
    if (args.command in ("read", "resume-log")) != (args.level is not None):
        parser.error("'read' and 'resume-log' take one level, A, B or C; the other commands "
                     "take none")
    setup = Setup(workers=args.workers)
    start = time.perf_counter()
    if args.command == "holm":
        run_holm(setup)
    elif args.command == "resume-log":
        resume_logging(args.level, setup)
    else:
        data = tree.load_tree_data(setup.root)
        print(f"{data!r}; lock {tree.LOCK_COMMIT}; code at {tree.git_head(setup.root)[:7]}")
        if args.command == "instrument":
            sections = run_instruments(data, setup)
            write_instruments(sections, setup)
            print(f"thresholds: {THRESHOLDS_FILE} ({len(sections)} sections); results: "
                  + ", ".join(RESULTS_FILE.format(level=level) for level in LEVELS)
                  + ". Nothing of the tree was read.")
        elif args.command == "read":
            run_read(args.level, data, setup)
        else:
            run_sensitivities(data, setup)
    print(f"done in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
