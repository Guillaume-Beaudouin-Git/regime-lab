"""The Bridgewater power script must refuse the real labels and compute its power honestly.

Synthetic data only: nothing here reads ``data/``. What is tested is what the pre-lock
power measurement relies on:
- the guard refuses a registered real label path, alone or inside a per-fold mapping,
  and lets placebo paths through;
- the percentile rule counts ties half, as `selection.protocol.placebo_percentile` does;
- the location-shift power is the nominal size at zero shift, grows with the shift,
  and respects the absolute floor;
- a null summary reports the thresholds and every power the script prints.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

from regime_lab.config import ROOT
from regime_lab.construction import evaluate as E
from regime_lab.selection.folds import walk_forward_folds
from regime_lab.selection.protocol import placebo_percentile

SCRIPT = ROOT / "scripts" / "measure_bridgewater_power.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("measure_bridgewater_power", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def world():
    sessions = pd.bdate_range("2004-01-02", periods=14 * 252)
    stamps = (pd.date_range(sessions[0] - pd.Timedelta(days=100), sessions[-1], freq="QS-FEB")
              + pd.Timedelta(days=13))
    rng = np.random.default_rng(0)
    stamped = pd.Series(rng.choice(4, len(stamps)).astype(float), index=stamps)
    returns = pd.DataFrame(rng.standard_normal((len(sessions), 5)) * 0.01, index=sessions,
                           columns=list(E.SLEEVES))
    folds = walk_forward_folds(sessions, n_folds=5, test_years=2.0, anchor="end")
    return sessions, stamped, returns, folds


def test_the_guard_refuses_the_real_path_and_passes_placebos(script, world):
    sessions, stamped, returns, folds = world
    engine = script.GuardedLevelA(returns, folds)
    real = E.expand_to_sessions(stamped, sessions)
    engine.forbid(real)
    with pytest.raises(RuntimeError, match="blindness guard"):
        engine.run(real)
    with pytest.raises(RuntimeError, match="blindness guard"):
        engine.run({f.number: real for f in folds})
    with pytest.raises(RuntimeError, match="blindness guard"):
        engine.run(real.copy().rename("a copy is refused too"))
    paths = E.placebo_session_paths(engine, stamped, 3, seed=1)
    for column in paths.columns:
        assert np.isfinite(engine.run(paths[column]).reduction)
    null = E.null_statistics(engine, stamped, 3, statistics={"r": lambda r: r.reduction},
                             seed=1)
    assert null["r"].finite and engine.runs == 6


def test_percentile_ties_count_half(script):
    null = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
    values = np.array([-1.0, 1.0, 2.5, 9.0])
    expected = [placebo_percentile(v, null) for v in values]
    np.testing.assert_allclose(script.pct_rank(null, values), expected)


def test_shift_power_is_the_size_at_zero_and_grows(script):
    null = np.random.default_rng(3).standard_normal(4000)
    assert script.power_shift(null, 0.0, 0.05) == pytest.approx(0.05, abs=0.005)
    powers = [script.power_shift(null, s, 0.05) for s in (0.5, 1.5, 2.5, 5.0)]
    assert all(b >= a for a, b in zip(powers, powers[1:], strict=False))
    assert powers[-1] == 1.0
    # z_0.95 + z_0.80 = 2.49 standard deviations buys 80% power under a Gaussian null.
    assert script.power_shift(null, 2.486, 0.05) == pytest.approx(0.80, abs=0.03)
    assert script.power_shift(null, 5.0, 0.05, floor=100.0) == 0.0


def test_a_null_summary_carries_thresholds_and_powers(script):
    null = E.NullDistribution(np.random.default_rng(4).normal(-0.02, 0.15, 2000))
    s = script.summarise("synthetic", null)
    assert s["q95"] < s["q_sidak"] <= s["q99"] + 1e-9
    assert s["holm_quantiles"][0] == pytest.approx(null.quantile(0.99))
    for label in ("log 1.5", "log 1.2", "0.204"):
        p = s["power"][label]
        assert p["sidak"] <= p["alpha_05"] and p["alpha_05_and_floor_0204"] <= p["alpha_05"]
    assert s["power"]["log 1.5"]["alpha_05"] > s["power"]["log 1.2"]["alpha_05"]


def test_label_codes_are_label_only(script):
    path = pd.Series([np.nan, 0.0, 3.0, 2.0])
    np.testing.assert_array_equal(script.codes_of(path), [-1, 0, 3, 2])
