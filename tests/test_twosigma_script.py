"""The Two Sigma CLI must run the lock's §13.1 in its order, and refuse everything else.

Every test runs on synthetic data (``tree.synthetic_tree_data``) inside a temporary git
repository holding a copy of ``uv.lock`` and two stand-in input files; nothing here reads
``data/`` or writes to the real trial register. What is tested is what a reading would
silently depend on: the reading refuses to run when the threshold file is untracked or
modified, when an input's SHA-256 or a package version differs, or when a threshold
differs bitwise; the readings run in the order A, B, C, once each; the instrument prints
no reading quantity, even with every reading function made to raise; the instrument will
not overwrite committed thresholds; and every criterion the results files quote is the
lock's own text.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from regime_lab.config import ROOT
from regime_lab.selection import level_c, protocol
from regime_lab.selection import tree as T

SCRIPT = ROOT / "scripts" / "run_twosigma_tree.py"
DRAWS = 6
BOOT = 50
INPUTS = ("data/raw/a.parquet", "data/cache/b.parquet")
VARIANTS = (T.PRIMARY, T.D025)


def _load_script():
    spec = importlib.util.spec_from_file_location("run_twosigma_tree", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_twosigma_tree"] = module
    spec.loader.exec_module(module)
    return module


S = _load_script()


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                    *args], check=True, capture_output=True)


def _commit(root: Path, *paths: str) -> None:
    _git(root, "add", "--", *paths)
    _git(root, "commit", "-q", "-m", "commit")


@pytest.fixture(scope="module")
def data() -> T.TreeData:
    return T.synthetic_tree_data(seed=0, n_industries=49, test_years=2.0, end="2019-12-31")


@pytest.fixture(scope="module")
def emitted() -> list[str]:
    return []


@pytest.fixture(scope="module")
def sections(data, emitted) -> dict:
    """The instruments at the primary and at d=0.25, with every reading function made to
    raise: the instrument must never reach one (§13.1 step 2)."""

    def refuse(*_args, **_kwargs):
        raise AssertionError("the instrument called a reading function")

    setup = S.Setup(draws=DRAWS, boot_draws=BOOT, workers=1)
    with pytest.MonkeyPatch.context() as patch:
        for owner, name in ((T, "sharpe"), (level_c, "sharpe"), (T, "pair_reading"),
                            (protocol, "realised_beta"), (protocol, "placebo_p_value"),
                            (protocol, "placebo_percentile")):
            patch.setattr(owner, name, refuse)
        return S.run_instruments(data, setup, variants=VARIANTS, emit=emitted.append)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for relative in INPUTS:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(relative.encode() * 10)
    shutil.copy(ROOT / "uv.lock", root / "uv.lock")
    (root / CODE).write_text("# stands for the code a reading runs\n")
    _git(root, "init", "-q")
    _commit(root, "uv.lock", CODE)
    return root


#: Stands for :data:`S.CODE_FILES` in the synthetic repository.
CODE = "code.py"


def _setup(repo: Path) -> S.Setup:
    return S.Setup(root=repo, inputs=INPUTS, draws=DRAWS, boot_draws=BOOT, workers=1,
                   trials_path=repo.parent / "trials.parquet", code_files=(CODE,))


def _results(level: str) -> str:
    return S.RESULTS_FILE.format(level=level)


# ------------------------------------------------------------------------ the plan


def test_the_plan_is_the_declared_instruments_and_rows():
    sections = [T.section_name(level, v) for v in S.INSTRUMENT_VARIANTS
                for level in S.levels_of(v)]
    assert len(sections) == len(set(sections)) == 17
    assert sections[:3] == ["A", "B", "C"]
    assert {"A:d=0.25", "A:d=1.00", "C:smooth21", "B:K=6"} <= set(sections)
    assert not any(s.startswith(("B:d", "C:d")) for s in sections)
    rows = sum(len(S.levels_of(v)) for v in S.SENSITIVITIES)
    assert rows + len(T.PRIMARIES) == len(T.DECLARED_ROWS) == 20


def test_every_criterion_is_copied_verbatim_from_the_lock():
    lock = " ".join((ROOT / "docs" / "PRESPEC_TWOSIGMA.md").read_text().split())
    for criterion in S.CRITERIA:
        for text in criterion.text:
            assert " ".join(text.split()) in lock, (criterion.lock, text[:60])
    assert {c.lock for c in S.CRITERIA} >= set(T.PRIMARIES)


# ------------------------------------------------------------------ the instrument


def test_the_instrument_prints_its_printouts_and_no_reading_quantity(sections, emitted):
    assert set(sections) == {"A", "B", "C", "A:d=0.25"}
    assert "A-2" not in sections["A:d=0.25"] and "nfci" in sections["A"]
    text = "\n".join(emitted)
    for section in sections:
        assert f"=== section {section} " in text
    for forbidden in ("sharpe", "Sharpe", "p_boot", "p_hac", "beta", "R_W", "S_C", "S_W",
                      "G_diag", "h_fk", "gate", "per_state"):
        assert forbidden not in text, forbidden
    assert repr(sections["A"]["A-1"]["T_A1"]) in text


def test_the_instrument_writes_thresholds_and_results_and_never_overwrites_them(
        repo, sections):
    setup = _setup(repo)
    payload = S.write_instruments(sections, setup)
    stored = T.read_thresholds(setup.thresholds)
    assert set(stored["thresholds"]) == set(sections) and payload["inputs"] == stored["inputs"]
    assert stored["thresholds"]["A"] == json.loads(json.dumps(sections["A"]))
    for level in S.LEVELS:
        text = (repo / _results(level)).read_text()
        assert "Nothing of the tree was read." in text
        for block in S.BLOCKS:
            assert f"<!-- {block}:BEGIN -->" in text and f"<!-- {block}:END -->" in text
        for criterion in S.CRITERIA:
            if criterion.level == level:
                assert f"**{criterion.lock}, {criterion.section}:**" in text
    assert "`A:d=0.25`" in (repo / _results("A")).read_text()
    S.write_instruments(sections, setup)  # untracked: the instrument may run again
    _commit(repo, T.THRESHOLDS_FILE)
    with pytest.raises(RuntimeError, match="amendment"):
        S.write_instruments(sections, setup)


# --------------------------------------------------------------------- the refusal


def test_the_reading_refuses_unless_its_thresholds_verify(repo, data, sections):
    setup = _setup(repo)
    S.write_instruments(sections, setup)
    results = [_results(level) for level in S.LEVELS]
    with pytest.raises(T.ReadingRefused, match="not tracked"):
        S.run_read("A", data, setup, emit=lambda _: None)  # results file not committed
    _commit(repo, *results)
    with pytest.raises(T.ReadingRefused, match="not tracked"):
        S.run_read("A", data, setup, emit=lambda _: None)  # thresholds not committed
    _commit(repo, T.THRESHOLDS_FILE)
    path = setup.thresholds
    original = path.read_text()

    path.write_text(original.replace('"lock"', '"lock" ', 1))
    with pytest.raises(T.ReadingRefused, match="differs from its committed version"):
        S.run_read("A", data, setup, emit=lambda _: None)
    _git(repo, "checkout", "-q", "--", T.THRESHOLDS_FILE)

    (repo / INPUTS[0]).write_bytes(b"republished")
    with pytest.raises(T.ReadingRefused, match="SHA-256"):
        S.run_read("A", data, setup, emit=lambda _: None)
    (repo / INPUTS[0]).write_bytes(INPUTS[0].encode() * 10)

    lock = (repo / "uv.lock").read_text()
    numpy_version = T.locked_versions(repo)["numpy"]
    (repo / "uv.lock").write_text(lock.replace(
        f'name = "numpy"\nversion = "{numpy_version}"', 'name = "numpy"\nversion = "0.0.1"'))
    with pytest.raises(T.ReadingRefused, match="package versions"):
        S.run_read("A", data, setup, emit=lambda _: None)
    _git(repo, "checkout", "-q", "--", "uv.lock")

    payload = json.loads(original)
    entry = payload["thresholds"]["A"]["A-1"]["T_A1"]
    nudged = np.nextafter(float.fromhex(entry["float.hex"]), np.inf)
    payload["thresholds"]["A"]["A-1"]["T_A1"] = {"float.hex": nudged.hex(), "repr": repr(nudged)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    _commit(repo, T.THRESHOLDS_FILE)
    with pytest.raises(T.ReadingRefused, match="bitwise"):
        S.run_read("A", data, setup, emit=lambda _: None)
    path.write_text(original)
    _commit(repo, T.THRESHOLDS_FILE)

    with pytest.raises(T.ReadingRefused, match="reading of level A"):
        S.run_read("B", data, setup, emit=lambda _: None)
    assert not setup.trials_path.exists()
    assert not (repo / S.READING_FILE.format(level="A")).exists()


def test_the_reading_refuses_uncommitted_code_and_a_logged_level(repo, data, sections):
    """Amendment of 2026-09-23: every code file a reading runs must be committed and
    unmodified, and a level already in the register is refused before any computation."""
    setup = _setup(repo)
    S.write_instruments(sections, setup)
    _commit(repo, T.THRESHOLDS_FILE, *(_results(level) for level in S.LEVELS))
    (repo / CODE).write_text("# an uncommitted edit\n")
    with pytest.raises(T.ReadingRefused, match="differs from its committed version"):
        S.run_read("A", data, setup, emit=lambda _: None)
    _git(repo, "checkout", "-q", "--", CODE)

    T.log_trial(test="A-1", variant=T.PRIMARY, sharpe=0.1, delta=0.1, threshold=0.338,
                p=0.5, verdict=T.UNDERPOWERED, sessions=10, path=setup.trials_path)
    with pytest.raises(T.ReadingRefused, match="already in the register"):
        S.run_read("A", data, setup, emit=lambda _: None)
    assert not (repo / S.READING_FILE.format(level="A")).exists()
    setup.refuse_logged(("B-1", "B-2"), T.PRIMARY.name)  # another level is not refused
    assert S.CODE_FILES and all((ROOT / f).exists() for f in S.CODE_FILES)
    assert "scripts/run_twosigma_tree.py" in S.CODE_FILES


def test_the_register_stores_a_column_as_text_when_families_mix_text_and_numbers(tmp_path):
    """The first primary reading of A crashed here: ``m_note`` held a flag ``1.0`` from
    another family and pyarrow refused the text note of the new row."""
    path = tmp_path / "trials.parquet"
    T.trials.log("other", {"x": 1}, {"note": 1.0, "sharpe": 0.1}, path=path)
    T.trials.log("twosigma", {"x": 2}, {"note": "", "sharpe": float("nan")}, path=path)
    T.trials.log("twosigma", {"x": 3}, {"note": "a sentence", "sharpe": 0.2}, path=path)
    frame = T.trials.read(path)
    assert frame["m_note"].tolist() == ["1.0", "", "a sentence"]
    assert frame["m_sharpe"].dtype == float and np.isnan(frame["m_sharpe"].iloc[1])


def test_a_reading_that_crashed_while_logging_resumes_without_recomputing(repo, data, sections,
                                                                         monkeypatch):
    setup = _setup(repo)
    S.write_instruments(sections, setup)
    _commit(repo, T.THRESHOLDS_FILE, *(_results(level) for level in S.LEVELS))
    quiet = dict(emit=lambda _: None)
    with pytest.raises(T.ReadingRefused, match="no reading artifact"):
        S.resume_logging("A", setup, **quiet)

    def crash(*_args, **_kwargs):
        raise RuntimeError("the register refused the row")

    with monkeypatch.context() as patch:
        patch.setattr(S, "_log_rows", crash)
        with pytest.raises(RuntimeError, match="refused the row"):
            S.run_read("A", data, setup, **quiet)
    artifact = repo / S.READING_FILE.format(level="A")
    computed = json.loads(artifact.read_text())
    assert computed["rows_logged"] is False and not setup.trials_path.exists()
    with pytest.raises(T.ReadingRefused, match="read already"):
        S.run_read("A", data, setup, **quiet)

    def recompute(*_args, **_kwargs):
        raise AssertionError("resuming must not recompute the reading")

    with monkeypatch.context() as patch:
        patch.setattr(S, "read_level", recompute)
        stored = S.resume_logging("A", setup, **quiet)
    assert stored["rows_logged"] is True and stored["logging_head"] == T.git_head(repo)
    assert stored["reading"] == computed["reading"]
    logged = [json.loads(c)["test"] for c in T.trials.read(setup.trials_path)["config"]]
    assert sorted(logged) == sorted(computed["reading"]["trial_rows"])
    assert "## Reading (§13.1 step 4)" in (repo / _results("A")).read_text()
    with pytest.raises(T.ReadingRefused, match="already logged"):
        S.resume_logging("A", setup, **quiet)


def test_the_register_takes_rows_only_at_the_lock_constants():
    with pytest.raises(ValueError, match="lock's"):
        S.Setup(draws=DRAWS).check_register()
    S.Setup().check_register()


# ------------------------------------------------------------- the order, end to end


def test_the_readings_run_once_in_order_then_holm_then_the_sensitivities(repo, data,
                                                                          sections):
    setup = _setup(repo)
    S.write_instruments(sections, setup)
    _commit(repo, T.THRESHOLDS_FILE, *(_results(level) for level in S.LEVELS))
    quiet = dict(emit=lambda _: None)

    readings = {}
    for level in S.LEVELS:
        readings[level] = S.run_read(level, data, setup, **quiet)
        artifact = S.READING_FILE.format(level=level)
        stored = json.loads((repo / artifact).read_text())
        assert stored["rows_logged"] and stored["reading"]["verdicts"] == json.loads(
            json.dumps(T.plain(readings[level]["verdicts"])))
        text = (repo / _results(level)).read_text()
        assert "Not read. The reading" not in text and "## Reading (§13.1 step 4)" in text
        with pytest.raises(T.ReadingRefused, match="already"):
            S.run_read(level, data, setup, **quiet)
        if level != "C":
            next_level = S.LEVELS[S.LEVELS.index(level) + 1]
            with pytest.raises(T.ReadingRefused, match=f"reading of level {level}"):
                S.run_read(next_level, data, setup, **quiet)
        _commit(repo, artifact, _results(level))

    logged = T.trials.read(setup.trials_path)
    tests = [json.loads(c)["test"] for c in logged["config"]]
    spent = [t for level in S.LEVELS for t in readings[level]["trial_rows"]]
    assert sorted(tests) == sorted(spent) and len(tests) <= len(T.PRIMARIES)

    with pytest.raises(T.ReadingRefused, match="Holm step"):
        S.run_sensitivities(data, setup, variants=(T.D025,), **quiet)
    holm = S.run_holm(setup, **quiet)
    assert [row["test"] for row in holm["table"]] and len(holm["table"]) == 6
    assert holm["p_entered"]["C-2"] == 1.0
    for level in S.LEVELS:
        assert holm["levels"][level] in T.LOCK_VERDICTS
    summary = (repo / S.SUMMARY_FILE).read_text()
    assert "Holm–Bonferroni over the six primaries (§13.3)" in summary
    _commit(repo, S.HOLM_FILE, S.SUMMARY_FILE)

    done = S.run_sensitivities(data, setup, variants=(T.D025,), **quiet)
    assert set(done["sections"]) == {"A:d=0.25"}
    assert done["labels"]["A"] == holm["levels"]["A"]  # a d row never labels (§12.13)
    again = S.run_sensitivities(data, setup, variants=(T.D025,), **quiet)

    def same(x):
        return json.dumps(x, sort_keys=True)

    assert same(again["sections"]["A:d=0.25"]) == same(done["sections"]["A:d=0.25"])
    configs = [json.loads(c) for c in T.trials.read(setup.trials_path)["config"]]
    assert sum(c["variant"] == "d=0.25" for c in configs) <= 1
    text = (repo / _results("A")).read_text()
    assert re.search(r"Level A label \(§12\.13\): \*\*", text)


def test_a_printout_line_is_every_leaf(sections):
    text = S.format_printout("C", sections["C"], 1.0)
    for key in ("MDE_C", "S_star", "q99-q20", "powered", "twin"):
        assert key in text
    assert S.fmt(float("nan")) == T.NON_FINITE and S.fmt(True) == "yes"
