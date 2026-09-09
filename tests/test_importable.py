"""Can a fresh checkout actually run?

The existing suite passed on a checkout whose package could not be imported:
every test happened to exercise one of the four modules that do not reach
``regime_lab.config``, and ``config.py`` was excluded by a *global* gitignore
rule that ``git status`` never mentions. A green suite said nothing about
whether anyone else could run the study.

This file closes that hole from both ends: every module must import, and every
script must import, so a missing file or an undeclared dependency fails here
rather than in a stranger's terminal.
"""

from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import regime_lab

ROOT = Path(regime_lab.__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))


def all_modules() -> list[str]:
    return [name for _, name, _ in pkgutil.walk_packages(regime_lab.__path__, "regime_lab.")]


@pytest.mark.parametrize("module", all_modules())
def test_every_module_imports(module):
    importlib.import_module(module)


def test_the_package_ships_its_configuration():
    """config.py is tracked. It was not, and nothing reported it."""
    listed = subprocess.run(
        ["git", "ls-files", "regime_lab/config.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert listed.stdout.strip() == "regime_lab/config.py", (
        "regime_lab/config.py is not tracked by git. A global gitignore rule can "
        "exclude it invisibly: git status stays silent and the public repository "
        "cannot be imported."
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_every_script_imports(script):
    """Compile and import each entry point without running it."""
    result = subprocess.run(
        [sys.executable, "-c", f"import runpy, ast; ast.parse(open({str(script)!r}).read())"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    module = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import importlib.util as u; "
            f"s=u.spec_from_file_location('m', {str(script)!r}); "
            f"m=u.module_from_spec(s); "
            f"__import__('sys').modules['m']=m; "
            f"exec(compile(open({str(script)!r}).read(), 'm', 'exec'), "
            f"{{'__name__': '__not_main__'}})",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert module.returncode == 0, module.stderr[-1500:]
