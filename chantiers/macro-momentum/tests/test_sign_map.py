"""The sign map in the code must match the one frozen in the pre-specification.

`macro_momentum/exposures.py` has claimed since it was written that "the test suite
checks that". It did not: `tests/` was empty. The claim was found on 2026-09-21 in the
same sweep that found three other places where this programme published something its
code did not produce, and it is the reason this file exists.

The map is the object most at risk in a study of this kind. H1 was falsified precisely
on it — the frozen map sits at the 7th percentile of 400 random sign maps, and 373 of
them beat it — so a silent drift between the document and the code would have made that
result unreadable after the fact. The test therefore **parses the frozen table** rather
than restating its values: a copy of the numbers would agree with itself forever.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from macro_momentum.exposures import CONTESTED, SIGN_MAP

PRESPEC = Path(__file__).resolve().parents[1] / "docs" / "PRESPEC.md"

#: The document names the themes in prose; the code names them in code.
ROW_TO_THEME = {
    "Growth": "growth",
    "Inflation": "inflation",
    "Policy tightening": "policy",
    "Credit spread": "risk",
}
COLUMNS = ("equities", "bonds", "commodities", "dollar")


def frozen_map() -> dict[str, dict[str, int]]:
    """Read the sign table out of docs/PRESPEC.md, signs and all."""
    parsed: dict[str, dict[str, int]] = {}
    for line in PRESPEC.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 5:
            continue
        theme = ROW_TO_THEME.get(cells[0])
        if theme is None:
            continue
        signs = []
        for cell in cells[1:]:
            # The table writes them bold and with a typographic minus.
            value = re.sub(r"[*\s]", "", cell).replace("−", "-")
            signs.append(int(value))
        parsed[theme] = dict(zip(COLUMNS, signs, strict=True))
    return parsed


def test_the_document_still_contains_a_readable_table():
    parsed = frozen_map()
    assert set(parsed) == set(ROW_TO_THEME.values()), (
        "the frozen table could not be parsed; a test that silently finds nothing "
        "is worse than no test"
    )
    assert all(len(row) == len(COLUMNS) for row in parsed.values())


def test_the_code_map_matches_the_frozen_map_cell_by_cell():
    parsed = frozen_map()
    assert parsed == SIGN_MAP, "the code and the pre-specification disagree"


@pytest.mark.parametrize("theme", sorted(ROW_TO_THEME.values()))
def test_every_sign_is_plus_or_minus_one(theme):
    assert set(SIGN_MAP[theme].values()) <= {-1, +1}


def test_the_contested_cells_are_named_in_the_document():
    text = PRESPEC.read_text().lower()
    assert CONTESTED, "the pre-specification flags contestable cells; the code must too"
    for theme, asset in CONTESTED:
        assert theme in text and asset in text, (
            f"({theme}, {asset}) is flagged contested in the code but the frozen "
            "document does not discuss it"
        )


def test_the_map_was_never_flipped():
    """H1 was falsified without reversing the signs, and that refusal is the result.

    Reversing them after seeing the outcome would have turned -0.38 into +0.32. This
    pins the two cells whose reversal would be most tempting, so a later edit cannot
    quietly do it.
    """
    assert SIGN_MAP["growth"]["equities"] == +1
    assert SIGN_MAP["risk"]["bonds"] == +1
