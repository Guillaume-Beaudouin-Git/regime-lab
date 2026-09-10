"""The cross-section, and the reason it is smaller than it looks.

Nineteen sovereigns have free monthly yield history back to 1990. They are not
nineteen independent bets: eight of them share the euro, which means one policy
rate and one currency between them. Their **bond** signals stay distinct, because
a Bund and a BTP price different credit even under one central bank, but their
**policy** and **currency** signals are the same series wearing different names.

Effective breadth is therefore reported alongside the nominal count everywhere,
because a cross-sectional book that looks like nineteen names and behaves like
eleven will overstate every statistic built on it.
"""

from __future__ import annotations

#: ``{code: (name, currency, equity index ticker or None)}``
COUNTRIES = {
    "US": ("United States", "USD", "^GSPC"),
    "DE": ("Germany", "EUR", "^GDAXI"),
    "GB": ("United Kingdom", "GBP", "^FTSE"),
    "JP": ("Japan", "JPY", "^N225"),
    "AU": ("Australia", "AUD", "^AXJO"),
    "FR": ("France", "EUR", "^FCHI"),
    "CA": ("Canada", "CAD", "^GSPTSE"),
    "IT": ("Italy", "EUR", "FTSEMIB.MI"),
    "CH": ("Switzerland", "CHF", "^SSMI"),
    "ES": ("Spain", "EUR", "^IBEX"),
    "NZ": ("New Zealand", "NZD", "^NZ50"),
    "SE": ("Sweden", "SEK", "^OMX"),
    "DK": ("Denmark", "DKK", None),
    "NL": ("Netherlands", "EUR", "^AEX"),
    "BE": ("Belgium", "EUR", "^BFX"),
    "NO": ("Norway", "NOK", None),
    "FI": ("Finland", "EUR", None),
    "IE": ("Ireland", "EUR", None),
    "PT": ("Portugal", "EUR", None),
}

#: Currency against the dollar, ``{currency: yahoo ticker}``. A rise in the
#: quoted rate means the currency strengthens against the dollar.
FX = {
    "EUR": "EURUSD=X", "GBP": "GBPUSD=X", "JPY": "JPY=X", "AUD": "AUDUSD=X",
    "CAD": "CAD=X", "CHF": "CHF=X", "NZD": "NZDUSD=X", "SEK": "SEK=X",
    "DKK": "DKK=X", "NOK": "NOK=X",
}

#: Yahoo quotes JPY, CAD, CHF, SEK, DKK and NOK as dollars-per-unit inverted —
#: the ticker is USD/XXX, so a rise means the dollar strengthens. Flipping them
#: here rather than at the point of use keeps one convention in the whole study.
FX_INVERTED = {"JPY", "CAD", "CHF", "SEK", "DKK", "NOK"}

#: Modified duration used to turn a ten-year yield into a total return.
DURATION_10Y = 8.0


def long_rate(code: str) -> str:
    return f"IRLTLT01{code}M156N"


def short_rate(code: str) -> str:
    return f"IR3TIB01{code}M156N"


def effective_breadth(codes: list[str]) -> dict[str, int]:
    """Independent policy and currency blocs among ``codes``."""
    currencies = {COUNTRIES[c][1] for c in codes if c in COUNTRIES}
    return {"sovereigns": len(codes), "currencies": len(currencies)}
