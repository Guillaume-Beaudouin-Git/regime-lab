"""The instruments and macro series the study draws on.

Every entry has to earn its place by feeding a feature the thesis needs. The
catalogue grows against that test, not to look comprehensive.
"""

from __future__ import annotations

#: Cross-asset daily prices, ``{series_id: yahoo_symbol}``.
PRICES = {
    "eq_us_large": "^GSPC",
    "eq_us_small": "^RUT",
    "eq_us_tech": "^IXIC",
    "eq_jp": "^N225",
    "bond_us_long": "^TYX",
    "bond_us_10y": "^TNX",
    "fx_dollar": "DX-Y.NYB",
    "vol_vix": "^VIX",
}

#: Daily prices sourced from FRED because Yahoo's history is too short,
#: ``{series_id: fred_id}``.
#:
#: Yahoo's continuous futures for oil and gold begin in late 2000. Keeping them
#: would have started the common sample in November 2001 and cost four of the
#: fourteen stress episodes, to buy two features out of forty-seven. WTI has a
#: full daily history on FRED and is substituted. Gold has no free long series:
#: the LBMA fixing was withdrawn from FRED, the bullion ETF starts in 2004, and
#: the Philadelphia index is gold *miners* — an equity sector that fell with the
#: market in 2008, so using it would inject equity beta into a feature labelled
#: as a safe haven. Gold is dropped rather than proxied badly.
PRICES_FRED = {
    "cmd_oil": "DCOILWTICO",
}

#: Series taken as first releases from ALFRED, ``{series_id: fred_id}``.
#:
#: These four carry the revision study. All have vintage history back to 1990,
#: which most FRED series do not: initial claims start in 2009 and the fourth
#: revision of the St. Louis stress index in 2022, so asking for their vintages
#: buys a short series rather than a real-time one.
MACRO_VINTAGED = {
    "macro_indpro": "INDPRO",
    "macro_payems": "PAYEMS",
    "macro_unrate": "UNRATE",
    "macro_cpi": "CPIAUCSL",
}

#: Series taken as current values plus a publication lag,
#: ``{series_id: (fred_id, frequency, revised)}``.
#:
#: ``revised=False`` means the lag path is exact, not an approximation: a market
#: rate printed on a given day is never restated.
MACRO_LAGGED = {
    "macro_claims": ("ICSA", "weekly", True),
    # NFCI only. The St. Louis stress index measures the same construct but
    # begins in 1994, which would cost four years including the 1990 recession.
    "fin_nfci": ("NFCI", "weekly", True),
    "fin_baa_spread": ("BAA10Y", "daily", False),
    "fin_aaa_spread": ("AAA10Y", "daily", False),
    "rate_curve_10y2y": ("T10Y2Y", "daily", False),
    "rate_curve_10y3m": ("T10Y3M", "daily", False),
    "rate_cash_3m": ("DTB3", "daily", False),
}
