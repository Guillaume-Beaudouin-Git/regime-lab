"""Two data layers, because one of them is biased and the other is not.

**Ken French portfolios** are built on the full CRSP universe, including every
company that later delisted. They carry no survivorship bias and no gaps, and
they are the only way to measure how large the reversal premium actually was.

**Individual tickers from a retail feed** carry the bias in its most damaging
form for this exact study: the stocks missing from today's listing are the ones
that fell and never came back — which is to say, precisely the losers a reversal
strategy would have bought. Any long-loser result on that layer is optimistic by
construction, and is only ever used to ask whether a *conditioner* helps, never
to size the effect.
"""

from __future__ import annotations

#: Unbiased CRSP-universe panels, ``{name: zip file}``.
FRENCH = {
    "st_reversal_factor": "F-F_ST_Reversal_Factor_daily_CSV.zip",
    "lt_reversal_factor": "F-F_LT_Reversal_Factor_daily_CSV.zip",
    "momentum_factor": "F-F_Momentum_Factor_daily_CSV.zip",
    "prior_1_0": "10_Portfolios_Prior_1_0_Daily_CSV.zip",
    "size_prior_1_0": "6_Portfolios_ME_Prior_1_0_Daily_CSV.zip",
    "factors_5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
}

#: Large, liquid US names for the conditioning layer. Survivorship-biased and
#: labelled as such everywhere it is used.
TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B", "JPM", "V", "JNJ", "WMT", "PG", "MA", "HD", "CVX", "ABBV", "KO", "PEP", "COST", "BAC", "AVGO", "MRK", "ADBE", "CSCO", "ACN", "LLY", "MCD", "TMO", "ABT", "CRM", "NKE", "DHR", "TXN", "LIN", "NEE", "VZ", "CMCSA", "PM", "WFC", "UPS", "RTX", "INTC", "AMD", "QCOM", "HON", "SPGI", "LOW", "UNP", "BA", "CAT", "GS", "IBM", "BLK", "DE", "ELV", "AMGN", "SBUX", "GILD", "MDT", "ISRG", "PLD", "BKNG", "ADP", "MMM", "AXP", "TJX", "CVS", "MDLZ", "SYK", "CI", "ZTS", "MO", "CB", "SO", "DUK", "BDX", "MU", "LRCX", "ADI", "KLAC", "SNPS", "CDNS", "ORLY", "MAR", "APD", "ECL", "SHW", "NSC", "CSX", "PSA", "AMT", "CCI", "EQIX", "SPG", "O", "WELL", "VTR", "DLR", "ITW", "EMR", "ETN", "PH", "ROK", "DOV", "XYL", "FDX", "LUV", "DAL", "UAL", "AAL", "CCL", "RCL", "MGM", "WYNN", "LVS", "F", "GM", "HOG", "WHR", "LEG", "NWL", "HAS", "MAT"]
