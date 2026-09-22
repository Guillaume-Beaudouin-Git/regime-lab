"""Does the modelled surprise resemble the market's, where the market's is free?

The pre-specification makes this a gate. A time-series model does not know what
a hurricane did to payrolls and economists do, so a proxy that fails to track
real consensus is not a weaker version of the signal — it is a different signal
wearing its name.

The Philadelphia Fed's Survey of Professional Forecasters is the only free
historical consensus available: quarterly, United States, published with its own
survey dates. Where it overlaps our indicators, the two surprises are compared.
"""

from __future__ import annotations

import io
import warnings

import numpy as np
import pandas as pd
import requests

from macro_momentum import config
from macro_momentum.surprise import first_releases, model_surprise

warnings.filterwarnings("ignore")
SPF = "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/survey-of-professional-forecasters/data-files/files"
RULE = "=" * 78


def spf_table(filename: str) -> pd.DataFrame:
    resp = requests.get(f"{SPF}/{filename}", timeout=60)
    resp.raise_for_status()
    return pd.read_excel(io.BytesIO(resp.content))


def main() -> None:
    print(RULE)
    print("VALIDATION DU PROXY DE SURPRISE contre le consensus reel (SPF)")
    print(RULE)

    table = spf_table("median_unemp_level.xlsx")
    cols = {c.upper(): c for c in table.columns}
    year, quarter = cols.get("YEAR"), cols.get("QUARTER")
    nowcast = cols.get("UNEMP2")   # prevision du trimestre en cours
    prior = cols.get("UNEMP1")     # trimestre precedent, connu a l'enquete

    spf = table.loc[:, [year, quarter, prior, nowcast]].dropna()
    spf.columns = ["year", "quarter", "known", "nowcast"]
    # pandas 3 dropped the year/quarter keywords on PeriodIndex.
    stamps = spf["year"].astype(int).astype(str) + "Q" + spf["quarter"].astype(int).astype(str)
    spf["period"] = pd.PeriodIndex(stamps, freq="Q").to_timestamp()
    print(f"\n   SPF chomage : {len(spf)} enquetes, "
          f"{spf['period'].min():%Y-%m} a {spf['period'].max():%Y-%m}")

    releases = first_releases(config.read("h3_macro", "unrate"))
    events = model_surprise(releases, kind="diff").dropna(subset=["z"])
    print(f"   nos surprises modelisees : {len(events)} publications mensuelles")

    # Realised quarterly average unemployment, from first releases only.
    monthly = releases.set_index("period")["value"]
    realised = monthly.resample("QS").mean().rename("realised")
    merged = spf.set_index("period").join(realised, how="inner").dropna()
    merged["spf_surprise"] = merged["realised"] - merged["nowcast"]

    ours = events.set_index("period")["surprise"].resample("QS").mean().rename("ours")
    pair = merged.join(ours, how="inner").dropna()
    pair = pair.loc["1998":]

    print(f"\n   trimestres communs : {len(pair)}  ({pair.index.min():%Y-%m} a "
          f"{pair.index.max():%Y-%m})\n")

    corr = pair["spf_surprise"].corr(pair["ours"])
    rank = pair["spf_surprise"].corr(pair["ours"], method="spearman")
    sign = float((np.sign(pair["spf_surprise"]) == np.sign(pair["ours"])).mean())
    print(f"   {'correlation de Pearson':<38} {corr:>+8.3f}")
    print(f"   {'correlation de rang':<38} {rank:>+8.3f}")
    print(f"   {'accord de signe':<38} {sign:>8.1%}")
    print(f"   {'ecart-type surprise SPF':<38} {pair['spf_surprise'].std():>8.3f} pt")
    print(f"   {'ecart-type notre surprise':<38} {pair['ours'].std():>8.3f} pt")

    print("\n   Un consensus d'economistes bat un modele autoregressif : c'est attendu.")
    print("   La question est de savoir si les deux mesurent la meme chose.")
    verdict = ("PROXY UTILISABLE" if abs(corr) > 0.4 and sign > 0.6
               else "PROXY FAIBLE — a declarer comme limite majeure")
    print(f"\n   verdict : {verdict}")
    print("\n" + RULE)


if __name__ == "__main__":
    main()
