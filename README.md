# regime-lab

Market regime detection under a point-in-time data contract, and the question of
whether the regimes it finds are worth conditioning a portfolio on.

## The question

> Does a machine-learned regime model carry tradable information beyond what a
> plain volatility measure already captures — and if so, does that information
> survive transaction costs, point-in-time data constraints and a correction for
> multiple testing?

It is a decomposition, not a yes/no question: the binary version is already
settled in the literature (Cederburg, O'Doherty, Wang & Yan, *JFE* 2020), while
the magnitudes are not.

The protocol is allowed to answer that the effect is absent, or that the data
cannot tell. Stopping rules and admissible outcomes are fixed before results are
looked at: see `docs/CHARTER.html`, frozen at

```
SHA-256  5791c1887c41d1f6b0449bafad2125013b82dc6b3506b4d72564e14ec1abc0db
```

and `docs/PROTOCOL_FREEZE.md` for what the freeze commits us to.

## Why point-in-time comes first

Macro series are revised. Industrial production for March 2008, as published
today, is not the figure an observer had in March 2008. Building features from
today's series leaks information backwards and inflates any model that keys on
recessions — invisibly, and in the direction that flatters the result.

Every observation here therefore carries two dates:

| column | meaning |
| --- | --- |
| `period` | the date the observation describes |
| `available_at` | the first instant a real-time observer could have known it |

A value may enter a model at `t` only if `available_at <= t`. Superseded
revisions are kept, so any past panel can be rebuilt exactly as it stood.
`tests/test_pit.py` enforces this: a panel built for a past date must not move
when later data arrives.

## Data

| block | source | coverage |
| --- | --- | --- |
| Cross-asset daily prices | Yahoo Finance, adjusted | 1990– |
| Macro, financial conditions, credit | FRED / ALFRED | 1990– |

Credit stress uses Moody's Baa and Aaa spreads over the 10-year Treasury rather
than ICE BofA option-adjusted spreads: FRED serves ICE BofA series for the
trailing two years only, which would leave the credit block empty before 2023.

**Macro vintages.** With a free [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html)
in `.env`, macro series are downloaded as full real-time histories and
`available_at` is the true publication date. Without a key the loader falls back
to the current vintage plus a conservative publication lag: this removes the
timing leak but not the revision leak, and every result on that path is
provisional.

## Setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
cp .env.example .env          # optional: paste a free FRED key
python scripts/fetch_data.py
pytest
```

## Layout

```
regime_lab/data/pit.py         the point-in-time contract: validate, as_of, build_panel
regime_lab/data/store.py       immutable parquet store with content-hashed manifests
regime_lab/data/sources/       one module per provider
regime_lab/data/universe.py    the catalogue of series the study draws on
tests/test_pit.py              temporal integrity regression tests
docs/CHARTER.html              the frozen charter (v2)
docs/PROTOCOL_FREEZE.md        content hash, amendment log, what the freeze binds
docs/DATA_NOTES.md             data traps found and how they were resolved
```

## Status

Phase 0 (data contract and ingestion) delivered. Still open in phase 0: power
calculation, per-series data quality screen, Ken French cross-sectional panels.
Phases 1-5 in `docs/CHARTER.html`.
