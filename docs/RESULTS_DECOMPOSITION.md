# Where the value of a regime overlay comes from

The charter names four attributions. `scripts/run_decomposition.py` computed them,
ran cleanly, and was published nowhere; this is that result, recovered rather than
produced. 6,377 out-of-sample days, 2002-04 to 2026-09, in excess of cash, certainty
equivalent at γ = 5, costs at 2 bp round trip declared in advance.

```bash
.venv/bin/python scripts/run_decomposition.py
```

The ladder climbs from the raw book to the traded overlay, one effect at a time, so
the gap between two rungs *is* that effect's contribution.

| family | base | exposure | vol timing | hindsight | latency | costs |
|---|---|---|---|---|---|---|
| A jump | 2.27% | +0.00% | +0.64% | −0.07% | −0.17% | −0.01% |
| A′ sparse jump | 2.27% | +0.01% | +0.75% | +0.08% | −0.29% | −0.01% |
| B filtered HMM | 2.27% | −0.17% | +0.31% | +0.05% | −0.59% | −0.04% |
| C gradient boost | 2.27% | −0.49% | +0.29% | −0.41% | +0.00% | −0.27% |
| C′ HAR-RV | 2.27% | −0.27% | +0.21% | +0.16% | +0.00% | −0.35% |

## What the ladder says

**The two rungs that need no regime model at all** are where the value is: cutting
exposure costs −0.18% on average, and timing volatility pays **+0.44%**.

**The two rungs that are the regime model's own contribution** are negative. Regime
information is worth **−0.04% with hindsight** — that is, knowing the state *in advance*
does not help — and detection lag takes a further −0.21%, for **−0.25% net of latency**.
Trading costs then take −0.14%.

This is the cleanest statement of the programme's central finding that exists, and it
was sitting uncomputed in a script. The volatility-timing rung needs no states, no
estimation and no parameter beyond a target. The regime rung is negative before latency
is charged and stays negative after.

## Two details worth keeping

**Hindsight is nearly free, which is the surprise.** Per family, the Sharpe cost of
replacing the hindsight state with the filtered one is 0.02 for A jump, 0.04 for A′,
0.10 for B, and exactly 0.00 for C and C′. A signal whose perfect-foresight version is
worth almost the same as its real-time version is not a signal being degraded by
latency; it is a signal with little to degrade.

**Rung 2 is invisible to the Sharpe ratio**, because a constant scaling cannot change
it. That is not a quirk of this table — it is the arithmetic that made the first version
of the study's decisive test compare a strategy with itself, and it is why the
certainty equivalent is the metric here.
