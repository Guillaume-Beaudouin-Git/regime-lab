# Final results

Out of sample April 2002 – September 2026, 6,377 sessions, returns in excess of
the three-month bill, states ordered by training volatility, costs charged at
two basis points round trip.

## Layer 1 — the classification is reliable

| family | mean run | vs chance | hindsight changes | NBER bal. acc | kappa |
| --- | --- | --- | --- | --- | --- |
| A jump model | 456 d | 97× | 0.9% | 93.8% | 0.49 |
| A′ sparse jump | 456 d | 83× | 0.6% | **95.1%** | **0.56** |
| B filtered HMM | 128 d | 54× | 0.7% | 84.7% | 0.24 |
| C gradient boosting | 18 d | 9× | 0.0% | 75.0% | 0.12 |
| C′ HAR-RV | 14 d | 7× | 0.0% | 77.4% | 0.16 |

Chance-corrected agreement between the five methods runs 0.28 to 0.83 — all
positive. Before the ordering was fixed the two camps sat at −0.39, which was
the inversion and nothing else.

Detection latency is nil. Seeing the rest of the block would change under one
percent of labels; the filtered and offline sequences agree at kappa 0.97-1.00.
Regimes here are recognised as they happen, not afterwards.

## Layer 2 — it separates variance, and only variance

| family | mean spread | MDE | detected | vol spread | MDE | detected |
| --- | --- | --- | --- | --- | --- | --- |
| A jump model | −1.02% | 19.5% | no | −6.51% | 4.5% | **yes** |
| A′ sparse jump | −0.46% | 21.9% | no | −6.59% | 5.0% | **yes** |
| B filtered HMM | −2.58% | 11.3% | no | −5.42% | 2.9% | **yes** |
| C gradient boosting | −0.84% | 8.0% | no | −4.54% | 2.2% | **yes** |
| C′ HAR-RV | −3.12% | 9.2% | no | −4.99% | 2.4% | **yes** |

**Five out of five detect the variance separation. Zero out of five detect the
mean separation.** Same data, same states, same method — the difference is that
a volatility spread carries a detection threshold of two to five percent while a
mean spread carries one of eight to twenty-two.

That asymmetry is the answer to the question the project set out to ask. The
timing question is unanswerable on 25 years of daily data; the variance question
is answered decisively.

Marginal information over a plain volatility quantile, on forward returns:
0.007% to 0.030% of incremental R². Once you know how volatile the market has
been, the state tells you nothing more about direction.

## Layer 3 — timing rule against sizing rule

| family | rule | Sharpe | net | vol | maxDD | CEQ |
| --- | --- | --- | --- | --- | --- | --- |
| base 60/40 | — | 0.47 | 0.47 | 10.5% | −35.6% | 2.16% |
| A jump | on/off | 0.54 | 0.54 | 7.5% | −22.3% | 2.66% |
| A jump | sized | 0.50 | 0.50 | 10.6% | −29.3% | 2.45% |
| A′ sparse | on/off | 0.55 | 0.55 | 7.9% | −22.3% | 2.80% |
| B HMM | sized | 0.50 | 0.49 | 10.6% | −30.4% | 2.42% |
| C gradient boosting | sized | 0.52 | 0.50 | 10.0% | −32.5% | 2.49% |
| C′ HAR-RV | on/off | 0.57 | 0.50 | 5.1% | −13.4% | 1.87% |

Every improvement over the base book sits well inside the 0.33-0.55 detection
threshold measured in T2. **At the portfolio layer nothing here is decidable**,
which was true before the ordering was fixed and is still true after.

Two honesty notes that belong beside this table. The jump model's move from 0.16
to 0.54 comes from a correction made *after* seeing that its labels were
inverted; the correction is principled — ranking states by volatility never
looks at returns — but the resulting figure is post-amendment and is logged as
such. And the sizing rule targets the base book's own volatility, so it does not
reduce risk: it is a like-for-like test of whether the state improves a
constant-volatility book, and it mostly does not.

## What the project establishes

1. **Regime classification works and is measurable.** 95% balanced accuracy
   against NBER dates, kappa 0.56, persistence 97× chance, five independent
   methods in agreement, recognised in real time.
2. **What it carries is variance, not mean.** Detected with power on every
   family for variance; undetectable on every family for mean.
3. **It is therefore a sizing signal, not a timing signal** — and an on/off rule
   is a timing rule.
4. **At the portfolio layer the question is underpowered**, by a factor of three
   to seven in sample length. That was the T2 finding and it survives.
5. The portfolio result was never evidence that regimes fail. It was a timing
   rule applied to a variance signal, on top of an inverted label.
