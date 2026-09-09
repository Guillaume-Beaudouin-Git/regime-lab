# T2 — the portfolio comparison

**Regenerated 2026-09-10.** The first version of this file was written before the
state-ordering correction and never rewritten. Its figures for families A and A′
described inverted labels — a negative certainty equivalent, a detectable effect
of 0.545, "188 years" — and were cited as current in `RESULTS_FINAL.md` for two
days. An independent audit found it by running the script. Everything below is
the output of `scripts/run_t2.py` on the current cache.

Out of sample April 2002 – September 2026, 6,377 sessions, returns in excess of
the three-month bill.

## No family beats a moving volatility target

The benchmark is the same 60/40 book sized to an ex-ante volatility target, with
the target set at each refit to the volatility the overlay realised on that
refit's training window. It uses no regimes at all.

| family | Sharpe overlay | Sharpe benchmark | ΔSharpe | CEQ overlay | CEQ benchmark | vol gap |
| --- | --- | --- | --- | --- | --- | --- |
| A jump model | 0.54 | 0.58 | −0.03 | 2.68% | 2.91% | 1.6% |
| A′ sparse jump | 0.55 | 0.59 | −0.03 | 2.82% | 3.04% | 2.4% |
| B filtered HMM | 0.47 | 0.57 | −0.10 | 1.87% | 2.42% | 1.1% |
| C gradient boosting | 0.48 | 0.55 | −0.07 | 1.67% | 2.08% | 5.7% |
| C′ HAR-RV | 0.59 | 0.56 | **+0.04** | 2.37% | 2.21% | 2.4% |

Every volatility gap is under 10%, so the comparisons are like-for-like.

Every difference sits inside its own detection threshold. The benchmark wins on
four families out of five and loses on one by a margin the sample cannot resolve.

## Costs remove the only positive margin

Cost levels were written into `strategies/costs.py` before any result was read.

| family | turnover | gross | 2 bp | 5 bp | 10 bp |
| --- | --- | --- | --- | --- | --- |
| A jump | 0.0020 | −0.03 | −0.04 | −0.04 | −0.04 |
| A′ sparse | 0.0020 | −0.03 | −0.03 | −0.04 | −0.04 |
| B HMM | 0.0077 | −0.10 | −0.10 | −0.11 | −0.13 |
| C gradient boosting | 0.0540 | −0.07 | −0.13 | −0.22 | −0.37 |
| C′ HAR-RV | 0.0707 | +0.04 | **−0.03** | −0.14 | −0.31 |

HAR-RV's edge dies at the cheapest assumption, and it had the highest turnover
in the table. The benchmark is left gross, which flatters the overlays.

## Stop rule 3 — and where it does not fire

The charter stops the study if the minimum detectable effect exceeds 0.30 of
Sharpe. Phase 0 measured 0.174 and flagged it as a lower bound, since it
compared two versions of one book correlated at 0.91.

| family | correlation with benchmark | MDE | stop rule | years for an MDE of 0.20 |
| --- | --- | --- | --- | --- |
| A jump model | 0.87 | 0.302 | fires | 58 |
| **A′ sparse jump** | 0.88 | **0.271** | **does not fire** | 46 |
| B filtered HMM | 0.78 | 0.328 | fires | 68 |
| C gradient boosting | 0.66 | 0.399 | fires | 101 |
| C′ HAR-RV | 0.73 | 0.384 | fires | 93 |

**The rule does not fire for A′ sparse jump**, which is the study's strongest
classifier. On that family the sample can resolve a Sharpe difference of 0.27,
and the measured difference is −0.03: the benchmark is not beaten, and this time
that is a measurement rather than an absence of one.

Settling the question at an effect size of 0.20 needs **46 to 101 years** against
the 25 available — a shortfall of a factor of **two to four**, not the "three to
seven" the superseded version claimed.

A caution on precision: the detectable effect is itself estimated from a
bootstrap. Recomputed on the first half and first quarter of the sample it
scales roughly as 1/√T but with 20–30% dispersion, so these figures are good to
about a factor of 1.3. "Several decades" is the honest reading; "188 years" was
false precision on top of a stale number.

## What this file no longer says

The superseded version reported that both jump models had a **negative**
certainty equivalent, that an investor with γ = 5 "would rather hold nothing
than hold them", and that the stopping rule fired on every family. All three
described a wrapper that was long the weakest state by construction. The
corrected certainty equivalents are +2.68% and +2.82%.
