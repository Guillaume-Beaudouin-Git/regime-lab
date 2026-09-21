# The trend book as the subject: it does not clear a research gate

The study locked in `docs/PRESPEC_TREND_VEHICLE.md` on 2026-09-21, evaluated against
its own six kill criteria, then attacked from four independent angles before this
was written.

```bash
.venv/bin/python scripts/apply_m1_realignment.py   # M1, the calendar repair
.venv/bin/python scripts/run_m3_construction.py    # M3, construction integrity
.venv/bin/python scripts/run_m3_evaluation.py      # K1-K6, the placebos, the gates
```

**Verdict: KILL.** The book dies on K1 under every admissible construction. The
research gate is missed on the level and on the maximum drawdown. H-a is not
established. H-b is not established, and **the programme's regime line closes**.

---

## 1. What the three repairs did

| | before | after |
|---|---|---|
| FX calendar (M1) | same-session correlation 0.057 / 0.107 / 0.064 against the matching CME future | **0.904 / 0.912 / 0.896** |
| portfolio volatility (M3) | floats 1.43% to 12.29%, realising 4.13% | **10.90% against a 10% target** |
| leverage cap (M3) | never tested | inert, binds on 0.00% of sessions |

The pound now falls on 24 June 2016 rather than on the 27th. Two of the ten spot FX
series, CAD and MXN, were measured as already aligned and deliberately left alone;
shifting them would have created the defect M1 exists to remove.

---

## 2. The six criteria

| | verdict | the number |
|---|---|---|
| **K1** level | **KILL** | net excess Sharpe **+0.359**, against a 0.50 line |
| **K2** walk-forward | PASS, at its own frontier | 4 folds positive of 5, one sign flip |
| **K3** sizing | **UNDERPOWERED**, H-a not established | gap **+0.068** against an MDE of 0.246 |
| **K4** regime | **UNDERPOWERED**, H-b not established | gap **+0.014**, against its own pairing's resolution of 0.043 |
| **K5** deflated Sharpe | **PASS not established** | excess runs −0.159 to +0.286 across defensible pools |
| **K6** cap integrity | PASS under the declared reading | binds 0.00% of sessions |

**K1 decides, and it is not close on the gate.** The committed construction reads
+0.359 net of costs and in excess of cash. The construction that §3's own sample
definition implies — the lookback taken before the sample rather than inside it —
reads +0.461 on 6,039 sessions. Both fall below the 0.50 kill line and neither
approaches the 0.70 gate. The headline was **not** moved to the higher figure after
seeing that it reads higher; that is the adjustment the protocol forbids, and it is
logged as a question for the next amendment rather than resolved here.

**K2 passes on a rule weaker than it reads.** A sign flip is defined against the
full-sample sign, which is positive, so "≥ 3 positive and < 2 flips" collapses to
"≥ 4 positive of 5". The test's size under a symmetric null is 0.1875, not 0.5000,
and the result sits exactly on its frontier. The folds also cover only 2014-2026,
because the tool trains on ten years first; the first decade is judged by K1 alone.

**K3 shrinks twice under scrutiny.** The published gap was +0.131. Funding the three
bond ETFs that had been missed takes it to +0.120. Separating the part that is
merely un-pinning gross exposure from volatility targeting proper takes it to
**+0.068** — 0.28× the locked MDE. Volatility targeting is not demonstrated as the
universal base layer practice claims it to be, net of cost, on this universe.

**K4 closes the regime line.** The gap is +0.014, which is 0.3× the resolution of
its own pairing. That is not a test too coarse to see the effect; it is a test fine
enough to say the effect is smaller than anything it can see. It fails under the
*lagged* states, which favour H-b, and under both sensitivities. **Sixth independent
refutation.**

---

## 3. What four adversarial passes found, including one of mine

Nothing rescued the book. The construction is causally clean to the bit — the only
angle on which nothing fell. Three published figures did not survive.

**The headline was not a net excess Sharpe, and the defect was mine.**
`fetch_trend_universe.py` downloads with `auto_adjust=True`, so every ETF column is a
*total return* series carrying its coupon. `NO_FUTURES_VEHICLE` was built for §6 by a
property of the market — an IEF exposure trades as ZN, so it pays the futures rate —
and then reused for §7, where the question is what the series already contains. IEF,
SHY and TLT fell through `cost_class`'s default branch and were never funded, while
six identical bond ETFs were. SHY, a 1-3 year Treasury ETF, drifts at 1.92% a year
against a mean cash rate of 1.77%: the series banks the cash and the study counted it
as return. Fixed by separating `FUNDED_ON_CASH` from `NO_FUTURES_VEHICLE`, with a
test that asserts the two answer different questions. It cost **0.041 of Sharpe**,
and it closed the study's own leading reservation — the §3-faithful construction had
missed the 0.50 line by 0.005 and now misses it by 0.039 in the other direction.

**"Both placebos fail to distinguish anything" is false.** P1's 88.5th percentile is
a statistic of the first 63 sessions: `unscaled_weights` returns zeros before the
book is live, so σ is estimated on zeros and the multiplier explodes — the 43.8×
maximum gross is that, and it falls to 7.96 once those sessions are dropped. On the
purged sample M1 reads **99.8%**, above the 95th percentile. And P3 cannot fail by
construction: Sharpe is scale-invariant to 1.1e-16, so its two lines are K3 and K4
recomputed.

**K5's PASS rests on an undeclared estimator.** Collapsing `trials.parquet` to one
Sharpe per configuration by taking the mean compresses the dispersion 4.9-fold, and
that dispersion is the quantity deciding K5. Across four defensible collapses the
deflated excess runs from −0.159 to +0.286 — it crosses zero. The pre-registration
fixes the trial *count* and says nothing about the variance estimator.

---

## 4. The gates

| condition | threshold | measured | |
|---|---|---|---|
| net excess Sharpe | > 0.70 | +0.359 | **fails** |
| maximum drawdown | > −25% | −29.6% | **fails** |
| folds positive | ≥ 3 of 5 | 4 of 5 | satisfied, at the frontier |
| DSR | > 0 | −0.159 to +0.286 | not established |

**RESEARCH_PASS: not cleared.** Stated at the right strength: the gate is a decision
rule on the measured quantity and it fails twice, but the sample cannot exclude a
true Sharpe above 0.70 — the stationary block bootstrap gives [−0.015, +0.779] at
block 126, and 0.70 sits inside it at all three block lengths. What is established is
that the gate is not cleared, not that the book is worthless.

---

## 5. What this means for the programme

**The regime line closes.** H-b was the sixth and last test, its prior was low and
written before the measurement, and it fails on a gap 0.3× its own resolution. Six
independent devices, one known mechanism — 13 state transitions in 6,377 sessions —
and a classification that carries variance rather than mean.

One correction travels with that closure: the mechanism the conclusion invoked is not
the one the evaluated arm shows. The 163.9% against 151.6% describes the *unscaled*
book, and at the level of the arm actually measured the regime covariate **raises**
book volatility, 10.71% against 10.57%. The line closes on a measured void, not on
the mechanism first named.

**And the benchmark the programme hoped for is not there either.** Correctly costed
and in excess, the book reads between +0.36 and +0.46 for a −29.6% drawdown and 58.9×
annual turnover. Eight instruments of 46 carry 69% of its cost, and they are the ones
whose §7 classification was just found wrong. What the study does deliver is the
denominator both repositories have lacked since the beginning: an audited reference
book, in excess, correctly costed, over 23.2 years.

A clean death on the criteria was declared in advance to be a valid and publishable
outcome. This is one.
