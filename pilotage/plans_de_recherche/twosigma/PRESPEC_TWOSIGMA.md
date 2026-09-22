# PRESPEC — Regime as a contextual selector across a signal library

**Status: DRAFT 2026-09-22.** Not locked. Written before any return on this angle was
computed. Locking requires Guillaume's sign-off and a commit; from that commit on, any
change to the universe, the library, the cost schedule, the kill criteria or the
hypotheses invalidates this document and must be logged as an amendment in
`docs/PROTOCOL_FREEZE.md` together with whatever it produced before the change.

**Programme:** `regime-lab`, post-study extensions.
**Firm template:** Two Sigma — unsupervised clustering over a context space, used to
overweight signals that historically perform in that context and to shut off those
exposed to it negatively.
**Parent documents:** `docs/CHARTER.html` (frozen), `docs/PROTOCOL_FREEZE.md`,
`docs/EXTENSIONS.md`, `docs/PRESPEC_TREND_VEHICLE.md` (the closed study).

---

## 0. What this study is, in one sentence

Six devices have asked a two-state, volatility-ordered classifier with thirteen
transitions in twenty-five years to size or to time a single trend book; this study
asks a volatility-orthogonal context partition with roughly seven transitions a year to
**arbitrate among ten signals whose regime profiles differ**, which is a question that
does not exist when there is only one signal — and there was only one.

---

## 1. Prior measurements, disclosed

Every figure below was computed before this document was written. None of them is a
return, a Sharpe or a mean. They are counts, dates, position-space correlations,
turnover, and bootstrap standard errors taken from **deliberately demeaned** series.
Scripts are in the companion directory and rerun as-is.

**P1. The 46-instrument universe cannot host this study.** Eleven candidate signals
built on `data/cache/trend_universe_m1.parquet`: mean |pairwise position correlation|
0.336, **effective rank 3.86**, first eigenvalue 4.82 of 11, six principal components
for 90% of the trace. `TSMOM_1M`/`XSREV_1M` correlate −0.990; `TSMOM_12_1`/`XSMOM_12_1`
+0.930; `TSMOM_3M`/`BRKOUT_100` +0.840. This is one trend factor at several horizons.

**P2. The 49-industry panel can.** Twelve candidate signals on
`data/raw/panels/industry_49.parquet`: mean |pairwise position correlation| **0.101**,
signed mean **+0.013**, **effective rank 8.37**, eight principal components for 90% of
the trace. `IND_ACCEL` correlates −0.912 with `IND_MOM_12_1` and is dropped as a
construction redundancy before any contact.

**P3. The context clock.** K-means, K=4, on a 20-feature context space holding no
`vol_*` feature: **223 transitions over 34.52 years = 6.46/year**, 224 episodes. On the
same space orthogonalised against log realised volatility: **287 transitions =
8.31/year**, 288 episodes, η²(realised vol) = **0.048** against 0.421 unorthogonalised.
Reference: `A' sparse jump` gives 13 transitions in 6,377 sessions = **0.532/year**.
Gaussian mixtures on the same space run far slower (0.90 to 2.43/year) and are not the
declared object.

**P4. Power under the declared pairing.** `regime_lab/analysis/power.py`, stationary
block bootstrap, both legs demeaned so that `observed` reads `+0.0000` by construction.
Hard switching one signal at a time: MDE **0.474**. Weight tilt at standard deviation
0.50: MDE **0.221 to 0.294 across eight draws, median 0.241**. At the family-wise
corrected α = 0.05/6: **0.338**. At α = 0.05/21: 0.376.

**P5. Cost geometry.** Equal-weight blend turnover 21.6×/year = 1.08%/yr at 5 bp RT.
Selector increment at tilt 0.50: **+2.6×/year = +0.129%/yr = +0.013 Sharpe** at a 10%
volatility target. `IND_REV_1W` runs 156.7×/year = 7.83%/yr at 5 bp and is excluded
before contact.

**P6. A matched placebo exists, at the third attempt.** Episode-pair permutation with
swap repair reproduces the real clock exactly (223 transitions in every one of 300
draws, standard deviation 0.0) and the occupancy exactly (maximum absolute deviation
0.0000). The two rejected constructions are recorded in §8.

---

## 2. Admission test

The programme's six refutations share four narrownesses. This study differs on all
four, and the differences are measured, not asserted.

| Axis | The six | This study | Measured gap |
|---|---|---|---|
| 3 — use | sizing, or an ON/OFF switch | **selection among signals** | undefined with one signal |
| 4 — conditioned object | one book, TSMOM 12-1 on 46 instruments | **10-signal library**, 49 legs | effective rank 8.37 vs 1 |
| 1 — latent variable | states ordered by training volatility | context partition orthogonal to realised volatility | η²(rv) **0.048** |
| 2 — time constant | 0.532 transitions/year | **6.46 to 8.31**/year | **×12 to ×16** |

Admission is granted on axis 3, which is the substantive one. Axes 1, 2 and 4 are
necessary conditions for axis 3 to be posable at all.

**Admission is granted on the 49-industry panel only.** On the 46-instrument universe
the library has effective rank 3.86 and the approach is declared **not transposable**.
That refusal is part of this pre-registration, not an escape from it.

---

## 3. Data

| File | Rows | Range | Frequency |
|---|---|---|---|
| `data/raw/panels/industry_49.parquet` | 451,388 (9,212 × 49) | 1990-01-02 → 2026-07-31 | daily |
| `data/raw/panels/factors_5.parquet` | 55,272 (9,212 × 6, incl. `ff_rf`) | 1990-01-02 → 2026-07-31 | daily |
| `data/raw/panels/size_bm_25.parquet` | 230,300 (9,212 × 25) | 1990-01-02 → 2026-07-31 | daily |
| `data/cache/features.parquet` | 9,572 × 50 | 1990-01-01 → 2026-09-08 | daily |

Quality, verified rather than assumed: **zero −99.99 sentinels, zero −999 sentinels,
100.00% of sessions fully populated across all 49 industries.** `ff_rf` has 9,212
observations and no NaN; excess returns are taken against it.

**Inferential sample.** The binding warm-up is the 1,260-session long-term reversal.
Where all signals exist simultaneously: **7,946 sessions, 1995-01-04 → 2026-07-31 =
31.57 years**, against 23.2 for the programme's locked sample.

**Declared PIT weakness.** `available_at == period` on **100.0%** of panel rows;
`kenfrench.py` documents and argues that choice for context features. For the portfolio
returns used here as signals the argument is weaker: the return is computable on the
evening of *t*, but the file is downloadable only weeks later. The T-1 rule covers
knowledge, not file availability. This is the single place where this design is more
optimistic than a live desk, and it is written here rather than discovered later.

**Not on disk and not sought.** No futures term structure, no options data: the carry
and volatility-carry families cannot be built. Four Ken French daily files
(`F-F_Momentum_Factor`, `F-F_ST_Reversal_Factor`, `F-F_LT_Reversal_Factor`,
`49_Industry_Portfolios`) answered HTTP 200 on 2026-09-22, verified by `curl -sIL`;
**this design uses none of them.**

---

## 4. Constructions, fixed here

**The library — 10 signals**, cross-sectional z-scores over 49 industries, clipped at
±3, gross-normalised to 1, lagged one session:
`IND_MOM_12_1`, `IND_MOM_6_1`, `IND_REV_1M`, `IND_LTREV`, `IND_LOWVOL`, `IND_VOLMOM`,
`IND_SKEW`, `IND_LOWBETA`, `IND_LOWCORR`, `IND_SEASON`.
Excluded before contact: `IND_ACCEL` (−0.912 against `IND_MOM_12_1`), `IND_REV_1W`
(156.7×/yr turnover).

**The context space — 20 features, no `vol_*` column**, orthogonalised session by
session against log 21-day realised volatility of `eq_us_large` on training data only:
`fin_nfci`, `fin_nfci_chg13w`, `cre_baa`, `cre_baa_chg63`, `cre_quality`,
`cre_quality_chg63`, `rat_slope_10y2y`, `rat_slope_10y2y_chg63`,
`rat_slope_10y3m_chg63`, `rat_cash_chg12m`, `xs_dispersion_ind`, `xs_dispersion_szbm`,
`xs_avg_corr`, `xs_absorption`, `xs_absorption_chg`, `xs_breadth_63`, `asy_vr_5`,
`asy_vr_20`, `asy_hurst`, `mom_breadth_200d`.

**The partition.** K-means, **K=4**, `n_init=20`, `random_state=0`, refitted on each
training window, labels assigned out of sample by nearest centroid. K ∈ {3,5,6} are
declared sensitivities and can never found a PASS.

**The selector.** A **tilt**, never a switch: weight *i* in state *k* is
`(1/n)(1 + d·m_ik)` renormalised and floored at zero, with **d = 0.50** and `m_ik`
estimated on training folds only. Hard switching is not tested: it costs twice the
resolution (MDE 0.474 against 0.241) and is excluded on power grounds, declared here.

**The control.** The fixed equal-weight blend of the same ten signals.

**Protocol inherited, non-negotiable.** Signal at T-1, trade at T · excess of `ff_rf` ·
HAC lag 6 · walk-forward with **5 test folds of 4.0 years**, expanding training, first
window 11.6 years · daily volatility targeting at 10% · stationary block bootstrap
(Politis–Romano), mean blocks 21 / 63 / 126, never iid · N = 49 legs.

---

## 5. Cost schedule, priced before testing

US equities: **5 bp round trip realistic, 10 bp conservative, 20 bp stress.**
Charged on `|Δw|`.

Measured: the equal-weight blend pays 1.08%/yr at 5 bp. The selector's **increment** is
+2.6×/yr = +0.129%/yr at 5 bp = **+0.013 Sharpe** at the 10% target.

**Cost kill rule, written before any measurement.** The overlay's own cost reaches the
MDE of 0.241 only at `2.6 × bps / 10,000 / 0.10 ≥ 0.241`, i.e. **bps ≥ 92.7** — ten
times the equity schedule. Cost is therefore **not** the binding constraint; power is.
Accordingly: **if the fitted map's incremental turnover exceeds 12×/year**, at which
the 20 bp stress schedule consumes 0.24 Sharpe — the whole MDE — **the overlay is
declared dead on cost whatever its gross gap.**

---

## 6. The escalation tree

Declared in full before first measurement. A tree declared in advance is not p-hacking;
an undeclared one is.

### Level A — the first-moment channel (the literal transposition)

State-conditional tilt on expected signal performance, fitted on training folds,
applied out of sample.

**Falsification — both locks must hold to pass:**
- **A-1.** Paired out-of-sample Sharpe difference against the equal-weight control,
  net, in excess, HAC lag 6, **< 0.338** (MDE at α = 0.05/6) ⇒ FAIL. A gap below the
  MDE is **UNDERPOWERED and never a PASS**.
- **A-2.** Rank-transfer statistic — Spearman between the within-state signal ranking
  estimated on training folds and the realised ranking on the test fold, pooled over
  5 folds × 4 states = 20 cells — **below the 99th percentile** of the matched placebo
  ⇒ FAIL.

**Prior: P(pass) ≈ 12%.** Low and stated in advance: the programme has measured that
the classifier carries variance and not mean (+0.030 R² point on forward returns,
t 0.27), and A conditions a mean. A is run regardless, because it is the documented
method and skipping it would assume the answer.

### Level B — the second-moment channel

**A, if it fails, fails on the first moment. B never touches it.** The map no longer
estimates which signal earns more in state *k*; it estimates the **covariance matrix
across signals, state by state**, which feeds the blend's construction (risk parity on
the state-conditional covariance). This is the channel where the classifier has
**+3.93 points of incremental R² on forward volatility, t −3.40**, established against
a volatility quantile.

**Pre-computed design constraint.** With signed mean pairwise correlation +0.013 over
11 signals, the share of blend variance running through the correlation matrix is
`11×10×0.013 / (11 + 1.43)` = **11.5%**; the other **88.5% is diagonal**. B therefore
conditions per-signal volatility by state first, correlations second. Writing this now
prevents reading a null correlation result as a null variance result.

**Falsification — both locks:**
- **B-1.** Out-of-sample covariance-forecast criterion: the state-conditional
  covariance must beat a single pooled covariance on a realised-dispersion loss by more
  than the **99th percentile** of the matched placebo ⇒ else FAIL.
- **B-2.** Paired Sharpe difference against the pooled-covariance blend **< 0.338** ⇒
  UNDERPOWERED, never a PASS.

**Prior: P(pass) ≈ 25%.** Higher than A because the channel has measured content;
capped because the library is already nearly orthogonal.

### Level C — the cost channel, whose verdict is partly written already

**If B falls, both moments of the return are spent.** What remains is the ledger:
condition the **rebalancing cadence** on the context state.

**C is pre-declared undecidable at the portfolio level.** The blend's entire cost is
1.08%/yr at 5 bp = **0.108 Sharpe** at a 10% target. Removing *all* of it yields at
most 0.108, **below the uncorrected MDE of 0.241 and far below the corrected 0.338**.
No Sharpe result is reachable here, and discovering that after six days of work would
be waste.

**C is therefore a turnover test, not a Sharpe test.**
- **C-1.** Paired difference in annual turnover against the control, with its own MDE.
- **C-2.** Its Sharpe translation, `Δturnover × bps / 10,000 / 0.10`, reported **as a
  bound and never as a PASS**.

**Prior: P(pass on C-1) ≈ 40%. P(a decidable Sharpe gain) ≈ 0%**, by the arithmetic
above.

### Closure statement, if all three fall

> A context partition with a 6–8 per year clock, orthogonal to volatility, does not
> transfer out of sample as a selector over a 10-signal US equity library on either the
> first or the second moment, over 31.6 years, at a resolution of 0.338 Sharpe. The
> only remaining channel is cost, and the book's entire cost — 0.108 Sharpe — is too
> small to be decidable on this sample.

That is the sentence this study is allowed to write. It is not "regimes do not
monetise."

---

## 7. Multiple testing, computed in advance over the whole tree

**21 declared evaluations:**

| Group | Count |
|---|---|
| Primary: 3 levels × 2 locks | **6** |
| Sensitivity: K ∈ {3,5,6} × 3 levels | 9 |
| Sensitivity: tilt d ∈ {0.25, 1.00} × 3 levels | 6 |
| **Total logged to `trials.parquet`** | **21** |

- **Holm–Bonferroni over the 6 primaries**, family-wise α = 0.05, strictest threshold
  **0.00833**. Corresponding **MDE 0.338**, against 0.272 uncorrected.
- The 15 sensitivities are reported and **can never found a PASS**. They still enter
  `trials.parquet` and therefore the deflated Sharpe ratio.
- Covering all 21 inferentially would put the MDE at **0.376**; the figure is stated so
  that restricting inference to 6 is visible and contestable.
- **Placebo threshold is the 99th percentile, not the 95th.** Motive, from the
  programme's own measurement: an index methodology tested against 160,000 random
  combinations places the official version at the **98th** percentile. The 95th is
  exactly where a searched map lands. The extra percentile costs power and buys
  credibility.

---

## 8. The matched placebo

The null must match the real partition on **both** the clock and the occupancy, because
T3 showed that an apparent gain can travel entirely through a dimension left
unmatched.

| Attempt | Transitions/yr (real 6.46) | Max occupancy deviation | Verdict |
|---|---|---|---|
| 1 — permute `(length, state)` episode pairs | 4.79 — loses 29% of the clock to merged joins | 0.0000 | rejected |
| 2 — permute lengths and states independently | 6.77 | **0.3043** — occupancy destroyed | rejected |
| 3 — **permute pairs, repair joins by swapping pairs** | **6.46, exact** | **0.0000, exact** | **adopted** |

The adopted construction keeps `(length, state)` pairs intact, so occupancy is exact by
construction, and repairs same-state joins by swapping whole pairs, which cannot alter
the multiset. 300 of 300 draws accepted, 223 transitions in each, standard deviation
0.0.

**Three controls, not one.**
1. **Clock placebo** — 1,000 matched partitions with no content, supplying the null for
   A-2 and B-1.
2. **Equal-weight control** — the fixed blend, which is the real adversary.
3. **Beta control, the T3 lesson** — the selector's realised beta against `ff_mkt-rf`
   is reported as a percentile of the placebo distribution. A selector that beats the
   control while sitting at the 95th percentile of beta has selected nothing; it has
   bought exposure, at 0.2785 per unit by the barrier study's measurement.

---

## 9. Effort and honest priors

| Item | Days |
|---|---|
| Library build, exclusions, unit tests | 2.0 |
| Context module, walk-forward refit, matched placebo | 2.0 |
| Level A | 1.5 |
| Level B | 2.0 |
| Level C | 1.0 |
| Write-up, prespec lock, protocol-freeze amendment | 1.5 |
| **Total** | **10.0** |

**P(the tree yields at least one inferential PASS) ≈ 30%**, being `1 − 0.88 × 0.75`
rounded down; level C contributes nothing by construction.
**P(the tree yields a writable result) = 1**, by §10.

---

## 10. What is learned if the whole tree falls

1. The programme's signal-library dimension, measured for the first time: **8.37
   effective of 12** on the equity cross-section against **3.86 of 11** on the
   46-instrument universe. This retroactively explains part of the six failures — the
   conditioned object had roughly four effective dimensions, three of them the same
   trend factor at different horizons.
2. A narrow, dated, quantified sentence in place of a generalisation.
3. A reusable placebo (clock-exact and occupancy-exact), with its two wrong versions
   documented so they are not rebuilt.
4. The cost geometry of any selection overlay in this programme, closed by
   measurement: +0.013 Sharpe at 5 bp, dead only above 93 bp round trip. A future
   failure of a selection overlay can no longer be blamed on fees.
5. The negative transposability result on the 46-instrument universe, which saves the
   next attempt.

---

## 11. Blinding, and what has not been done

No strategy was built and no backtest return was read. Correlations are correlations of
weight vectors; standard errors come from series demeaned before entering
`power.py`, whose `observed` field is verified at `+0.0000`; turnover is `|Δw|`. What
distinguishes one signal from another across states — the only quantity that would
decide the tree — has not been measured and must not be until this document is locked.

**Open and unverified, stated plainly:** whether any signal in this library has a
regime profile different enough to be worth selecting on is **unknown**. This
pre-registration establishes that the question is now askable — a library of effective
rank 8.37 over 31.57 years, a clock at 6.46 to 8.31 per year, a matched placebo, and a
resolution of 0.338 — and nothing more.
