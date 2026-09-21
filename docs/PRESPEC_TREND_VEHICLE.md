# PRESPEC — The trend book as the subject: vehicle, sizing, and regime as one covariate

**Status: LOCKED 2026-09-21**, by this commit, before the repaired book produced a
single return. Any edit to the universe, the cost model, the kill
criteria or the hypotheses after data contact invalidates it and must be logged
as an amendment with what it produced before the change.

**Programme:** `regime-lab`, post-study extensions.
**Author:** Guillaume Beaudouin.
**Lock date:** 2026-09-21.
**Parent documents:** `docs/CHARTER.html` (frozen, never edited),
`docs/PROTOCOL_FREEZE.md` (amendment log), `docs/EXTENSIONS.md`.

---

## 0. What this study is, in one sentence

Every measurement this programme has published treats a 46-instrument trend book
as the *vehicle* for testing a regime overlay; this study makes the book the
*subject*, specifies a vehicle its own cost schedule can support, applies the
portfolio-level volatility target its own hard rules require, and asks whether
what remains clears a research gate.

---

## 1. Prior measurements, disclosed

This section exists because the choice of hypothesis below was not blind. Three
things were measured on **already-published constructions** before this
pre-registration was written. They are audits of existing artefacts, not new
trials, and they carry no deflation burden — but they informed the design and
must therefore be declared.

**D1. The published reference book is gross, unsized, and cost-fragile.**
`regime_lab/extensions/trend.py` renormalises gross exposure to 1.0 every
session, so portfolio volatility floats (rolling 63-day annualised: 0.00% to
15.14%, mean 4.54%). Published Sharpe 0.510, gross annual return 2.315%, gross
turnover 14.79× per year, breakeven 15.7 bp round trip. Charging this
programme's own cost schedule by asset class costs 1.130%/yr — **48.8% of the
gross return** — for a net Sharpe of 0.261. Mean net directional exposure is
+0.268 against a mean 3-month cash rate of 1.73%, so the excess-return
correction takes 0.520 to 0.413, or to 0.145 if the whole gross must be funded.

**D2. The published winner's advantage is vehicle-dependent.** The
parameter-free dynamic volatility target reproduces at gap +0.244, t +2.91
(published: +0.244, t +2.89). Net of costs on the common sample:

| cost schedule | book net SR | vol-target net SR | gap | t |
|---|---|---|---|---|
| retail mix implied by the stored universe | 0.163 | 0.269 | +0.106 | +1.23 |
| all-futures, 1.0 bp RT | 0.410 | 0.638 | +0.228 | +2.71 |
| all-futures, 2.0 bp RT | 0.375 | 0.586 | +0.211 | +2.51 |
| futures + 5 bp FX/bond | 0.316 | 0.498 | +0.182 | +2.15 |

Sample MDE on this comparison: 0.205 as published, 0.246–0.273 as recomputed
here by stationary block bootstrap at mean blocks 21 / 63 / 126.

**D3. Two repairable data defects.** (a) The **ten** spot FX series (the tickers
ending `=X`: AUDUSD, CAD, CHF, EURUSD, GBPUSD, JPY, MXN, NOK, NZDUSD, SEK) are
stamped one session late: against the matching CME futures, corr(t, t) is 0.057 / 0.108 /
0.064 for EURUSD=X / GBPUSD=X / AUDUSD=X while corr(t+1, t) is 0.890 / 0.896 /
0.877; the Brexit session confirms it by event (6B records −8.10% on 2016-06-24,
the spot series records −1.58% that day and −7.91% on the 27th). Equity indices
are correctly aligned (^GSPC vs ES: 0.9709 same-day). Three of the ten are verified
against a future; the other seven are inferred from the same data endpoint and are
**not verified**. `DX-Y.NYB`, the eleventh currency-family instrument, is an ICE
index rather than a spot pair, quotes on a different venue, and its alignment is
**not verified either way** — it is treated as unrepaired and its sensitivity is
reported separately. (b) GC=F and SI=F correlate 1.0000 with raw
front-month and 0.984 / 0.990 with back-adjusted series, i.e. they are raw and
carry roll gaps; on CL between 2012 and 2026, roll-day returns sum to −22.4% raw
against −13.9% back-adjusted.

**Consequence for blindness, stated plainly.** The decision to specify a futures
vehicle was taken *after* seeing D2 — that the retail mix drives the only
surviving device below its own detection threshold. That decision is therefore
not out of sample, in the same way and for the same reason that
`docs/EXTENSIONS.md` declares the choice to condition on market volatility was
not. It is disclosed here so that a reader can discount it.

What has **not** been measured, and is the object of this pre-registration: any
return, Sharpe, drawdown or turnover of a book built on repaired data, or of any
portfolio-level volatility target applied to it.

---

## 2. Hypothesis

**H.** A time-series trend book on the 46 instruments of
`data/cache/trend_universe.parquet`, restricted to the sample that satisfies the
programme's N ≥ 30 rule, built on a correctly specified vehicle and sized by a
daily portfolio-level volatility target, achieves a **net excess Sharpe above
0.70** over the full sample.

**H-a (subsidiary, and the one that is publishable either way).** The
portfolio-level volatility target improves net excess Sharpe by at least the
sample's minimum detectable effect.

**H-b (subsidiary, terminal).** Regime state, entered as one covariate among
others in the risk-budget module and never as a switch, improves net excess
Sharpe by at least the MDE over the parameter-free volatility target.

H-b is the sixth and last test of the regime thesis this programme will run. Its
prior is low and stated: five independent devices have already failed, and the
mechanism is known (13 state transitions in 6,377 sessions). It is included
because the book it would condition is, for the first time, a book worth
conditioning.

---

## 3. Sample

- **Universe:** the 46 instruments already in `data/cache/trend_universe.parquet`.
  No instrument is added, removed or substituted for any reason, including
  performance. Substituting a *vehicle* for the same underlying exposure (a
  futures series for a spot or ETF series) is not a universe change and is
  covered by §4.
- **Start:** 2003-07-17, the first session on which at least 30 instruments have
  252 sessions of price history, i.e. the first session on which a 12-minus-1
  trend signal exists for an N ≥ 30 cross-section.
- **End:** 2026-09-10, the last session in the stored panel.
- **Length:** 23.2 years.
- **No data is re-downloaded.** Back-adjusted futures series, where used, come
  from a private sibling repository's back-adjusted daily futures set, validated against
  Databento at 0.975 weekly return correlation.

Declared breadth, to be used in any appeal to the fundamental law: over 2008+ on
the 46-column complete panel, mean pairwise correlation 0.098, participation
ratio 11.4, entropy-effective bets 20.7, first eigenvalue 22.5% of variance, 24
eigenvalues to reach 90%. **The breadth of this book is 11 to 21, not 46.**

---

## 4. The three modifications, and no others

The programme allows three modifications per hypothesis. These are they. Nothing
else changes relative to the published construction.

**M1 — Calendar realignment of spot FX.** The eleven FX series are shifted so
that each return sits on the session in which it occurred, as established in
D3(a). This is not a fitted parameter; it is a calendar correction verified
against an external instrument and against a dated event. The realignment is
applied to all eleven, and the report states that three are verified and eight
inferred. A sensitivity is reported with the eight unverified series left
unshifted.

**M2 — Vehicle specification.** Each exposure is represented by the instrument
that would actually be traded: index futures for equity indices, currency
futures for FX, note and bond futures for the fixed-income ETFs, back-adjusted
continuous futures for commodities. Where a back-adjusted series is not
available, the stored series is kept and the instrument is flagged in the
report; the headline result is reported both with and without flagged
instruments.

**M3 — Portfolio-level volatility target.** Gross exposure ceases to be pinned
at 1.0. The book is scaled daily to a constant target volatility by an estimator
that uses only past data, with the scaling factor lagged one session. The
estimator, its window and its leverage cap are fixed in §5 before any return is
computed.

Explicitly **not** modifications, and therefore not permitted: changing the
lookback or skip of the trend signal, adding a second signal, adding a
cross-sectional rank, adding instrument selection, adding a stop, adding
pyramiding, or changing the target volatility after seeing a result.

---

## 5. Frozen construction parameters

Fixed here, before any return is computed. Each value is inherited from existing
committed code rather than chosen, and the source is named.

| parameter | value | source |
|---|---|---|
| signal | `sign(P[t−21] / P[t−252] − 1)` | `extensions/trend.py`, unchanged |
| per-instrument risk scaling | `0.10 / σ_63d`, capped at 3.0 | `extensions/trend.py`, unchanged |
| signal lag | one session | hard rule, unchanged |
| portfolio volatility estimator | 63-day rolling standard deviation of book returns, lagged one session | the control that wins in `EXTENSIONS.md` |
| portfolio volatility target | 10% annualised | `VOL_TARGET` already in `extensions/trend.py` |
| portfolio leverage cap | 3.0 | `MAX_LEVERAGE` already in `extensions/trend.py` |
| cost charged | on the absolute change in each instrument's exposure | `strategies/costs.py`, unchanged |

A leverage cap that binds on more than 5% of sessions is a design failure, not a
result: the barrier study found exactly this failure mode, where a cap bit on
76.8% of sessions and cut 39% of the contrast under test in the direction that
favoured the thesis. The binding rate is reported in every table.

---

## 6. Cost model, declared before any result

Round-trip basis points by class, all applied to the change in exposure. Values
are the programme's own schedule; futures values are the range that sibling repository
measured per contract from tick size, notional and commission (0.57–2.11 bp per
side including roll cost).

| class | headline | conservative | stress |
|---|---|---|---|
| equity index futures | 1.0 | 2.0 | 5.0 |
| currency futures | 1.0 | 2.0 | 5.0 |
| fixed-income futures | 1.0 | 2.0 | 5.0 |
| commodity futures | 1.5 | 3.0 | 7.5 |
| any instrument kept on a non-futures vehicle | 7.5 | 15.0 | 30.0 |

All three columns are reported. The headline column is the decision column. The
sign of a result that moves between the headline and conservative columns is
reported as undecided, not as a pass.

Reference points already measured, for calibration: the published book's
breakeven is 15.7 bp round trip, and its turnover is 14.79× per year. A
volatility-targeted book turns over more, because the leverage path itself
trades; on the published device that took turnover from 15.15× to 24.71× per
year. This is expected and is not a reason to alter the construction.

---

## 7. Returns convention

**Net excess returns are the only decision metric.** Futures returns are excess
by construction, since the notional is financed in the term structure. Any
instrument kept on a cash vehicle has its funded exposure charged at the
3-month cash rate from `data/raw/macro/rate_cash_3m.parquet`, at the
`available_at` date, never the `period` date.

Gross Sharpe may be reported alongside for continuity with `EXTENSIONS.md`, and
must be labelled as such. The programme published gross figures for a year
without saying so; this is the correction.

---

## 8. Kill criteria, locked

Forward-looking. Sunk cost is not a justification. Any one of these fires and the
corresponding claim is dead.

- **K1 — level.** Net excess Sharpe ≤ 0.50 on the full sample → **KILL H**. The
  repaired book will not have beaten the unrepaired one and the vehicle thesis
  is false.
- **K2 — walk-forward.** Fewer than 3 of 5 folds positive, or a sign flip in ≥ 2
  folds → **KILL H**. Five folds, genuinely evaluated, using
  `regime_lab.models.protocol.walk_forward`; the folds are partitioned before any
  fold result is read.
- **K3 — sizing.** The portfolio volatility target does not improve net excess
  Sharpe by at least the sample MDE → **H-a is refuted**, and the finding is
  written up as such: volatility targeting is not the universal base layer that
  practice claims, at least not net of cost on this universe.
- **K4 — regime.** The regime covariate does not improve net excess Sharpe by at
  least the sample MDE over the parameter-free volatility target → **H-b is
  refuted**, the regime leaves the risk-budget module permanently, and the
  programme's regime line closes.
- **K5 — deflated Sharpe.** DSR ≤ 0 after deflating by the full trial count in
  `data/trials.parquet` plus every configuration run under this pre-registration
  → **KILL H**.
- **K6 — cap integrity.** The leverage cap binds on more than 5% of sessions →
  the construction is rebuilt, and the pre-cap result is published alongside.

A clean kill on all six is a valid and reportable outcome of this study.

---

## 9. Gates

- **RESEARCH_PASS:** net excess Sharpe > 0.70, MaxDD > −25%, ≥ 3 of 5 folds
  positive, DSR > 0.
- **PROPFIRM_PASS:** net excess Sharpe > 1.00, MaxDD > −10%, ≥ 4 of 5 folds
  positive. Reported for context only; it is not the bar for this study and is
  not expected to be met.

---

## 10. Power, computed before the model

Sample length 23.2 years. Standalone Sharpe standard error by Lo (2002),
`SE = sqrt((1 + SR²/2) / T)`:

| true Sharpe | SE | measured Sharpe needed for t > 1.96 |
|---|---|---|
| 0.50 | 0.226 | 0.44 |
| 0.70 | 0.232 | 0.45 |
| 1.00 | 0.254 | 0.50 |

Years required to resolve a true Sharpe of 0.70 against zero at 80% power and
α = 0.05: **19.9**. Available: **23.2**. The level question H is **powered**.

Paired comparison between two correlated constructions of the same book,
stationary block bootstrap on the active sample (6,569 sessions, 2001-07-05 to
2026-09-10), via `regime_lab.analysis.power.minimum_detectable_sharpe_difference`:

| mean block | SE | MDE at 80% power |
|---|---|---|
| 21 | 0.098 | **0.273** |
| 63 | 0.094 | **0.264** |
| 126 | 0.088 | **0.246** |

**The MDE for K3 and K4 is 0.246**, the most favourable of the three, and all
three are reported. Any gap below it is declared UNDERPOWERED and is not a pass —
the failure mode that `EXTENSIONS.md` caught on the Carver attenuator, whose
+0.144 sat under an MDE of 0.205.

Stated consequence: this sample can decide whether the book is good. It can
barely decide whether one construction of it beats another, and it cannot decide
anything about an effect smaller than a quarter of a Sharpe point. No claim below
that threshold will be made.

---

## 11. Placebos, specified before any result

Both are mandatory. Both already exist in committed code.

**P1 — circular rotation of the leverage path.** For every sizing or conditioning
claim, the leverage series is rotated circularly against the dates, 400 draws.
This preserves the autocorrelation of the leverage path exactly and destroys only
its calendar alignment. Implementation: `scripts/run_extensions.py`. Its
unbiasedness is checked by confirming that the placebo mean lands on the
unconditional book Sharpe, as it did at +0.428 against 0.44.

**P2 — signals re-derived on block-shuffled returns.** Never a permutation of the
P&L column. The signal is recomputed from shuffled prices so that the placebo
inherits the estimation noise of the signal itself. Convention taken from
that repository's own pre-registration, whose K2 is stricter than this
repository's.

**P3 — matched-exposure control.** Every conditioned arm is compared against a
constant-exposure book carrying the same mean gross exposure. The barrier study
showed that p(pass) rises at 0.2785 per unit of exposure and that four devices
out of four delivered nothing beyond their own average exposure; T3 showed the
same illusion through beta. Any Sharpe gain must be shown not to be an exposure
gain.

---

## 12. Trial accounting

Every configuration evaluated is appended to `data/trials.parquet` through
`regime_lab.analysis.trials`, including configurations abandoned after one look.
The DSR deflation in K5 uses the full count, existing 83 configurations included.

Sensitivities that are *reported but not chosen between* — the three cost
columns, the three bootstrap block lengths, the M1 sensitivity with eight FX
series unshifted, the M2 variant excluding flagged instruments — are not separate
trials, because no selection is made among them. If a selection is ever made, it
becomes a trial retroactively and the deflation is recomputed.

---

## 13. Hard rules this study inherits

Repeated so that a reader does not have to look them up, and so that a violation
is visible.

- Signal at T−1, trade at T. No exception.
- Excess returns, HAC lag 6, multiple-testing correction whenever n_tests > 1.
- Walk-forward with at least 5 folds, genuinely evaluated, never a single split.
- Daily volatility targeting. *(This study is the first in the programme to
  actually satisfy this rule at book level.)*
- N ≥ 30 instruments for any cross-sectional claim. Satisfied from 2003-07-17.
- At most three modifications per hypothesis. They are M1, M2, M3.
- Stationary block bootstrap, never iid.
- Matched placebo on any conditional claim.

---

## 14. Operational constraints

- Never re-run `scripts/run_phase2.py`. It takes 15–20 minutes and overwrites
  `states.parquet`, on which every published result depends. This study needs the
  state series only as a covariate for H-b, and reads them from cache.
- Never edit `docs/CHARTER.html`. Deviations go to `docs/PROTOCOL_FREEZE.md`.
- Re-download nothing.
- `~/.gitignore_global` contains `config.py`; any new file of that name needs
  `git add -f` or it vanishes from the repository without `git status` saying so.
- Use the repository's own venv.

---

## 15. What gets published, in every outcome

| outcome | what is written |
|---|---|
| H passes | an audited, cost-correct, excess-return, 23.2-year, 46-instrument trend book that clears a research gate — the first deployable result of the programme |
| H fails, H-a passes | the benchmark book the programme has always lacked, plus a measured statement that volatility targeting is the active ingredient and the trend signal is not enough |
| H fails, H-a fails | the vehicle-dependence result: the only device this programme could not beat loses its significance under its own declared cost schedule, and practice's universal base layer is not universal net of cost |
| H-b passes | the single surviving use of regime classification in this programme, found on the sixth attempt and on the first book worth conditioning |
| H-b fails | the regime line closes, on the record, with six independent refutations and a measured mechanism |

Every row is a result. That is the point of writing this before the measurement.
