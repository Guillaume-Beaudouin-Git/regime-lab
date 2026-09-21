# Protocol freeze

On data everyone has already seen, pre-registration cannot mean "data unseen".
It means **the protocol was fixed before the results were looked at**, and that
the fixing is verifiable by a third party. That is what this file records.

## Frozen document

`docs/CHARTER.html` — charter v2, 8 September 2026.

```
SHA-256  5791c1887c41d1f6b0449bafad2125013b82dc6b3506b4d72564e14ec1abc0db
```

Verify with:

```bash
shasum -a 256 docs/CHARTER.html
```

The charter is never amended in place. Revisions go in the amendment log below,
each dated and justified, so that a reader can see what changed and when.

## What the freeze commits us to

- The decisive test (T2) as specified: dynamic ex-ante volatility targeting,
  target calibrated on training folds only, paired block bootstrap, minimum
  detectable effect reported, metrics that are not scale-invariant.
- The two base strategies, fixed before any regime is estimated: 60/40, and an
  equal-risk multi-asset momentum book.
- The three stopping rules and the three admissible outcomes.
- Transaction costs in basis points, written before results.

## Amendment log

| date | version | change | reason |
| --- | --- | --- | --- |
| 2026-09-08 | state ordering | **APPLIED** — states are ranked by the volatility realised in each on the training window | A sign error in this repository's wrapper, not a property of the reference implementation. See the correction below. |
| 2026-09-10 | audit corrections | **APPLIED** — NBER scored against the calendar month rather than through a 45-day publication lag; a causal volatility-quantile placebo added to every table; marginal information reported on forward volatility as well as forward returns | Found by an independent audit. Each overstated or omitted a result; the corrected figures are lower and are the ones published. |
| 2026-09-08 | sizing rule | **ADDED** — a graded inverse-volatility sizing rule alongside the frozen on/off rule; the on/off rule is still reported | The frozen rule is a timing rule and the measured signal separates variance, not mean. Adding the rule the evidence supports is disclosed here rather than substituted quietly, and the original stays in every table. |
| 2026-09-08 | grid | **DECLINED** — the jump-penalty grid was not widened despite the optimum landing on its floor | Extending a grid after seeing which end wins is an amendment made in the light of results. Recorded in docs/CALIBRATION_NOTES.md. |
| 2026-09-08 | v1 → v2 | T2 respecified; families D cut; strategies frozen; power section added; data coverage corrected | Adversarial review found T2 was scale-invariant, hence vacuous on Sharpe, and that deferring the strategy choice to phase 4 was circular. |
| 2026-09-13 | T1, T3, T5 | **APPLIED** — the three controls the charter promised and `RESULTS_FINAL.md` declared missing are implemented and run: `scripts/run_t1_control.py`, `run_t3_control.py`, `run_t5_refit.py` + `run_t5_control.py`. Results in `docs/RESULTS_FALSIFICATION.md` | A promise in a frozen, publicly tagged document is an obligation, not an option. They were implemented after the primary results were known, which is disclosed here: each was specified from the charter text before its own output was read, and each reports the outcome it produced rather than the outcome that would have been convenient. T1 and T3 weaken the study's position; they are published unchanged. |
| 2026-09-13 | five folds | **APPLIED** — the five declared folds are evaluated for the first time: fold fixed effects, leave-one-fold-out, Cochran's Q. Results in `docs/RESULTS_FOLDS.md` | `walk_forward` had one call site and was used only to read the first fold's start date. The central result survives all three checks; the weakest leave-one-out cell is 1.74 points at t −2.42, dropping the financial crisis. |
| 2026-09-13 | walk-forward of record | **APPLIED** — the executed protocol (one expanding block, 49 semi-annual refits, online prediction) is declared the walk-forward of record; the five-fold slice is retained as a dispersion diagnostic, not as the estimator | Not a preference. Two of A′ sparse jump's five folds contain a **single state across 1,279 trading days**, and one of A jump's does, so the charter's "three folds of five" stopping rule is arithmetically uncomputable for the families the study leads with. An object with a 456-day mean run length cannot be cut into five five-year windows and still vary inside each one. Measured in `docs/RESULTS_FOLDS.md` §2. |
| 2026-09-13 | heterogeneity p-values | **WITHDRAWN** — the fold-interaction Wald test on all five folds, which reported A′ sparse jump at p = 4.5 × 10⁻⁵ | Not identified. With two degenerate folds the design matrix has rank 9 for 11 columns in A′ and 10 for A jump. Superseded by `scripts/run_fold_heterogeneity.py`, which checks rank first and reports A′ at p 0.218. The rank-deficient figure was found and retracted by the same work that produced it. |
| 2026-09-21 | new study | **OPENED** — `docs/PRESPEC_TREND_VEHICLE.md`, locked before the repaired book produced a return. The 46-instrument trend book becomes the *subject* rather than the vehicle: three modifications and no others (realign the ten spot FX series, back-adjust GC/SI, apply a portfolio-level volatility target), with regime state entered as one covariate in the risk budget and never as a switch | Audits of the published artefacts found the reference book of every comparison in `EXTENSIONS.md` is gross, unsized, and cost-fragile: gross turnover 14.79×/yr against a 15.7 bp breakeven, so this programme's own cost schedule takes 48.8% of the gross return and the net Sharpe is 0.261. Ten spot FX series are stamped one session late — corr(t+1,t) 0.877-0.896 against the matching CME futures, confirmed on the Brexit session — and GC=F/SI=F correlate 1.0000 with raw front-month, so roll gaps are counted as returns. And the winner of `EXTENSIONS.md` is **vehicle-dependent**: +0.244, t +2.91 on futures at 1 bp, but +0.106, t +1.23 on the retail mix implied by the stored universe, under its own MDE. None of the five refutations tested the construction of the book; all tested a conditioner for which it was the vehicle. The non-blindness is disclosed in §1 of the pre-registration: the futures vehicle was chosen after seeing that cost sensitivity. |
| 2026-09-14 | `external_validation` | **APPLIED** — the function now returns NaN when either the state or the reference has fewer than two distinct values, instead of scoring the degenerate case; every branch returns the same six keys, where the short-circuit branch previously returned three. Covered by `tests/test_reliability.py` | A defect with two halves, and the larger half was not the one first noticed. A constant state satisfies `state == state.min()` everywhere, so the classifier was graded as if it had called a recession every day — three per-fold cells read 0.0%. A **one-class reference** has no balanced accuracy at all: the mean collapses onto the recall of the only class present and kappa's denominator vanishes, returning 0.000 indistinguishably from real incompetence. That affected 15 of 25 per-fold cells, three of which read *high* — A jump showed 75.6% and 98.0% in folds 0 and 4, C gradient boost 16.8% in fold 4, all of them specificities presented as balanced accuracies. Six misleading cells, three low and three high, from one defect with opposite signs. **No published figure changes**: every published number is full-sample, where both series move, and A′ sparse jump stays at 0.9316 / 0.5342. NBER fires in only 2 of these 5 folds, so per-fold external validation is structurally impossible on this sample and the honest table has two rows. |

## Deviations from the frozen text

The charter is not edited to match reality; deviations are recorded here.

| item | charter says | actual | why |
| --- | --- | --- | --- |
| tag signature | "signed git tag pushed publicly" | annotated tag, unsigned, pushed publicly | No signing key is configured on the author's machine. The property that matters for a third party is the **push date**, which GitHub records independently of the signature. To be upgraded if a key is set up; the tag will be re-cut and this row updated. |
| stopping rule 1 | a positive gap above the MDE on three folds of five | uncomputable for A and A′; **0 of 5 for every family** on the folds that are identified | Two of A′'s five folds hold one state, so three of five cannot be assessed for it. Where it can be assessed, no family clears the per-fold MDE in any fold: per-fold gaps swing from −0.71 to +1.01 against MDEs of 0.35 to 1.36, so the fold-level test has no resolution. Reported in `docs/RESULTS_FALSIFICATION.md` §T1 and `docs/RESULTS_FOLDS.md` §2. |
| T1 timing | a control run before the primary results | run after them, 2026-09-13 | The control was not implemented during the study. It is specified from the charter text, its sensitivities were declared before its own primary output was read, and it produced a result that weakens the study. Recording the order is the only defence available after the fact. |


## Correction — the inversion was a sign error here, not a convention there

The first version of this log explained the inverted labels as a property of the
reference implementation: that the volatile state had earned more cumulatively
in training, so ranking by cumulative return put it first. **That explanation is
wrong, and the repository's own data contradicts it.**

Refitting with `ordering="cumret"` on the training window through March 2002:

```
state 0 : cumulative return +0.363   volatility 0.0050   1,607 days   (the CALM state)
state 1 : cumulative return +0.167   volatility 0.0079   1,023 days   (the VOLATILE state)
```

The volatile state earned **less**, not more. The real cause is that
`jumpmodels` sorts **descending** — index 0 carries the highest cumulative
return — while `mapping.py` documented the opposite ("states arrive ordered by
the return earned in them, so the strongest is the last index") and held
`n_states - 1`. The wrapper was long the weakest state by construction.

The filtered HMM sorted ascending, correctly, which is why the two families
disagreed at kappa −0.39. That gap was never a difference of convention between
methods; it was one bug in one wrapper.

The fix stands — ranking by volatility never touches returns and cannot invert —
but the justification recorded here was counterfactual, and an amendment log
that misstates its own reason is worse than no log. Anyone can reproduce the
three lines above in a single fit.
