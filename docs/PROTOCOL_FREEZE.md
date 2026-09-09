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

## Deviations from the frozen text

The charter is not edited to match reality; deviations are recorded here.

| item | charter says | actual | why |
| --- | --- | --- | --- |
| tag signature | "signed git tag pushed publicly" | annotated tag, unsigned, pushed publicly | No signing key is configured on the author's machine. The property that matters for a third party is the **push date**, which GitHub records independently of the signature. To be upgraded if a key is set up; the tag will be re-cut and this row updated. |


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
