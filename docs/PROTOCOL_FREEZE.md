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
| 2026-09-08 | state ordering | **APPLIED** — states are ranked by the volatility realised in each on the training window, replacing the reference implementation's ranking by cumulative return | A diagnosis, not a search. Scored against NBER dates the old convention gave the jump model 6.0% balanced accuracy — anti-correlated with every external reference, with all 434 recession days in the state it called strong. The new rule never touches returns and cannot invert. Both orderings are reported. See docs/RESULTS_CLASSIFIER.md. |
| 2026-09-08 | sizing rule | **ADDED** — a graded inverse-volatility sizing rule alongside the frozen on/off rule; the on/off rule is still reported | The frozen rule is a timing rule and the measured signal separates variance, not mean. Adding the rule the evidence supports is disclosed here rather than substituted quietly, and the original stays in every table. |
| 2026-09-08 | grid | **DECLINED** — the jump-penalty grid was not widened despite the optimum landing on its floor | Extending a grid after seeing which end wins is an amendment made in the light of results. Recorded in docs/CALIBRATION_NOTES.md. |
| 2026-09-08 | v1 → v2 | T2 respecified; families D cut; strategies frozen; power section added; data coverage corrected | Adversarial review found T2 was scale-invariant, hence vacuous on Sharpe, and that deferring the strategy choice to phase 4 was circular. |

## Deviations from the frozen text

The charter is not edited to match reality; deviations are recorded here.

| item | charter says | actual | why |
| --- | --- | --- | --- |
| tag signature | "signed git tag pushed publicly" | annotated tag, unsigned, pushed publicly | No signing key is configured on the author's machine. The property that matters for a third party is the **push date**, which GitHub records independently of the signature. To be upgraded if a key is set up; the tag will be re-cut and this row updated. |
