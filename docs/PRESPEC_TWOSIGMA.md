# PRESPEC — Regime as a contextual selector across a signal library

**Status: LOCKED 2026-09-23, by the commit that adds this file, before any statistic of
the escalation tree was read. Any edit from here is an amendment in
`docs/PROTOCOL_FREEZE.md`.**

**Sign-off:** given. Guillaume read and approved this text on 2026-09-23, before the
commit that adds it. He asked for two further independent reviews before the commit: a
second validator, on the substance, and a pre-commit auditor, on blindness,
reproducibility and hygiene. Both read the text, their blocking findings are applied here
(§12.0 rows 28-36, and wording), and both confirmed the corrected text before the commit.
The lock also rests on his instruction of 2026-09-22 to run the Two Sigma plan ("GO"),
renewed on 2026-09-23. Any change asked for from here on is logged as an amendment in
`docs/PROTOCOL_FREEZE.md`.

**Programme:** `regime-lab`, post-study extensions.
**Firm template:** Two Sigma — unsupervised clustering over a context space, used to
overweight signals that historically perform in that context and to shut off those
exposed to it negatively.
**Parent documents:** `docs/CHARTER.html` (frozen), `docs/PROTOCOL_FREEZE.md`,
`docs/EXTENSIONS.md`, `docs/PRESPEC_TREND_VEHICLE.md` (the closed study),
`docs/PRESPEC_AHL.md` and `docs/RESULTS_AHL_LEVEL_B.md` (the discipline this lock
follows: the instrument measured under the null and its criterion committed before the
reading), `pilotage/plans_de_recherche/ARBITRAGE.md` §4.
**Build locked against:** commit `3b64ab25055dd367cbded13a1bdea75c6136c1a3`, which
adds to the build commit `22abbe0dc59eed3be64639f8c4cef5438f33b081` what the three
audits of this text asked code to pin (the cells of §12.4, the smoothing of §12.13, the
kill's base rate of §12.6, the count of §7, the non-finite placebo rule of §13.4) —
`regime_lab/selection/{library,folds,context,protocol}.py`,
`regime_lab/analysis/placebo.py`, and the three blind measurement scripts
`scripts/measure_twosigma_library.py`, `scripts/measure_twosigma_context.py`,
`scripts/measure_twosigma_power.py`. All three exit 0 at that commit (library 16/20,
context 43/43, power 10/10), rerun on 2026-09-23. Every figure this text quoted from the
build commit is unchanged; the new sections add figures and move none. The environment
is the committed `uv.lock` (at this writing scikit-learn 1.9.1, numpy 2.5.3, scipy
1.18.1, pandas 3.0.6, statsmodels 0.15.0), on which bit-reproducibility depends
(§13.1).
**The draft:** `pilotage/plans_de_recherche/twosigma/PRESPEC_TWOSIGMA.md` (DRAFT
2026-09-22) stays where it is, unedited, as the record of what was written before the
declared object was measured. `PLAN.md` beside it is the longer French plan. Where
either disagrees with this file, this file wins.

**How to read this file.** §I states what the instrument measured before the lock and
which disclosed figures did not describe the declared object. §0 to §11 keep the
draft's structure. Every figure there that did not hold is replaced in place and marked
*(lock)*. §12.0 lists every change from the draft, what forced it, and its direction.
§12 fixes every choice the draft left open, with its reason and the direction it
pushes, and §13 is the decision procedure. **Where §0-§11 and §12-§13 differ in
wording, §12-§13 decide.** Where §12 and §13 differ, the more specific paragraph
decides: §12.6-§12.10 for a lock's own statistic, verdict lines and instrument, §13 for
the order of work, the precedence between locks, Holm and logging. §14 is the scope and
what stays open.

**Audited before the commit.** Three independent audits read this text against the
build commit before it was committed. Their blocking findings were applied, or refused
with a reason, before the lock (§12.0 records each change they caused). They confirmed
that the build computed no mean, Sharpe, IC, hit rate, alpha, `m_ik`, or per-state
statistic of any signal return, and they judged the unconditional sds and cap shares
cited in §I and §11 allowed (§14). The build does compute per-state quantities of the
labels and of volatility (occupancy, episode counts, η² against log volatility); none
touches a signal return. A pre-commit auditor and a second validator then read the
final text on 2026-09-23 (see Sign-off).

**No statistic of the escalation tree has been read at the time of this lock.** No
mean, Sharpe, IC, hit rate or alpha of any signal, blend or selector has been computed,
and no per-state quantity of any signal return either. No `m_ik` has been estimated
from returns, and no covariance-forecast loss has been computed.

---

## I. What the instrument measured before the lock

The draft's §1 disclosed figures produced by pre-lock scripts that were never committed.
The build commit ported their logic into tested package code. Its three scripts first
rerun each disclosed figure through committed code, then measure the object that §4
actually declares. Everything below is a count, a date, a transition, an occupancy, a
correlation between weight vectors, an effective rank, a turnover `|Δw|`, an η² of a
partition against realised volatility, or a bootstrap standard error or MDE taken from
legs **demeaned** before the bootstrap. The demeaning is enforced by
`protocol.blinded_mde`, and the largest `|observed|` over every call is 7.4e-16.

### I.1 The disclosed figures, re-measured from committed code

| figure | disclosed | re-measured | match | comment |
|---|---|---|---|---|
| P1 mean abs pairwise position corr (46 instr., 11 signals) | 0.336 | 0.3363 | yes | |
| P1 effective rank | 3.86 | 3.862 | yes | admission-critical |
| P1 first eigenvalue | 4.82 | 4.817 | yes | |
| P1 components for 90% | 6 | 6 | yes | |
| P1 TSMOM_1M / XSREV_1M | -0.990 | -0.9897 | yes | |
| P1 TSMOM_12_1 / XSMOM_12_1 | 0.930 | 0.9279 | at 2 dp | the pre-lock printout used `round(c, 2)`; the draft wrote 3 decimals |
| P1 TSMOM_3M / BRKOUT_100 | 0.840 | 0.8387 | at 2 dp | same rounding |
| P2 mean abs corr, 12 candidates | 0.101 | 0.1005 | yes | |
| P2 signed mean corr, 12 | +0.013 | +0.0130 | yes | |
| P2 effective rank, 12 | 8.37 | 8.372 | yes | admission-critical |
| P2 components for 90%, 12 | 8 | 8 | yes | |
| P2 IND_ACCEL / IND_MOM_12_1 | -0.912 | -0.9118 | yes | |
| §2/§10 effective rank of the 10-signal LIBRARY | 8.37 | **7.965** | **NO** | the draft attached the 12-candidate figure; the axis-4 floor of 4 is still cleared |
| §3 common sample | 7,946 sessions, 31.57 y | 7,946, 31.570 y (1995-01-04 to 2026-07-31) | yes | the start is set by `IND_SEASON`, not by the 1,260-session LTREV as §3 said (LTREV is complete from 1994-12-23) |
| P3 context span | 34.52 y | 34.519 y | yes | |
| P3 K=4 raw: transitions / per yr / η² | 223 / 6.46 / 0.421 | 223 / 6.460 / 0.4215 | yes | η² against rv LEVEL |
| P3 K=4 orth: transitions / per yr / η² / episodes | 287 / 8.31 / 0.048 / 288 | 287 / 8.314 / 0.0476 / 288 | yes | η² against LOG rv. Like for like: 0.048 vs 0.441 (log), or 0.085 vs 0.421 (level) |
| P3 K=4 raw episodes, occupancy | 224; 0.261/0.427/0.091/0.221 | 224; 0.2615/0.4266/0.0907/0.2213 | yes | |
| `clock.csv` K=3/5/6 raw | 141, 290, 293; 4.08, 8.40, 8.49; 0.371, 0.487, 0.466 | identical at the draft's precision | yes | |
| `clock_resid.csv` K=3/5/6 orth | 200, 266, 264; 5.79, 7.71, 7.65; 0.008, 0.098, 0.143 | identical at the draft's precision | yes | |
| pre-lock episodes on the toy sample | 215; 46/71/20/78; median 8; mean 37.0 | 215; 46/71/20/78; 8; 36.99 | yes | |
| §8 table reference clock for attempts 1-2 | "real 6.46" | **214 transitions = 6.77/yr** on the toy sample | **NO** | wrong reference in the draft table; `placebo.py` states 6.77 |
| P4 hard switch, block 63 | 0.474 | 0.4741 (blocks 21/126: 0.515/0.425) | yes | TOY 5-signal library, random switch every 21 sessions, full 31.6 y, gross |
| P4 tilt d=0.50, 8 generators, min/median/max | 0.221/0.241/0.294 | 0.2211/0.2406/0.2945 | yes | toy |
| §7 uncorrected MDE | 0.272 | 0.2717 | yes | generator 0 alone |
| P4 α 0.05/6 | 0.338 | 0.3375 | yes | ONE draw (0.272) scaled by 1.2421, not the 8-draw median; the median at 0.05/6 is 0.299 |
| P4 α 0.05/21 | 0.376 | 0.3763 | yes | same single draw |
| `measure_fix.py` single-seed figure | 0.238 | 0.2377 | yes | in the old printout only, not in the draft |
| P5 blend turnover | 21.6×/yr | 21.59 (11-signal blend incl. `IND_REV_1W`) | yes | 11-signal blend |
| P5 control turnover (10 signals, §4) | 21.6×/yr | **13.64** unscaled | **NO** | the draft calls the 11-signal figure the control's |
| P5 blend cost at 5 bp | 1.08%/yr | 1.079 (11 signals) | yes | 0.68%/yr for the 10-signal control, unscaled |
| P5 selector increment | +2.6×/yr (+0.013 Sharpe) | +2.58 (21.59 to 24.2) | yes | 11 signals, in-sample raw clock, lag 0, UNSCALED weights: none of these is the declared object |
| P5 `IND_REV_1W` | 156.7×/yr, 7.83%/yr | 156.66, 7.833 | yes | |
| P6 swap placebo, full-sample raw clock, 1 block | 300/300; 223 each; sd 0.0; occ. dev. 0.0000 | 300/300; 223; 0.0; 0.0000; max tries 33 | yes | validated on ONE block of the in-sample unorthogonalised clock only |
| §5 kill-rule arithmetic | 92.7 bp; 12×/yr at 20 bp vs 0.241 | 92.7; 12.05 | yes | arithmetic only; its inputs do not hold on the declared object |
| Lo standalone threshold | 0.533 at 31.57 y | 0.533 | yes | over the 20.0 paired years: 0.699 / 0.932 / 1.099 at α 0.05, 0.05/6, 0.05/21 |
| script totals | | library 16/20, context 43/43, power 10/10 | | the library script runs in about 1 s, context about 4 min, power about 2 min 20 s on 6 workers |

### I.2 The declared object against the draft

These rows measure what §4 declares, not what the pre-lock scripts measured. None of
them reproduces a disclosed figure, because none of the disclosed figures was taken on
this object.

| quantity | draft | declared object | comment |
|---|---|---|---|
| OOS clock (K=4 orth, anchor end, nearest centroid, 5 test folds = 20.0 y) | 8.31/yr (P3), "6.46 to 8.31" (§2) | **12.55/yr** (context training from 1995) / 12.80 (from 1992); per fold 12.00/17.75/19.25/10.00/3.75 | the disclosed object over the same 20 y runs at 9.65/yr; the refit adds the rest (+30%) |
| η² vs log rv, pooled OOS aligned labels | 0.048 | **0.142** / 0.162; fold 3: 0.608 / 0.629 | admission axis 1 is much weaker than disclosed |
| A-1 MDE, tilt d 0.50 vs control, block 63, α 0.05/6 | 0.338 | median **0.262** (0.178 to 0.361) over 8 random N(0,1) m tables | 0.77 times the draft; blocks 21/126: 0.263/0.264 |
| MDE uncorrected / α 0.05/21 | 0.241 / 0.376 | 0.211 (0.144 to 0.291) / 0.292 (0.199 to 0.403) | |
| hard switch MDE, α 0.05 / 0.05/6 | 0.474 (toy) | 0.709 (0.523 to 0.832) / 0.881 | reference only; §4 declines the switch |
| control turnover, held weights, test folds | 21.6 | **36.7×/yr** (13.6 unscaled) | the volatility multiplier scales the book up about 2.7 times |
| selector increment, held, d 0.50 | +2.6×/yr | **+7.85** (1.78 to 11.75) | |
| increment cost at 5/10/20 bp | 0.013 at 5 bp | 0.039 / 0.079 / 0.157 Sharpe | at the 10% nominal target |
| control's whole cost at 5 bp (level-C ceiling) | 0.108 | **0.183** at the 10% target; **0.233** at the realised 7.88% sd | at the realised sd, exceeds the re-measured uncorrected MDE of 0.211 |
| breakeven | 92.7 bp | **27 bp** (vs 0.211) / 33 bp (vs 0.262) | |
| kill turnover at 20 bp | 12×/yr | **10.54** (vs 0.211) / 13.09 (vs 0.262), at the 10% nominal target | 2 of 8 RANDOM maps already exceed 10.54 |
| the kill as the proposed lock re-derived it, map by map: `min(12, MDE(0.05, 63) × σ_arm × 500)` *(audit)* | — | fires on **4 of 8** random maps, and on **3 of 8** random maps given §12.5's shape of `m`; 4 of 8 and 5 of 8 exceed the median-based 8.31 | kills by construction, whatever the map holds: **not adopted** (§12.6) |
| the kill anchored to the decision bar: `min(12, T × σ_arm × 500)`, T = max(0.338, the map's MDE(0.05/6)) *(audit)* | — | 12.00 on every map; fires on **0 of 8** random maps and **1 of 8** shaped maps | **adopted** (§12.6) |
| selector increment, random maps given §12.5's shape *(audit)* | — | +8.55 (6.19 to 13.59); MDE(0.05/6) at block 63 0.260 (0.195 to 0.367), largest block 0.269 | the shape puts the whole RMS in the part that varies by state |
| qualifying cells, §12.4's rule on the stamped paths *(audit)* | — | K=4: **19 of 20** training cells, **14 of 20** A-2 cells (2/3/4/3/2 by fold); **372 of 5,031** test sessions (7.4%) abstain. K=3: 15/15 and 11/15, 0 abstain. K=5: 24/25 and 15/25, 34. K=6: 28/30 and 17/30, 30 | every UNDECIDABLE floor (§13.4) is cleared |
| 21-session smoothing, `PLAN.md` §f.4 *(audit)* | 2.64/yr (`PLAN.md`, pre-lock, not on the declared object) | **3.80/yr** out of sample (12.55 unsmoothed), 2.12 in sample; median episode **41** sessions; η²(log rv) 0.100; 13 of 20 A-2 cells; uniform placebo exact | reinstated as a sensitivity (§12.13) |
| realised sd of the control; cap binding | 10% assumed | **7.88%/yr**; the cap of 3 binds on **70.5%** of test sessions | unscaled gross median 0.348 |
| swap placebo on (fold, segment) blocks | 300/300 (P6) | K=4: **4/1,000** (1995) and 36/1,000 (1992); K=3: **0** and 0 | the uniform construction is exact on all 8 objects |
| Level-B design constraint (§6 B) | 11.5% of variance through correlations (n=11, ρ=+0.013) | the same formula on the 10-signal library (ρ=+0.035): **23.9%** | position space only; the formula uses position correlation as a stand-in for return correlation |

### I.3 What did not describe the declared object, why, and what the lock does

1. **Effective rank of the ten-signal library: 7.97, not 8.37.** The draft attached the
   twelve-candidate figure (8.372, which reproduces) to the ten-signal library. *Lock:*
   §2 and §10 are restated at 7.97 of 10. Axis 4's floor of 4 is still cleared, so
   admission is unaffected.
2. **Control turnover: 13.6×/yr unscaled and 36.7×/yr held, not 21.6.** The 21.6 is the
   eleven-signal blend that still held `IND_REV_1W`. §4 excludes that signal before
   contact, so the control holds ten. *Lock:* §5 is restated on held weights, which are
   what costs are charged on.
3. **P4 was measured on a toy, not on the declared pairing.** The pre-lock scripts
   (`measure_power*.py`, `measure_fix.py`, `measure_amtp.py`) built a five-signal
   library: 12-1 momentum, 1-month reversal, a 1,008-session long-term reversal, low
   volatility, and a "seasonality" that is the 12-month return lagged 12 months. They
   redrew a random tilt every 21 sessions, used the full 31.6 years, gross, in raw
   returns. A-1 compares out-of-sample net excess returns on the five test folds only,
   20.0 years. The 0.338 is one draw (0.272) rescaled to α = 0.05/6, not the eight-draw
   median, which would give 0.299. *Lock:* 0.338 is kept as a **floor**, not as a
   measurement. The A-1 bar is the procedure of §12.6, which reads the declared object's
   own MDE and never goes below 0.338.
4. **P3's clock and η² describe one full-sample fit with in-sample labels, over 34.52
   years.** The declared object is a walk-forward refit, orthogonalised, labelled out of
   sample by nearest centroid. It runs at 12.55 transitions a year out of sample, not
   8.31, and its pooled out-of-sample η² against log rv is 0.142, not 0.048. The draft's
   η² pair also mixes scales: 0.048 is against log rv, 0.421 against rv levels. *Lock:*
   §0, §2, §6 and §11 are restated on the declared object. The full-sample figures stay
   in §1 as disclosed history. η² is reported against **log** rv only.
5. **P6's placebo was validated on one block of the unorthogonalised full-sample clock.**
   On the declared object, cut into one block per (fold, segment), the swap repair
   completes 4 draws of 1,000 at K=4 (36 with the 1992 start) and none at K=3. *Lock:*
   construction 4, uniform over join-free orders, is adopted for the primary and for
   every sensitivity (§12.11).
6. **P5's selector increment (+2.6×/yr, +0.013 Sharpe) and the §5 cost figures derived
   from it (92.7 bp, 12×/yr).** They used eleven signals, the in-sample raw clock, lag 0
   and unscaled weights. On the declared object the increment is +7.85×/yr (random-map
   median), 0.039 Sharpe at 5 bp, breakeven 27 to 33 bp. *Lock:* §5 is restated, and the
   kill rule is re-derived in §12.6.
7. **Level C's premise, "0.108 < 0.241, undecidable by arithmetic".** The control's
   whole cost at 5 bp is 0.183 Sharpe at the 10% target and 0.233 at the book's realised
   7.88%. The realised-sd figure, which is the right conversion (§12.3), exceeds the
   declared object's uncorrected MDE of 0.211; the nominal one sits just below it. Both
   stay below the corrected bar of at least 0.338. *Lock:* C's Sharpe channel is undecidable
   against the corrected bar only. C stays a turnover test, with its own comparator and
   its own null (§6 C, §12.10).
8. **Level B's design constraint: 11.5% used n = 11 and the twelve candidates' ρ.** On
   the ten-signal library the same formula gives 23.9%. *Lock:* restated in §6 B. The
   return-space version is printed by the level-B instrument before B is read (§12.8).
9. **The book does not run at 10%.** The inherited cap of 3 on the multiplier binds on
   70.5% of the control's test sessions, and the control realises 7.88%. *Lock:* the cap
   is kept, and every cost is converted to Sharpe at the realised sd (§12.3).
10. **Text errors needing no decision.** §8's "real 6.46" is the toy sample's 6.77. §3's
    binding warm-up is `IND_SEASON`, not LTREV; the sample is unchanged. §10.4's
    "+0.013 Sharpe, dead only above 93 bp" does not hold on the declared object. The two
    P1 pairs match at two decimals, a rounding of the old printout. All are corrected in
    place.

### I.4 The declared object, as measured

**Object.** The ten-signal library on 49 industries, in excess of `ff_rf`. A 10% daily
volatility target with the inherited cap of 3, net of 5 bp on held weights. K=4 K-means
on 20 context features orthogonalised on log rv of `eq_us_large`, refitted per fold,
labelled out of sample by nearest centroid, then lagged one session. Walk-forward
anchor `end`.

**Folds (anchor `end`).** Test sessions 1,007 / 1,007 / 1,007 / 1,006 / 1,004. Test
windows run from 2006-08-01 to 2026-07-31: 5,031 sessions, 20.00 years. The first
training window is 11.57 years (2,915 sessions, cut-off 2006-07-31). The training
windows expand to 3,922 / 4,929 / 5,936 / 6,942 sessions. With anchor `start`, the
first window is exactly 11.6 years and the last fold is cut to 3.97 years (996 sessions).

**Clock** (context training from 1995-01-04 | from 1992-03-02):

| quantity | 1995 | 1992 |
|---|---|---|
| OOS transitions/yr, pooled | 12.55 (251) | 12.80 |
| per fold | 12.00/17.75/19.25/10.00/3.75 | 3.75/21.25/19.50/13.00/6.50 |
| in-sample clock, mean over fold models | 6.41 | 6.87 |
| disclosed object on the same 20 y | 9.65 | 9.65 |
| η² vs log rv, pooled | 0.142 | 0.162 |
| η² per fold | 0.065/0.370/0.608/0.152/0.131 | 0.009/0.427/0.629/0.142/0.116 |
| refit agreement, folds 2-5 | 0.845/0.843/0.916/0.954 | 0.607/0.920/0.962/0.927 |
| median / mean episode (sessions) | 4 / 19.7 | 3 / 19.3 |
| (fold, state) test cells with 0 episodes | 3 | 2 |
| fewest episodes in a training cell | 2 (fold 1) | 4 |
| fold-5 test episodes per state | 8/0/0/8 (2 states) | 10/11/4/2 |

Pooled OOS occupancy (aligned numbering): 0.122 / 0.269 / 0.388 / 0.220. Episodes per
state in each (fold, segment) block, 1995 start: 1:train 21/2/19/7, 1:test 23/3/22/1,
2:train 44/8/7/38, 2:test 24/32/0/16, 3:train 18/36/44/44, 3:test 15/23/18/22,
4:train 35/62/49/32, 4:test 8/12/8/13, 5:train 65/32/51/42, 5:test 8/0/0/8.
Variants (1995 | 1992): unorthogonalised 10.10 | 9.55/yr (η² 0.353 | 0.369); anchor
`start` 12.97 | 12.32; K=3 10.90 | 10.90; K=5 15.40 | 14.35; K=6 16.30 | 20.90.

**Labels and lag.** No test session lacks a label (0 of 5,031). With each fold's own
path (`session_paths` plus `map_states`) there are 0 lag mismatches, and the first test
session is correct in 5 of 5 folds. The power script's pooled-aligned path agrees with
it on 5,030 of 5,030 comparable test sessions (1995 start). The only difference is fold
1's first session, which holds the equal-weight fallback. With the 1992 start there is 1
mismatch.

**Placebo feasibility** (blocks = (fold, segment), 1,000 draws). The swap repair
completes: K=4 4 draws (1995) / 36 (1992); K=3 0 / 0; K=5 3 / 465; K=6 108 / 724. The
uniform construction is exact on all 8 objects: transition sd 0.0, occupancy deviation
0.0000, episode multisets kept. It takes about 10 s per 1,000 draws. Freedom per block,
as log10 of the number of join-free state orders, runs from 0.3 in block 5:test (1995),
where only 2 orders exist, up to 82.3. With the 1992 start it runs from 1.5 to 102.7.

**Library geometry.** Effective rank 7.965 of 10, 8 components for 90%. Mean abs corr
0.090, signed +0.035, max abs 0.673 (MOM_12_1/MOM_6_1).

**Power (A-1).** Tilt d = 0.50 against the EW control, 8 random N(0,1) m tables,
blinded bootstrap with 2,000 draws; median (range):
- block 63: SE 0.075 (0.051 to 0.104);
- MDE at α 0.05: 0.211 (0.144 to 0.291). At 0.05/6: 0.262 (0.178 to 0.361). At
  0.05/21: 0.292 (0.199 to 0.403);
- at 0.05/6, block 21: 0.263 (0.187 to 0.364); block 126: **0.264** (0.167 to 0.323);
- d = 0.25: 0.109 (α 0.05) and 0.136 (0.05/6). d = 1.00: 0.325 and 0.404;
- hard switch: 0.709 and 0.881;
- m demeaned across states: 0.192 and 0.238;
- conditional tilt against a static K=1 twin: 0.196 and 0.244;
- Lo standalone over 20.0 y: 0.699 / 0.932 / 1.099.

**Costs (held weights, test folds).**
- Control: 36.7×/yr (13.6 unscaled); unscaled gross median 0.348; the cap binds on 70.5%
  of sessions; realised sd 7.88%.
- Control's whole cost: 0.183 / 0.367 / 0.733 Sharpe at 5 / 10 / 20 bp (10% nominal);
  0.233 at 5 bp on the realised sd.
- Selector, d 0.50: 44.5×/yr held, increment +7.85 (1.78 to 11.75). Increment at d 0.25:
  +2.78; at d 1.00: +14.49. With m demeaned: +6.89. Over the static twin: +7.03 (5.06
  to 11.92).
- Cost of the increment: 0.039 / 0.079 / 0.157 Sharpe at 5 / 10 / 20 bp.
- Breakeven: 27 bp (against 0.211), 33 bp (against 0.262), at the 10% nominal target.
  At the control's realised 7.88% the increment costs 0.050 / 0.100 / 0.199 Sharpe at
  5 / 10 / 20 bp, and the breakevens are about 21 and 26 bp.
- Kill turnover at 20 bp, 10% nominal: 10.54 / 13.09×/yr; 2 of 8 random maps exceed
  10.54. Map by map, at each map's own MDE and realised sd, see §I.2: the re-derived
  kill fires on 4 of 8 random maps, the kill anchored to the decision bar on 0 of 8
  (1 of 8 once `m` has §12.5's shape).
- The d = 0.50 tilt arms realise 7.6 to 8.8% sd, and the cap binds on 61 to 76% of their
  sessions. Across every random-map arm measured, the hard switch included, the range is
  7.6 to 10.8%.

### I.5 The admission test, re-checked on the re-measured figures

| Axis | The six | This study, declared object | Measured gap | Holds? |
|---|---|---|---|---|
| 3 — use | sizing, or an ON/OFF switch | **selection among signals** | undefined with one signal | **yes**, unchanged |
| 4 — conditioned object | one book, TSMOM 12-1 on 46 instruments | **10-signal library**, 49 legs | effective rank **7.97** vs 1 (floor 4) | **yes** |
| 1 — latent variable | states ordered by training volatility | context partition orthogonalised on log rv, walk-forward | pooled OOS η²(log rv) **0.142** (per fold 0.065 to 0.608), against 0.353 for the same refit unorthogonalised | **yes, weakened**: not orthogonal to volatility, less than half as volatility-bound as its unorthogonalised twin, and largely a volatility partition in fold 3 |
| 2 — time constant | 0.532 transitions/year | **12.55**/year out of sample (6.41 in sample) | **×23.6** | **yes**, more strongly, at the price of flicker: the median OOS episode lasts 4 sessions |

**A device the draft could not compare against.** Since the draft was written, AHL
level A has asked a selection question: which of two trend speeds each instrument
runs, on a change-point state, on the 46-instrument universe
(`docs/RESULTS_AHL_LEVEL_A.md`). On the leg that was read, the state did not move the
speed ranking. That leg is a declared sensitivity, and the headline reading is still
owed. AHL A shares axis 3 with this study, so axis 3 no longer separates this study from
everything the programme has tried. The difference from AHL A lies on axis 4 (two legs
of one trend factor, on the universe this study declares unable to host a selector) and
on axis 1 (a path-shape latent, not a context partition).

**Admission holds.** It is granted on axis 3 against the six, as the draft said, and on
axes 4 and 1 against AHL A. Axes 1, 2 and 4 remain necessary conditions for axis 3 to
be posable, and all three are met on the declared object. Axis 1 is the weak one, and
the lock answers it with a volatility witness at every level (§8 control 4; two
witnesses at B-1, §12.8) and per-fold reporting (§12.14). The refusal on the
46-instrument universe (effective rank 3.862) is unchanged.

---

## 0. What this study is, in one sentence

Six devices have asked a two-state, volatility-ordered classifier with thirteen
transitions in twenty-five years to size or to time a single trend book; this study
asks a context partition orthogonalised on volatility, refitted walk-forward and
switching about **twelve and a half times a year out of sample** *(lock; the draft said
roughly seven)*, to **arbitrate among ten signals whose regime profiles differ** — a
question that does not exist when there is only one signal, and there was only one.

---

## 1. Prior measurements, disclosed

Every figure below was computed before the draft was written, and is kept here as it
was disclosed. None of them is a return, a Sharpe or a mean. They are counts, dates,
position-space correlations, turnover, and bootstrap standard errors taken from
**deliberately demeaned** series. Each is followed by what the lock found (§I).

**P1. The 46-instrument universe cannot host this study.** Eleven candidate signals
built on `data/cache/trend_universe_m1.parquet`: mean |pairwise position correlation|
0.336, **effective rank 3.86**, first eigenvalue 4.82 of 11, six principal components
for 90% of the trace. `TSMOM_1M`/`XSREV_1M` correlate −0.990; `TSMOM_12_1`/`XSMOM_12_1`
+0.930; `TSMOM_3M`/`BRKOUT_100` +0.840. This is one trend factor at several horizons.
*Lock:* reproduces (3.862); the two pairs are +0.928 and +0.839 at three decimals.

**P2. The 49-industry panel can.** Twelve candidate signals on
`data/raw/panels/industry_49.parquet`: mean |pairwise position correlation| **0.101**,
signed mean **+0.013**, **effective rank 8.37**, eight principal components for 90% of
the trace. `IND_ACCEL` correlates −0.912 with `IND_MOM_12_1` and is dropped as a
construction redundancy before any contact. *Lock:* reproduces for the twelve. The
ten-signal inferential library has effective rank **7.965**, mean |corr| 0.090, signed
+0.035.

**P3. The context clock.** K-means, K=4, on a 20-feature context space holding no
`vol_*` feature: 223 transitions over 34.52 years = 6.46/year, 224 episodes. On the
same space orthogonalised against log realised volatility: 287 transitions = 8.31/year,
288 episodes, η²(realised vol) = 0.048 against 0.421 unorthogonalised. Reference:
`A' sparse jump` gives 13 transitions in 6,377 sessions = **0.532/year**. Gaussian
mixtures on the same space run far slower (0.90 to 2.43/year) and are not the declared
object. *Lock:* reproduces, but it is one full-sample fit with in-sample labels. The
declared walk-forward object runs at **12.55/year** out of sample, with pooled η²(log
rv) **0.142**. The 0.048/0.421 pair mixes log and level scales (§I.3 item 4).

**P4. Power under the declared pairing.** Hard switching one signal at a time: MDE
0.474. Weight tilt at standard deviation 0.50: MDE 0.221 to 0.294 across eight draws,
median 0.241. At the family-wise corrected α = 0.05/6: 0.338. At α = 0.05/21: 0.376.
*Lock:* reproduces, but it was **not** the declared pairing: a toy five-signal library
with a random tilt, over the full sample, gross. On the declared object: tilt 0.211
(α 0.05), 0.262 (0.05/6), 0.292 (0.05/21); switch 0.709 (§I.3 item 3).

**P5. Cost geometry.** Equal-weight blend turnover 21.6×/year = 1.08%/yr at 5 bp RT.
Selector increment at tilt 0.50: +2.6×/year = +0.129%/yr = +0.013 Sharpe at a 10%
volatility target. `IND_REV_1W` runs 156.7×/year = 7.83%/yr at 5 bp and is excluded
before contact. *Lock:* 21.6 and +2.6 were an eleven-signal, unscaled, lag-0
construction. On the declared object the control turns over **36.7×/yr** on held
weights, and the increment is **+7.85×/yr = 0.039 Sharpe** at 5 bp. The `IND_REV_1W`
figures reproduce.

**P6. A matched placebo exists, at the third attempt.** Episode-pair permutation with
swap repair reproduces the real clock exactly (223 transitions in every one of 300
draws, standard deviation 0.0) and the occupancy exactly (maximum absolute deviation
0.0000). *Lock:* reproduces on the one block it was run on, but it is **infeasible on
the declared object**: 4 draws in 1,000 at K=4, none at K=3. A fourth construction is
adopted (§8, §12.11).

---

## 2. Admission test

The programme's six refutations share four narrownesses. This study differs on all
four, and the differences are measured, not asserted. *(lock: re-measured on the
declared object; the draft's figures are in §I.2.)*

| Axis | The six | This study | Measured gap |
|---|---|---|---|
| 3 — use | sizing, or an ON/OFF switch | **selection among signals** | undefined with one signal |
| 4 — conditioned object | one book, TSMOM 12-1 on 46 instruments | **10-signal library**, 49 legs | effective rank **7.97** vs 1 |
| 1 — latent variable | states ordered by training volatility | context partition orthogonalised on log realised volatility, refitted walk-forward | pooled OOS η²(log rv) **0.142** (unorthogonalised twin 0.353) |
| 2 — time constant | 0.532 transitions/year | **12.55**/year out of sample | **×23.6** |

Admission is granted on axis 3, which is the substantive one. Axes 1, 2 and 4 are
necessary conditions for axis 3 to be posable at all; §I.5 re-checks each and finds all
three met, axis 1 weakened.

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
| `data/raw/prices/cross_asset.parquet`, series `eq_us_large` *(lock: added; the draft omitted it)* | 9,238 | 1990-01-02 → 2026-09-08 | daily |

`eq_us_large` enters only through its log 21-session realised volatility, the variable
the context is orthogonalised against and the level-B adversary's variable (§12.8).

Quality, verified rather than assumed: **zero −99.99 sentinels, zero −999 sentinels,
100.00% of sessions fully populated across all 49 industries.** `ff_rf` has 9,212
observations and no NaN; excess returns are taken against it.

**Inferential sample.** Where all ten signals exist simultaneously: **7,946 sessions,
1995-01-04 → 2026-07-31 = 31.57 years**, against 23.2 for the programme's locked
sample. *(lock)* The binding warm-up is `IND_SEASON`, whose same-month mean needs five
prior years. It is not the 1,260-session long-term reversal as the draft said, since
`IND_LTREV` is complete from 1994-12-23. The sample is unchanged. The paired
out-of-sample span on which every Sharpe comparison of this tree is read is the union
of the five test folds: **5,031 sessions, 2006-08-01 → 2026-07-31, 20.00 years**.

**Declared PIT weakness.** `available_at == period` on **100.0%** of panel rows;
`kenfrench.py` documents and argues that choice for context features. It covers the six
`xs_*` context features (`xs_dispersion_ind`, `xs_dispersion_szbm`, `xs_avg_corr`,
`xs_absorption`, `xs_absorption_chg`, `xs_breadth_63`), which are built from Ken French
panels. For the portfolio returns used here as signals the argument is weaker: the
return is computable on the evening of *t*, but the file is downloadable only weeks
later. The T-1 rule covers knowledge, not file availability. It is written here rather
than discovered later.

**Second declared PIT weakness** *(lock, from an audit)*. `fin_nfci` and
`fin_nfci_chg13w` are built from the **current NFCI vintage**: the FRED lag path, one
row per week (1,913 rows), `available_at = period + 7 days` on every row, and
`revised=True` in `regime_lab/data/universe.py`. The Chicago Fed re-estimates the NFCI
over its whole history at every weekly release, so its 1995-2026 values embed
information from later releases. The timing stamp is right; the values are not
point-in-time. **Direction: EASIER** (the partition may know more than a live desk
did). Declared, not corrected. The level-A instrument measures how much the partition
leans on these two columns (§12.6, instrument step 6).

These are the two places where this design is more optimistic than a live desk. The
other twelve context features, as the audit traced them, come from FRED daily series
that are never revised (`BAA10Y`, `AAA10Y`, `T10Y2Y`, `T10Y3M`, `DTB3`, stamped at
`period + 1 day`) or from prices known at the close (`asy_vr_5`, `asy_vr_20`,
`asy_hurst`, `mom_breadth_200d`).

**Not on disk and not sought.** No futures term structure, no options data: the carry
and volatility-carry families cannot be built. Four Ken French daily files
(`F-F_Momentum_Factor`, `F-F_ST_Reversal_Factor`, `F-F_LT_Reversal_Factor`,
`49_Industry_Portfolios`) answered HTTP 200 on 2026-09-22, verified by `curl -sIL`;
**this design uses none of them.**

---

## 4. Constructions, fixed here

**The library — 10 signals**, cross-sectional z-scores over 49 industries, clipped at
±3, gross-normalised to 1, lagged one session (`selection.library.industry_library`):
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
`asy_vr_20`, `asy_hurst`, `mom_breadth_200d`. The two `fin_nfci*` columns carry the
revised-vintage weakness declared in §3, the six `xs_*` columns the Ken French one.

**The loaders** *(lock)*. Industry returns: `selection.library.load_panel("industry_49")`.
The library's market regressor (used by `IND_LOWBETA` only) is `ff_mkt-rf` of
`load_panel("factors_5")`, at `period`, as in `scripts/measure_twosigma_power.py`.

**The partition.** K-means, **K=4**, `n_init=20`, `random_state=0`, refitted on each
training window, labels assigned out of sample by nearest centroid
(`selection.context.walk_forward_context`). Every fit and prediction runs on one thread,
so the partition is bit-reproducible *(lock)*. K ∈ {3,5,6} are declared sensitivities
and can never found a PASS. The mapping from labels to trading sessions and the lag are
fixed in §12.2.

**The selector.** A **tilt**, never a switch: weight *i* in state *k* is
`(1/n)(1 + d·m_ik)`, **floored at zero, then renormalised** to sum to one
*(lock: the draft's wording put the renormalisation first; the floor-first order is the
pre-lock scripts' and the only one that sums to one)*, with **d = 0.50** and `m_ik`
estimated on training folds only by the estimator of §12.5. Hard switching is not
tested: on the declared object it costs **3.4 times** the resolution of the tilt (MDE
0.709 against 0.211 at α 0.05) *(lock; the draft's 0.474 against 0.241 was the toy
library)*, and it is excluded on power grounds, declared here.

**The control.** The fixed equal-weight blend of the same ten signals.

**Protocol inherited, non-negotiable.** Signal at T-1, trade at T · excess of `ff_rf`,
taken at the instrument level · HAC lag 6 · walk-forward with **5 test folds of exactly
4.0 years ending 2026-07-31**, expanding training from 1995-01-04, first window **11.57
years** *(lock: anchor `end`, §12.1; the draft's 11.6 + 5 × 4.0 overshoots the sample by
eight sessions)* · daily volatility targeting at 10%, 63-session window, with the
inherited cap of 3 on the multiplier · stationary block bootstrap (Politis–Romano),
mean blocks 21 / 63 / 126, never iid · N = 49 legs.

---

## 5. Cost schedule, priced before testing

US equities: **5 bp round trip realistic, 10 bp conservative, 20 bp stress.** Charged
on `|Δw|` of the weights **actually held**, after the volatility multiplier. **The
column that decides is 5 bp** (§13.2).

*(lock: re-measured on the declared object, test folds, held weights.)* The control
turns over **36.7×/yr**, which is 13.6×/yr on unscaled weights times a multiplier that
holds the book at about 2.7 times its unscaled size. That is 1.84%/yr at 5 bp,
**0.183 Sharpe** at the 10% target and **0.233** at the 7.88% the book actually realises.
The selector's **increment** at d = 0.50, the median over eight random N(0,1) maps, is
**+7.85×/yr** (1.78 to 11.75): **0.039 / 0.079 / 0.157 Sharpe** at 5 / 10 / 20 bp at
the 10% nominal target, and **0.050 / 0.100 / 0.199** at the control's realised 7.88%,
the conversion §12.3 uses. The breakeven is **27 bp** against the uncorrected MDE of
0.211, and 33 bp against 0.262, at the nominal target; about 21 and 26 bp at the
realised sd.

The draft's reading, "cost is not the binding constraint, power is", does not survive
the re-measurement. At the stress schedule and the realised sd, the increment consumes
about 94% of the uncorrected MDE (0.199 against 0.211; three quarters at the nominal
10%). Both constraints bind.

**Cost kill rule, re-derived at the lock (§12.6).** The draft wrote: "if the fitted
map's incremental turnover exceeds 12×/year, at which the 20 bp stress schedule
consumes the whole MDE, the overlay is declared dead on cost whatever its gross gap."
The rule is kept in form, and its 12×/yr is kept as the cap. *(lock, forced by
measurement)* Its anchor is the **decision bar** T_A1 at the arm's realised sd, not
the uncorrected MDE: `K_kill = min(12, T_A1 × σ_sel × 10,000 / 20)`. A first
re-derivation anchored it to the fitted pair's uncorrected MDE at the realised sd
(about 8.3×/yr at the random-map median). Measured map by map on content-free maps
(§I.2), that version fires on **4 of 8** random maps and **3 of 8** random maps shaped
as §12.5 shapes `m`. It would kill by construction, whatever the fitted map holds. **The
adopted kill fires on 0 of 8 random maps and on 1 of 8 shaped maps**: at T ≥ 0.338 and
a realised sd near 8%, it sits at the draft's 12×/yr. Against the draft's number it is
therefore NEUTRAL. Against the draft's anchor (the uncorrected MDE, 0.241 at the
nominal 10%), and against the first re-derivation that kept that anchor, it is
**EASIER**, forced by that measurement.

---

## 6. The escalation tree

Declared in full before first measurement. A tree declared in advance is not p-hacking;
an undeclared one is. *(lock)* **All three levels are read, in the order A, B, C,
whatever each outcome.** "If A fails… B" below says why each level escapes what may
have killed the previous one, not that a level is read only on failure. The Holm
correction of §7 already counts all six locks, and the programme's rule is that it does
not stop at the first negative result. Every level's instrument is measured under the
null and its thresholds committed **before A is read** (§13.1).

### Level A — the first-moment channel (the literal transposition)

State-conditional tilt on expected signal performance, fitted on training folds,
applied out of sample.

**Falsification — both locks must hold to pass** (exact definitions in §12.6, §12.7):
- **A-1.** Paired out-of-sample Sharpe difference against the equal-weight control,
  pooled over the five test folds, net of 5 bp, in excess, at the 10% target. It must
  reach **T_A1 = max(0.338, the fitted pair's own MDE at α = 0.05/6)**, with its
  p-value (the larger of the bootstrap and HAC-6 readings) surviving Holm. A positive
  gap below T_A1 is **UNDERPOWERED and never a PASS**; a non-positive gap is a FAIL. If
  the same statistic built on the one-line volatility witness is at least as large, a
  pass is reported **DOMINATED** (§12.6).
- **A-2.** Rank-transfer statistic: in each qualifying (fold, state) cell, the Spearman
  correlation, across the ten signals, between their static-neutralised within-state
  **mean-contrast** profiles on the training fold and on the test fold *(lock, §12.5)*;
  then the mean over the cells. It must lie **above the 99th percentile** of 1,000
  matched placebo partitions (p ≤ 0.01, §12.7) and survive Holm. Otherwise it reads
  **NOT SHOWN**: no power was measured for A-2, so it cannot read FAIL (§12.7, §13.2).
  If the witness partition's statistic is at least as large, a pass is DOMINATED.

**Prior: P(pass) ≈ 12%.** Low and stated in advance: the programme has measured that
the classifier carries variance and not mean (+0.030 R² point on forward returns,
t 0.27), and A conditions a mean. A is run regardless, because it is the documented
method and skipping it would assume the answer.

### Level B — the second-moment channel

**A, if it fails, fails on the first moment. B never touches it.** The map no longer
estimates which signal earns more in state *k*; it estimates the **second-moment matrix
across signals, state by state** *(lock: about zero, so that no mean is estimated,
§12.8)*, which feeds the blend's construction: risk parity on the state-conditional
matrix. This is the channel where the classifier has **+3.93 points of incremental R²
on forward volatility, t −3.40**, established against a volatility quantile.

**Pre-computed design constraint** *(lock: re-computed)*. With signed mean pairwise
**position** correlation +0.035 over the ten signals, the share of blend variance
running through the correlation matrix is `10×9×0.035 / (10 + 3.15)` = **23.9%**; the
other **76.1% is diagonal**. The draft's 11.5% used n = 11 and the twelve candidates'
+0.013. The formula uses position correlation as a stand-in for return correlation. The
return-space share is printed by the level-B instrument before B is read. B therefore
reads the diagonal first and the correlations second, through a diagnostic that splits
any gain between them (§12.8). Writing this now prevents reading a null correlation
result as a null variance result.

**Falsification — both locks** (§12.8, §12.9):
- **B-1.** Out-of-sample covariance-forecast criterion: the state-conditional
  second-moment matrix must beat a single pooled one on the **Gaussian QLIKE** loss of
  the test-fold daily signal-return vectors, by more than the **99th percentile** of
  the matched placebo (p ≤ 0.01) and survive Holm. Otherwise it reads **NOT SHOWN**
  (no power was measured for B-1). *(lock)* The placebo does not match the partition's
  tie to volatility, so a B-1 pass must also **beat two volatility witnesses by a test,
  not by a point comparison**: quantile bins of the trailing realised volatility of
  `eq_us_large`, and quantile bins of the trailing realised variance of the
  equal-weight blend itself. Otherwise it is reported **DOMINATED**, not as a PASS
  (§12.8).
- **B-2.** Paired Sharpe difference, risk parity on the state-conditional matrix
  against risk parity on the pooled one, same gross, same target, net of 5 bp. It must
  reach **T_B2 = max(0.338, the pair's own MDE at α = 0.05/6)**. A positive gap below it
  is UNDERPOWERED, never a PASS.

**Prior: P(pass) ≈ 25%.** Higher than A because the channel has measured content;
capped because the library is already nearly orthogonal.

### Level C — the cost channel

**If B falls, both moments of the return are spent.** What remains is the ledger:
condition the **rebalancing cadence** on the context state.

*(lock: the premise re-examined.)* The control's entire cost at 5 bp is **0.183 Sharpe**
at the 10% target and **0.233** at its realised 7.88% sd, the conversion this lock uses
(§12.3). That is above the declared object's uncorrected MDE (0.211) but below the
corrected bar, which is at least 0.338 (§12.6). Removing all of it would still not reach
a decidable Sharpe gain at the level at which this tree decides. But the whole cost now
reaches about 70% of that bar (0.233 against 0.338), where the draft put it at a third,
and it clears the uncorrected bar.

**C is therefore a turnover test, not a Sharpe test** (§12.10):
- **C-1.** The state-conditional cadence (per-state rebalancing intervals chosen on
  training folds to minimise turnover under a **pooled** tracking budget equal to the
  twin's) against a **state-blind weekly twin**. The statistic is the held-weight
  turnover saving over the twin on the test folds. It must lie above the **99th
  percentile** of the same rule fitted on the matched placebo (p ≤ 0.01), survive Holm,
  and the conditional book must not track the target worse than the twin. If the gate
  fails, C-1 FAILs. If the saving does not clear the placebo, C-1 FAILs only if its
  instrument showed, before the reading, that the null resolves a saving worth 0.05
  Sharpe at 5 bp; otherwise it reads NOT SHOWN. If the same rule on the volatility
  witness saves at least as much, a pass is DOMINATED.
  *(lock: the draft compared against the daily control, which any cadence longer than a
  day beats, so its C-1 could not fail. A first per-state version of the rule, in which
  each state had to keep its own tracking within the twin's average, failed by
  construction instead: it sends the state with the most drift to daily trading, §12.10.)*
- **C-2.** Its Sharpe translation, `saving × bps / 10,000 / σ_realised`, reported **as
  a bound and never as a PASS**.

**Prior: P(pass on C-1) ≈ 40%. P(a decidable Sharpe gain) ≈ 0%**, by the arithmetic
above.

### Closure statement, if all three fall

> *(lock: restated on the declared object; T_A1 filled in at the reading)* A context
> partition refitted walk-forward on twenty macro-financial and cross-sectional
> features, orthogonalised on log realised volatility (pooled out-of-sample η² 0.14
> against it) and switching about 12.5 times a year out of sample, does not transfer
> out of sample as a selector over a ten-signal US industry library on either the first
> or the second moment, over 20.0 years of paired out-of-sample sessions (August 2006 to
> July 2026), at a decision bar of T_A1 Sharpe after correction for six primary tests
> (resolution 0.26 at α 0.05/6 on the declared object). Nor does it let the
> equal-weight book trade less than a state-blind weekly cadence at the same tracking
> distance. The book's entire cost at 5 bp, 0.18 to 0.23 Sharpe, is below that bar, so
> no Sharpe gain from cost alone is decidable on this sample.

That is the sentence this study is allowed to write. It is not "regimes do not
monetise." Where a level ends UNDERPOWERED, NOT SHOWN or UNDECIDABLE rather than FAIL,
"does not transfer" becomes "is not shown to transfer" for that level, and each other
status has its own wording (§13.6). Under this lock A-2 and B-1 cannot read FAIL, so the
first-moment and second-moment clauses can be written "does not transfer" only through
A-1 and B-2.

---

## 7. Multiple testing, computed in advance over the whole tree

**20 declared evaluations** *(lock: 21 in the draft)*:

| Group | Count |
|---|---|
| Primary: 3 levels × 2 locks | **6** |
| Sensitivity: K ∈ {3,5,6} × 3 levels, one row per (K, level) | 9 |
| Sensitivity: tilt d ∈ {0.25, 1.00}, at A-1 only | 2 |
| Sensitivity: 21-session smoothing (`PLAN.md` §f.4) × 3 levels, K = 4, d = 0.50 | 3 |
| **Total logged to `trials.parquet`** | **20** |

*(lock)* The draft counted d ∈ {0.25, 1.00} × 3 levels = 6. The tilt strength has no
meaning at level B, which is risk parity, or at level C, which is a cadence. It does not
enter A-2 either, whose ranks do not depend on d. Four of the six draft rows would have
been exact duplicates of the primary B and C configurations. The d rows are therefore 2,
not 6: **EASIER on the trial count by 4, not forced by measurement** (§12.0). The
smoothing sensitivity, pre-registered in `PLAN.md` §f.4 and absent from the draft's §7,
is reinstated (§12.13): **HARDER**, +3 rows. The row schema is §13.5.

- **Holm–Bonferroni over the 6 primaries**, family-wise α = 0.05, two-sided, strictest
  threshold **0.00833** (§13.3). On the declared object the corresponding MDE of the
  tilt is **0.262** at the random-map median (block 63; 0.264 at block 126), against
  **0.211** uncorrected. The draft's 0.338 against 0.272 was one toy draw. **The
  decision bar never goes below 0.338** (§12.6).
- The 14 sensitivities are reported and **can never found a PASS**. They still enter
  `trials.parquet` and therefore the deflated Sharpe ratio.
- Covering all 20 inferentially would put the random-map median MDE at **0.291**
  (α = 0.05/20, block 63, printed by `scripts/measure_twosigma_power.py`; 0.292 at the
  draft's 0.05/21). The figure is stated so that restricting inference to 6 is visible
  and contestable.
- **Placebo threshold is the 99th percentile, not the 95th.** Motive, from the
  programme's own measurement: an index methodology tested against 160,000 random
  combinations places the official version at the **98th** percentile. The 95th is
  exactly where a searched map lands. The extra percentile costs power and buys
  credibility.

---

## 8. The matched placebo

The null must match the real partition on **both** the clock and the occupancy, because
T3 showed that an apparent gain can travel entirely through a dimension left unmatched.

| Attempt | Transitions/yr (real **6.77**, toy sample) *(lock: the draft wrote 6.46)* | Max occupancy deviation | Verdict |
|---|---|---|---|
| 1 — permute `(length, state)` episode pairs | 4.79 — loses 29% of the clock to merged joins | 0.0000 | rejected |
| 2 — permute lengths and states independently | 6.77 | **0.3043** — occupancy destroyed | rejected |
| 3 — permute pairs, repair joins by swapping pairs | exact on one full-sample block (223 in each of 300 draws) | 0.0000 | *(lock)* **infeasible on the declared object**: completes 4 of 1,000 draws at K=4 (36 with the 1992 start), 0 at K=3 |
| 4 — **uniform over join-free orders** *(lock)* | **exact** in every block | **0.0000, exact** | **adopted** |

Construction 3 keeps `(length, state)` pairs intact and repairs same-state joins by
swapping whole pairs. Its greedy repair fails when one state holds close to half of a
block's episodes, and on the declared object that is the normal case. A "normal" state
sitting between every excursion holds 45 to 50% of a block's episodes. Retrying a fresh
permutation conditions the null on the permutations the greedy can repair, and dropping
a draw conditions it further.

Construction 4 (`analysis.placebo`, `method="uniform"`) draws the state sequence
uniformly among all sequences with the real per-state episode counts and no two equal
neighbours, then permutes each state's own lengths among its slots. It is exactly
attempt 1 conditioned on the clock, which is the null construction 3 was approximating.
Clock, occupancy and the episode multiset are exact by construction and verified on
every draw. It never fails when a join-free order exists (the real partition is one),
and it needs no retry. It is exact on all eight objects measured: K ∈ {3,4,5,6} × two
context training starts. It takes about 10 s per 1,000 draws. Blocks, seeds and the
treatment of thin blocks are fixed in §12.11.

**Three controls, not one** *(lock: and a fourth, the volatility witness, and a fifth, the NFCI point-in-time control)*.
1. **Clock placebo** — 1,000 matched partitions with no content. They supply the null
   for A-2, B-1 and C-1, and the placebo arms of A-1 and B-2 (§12.11).
2. **Equal-weight control** — the fixed blend, which is the real adversary.
3. **Beta control, the T3 lesson** — the realised beta of the A-1 selector, and of the
   B-2 state arm, against `ff_mkt-rf`, reported as a percentile of the same arm rebuilt
   on each placebo partition. A selector that beats the control while sitting at or
   above the **95th percentile** of beta has selected nothing; it has bought exposure.
   *(lock)* It is a
   downgrade-only diagnostic: it can turn a PASS into a non-PASS, never the reverse
   (§12.6). Its inputs are pinned in §12.6 ("Inputs"): the arm's daily excess returns
   net of 5 bp and `protocol.market_excess`, both on the 5,031 test sessions.
4. *(lock)* **Volatility witness, downgrade only** — each level's statistic rebuilt on a
   partition made of quantile bins of trailing realised volatility, the programme's
   standing one-line adversary. The clock placebo does not match the real partition's
   tie to volatility (η² about 0 by construction, against 0.142 pooled and 0.608 in
   fold 3), so a gain that travels through volatility would pass the placebo and be
   credited to the context. §12.6-§12.10 fix it per lock.
5. *(lock, from the second validator)* **Point-in-time control for the NFCI, downgrade
   only.** Two of the twenty features, `fin_nfci` and `fin_nfci_chg13w`, are read from
   the NFCI's current, revised vintage, which carries knowledge from after *t*. The
   T-1 rule covers knowledge, and no real-time vintage exists before 2011 (the Chicago
   Fed first published the NFCI in 2011), so the
   weakness cannot be corrected on the 2006-2011 test years. A lock that would
   otherwise PASS is therefore rebuilt on the 18-feature partition of §12.6 step 6,
   with the same K, `n_init`, seed, folds and training start, its own 1,000 uniform
   draws (seed 0), its thresholds by the same procedures and its witnesses as for the
   primary. If the rebuilt lock does not meet its own criterion, the lock is reported
   **DOWNGRADED (PIT)**. It is downgrade only, spends no trial row, and is computed only
   for a lock that would otherwise PASS. Its code is part of the instruments committed
   before A is read (§13.1). Wherever a rebuilt lock's criterion calls for Holm, the
   rebuild's p-value is compared with 0.05/6, the strictest Holm threshold; the rebuild
   never enters or changes the six-p family. Its thresholds (T′_A1, T′_B2, and the null
   percentiles and MDE′_C on its own 1,000 draws) are computed at the reading by the
   committed instrument code, only for a lock that would otherwise PASS. They are not
   part of the threshold JSON of §13.1 step 3, and their absence from it voids neither
   this control nor the reading. Because the control can only downgrade, computing them
   after the reading cannot help a PASS.

---

## 9. Effort and honest priors

| Item | Days |
|---|---|
| Library build, exclusions, unit tests | 2.0 — *done, build commit `22abbe0`* |
| Context module, walk-forward refit, matched placebo | 2.0 — *done, same commit* |
| Level A: instrument, criterion commit, reading | 1.5 |
| Level B: instrument, criterion commit, reading | 2.0 |
| Level C: instrument, criterion commit, reading | 1.0 + 0.5 *(lock: C now has a twin and a null)* |
| Write-up, prespec lock, protocol-freeze amendment | 1.5 |
| **Total** | **10.5** |

**P(the tree yields at least one inferential PASS) ≈ 30%**, being `1 − 0.88 × 0.75`
rounded down; level C contributes nothing to a Sharpe PASS by construction.
**P(the tree yields a writable result) = 1**, by §10.
*(lock)* The priors are kept as they were written, before the declared object was
measured. They are a record of expectation, not a decision input. The measurements since
(a faster, flickering out-of-sample clock, a partition less orthogonal to volatility
than disclosed, a bar that stays at 0.338) do not make them more favourable.

---

## 10. What is learned if the whole tree falls

1. The programme's signal-library dimension, measured for the first time: **8.37
   effective of 12** candidates and **7.97 of the 10** retained *(lock)* on the equity
   cross-section, against **3.86 of 11** on the 46-instrument universe. This
   retroactively explains part of the six failures: the conditioned object had roughly
   four effective dimensions, three of them the same trend factor at different horizons.
2. A narrow, dated, quantified sentence in place of a generalisation.
3. A reusable placebo, clock-exact, occupancy-exact and episode-exact, on refitted
   walk-forward partitions. Its three wrong or infeasible versions are documented so
   they are not rebuilt *(lock: the swap repair joins the two the draft had rejected)*.
4. *(lock)* The cost geometry of a selection overlay on this library, closed by
   measurement on the object actually traded. The control costs 0.18 to 0.23 Sharpe at
   5 bp. A d = 0.50 overlay on a 12.5-a-year clock adds about 7.9×/yr of held turnover:
   0.039 Sharpe at 5 bp and a breakeven of 27 to 33 bp against the re-measured MDE at
   the nominal 10%, or 0.050 and 21 to 26 bp at the realised 7.88%. **A future
   failure of a selection overlay here can partly be blamed on fees**: the draft's
   "+0.013 Sharpe, dead only above 93 bp" does not hold.
5. The negative transposability result on the 46-instrument universe, which saves the
   next attempt.

---

## 11. Blinding, and what has not been done

No strategy was built and no backtest return was read. Correlations are correlations of
weight vectors; standard errors come from series demeaned before entering `power.py`,
through `protocol.blinded_mde`, whose `observed` field is verified below 1e-9 (largest
7.4e-16); turnover is `|Δw|`. *(lock)* The build also printed the unconditional
standard deviation of demeaned net legs (control 7.88%, random-map arms 7.6 to 10.8%)
and the share of sessions on which the leverage cap binds. These are unconditional
second moments, with no mean and no per-state content, and are cited here on that
judgement, which the three audits confirmed (§14). The audit commit `3b64ab2` added
counts of cells, the smoothed clock, and the kill's base rate on content-free maps
(turnover, sd and standard errors of demeaned legs only). What distinguishes one signal
from another across states — the only
quantity that would decide the tree — has not been measured and must not be until this
document is committed.

**Open and unverified, stated plainly:** whether any signal in this library has a
regime profile different enough to be worth selecting on is **unknown**. This
pre-registration establishes that the question is now askable, on the declared object:
a library of effective rank 7.97 over 31.57 years, a walk-forward clock at 12.55 per
year out of sample over 20.0 paired years, an exact matched placebo, and a decision bar
of at least 0.338 (resolution 0.26 at α 0.05/6). It establishes nothing more.

---

## 12. Operationalisation fixed at lock

Every choice below is fixed before any statistic of the tree is read. Each carries its
reason and the **direction** it pushes relative to the draft:
- **HARDER** means a PASS needs more;
- **NEUTRAL** means it fills a gap the draft left without a default, or moves nothing;
- **EASIER** is never taken unless a measurement forces it, and then it is said.

### 12.0 Changes from the draft *(lock, added after the audits)*

Every change this text makes to the draft, what forced it, and its direction. "Choice"
means **not forced by measurement**. "Logic" means forced by an argument, not by a
figure. The sections that fix each change give the detail.

| # | change | forced by | direction |
|---|---|---|---|
| 1 | Folds anchored at `end`, first training window 11.57 y (§12.1) | arithmetic: 11.6 + 5 × 4.0 overshoots the sample by eight sessions | NEUTRAL |
| 2 | Figures restated on the declared object: rank 7.97, control 36.7×/yr held, clock 12.55/yr, η² 0.142 (§I) | measurement §I.1-§I.2 | NEUTRAL (text) |
| 3 | Tilt floored at zero, then renormalised (§4) | logic: the only order that sums to one, and the pre-lock scripts' | NEUTRAL |
| 4 | A-1 bar `T_A1 = max(0.338, fitted MDE(0.05/6))` (§12.6) | measurement §I.2: random maps reach 0.361-0.364, above 0.338 | HARDER |
| 5 | Every turnover-to-Sharpe conversion at the arm's realised sd (§12.3) | measurement §I.2: the book realises 7.88%, the cap binds 70.5% | HARDER for the cost rules, NEUTRAL for the paired tests |
| 6 | Cost kill anchored to the decision bar, `min(12, T × σ × 500)` (§5, §12.6) | measurement §I.2: the version anchored to the uncorrected MDE fires on 4 of 8 content-free maps (3 of 8 shaped) | **EASIER** than the draft's anchor (the uncorrected MDE) and than the first re-derivation, forced by that measurement; NEUTRAL against the draft's number, 12×/yr |
| 7 | Excess of `ff_rf` taken at the **instrument** level (§12.3) | choice, not forced by measurement. The reason is the level: a long-short book pays cash only on its signed net exposure, so charging `rf` on the whole book would bill capital it does not use. The stamp (`available_at`) is a separate choice | direction unknown ex ante; it moves a paired difference only through the arms' different net exposures |
| 8 | `m_ik` **static-neutralised** (§12.5) | choice, not forced by measurement (the T3 lesson) | HARDER: removes the static channel |
| 9 | `m_ik` and A-2 profiles as **mean contrasts** scaled by the pooled sd, not Sharpe contrasts (§12.5, §12.7) | logic (audit): with constant means and state-dependent volatility a Sharpe contrast equals `S_i(σ_i/σ_ik − 1)`, a persistent ranking that the placebo cannot reproduce | HARDER: removes a PASS route |
| 10 | A-2's statistic: static-neutralised profiles, where the draft said "within-state signal ranking" (§12.7) | choice, not forced by measurement | **direction unknown** ex ante |
| 11 | B's matrices are **second moments about zero**, not covariances (§12.8) | choice, made for blinding, not forced by measurement | direction unknown; the difference is of the order of the squared daily Sharpe |
| 12 | B-1's loss fixed at **Gaussian QLIKE**, where the draft said "realised-dispersion loss" (§12.8) | choice, not forced by measurement | **direction unknown** ex ante |
| 13 | B-1 witnesses: two volatility partitions, beaten by a test, not a point comparison (§12.8) | logic (audit): the placebo has η² about 0 against volatility | HARDER |
| 14 | C-1 redefined: a state-blind weekly twin, a pooled tracking budget, a placebo null and a tracking gate (§12.10) | logic, not forced by measurement: the draft's C-1 could not fail; a first per-state rule failed by construction | HARDER than the draft's C-1; against the per-state rule it removes a forced FAIL |
| 15 | Placebo construction 4, uniform over join-free orders (§8, §12.11) | measurement §I.2: the swap repair completes 4 of 1,000 draws | forced; not EASIER |
| 16 | Minimum-cell rule; UNDECIDABLE below 10 A-2 cells or above 50% fallback (§12.4, §13.4) | choice, not forced by measurement | excluding thin cells is HARDER. The two UNDECIDABLE rules would be **EASIER if they bound**, since they turn a FAIL into a verdict that spends no trial. Measured (§I.2): at K=4, 14 of 20 cells qualify and 7.4% of test sessions (372 of 5,031) sit at the abstaining row; at K=3, 5 and 6, 11/15, 15/25 and 17/30. **They do not bind on this data** |
| 17 | d sensitivities cut from 6 to 2 (§7) | logic: d has no meaning at B, C or A-2; four rows would duplicate primaries. Not forced by measurement | **EASIER** on the trial count by 4 |
| 18 | The 21-session smoothing of `PLAN.md` §f.4 reinstated as a sensitivity, 3 rows (§12.13) | `PLAN.md` pre-registered it; the audit found the first version of this lock had dropped it | HARDER on the trial count; restores the one declared check on flicker |
| 19 | Volatility witness at A-1, A-2 and C-1, downgrade only (§12.6, §12.7, §12.10) | logic (audit) | HARDER |
| 20 | NOT SHOWN, not FAIL, for a placebo lock whose power was not measured (§13.2) | logic (audit): the programme's rule that an effect below the resolution is never read as a verdict on the effect | NEUTRAL on a PASS; softens the wording of a non-pass |
| 21 | Holm: an UNDECIDABLE lock, a Δ ≤ 0, or a HAC t of the wrong sign enters with p := 1 (§13.3) | logic | HARDER |
| 22 | One non-finite placebo draw makes its lock UNDECIDABLE; no draw is dropped (§13.4) | logic (audit) | HARDER |
| 23 | The instruments of A, B and C are committed before A is read (§13.1) | logic (audit): choices resolved in code after A's reading could be steered by it | HARDER |
| 24 | The placebo arms' Δ and β are built by the reading script, not the instrument (§12.6) | blindness (audit): they are Sharpe differences computed on returns | NEUTRAL |
| 25 | NFCI's revised vintage declared as a PIT weakness, with a blind diagnostic (§3, §12.6) | audit of the store | the weakness is **EASIER**, declared and not corrected; guarded by the PIT downgrade of row 35 |
| 26 | Mean block that gives the largest MDE; two-sided α (§12.12) | choice | HARDER / NEUTRAL |
| 27 | Context training starts 1995-01-04; label source for `m` = in-sample training labels (§12.2, §12.5) | choice, measured trade-off | NEUTRAL; the known cost of the second is declared in §12.5 |
| 28 | Sensitivities never revoke a PASS; a level PASS with a K or smoothing sensitivity whose `delta ≤ 0` is labelled **PASS (not robust)** (§12.13) *(second validator)* | choice, not forced by measurement: the draft only said sensitivities "can never found a PASS" | "never revoke" is **EASIER**; the label is HARDER on the wording and changes no verdict |
| 29 | A-1: `0 < Δ < T` reads UNDERPOWERED, where the draft's A-1 said "< 0.338 ⇒ FAIL" (§12.6, §13.2); the draft's B-2 already read so *(second validator)* | the programme's rule that a gap below the MDE is never read as a verdict on the effect | **EASIER on the wording of a non-pass**: with row 20, levels A and B read FAIL only when Δ ≤ 0 or the kill fires, so the likeliest closure is "is not shown to transfer"; NEUTRAL on a PASS |
| 30 | All three levels are read whatever each outcome (§13.1) *(second validator)* | logic (audit): the draft opened B only when A failed | NEUTRAL: the Holm family stays at six whatever is read |
| 31 | The 5 bp column decides; 10 bp can only turn a positive reading into UNDECIDED (§12.6, §13.2) *(second validator)* | fills a gap: the draft priced three columns and named no decider | NEUTRAL for the deciding column (5 bp is the draft's realistic schedule); HARDER through the 10 bp line |
| 32 | Two DOWNGRADED lines at A-1 and B-2: beta at or above the placebo's 95th percentile, and a difference below the placebo's 95th percentile (§12.6) *(second validator)* | the draft's beta control made operative (§8, control 3) | HARDER |
| 33 | B-2: the fitted pair's MDE enters `T_B2`, and B-2 has its own cost kill (§12.9) *(second validator)* | as rows 4 and 6 | HARDER |
| 34 | A-1 and B-2: `p = max(p_boot, p_HAC)`, with `p_HAC := 1` on a wrong-signed *t* (§12.6) *(second validator)* | fills a gap: the draft named HAC lag 6 and a bootstrap without saying which decides | HARDER |
| 35 | **DOWNGRADED (PIT)**: a lock that would PASS must also meet its criterion on the 18-feature partition without the NFCI (§8, control 5) *(second validator)* | logic: the revised NFCI vintage is a look-ahead the T-1 rule forbids, and no real-time vintage exists before 2011 | HARDER |
| 36 | Level status: when a level's locks read UNDERPOWERED and NOT SHOWN, the level reads UNDERPOWERED (§13.2) *(second validator)* | fills a gap: two analysts could have logged different labels | NEUTRAL |

### 12.1 Sample and folds

- **Sessions:** `folds.inferential_sessions()`, the 7,946 industry sessions from
  1995-01-04 to 2026-07-31.
- **Folds:** `folds.walk_forward_folds(sessions, anchor="end")`. Five test windows of
  exactly 4.0 calendar years (1,461 days), cut-off to cut-off: (2006-07-31,
  2010-07-31], (2010-07-31, 2014-07-31], (2014-07-31, 2018-07-31], (2018-07-31,
  2022-07-31], (2022-07-31, 2026-07-31]. Test sessions 1,007 / 1,007 / 1,007 / 1,006 /
  1,004. Expanding training from 1995-01-04: 2,915 / 3,922 / 4,929 / 5,936 / 6,942
  sessions.
- **Paired sample:** the union of the five test windows, 5,031 sessions, 20.00 years,
  pooled.
- **Reason.** Equal test lengths give every fold roughly the same calendar weight.
  Every measurement of the declared object used `end`. `start` cuts fold 5 to 3.97
  years. The draft's "11.6" is a rounding of 11.57.
- **Direction: NEUTRAL.** Boundaries move by eleven days, and fold 1's training window
  is eight sessions shorter than 11.6 years.

### 12.2 The partition, the label-to-session mapping, and the lag

- **Fit:** `context.walk_forward_context(context_panel(load_context_features(),
  load_log_realised_vol()), folds, orthogonalise=True)` with the default `train_from`:
  K = 4, `n_init` 20, `random_state` 0. Standardisation and orthogonalisation slopes come
  from training rows only, and everything runs on one thread.
- **Context training start: 1995-01-04**, each fold's own first training session.
  - *Reason:* the literal reading of §4, which refits "on each training window", and
    the walk-forward's training window starts with the inferential sample. The
    partition, `m_ik`, the second-moment matrices and the cadences are then all fitted
    on the same window.
  - *Measured trade-off, neither dominant:* 1995 leaves fold 1 with a training state of
    2 episodes, and a fold-5 test window that visits two states. 1992 gives weaker
    fold-2 refit agreement (0.607) and a nearly frozen fold 1 (3.75 transitions/yr).
  - *Direction: NEUTRAL.* The 1992 start is not a sensitivity and is not logged.
- **Label → session:** `context.session_paths(wf, folds, sessions)`. Fold *f*'s path is
  its model's in-sample training labels on `fold.train(sessions)`, followed by its
  out-of-sample labels on `fold.test(sessions)`. It is in fold *f*'s own numbering, and
  each session reads the last label stamped at or before it on the feature calendar.
- **Lag:** `protocol.map_states(path_f, table, lag=1)` on each fold's own path. Session
  *t* holds the state fold *f*'s model stamped at the close of *t−1*. The first test
  session therefore holds the label that model stamped on the last training session.
  Measured: 0 of 5,031 test sessions without a label, 0 lag mismatches, first test
  session right in 5 of 5 folds.
- **The pooled aligned series** (`wf.aligned`) serves only pooled descriptive
  statistics: occupancy, η², transitions. It is never used to trade or to estimate.
  - *Reason:* a map estimated per fold lives in that fold's numbering. Lagging the
    pooled raw series crosses models at 3 or 4 of 4 fold starts, and the aligned path
    leaves fold 1's first session at the fallback.
  - *Direction: NEUTRAL.* The two paths agree on 5,030 of 5,030 comparable sessions.

### 12.3 Returns, signal legs, books

- **Excess returns:** `protocol.excess_returns(industries, protocol.risk_free(factors,
  sessions))`. `ff_rf` is stamped at `available_at` and carried forward, and subtracted
  at the instrument level: a long-short book pays cash only on its signed net exposure.
  The market series for beta is `protocol.market_excess` at `period`, a measurement made
  afterwards.
  - *Reason for the level (choice, not forced by measurement):* a cross-sectional
    long-short book is close to self-financing and pays the cash rate only on its signed
    net exposure. Subtracting `rf` from the book return as a whole would charge it for
    capital it does not use. This is the "signed" funding reading of
    `scripts/run_m3_evaluation.py`.
  - *Reason for the stamp:* `available_at` is the point-in-time reading. On today's
    store it equals `period` on every `ff_rf` row.
  - *Direction:* the stamp is NEUTRAL today. The level's direction is **unknown ex
    ante**: it moves a paired difference only through the arms' different net
    exposures (§12.0 item 7).
- **Signal leg** of signal *i* (used for `m_ik`, the second-moment matrices, and every
  placebo statistic; never for the books themselves):
  `x_i,t = Σ_n w_i,n,t · (r_n,t − rf_t) − 0.0005 · Σ_n |w_i,n,t − w_i,n,t−1|`. This is
  the signal traded alone at unit gross, net of 5 bp: `protocol.net_of_costs` applied
  to `(W_i × excess).sum(axis=1)` with the signal's own weights. It is not
  volatility-targeted.
- **Books:** `protocol.blend(signals, mix)` → `protocol.target_volatility` (10%,
  63-session window lagged one session, `MAX_LEVERAGE` = 3 on the multiplier) →
  `protocol.net_of_costs(book.returns, book.weights, bps=5)`.
- **The traded path.** An arm holds the control's equal-weight mix on every session
  before the first test session, when no map exists yet, and fold *f*'s mix on fold
  *f*'s test sessions. The unscaled blend over the 7,946 sessions is volatility-targeted
  as one path, and every statistic is read on the 5,031 test sessions.
  - *Warm-up:* the multiplier at the start of fold 1, and at each fold boundary, uses
    the trailing 63 sessions of what the book held then. That is what a desk would have
    traded, and it uses no future data. This affects at most 63 sessions per boundary,
    through a scale, never through a sign.
  - *Direction: NEUTRAL.*
- **Leverage cap kept at 3 on the multiplier**, the reading resolved in
  `docs/PROTOCOL_FREEZE.md` on 2026-09-21. The books run below target: the control
  realises 7.88%, and the cap binds on 70.5% of its test sessions. Realised sd and the
  cap share are reported for every arm.
  - *Reason:* the cap is inherited, and it is the object on which power was measured. A
    paired Sharpe difference is unaffected by a common scale.
  - *Consequence, fixed here:* **every conversion of turnover into Sharpe uses the arm's
    realised annualised sd**, not the nominal 10% (`cost_in_sharpe` is valid only at the
    volatility the book is held at).
  - *Direction: NEUTRAL* for the paired tests, **HARDER** for the cost rules.
- **Sharpe:** `√252 × mean / sd (ddof 1)` of daily net excess returns over the pooled
  test sessions.

### 12.4 Cells, and the minimum-cell rule

- A **cell** is the set of sessions of one segment (training or test) of fold *f*'s
  path whose **lagged** state is *k*.
- **Qualification** is decided on the **stamped** path of the (fold, segment) block:
  state *k* qualifies in a block if it holds **at least 63 sessions and at least 3
  episodes** there.
  - A training cell must qualify for any estimate to use it.
  - A (fold, state) pair enters A-2 only if both its training and its test cell qualify.
- **Reasons:**
  - 63 sessions is one quarter: the volatility window, and the shortest window over
    which the programme estimates any second moment. Below it, a ten-signal
    second-moment matrix is ill-conditioned (p/n above 0.16) and a Sharpe is noise.
  - Three episodes, because a one- or two-episode "state" is a stretch of calendar, a
    period effect.
  - Deciding on the stamped path means every placebo draw qualifies exactly the same
    cells: the uniform construction reproduces each block's per-state session and
    episode counts exactly.
- **A non-qualifying training cell abstains:** the equal-weight row at A, the pooled
  matrix at B, the twin's interval at C.
- **Code:** `context.qualifying_cells(paths)` (`MIN_CELL_SESSIONS` 63,
  `MIN_CELL_EPISODES` 3), at commit `3b64ab2`. A session with no label belongs to no
  state and still separates the episodes on either side of it.
- **Known before the reading, measured by `scripts/measure_twosigma_context.py`
  (section C2, labels only):**
  - K = 4: **19 of 20** training cells qualify. Fold 1's training state with 2 episodes
    abstains. It covers **372 of 5,031** test sessions (7.4% of the paired sample, 37%
    of fold 1's).
  - Three test cells hold no episode; fold 1's test state with 1 episode fails; fold 4's
    test state with 5% occupancy (about 50 sessions) fails; fold 5's test window visits
    two states.
  - **14 of the 20** (fold, state) pairs qualify on both segments and enter A-2 (2 / 3 /
    4 / 3 / 2 by fold).
  - K = 3: 15 of 15 training cells, 11 of 15 A-2 cells, 0 abstaining sessions. K = 5:
    24 of 25, 15 of 25, 34. K = 6: 28 of 30, 17 of 30, 30. The smoothing sensitivity
    (§12.13): 19 of 20, 13 of 20, 372.
  - Every one clears the UNDECIDABLE floor of §13.4. The instrument prints the counts
    again before the reading.
- **Direction:** excluding thin cells is **HARDER** (fewer cells, the same cells in the
  null). The UNDECIDABLE floors built on these counts (§13.4) would be EASIER if they
  bound; on this data they do not (§12.0 item 16).

### 12.5 `m_ik` — the level-A map, fixed before any return is read

For fold *f*, on its training segment (lagged states from its own path; the first
training session has no lagged state and is skipped):

1. Let `Q_f` be the union of fold *f*'s **qualifying** training cells (§12.4), and
   `n_k` the number of sessions of cell (*f*, *k*).
2. For each qualifying *k*: `μ_ik` = mean of the leg `x_i` over cell (*f*, *k*).
   `μ̄_i` and `σ̄_i` = the mean and the sd (ddof 1) of `x_i` over `Q_f`.
3. `D_ik = √252 × (μ_ik − μ̄_i) / σ̄_i`: how much more or less signal *i* earns in state
   *k* than over the qualifying training sessions, in units of its own pooled sd. The
   row is NaN for a non-qualifying state. By construction `Σ_k n_k D_ik = 0` exactly on
   the training window, and it stays so after step 4.
4. Row-centre: `D̃_ik = D_ik − mean_j D_jk` (across the ten signals, within the state).
5. Scale: `m_ik = D̃_ik / sqrt(mean of D̃² over all qualifying (i, k))`, so the table
   has unit root-mean-square.
6. Tilt: `protocol.tilt_table(m, d=0.50)`, which gives `(1/10)(1 + d·m_ik)`, floored at
   zero, then renormalised. A NaN row holds 1/10 on every signal.
7. If no state qualifies, or the root-mean-square is zero, fold *f* holds the control.

**Reasons.**
- **A mean contrast in units of the pooled sd**, not a Sharpe contrast *(lock, from an
  audit)*. The ten legs run at different volatilities at unit gross, and dividing by
  each signal's own pooled sd `σ̄_i` puts them on one scale. Dividing each **state's**
  mean by that state's own sd would not be a first-moment quantity. With constant means
  and state-dependent volatility, a Sharpe contrast is `S_i(σ_i/σ_ik − 1)`: each state's
  profile becomes plus or minus the ranking of the signals' unconditional Sharpes. That
  ranking persists from training to test, so A-2 would pass on any partition tied to
  volatility, and the placebo (η² about 0 against volatility) cannot reproduce it. The
  fitted map would also become a volatility-managed static tilt, the "variance, not
  mean" channel the programme has already measured. The mean contrast has expectation
  zero under constant means whatever the volatility. **HARDER**: it removes a PASS
  route.
- **Static component neutralised** (steps 2-3), which is the T3 lesson. A map that tilts
  toward the signals that are good in every state can pass A-1 without selecting
  anything. Centring each signal on its own mean over the qualifying training sessions
  leaves only its state profile, and the occupancy-weighted profile is exactly zero on
  the training window. Centring on `Q_f` rather than on every training session leaves no
  residual static tilt from non-qualifying states. The static tilt is not zero on the
  test window, whose occupancies differ. *Choice, not forced by measurement* (§12.0
  item 8).
  - The random-map measurement prices this: "m demeaned" gives an MDE of 0.238 against
    0.262 at α 0.05/6.
  - It also prices the alternative left unchosen, a static K = 1 twin arm: 0.244, which
    would add a seventh primary.
- **Unit root-mean-square** (step 5), so that d = 0.50 means what the N(0,1) stand-in
  assumed. The scaling is one per fold, not one per state: a state in which the signals
  barely differ in training gets a weak tilt, not a noise-amplified one. The shape
  steps 2-5 give `m` puts the whole root-mean-square in the part that varies by state.
  The same eight N(0,1) draws given that shape (`scripts/measure_twosigma_power.py`,
  "m shaped") turn over +8.55×/yr more than the control at the median (6.19 to 13.59),
  not +7.85, with an MDE(0.05/6) of 0.260 at block 63 and 0.269 at the largest block.
  The fitted map's own MDE enters T_A1 (§12.6), so the bar does not rely on the
  stand-in.
- **Row-centring** (step 4) makes the pre-floor sum exactly one. It is not neutral after
  renormalisation: adding a constant `c` to row *k* scales that state's effective d by
  `1/(1 + d·c)`. It is kept because it is the only centring under which d = 0.50 means
  the same thing in every state. The floor binds only where `m_ik < −2` at d = 0.50.
- **Label source:** fold *f*'s in-sample training labels. They are the only labels that
  exist for fold 1, and they come from the same model, in the same numbering, as the
  test labels the map is applied to.
  - *Known cost, declared and not corrected:* in-sample labels run at 6.41
    transitions/yr against 12.55 out of sample, so the map is estimated on smoother
    states than it is applied to.
  - Correcting this would be a modification of the hypothesis.
- **No other estimator** (an unscaled mean, a Sharpe contrast, HAC t, James–Stein
  shrinkage, within-state rank) is computed or logged.

**Direction: HARDER**: the static channel is removed, the route through state-dependent
volatility is removed, and the minimum-cell rule removes the thinnest cells.

### 12.6 A-1, and the cost-kill rule

**Arms.** The selector (fold *f*'s `tilt_table` via `map_states` on fold *f*'s path, on
the traded path of §12.3) against the control. Both are net of 5 bp. A **fallback
session** is a test session on which the selector holds the equal-weight row: its lagged
state is missing, or its lagged state's training cell does not qualify, or its fold holds
the control (§12.5 step 7). Measured from labels (§12.4): 372 of 5,031 at K = 4.

**Inputs, pinned** *(lock, from an audit)*. Every A-1 quantity — `blinded_mde`,
`paired_hac_t`, σ, the Sharpes, `realised_beta`, and the placebo arms' Δ and β — is
computed on the two arms' **daily excess returns net of 5 bp** (the 10 and 20 bp series
only for verdict line 4 and the reported cost columns). The full 7,946-session path is
first volatility-targeted and costed; the series are then **restricted to the 5,031 test
sessions**, in date order. Nothing before 2006-08-01 enters a statistic, so identical
pre-test legs cannot shrink a standard error.
- σ = √252 × sd (ddof 1) of that series.
- Turnover is `protocol.annual_turnover(book.weights.loc[test])`: restricted **first**,
  so the entry change on 2006-08-01 is not counted, as in
  `scripts/measure_twosigma_power.py`.
- β = `protocol.realised_beta(net5.loc[test], protocol.market_excess(factors,
  sessions).loc[test])`.

**Statistic.** `Δ_A1 = SR(selector) − SR(control)` over the 5,031 pooled test sessions.

**Instrument, measured before the reading and committed with its thresholds (§13.1).**
It prints no mean and no Sharpe, of the real selector, of the control or of any placebo
arm. To build the selector's legs it computes `m` from returns on the training folds
(§12.5); it never prints `m` or any per-state quantity.
1. `protocol.blinded_mde(selector, control, mean_block=b, draws=2000, seed=0)` for
   b ∈ {21, 63, 126}, giving `MDE(α, b) = SE_b × mde_z(α)`.
2. **Threshold:** `T_A1 = max(0.338, max_b MDE(0.05/6, b))`.
   - 0.338 is the draft's bar, one toy draw rescaled, kept as a floor.
   - Reference, not a term of the formula: the declared object's random-map median at
     α 0.05/6 is 0.262 at block 63, 0.263 at block 21 and 0.264 at block 126. It is
     below the floor and cannot bind.
   - The fitted pair's own MDE is included because the random tables reach 0.361 (block
     63) and 0.364 (block 21), above 0.338. A fitted map with a wider tracking error
     than the median would otherwise be read against a bar below its own resolution.
   - **SE\*** is the standard error at the block, among 21 / 63 / 126, that gives the
     largest MDE(0.05/6), which is the block with the largest standard error.
3. **Kill turnover** *(lock, forced by measurement, §5)*:
   `K_kill = min(12.0, T_A1 × σ_sel × 10,000 / 20)`, where `σ_sel` is the selector's
   realised annualised sd on the test sessions (Inputs), and
   `ΔT = annual_turnover(selector held weights on the test sessions) −
   annual_turnover(control held weights on the test sessions)`.
   - **The kill fires if `ΔT > K_kill`.**
   - At T_A1 ≥ 0.338 and σ_sel near 8%, K_kill is the draft's 12×/yr.
   - Base rate on content-free maps, measured before the lock (§I.2): **the adopted
     kill fires on 0 of 8 random N(0,1) maps, and on 1 of 8 once `m` has §12.5's
     shape.** The first re-derivation, anchored to MDE(0.05, 63), fired on 4 of 8 and
     3 of 8 and is not used: it would have killed by construction.
   - The instrument prints this verdict line, and that is intended. It reads turnover, a
     demeaned-leg MDE and a realised sd only, and reveals no mean.
4. **Placebo arms: built by the reading script, after the instrument commit, not by the
   instrument.** They are Sharpe differences computed on returns. On each of the 1,000
   placebo draws (§12.11), `m` is re-estimated on the draw's training labels and the
   selector rebuilt; its paired difference `Δ^(j)` against the same control and its
   realised beta `β^(j)` are recorded. The percentile rules of verdict lines 6-7 are
   fixed by this text, so no threshold needs committing.
5. **Counts:** abstaining (fold, state) rows, fallback sessions, qualifying cells, the
   cap share, and realised sd.
6. **NFCI diagnostic** *(lock, from an audit; §3)*, no trial row: the walk-forward
   partition refitted on the 18 features without `fin_nfci` and `fin_nfci_chg13w`, with
   the same K, `n_init`, seed, folds and training start. The instrument prints its
   out-of-sample label agreement with the declared partition (per fold,
   `context.align_labels` on the test sessions, then the share of the 5,031 test
   sessions that agree), its out-of-sample transitions per year, and its pooled η²
   against log rv. Counts only, no return.

**Reading.**
- `Δ_A1` at 5 bp decides. It is also computed at 10 bp (verdict line 4) and 20 bp
  (reported).
- `p_boot = 2(1 − Φ(|Δ_A1| / SE*))`.
- `p_HAC = 2(1 − Φ(|t|))`, with `t = protocol.paired_hac_t(selector, control, lags=6)`
  on the daily net differences (Inputs). *(lock)* If the sign of *t* differs from the
  sign of `Δ_A1`, `p_HAC := 1`: the HAC test is on the mean difference of two arms that
  run at different realised sd, and a *t* of the wrong sign is no evidence for Δ.
- **`p_A1 = max(p_boot, p_HAC)`.** The two coincide only at equal realised volatility,
  which the cap breaks, and requiring both removes the choice between them. If
  `Δ_A1 ≤ 0`, `p_A1 := 1` in Holm (§13.3).
- The beta percentile is `protocol.placebo_percentile(β_selector, β_placebo)`, and the
  difference percentile is `placebo_percentile(Δ_A1, Δ_placebo)`. Both are NaN if any
  draw is not finite (§13.4).
- **Volatility witness** *(lock, from an audit; §8 control 4)*, downgrade only, no trial
  row, printed after the real statistic. Witness W1 of §12.8 (K quantile bins of log
  rv, cut-offs from fold *f*'s training sessions, stamped at *t*, lagged one session)
  runs through the same code as the real partition: qualification on its stamped path,
  the estimator of §12.5 fitted per fold, the same selector and traded path. It gives
  `Δ_W`.
- Per-fold components of `Δ_A1` are reported as diagnostics (§12.14).

**Verdict — the first line that applies:**
1. **UNDECIDABLE**: any instrument or reading value is non-finite, including any one of
   the 1,000 placebo draws of Δ or β; a paired leg is missing on a test session; or the
   selector is at the fallback on more than half of the test sessions.
2. **FAIL (cost)**: the kill fired. The reading is still made and reported; it cannot
   pass.
3. **FAIL**: `Δ_A1 ≤ 0` at 5 bp.
4. **UNDECIDED**: `Δ_A1 > 0` at 5 bp and `≤ 0` at 10 bp.
5. **UNDERPOWERED**: `0 < Δ_A1 < T_A1`, or `Δ_A1 ≥ T_A1` with `p_A1` not rejected by
   Holm (§13.3). Never a PASS.
6. **DOWNGRADED (beta)**: the beta percentile is ≥ 0.95.
7. **DOWNGRADED (not conditional)**: the difference percentile among the placebo arms is
   < 0.95. The gain is then no larger than what a content-free partition with the same
   clock produces.
8. **DOMINATED (volatility)**: `Δ_W ≥ Δ_A1`. The one-line volatility partition does at
   least as well, so the gain cannot be credited to the context.
9. **DOWNGRADED (PIT)** *(lock, second validator)*: the same lock rebuilt on the
   18-feature partition without the NFCI (§8, control 5) does not meet this lock's
   criterion (lines 1-8 applied to the rebuild give anything but PASS).
10. **PASS.**

**Direction: HARDER.** The p-value is the larger of two and loses a wrong-signed *t*, the
fitted MDE enters the bar, there are four downgrade diagnostics, and the floor stays at
0.338 although the declared object resolves 0.264 at the median. Lowering it would be
EASIER and was not forced. The one EASIER move, the kill's anchor, is forced by the
measurement in §I.2 and said in §5 and §12.0 item 6.

### 12.7 A-2 — rank transfer

- **Per qualifying (fold *f*, state *k*)**, a pair whose training and test cells both
  qualify (§12.4; 14 at K = 4):
  - the training profile is `D_·k` of §12.5, step 3;
  - the test profile is computed identically on fold *f*'s test sessions (lagged states
    along fold *f*'s path, first test session included) *(lock, from an audit)*: let
    `Q^te_f` be the union of fold *f*'s qualifying **test** cells; `μ^te_ik` = mean of
    `x_i` over the test cell (*f*, *k*); `μ̄^te_i` and `σ̄^te_i` = mean and sd (ddof 1)
    of `x_i` over `Q^te_f`; `D^te_ik = √252 × (μ^te_ik − μ̄^te_i) / σ̄^te_i`;
  - `ρ_fk` is the Spearman correlation across the ten signals, average ranks for ties
    (`scipy.stats.spearmanr`). If either side has zero rank variance, `ρ_fk = 0`. (A
    fold whose test window has one qualifying cell gives that cell a zero test profile,
    hence `ρ = 0`.)
- **Statistic:** `R = plain arithmetic mean of ρ_fk` over the qualifying cells.
- **Why the plain mean.** Every `ρ_fk` is a rank correlation across the same ten
  signals, bounded in [−1, 1] whatever the cell size, so no cell dominates by scale. The
  placebo reproduces every cell's size exactly, so any weighting would be valid. The
  plain mean is the draft's "pooled over 5 folds × 4 states" in its plainest reading. A
  session-weighted mean would let the two or three largest cells decide.
  **NEUTRAL.**
- **Why static-neutralised profiles, not raw ranks.** A signal that ranks high in every
  state lifts both the real statistic and every placebo draw. That is persistence, not
  selection, and it is not what A asks. The profiles are those of the map. *Choice, not
  forced by measurement; direction unknown ex ante* (§12.0 item 10): the null carries
  the same persistence, and which reading gives the lower p cannot be known before the
  data.
- **Why mean contrasts, not Sharpe contrasts:** §12.5. A Sharpe contrast would let any
  partition tied to volatility pass through a persistent unconditional ranking that the
  placebo cannot reproduce. **HARDER** (§12.0 item 9).
- **Null:** the 1,000 uniform placebo draws of §12.11. On each draw, both profiles are
  recomputed through the same code with the draw's labels, training and test, in place
  of the real ones, giving `R^(j)`.
- **p-value:** `p_A2 = protocol.placebo_p_value(R, R^(·)) = (1 + #{j : R^(j) ≥ R}) /
  1,001`. Ties count against the real partition.
- **The lock holds** if `p_A2 ≤ 0.01` (at most 9 of 1,000 draws at or above R) **and**
  Holm rejects (§13.3). If A-2 ranks first in Holm, its bar is `p ≤ 0.00833`, at most 7
  draws.
- **Volatility witness** *(lock, from an audit)*, downgrade only, no trial row, printed
  after `R`: `R_W`, the same statistic on witness W1 of §12.8, with its cells qualified
  on W1's stamped path and the same estimator, fitted per fold.
- **Verdict:**
  - **UNDECIDABLE**: `R` or any `R^(j)` is not finite, the placebo check is not exact, or
    fewer than 10 cells qualify (half of the 5K cells at a K sensitivity);
  - **PASS**: the lock holds and `R_W < R`, and the lock rebuilt on the 18-feature
    partition (§8, control 5) holds with its own witness;
  - **DOWNGRADED (PIT)**: the lock holds, `R_W < R`, and the lock rebuilt on the
    18-feature partition does not hold with its own witness;
  - **DOMINATED**: the lock holds and `R_W ≥ R`;
  - **NOT SHOWN**, otherwise. No power was measured for A-2 against any declared
    alternative, so under this lock A-2 **never reads FAIL**. A FAIL would need, before
    the reading, a power of at least 0.80 measured blind against an effect planted
    into demeaned legs. That needs a blindness ruling this lock does not give. Any such
    instrument would be an amendment in `docs/PROTOCOL_FREEZE.md`, committed before
    any reading.
- **Instrument before the reading:** the number of qualifying cells, and the null's
  50th, 95th and 99th percentiles and `q99 − q50`, the resolution quoted in a NOT SHOWN.
  Nothing is printed per cell. The null is computed on real returns under content-free
  partitions; its percentiles do not read the real partition's labels.

### 12.8 B-1 — the covariance forecast, with the volatility witness

- **Second-moment matrices, about zero:**
  - `Σ_f = (1/n) Σ_t x_t x_tᵀ` over fold *f*'s training sessions that have a lagged
    state;
  - `Σ_fk` is the same over the training cell (*f*, *k*), for qualifying *k*, and
    `Σ_f` otherwise;
  - `x_t` is the 10-vector of signal legs (§12.3);
  - sample estimates, no shrinkage.
  - A qualifying cell whose matrix is not positive definite (smallest eigenvalue ≤
    1e-12 × trace) falls back to `Σ_f` and is counted. A pooled matrix that is not
    positive definite makes the level UNDECIDABLE.
  - *Reasons:*
    - B must not estimate a mean. About zero, the difference from a centred covariance
      is of the order of the squared daily Sharpe.
    - No shrinkage, because the test is placebo-relative. The estimation noise of a
      63-session cell is borne equally by every placebo draw, and a shrinkage target
      would be one more choice to make.
    - *Choice, made for blinding, not forced by measurement; direction unknown* (§12.0
      item 11).
- **Loss:** the Gaussian QLIKE of each test session *t* of fold *f*,
  `ℓ_t(Σ) = log det Σ + x_tᵀ Σ⁻¹ x_t`.
  - *Reason.* The daily outer product `x_t x_tᵀ` is an unbiased but very noisy proxy of
    the true covariance. Against such a proxy, QLIKE, like the squared-error Frobenius
    loss, ranks forecasts consistently (Patton 2011; Laurent, Rombouts and Violante
    2013).
  - Unlike a Frobenius distance, QLIKE is scale-free. It measures relative error, so a
    quiet signal counts as much as a loud one. It needs no realised-covariance window:
    no window to choose, no overlap.
  - *Choice, not forced by measurement; direction unknown ex ante* (§12.0 item 12). The
    draft said "realised-dispersion loss" and named none.
- **Statistic:** `G = mean over the 5,031 test sessions of [ℓ_t(Σ_f) − ℓ_t(Σ_f,k(t))]`,
  where *k(t)* is the lagged state along fold *f*'s path. G is positive when the
  state-conditional forecast is better.
- **Null and p-value:** G on the same 1,000 placebo draws, with every matrix
  re-estimated on the draw's training labels. `p_B1 = protocol.placebo_p_value(G,
  G^(·)) = (1 + #{G^(j) ≥ G}) / 1,001`. The lock holds if `p_B1 ≤ 0.01` and Holm
  rejects.
- **Two witnesses, downgrade only** *(lock, from an audit)*. The uniform placebo has η²
  about 0 against volatility, while the real partition has 0.142 pooled and 0.608 in
  fold 3. Against the placebo alone, B-1 would pass for almost any partition tied to
  volatility. The witnesses are therefore the real test of the context's content, and
  they are read by a test, not by a point comparison.
  - **W1**, quantile bins of `log_rv`, the 21-session log realised volatility of
    `eq_us_large` (the variable the context is orthogonalised against): read as of each
    industry session; K bins with cut-offs at the quantiles of `log_rv` over fold *f*'s
    training sessions (quartiles at K = 4); stamped at *t* and lagged one session like
    the context. This is the programme's standing one-line adversary, which has beaten
    every regime device it was set against.
  - **W2**, quantile bins of the log trailing realised variance of the equal-weight
    blend's unscaled leg: `x̄_t = (1/10) Σ_i x_i,t`, the mean of the ten signal legs of
    §12.3, and `v_t = mean of x̄²` over the 63 sessions ending at *t* (about zero, like
    B's matrices). K bins of `log v_t` with cut-offs at its quantiles over fold *f*'s
    training sessions; stamped at *t* and lagged one session. The first 62 sessions of
    the sample have no `v_t` and hold no bin. *Reason:* the context
    carries `xs_dispersion_ind`, `xs_avg_corr`, `xs_absorption` and
    `xs_absorption_chg`, trailing realised second moments of the same 49 industries,
    and they are not orthogonalised. The object forecast is the second moment of
    long-short industry legs. A B-1 pass could be nothing more than the persistence of
    realised dispersion in the signals themselves.
  - Each witness gets second-moment matrices by bin, with the same estimator, the same
    qualification on its own stamped path, and the same fallbacks.
  - For each witness W, the daily loss differential over the 5,031 test sessions is
    `d^W_t = ℓ_t(Σ_W,k_W(t)) − ℓ_t(Σ_f,k(t))`, positive when the context forecasts
    better. `p_W` is the one-sided p-value that its mean is positive: the **larger** of
    the HAC-6 reading, `1 − Φ(t_HAC)`, and the stationary-bootstrap reading
    `1 − Φ(mean / SE_boot)`. `SE_boot` is the sd of the resampled mean under
    `analysis.bootstrap.stationary_indices`, mean blocks 21 / 63 / 126, 2,000 draws,
    seed 0, at the block that gives the **largest** SE.
  - **A B-1 that holds is reported DOMINATED unless `p_W1 ≤ 0.05` and `p_W2 ≤ 0.05`.**
    **HARDER.**
  - *Known bias, declared:* the context's small cells carry a larger inverse-Wishart
    penalty (about n/(n − p − 1), 1.21 at n = 63, p = 10) than the witnesses' equal
    quantile bins. That tilts the comparison toward DOMINATED, which is conservative.
- **Verdict:**
  - **UNDECIDABLE**: `G` or any `G^(j)` is not finite, the placebo check is not exact,
    or a pooled training matrix is not positive definite (§13.4);
  - **PASS**: the lock holds and neither witness dominates, and the lock rebuilt on the
    18-feature partition (§8, control 5) holds with its own witnesses;
  - **DOWNGRADED (PIT)**: the lock holds, neither witness dominates, and the lock rebuilt
    on the 18-feature partition does not hold with its own witnesses;
  - **DOMINATED**: the lock holds and a witness dominates;
  - **NOT SHOWN**, otherwise. As at A-2, no power was measured for B-1, so under this
    lock B-1 **never reads FAIL** (§12.7).
- **Diagnostic, never decisive, no trial row:** `G_diag` for a state-conditional
  diagonal, with per-state variances and the pooled correlation matrix. It tells whether
  any gain runs through variances or through correlations. This is the draft's
  "diagonal first".
- **Instrument before the reading:**
  - qualifying cells and fallbacks;
  - the null's 50th, 95th and 99th percentiles and `q99 − q50`;
  - the **return-space** share of blend variance through correlations, from fold 5's
    pooled training matrix. This is an unconditional second moment, the input of the
    comparison arm, silent on the conditional side. It was not computed before this
    lock. This text rules it blind-safe from the lock on.
- **Direction:** the estimator and the loss fill a gap the draft left, as choices of
  unknown direction (§12.0 items 11-12). The two witnesses and their test are
  **HARDER**.

### 12.9 B-2 — risk parity, state-conditional against pooled

- **Arms:**
  - the **pooled** arm holds `protocol.risk_parity_mix(Σ_f)` on every test session of
    fold *f*;
  - the **state** arm holds `risk_parity_mix(Σ_fk)` for lagged state *k*, with
    `risk_parity_mix(Σ_f)` as the fallback (`map_states(..., fallback=...)`);
  - instrument weights come from `blend`, and the state arm is rescaled session by
    session to the pooled arm's gross (`protocol.match_gross`);
  - both are then volatility-targeted and netted of 5 bp on the traded path of §12.3.
- **Inputs:** pinned exactly as in §12.6 ("Inputs"), with the state arm in the
  selector's place and the pooled arm in the control's: daily excess returns net of
  5 bp, restricted to the 5,031 test sessions after the full path is targeted and
  costed; σ, turnover (restricted first) and β as there.
- **Statistic:** `Δ_B2 = SR(state) − SR(pooled)` over the pooled test sessions.
- **Threshold:** `T_B2 = max(0.338, max_b MDE_B2(0.05/6, b))` on the pair's demeaned
  legs, measured by the instrument; SE\* at the block that gives the largest
  MDE_B2(0.05/6), as in §12.6.
- **p-value:** `p_B2 = max(p_boot, p_HAC)` as in §12.6, with `p_HAC := 1` when the sign
  of *t* differs from that of `Δ_B2`, and `p_B2 := 1` in Holm when `Δ_B2 ≤ 0`.
- **Kill rule** *(lock, from an audit: the comparator is B-2's own pair)*:
  - `ΔT = protocol.annual_turnover(state arm held weights restricted to the 5,031 test
    sessions) − protocol.annual_turnover(pooled arm held weights restricted to the same
    sessions)`. It is **not** measured against the equal-weight control: pooled risk
    parity overweights the slow legs and turns over far less than the control.
  - `K_kill = min(12.0, T_B2 × σ_state × 10,000 / 20)`, where σ_state is √252 × sd
    (ddof 1) of the state arm's 5 bp net daily excess returns on the test sessions.
    Anchored to the decision bar, as at A-1 (§12.6), and not to the pair's own
    uncorrected MDE. The state and pooled legs are nearly identical, so that MDE is
    small, and a kill scaled to it would fire almost automatically.
  - The kill fires if `ΔT > K_kill`.
- **Fallback at B-2:** a test session on which the state arm holds
  `risk_parity_mix(Σ_f)` because its lagged state is missing, its lagged state's
  training cell does not qualify, or its `Σ_fk` failed the positive-definite check.
  The >50% UNDECIDABLE line of §12.6 applies to that count.
- **Beta and difference downgrades:** as in §12.6, with the placebo arms (risk parity
  on each draw's state matrices) built by the reading script, not by the instrument.
- **Volatility witness:** `Δ_W,B2`, the state arm rebuilt on witness W1's bins (§12.8)
  through the same code, against the same pooled arm. Downgrade only, no trial row,
  printed after the real statistic.
- **Verdict lines:** those of §12.6, in the same order, with `Δ_B2`, `T_B2`, `p_B2`,
  this kill and `Δ_W,B2` in place of A-1's.
- **Instrument before the reading:** as at A-1: the demeaned-leg MDEs and `T_B2`,
  `ΔT`, `K_kill` and the kill line, σ_state, the cap share, fallbacks and cells. It
  computes the state matrices and never prints them or any per-state quantity.
- **Direction: NEUTRAL** against the draft's B-2, plus **HARDER** (the kill, the
  downgrades, the witness, the fitted-MDE term).

### 12.10 C-1 and C-2 — cadence

- **Target:** the control's held weights `w*_t` (after the multiplier).
- **Cadence books** *(lock: phase, counter and NaN handling pinned after an audit)*:
  - a book is defined by an interval `h_t ≥ 1` on each session;
  - it starts on the first session on which `w*` is defined (the first session after
    the volatility target's warm-up); that session is a rebalance, with counter 0;
  - it rebalances on *t* when the number of sessions since its last rebalance is
    **≥ `h_t`**, so a shorter interval that takes over after a longer count rebalances
    at once;
  - on a rebalance session it holds `w*_t`; otherwise it keeps `w_{t−1}`;
  - its return is `w_t · (r_t − rf_t)`, net of 5 bp on `|Δw|`;
  - a **state-blind book at h** has `h_t = h` on every session; h = 1 is the control;
  - **menu H = {1, 2, 3, 5, 8, 13, 21}** *(lock, from an audit: {1, 5, 21} was too
    coarse for any rule to move)*.
- **Tracking distance:** `δ_t = Σ_n |w_n,t − w*_n,t|`, on weights only. `δ_t`, and
  every mean of it, is taken over the sessions on which both `w_t` and `w*_t` are
  defined.
- **Twin:** the state-blind book at h = 5.
- **Training runs:** fold *f*'s training quantities are read on the books above, over
  the same full path, restricted to fold *f*'s training sessions that have a lagged
  state along fold *f*'s path. No book is restarted.
- **Per-state intervals, on training only (fold *f*)** *(lock, from an audit: a pooled
  budget)*:
  - for each state *k* whose training cell qualifies: `π_k` = the share of those
    training sessions whose lagged state is *k*; `T_k(h)` = the mean of
    `Σ_n |Δw_n,t|` of the state-blind book at h over them; `δ_k(h)` = the mean of its
    `δ_t` over them;
  - budget: `B_f = Σ_k π_k δ_k(5)` over the qualifying states. A non-qualifying state
    takes h = 5 and so spends exactly its share of the twin's tracking;
  - for λ ≥ 0, `h_k(λ) = argmin_{h ∈ H} [T_k(h) + λ δ_k(h)]`, ties to the larger h;
  - λ\* is the smallest λ with `Σ_k π_k δ_k(h_k(λ)) ≤ B_f`: λ\* = 0 if λ = 0 meets
    the budget; otherwise λ_hi starts at 1 and doubles until the budget holds (it
    holds once every `h_k` = 1, since `δ_k(1) = 0`), followed by 100 bisection steps
    between the last failing λ and λ_hi. The intervals are `h_fk = h_k(λ_hi)`;
  - a training check follows: a cadence book whose interval is `h_f,k(t)` along fold
    *f*'s lagged training path (h = 5 for a missing or non-qualifying state), started
    like every book on the first session on which `w*` is defined, is read on fold
    *f*'s training sessions. **If its turnover there is not strictly below the twin's,
    or its mean δ there exceeds the twin's, fold *f* holds the twin** (every h = 5). The
    training saving is therefore ≥ 0 by construction.
- **Conditional arm:**
  - it runs as the twin (h = 5) on every session before the first test session;
  - on fold *f*'s test sessions, `h_t = h_f,k(t)`, where *k(t)* is the lagged state
    along fold *f*'s path, and h = 5 for a missing or non-qualifying state;
  - the counter carries across fold boundaries.
- **Statistic:** `S_C = annual_turnover(twin held weights on the test sessions) −
  annual_turnover(conditional held weights on the test sessions)`, each restricted to
  the test sessions **first**, in ×/yr.
- **Gate:** the conditional arm's mean `δ_t` over the test sessions must not exceed the
  twin's.
- **Null:** `S_C` on the 1,000 placebo draws. On each draw the whole rule (λ, the
  intervals, the training check) is re-run on the draw's training labels, and the arm
  is run on the draw's test labels. `p_C1 = protocol.placebo_p_value(S_C, S^(·))`. The
  re-optimisation on every draw absorbs the rule's in-sample optimisation bias.
- **C-1 holds** if `p_C1 ≤ 0.01`, Holm rejects, and the gate holds.
- **Resolution and power, before the reading** *(lock, from an audit)*. From the null,
  the instrument prints q50, q95, q99 and **`MDE_C = q99 − q20`**: the shift of the null
  that puts 80% of it above its 99th percentile, under a location shift. The saving
  worth 0.05 Sharpe at 5 bp is **`S* = 0.05 × σ_twin × 10,000 / 5`**, where σ_twin is
  the twin's realised annualised sd of its 5 bp net daily excess returns on the test
  sessions (about 7.9×/yr at the control's 7.88%). **If `MDE_C ≤ S*`**, the instrument
  has a power of at least 0.80 against a saving of S\*, and a C-1 that does not hold
  reads FAIL; otherwise it reads NOT SHOWN.
- **Volatility witness** *(lock, from an audit)*: `S_W`, the same rule on witness W1 of
  §12.8, with its cells qualified on W1's stamped path and the same λ procedure, per
  fold. Downgrade only, no trial row, printed after `S_C`. Held-weight drift depends on
  volatility through the multiplier, whose cap does not bind in volatile states, so a
  saving could travel through volatility alone.
- **Verdict:**
  - **UNDECIDABLE**: `S_C` or any `S^(j)` is not finite, the placebo check is not
    exact, or a leg is missing on a test session;
  - **FAIL**: the gate fails; or the lock does not hold and `MDE_C ≤ S*`;
  - **NOT SHOWN**: the lock does not hold and `MDE_C > S*`;
  - **DOMINATED**: the lock holds and `S_W ≥ S_C`;
  - **PASS**: the lock holds and `S_W < S_C`, and the lock rebuilt on the 18-feature
    partition (§8, control 5) holds with its own witness and gate;
  - **DOWNGRADED (PIT)**: the lock holds, `S_W < S_C`, and the lock rebuilt on the
    18-feature partition does not hold with its own witness and gate.
- **Instrument before the reading** *(lock, from an audit)*. At level C the instrument
  builds only the state-blind books (the twin and the menu) and the conditional arms of
  the 1,000 placebo draws. It prints the twin's held turnover and mean δ on the test
  sessions, σ_twin, the null's 50th, 95th and 99th percentiles, `MDE_C` and S\*. **It
  does not build the conditional arm on the real partition**, and prints none of its
  turnover, its δ, `S_C`, the gate, or the intervals `h_fk`. These are first computed by
  the reading script. Level C is decided in position space alone, so any per-state
  turnover, δ or cadence rule on the real partition is forbidden until the level-C
  instrument is committed.
- **Reasons.**
  - Against the daily control, any cadence longer than a day cuts turnover. The draft's
    C-1 could not fail, and a placebo partition would have passed it.
  - *Why a pooled budget.* A first version of this rule gave each state the longest
    interval whose own training tracking stayed within the twin's pooled mean δ̄_f.
    That mean is the occupancy-weighted average of the per-state `δ_k(5)`, so whenever
    the states differ, at least one qualifying state lies above it. With tracking
    rising in h, that state got h = 1 and traded daily exactly where the drift was
    highest. A partition with cadence content was then penalised more than a random
    one, whose states sit near the average: the rule failed by construction, and the
    closure sentence of §6 would have been written by it. An audit's synthetic
    simulation found the same; its figures are not committed and are not quoted here.
    The pooled budget lets a state with more drift trade more often and a calmer one
    less, as long as the book as a whole tracks no worse than the twin on training.
  - The placebo asks whether any saving comes from the state or from the mechanics of
    the rule. The gate refuses a saving bought with tracking.
- **C-2:** the bound `S_C × bps / 10,000 / σ_twin` at 5, 10 and 20 bp. It is reported
  beside the control's whole cost, never a PASS, and **`p_C2 := 1`** in Holm.
- **Level C's verdict is C-1's.**
- **Direction:** **HARDER** than the draft's C-1, which could not fail. Against the first
  per-state version, the pooled budget removes a FAIL that was forced by construction
  (logic, not forced by measurement; §12.0 item 14). The power condition on FAIL, the
  witness and the restriction on the instrument are HARDER or NEUTRAL.

### 12.11 The placebo, as run

- **Construction 4:** `analysis.placebo.matched_placebos(paths, 1000, seed=0,
  groups=context.placebo_groups(paths), method="uniform")`, where `paths =
  context.session_paths(...)` at the partition's K.
- **Blocks:** one per (fold, segment) on the industry sessions. Each fold's training
  labels and test labels are redrawn separately, each block keeping its own clock,
  occupancy and episode multiset exactly. Transitions across block boundaries are not
  controlled. `analysis.placebo.check_placebos` must report `exact`; otherwise the
  levels that use the placebo are UNDECIDABLE.
- **The same 1,000 draws** serve A-2, B-1 and C-1 and the placebo arms of A-1 and B-2.
  On every draw, everything estimated from training labels (`m`, the matrices, the
  intervals) is re-estimated from the draw's training labels through the same code. The
  draw's path is lagged exactly as the real one. The volatility witnesses (§8 control
  4) are not placebos and are not redrawn.
- **Sensitivities** draw their own 1,000 (seed 0) on their own paths. The smoothing
  sensitivity draws on the smoothed stamped paths, with the unlabelled first 20
  sessions of each fold's training block left out of the block (§12.13); the uniform
  construction is exact on it (measured, §I.2).
- **Any non-finite draw** of any statistic a lock uses makes that lock UNDECIDABLE; no
  draw is dropped, replaced or redrawn (§13.4, `protocol.placebo_p_value`,
  `protocol.placebo_percentile`).
- **No minimum freedom per block; thin blocks are kept.** Fold 5's test block (1995
  start) holds two states of 8 episodes each: only 2 join-free state orders exist,
  although each state's lengths are still permuted among its slots. A block with little
  freedom makes the null resemble the real partition there, which can only make the
  99th percentile harder to clear.
- **The swap repair is not used.**
- **Reasons.**
  - The swap repair completes 4 of 1,000 draws on the declared object, and retrying
    conditions the null. The uniform construction is exact on every object measured and
    conditions on nothing but the clock.
  - The blocks are exactly what the statistics read: every estimate is fitted on a
    fold's training segment and applied on its test segment.
  - The feature calendar carries non-trading days that no statistic reads.
- **Direction: forced by measurement.** Construction 3 cannot be run. Construction 4 is
  exact where construction 3 was approximate. It is not EASIER.

### 12.12 α, sidedness, bootstrap

- **Two-sided** α (`protocol.mde_z`). One-sided would scale the MDEs by about 0.93 at
  0.05/6. That would be EASIER and is not taken. The draft's figures were two-sided.
  **NEUTRAL.**
- **Mean block:** the one that gives the largest MDE among 21 / 63 / 126, as in the AHL
  level-B precedent. **HARDER**, by at most 0.002 at the random-map median.
- **Bootstrap:** `analysis.power` through `protocol.blinded_mde`, 2,000 draws, seed 0.
  Power 0.80.
- **One power convention** (ARBITRAGE §4.3): the paired blinded bootstrap decides. Lo's
  standalone threshold (0.932 at α 0.05/6 over the 20.0 paired years) is printed beside
  it for comparison and decides nothing.
- **Family:** Holm over this tree's six primaries. The programme-wide count of
  ARBITRAGE §4.5 is registered (§14), not used as α. **NEUTRAL**: it is the draft's
  family.

### 12.13 Sensitivities

- **K ∈ {3, 5, 6}:** each level's full evaluation is rerun at that K. That means the
  partition, the placebo at that K, and the thresholds by the same procedures, with the
  UNDECIDABLE cell floor at half of the 5K cells. One row per (K, level): 3 × 3 = **9**
  rows.
- **d ∈ {0.25, 1.00}:** A-1 only, with the same `m`. The threshold is recomputed by the
  §12.6 procedure; the random-map median at d 1.00 is 0.404, above the floor. **2**
  rows.
- **21-session smoothing** *(lock, reinstated after an audit)*. `PLAN.md` §f.4
  pre-registered it as a sensitivity aimed at flicker, the main weakness this lock
  measured (median out-of-sample episode 4 sessions). The first version of this lock
  had dropped it on a documentary argument; that would have been EASIER on the trial
  count and would have removed the one declared check on flicker.
  - *Rule:* `context.trailing_mode(paths, 21)` at commit `3b64ab2`. Per fold, over its
    whole stamped path (training then test sessions), session *t* takes the most
    frequent label of the 21 sessions ending at *t*, the smallest label on ties. It is
    applied before the lag. The first 20 sessions of each path have no label and hold
    the fallback, with no back-fill; they are left out of every cell and every placebo
    block. Everything else (qualification, estimators, placebo, thresholds, witnesses)
    runs as for the primary.
  - *At K = 4, d = 0.50, all three levels:* one row per level, **3** rows.
  - *Measured before the lock, labels only (§I.2):* 3.80 transitions/yr out of sample,
    2.12 in sample, median episode 41 sessions, η²(log rv) 0.100, 13 of 20 A-2 cells,
    uniform placebo exact.
- They are read **after** all six primaries have been read and logged. They **never
  found a PASS and never revoke one**, and they are reported beside their primary. A
  sign reversal in a sensitivity is written into the results text.
- *(lock, second validator; §12.0 row 28)* "Never revoke" is a choice and is EASIER. A
  level PASS is labelled **PASS (not robust)** when any K or smoothing sensitivity row
  at that level has `delta ≤ 0` (the `delta` field of §13.5: A-1's, B-2's, or C-1's
  `S_C`). The label changes no verdict and no Holm step; it is written into the results
  text and into the `note` of each sensitivity row with `delta ≤ 0`; the primary rows,
  already logged, are not edited (`trials.log` only appends).
- **Nothing else is a sensitivity:** not the 1992 context start, not the `start` anchor,
  not the unorthogonalised partition. None is computed on returns, none is logged. The
  diagnostics of §12.6-§12.10 (NFCI refit, witnesses, per-fold components) are not
  sensitivities and spend no row.

### 12.14 Recorded, no choice needed

- **K-means runs on one thread:** a parallel reduction made two identical fits differ by
  2e-16. The partition is bit-reproducible.
- **Per-fold heterogeneity.** The out-of-sample clock runs at 3.75 to 19.25
  transitions/yr across folds, so power is unequal across folds. Per-fold components of
  A-1 and A-2 are reported as diagnostics only.
- **Flicker.** The median out-of-sample episode lasts 4 sessions, and the clock runs at
  12.55/yr out of sample against 6.41 in sample. The primary has no smoothing or
  minimum dwell. Adding one to the primary would be a modification of the hypothesis,
  counted against the three-modification limit, so none is added. The 21-session
  smoothing is a declared sensitivity (§12.13), and never founds a PASS.
- **Fold 3** (2014-08 to 2018-07) has η²(log rv) 0.608. There, the partition is largely a
  volatility partition. The volatility witnesses at every level (§8 control 4) are the
  guard, and per-fold readings are reported.
- **The unconditional realised sd** of demeaned net legs and the cap-binding share are
  cited in §I and §11, judged blind-safe (§14).

---

## 13. Decision procedure

### 13.1 Order

**All three levels are read, whatever each outcome.** The steps:

1. **Every instrument first** *(lock, from an audit)*. Before the reading of A begins,
   the instruments of A, B and C, and every new function they call, are written,
   unit-tested on synthetic data and committed, and each level's thresholds are
   committed. The new functions are the selector and its `m`, the second-moment
   matrices, the QLIKE loss, the witness bins, the cadence books with the pooled-budget
   rule, and the placebo arms. **Only the readings follow the order A, B, C.** An
   instrument changed after any reading is an amendment in `docs/PROTOCOL_FREEZE.md`.
   Otherwise, choices resolved in B's or C's code after A had been read could be
   steered by A's outcome.
2. **What an instrument prints.** Only:
   - counts: qualifying cells, abstentions, fallbacks;
   - the turnover of the arms it builds. At A these are the selector and the control,
     with ΔT; at B the state and pooled arms, with ΔT; at C only the twin and the menu
     of state-blind books (§12.10). At A and B it also prints the kill verdict line.
     That is intended: the line reads turnover, a demeaned-leg MDE and a realised sd,
     and reveals no mean;
   - the realised sd of the legs and the cap share;
   - the MDEs of demeaned legs, `T_A1`, `T_B2`, `MDE_C` and S\*;
   - the 50th, 95th and 99th percentiles of the nulls of A-2, B-1 and C-1. These are
     computed on real returns under content-free partitions, and they do not read the
     real partition's labels;
   - at A, the NFCI diagnostic (§12.6 step 6).

   It prints no mean, Sharpe, IC or per-state statistic of the real partition or of any
   real arm, and no Sharpe difference or beta of any placebo arm. The placebo arms of
   A-1 and B-2 are built by the reading script (§12.6 step 4). It computes `m` and the
   state matrices, which the arms' demeaned legs need, and never prints them. At C it
   does not build the conditional arm on the real partition (§12.10). It refuses a
   verdict on any non-finite reading.
3. **Commit.** The three scripts and the thresholds they produced are committed, in the
   instrument sections of `docs/RESULTS_TWOSIGMA_LEVEL_{A,B,C}.md` and as one committed
   JSON of every threshold, **before any reading**. The JSON also records the SHA-256
   of every input file the instruments read (`data/` is not versioned, and Ken French
   republishes its files).
4. **Readings, A then B then C.** A reading script recomputes its instrument; it is
   seeded and bit-reproducible. It prints the versions of the packages pinned by the
   committed `uv.lock` and refuses to read if one differs. It refuses to read if an
   input file's SHA-256 differs from the JSON's, or if any recomputed threshold differs
   **bitwise** from it. It then computes
   the real statistics, the placebo arms and the witnesses, and, for a lock that would
   otherwise PASS, the NFCI point-in-time rebuild (§8, control 5), applies §12 and this
   section, logs the trial rows (§13.5), and writes the reading section of the same
   results file, which is committed.
5. Then the Holm step (§13.3), then the sensitivities (§12.13).

A reading made before the commit of step 3 is void and reported as a protocol breach.

### 13.2 Verdicts and precedence

The lock verdicts are:
- **PASS**;
- **FAIL**, and FAIL (cost) at A-1 and B-2. A placebo lock (A-2, B-1, C-1) that does
  not hold reads FAIL only when its instrument measured, before the reading, a power of
  at least 0.80 against a declared alternative (C-1 when `MDE_C ≤ S*`, §12.10), or when
  C-1's tracking gate fails;
- **NOT SHOWN** *(lock, from an audit)*: a placebo lock that does not hold and whose
  power was not measured. Under this lock that is always the case for A-2 and B-1;
- **UNDERPOWERED** (A-1, B-2: positive but below the bar or not surviving Holm);
- **UNDECIDED** (A-1, B-2: positive at 5 bp, not at 10 bp);
- **DOWNGRADED** (beta, or not conditional: A-1, B-2; **PIT**: any lock, §8 control 5);
- **DOMINATED** (a volatility witness does at least as well: A-1, A-2, B-1, B-2, C-1);
- **UNDECIDABLE**.

**The cost column that decides is 5 bp.** 10 bp can only turn a positive reading into
UNDECIDED, and 20 bp enters only through the kill rule. A level's verdict is PASS only
if its locks pass: both at A and at B, and C-1 at C. Otherwise the level takes the
first status among its locks in this order: FAIL (FAIL (cost) included), then
UNDECIDABLE, then UNDECIDED, then UNDERPOWERED, then NOT SHOWN, then DOWNGRADED, then
DOMINATED. Following ARBITRAGE §5(b), no
PASS in one channel is traded for a FAIL in another: a turnover saving, a QLIKE gain
and a Sharpe difference are not commensurable.

### 13.3 Holm–Bonferroni over the six primaries

| primary | statistic | null or standard error | threshold as applied | p-value entering Holm |
|---|---|---|---|---|
| A-1 | Δ SR selector − control, net 5 bp, 5,031 test sessions | SE\* = the blinded bootstrap SE at the mean block, among 21 / 63 / 126, that gives the largest MDE(0.05/6) (§12.6 step 2); HAC-6 | Δ ≥ T_A1 = max(0.338, MDE_fit(0.05/6)) | max(p_boot, p_HAC); p_HAC := 1 if sign(t) ≠ sign(Δ); **1 if Δ ≤ 0** |
| A-2 | mean Spearman of static-neutralised mean-contrast profiles, training → test | 1,000 uniform placebo draws | p ≤ 0.01 (above the 99th percentile) | (1 + #≥)/1,001 |
| B-1 | mean QLIKE gain, state-conditional over pooled | same draws | p ≤ 0.01; DOMINATED unless p_W1 ≤ 0.05 and p_W2 ≤ 0.05 | (1 + #≥)/1,001 |
| B-2 | Δ SR state-RP − pooled-RP | SE\* as for A-1, on the B-2 pair; HAC-6 | Δ ≥ T_B2 = max(0.338, MDE_fit(0.05/6)) | as A-1 |
| C-1 | held-turnover saving over the weekly twin, tracking gate | same draws | p ≤ 0.01 and gate | (1 + #≥)/1,001 |
| C-2 | S_C × bps / 10,000 / σ_twin | — | never PASS | 1 |

Order the six p-values `p_(1) ≤ … ≤ p_(6)`. Find the largest *j* such that
`p_(i) ≤ 0.05 / (6 − i + 1)` for every `i ≤ j`. The primaries ranked 1 to *j* are
rejected. Rejection is **necessary, never sufficient**: each lock must also meet its own
criterion in §12. The final Holm step runs after C is read. Each level's results file
states its provisional verdicts at the Bonferroni level, 0.05/6, which implies Holm
rejection.

*(lock, from an audit)* **The Holm family is always the six primaries.** A lock that is
UNDECIDABLE enters Holm with p := 1, as C-2 does, and so does A-1 or B-2 with Δ ≤ 0: a
significantly negative difference must not count as a rejection that loosens the other
locks' thresholds. A p that is missing for any reason enters as 1, so the family never
shrinks below six while the bar is set at 0.05/6. Every other lock enters with its p
as defined in the table, whatever its own criterion gives. A placebo lock that does not
hold — p above 0.01, Holm not rejecting it, or at C-1 the gate failing — is FAIL or NOT
SHOWN by §13.2. When a placebo lock ranks first in Holm, its bar is p ≤ 0.00833: at most
7 of 1,000 draws at or above the real statistic.

### 13.4 What makes a level UNDECIDABLE

- **At every level:**
  - a non-finite instrument reading, or a non-finite real statistic;
  - *(lock, from an audit)* **any one** of the 1,000 draws of any placebo statistic a
    lock uses (`R`, `G`, `S_C`, `Δ^(j)`, `β^(j)`) is not finite. No draw is dropped,
    replaced or redrawn. `protocol.placebo_p_value` and
    `protocol.placebo_percentile` return NaN in that case (commit `3b64ab2`);
  - a paired leg missing on any test session;
  - a placebo check that is not exact.
- **At A and B, in addition:**
  - fewer than 10 qualifying cells (half of the 5K cells at a K sensitivity). At A the
    cells counted are the (fold, state) pairs that enter A-2 (§12.4); at B they are the
    qualifying training cells;
  - more than half of the test sessions at the fallback (§12.6 at A, §12.9 at B);
  - at B, a pooled training matrix that is not positive definite.
- **Measured before the lock (§12.4), labels only:** A-2 cells 14 of 20 at K = 4 (floor
  10), 11 of 15 at K = 3 (7.5), 15 of 25 at K = 5 (12.5), 17 of 30 at K = 6 (15), 13 of
  20 smoothed (10). Training cells 19 of 20, 15 of 15, 24 of 25, 28 of 30, 19 of 20.
  Fallback sessions at K = 4: 372 of 5,031. No floor binds on this data.

UNDECIDABLE is written with the reading that caused it, as AHL B1 was. No trial row is
spent on a lock that is UNDECIDABLE before its reading. *(lock)* A lock that becomes
UNDECIDABLE during its reading, for instance through a non-finite real statistic or a
broken placebo arm, has been read. It logs its row with verdict UNDECIDABLE and enters
Holm with p := 1.

### 13.5 Trial logging

`regime_lab.analysis.trials.log("twosigma", config, metrics)`. One row per primary at
its reading, and one per sensitivity row of §7: **at most 20 rows**, since a lock that
is UNDECIDABLE before its reading spends none (§13.4).

- **config:**
  - `prespec` ("docs/PRESPEC_TWOSIGMA.md, LOCKED 2026-09-23"), `test`, `role`
    (primary | sensitivity), `variant` (`primary`, `K=3`, `K=5`, `K=6`, `d=0.25`,
    `d=1.00`, `smooth21`), `K`, `d` (null where N/A);
  - `library` (the ten names), `context` (20 features, orthogonalised on log rv of
    `eq_us_large`, K-means n_init 20 rs 0, walk-forward refit, OOS nearest centroid,
    lag 1), `context_train_start` 1995-01-04, `anchor` end;
  - `folds` 5 × 4.0 y, `cost_bps` 5, `vol_target` 0.10, `cap` 3;
  - `placebo` (uniform, 1,000, seed 0, (fold, segment) blocks), `bootstrap`
    (stationary, blocks 21/63/126, 2,000 draws, seed 0), `estimator` (§12.5, mean
    contrast), `min_cell` (63 sessions, 3 episodes), `build` (`3b64ab2`).
- **A primary row** (`test` = A-1 … C-2):
  - `sharpe`: the net Sharpe of the arm the row evaluates, where one exists (A-1 the
    selector, B-2 the state arm, C-1 the conditional cadence arm); NaN otherwise;
  - `delta`, `threshold`, `p`, `placebo_pct`, `sessions` (5,031), `verdict` (the
    provisional Bonferroni-level lock verdict), `note`.
- **A sensitivity row** *(lock, from an audit: one schema, so that two analysts log the
  same rows)*:
  - K and smoothing rows: **one row per (variant, level)**, `test` = `A` | `B` | `C`.
    `delta`, `threshold` and `p` are those of A-1, B-2 and C-1 respectively. `p2` is the
    p of A-2 or B-1, NaN at C. `verdict` is the level verdict by §13.2. `sharpe` is the
    A-1 selector's, the B-2 state arm's or the C-1 conditional arm's at that variant;
  - d rows: `test` = `A-1`, A-1's metrics only, `p2` NaN, `verdict` = A-1's lock
    verdict.

The Holm step is computed from the six primaries' p-values, with p := 1 where §13.3 sets
it, and published in `docs/RESULTS_TWOSIGMA.md`. It is **not** logged as extra rows.
The
register holds, on 2026-09-23, **807 rows and 85 distinct configuration hashes over 8
families, none named `twosigma`**.

*(lock, second validator)* The logged `sharpe` is that of a near-zero-net long-short book
of industry portfolios with no borrow cost. It is not a measure of progress toward the
programme's ambition of a net Sharpe of 1 to 2 (`AVANCEMENT.md` §0), and every results
document that quotes it says so.

### 13.6 Closure

The closure statement of §6 is written only if no level passes, with T_A1 at its
applied value. Each non-PASS status has its own wording, and no sentence reaches beyond
the declared object: a ten-signal US industry library, this partition, 20.0 paired
years.
- **FAIL:** "does not transfer". **FAIL (cost):** "does not survive its own turnover".
- **UNDERPOWERED, NOT SHOWN, UNDECIDABLE:** "is not shown to transfer", with the
  resolution: T_A1 or T_B2 at A-1 and B-2; the null's `q99 − q50` at A-2 and B-1;
  `MDE_C` at C-1; for UNDECIDABLE, the reading that caused it.
- **UNDECIDED:** "is not shown to survive a conservative cost schedule".
- **DOMINATED:** "passes its test, but a one-line volatility partition does at least as
  well; it is not credited to the context".
- **DOWNGRADED:** "passes its test, but the gain is market exposure (beta), or is no
  larger than a content-free partition with the same clock produces (not
  conditional), or does not hold without the revised NFCI (PIT); it is not credited to
  the context".

Any other outcome is written as the six lock verdicts with their numbers.

---

## 14. Scope, and what this lock leaves open

- **The AQR sealed holdout (1971-04 to 1989-12) is not touched by this study.**
  ARBITRAGE §4.2 suggested merging AQR into this plan. The merge is not taken here: the
  instruction was the Two Sigma plan, and the holdout opens once only. Whoever runs AQR
  decides.
- **ARBITRAGE §4.1**, which promoted the swap repair as the programme-wide placebo, is
  affected. On a walk-forward object cut into blocks, the swap repair is infeasible.
  Every plan that uses the shared placebo must measure its own feasibility, and the
  uniform construction is the one this study runs. `analysis.placebo`'s default method
  is still `swap`. Changing it is outside this file.
- **Programme-wide count (ARBITRAGE §4.5):** still 34 primaries. Two Sigma now adds 14
  sensitivities, not 15, so the count with sensitivities is 48, not 49.
- **ARBITRAGE §4.4's** 10.84 to 10.90% realised vehicle volatility does not hold for this
  object, which realises 7.88% under the same cap.
- **Code state, at commit `3b64ab2`:**
  - `protocol.DECLARED_EVALUATIONS` is 20, the count of §7; the draft's 21 survives in
    `scripts/measure_twosigma_power.py` only to reproduce the draft's 0.376;
  - `protocol.py`'s docstring names this file as the text that decides;
  - `analysis.placebo`'s default method is still `swap`; every call in this study must
    pass `method="uniform"` explicitly (§12.11);
  - `context.py` imports `threadpoolctl`, which is declared only transitively through
    scikit-learn;
  - `library.annual_turnover` (refuses NaN) and `protocol.annual_turnover` (skips
    partial rows) share a name. They agree on complete frames, and §12 uses the latter;
  - `library.load_panel`, `context.load_log_realised_vol` and
    `folds.inferential_sessions` pivot on `period`, not `available_at`. That is right
    today, since the two coincide on every row they read, but it would go silent if the
    store ever encoded a delay. An assertion belongs there before the instruments;
  - pandas' rolling correlation (`IND_LOWCORR`) changes in the last bits (about 4e-16)
    when rows after a date change, so "bit-identical" holds only on identical input
    files: hence the SHA-256 check of §13.1.
- **Handoff documents** — `AVANCEMENT.md`, `CLAUDE.md`,
  `pilotage/feuille_de_route/TACHES.md` — recorded the build and the pending lock at
  `f5f04ec`, and are updated to LOCKED by the commit that adds this file, together with
  its row in `docs/PROTOCOL_FREEZE.md`.
- **Blindness judgement, confirmed.** The build printed, and this file cites, the
  unconditional realised sd of demeaned net legs (control 7.88%, random-map arms 7.6 to
  10.8%) and the share of sessions on which the cap binds. They were judged allowed:
  unconditional second moments, no mean, no per-state content. The three audits of this
  text concurred. The audit commit `3b64ab2` added counts of cells, the smoothed clock
  and the kill's base rate on content-free maps, all on the same allow-list.
- **Open: a blindness ruling for the power of A-2 and B-1.** Measuring their power needs
  an effect planted into demeaned legs. Until such an instrument is ruled blind and
  committed as an amendment before any reading, A-2 and B-1 can read NOT SHOWN but
  never FAIL (§12.7, §12.8).
- **Not verified here:** the level instruments of §13.1 do not exist yet. Every
  procedure in §12 is written against the committed functions it names. The new code —
  the mean-contrast `m`, the second-moment matrices, the QLIKE loss, the witnesses W1
  and W2 with their test, the cadence books with the pooled-budget rule, the placebo
  arms, the NFCI refit — must be written, tested on synthetic data and committed
  **before A is read** (§13.1). The audits' synthetic simulations of the Sharpe-profile
  and per-state-cadence defects (§12.5, §12.10) were not committed; this text relies on
  the logical arguments and quotes none of their figures.
- **Audit findings not applied as written, and why** (the blocking findings were all
  acted on; these four were resolved differently from the audit's first suggestion):
  - *The cost kill.* The audit's first option replaced the kill by "Δ at 20 bp ≤ 0".
    Its second option, a turnover kill anchored to the decision bar, is taken: it is
    the harder of the two (at Δ ≥ 0.338 it fires at a lower turnover), it is the
    draft's own 12×/yr in practice, and it keeps the UNDECIDED line reachable. Its base
    rate on content-free maps was measured before the lock: 0 of 8, 1 of 8 shaped
    (§12.6).
  - *B-2's kill budget.* The audit that pinned B-2's comparator scaled the budget to
    the pair's own uncorrected MDE. Another audit showed that such a budget fires
    almost automatically, since the two legs are nearly identical. The comparator is
    taken, and the budget is anchored to T_B2 (§12.9).
  - *"A placebo lock that does not hold is FAIL."* One audit asked for it, and another
    showed that no power was measured for A-2, B-1 or C-1. The two are reconciled: FAIL
    where power ≥ 0.80 was measured before the reading (C-1 only) or C-1's gate fails,
    NOT SHOWN otherwise (§13.2).
  - *C-1's MDE.* The audit wrote `q99 − q50` and asked for power 0.80. Under a location
    shift, `q99 − q50` gives power 0.50, so the lock uses `q99 − q20` (§12.10), which is
    harder to meet before a FAIL can be written.
- **Not taken, reported:** an adjacency diagnostic (A-1 and A-2 recomputed without the
  first 63 test sessions of each fold, against factor momentum carried across the
  training/test boundary) and a circular-rotation second null were suggested by an
  audit as non-blocking. Neither is part of this lock. The second validator's
  non-blocking suggestions are not part of it either: a paired test instead of a point
  comparison for the volatility witness at A-1, A-2, B-2 and C-1; the adjacency
  diagnostic above, also at 252 sessions; renaming UNDECIDED or UNDECIDABLE, which are
  easy to confuse; and an effective rank measured in return space, where axis 4 is
  measured in position space only.
