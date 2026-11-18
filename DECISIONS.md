# Decisions log

Every analysis decision, with its reason. Re-read at the start of each phase; append new decisions; never reverse a
logged decision without the project owner's approval. Status: **approved** = confirmed by the owner;
**proposed** = awaiting approval; **superseded by X** = replaced by a later decision (the final method is P7F-1/P7F-2, P7F-3, P7G-1).

## Environment
| # | Decision | Reason | Status |
|---|---|---|---|
| E1 | `.venv` built from system Python 3.9.6 (`/usr/bin/python3`), never the FSL Python | reproducibility, independent of other installs | approved |
| E2 | LightGBM not used; scikit-learn HistGradientBoosting instead; no system libraries installed | LightGBM needs libomp, not available | approved |
| E3 | Seeds fixed (42), n_jobs = 2, OMP_NUM_THREADS = 2; one command `python run_all.py` rebuilds everything | reproducibility, MacBook Air limits | approved |
| E4 | Git: `.venv/`, `data_processed/` ignored; raw CSVs committed read-only; commit after every phase | version control | approved |

## Data (Phase 1)
| # | Decision | Reason | Status |
|---|---|---|---|
| D1 | Total runs = `runs + extras1_runs + extras_2_runs` | reconciles 804/809 decided un-revised results; dropping extras_2 breaks ~8% | approved |
| D2 | `extras_type = other` is a no-ball (height no-balls/beamers) | treating it as legal gives 244 overs of 7–9 legal balls | approved |
| D3 | Legal delivery = not wide, not no-ball (Overstep/other), not the penalty row | standard scoring | approved |
| D4 | Bowler charged: bat runs + all wide runs + 1 per no-ball; not byes, leg byes (incl. off no-balls), penalty runs | standard scoring | approved |
| D5 | Bowler wickets: Caught, Bowled, LBW, Stumped, Hit Wicket. Not: run out, retired out, obstructing | standard scoring | approved |
| D6 | Super-over innings (3–4) excluded; no-result and abandoned matches (40) excluded | not regular play / no outcome | approved |
| D7 | IPL identified by competition name; the two A-team Asia Cup events (in `match_format = T20`) go to the weaker group | `T20` format ≠ IPL (IPL = 371 matches, not 401) | approved |
| D8 | Strength groups: IPL; FM_T20I (both teams ICC full members); Weaker_T20I (any associate or A side) | brief | approved |
| D9 | Ties kept for player analysis, excluded from WP training | no win label | approved |
| D10 | Rain-revised matches: revised fields are in balls; unrecorded rain detected (≥1 over short, ≤8 down, chase incomplete) and lengths inferred for pre-match reductions; excluded from WP training | cleaner WP labels | approved |
| D11 | Player identity: id 4458 = "Aneurin Donald" (mapping file's "AHT Donald" ignored). Same-name different players kept apart by id (Abdul Samad, Haider Ali, Rashid Khan, …). Basil Hameed ids 8638/39444 probably one person but kept separate (cannot prove; negligible sample). Final lists add a team tag where a name maps to two ids | identity check, Phase 6 Part 1 | proposed |

## Win probability (Phase 2)
| # | Decision | Reason | Status |
|---|---|---|---|
| W1 | Leak-free venue par: same-group league mean over previous 730 days + venue residual shrunk by n/(n+k), k estimated (12.0) | brief; no future data | approved |
| W2 | 1st innings: sign-constrained logistic; 2nd innings: sign-constrained logistic + endgame logistic blended 24→12 balls left | beat monotonic HGB, spline logistic, Markov chain and hybrid on out-of-fold log-loss/Brier for ≤30 balls left | approved |
| W3 | 5-fold CV grouped by match; non-training matches scored by the all-data model | no leakage | approved |
| W4 | Ties = 0.5 in all WP values and swings | brief | approved |
| W5 | End of 1st innings → 1 − WP(start of chase, target = score + 1, chase length = 1st-innings length) | later rain revisions are unknowable at the break | approved |
| W6 | 1st-innings stage interactions (score−par × balls left, wickets × balls left) tested and **rejected** (0.6242 → 0.6240, folds mixed; unconstrained interaction slightly negative). Limitation noted | keep only if out-of-fold log-loss improves | approved |

## Leverage (Phase 3)
| # | Decision | Reason | Status |
|---|---|---|---|
| L1 | Centred LI: LI_raw = Σ p(o)·\|WP(s_o) − Σ p(o')·WP(s_o')\|; specified version kept as `LI_spec` (appendix) | removes the 1st→2nd-innings model hand-over jump; identical elsewhere (Spearman 0.996) | approved |
| L2 | Primary H = top 25% LI **within each innings**; H20/H33 within innings for robustness; L = the rest | global cut-off put 83% of H in the 2nd innings and none in 1st-innings middle overs | approved |
| L3 | Rain-shortened matches (32) excluded from H and L; kept for conventional statistics | WP model never trained on shortened innings | approved |
| L4 | Outcome model: gradient boosting (8 classes), player-agnostic, out-of-fold by match | lower log-loss than multinomial logistic | approved |

## Value and metrics (Phases 4–5)
| # | Decision | Reason | Status |
|---|---|---|---|
| V1 | Opposition quality = leave-one-match-out run-value ratings, empirical-Bayes shrunk (c: batters 132, bowlers 564) | brief; no own-match leakage | approved |
| V2 | Value = actual WP change − expected (opposition-aware); v = Value / LI_raw | brief | approved |
| V3 | R centred on the population P − B (delivery-weighted); R_raw kept | removes a population-wide shift (bat +0.019, bowl −0.018) that would bias the "R significantly < 0" rule | approved |
| V4 | Phase 7 will test an informed prior for P (shrink toward a + b·B̂) vs the population mean, chosen by next-season prediction; keep the simple version if the gain is negligible | P and B are related | approved |
| M1 | Yorker execution: report only, not ranked | the delivery label depends on the outcome (missed yorker ≈ full toss) | approved |
| M2 | Squeeze response: H and L trigger deliveries pooled | the trigger is itself pressure; only 3 batters have 30 H trigger balls | approved |
| M3 | Entry resilience kept; compared between players only | carries a population new-batter effect | approved |
| M4 | Run-out avoidance: candidate; Phase 6 decides | rare events | approved |
| M5 | Partner effect: candidate, "on trial" | | approved |
| M6 | Stroke intent: profile only | | approved |
| M7 | **Candidate metric list locked** (no new metrics) | | approved |

## Phase 6
| # | Decision | Reason | Status |
|---|---|---|---|
| P6-B | **H25 within innings stays primary** after the band check (H25/H33/H50/LI-weighted) | no version clearly more reliable (intervals overlap); H33 loses a quarter of top-25% concentration | approved |
| P6-1 | Primary split-half reliability = mean Spearman–Brown over 50 random match splits per player, with bootstrap 95% intervals over players; odd/even (date order) and chronological (earlier vs later half) reported alongside | single splits were unstable (batters' P: 0.33 by id parity vs 0.05 by date order) | approved |
| P6-2 | Target candidates: composite run value per ball (actual − expected runs + wicket value from the Phase 2 resource table; opposition-aware expectations) on H, on all deliveries, and on all deliveries weighted by LI; and an equal-weight composite of standardised process metrics (bat: strike-rate delta, dot suppression, boundary elevation; bowl: dot rate, boundary suppression, economy suppression, wicket-taking) on H and on all deliveries. These are targets, not new candidate metrics | owner's instruction; bowler H metrics individually unreliable | approved |
| P6-3 | Target = most reliable measure that still reflects pressure (H or LI-weighted preferred; all-deliveries only if pressure versions are not clearly positive = random-split interval lower bound ≤ 0) | owner's instruction | approved |
| P6-4 | Phase 7 rating = empirical Bayes estimate of true H performance: prior from all-delivery skill, updated by the H record in proportion to the estimated true variance of the pressure-specific deviation (estimated from data; if ≈ 0 the update is ≈ 0, stated) | owner's instruction | superseded by P7F-1 (batters: L-only component prior) and P7F-2 (bowlers: bivariate) |
| P6-5 | Retention stays a condition: exclude a player whose R 95% interval is entirely below 0 | owner's instruction | approved |
| P6-6 | Batter target: process composite on H (reliability 0.41 [0.23, 0.52]) | most reliable measure that reflects pressure | superseded by P6B-4 (rating target = run value per ball on H) |
| P6-7 | Bowler target: no pressure measure is clearly positive (LI-weighted run value 0.19 [0.00, 0.32] passes only at the boundary). **Resolved by Phase 6B** (see P6B-3) | rule P6-3 cannot be satisfied cleanly | approved (via 6B instruction) |
| P6-8 | Weights: baseline skill only beats ridge and equal weights out of sample on both sides; ridge weights not used. **Resolved by Phase 6B** (see P6B-4) | D3 comparison | approved (via 6B instruction) |

## Headline finding (Phase 6)
**What repeats under pressure is process, not outcome.** Split-half reliability (mean of 50 random match splits, Spearman–Brown, 95% bootstrap interval), H = top 25% LI within innings:
- Repeats (batters): strike-rate delta 0.43 [0.26, 0.55], dot suppression 0.42 [0.26, 0.54], boundary elevation 0.45 [0.29, 0.58]; process composite on H 0.41 [0.23, 0.52]; run-out avoidance 0.23 [0.05, 0.37].
- Does not repeat (batters): WP value P 0.02 [−0.49, 0.31], wicket security 0.19 [−0.11, 0.36], retention R 0.06 [−0.35, 0.31], entry resilience, collapse resistance, post-wicket response, squeeze (pooled), partner effect.
- Bowlers: no H metric repeats individually — P −0.14, economy −0.12, dots 0.06, wickets 0.14, boundaries −0.11, bounce-back, post-wicket, execution drift, retention R (all intervals span 0). Only variation (entropy H − L) passes, 0.22 [0.06, 0.38], and it predicts next-season performance *negatively* (r −0.13 [−0.22, −0.03]).
- True variance of pressure-specific run value (H beyond what L predicts): batters 0.0029 runs²/ball [−0.012, 0.014] (26% of true H variance, interval spans 0); bowlers ≈ 0 (true H variance itself estimated negative). Phase 7's empirical Bayes H update is therefore small for batters and ≈ 0 for bowlers.
- Out of sample (next season, grouped by player), baseline skill alone predicts the pressure target better than weighted pressure metrics: batters r 0.16 [0.01, 0.30] vs ridge 0.07 / equal weights 0.10; bowlers 0.07 [−0.02, 0.18] vs ridge −0.12 / equal −0.13.

## Phase 6B — signal recovery (pre-specified; no other methods added)
| # | Decision | Reason | Status |
|---|---|---|---|
| P6B-0 | Methods A–E fixed in advance. Pass = random-split reliability 95% interval above 0 AND bootstrap 95% interval of the out-of-sample gain (weighted r, next-season run value on H, folds grouped by player, over the best baseline) above 0 | owner's stopping rule | approved |
| P6B-1 | Mixed-model slope-variance interval by parametric bootstrap (100 resamples), not profile likelihood: statsmodels' parameter-fixing did not hold the slope variance fixed in either parameterisation. Split-half/season slopes = exact BLUPs from sufficient statistics with variance components fixed at full-data values | a correct profile could not be computed; bootstrap is the allowed alternative | approved (method listed by owner) |
| P6B-2 | **Result: no method passes on either side.** Best baseline = all-delivery run value on both sides (LI-weighted was worse: batters r 0.076 vs 0.140; bowlers −0.141 vs −0.057) | see outputs/06b_signal_recovery.md | recorded |
| P6B-3 | **Pressure-specific bowler skill not detectable in this data.** | stopping rule | recorded |
| P6B-4 | Phase 7 rating = empirical Bayes estimate of H performance (run value per ball on H): prior from all-delivery run value (best baseline), H update in proportion to the estimated pressure-specific deviation variance (Phase 6: batters 0.0029 runs²/ball, interval spans 0; bowlers ≈ 0). No validated pressure-specific component (slope, episode rate) is added | owner's 6B instruction | superseded by P7F-1 (batters) and P7F-2 (bowlers); 'no validated pressure component added' still holds |
| P6B-5 | Retention R stays a condition (exclude if R 95% interval entirely below 0) | P6-5 | approved |

## Headline findings after Phase 6B (owner-approved wording)
- **Pressure-specific variation exists on both sides** (mixed-model LRT p = 2.7e-8 batters, 6.2e-6 bowlers; slope SD ≈ 0.065 runs/ball), **but is too small relative to sampling noise to estimate reliably per player with 2022–26 data. Core skill is the best predictor of pressure performance.**
- **Death pace specialists** (method E within H) show the strongest pressure signal (reliability: P 0.47, economy 0.40, boundary suppression 0.53; n ≈ 45–51 bowlers) but fail the pre-specified rule; reported as a lead for further work, **not used in the ranking**.
- **Role-group pooling raised reliability without improving prediction**: the gain came from shared role means (both split halves share the same role mean), not from player-level signal.

## Phase 7 — rating and ranking
| # | Decision | Reason | Status |
|---|---|---|---|
| P7-1 | Bowler prior = **all-delivery economy suppression** (runs vs expected, no wicket cost). Out-of-sample r for next-season run value on H: run value −0.057 [−0.164, 0.079], economy 0.033 [−0.081, 0.153], process composite −0.027 [−0.135, 0.101]; all intervals overlap, so the most reliable was chosen (economy: split-half 0.25 [0.09, 0.39], season 0.19 [0.08, 0.28]). Note: no bowler baseline predicts next-season H run value | owner's rule P7.1 | superseded by P7F-2 (bowlers: bivariate component rating; wicket component dropped per P7B-2) |
| P7-2 | Batter prior = all-delivery run value | owner's rule | superseded by P7F-1 (batters: L-only component prior) |
| P7-3 | Recency weighting tested (λ = 0.8/0.6/0.4 vs 1) and **not kept**: batters best λ 0.4 gain +0.017 [−0.033, 0.063]; bowlers best λ 0.8 gain −0.002 [−0.015, 0.010] | keep only if out-of-sample gain is clearly positive | superseded by P7R-4 (re-tested on the component prior; still not kept) |
| P7-4 | Rating = prior (a + b × shrunk baseline, a/b by weighted regression of observed H run value on the baseline) + k × (H mean − prior), k = τ²_dev / (τ²_dev + σ²_H / n_H); τ²_dev batters 0.00288, bowlers 0 (estimate −0.0017, clipped) → bowler ratings are the prior | P6B-4 | superseded by P7F-1 (batters) and P7F-2 (bowlers) |
| P7-5 | SE = bootstrap over the player's matches (200); rank score = rating − SE | brief | superseded by P7F-4 (200 → 2,000 resamples for SEs) and P7G-1 (retention gate on pooled 10,000 resamples); rank score = rating − SE unchanged |
| P7-6 | Eligibility: ≥ 2 × median H deliveries in regular player-seasons (≥10 matches): batters 108 (median 54), bowlers 154 (median 77); played in 2025 or 2026; not excluded by retention (R 95% interval entirely below 0; R unreliable, gate only). Eligible: 118 batters, 80 bowlers; retention excluded 2 batters, 6 bowlers | owner's rule P7.4 | partly superseded: thresholds (108 / 154) and 2025–26 activity rule kept; gates superseded by P7F-3 (delivery requirement) and P7G-1 (pooled retention gate) |
| P7-7 | Duplicate data names get a team tag; all other names exactly as in the data | P6 Part 1 | recorded |

## Phase 7 review fixes (owner instruction)
| # | Decision | Reason | Status |
|---|---|---|---|
| P7R-1 | Prior uses only non-H (L) deliveries; pressure-specific variance re-estimated with that prior (batters 0.0035 [−0.0055, 0.0123] → used; bowlers −0.0022 [−0.0078, 0.0038] → 0). Supersedes P7-4's all-delivery prior | no double counting of H deliveries | final for batters (P7F-1); superseded for bowlers by P7F-2 |
| P7R-2 | Component-wise prior: runs component + wicket component (wicket cost from the Phase 2 resource table), each shrunk with its own reliability (batters c_runs 153, c_wkt 78; reliability at median L sample 0.22 / 0.36). Bowlers: the wicket component has no estimable between-player variance on L (reliability 0) → it is a constant and is dropped (coefficient 0); bowler prior = economy component on L. Out of sample vs the reviewed rating: batters +0.051 [−0.022, 0.103], bowlers −0.012 [−0.069, 0.048] → not clearly worse, component version kept on both sides. Supersedes P7-1 for the rating | owner's rule: keep unless clearly worse | final for batters (P7F-1); superseded for bowlers by P7F-2 (bivariate; wicket component dropped per P7B-2) |
| P7R-3 | Delivery condition: eligible only if own raw run value on H > 0 (in addition to volume, 2025/26 activity and the retention gate). Removed 36 batters and 38 bowlers (names in outputs/07_rating.md) | brief: "deliver when it matters"; ranking supported by data | superseded by P7B-3, then reinstated as P7F-3 (same rule, framed as a requirement) |
| P7R-4 | Recency on the component prior re-tested: not kept (batters +0.015 [−0.034, 0.056], bowlers −0.001 [−0.013, 0.014]) | P7-3 rule | recorded |
| P7R-5 | Numerical safeguard: a prior component whose between-player variance is at the estimation floor is dropped rather than entering a collinear regression (first run produced coefficients of ~10^8 for the bowler wicket component) | numerical stability; no effect on any estimable component | recorded |

## Phase 7 second review (owner instruction)
| # | Decision | Reason | Status |
|---|---|---|---|
| P7B-1 | Bivariate, component-wise empirical Bayes replaces the L-only prior (P7R-1): H performance = core skill (all deliveries, shrunk by its between-player variance) + pressure-specific deviation ((H − L) − population shift, shrunk by var(δ)). Batters: c_runs 157, c_wkt 106; var(δ) runs 0.0048 [−0.0030, 0.0114] (used), wicket −0.0004 (→ 0). Bowlers: c_runs 318; var(δ) runs 0.0022 [−0.0043, 0.0099] (used, so bowlers now get a small H update) | uses pressure deliveries as evidence of core skill without double counting | final for bowlers (P7F-2); superseded for batters by P7F-1 (bivariate kept as a tested alternative) |
| P7B-2 | Bowler wicket component re-estimated on ALL deliveries: between-player variance −0.0004 → not estimable → dropped (stated in report) | owner's rule | recorded |
| P7B-3 | Noise-aware delivery condition replaces raw H > 0 (P7R-3): exclude only if raw H + 1.28·SE < 0 (SE = bootstrap over matches). Removed 5 batters, 9 bowlers | owner's rule | superseded by P7F-3 (noise-aware version kept as a sensitivity check) |
| P7B-4 | Tiers: new tier when the gap to the previous rank score exceeds the larger adjacent SE; order still by rank score | owner's rule | approved |
| P7B-5 | Re-validation (next-season run value on H): batters bivariate 0.103 vs previous 0.184, difference −0.082 [−0.144, +0.011]; bowlers 0.028 vs 0.019, +0.008 [−0.073, +0.080]. **Closed by P7F-1/P7F-2** | | closed |

## Phase 7 final (owner decisions; rule: out-of-sample prediction first, then use of evidence)
| # | Decision | Reason | Status |
|---|---|---|---|
| P7F-1 | **Batters: L-only component prior + component-wise H update** (P7R-1/2). Bivariate batter rating kept as a tested alternative (appendix table) | prediction r 0.184 vs 0.103 | approved |
| P7F-2 | **Bowlers: bivariate component rating** (P7B-1) | prediction equivalent (0.028 vs 0.019); uses all deliveries as evidence of core skill | approved |
| P7F-3 | **Delivery condition = requirement of demonstrated delivery under pressure**: raw run value on H > 0 to appear in a final ranked list (brief: "deliver when it matters"). It is a requirement, not an estimate. Noise-aware version (exclude only if raw H + 1.28·SE < 0) reported as a sensitivity check | owner | approved |
| P7F-4 | Stable gates: SEs, raw-H SEs and retention intervals with 2,000 bootstrap resamples (fixed seed); primary run repeated at 4 other seeds; borderline players excluded. **Exclusion part superseded by P7G-1** | consistency is part of the brief | superseded by P7G-1 for the exclusion rule; 2,000 resamples for SEs kept |
| P7F-5 | Result of P7F-4: one borderline bowler — Jasprit Bumrah (player_id 2933): retention 95% intervals by seed [−0.333, +0.021], [−0.338, +0.015], [−0.343, +0.017], [−0.336, +0.020], [−0.336, −0.002] (excluded only at seed 4). **Reversed by P7G-1** | superseded by P7G-1 / P7G-2 (Bumrah passes the pooled gate) |
| P7F-6 | Names: every final name matches the data spelling; names shared by two player_ids anywhere in the data (incl. the mapping file) carry a team tag — "Rashid Khan (Afghanistan, Gujarat Titans)" (the other Rashid Khan is Nepal's, id 20172, mapping file only) | D11 | recorded |
| P7G-1 | **Retention gate on pooled resamples**: pool the bootstrap resamples of all 5 seeds (5 × 2,000 = 10,000) and apply the gate once (exclude only if the pooled 95% interval is entirely below 0). Players whose single-seed decisions disagree are reported as borderline but **not excluded**. SEs and raw-H SEs stay on seed 0 (2,000) | a flip across seeds reflects Monte Carlo error at the interval edge, not the player | approved |
| P7G-2 | Result: Jasprit Bumrah pooled interval [−0.338, +0.007] → passes, returns at bowler #1; Mohammed Siraj moves to #11. No batter changes. With the pooled run no player's single-seed decisions disagree (borderline list empty) | P7G-1 | recorded |
| P7G-3 | Name note: the only final name shared with another player_id is **Rashid Khan** → shown as "Rashid Khan (Afghanistan, Gujarat Titans)", **player_id 4722** (3,320 deliveries in the ball data). The other id, 20172, is Nepal's Rashid Khan (mapping file only, 0 deliveries). All other final names are unique and match the data spelling exactly | D11 | recorded |

## LOCKED final lists (git tag `final-lists`)
From here, validation results are reported, not used to change the lists (exception: a genuine bug, reported to the owner first). Frozen copies: `outputs/tables/locked_top10_batters.csv`, `outputs/tables/locked_top10_bowlers.csv` (checked automatically by checks.py).

**Batters (L-only component rating)** — rank. player (player_id) · rating / SE / rank score, runs per 100 H balls · tier
1. Vaibhav Sooryavanshi (50284) · 52.28 / 8.97 / 43.31 · tier 1
2. Abhishek Sharma (6281) · 35.93 / 5.35 / 30.58 · tier 2
3. Finn Allen (4847) · 27.33 / 5.53 / 21.80 · tier 3
4. Phil Salt (5553) · 25.66 / 4.04 / 21.62 · tier 3
5. Suryakumar Yadav (1699) · 24.05 / 3.63 / 20.42 · tier 3
6. Tim David (6396) · 24.77 / 5.27 / 19.51 · tier 3
7. Heinrich Klaasen (4260) · 19.73 / 3.72 / 16.01 · tier 3
8. Andre Russell (1318) · 21.11 / 5.93 / 15.18 · tier 3
9. Travis Head (2367) · 20.43 / 6.35 / 14.08 · tier 3
10. Priyansh Arya (25669) · 22.57 / 8.82 / 13.75 · tier 3  — top 10
11. Josh Inglis (4736) · 17.88 / 5.68 / 12.20 · tier 3
12. Glenn Maxwell (838) · 17.00 / 4.96 / 12.03 · tier 3
13. Nicholas Pooran (3125) · 15.95 / 4.65 / 11.30 · tier 3
14. Romario Shepherd (6726) · 15.01 / 4.67 / 10.33 · tier 3
15. Jos Buttler (813) · 13.56 / 3.51 / 10.05 · tier 3
16. Mitchell Marsh (818) · 13.75 / 3.86 / 9.89 · tier 3
17. Harry Brook (6280) · 14.81 / 5.46 / 9.35 · tier 3
18. Prabhsimran Singh (7216) · 13.63 / 4.38 / 9.25 · tier 3
19. Glenn Phillips (4853) · 12.71 / 3.81 / 8.90 · tier 3
20. Cameron Green (5939) · 12.92 / 4.29 / 8.63 · tier 3

**Bowlers (bivariate component rating)** — rank. player (player_id) · rating / SE / rank score, runs per 100 H balls · tier
1. Jasprit Bumrah (2933) · 25.27 / 5.05 / 20.22 · tier 1
2. Sunil Narine (781) · 20.86 / 3.64 / 17.21 · tier 1
3. Matheesha Pathirana (8661) · 16.36 / 4.09 / 12.27 · tier 2
4. Nathan Ellis (7920) · 15.40 / 3.23 / 12.17 · tier 2
5. Rashid Khan (Afghanistan, Gujarat Titans) (4722) · 12.24 / 3.88 / 8.37 · tier 2
6. Eshan Malinga (39657) · 11.94 / 4.36 / 7.57 · tier 2
7. Josh Hazlewood (830) · 11.91 / 4.63 / 7.28 · tier 2
8. Bhuvneshwar Kumar (360) · 9.88 / 3.99 / 5.88 · tier 2
9. Mujeeb Ur Rahman (6450) · 9.82 / 4.42 / 5.41 · tier 2
10. Varun Chakravarthy (6256) · 7.61 / 3.68 / 3.92 · tier 2  — top 10
11. Mohammed Siraj (5944) · 8.63 / 4.76 / 3.88 · tier 2
12. Mitchell Santner (4101) · 7.35 / 4.07 / 3.28 · tier 2
13. Joshua Little (4807) · 7.44 / 4.40 / 3.04 · tier 2
14. Taskin Ahmed (2681) · 7.50 / 4.59 / 2.91 · tier 2
15. Lungi Ngidi (4641) · 7.92 / 5.40 / 2.52 · tier 2
16. Krunal Pandya (4219) · 6.12 / 4.34 / 1.78 · tier 2
17. Jason Holder (2633) · 4.59 / 3.65 / 0.95 · tier 2
18. Mark Adair (5379) · 3.94 / 3.58 / 0.36 · tier 2
19. Haris Rauf (7364) · 4.64 / 4.48 / 0.16 · tier 2
20. Shaheen Afridi (6355) · 5.34 / 5.19 / 0.15 · tier 2

## Phases 8–9
| # | Decision | Reason | Status |
|---|---|---|---|
| P8-1 | Out-of-time test rebuilds the Phase 7 rating stage on 2022–24 (activity window 2023–24); per-ball values come from league-level models fitted out-of-fold over all seasons (stated as a limitation) | full model retraining per window not implemented | recorded |
| P9-1 | Runs-to-wins conversion = WP value of one run on an average H delivery (WP model): 0.0143 → ≈70 runs per win; replacement = median rating of active players with ½·min to min H balls | data-derived | recorded |
| FIX-1 | Reproducibility bug (report text only, no effect on lists or numbers): owner-approved Phase 6/6B headline text and the Phase 6 band-check section were inserted into reports by hand and lost on a clean rebuild. Now generated by code; full rebuild verified (all phase checks pass, lists byte-identical to the locked copies) | genuine bug, fixed without changing results | recorded |
