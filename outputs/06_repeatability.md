# Phase 6 — Repeatability, target and weights

## B. Pressure-band check (run first; decision P6-B: H25 within innings kept)

Split-half reliability (odd vs even matches in date order within player, Spearman–Brown corrected, 200 bootstrap resamples over players for 95% intervals). Eligible: ≥100 band deliveries and ≥25 in each half. `common set` = the players eligible under H25, so differences there come from the band, not from who is included.

### How much each version concentrates leverage

| version | share_of_deliveries | mean_LI_in_set | share_of_top25_deliveries_in_set |
|---|---|---|---|
| H25 (primary) | 0.25 | 1.816 | 1.0 |
| H33 | 0.333 | 1.669 | 0.75 |
| H50 | 0.5 | 1.465 | 0.5 |
| LI-weighted, LI > median | 0.5 | 2.395 | 0.62 |

### Reliability by version

| side | metric | version | players | reliability | ci95 | players_common_set | reliability_common_set | ci95_common_set |
|---|---|---|---|---|---|---|---|---|
| bat | pressure_v | H25 (primary) | 135 | 0.051 | [-0.49, 0.35] | 135 | 0.051 | [-0.44, 0.41] |
| bat | pressure_v | H33 | 168 | 0.155 | [-0.14, 0.40] | 135 | 0.246 | [-0.12, 0.49] |
| bat | pressure_v | H50 | 203 | -0.196 | [-0.68, 0.19] | 135 | -0.111 | [-0.89, 0.28] |
| bat | pressure_v | LI-weighted, LI > median | 203 | -0.219 | [-0.84, 0.18] | 135 | -0.092 | [-0.56, 0.32] |
| bat | sr_delta | H25 (primary) | 135 | 0.369 | [0.11, 0.56] | 135 | 0.369 | [0.09, 0.56] |
| bat | sr_delta | H33 | 168 | 0.51 | [0.33, 0.66] | 135 | 0.515 | [0.23, 0.68] |
| bat | sr_delta | H50 | 200 | 0.366 | [0.11, 0.53] | 135 | 0.481 | [0.24, 0.64] |
| bat | sr_delta | LI-weighted, LI > median | 200 | 0.248 | [-0.09, 0.47] | 135 | 0.237 | [-0.33, 0.52] |
| bat | dot_suppression | H25 (primary) | 135 | 0.477 | [0.29, 0.63] | 135 | 0.477 | [0.27, 0.63] |
| bat | dot_suppression | H33 | 168 | 0.426 | [0.25, 0.55] | 135 | 0.495 | [0.34, 0.64] |
| bat | dot_suppression | H50 | 200 | 0.266 | [0.01, 0.46] | 135 | 0.453 | [0.27, 0.60] |
| bat | dot_suppression | LI-weighted, LI > median | 200 | 0.349 | [0.13, 0.53] | 135 | 0.475 | [0.29, 0.61] |
| bat | boundary_elevation | H25 (primary) | 135 | 0.392 | [0.17, 0.59] | 135 | 0.392 | [0.11, 0.56] |
| bat | boundary_elevation | H33 | 168 | 0.567 | [0.35, 0.71] | 135 | 0.515 | [0.23, 0.68] |
| bat | boundary_elevation | H50 | 200 | 0.516 | [0.34, 0.63] | 135 | 0.555 | [0.31, 0.70] |
| bat | boundary_elevation | LI-weighted, LI > median | 200 | 0.455 | [0.24, 0.61] | 135 | 0.408 | [0.08, 0.63] |
| bat | wicket_security | H25 (primary) | 135 | 0.032 | [-0.52, 0.36] | 135 | 0.032 | [-0.44, 0.39] |
| bat | wicket_security | H33 | 168 | 0.045 | [-0.30, 0.34] | 135 | 0.001 | [-0.45, 0.38] |
| bat | wicket_security | H50 | 200 | 0.04 | [-0.28, 0.28] | 135 | 0.124 | [-0.27, 0.38] |
| bat | wicket_security | LI-weighted, LI > median | 200 | -0.161 | [-0.55, 0.14] | 135 | -0.211 | [-0.80, 0.22] |
| bowl | pressure_v | H25 (primary) | 148 | -0.281 | [-0.76, 0.09] | 148 | -0.281 | [-0.67, 0.04] |
| bowl | pressure_v | H33 | 183 | -0.051 | [-0.35, 0.22] | 148 | 0.003 | [-0.37, 0.31] |
| bowl | pressure_v | H50 | 223 | -0.107 | [-0.48, 0.20] | 148 | -0.124 | [-0.48, 0.14] |
| bowl | pressure_v | LI-weighted, LI > median | 223 | 0.03 | [-0.33, 0.29] | 148 | -0.022 | [-0.50, 0.27] |
| bowl | economy_suppression | H25 (primary) | 148 | 0.129 | [-0.19, 0.36] | 148 | 0.129 | [-0.15, 0.37] |
| bowl | economy_suppression | H33 | 183 | -0.125 | [-0.55, 0.16] | 148 | 0.129 | [-0.21, 0.41] |
| bowl | economy_suppression | H50 | 223 | -0.036 | [-0.44, 0.19] | 148 | -0.201 | [-0.67, 0.21] |
| bowl | economy_suppression | LI-weighted, LI > median | 223 | 0.09 | [-0.21, 0.30] | 148 | -0.103 | [-0.50, 0.22] |
| bowl | dot_rate | H25 (primary) | 141 | 0.134 | [-0.25, 0.45] | 141 | 0.134 | [-0.34, 0.44] |
| bowl | dot_rate | H33 | 178 | 0.119 | [-0.25, 0.34] | 141 | 0.177 | [-0.12, 0.42] |
| bowl | dot_rate | H50 | 218 | -0.009 | [-0.32, 0.23] | 141 | 0.366 | [0.05, 0.57] |
| bowl | dot_rate | LI-weighted, LI > median | 218 | -0.033 | [-0.46, 0.26] | 141 | 0.25 | [-0.06, 0.47] |
| bowl | wicket_taking | H25 (primary) | 141 | -0.04 | [-0.37, 0.24] | 141 | -0.04 | [-0.46, 0.25] |
| bowl | wicket_taking | H33 | 178 | 0.042 | [-0.42, 0.31] | 141 | 0.005 | [-0.53, 0.34] |
| bowl | wicket_taking | H50 | 218 | -0.079 | [-0.47, 0.18] | 141 | 0.148 | [-0.25, 0.41] |
| bowl | wicket_taking | LI-weighted, LI > median | 218 | 0.049 | [-0.29, 0.29] | 141 | 0.056 | [-0.35, 0.33] |
| bowl | boundary_suppression | H25 (primary) | 141 | 0.082 | [-0.21, 0.36] | 141 | 0.082 | [-0.22, 0.33] |
| bowl | boundary_suppression | H33 | 178 | -0.131 | [-0.55, 0.16] | 141 | -0.046 | [-0.40, 0.26] |
| bowl | boundary_suppression | H50 | 218 | 0.11 | [-0.20, 0.34] | 141 | -0.144 | [-0.68, 0.23] |
| bowl | boundary_suppression | LI-weighted, LI > median | 218 | 0.144 | [-0.16, 0.35] | 141 | -0.127 | [-0.68, 0.24] |

## Headline finding
**What repeats under pressure is process, not outcome.** Split-half reliability (mean of 50 random match splits, Spearman–Brown, 95% bootstrap interval), H = top 25% LI within innings:
- Repeats (batters): strike-rate delta 0.43 [0.26, 0.55], dot suppression 0.42 [0.26, 0.54], boundary elevation 0.45 [0.29, 0.58]; process composite on H 0.41 [0.23, 0.52]; run-out avoidance 0.23 [0.05, 0.37].
- Does not repeat (batters): WP value P 0.02 [−0.49, 0.31], wicket security 0.19 [−0.11, 0.36], retention R 0.06 [−0.35, 0.31], entry resilience, collapse resistance, post-wicket response, squeeze (pooled), partner effect.
- Bowlers: no H metric repeats individually — P −0.14, economy −0.12, dots 0.06, wickets 0.14, boundaries −0.11, bounce-back, post-wicket, execution drift, retention R (all intervals span 0). Only variation (entropy H − L) passes, 0.22 [0.06, 0.38], and it predicts next-season performance *negatively* (r −0.13 [−0.22, −0.03]).
- True variance of pressure-specific run value (H beyond what L predicts): batters 0.0029 runs²/ball [−0.012, 0.014] (26% of true H variance, interval spans 0); bowlers ≈ 0 (true H variance itself estimated negative). Phase 7's empirical Bayes H update is therefore small for batters and ≈ 0 for bowlers.
- Out of sample (next season, grouped by player), baseline skill alone predicts the pressure target better than weighted pressure metrics: batters r 0.16 [0.01, 0.30] vs ridge 0.07 / equal weights 0.10; bowlers 0.07 [−0.02, 0.18] vs ridge −0.12 / equal −0.13.

## C. Repeatability of every candidate (trust table)

Primary figure: mean Spearman–Brown split-half over 50 random splits of each player's matches into two halves; 95% interval from 200 bootstrap resamples of players. Also: odd/even matches in date order, and chronological (earlier half vs later half of each player's matches). Season t → t+1: all adjacent calendar-year pairs pooled (match dates, so split seasons such as 2022/23 fall in the year of each match), weighted by the harmonic mean of deliveries; **season results reflect regular players only** (≥ half the minimum deliveries in both seasons). Minimum career deliveries in the measured set: skill/value/target 100, temperament 30, variation 60 (H), partner 100; ≥4 matches; each half needs ≥ ¼ of the minimum. Eligible players differ by metric (column `players`).

Pass rule: the random-split 95% interval lies above 0.

| side | measure | role | set | min_deliveries | players | split_half_random50 | ci95_random50 | players_in_splits | split_half_odd_even | ci95_odd_even | split_half_chronological | ci95_chronological | season_pairs | season_players | season_r_weighted | ci95_season | passes (random-split CI > 0) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bat | pressure_v | candidate | H | 100 | 136 | 0.023 | [-0.49, 0.31] | 135 | 0.051 | [-0.41, 0.38] | 0.28 | [-0.09, 0.55] | 125 | 69 | 0.062 | [-0.10, 0.25] | False |
| bat | sr_delta | candidate | H | 100 | 136 | 0.426 | [0.26, 0.55] | 135 | 0.369 | [0.06, 0.55] | 0.542 | [0.30, 0.69] | 122 | 69 | 0.171 | [-0.02, 0.34] | True |
| bat | dot_suppression | candidate | H | 100 | 136 | 0.416 | [0.26, 0.54] | 135 | 0.477 | [0.28, 0.61] | 0.413 | [0.16, 0.58] | 122 | 69 | 0.322 | [0.12, 0.48] | True |
| bat | boundary_elevation | candidate | H | 100 | 136 | 0.453 | [0.29, 0.58] | 135 | 0.392 | [0.17, 0.60] | 0.49 | [0.17, 0.65] | 122 | 69 | 0.18 | [0.01, 0.34] | True |
| bat | wicket_security | candidate | H | 100 | 136 | 0.189 | [-0.11, 0.36] | 135 | 0.032 | [-0.42, 0.40] | 0.343 | [0.07, 0.54] | 122 | 69 | 0.082 | [-0.09, 0.24] | False |
| bat | squeeze_response (pooled H+L) | candidate | pooled | 30 | 46 | 0.155 | [-0.18, 0.39] | 46 | 0.058 | [-0.57, 0.44] | 0.223 | [-0.23, 0.54] | 13 | 11 | 0.458 | [-0.11, 0.79] | False |
| bat | entry_resilience | candidate | H | 30 | 135 | -0.042 | [-0.35, 0.19] | 133 | 0.128 | [-0.22, 0.33] | -0.388 | [-0.93, 0.09] | 91 | 53 | -0.021 | [-0.22, 0.22] | False |
| bat | collapse_resistance | candidate | H | 30 | 87 | 0.132 | [-0.17, 0.35] | 83 | 0.174 | [-0.18, 0.43] | 0.193 | [-0.09, 0.41] | 49 | 36 | 0.128 | [-0.09, 0.36] | False |
| bat | post_wicket_response | candidate | H | 30 | 184 | -0.037 | [-0.34, 0.20] | 183 | -0.125 | [-0.63, 0.27] | -0.011 | [-0.46, 0.29] | 170 | 94 | 0.012 | [-0.10, 0.11] | False |
| bat | runout_avoidance | candidate | H | 30 | 349 | 0.226 | [0.05, 0.37] | 340 | 0.301 | [0.02, 0.53] | 0.307 | [0.04, 0.48] | 526 | 234 | 0.129 | [-0.03, 0.27] | True |
| bat | partner_effect (on trial) | candidate | H | 100 | 135 | 0.153 | [-0.07, 0.28] | 134 | 0.262 | [0.01, 0.43] | 0.172 | [-0.10, 0.42] | 120 | 66 | 0.004 | [-0.21, 0.22] | False |
| bat | retention R (P − B) | condition | H-L | 100 | 136 | 0.057 | [-0.35, 0.31] | 135 | -0.053 | [-0.52, 0.29] | 0.147 | [-0.29, 0.47] | 125 | 69 | 0.085 | [-0.10, 0.28] | False |
| bowl | pressure_v | candidate | H | 100 | 148 | -0.143 | [-0.41, 0.06] | 147 | -0.281 | [-0.76, 0.06] | -0.276 | [-0.72, 0.11] | 144 | 71 | 0.074 | [-0.03, 0.20] | False |
| bowl | economy_suppression | candidate | H | 100 | 148 | -0.122 | [-0.45, 0.12] | 147 | 0.129 | [-0.17, 0.36] | -0.076 | [-0.44, 0.21] | 144 | 71 | 0.144 | [-0.01, 0.32] | False |
| bowl | dot_rate | candidate | H | 100 | 141 | 0.062 | [-0.24, 0.28] | 140 | 0.134 | [-0.43, 0.44] | 0.229 | [-0.11, 0.44] | 137 | 69 | 0.189 | [-0.02, 0.38] | False |
| bowl | wicket_taking | candidate | H | 100 | 141 | 0.144 | [-0.07, 0.35] | 140 | -0.04 | [-0.51, 0.24] | 0.121 | [-0.28, 0.39] | 137 | 69 | 0.102 | [-0.03, 0.22] | False |
| bowl | boundary_suppression | candidate | H | 100 | 141 | -0.114 | [-0.44, 0.15] | 140 | 0.082 | [-0.27, 0.33] | -0.116 | [-0.52, 0.18] | 137 | 69 | 0.135 | [-0.04, 0.29] | False |
| bowl | bounce_back | candidate | H | 30 | 94 | 0.019 | [-0.33, 0.24] | 94 | -0.23 | [-0.77, 0.18] | 0.183 | [-0.17, 0.46] | 89 | 50 | 0.133 | [-0.09, 0.31] | False |
| bowl | post_wicket_response | candidate | H | 30 | 181 | 0.117 | [-0.15, 0.31] | 179 | -0.039 | [-0.44, 0.26] | 0.232 | [-0.04, 0.43] | 180 | 94 | -0.044 | [-0.19, 0.09] | False |
| bowl | execution_drift (H − L) | candidate | H-L | 100 | 148 | -0.07 | [-0.40, 0.15] | 146 | -0.285 | [-0.68, 0.05] | -0.404 | [-1.22, 0.12] | 144 | 71 | -0.004 | [-0.16, 0.16] | False |
| bowl | variation (entropy H − L) | candidate | H-L | 60 | 203 | 0.222 | [0.06, 0.38] | 200 | 0.259 | [-0.04, 0.50] | 0.089 | [-0.24, 0.34] | 233 | 110 | 0.113 | [-0.01, 0.22] | True |
| bowl | retention R (P − B) | condition | H-L | 100 | 148 | 0.023 | [-0.20, 0.22] | 146 | 0.004 | [-0.36, 0.27] | -0.232 | [-0.82, 0.15] | 144 | 71 | -0.055 | [-0.23, 0.15] | False |
| bat | TARGET run value on H | target | H | 100 | 136 | 0.196 | [-0.11, 0.37] | 135 | 0.125 | [-0.28, 0.39] | 0.386 | [0.10, 0.54] | 122 | 69 | 0.067 | [-0.13, 0.27] | False |
| bat | TARGET run value, all deliveries | target | all | 100 | 317 | 0.258 | [0.10, 0.39] | 313 | 0.336 | [0.13, 0.48] | 0.274 | [0.03, 0.42] | 503 | 202 | 0.191 | [0.08, 0.31] | True |
| bat | TARGET run value, all, LI-weighted | target | LIw | 100 | 317 | 0.222 | [0.05, 0.35] | 314 | 0.226 | [-0.04, 0.41] | 0.248 | [0.01, 0.42] | 503 | 202 | 0.185 | [0.06, 0.28] | True |
| bat | TARGET process composite on H | target | H | 100 | 136 | 0.412 | [0.23, 0.52] | 135 | 0.386 | [0.13, 0.56] | 0.506 | [0.31, 0.65] | 122 | 69 | 0.144 | [-0.06, 0.30] | True |
| bat | TARGET process composite, all deliveries | target | all | 100 | 317 | 0.47 | [0.34, 0.59] | 313 | 0.503 | [0.37, 0.62] | 0.465 | [0.29, 0.59] | 503 | 202 | 0.308 | [0.14, 0.44] | True |
| bowl | TARGET run value on H | target | H | 100 | 148 | -0.207 | [-0.46, 0.02] | 147 | -0.225 | [-0.64, 0.12] | -0.167 | [-0.53, 0.08] | 144 | 71 | 0.104 | [-0.07, 0.23] | False |
| bowl | TARGET run value, all deliveries | target | all | 100 | 338 | 0.126 | [-0.04, 0.26] | 337 | 0.079 | [-0.15, 0.29] | 0.071 | [-0.21, 0.28] | 491 | 212 | 0.095 | [0.01, 0.19] | False |
| bowl | TARGET run value, all, LI-weighted | target | LIw | 100 | 338 | 0.189 | [0.00, 0.32] | 337 | 0.091 | [-0.19, 0.29] | 0.161 | [-0.05, 0.34] | 491 | 212 | 0.063 | [-0.02, 0.15] | True |
| bowl | TARGET process composite on H | target | H | 100 | 148 | -0.127 | [-0.44, 0.07] | 148 | -0.049 | [-0.46, 0.25] | -0.137 | [-0.58, 0.19] | 144 | 71 | 0.098 | [-0.08, 0.24] | False |
| bowl | TARGET process composite, all deliveries | target | all | 100 | 338 | 0.127 | [-0.07, 0.25] | 337 | 0.04 | [-0.16, 0.23] | 0.028 | [-0.26, 0.26] | 491 | 212 | 0.143 | [0.04, 0.22] | False |

Charts: `outputs/charts/repeatability_batters.png`, `repeatability_bowlers.png`, `repeatability_targets.png` (earlier vs later half per metric).

## D. Target

- **Batters: TARGET process composite on H** — random-split reliability 0.412 [0.23, 0.52]; season-to-season r 0.144 [-0.06, 0.30].
- **Bowlers: TARGET run value, all, LI-weighted** — random-split reliability 0.189 [0.00, 0.32]; season-to-season r 0.063 [-0.02, 0.15].

### Pressure-specific deviation (for the Phase 7 empirical Bayes update)

Method of moments on career run value per delivery, players with ≥100 deliveries on both H and L (L used as the clean baseline: its sampling noise is independent of H). tau2 = observed between-player variance − mean sampling variance. The pressure-specific residual = true H variance not explained by true L performance.

| side | quantity | estimate | ci_lo | ci_hi | share_of_tau2_H | players | mean_sampling_var_H |
|---|---|---|---|---|---|---|---|
| bat | true var of H run value (tau2_H) | 0.011021 | -0.000264 | 0.021748 | 1.0 | 136 | 0.028862 |
| bat | true var of L run value (tau2_L) | 0.0136 | 0.003277 | 0.024878 | 1.233991 | 136 | 0.028862 |
| bat | true cov(H, L) | 0.010524 | 0.003382 | 0.017535 | 0.954883 | 136 | 0.028862 |
| bat | true var of (H − L) deviation | 0.003573 | -0.010928 | 0.018406 | 0.324224 | 136 | 0.028862 |
| bat | pressure-specific residual var: tau2_H − cov²/tau2_L | 0.002878 | -0.01163 | 0.014256 | 0.261095 | 136 | 0.028862 |
| bowl | true var of H run value (tau2_H) | -0.001524 | -0.006108 | 0.002806 | 1.0 | 147 | 0.028276 |
| bowl | true var of L run value (tau2_L) | 0.006853 | 0.00159 | 0.012616 | -4.495664 | 147 | 0.028276 |
| bowl | true cov(H, L) | 0.001003 | -0.003525 | 0.005141 | -0.657767 | 147 | 0.028276 |
| bowl | true var of (H − L) deviation | 0.003323 | -0.007577 | 0.013306 | -2.180127 | 147 | 0.028276 |
| bowl | pressure-specific residual var: tau2_H − cov²/tau2_L | -0.001671 | -0.008224 | 0.00258 | 1.096239 | 147 | 0.028276 |

### Weights

Non-negative ridge predicting next-season target from this season's shrunk, centred, standardised metrics (only metrics whose random-split interval is above 0). Folds grouped by player; ridge penalty chosen by inner grouped CV inside each training fold, and by grouped CV on all data for the final weights. Shrinkage θ = n/(n + c)·(M − mean); c from within/between variance (mean-type metrics) or from measured reliability.

| side | metric | ridge_coef | weight | shrinkage_c |
|---|---|---|---|---|
| bat | sr_delta | 0.029009 | 0.2285 | 223.3 |
| bat | dot_suppression | 0.059926 | 0.472 | 253.9 |
| bat | boundary_elevation | 0.006558 | 0.0516 | 343.6 |
| bat | runout_avoidance | 0.031477 | 0.2479 | 17.3 |
| bowl | variation (entropy H − L) | 0.006232 | 1.0 | 210.3 |

Ridge details: {'bat': {'alpha_per_fold': [1000, 1000, 1000, 1000, 1000], 'alpha_final': 1000, 'ridge_minus_equal_r': '-0.019 [-0.077, 0.025]', 'z_feature_means': [-0.0, 0.0, 0.0, 0.0]}, 'bowl': {'alpha_per_fold': [1000, 1000, 1000, 1000, 1000], 'alpha_final': 1000, 'ridge_minus_equal_r': '0.009 [0.002, 0.018]', 'z_feature_means': [0.0]}}

### D3 — out-of-sample comparison (grouped by player)

| side | model | player_seasons | players | oof_weighted_r | ci95 | oof_R2 |
|---|---|---|---|---|---|---|
| bat | ridge (non-negative) | 297 | 140 | 0.072 | [-0.07, 0.22] | -0.0072 |
| bat | equal weights (passed metrics) | 297 | 140 | 0.096 | [-0.03, 0.22] | 0.0046 |
| bat | baseline skill only | 297 | 140 | 0.161 | [0.01, 0.30] | 0.0248 |
| bowl | ridge (non-negative) | 453 | 198 | -0.123 | [-0.21, -0.05] | -0.013 |
| bowl | equal weights (passed metrics) | 453 | 198 | -0.131 | [-0.22, -0.03] | -0.0138 |
| bowl | baseline skill only | 453 | 198 | 0.068 | [-0.02, 0.18] | 0.0017 |