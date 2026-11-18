# Phase 6B — Signal recovery (pre-specified methods A–E)

Phase 6B: pre-specified signal-recovery methods (A–E). Every result is reported, including failures.

**Pass rule (fixed before running):** a method recovers signal only if
  (1) its random-split reliability 95% interval lies above 0, AND
  (2) adding it to the best baseline improves out-of-sample prediction of next-season run value on H:
      bootstrap 95% interval (over players) of the gain in weighted correlation lies above 0.
Out-of-sample design: player-season pairs t -> t+1 (adjacent calendar years); this season ≥ 60 deliveries, next season
≥ 30 deliveries on H; weight = harmonic mean of the two; 5 folds grouped by player; linear calibration fitted on the
training folds. Baseline = shrunk all-delivery run value or shrunk LI-weighted run value, whichever predicts better.

## Headline findings
- **Pressure-specific variation exists on both sides** (mixed-model LRT p = 2.7e-8 batters, 6.2e-6 bowlers; slope SD ≈ 0.065 runs/ball), **but is too small relative to sampling noise to estimate reliably per player with 2022–26 data. Core skill is the best predictor of pressure performance.**
- **Death pace specialists** (method E within H) show the strongest pressure signal (reliability: P 0.47, economy 0.40, boundary suppression 0.53; n ≈ 45–51 bowlers) but fail the pre-specified rule; reported as a lead for further work, **not used in the ranking**.
- **Role-group pooling raised reliability without improving prediction**: the gain came from shared role means (both split halves share the same role mean), not from player-level signal.


## Summary: method × side × reliability × out-of-sample gain × pass/fail

| method | side | variant | players | reliability (random-50 SB) [95% CI] | OOS r baseline | OOS r with method | OOS gain [95% CI] | pairs | PASS |
|---|---|---|---|---|---|---|---|---|---|
| A pressure slope (mixed model BLUP) | bat | slope on standardised log(LI) | 317 | 0.226 [0.04, 0.41] | 0.14 | 0.107 | -0.033 [-0.062, -0.000] | 297.0 | False |
| B robust values | bat | P on H (raw) | 136 | 0.042 [-0.38, 0.35] |  |  | None  |  | False |
| B robust values | bat | P on H (winsorised 1/99) | 136 | 0.067 [-0.36, 0.35] |  |  | None  |  | False |
| B robust values | bat | run value on H (raw) | 136 | 0.206 [-0.08, 0.39] | 0.14 | 0.16 | 0.02 [-0.036, 0.071] | 297.0 | False |
| B robust values | bat | run value on H (winsorised 1/99) | 136 | 0.217 [-0.04, 0.40] | 0.14 | 0.158 | 0.018 [-0.036, 0.067] | 297.0 | False |
| B robust values | bat | process composite on H (raw) | 136 | 0.397 [0.19, 0.53] |  |  | None  |  | False |
| B robust values | bat | process composite on H (winsorised) | 136 | 0.419 [0.26, 0.54] |  |  | None  |  | False |
| B robust values | bat | process composite, all (raw) | 317 | 0.471 [0.35, 0.57] |  |  | None  |  | False |
| B robust values | bat | process composite, all (winsorised) | 317 | 0.469 [0.37, 0.57] |  |  | None  |  | False |
| B robust values | bat | run value on H (Huber, k=1.345, scale=1.191) | 136 | 0.336 [0.15, 0.47] | 0.14 | 0.12 | -0.02 [-0.043, 0.008] | 297.0 | False |
| C role-group pooling | bat | population pooling (c = 498.6) | 136 | 0.322 [0.06, 0.47] | 0.14 | 0.142 | 0.002 [-0.029, 0.028] | 297.0 | False |
| C role-group pooling | bat | role-group pooling (c = 592.5) | 136 | 0.581 [0.37, 0.71] | 0.14 | 0.14 | 0.0 [-0.024, 0.029] | 297.0 | False |
| D episode success rate | bat | run of consecutive balls faced in H | 133 | 0.347 [0.11, 0.49] | 0.14 | 0.133 | -0.007 [-0.018, 0.007] | 297.0 | False |
| A pressure slope (mixed model BLUP) | bowl | slope on standardised log(LI) | 338 | -0.071 [-0.29, 0.10] | -0.057 | 0.007 | 0.064 [-0.025, 0.167] | 292.0 | False |
| B robust values | bowl | P on H (raw) | 148 | -0.153 [-0.42, 0.05] |  |  | None  |  | False |
| B robust values | bowl | P on H (winsorised 1/99) | 148 | -0.167 [-0.42, 0.02] |  |  | None  |  | False |
| B robust values | bowl | run value on H (raw) | 148 | -0.187 [-0.46, 0.02] | -0.057 | -0.017 | 0.04 [-0.045, 0.128] | 292.0 | False |
| B robust values | bowl | run value on H (winsorised 1/99) | 148 | -0.238 [-0.50, -0.01] | -0.057 | -0.018 | 0.039 [-0.045, 0.130] | 292.0 | False |
| B robust values | bowl | process composite on H (raw) | 148 | -0.159 [-0.48, 0.05] |  |  | None  |  | False |
| B robust values | bowl | process composite on H (winsorised) | 148 | -0.169 [-0.45, 0.05] |  |  | None  |  | False |
| B robust values | bowl | process composite, all (raw) | 338 | 0.133 [-0.06, 0.25] |  |  | None  |  | False |
| B robust values | bowl | process composite, all (winsorised) | 338 | 0.109 [-0.07, 0.25] |  |  | None  |  | False |
| B robust values | bowl | run value on H (Huber, k=1.345, scale=1.153) | 148 | -0.115 [-0.40, 0.11] | -0.057 | -0.095 | -0.038 [-0.118, 0.035] | 292.0 | False |
| C role-group pooling | bowl | population pooling (c = 1926.3) | 148 | 0.057 [-0.21, 0.23] | -0.057 | -0.017 | 0.04 [-0.028, 0.122] | 292.0 | False |
| C role-group pooling | bowl | role-group pooling (c = 3360.6) | 148 | 0.526 [0.31, 0.67] | -0.057 | -0.068 | -0.011 [-0.068, 0.041] | 292.0 | False |
| D episode success rate | bowl | high-leverage over | 122 | 0.107 [-0.24, 0.36] | -0.057 | -0.067 | -0.01 [-0.091, 0.059] | 292.0 | False |
| E death pace specialists | bowl | pressure_v — death overs, all | 51 | -0.11 [-1.10, 0.33] |  |  | None  |  | False |
| E death pace specialists | bowl | pressure_v — death overs ∩ H | 45 | 0.473 [0.17, 0.60] |  |  | None  |  | False |
| E death pace specialists | bowl | economy_suppression — death overs, all | 51 | 0.087 [-0.40, 0.45] |  |  | None  |  | False |
| E death pace specialists | bowl | economy_suppression — death overs ∩ H | 45 | 0.401 [0.06, 0.59] |  |  | None  |  | False |
| E death pace specialists | bowl | dot_rate — death overs, all | 48 | 0.271 [-0.15, 0.52] |  |  | None  |  | False |
| E death pace specialists | bowl | dot_rate — death overs ∩ H | 43 | 0.215 [-0.34, 0.52] |  |  | None  |  | False |
| E death pace specialists | bowl | wicket_taking — death overs, all | 48 | -0.253 [-0.93, 0.17] |  |  | None  |  | False |
| E death pace specialists | bowl | wicket_taking — death overs ∩ H | 43 | -0.216 [-0.94, 0.12] |  |  | None  |  | False |
| E death pace specialists | bowl | boundary_suppression — death overs, all | 48 | -0.08 [-0.85, 0.30] |  |  | None  |  | False |
| E death pace specialists | bowl | boundary_suppression — death overs ∩ H | 43 | 0.533 [0.19, 0.70] |  |  | None  |  | False |
| E death pace specialists | bowl | rv — death overs, all | 51 | 0.003 [-0.78, 0.40] | -0.253 | 0.04 | 0.293 [0.162, 0.471] | 103.0 | False |
| E death pace specialists | bowl | rv — death overs ∩ H | 45 | 0.433 [0.07, 0.63] | -0.253 | -0.131 | 0.122 [-0.027, 0.291] | 103.0 | False |

## Batters

Out-of-sample pairs: 297 player-seasons, 140 players. Baselines (OOS weighted r for next-season run value on H): {'all-delivery run value': {'pairs': 297, 'players': 140, 'r_baseline': np.float64(0.14)}, 'LI-weighted run value': {'pairs': 297, 'players': 140, 'r_baseline': np.float64(0.076)}}. **Best baseline: all-delivery run value.**

### A. Mixed model

| side | deliveries | players | rv_sd_per_ball | slope_var (y-sd units²) | slope_var_ci95_parametric_bootstrap | intercept_var | cov | slope_sd_in_runs_per_ball | LRT_vs_intercept_only | p_value (chi-bar², 1/2 df mixture) | bootstrap_resamples | full_converged | null_converged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bat | 181630 | 555 | 2.2995 | 0.000842 | [0.000432, 0.001267] | 0.004269 | -0.001087 | 0.0667 | 33.72 | 2.69e-08 | 100 | True | True |

### B. Robust values (before/after)

| variant | players | split_half_random50 | ci95_random50 | split_half_odd_even | split_half_chronological | season_r_weighted | ci95_season |
|---|---|---|---|---|---|---|---|
| P on H (raw) | 136 | 0.042 | [-0.38, 0.35] | 0.051 | 0.28 | 0.062 | [-0.12, 0.25] |
| P on H (winsorised 1/99) | 136 | 0.067 | [-0.36, 0.35] | 0.045 | 0.283 | 0.063 | [-0.15, 0.24] |
| run value on H (raw) | 136 | 0.206 | [-0.08, 0.39] | 0.125 | 0.386 | 0.067 | [-0.14, 0.23] |
| run value on H (winsorised 1/99) | 136 | 0.217 | [-0.04, 0.40] | 0.133 | 0.393 | 0.074 | [-0.09, 0.29] |
| process composite on H (raw) | 136 | 0.397 | [0.19, 0.53] | 0.386 | 0.506 | 0.144 | [-0.01, 0.30] |
| process composite on H (winsorised) | 136 | 0.419 | [0.26, 0.54] | 0.391 | 0.51 | 0.148 | [-0.05, 0.34] |
| process composite, all (raw) | 317 | 0.471 | [0.35, 0.57] | 0.503 | 0.465 | 0.308 | [0.15, 0.42] |
| process composite, all (winsorised) | 317 | 0.469 | [0.37, 0.57] | 0.506 | 0.467 | 0.309 | [0.14, 0.47] |
| run value on H (Huber, k=1.345, scale=1.191) | 136 | 0.336 | [0.15, 0.47] | 0.267 | 0.447 | 0.174 | [-0.09, 0.40] |

### C. Role-group pooling

| variant | players | shrinkage_c | between_player_var | split_half_random50 | ci95_random50 | season_r_weighted |
|---|---|---|---|---|---|---|
| population pooling | 136 | 498.6 | 0.01119 | 0.322 | [0.06, 0.47] | 0.174 |
| role-group pooling | 136 | 592.5 | 0.00942 | 0.581 | [0.37, 0.71] | 0.177 |

Role means of run value on H: {'No.3': -0.0228, 'finisher (6-7)': 0.0773, 'lower order (8+)': -0.1144, 'middle order (4-5)': 0.0606, 'opener': 0.0721}

### D. Episode success rate

| episode_unit | episodes | mean_success | prior_a+b (episodes) | players | split_half_random50 | ci95_random50 | season_r_weighted | ci95_season |
|---|---|---|---|---|---|---|---|---|
| run of consecutive balls faced in H | 8358 | 0.466 | 83.4 | 133 | 0.347 | [0.11, 0.49] | 0.081 | [-0.15, 0.31] |

## Bowlers

Out-of-sample pairs: 292 player-seasons, 143 players. Baselines (OOS weighted r for next-season run value on H): {'all-delivery run value': {'pairs': 292, 'players': 143, 'r_baseline': np.float64(-0.057)}, 'LI-weighted run value': {'pairs': 292, 'players': 143, 'r_baseline': np.float64(-0.141)}}. **Best baseline: all-delivery run value.**

### A. Mixed model

| side | deliveries | players | rv_sd_per_ball | slope_var (y-sd units²) | slope_var_ci95_parametric_bootstrap | intercept_var | cov | slope_sd_in_runs_per_ball | LRT_vs_intercept_only | p_value (chi-bar², 1/2 df mixture) | bootstrap_resamples | full_converged | null_converged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bowl | 191967 | 521 | 2.2802 | 0.000794 | [0.000365, 0.001192] | 0.001451 | -9.2e-05 | 0.0642 | 22.9 | 6.17e-06 | 100 | True | True |

### B. Robust values (before/after)

| variant | players | split_half_random50 | ci95_random50 | split_half_odd_even | split_half_chronological | season_r_weighted | ci95_season |
|---|---|---|---|---|---|---|---|
| P on H (raw) | 148 | -0.153 | [-0.42, 0.05] | -0.281 | -0.276 | 0.074 | [-0.06, 0.19] |
| P on H (winsorised 1/99) | 148 | -0.167 | [-0.42, 0.02] | -0.268 | -0.269 | 0.079 | [-0.06, 0.18] |
| run value on H (raw) | 148 | -0.187 | [-0.46, 0.02] | -0.225 | -0.167 | 0.104 | [-0.04, 0.22] |
| run value on H (winsorised 1/99) | 148 | -0.238 | [-0.50, -0.01] | -0.22 | -0.165 | 0.104 | [-0.04, 0.23] |
| process composite on H (raw) | 148 | -0.159 | [-0.48, 0.05] | -0.049 | -0.137 | 0.098 | [-0.11, 0.28] |
| process composite on H (winsorised) | 148 | -0.169 | [-0.45, 0.05] | -0.056 | -0.143 | 0.094 | [-0.05, 0.23] |
| process composite, all (raw) | 338 | 0.133 | [-0.06, 0.25] | 0.04 | 0.028 | 0.143 | [0.05, 0.24] |
| process composite, all (winsorised) | 338 | 0.109 | [-0.07, 0.25] | 0.038 | 0.025 | 0.142 | [0.04, 0.24] |
| run value on H (Huber, k=1.345, scale=1.153) | 148 | -0.115 | [-0.40, 0.11] | -0.066 | -0.211 | 0.045 | [-0.13, 0.22] |

### C. Role-group pooling

| variant | players | shrinkage_c | between_player_var | split_half_random50 | ci95_random50 | season_r_weighted |
|---|---|---|---|---|---|---|
| population pooling | 148 | 1926.3 | 0.00278 | 0.057 | [-0.21, 0.23] | -0.202 |
| role-group pooling | 148 | 3360.6 | 0.00160 | 0.526 | [0.31, 0.67] | -0.159 |

Role means of run value on H: {'pace, mainly death': 0.0025, 'pace, mainly middle': -0.0845, 'pace, mainly powerplay': 0.0012, 'spin, mainly death': 0.0266, 'spin, mainly middle': 0.0039, 'spin, mainly powerplay': 0.0396}

### D. Episode success rate

| episode_unit | episodes | mean_success | prior_a+b (episodes) | players | split_half_random50 | ci95_random50 | season_r_weighted | ci95_season |
|---|---|---|---|---|---|---|---|---|
| high-leverage over | 8085 | 0.477 | 1838.3 | 122 | 0.107 | [-0.24, 0.36] | 0.059 | [-0.12, 0.20] |

### E. Death pace specialists (145 bowlers: pace, ≥30% of legal balls at the death)

| variant | players | split_half_random50 | ci95_random50 | split_half_odd_even | split_half_chronological | season_r_weighted | ci95_season |
|---|---|---|---|---|---|---|---|
| pressure_v — death overs, all | 51 | -0.11 | [-1.10, 0.33] | -0.068 | -0.865 | 0.061 | [-0.25, 0.35] |
| pressure_v — death overs ∩ H | 45 | 0.473 | [0.17, 0.60] | 0.451 | 0.296 | 0.132 | [-0.19, 0.40] |
| economy_suppression — death overs, all | 51 | 0.087 | [-0.40, 0.45] | -0.17 | -0.61 | 0.146 | [-0.14, 0.40] |
| economy_suppression — death overs ∩ H | 45 | 0.401 | [0.06, 0.59] | 0.412 | 0.188 | 0.152 | [-0.19, 0.46] |
| dot_rate — death overs, all | 48 | 0.271 | [-0.15, 0.52] | 0.093 | 0.094 | 0.251 | [0.09, 0.38] |
| dot_rate — death overs ∩ H | 43 | 0.215 | [-0.34, 0.52] | 0.19 | 0.173 | -0.016 | [-0.26, 0.33] |
| wicket_taking — death overs, all | 48 | -0.253 | [-0.93, 0.17] | -0.092 | -0.449 | -0.007 | [-0.19, 0.16] |
| wicket_taking — death overs ∩ H | 43 | -0.216 | [-0.94, 0.12] | 0.091 | 0.023 | -0.195 | [-0.42, -0.00] |
| boundary_suppression — death overs, all | 48 | -0.08 | [-0.85, 0.30] | -0.366 | -0.875 | 0.114 | [-0.19, 0.38] |
| boundary_suppression — death overs ∩ H | 43 | 0.533 | [0.19, 0.70] | 0.529 | 0.465 | -0.009 | [-0.38, 0.38] |
| rv — death overs, all | 51 | 0.003 | [-0.78, 0.40] | -0.001 | -0.778 | 0.145 | [-0.21, 0.41] |
| rv — death overs ∩ H | 45 | 0.433 | [0.07, 0.63] | 0.447 | 0.171 | 0.13 | [-0.18, 0.39] |