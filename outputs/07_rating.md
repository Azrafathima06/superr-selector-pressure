# Phase 7 — Final rating and ranking

Phase 7: rating and ranking (DECISIONS.md P6B-4, P7-*, P7R-*, P7B-*).

Bivariate, component-wise empirical Bayes (second review). For each component k (runs; wicket = dismissals vs expected
x wicket cost from the Phase 2 resource table):
  H performance = core skill mu_k + pressure-specific deviation delta_k
  mu_k   : mean over ALL deliveries, shrunk toward the population by its own between-player variance
           (c_k = within-player per-ball variance / between-player variance, method of moments)
  delta_k: d = (H mean − L mean) − population (H − L) shift, shrunk by var(delta_k) estimated from data
           (var(d) − mean sampling variance; 0 if not positive -> delta_hat = 0)
  rating = sum_k (mu_hat_k + delta_hat_k). A component whose between-player variance is not estimable is a constant
  (dropped). Pressure deliveries count as evidence of core skill once; the H−L contrast carries only the deviation.
SE = bootstrap over the player's matches (200; hyper-parameters fixed). Rank score = rating − SE.
Eligibility: ≥ 2 × median H deliveries of regular player-seasons (≥10 matches); played in 2025 or 2026; retention gate
(exclude if centred R 95% interval entirely below 0); noise-aware delivery condition: exclude only if raw run value on H is
clearly below expectation (raw H + 1.28·SE < 0, i.e. the 80% interval lies below 0).
Tiers: walking down the ordered list, a new tier starts when the gap to the previous player's rank score exceeds the larger
of the two players' SEs.

## Final method (owner decision, Phase 7 rule: out-of-sample prediction first, then use of evidence)

- **Batters: L-only component prior + component-wise H update** (prediction r 0.184 vs 0.103 for bivariate). The bivariate batter rating is kept as a tested alternative (`outputs/tables/appendix_bivariate_batters.csv`).
- **Bowlers: bivariate component rating** (prediction equivalent, 0.028 vs 0.019; uses all deliveries as evidence of core skill).
- **Delivery condition: a requirement, not an estimate.** To appear in a final ranked list a player must have demonstrated delivery under pressure — raw run value on H above expectation (> 0) (brief: 'deliver when it matters'). The noise-aware version (exclude only if raw H + 1.28·SE < 0) is reported as a sensitivity check.
- **Retention gate on pooled resamples:** SEs and raw-H SEs use 2,000 bootstrap resamples (fixed seed 0); the retention interval pools 5 seeds x 2,000 = 10,000 resamples and the gate is applied once (exclude only if the 95% interval is entirely below 0). Players whose single-seed decisions would disagree are reported as borderline, not excluded.

## Changes after the second Phase 7 review (owner instruction)

1. Bivariate empirical Bayes replaces the L-only prior: core skill from all deliveries + shrunk pressure-specific deviation from (H − L), component-wise.
2. Bowler wicket component re-estimated on all deliveries.
3. Noise-aware delivery condition: excluded only if raw H + 1.28·SE < 0.
4. Tiers mark positions that are not statistically separable.

## Out-of-sample: bivariate rating vs the previous (L-only prior) rating

Prediction of next-season run value on H (season-t ratings, hyper-parameters from career data; folds grouped by player).

| side | rating | oos_r_next_season_rv_H | ci95 | pairs |
|---|---|---|---|---|
| bat | previous (L-only component prior + H update) | 0.184 | [-0.018, 0.366] | 297 |
| bat | bivariate (core skill from all deliveries + shrunk H−L deviation) | 0.103 | [-0.038, 0.246] | 297 |
| bowl | previous (L-only component prior + H update) | 0.019 | [-0.103, 0.157] | 292 |
| bowl | bivariate (core skill from all deliveries + shrunk H−L deviation) | 0.028 | [-0.087, 0.140] | 292 |

| side | bivariate_minus_previous_r | ci95 | clearly_worse |
|---|---|---|---|
| bat | -0.082 | [-0.144, 0.011] | False |
| bowl | 0.008 | [-0.073, 0.080] | False |

## Recency weighting on core skill (all deliveries, both components)

**Batters**
| lambda | oos_r | ci95 |
|---|---|---|
| 1.0 | 0.122 | [-0.013, 0.266] |
| 0.8 | 0.133 | [-0.008, 0.288] |
| 0.6 | 0.14 | [-0.009, 0.296] |
| 0.4 | 0.142 | [-0.007, 0.301] |

{'best_lambda': 0.4, 'gain_vs_no_recency': np.float64(0.02), 'gain_ci95': '[-0.033, 0.066]', 'kept': False}

**Bowlers**
| lambda | oos_r | ci95 |
|---|---|---|
| 1.0 | 0.032 | [-0.082, 0.134] |
| 0.8 | 0.03 | [-0.084, 0.134] |
| 0.6 | 0.029 | [-0.084, 0.130] |
| 0.4 | 0.03 | [-0.090, 0.128] |

{'best_lambda': 0.8, 'gain_vs_no_recency': np.float64(-0.002), 'gain_ci95': '[-0.016, 0.011]', 'kept': False}

## Fitted parameters and eligibility counts per variant

| variant | side | method | H | min_H | tau2_dev_used | tau2_dev_estimate | tau2_dev_ci95 | prior_coef | dropped_components | c_runs | c_wkt | reliability_at_median_nL | volume_and_active | removed_by_retention | removed_by_delivery_condition | eligible | runs: c (core shrinkage) | runs: tau2_core | runs: tau2_delta est [95% CI] | wkt: c (core shrinkage) | wkt: tau2_core | wkt: tau2_delta est [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bat: primary | bat | component | H | 108 | 0.003542 | 0.003542 | [-0.00551, 0.01229] | {'intercept': 0.0605, 'rv_runs': 0.727, 'rv_wkt': 0.1164} | [] | 153.4 | 78.3 | {'rv_runs': np.float64(0.219), 'rv_wkt': np.float64(0.355)} | 120 | 2 | 36 | 82 |  |  |  |  |  |  |
| bat: H20 | bat | component | H20_within | 108 | 0.004949 | 0.004949 | [-0.00668, 0.02052] | {'intercept': 0.074, 'rv_runs': 0.6669, 'rv_wkt': -0.137} | [] | 148.9 | 85.4 | {'rv_runs': np.float64(0.228), 'rv_wkt': np.float64(0.34)} | 100 | 2 | 26 | 72 |  |  |  |  |  |  |
| bat: H33 | bat | component | H33_within | 108 | 0.002167 | 0.002167 | [-0.00338, 0.00873] | {'intercept': 0.034, 'rv_runs': 0.7611, 'rv_wkt': 0.1955} | [] | 150.8 | 70.1 | {'rv_runs': np.float64(0.205), 'rv_wkt': np.float64(0.357)} | 146 | 5 | 52 | 89 |  |  |  |  |  |  |
| bat: threshold −25% | bat | component | H | 81 | 0.003542 | 0.003542 | [-0.00551, 0.01229] | {'intercept': 0.0605, 'rv_runs': 0.727, 'rv_wkt': 0.1164} | [] | 153.4 | 78.3 | {'rv_runs': np.float64(0.219), 'rv_wkt': np.float64(0.355)} | 142 | 3 | 47 | 92 |  |  |  |  |  |  |
| bat: threshold +25% | bat | component | H | 135 | 0.003542 | 0.003542 | [-0.00551, 0.01229] | {'intercept': 0.0605, 'rv_runs': 0.727, 'rv_wkt': 0.1164} | [] | 153.4 | 78.3 | {'rv_runs': np.float64(0.219), 'rv_wkt': np.float64(0.355)} | 100 | 2 | 30 | 68 |  |  |  |  |  |  |
| bat: IPL only | bat | component | H | 108 | 0.0 | -0.002578 | [-0.01197, 0.00642] | {'intercept': 0.0895, 'rv_runs': 1.1362, 'rv_wkt': -0.0118} | [] | 178.9 | 99.9 | {'rv_runs': np.float64(0.194), 'rv_wkt': np.float64(0.301)} | 69 | 1 | 21 | 47 |  |  |  |  |  |  |
| bat: noise-aware delivery condition | bat | component | H | 108 | 0.003542 | 0.003542 | [-0.00551, 0.01229] | {'intercept': 0.0605, 'rv_runs': 0.727, 'rv_wkt': 0.1164} | [] | 153.4 | 78.3 | {'rv_runs': np.float64(0.219), 'rv_wkt': np.float64(0.355)} | 120 | 2 | 5 | 113 |  |  |  |  |  |  |
| bat: bivariate method | bat | bivariate | H | 108 | 0.004836 |  |  |  |  |  |  |  | 120 | 2 | 36 | 82 | 157.1 | 0.018801 | 0.00484 [-0.00301, 0.01136] | 105.7 | 0.016657 | -0.00037 [-0.00453, 0.00409] |
| bowl: primary | bowl | bivariate | H | 154 | 0.002214 |  |  |  |  |  |  |  | 86 | 6 | 39 | 41 | 317.9 | 0.009036 | 0.00221 [-0.00425, 0.00992] | not estimable → dropped | 0.0 | -0.00094 [-0.00309, 0.00137] |
| bowl: H20 | bowl | bivariate | H20_within | 154 | 0.005333 |  |  |  |  |  |  |  | 73 | 5 | 35 | 33 | 317.9 | 0.009036 | 0.00533 [-0.00059, 0.01091] | not estimable → dropped | 0.0 | -0.00054 [-0.00296, 0.00210] |
| bowl: H33 | bowl | bivariate | H33_within | 154 | 0.005292 |  |  |  |  |  |  |  | 120 | 5 | 64 | 51 | 317.9 | 0.009036 | 0.00485 [-0.00142, 0.01187] | not estimable → dropped | 0.0 | 0.00044 [-0.00234, 0.00369] |
| bowl: threshold −25% | bowl | bivariate | H | 116 | 0.002214 |  |  |  |  |  |  |  | 115 | 6 | 54 | 55 | 317.9 | 0.009036 | 0.00221 [-0.00425, 0.00992] | not estimable → dropped | 0.0 | -0.00094 [-0.00309, 0.00137] |
| bowl: threshold +25% | bowl | bivariate | H | 192 | 0.002214 |  |  |  |  |  |  |  | 76 | 5 | 36 | 35 | 317.9 | 0.009036 | 0.00221 [-0.00425, 0.00992] | not estimable → dropped | 0.0 | -0.00094 [-0.00309, 0.00137] |
| bowl: IPL only | bowl | bivariate | H | 150 | 0.003423 |  |  |  |  |  |  |  | 52 | 3 | 32 | 17 | 255.4 | 0.011908 | 0.00342 [-0.00899, 0.02042] | not estimable → dropped | 0.0 | -0.00035 [-0.00310, 0.00294] |
| bowl: noise-aware delivery condition | bowl | bivariate | H | 154 | 0.002214 |  |  |  |  |  |  |  | 86 | 6 | 7 | 73 | 317.9 | 0.009036 | 0.00221 [-0.00425, 0.00992] | not estimable → dropped | 0.0 | -0.00094 [-0.00309, 0.00137] |
| bowl: L-only (previous) method | bowl | component | H | 154 | 0.0 | -0.002227 | [-0.00784, 0.00379] | {'intercept': -0.0268, 'rv_runs': 0.3153} | ['rv_wkt'] | 226.5 | not estimable (between-player variance ≈ 0) | {'rv_runs': np.float64(0.279), 'rv_wkt': 0.0} | 86 | 6 | 39 | 41 |  |  |  |  |  |  |


## Batters: top 10 and next 10 (method: component)

Eligibility: ≥ 108 H deliveries (2 × median 54 in 484 regular player-seasons), played in 2025 or 2026, retention gate, noise-aware delivery condition. Eligible: 82. Ratings: runs per 100 H balls vs expectation = core skill + pressure deviation; `k_update` = shrinkage weight on the player's own H − L deviation (runs component). Tier: positions in one tier are within one SE of their neighbours (not separable).

| rank | tier | player | latest_team | role | rating_runs_per_100_H_balls | se_per_100 | rank_score_per_100 | core_skill_per_100 | pressure_dev_per_100 | raw_H_per_100 | raw_H_se_per_100 | k_update | n_H | seasons_played | R | R_ci95 | strongest_process_on_H | name_matches_data | name_shared_by_two_ids |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | Vaibhav Sooryavanshi | India | opener | 52.28 | 8.97 | 43.31 | 51.18 | 1.1 | 59.03 | 22.23 | 0.14 | 256.0 | 2025–2026 (2 seasons) | -0.137 | [-0.522, +0.213] | strike-rate delta +2.62σ; boundary elevation +2.45σ; wicket security -0.26σ | True | False |
| 2 | 2 | Abhishek Sharma | India | opener | 35.93 | 5.35 | 30.58 | 36.13 | -0.2 | 35.54 | 10.55 | 0.337 | 801.0 | 2022–2026 (5 seasons) | -0.034 | [-0.190, +0.127] | strike-rate delta +1.41σ; boundary elevation +1.15σ; dot suppression +1.03σ | True | False |
| 3 | 3 | Finn Allen | New Zealand | opener | 27.33 | 5.53 | 21.8 | 28.91 | -1.58 | 16.94 | 20.2 | 0.132 | 239.0 | 2022–2026 (5 seasons) | -0.175 | [-0.489, +0.117] | strike-rate delta +1.28σ; boundary elevation +1.09σ; dot suppression +0.98σ | True | False |
| 4 | 3 | Phil Salt | England | opener | 25.66 | 4.04 | 21.62 | 22.89 | 2.77 | 35.66 | 12.27 | 0.217 | 436.0 | 2022–2026 (5 seasons) | 0.01 | [-0.190, +0.215] | dot suppression +1.63σ; strike-rate delta +1.04σ; boundary elevation +0.42σ | True | False |
| 5 | 3 | Suryakumar Yadav | Mumbai Indians | middle order (4-5) | 24.05 | 3.63 | 20.42 | 24.09 | -0.04 | 23.93 | 11.14 | 0.245 | 512.0 | 2022–2026 (5 seasons) | -0.114 | [-0.291, +0.058] | boundary elevation +0.99σ; strike-rate delta +0.63σ; wicket security -0.47σ | True | False |
| 6 | 3 | Tim David | Australia | finisher (6-7) | 24.77 | 5.27 | 19.51 | 21.66 | 3.11 | 39.18 | 12.37 | 0.178 | 340.0 | 2022–2026 (5 seasons) | -0.008 | [-0.206, +0.191] | strike-rate delta +1.56σ; boundary elevation +1.06σ; wicket security +0.82σ | True | False |
| 7 | 3 | Heinrich Klaasen | Sunrisers Hyderabad | middle order (4-5) | 19.73 | 3.72 | 16.01 | 15.86 | 3.87 | 51.84 | 12.27 | 0.108 | 190.0 | 2022–2026 (5 seasons) | 0.129 | [-0.101, +0.344] | strike-rate delta +2.17σ; boundary elevation +1.97σ; dot suppression +0.79σ | True | False |
| 8 | 3 | Andre Russell | West Indies | finisher (6-7) | 21.11 | 5.93 | 15.18 | 20.53 | 0.58 | 25.79 | 19.15 | 0.11 | 195.0 | 2022–2025 (4 seasons) | -0.129 | [-0.502, +0.184] | boundary elevation +1.39σ; strike-rate delta +1.01σ; wicket security +0.23σ | True | False |
| 9 | 3 | Travis Head | Sunrisers Hyderabad | opener | 20.43 | 6.35 | 14.08 | 17.37 | 3.07 | 30.52 | 15.49 | 0.233 | 479.0 | 2023–2026 (4 seasons) | 0.196 | [-0.046, +0.431] | boundary elevation +1.14σ; strike-rate delta +0.98σ; dot suppression +0.89σ | True | False |
| 10 | 3 | Priyansh Arya | Punjab Kings | opener | 22.57 | 8.82 | 13.75 | 21.11 | 1.46 | 34.06 | 23.22 | 0.113 | 200.0 | 2025–2026 (2 seasons) | 0.108 | [-0.251, +0.526] | dot suppression +2.49σ; strike-rate delta +2.22σ; boundary elevation +1.48σ | True | False |
| 11 | 3 | Josh Inglis | Australia | No.3 | 17.88 | 5.68 | 12.2 | 18.16 | -0.29 | 15.68 | 20.63 | 0.116 | 206.0 | 2022–2026 (5 seasons) | -0.075 | [-0.405, +0.258] | boundary elevation +0.54σ; strike-rate delta +0.41σ; dot suppression -0.42σ | True | False |
| 12 | 3 | Glenn Maxwell | Australia | middle order (4-5) | 17.0 | 4.96 | 12.03 | 16.56 | 0.44 | 21.86 | 28.86 | 0.083 | 142.0 | 2022–2025 (4 seasons) | 0.024 | [-0.551, +0.430] | boundary elevation +1.38σ; strike-rate delta +1.10σ; dot suppression +0.52σ | True | False |
| 13 | 3 | Nicholas Pooran | Lucknow Super Giants | middle order (4-5) | 15.95 | 4.65 | 11.3 | 15.93 | 0.02 | 16.05 | 15.86 | 0.182 | 350.0 | 2022–2026 (5 seasons) | -0.0 | [-0.226, +0.222] | boundary elevation +0.15σ; wicket security +0.04σ; strike-rate delta -0.05σ | True | False |
| 14 | 3 | Romario Shepherd | West Indies | finisher (6-7) | 15.01 | 4.67 | 10.33 | 15.65 | -0.65 | 8.53 | 24.84 | 0.091 | 157.0 | 2022–2026 (5 seasons) | -0.142 | [-0.518, +0.196] | wicket security +0.30σ; boundary elevation +0.17σ; strike-rate delta -0.53σ | True | False |
| 15 | 3 | Jos Buttler | England | opener | 13.56 | 3.51 | 10.05 | 12.44 | 1.11 | 15.86 | 8.26 | 0.325 | 759.0 | 2022–2026 (5 seasons) | -0.009 | [-0.138, +0.123] | wicket security +0.39σ; boundary elevation -0.41σ; strike-rate delta -0.45σ | True | False |
| 16 | 3 | Mitchell Marsh | Australia | opener | 13.75 | 3.86 | 9.89 | 16.01 | -2.26 | 6.78 | 11.26 | 0.245 | 510.0 | 2022–2026 (5 seasons) | -0.1 | [-0.287, +0.080] | wicket security -0.21σ; dot suppression -0.66σ; boundary elevation -0.76σ | True | False |
| 17 | 3 | Harry Brook | England | middle order (4-5) | 14.81 | 5.46 | 9.35 | 13.69 | 1.12 | 29.39 | 30.83 | 0.071 | 121.0 | 2022–2026 (5 seasons) | -0.032 | [-0.451, +0.355] | dot suppression +2.30σ; strike-rate delta +1.29σ; boundary elevation +0.41σ | True | False |
| 18 | 3 | Prabhsimran Singh | Punjab Kings | opener | 13.63 | 4.38 | 9.25 | 9.46 | 4.17 | 28.49 | 12.12 | 0.219 | 442.0 | 2022–2026 (5 seasons) | 0.29 | [+0.048, +0.541] | boundary elevation +0.63σ; wicket security +0.26σ; strike-rate delta +0.17σ | True | False |
| 19 | 3 | Glenn Phillips | New Zealand | middle order (4-5) | 12.71 | 3.81 | 8.9 | 13.15 | -0.45 | 7.83 | 16.18 | 0.084 | 144.0 | 2022–2026 (5 seasons) | 0.016 | [-0.269, +0.292] | wicket security +1.48σ; dot suppression -1.07σ; strike-rate delta -1.54σ | True | False |
| 20 | 3 | Cameron Green | Kolkata Knight Riders | middle order (4-5) | 12.92 | 4.29 | 8.63 | 13.53 | -0.61 | 8.78 | 14.54 | 0.129 | 234.0 | 2022–2026 (5 seasons) | -0.059 | [-0.310, +0.163] | wicket security +0.37σ; dot suppression -0.09σ; strike-rate delta -0.44σ | True | False |

### Removed by the eligibility conditions (had enough H deliveries and were active)

- **retention gate (pooled 10,000 resamples)** (2): Devon Conway (R -0.229, CI [-0.394, -0.076]); Charith Asalanka (R -0.608, CI [-1.103, -0.208])
- **delivery condition (raw H run value ≤ 0)** (36): Rajat Patidar (raw H -0.30 ± 20.83/100 balls, rating +18.94); Mark Chapman (raw H -0.40 ± 23.54/100 balls, rating +12.69); Johnson Charles (raw H -10.36 ± 23.52/100 balls, rating +12.65); Shivam Dube (raw H -11.18 ± 9.99/100 balls, rating +10.71); Paul Stirling (raw H -16.10 ± 22.12/100 balls, rating +10.37); Quinton de Kock (raw H -4.68 ± 14.22/100 balls, rating +8.12); Tim Robinson (raw H -4.94 ± 18.78/100 balls, rating +7.72); Brian Bennett (raw H -8.96 ± 19.40/100 balls, rating +7.18); Faf du Plessis (raw H -2.23 ± 12.00/100 balls, rating +6.69); Tanzid Hasan (raw H -22.88 ± 15.83/100 balls, rating +6.65); Sherfane Rutherford (raw H -6.76 ± 12.42/100 balls, rating +6.45); William Jacks (raw H -18.16 ± 21.70/100 balls, rating +6.38); Nehal Wadhera (raw H -12.80 ± 13.27/100 balls, rating +6.09); Rachin Ravindra (raw H -6.34 ± 22.85/100 balls, rating +4.49); Venkatesh Iyer (raw H -12.65 ± 23.88/100 balls, rating +4.42); Rahul Tripathi (raw H -3.19 ± 16.99/100 balls, rating +4.31); Dhruv Jurel (raw H -7.48 ± 12.46/100 balls, rating +3.93); Abishek Porel (raw H -9.26 ± 22.13/100 balls, rating +3.62); Liton Das (raw H -6.85 ± 17.07/100 balls, rating +3.57); Aiden Markram (raw H -23.76 ± 14.85/100 balls, rating +3.21); Ayush Badoni (raw H -6.30 ± 15.40/100 balls, rating +2.92); Axar Patel (raw H -8.88 ± 9.91/100 balls, rating +2.68); Deepak Hooda (raw H -19.05 ± 16.23/100 balls, rating +2.64); Angkrish Raghuvanshi (raw H -8.91 ± 17.12/100 balls, rating +2.48); Brandon King (raw H -9.72 ± 14.08/100 balls, rating +2.16); Babar Azam (raw H -10.40 ± 13.89/100 balls, rating +1.74); Hardik Pandya (raw H -1.58 ± 7.99/100 balls, rating +1.56); Vijay Shankar (raw H -10.17 ± 27.20/100 balls, rating +0.07); Kamil Mishara (raw H -15.89 ± 30.12/100 balls, rating -0.30); Rassie van der Dussen (raw H -25.54 ± 25.29/100 balls, rating -1.04); Rahmanullah Gurbaz (raw H -21.80 ± 15.96/100 balls, rating -1.32); Devdutt Padikkal (raw H -0.94 ± 16.79/100 balls, rating -2.05); Mayank Agarwal (raw H -37.01 ± 18.33/100 balls, rating -2.42); Krunal Pandya (raw H -14.08 ± 19.55/100 balls, rating -3.65); Ibrahim Zadran (raw H -13.01 ± 13.34/100 balls, rating -4.50); Tadiwanashe Marumani (raw H -40.26 ± 27.28/100 balls, rating -5.45)
- **borderline (single-seed gate decisions disagree; reported, not excluded)** (0): none

### Retention gate: pooled bootstrap (5 seeds x 2,000 = 10,000 resamples)

Borderline = players whose single-seed retention decisions disagree (Monte Carlo error at the interval edge). They are reported, not excluded; the gate uses the pooled interval (P7G-1).

No borderline players.

### Changes in the top 10 vs the previous version

- Left: none
- Entered: none

### Comparisons (overlap with our top 10)

| comparison | overlap |
|---|---|
| plain all-delivery run value | 9 |
| raw run value on H | 6 |
| batting average | 2 |
| strike rate | 8 |

Conventional top 10s: {'batting average': ['Sai Sudharsan', 'Virat Kohli', 'Lokesh Rahul', 'Vaibhav Sooryavanshi', 'Namboori Tilak Varma', 'Jos Buttler', 'Shubman Gill', 'Heinrich Klaasen', 'Rinku Singh', 'David Miller'], 'strike rate': ['Vaibhav Sooryavanshi', 'Priyansh Arya', 'Abhishek Sharma', 'Ayush Mhatre', 'Tim David', 'Finn Allen', 'Travis Head', 'Andre Russell', 'Sunil Narine', 'Phil Salt']}

Role mix of the top 20:

| role | in_top20 |
|---|---|
| opener | 9 |
| middle order (4-5) | 7 |
| finisher (6-7) | 3 |
| No.3 | 1 |

Rating vs role (eligible players): η² = 0.050

| role | mean_rating_per_100 | eligible |
|---|---|---|
| No.3 | 7.41 | 9 |
| finisher (6-7) | 8.34 | 17 |
| lower order (8+) | 11.47 | 1 |
| middle order (4-5) | 10.06 | 24 |
| opener | 12.46 | 31 |

### Sensitivity — top-10 overlap with the primary list

| variant | top10_overlap_with_primary | only_in_primary | only_in_variant |
|---|---|---|---|
| H20 | 9 | Andre Russell | Glenn Maxwell |
| H33 | 8 | Travis Head, Priyansh Arya | Rajat Patidar, Muhammad Waseem |
| threshold −25% | 10 |  |  |
| threshold +25% | 10 |  |  |
| IPL only | 7 | Finn Allen, Tim David, Andre Russell | Nicholas Pooran, Sunil Narine, Liam Livingstone |
| noise-aware delivery condition | 10 |  |  |
| bivariate method | 8 | Travis Head, Priyansh Arya | Shreyas Iyer, Reeza Hendricks |

## Bowlers: top 10 and next 10 (method: bivariate)

Eligibility: ≥ 154 H deliveries (2 × median 77 in 349 regular player-seasons), played in 2025 or 2026, retention gate, noise-aware delivery condition. Eligible: 41. Ratings: runs per 100 H balls vs expectation = core skill + pressure deviation; `k_update` = shrinkage weight on the player's own H − L deviation (runs component). Tier: positions in one tier are within one SE of their neighbours (not separable).

| rank | tier | player | latest_team | role | rating_runs_per_100_H_balls | se_per_100 | rank_score_per_100 | core_skill_per_100 | pressure_dev_per_100 | raw_H_per_100 | raw_H_se_per_100 | k_update | n_H | seasons_played | R | R_ci95 | strongest_process_on_H | name_matches_data | name_shared_by_two_ids |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | Jasprit Bumrah | India | pace, mainly powerplay | 25.27 | 5.05 | 20.22 | 27.58 | -2.3 | 10.7 | 11.46 | 0.207 | 539.0 | 2022–2026 (5 seasons) | -0.165 | [-0.338, +0.007] | boundary suppression +2.00σ; economy suppression +1.46σ; dot rate +1.42σ | True | False |
| 2 | 1 | Sunil Narine | Kolkata Knight Riders | spin, mainly middle | 20.86 | 3.64 | 17.21 | 20.66 | 0.19 | 24.05 | 13.67 | 0.108 | 203.0 | 2022–2026 (5 seasons) | 0.046 | [-0.207, +0.316] | dot rate +1.74σ; economy suppression +1.54σ; boundary suppression +1.18σ | True | False |
| 3 | 2 | Matheesha Pathirana | Kolkata Knight Riders | pace, mainly death | 16.36 | 4.09 | 12.27 | 16.71 | -0.35 | 19.07 | 10.82 | 0.144 | 331.0 | 2022–2026 (5 seasons) | 0.104 | [-0.104, +0.331] | economy suppression +0.91σ; boundary suppression +0.77σ; wicket-taking +0.41σ | True | False |
| 4 | 2 | Nathan Ellis | Australia | pace, mainly death | 15.4 | 3.23 | 12.17 | 14.78 | 0.61 | 22.29 | 9.84 | 0.133 | 344.0 | 2022–2026 (5 seasons) | 0.008 | [-0.203, +0.212] | boundary suppression +1.70σ; economy suppression +1.25σ; dot rate -0.31σ | True | False |
| 5 | 2 | Rashid Khan (Afghanistan, Gujarat Titans) | Afghanistan | spin, mainly middle | 12.24 | 3.88 | 8.37 | 12.11 | 0.13 | 10.07 | 10.64 | 0.199 | 424.0 | 2022–2026 (5 seasons) | -0.071 | [-0.225, +0.093] | boundary suppression +0.77σ; economy suppression +0.34σ; wicket-taking -0.13σ | True | True |
| 6 | 2 | Eshan Malinga | Sri Lanka | pace, mainly middle | 11.94 | 4.36 | 7.57 | 10.2 | 1.74 | 20.61 | 12.11 | 0.108 | 227.0 | 2024–2026 (3 seasons) | 0.163 | [-0.017, +0.364] | boundary suppression +1.91σ; economy suppression +1.50σ; dot rate +0.07σ | True | False |
| 7 | 2 | Josh Hazlewood | Royal Challengers Bengaluru | pace, mainly powerplay | 11.91 | 4.63 | 7.28 | 10.05 | 1.86 | 25.87 | 10.32 | 0.197 | 570.0 | 2022–2026 (5 seasons) | 0.025 | [-0.147, +0.188] | wicket-taking +1.26σ; dot rate +1.06σ; economy suppression +0.79σ | True | False |
| 8 | 2 | Bhuvneshwar Kumar | India | pace, mainly powerplay | 9.88 | 3.99 | 5.88 | 11.08 | -1.21 | 11.84 | 9.73 | 0.266 | 771.0 | 2022–2026 (5 seasons) | -0.003 | [-0.159, +0.151] | boundary suppression +0.42σ; economy suppression -0.02σ; wicket-taking -0.37σ | True | False |
| 9 | 2 | Mujeeb Ur Rahman | Afghanistan | spin, mainly powerplay | 9.82 | 4.42 | 5.41 | 7.64 | 2.19 | 30.89 | 15.42 | 0.083 | 162.0 | 2022–2026 (5 seasons) | 0.1 | [-0.099, +0.328] | dot rate +2.07σ; economy suppression +2.04σ; boundary suppression +1.20σ | True | False |
| 10 | 2 | Varun Chakravarthy | India | spin, mainly middle | 7.61 | 3.68 | 3.92 | 7.14 | 0.47 | 7.98 | 11.24 | 0.175 | 367.0 | 2022–2026 (5 seasons) | -0.116 | [-0.294, +0.059] | dot rate +0.51σ; economy suppression -0.00σ; wicket-taking -0.33σ | True | False |
| 11 | 2 | Mohammed Siraj | Gujarat Titans | pace, mainly powerplay | 8.63 | 4.76 | 3.88 | 7.63 | 1.0 | 12.1 | 10.79 | 0.242 | 733.0 | 2022–2026 (5 seasons) | 0.087 | [-0.060, +0.243] | dot rate +0.73σ; economy suppression +0.16σ; wicket-taking -0.18σ | True | False |
| 12 | 2 | Mitchell Santner | New Zealand | spin, mainly middle | 7.35 | 4.07 | 3.28 | 5.4 | 1.95 | 27.0 | 19.01 | 0.089 | 157.0 | 2022–2026 (5 seasons) | 0.175 | [-0.087, +0.437] | economy suppression +1.48σ; boundary suppression +0.95σ; dot rate +0.80σ | True | False |
| 13 | 2 | Joshua Little | Ireland | pace, mainly powerplay | 7.44 | 4.4 | 3.04 | 4.99 | 2.45 | 15.16 | 11.96 | 0.105 | 258.0 | 2022–2025 (4 seasons) | 0.012 | [-0.237, +0.243] | economy suppression +1.17σ; dot rate +1.06σ; boundary suppression +0.70σ | True | False |
| 14 | 2 | Taskin Ahmed | Bangladesh | pace, mainly powerplay | 7.5 | 4.59 | 2.91 | 4.94 | 2.55 | 13.7 | 9.85 | 0.15 | 400.0 | 2022–2026 (5 seasons) | 0.054 | [-0.143, +0.247] | economy suppression +0.68σ; boundary suppression +0.35σ; dot rate -0.04σ | True | False |
| 15 | 2 | Lungi Ngidi | Delhi Capitals | pace, mainly powerplay | 7.92 | 5.4 | 2.52 | 6.37 | 1.55 | 11.69 | 13.65 | 0.119 | 297.0 | 2022–2026 (5 seasons) | 0.082 | [-0.138, +0.288] | boundary suppression +0.84σ; dot rate +0.74σ; economy suppression +0.70σ | True | False |
| 16 | 2 | Krunal Pandya | Royal Challengers Bengaluru | spin, mainly middle | 6.12 | 4.34 | 1.78 | 4.19 | 1.93 | 22.81 | 14.76 | 0.102 | 194.0 | 2022–2026 (5 seasons) | 0.201 | [-0.051, +0.470] | wicket-taking +1.20σ; economy suppression +1.02σ; dot rate +0.91σ | True | False |
| 17 | 2 | Jason Holder | West Indies | pace, mainly middle | 4.59 | 3.65 | 0.95 | 4.88 | -0.29 | 4.07 | 8.41 | 0.242 | 663.0 | 2022–2026 (5 seasons) | 0.003 | [-0.146, +0.157] | boundary suppression +0.10σ; wicket-taking +0.10σ; economy suppression -0.49σ | True | False |
| 18 | 2 | Mark Adair | Ireland | pace, mainly powerplay | 3.94 | 3.58 | 0.36 | 2.85 | 1.09 | 3.58 | 10.24 | 0.128 | 365.0 | 2022–2026 (5 seasons) | -0.013 | [-0.222, +0.190] | boundary suppression +0.15σ; economy suppression -0.08σ; dot rate -0.46σ | True | False |
| 19 | 2 | Haris Rauf | Pakistan | pace, mainly powerplay | 4.64 | 4.48 | 0.16 | 4.54 | 0.1 | 10.25 | 12.73 | 0.148 | 397.0 | 2022–2025 (4 seasons) | -0.013 | [-0.260, +0.250] | wicket-taking +1.04σ; dot rate -0.04σ; economy suppression -0.32σ | True | False |
| 20 | 2 | Shaheen Afridi | Pakistan | pace, mainly powerplay | 5.34 | 5.19 | 0.15 | 6.56 | -1.22 | 8.48 | 16.13 | 0.133 | 319.0 | 2023–2026 (4 seasons) | -0.042 | [-0.283, +0.181] | wicket-taking -0.15σ; boundary suppression -0.69σ; economy suppression -0.71σ | True | False |

### Removed by the eligibility conditions (had enough H deliveries and were active)

- **retention gate (pooled 10,000 resamples)** (6): Deepak Chahar (R -0.239, CI [-0.433, -0.034]); Jofra Archer (R -0.222, CI [-0.412, -0.035]); Hardik Pandya (R -0.227, CI [-0.386, -0.059]); Mustafizur Rahman (R -0.187, CI [-0.348, -0.021]); Azmatullah Omarzai (R -0.293, CI [-0.512, -0.075]); Prince Yadav (R -0.368, CI [-0.635, -0.104])
- **delivery condition (raw H run value ≤ 0)** (39): Blessing Muzarabani (raw H -3.77 ± 18.19/100 balls, rating +10.26); Maheesh Theekshana (raw H -0.48 ± 10.16/100 balls, rating +7.52); Ravichandran Ashwin (raw H -9.95 ± 12.65/100 balls, rating +5.85); Sandeep Sharma (raw H -7.51 ± 10.20/100 balls, rating +4.72); Barry McCarthy (raw H -10.35 ± 11.55/100 balls, rating +3.11); Noor Ahmad (raw H -8.97 ± 12.35/100 balls, rating +2.79); T Natarajan (raw H -0.52 ± 10.64/100 balls, rating +2.66); Harshal Patel (raw H -8.53 ± 8.40/100 balls, rating +2.11); Reece Topley (raw H -21.38 ± 14.86/100 balls, rating +1.71); Ravi Bishnoi (raw H -9.60 ± 8.46/100 balls, rating +0.54); Kuldeep Yadav (raw H -7.55 ± 14.15/100 balls, rating +0.52); Wanindu Hasaranga (raw H -11.77 ± 10.18/100 balls, rating +0.50); Prasidh Krishna (raw H -11.55 ± 17.26/100 balls, rating +0.36); Khaleel Ahmed (raw H -7.44 ± 12.36/100 balls, rating +0.10); Ishant Sharma (raw H -1.48 ± 21.03/100 balls, rating +0.02); Patrick Cummins (raw H -7.32 ± 13.93/100 balls, rating -0.20); FazalHaq Farooqi (raw H -13.32 ± 13.43/100 balls, rating -0.48); Naseem Shah (raw H -5.61 ± 12.67/100 balls, rating -0.72); Sam Curran (raw H -9.45 ± 9.60/100 balls, rating -0.96); Tushar Deshpande (raw H -12.43 ± 10.10/100 balls, rating -1.22); Obed McCoy (raw H -7.06 ± 12.87/100 balls, rating -2.20); Dushmantha Chameera (raw H -19.25 ± 15.41/100 balls, rating -2.32); Kagiso Rabada (raw H -5.52 ± 8.70/100 balls, rating -2.47); Marco Jansen (raw H -15.41 ± 12.03/100 balls, rating -3.06); Mohammed Shami (raw H -3.81 ± 16.85/100 balls, rating -3.36); Mohit Sharma (raw H -30.28 ± 18.67/100 balls, rating -3.53); Adam Milne (raw H -5.41 ± 19.50/100 balls, rating -3.97); Shardul Thakur (raw H -2.60 ± 13.88/100 balls, rating -4.96); Vaibhav Arora (raw H -14.68 ± 15.45/100 balls, rating -5.00); Tanzim Sakib (raw H -14.54 ± 20.17/100 balls, rating -5.26); Cameron Green (raw H -20.62 ± 18.58/100 balls, rating -5.28); Xavier Bartlett (raw H -12.33 ± 15.81/100 balls, rating -5.37); Mitchell Starc (raw H -2.72 ± 17.05/100 balls, rating -5.45); Mukesh Choudhary (raw H -16.11 ± 17.14/100 balls, rating -5.81); Matt Henry (raw H -26.94 ± 15.80/100 balls, rating -6.12); Kyle Jamieson (raw H -24.57 ± 14.94/100 balls, rating -10.08); Romario Shepherd (raw H -16.95 ± 7.93/100 balls, rating -10.63); Anrich Nortje (raw H -36.21 ± 15.04/100 balls, rating -11.80); Andre Russell (raw H -21.61 ± 16.64/100 balls, rating -14.25)
- **borderline (single-seed gate decisions disagree; reported, not excluded)** (0): none

### Retention gate: pooled bootstrap (5 seeds x 2,000 = 10,000 resamples)

Borderline = players whose single-seed retention decisions disagree (Monte Carlo error at the interval edge). They are reported, not excluded; the gate uses the pooled interval (P7G-1).

No borderline players.

### Changes in the top 10 vs the previous version

- Left: **Mohammed Siraj** — now rank 11: rating +8.63 (core +7.63, pressure dev +1.00), rank score +3.88
- Entered: **Jasprit Bumrah** — was excluded as borderline under the seed-flip rule (now passes the pooled retention gate); now rating +25.27 (core +27.58, pressure dev -2.30, raw H +10.70), rank score +20.22

### Comparisons (overlap with our top 10)

| comparison | overlap |
|---|---|
| plain all-delivery run value | 7 |
| raw run value on H | 6 |
| economy (lowest) | 5 |
| wickets (most) | 4 |

Conventional top 10s: {'economy (lowest)': ['Sunil Narine', 'Mujeeb Ur Rahman', 'Jasprit Bumrah', 'Rashid Khan (Afghanistan, Gujarat Titans)', 'Mitchell Santner', 'Akeal Hosein', 'Krunal Pandya', 'Axar Patel', 'Varun Chakravarthy', 'Shaheen Afridi'], 'wickets (most)': ['Arshdeep Singh', 'Rashid Khan (Afghanistan, Gujarat Titans)', 'Varun Chakravarthy', 'Yuzvendra Chahal', 'Axar Patel', 'Bhuvneshwar Kumar', 'Jason Holder', 'Josh Hazlewood', 'Mohammed Siraj', 'Mitchell Santner']}

Role mix of the top 20:

| role | in_top20 |
|---|---|
| pace, mainly powerplay | 10 |
| spin, mainly middle | 5 |
| pace, mainly death | 2 |
| pace, mainly middle | 2 |
| spin, mainly powerplay | 1 |

Rating vs role (eligible players): η² = 0.042

| role | mean_rating_per_100 | eligible |
|---|---|---|
| pace, mainly death | 7.46 | 4 |
| pace, mainly middle | 3.36 | 5 |
| pace, mainly powerplay | 4.7 | 22 |
| spin, mainly middle | 6.3 | 9 |
| spin, mainly powerplay | 9.82 | 1 |

### Sensitivity — top-10 overlap with the primary list

| variant | top10_overlap_with_primary | only_in_primary | only_in_variant |
|---|---|---|---|
| H20 | 7 | Bhuvneshwar Kumar, Mujeeb Ur Rahman, Varun Chakravarthy | Mohammed Siraj, Joshua Little, Taskin Ahmed |
| H33 | 7 | Bhuvneshwar Kumar, Mujeeb Ur Rahman, Varun Chakravarthy | Joshua Little, Mohammad Wasim, Taskin Ahmed |
| threshold −25% | 9 | Varun Chakravarthy | Abrar Ahmed |
| threshold +25% | 9 | Mujeeb Ur Rahman | Mohammed Siraj |
| IPL only | 6 | Nathan Ellis, Eshan Malinga, Mujeeb Ur Rahman, Varun Chakravarthy | Mohammed Siraj, Krunal Pandya, Arshdeep Singh, Avesh Khan |
| noise-aware delivery condition | 9 | Varun Chakravarthy | Blessing Muzarabani |
| L-only (previous) method | 7 | Eshan Malinga, Josh Hazlewood, Mujeeb Ur Rahman | Mohsin Khan, Shaheen Afridi, Arshdeep Singh |