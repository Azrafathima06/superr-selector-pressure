# Phase 8 — Validation (lists LOCKED; results reported, not used to change them)

Phase 8: validation of the LOCKED lists (results are reported, never used to change the lists).

1. Out-of-time: the Phase 7 rating stage is rebuilt on 2022–2024 data only (same methods and rules; activity window
   2023–24, the analogue of 2025–26), then used to predict each player's 2025–2026 raw run value on H.
   Benchmarks (all from 2022–24): (a) career average run value (all deliveries), (b) raw death-over strike rate (bat) /
   economy (bowl), (c) simple pressure delta: strike rate on H − on L (bat) / economy on H − on L (bowl).
   Note: per-ball values come from league-level models (WP, outcome) that were fitted out-of-fold over all seasons; the
   player-level rating stage uses no 2025–26 data.
2. Knockout check (31 matches): top-20 players' run value on H vs all other players. Small sample.
3. Face validity: where named players sit, with a one-line data reason.
4. Vignesh Puthur through the model.
5. Charts.

## 1. Out-of-time test

Setup per side: {'bat': {'min_H_2022_24': 107, 'eligible_2022_24': 50, 'evaluated_players': 61, 'test_min_H_2025_26': 30}, 'bowl': {'min_H_2022_24': 156, 'eligible_2022_24': 30, 'evaluated_players': 50, 'test_min_H_2025_26': 30}}. Evaluated players: eligible by volume in 2022–24 and ≥ 30 H deliveries in 2025–26. Bootstrap over players (2,000).

| side | predictor | players | pearson_r | pearson_ci95 | spearman | spearman_ci95 |
|---|---|---|---|---|---|---|
| bat | Our rating (2022–24 rebuild) | 61 | 0.149 | [-0.078, 0.350] | 0.124 | [-0.122, 0.358] |
| bat | (a) career average run value | 61 | 0.199 | [0.006, 0.391] | 0.214 | [-0.025, 0.426] |
| bat | (b) raw death-over strike rate | 61 | -0.111 | [-0.393, 0.286] | -0.04 | [-0.305, 0.232] |
| bat | (c) simple pressure SR delta (H − L) | 61 | 0.13 | [-0.119, 0.370] | 0.126 | [-0.124, 0.376] |
| bowl | Our rating (2022–24 rebuild) | 50 | 0.419 | [0.147, 0.629] | 0.354 | [0.063, 0.581] |
| bowl | (a) career average run value | 50 | 0.455 | [0.204, 0.677] | 0.484 | [0.199, 0.711] |
| bowl | (b) raw death-over economy (negated) | 50 | 0.39 | [0.166, 0.585] | 0.414 | [0.133, 0.634] |
| bowl | (c) simple pressure economy delta (negated) | 50 | -0.068 | [-0.346, 0.175] | -0.082 | [-0.358, 0.228] |

Top-20 of the 2022–24 rebuild (full rules) vs a random eligible player — share above the average of evaluated players on H in 2025–26:

| side | top20_2022_24 | top20_with_2025_26_H_data | top20_above_average_on_H_2025_26 | share | share_ci95_wilson | share_for_random_eligible_player | average_used (runs per H ball, eligible test players) | top20_above_expectation_(>0) |
|---|---|---|---|---|---|---|---|---|
| bat | 20 | 16 | 10 | 0.625 | [0.39, 0.82] | 0.557 | 0.10010000318288803 | 12 |
| bowl | 20 | 18 | 13 | 0.722 | [0.49, 0.88] | 0.58 | -0.07900000363588333 | 11 |

Chart: `outputs/charts/out_of_time_validation.png`

## 2. Knockout check (31 knockout matches — SMALL SAMPLE)

Run value per 100 H balls vs expectation; 95% interval by bootstrap over matches.

| side | matches | group | players | H_deliveries | n_matches | run_value_per_100_H_balls | ci95 (bootstrap over matches) |
|---|---|---|---|---|---|---|---|
| bat | knockout matches | final top 20 | 13 | 197 | 16 | -30.40999984741211 | [-101.10, 21.27] |
| bat | knockout matches | all other players | 135 | 1541 | 29 | 1.7599999904632568 | [-11.95, 13.15] |
| bat | all other matches | final top 20 | 20 | 6516 | 411 | 27.5 | [20.74, 34.25] |
| bat | all other matches | all other players | 686 | 37816 | 762 | -2.0 | [-4.44, 0.29] |
| bowl | knockout matches | final top 20 | 12 | 337 | 19 | 17.229999542236328 | [-9.09, 46.33] |
| bowl | knockout matches | all other players | 104 | 1469 | 30 | -0.6499999761581421 | [-14.92, 16.95] |
| bowl | all other matches | final top 20 | 20 | 7384 | 478 | 13.649999618530273 | [8.42, 19.11] |
| bowl | all other matches | all other players | 506 | 39263 | 764 | -5.320000171661377 | [-7.61, -2.86] |

## 3. Face validity (data reasons only)

| player | side | status | reason |
|---|---|---|---|
| Rinku Singh | bat | eligible, rank 38 of 82 | rating +9.2 (core +7.0, pressure dev +2.2); rank score +5.5; raw H +24.0 ± 13.0 over 229 H balls |
| Hardik Pandya | bat | not eligible | raw H run value -1.6 per 100 balls ≤ 0 (delivery requirement). Rating +1.6 (core +2.3, pressure dev -0.7); raw H -1.6 |
| Virat Kohli | bat | eligible, rank 46 of 82 | rating +6.4 (core +5.1, pressure dev +1.4); rank score +3.7; raw H +9.3 ± 7.3 over 745 H balls |
| Mahendra Singh Dhoni | bat | eligible, rank 76 of 82 | rating +2.8 (core +1.8, pressure dev +1.0); rank score -2.5; raw H +14.6 ± 21.8 over 139 H balls |
| Arshdeep Singh | bowl | eligible, rank 27 of 41 | rating +2.2 (core +4.2, pressure dev -2.0); rank score -1.6; raw H +2.1 ± 7.3 over 1118 H balls |
| Kuldeep Yadav | bowl | not eligible | raw H run value -7.6 per 100 balls ≤ 0 (delivery requirement). Rating +0.5 (core +1.8, pressure dev -1.3); raw H -7.6 |
| Hardik Pandya | bowl | not eligible | retention interval entirely below 0 [-0.386, -0.059]. Rating -4.8 (core -0.8, pressure dev -4.0); raw H -29.9 |

## 4. Vignesh Puthur

| role | in_data | player_id | matches | seasons | deliveries | H_deliveries | min_H_for_eligibility | eligible | rating_per_100 (model, shrunk) | core_skill_per_100 | raw_H_per_100 | raw_H_se_per_100 (simple) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bat | False |  |  |  |  |  |  |  |  |  |  |  |
| bowler |  | 48557.0 | 5.0 | 2025 | 73.0 | 4.0 | 154.0 | False | -4.09 | -3.97 | -77.7 | 131.91 |

## 5. Charts

- WP calibration: `outputs/charts/wp_calibration.png`
- Leverage by innings and phase: `outputs/charts/leverage_share_by_innings_phase.png`
- Trust table: `outputs/charts/trust_table.png`
- Core skill vs pressure (H vs L): `outputs/charts/core_vs_pressure.png`
- Out-of-time validation: `outputs/charts/out_of_time_validation.png`