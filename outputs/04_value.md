# Phase 4 — Per-ball performance value

Deliveries: 193,806 (rain-shortened matches excluded, as decided in Phase 3). WP is the batting side's; ties = 0.5.

## 1. Opposition quality (leave-one-match-out run-value ratings)

Run value of a delivery, batting side's view: **RV = (runs − E[runs]) − wicket_cost × (wicket − p_wicket)**. E[runs] and p_wicket come from the Phase 3 player-agnostic outcome model (out-of-fold). wicket_cost = expected runs still to come at this state minus the same with one more wicket down, from the Phase 2 Markov resource table (match's own fold). Batter rating: shrunk mean RV over deliveries faced (wides excluded). Bowler rating: shrunk mean −RV over everything bowled. Shrinkage toward 0 (league average) with c = σ²_within / τ²_between (method of moments): {'batters': {'c': np.float64(132.4), 'sigma2': np.float64(5.468), 'tau2': np.float64(0.0413)}, 'bowlers': {'c': np.float64(563.7), 'sigma2': np.float64(5.291), 'tau2': np.float64(0.00939)}}. **Every delivery uses ratings computed without its own match** (later matches are used; this affects only the opposition adjustment).

Highest-rated (≥500 deliveries, all-data ratings, sanity check only):

Batters:

| player | n | rating |
|---|---|---|
| Vaibhav Sooryavanshi | 662 | 0.6294 |
| Abhishek Sharma | 2261 | 0.3927 |
| Tim David | 1166 | 0.3383 |
| Suryakumar Yadav | 2634 | 0.305 |
| Phil Salt | 1592 | 0.299 |
| Andre Russell | 700 | 0.2951 |
| Finn Allen | 762 | 0.2838 |
| Heinrich Klaasen | 1410 | 0.261 |
| Rajat Patidar | 869 | 0.2506 |
| Romario Shepherd | 663 | 0.2465 |

Bowlers:

| player | n | rating |
|---|---|---|
| Jasprit Bumrah | 1674 | 0.2019 |
| Sufiyan Muqeem | 525 | 0.1731 |
| Sunil Narine | 1485 | 0.1554 |
| Mustafizur Rahman | 1566 | 0.1157 |
| Rashid Khan | 2684 | 0.1102 |
| Varun Chakravarthy | 2108 | 0.1079 |
| Blessing Muzarabani | 846 | 0.1053 |
| Matheesha Pathirana | 1149 | 0.1029 |
| Josh Hazlewood | 1403 | 0.1001 |
| Abrar Ahmed | 763 | 0.0984 |

## 2. Opposition-aware expectation models

Same features and folds as the Phase 3 outcome model plus one opposition rating; gradient boosting with 53 trees (median of the Phase 3 held-out choice). Out-of-fold log-loss:

| model | log_loss |
|---|---|
| Phase 3 player-agnostic (context only) | 1.58611 |
| Batter-side: context + bowler rating (excl. match) | 1.58489 |
| Bowler-side: context + batter rating (excl. match) | 1.58395 |

## 3. Value per delivery

Value = actual WP change − Σ_o p(o|s,x)·[WP(s_o) − WP(s)] = WP_after − Σ_o p(o|s,x)·WP(s_o). Batters use the batter-side model (opposition = bowler), bowlers the bowler-side model (opposition = batter) and −Value. **v = Value / LI_raw** (LI_raw = centred Phase 3 leverage).

Distribution of v:

| series | count | mean | std | min | 0.1% | 1% | 25% | 50% | 75% | 99% | 99.9% | max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v_bat | 193806.0 | 0.008 | 1.743 | -13.781 | -9.482 | -7.2 | -0.535 | -0.042 | 0.568 | 3.733 | 4.672 | 49.233 |
| v_bowl | 193806.0 | -0.009 | 1.743 | -49.114 | -4.67 | -3.725 | -0.569 | 0.038 | 0.531 | 7.199 | 9.485 | 13.812 |

Where extreme v comes from:

| group | deliveries | mean_abs_v_bat | p99_abs_v_bat |
|---|---|---|---|
| LI_raw in lowest 1% | 1939 | 1.02 | 6.291 |
| other deliveries | 191867 | 1.0 | 7.211 |

Calibration check (mean over all deliveries should be ≈ 0):

| side | mean_v_H | mean_v_L | mean_Value_H | mean_Value_L |
|---|---|---|---|---|
| batters | -0.0085 | -0.0274 | -9e-05 | -0.00025 |
| bowlers | -0.023 | -0.0048 | -0.00067 | -4e-05 |

## 4. Per-player pressure level, baseline and retention

P = mean v over H (primary `H25_within`), B = mean v over L, R = P − B. Also P/R on H20 and H33 within innings. Batters: deliveries faced as striker (wides excluded). Bowlers: all deliveries bowled. Simple standard errors here; Phase 7 bootstraps over matches. Tables: `outputs/tables/player_pressure_batters.csv`, `outputs/tables/player_pressure_bowlers.csv`.

Players: 881 batters, 640 bowlers. High-leverage deliveries per player:

| role | count | mean | std | min | 25% | 50% | 75% | 90% | max |
|---|---|---|---|---|---|---|---|---|---|
| batters | 881.0 | 52.6 | 106.8 | 0.0 | 2.0 | 10.0 | 43.0 | 160.0 | 809.0 |
| bowlers | 640.0 | 75.7 | 128.7 | 0.0 | 5.0 | 20.5 | 86.5 | 226.1 | 1118.0 |

Correlation of P with B across players with ≥100 H deliveries: batters 0.17, bowlers 0.02 (`outputs/charts/value_P_vs_B_batters.png`, `..._bowlers.png`).

No player is ranked in this phase.