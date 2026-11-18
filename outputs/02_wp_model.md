# Phase 2 — Win probability model

WP = probability that the **batting side of the ball** wins. Separate models for the 1st and 2nd innings.

## Venue par (leak-free)

- League mean for a match = mean first-innings total of the same competition group in the previous 730 days, strictly before the match date (fallback to all groups if <10 such matches; to the overall mean if none). Sources: {'group': 812, 'all_groups': 30, 'global_fallback': 1}.
- Venue adjustment = mean residual (total − its own league mean) of earlier matches at the venue in the window, shrunk by n/(n+k), k = within-venue variance / between-venue variance = 1216 / 101.4 = **12.0** (estimated from data).
- History pool: 811 full-length, un-revised first innings. Par totals 97–207 (median 174).
- Par at a stage = par total × f(k), f = average share of the final total reached after k legal balls (f(36) = 0.29, f(90) = 0.71). This is a shape curve and uses no match outcomes.

## How the model was chosen (and what failed)

1. **Planned:** LightGBM with monotonic constraints. LightGBM cannot load on this Mac (missing libomp; no system libraries installed, as agreed), so the boosted candidate is scikit-learn HistGradientBoosting with the same monotonic constraints. It **lost to logistic regression** on grouped out-of-fold log-loss in both innings, even when regularised (≥3000 balls per leaf). Balls within a match share one outcome, so trees fit individual matches.
2. **Logistic regression with sign constraints** won overall, but a first version gave a side needing 2 off the last ball with 8 wickets in hand **99.6%**. Additive terms (especially wickets in hand) do not fade as balls run out.
3. **Fix (2nd innings):** features from a Markov ball-outcome model of the chase (below), wickets entered only as wickets × min(balls left, k)/k so their value can fade, plus a separate endgame model fitted on the last 18 balls and blended in linearly between 24 and 12 balls left (weight 1 at ≤12). Last-6-balls log-loss: **0.1745 → 0.1673**.

**Markov ball-outcome model.** p(outcome | overs left, wickets lost) for outcomes 0/1/2/3/4/6/wicket/wide-or-no-ball, from 2nd-innings deliveries of the fold's own training matches, smoothed cell → (over, wicket group) → over with pseudo-count α chosen by grouped-CV outcome log-loss (α = 1000; {10: np.float64(1.60446), 30: np.float64(1.60286), 100: np.float64(1.60128), 300: np.float64(1.60026), 1000: np.float64(1.60023)}). From it: (a) **dp_logit** = logit of the exact chase win probability by backward induction over (balls left, wickets lost, runs needed ≤ 400); extras add a run without using a ball; level at the end = tie = 0.5; a monotone projection guarantees more balls / more wickets in hand / fewer runs needed never lowers it. (b) **zmv** = (expected runs still to come − runs needed) / s.d. of runs still to come, both computed exactly from the same chain (a data-estimated, DLS-like resource table).

### Features, required signs, fitted coefficients (all-training-data models, raw units)

| Model | Feature | Required sign | Coefficient |
|---|---|---|---|
| 1st inns | score_vs_par | + | 0.03397 |
| 1st inns | balls_left | free | -0.02181 |
| 1st inns | wkts_in_hand | + | 0.17836 |
| 1st inns | resources | + | 0.16066 |
| 1st inns | g_FM_T20I | free | 0.2137 |
| 1st inns | g_Weaker_T20I | free | 0.65053 |
| 2nd inns main | zmv | + | 0.0 |
| 2nd inns main | runs_needed | − | -0.00735 |
| 2nd inns main | ln_need | − | -1.82804 |
| 2nd inns main | rrr_c | − | -0.16424 |
| 2nd inns main | chase_z | + | 0.35341 |
| 2nd inns main | wb3 | + | 0.00107 |
| 2nd inns main | wb6 | + | 0.22473 |
| 2nd inns main | wb12 | + | 0.25651 |
| 2nd inns main | wb24 | + | 0.17591 |
| 2nd inns main | wb48 | + | 0.0 |
| 2nd inns main | wb120 | + | 0.0 |
| 2nd inns main | g_FM_T20I | free | -0.22349 |
| 2nd inns main | g_Weaker_T20I | free | -1.24896 |
| 2nd inns endgame | zmv | + | 0.22362 |
| 2nd inns endgame | dp_logit | + | 0.41358 |
| 2nd inns endgame | runs_needed | − | 0.0 |
| 2nd inns endgame | ln_need | − | -1.3205 |
| 2nd inns endgame | rrr_c | − | -0.03372 |
| 2nd inns endgame | wb3 | + | 0.05544 |
| 2nd inns endgame | wb6 | + | 0.26492 |
| 2nd inns endgame | wb12 | + | 0.22571 |
| 2nd inns endgame | wb24 | + | 0.00291 |
| 2nd inns endgame | wb48 | + | 0.00581 |
| 2nd inns endgame | wb120 | + | 0.01453 |
| 2nd inns endgame | g_FM_T20I | free | -0.13208 |
| 2nd inns endgame | g_Weaker_T20I | free | -1.27912 |

Definitions: score_vs_par = score − par at this stage; resources = wickets in hand × balls left / 120; rrr_c = (required rate − par total/20) clipped to [−12, 24]; ln_need = log(1 + runs needed); chase_z = (balls left × par total/120 − runs needed) / (σ√balls left), σ = s.d. of runs per legal-ball step = 1.779; wbK = wickets in hand × min(balls left, K)/K; g_* = competition-group dummies (IPL = reference). Coefficients are fitted with L2 penalty (as sklearn C=1 on standardised inputs) and sign bounds (L-BFGS-B). A coefficient of exactly 0 means the sign bound was active: the data wanted the 'wrong' sign, so the feature is unused.

Terminal states are exact: chase reached → 1; all out / overs done → 0 (tie → 0.5); end of 1st innings → 1 − WP(start of chase with target = score + 1).

## Training data and validation

- Training matches: 800 (excludes ties, rain-revised recorded and unrecorded, non-reconciling). Deliveries: 191,062.
- 5-fold cross-validation grouped by match. Each training-match ball is scored by models and Markov tables built without that match. Other kept matches are scored by the all-training-data model (never trained on them).

### Out-of-fold metrics (ball level)

| model | innings | n | log_loss | brier | auc |
|---|---|---|---|---|---|
| CHOSEN: sign-constrained logistic | 1 | 99603 | 0.6242 | 0.2178 | 0.7048 |
| Logistic, unconstrained (same features) | 1 | 99603 | 0.6242 | 0.2178 | 0.7048 |
| Logistic naive (score or runs needed + balls left) | 1 | 99603 | 0.6321 | 0.2215 | 0.6909 |
| Monotonic gradient boosting (HGB) | 1 | 99603 | 0.6395 | 0.2236 | 0.688 |
| Logistic, unconstrained (same features) | 2 | 91459 | 0.3837 | 0.1242 | 0.9064 |
| CHOSEN: sign-constrained logistic + endgame blend | 2 | 91459 | 0.3844 | 0.1244 | 0.906 |
| Sign-constrained logistic, no endgame model | 2 | 91459 | 0.3845 | 0.1244 | 0.906 |
| Monotonic gradient boosting (HGB) | 2 | 91459 | 0.393 | 0.1262 | 0.9028 |
| Logistic naive (score or runs needed + balls left) | 2 | 91459 | 0.4749 | 0.154 | 0.8562 |

### By competition group (chosen model, out-of-fold)

| strength | innings | n | log_loss | brier | auc |
|---|---|---|---|---|---|
| FM_T20I | 1 | 46095 | 0.6219 | 0.2168 | 0.7064 |
| FM_T20I | 2 | 42043 | 0.3632 | 0.1178 | 0.9159 |
| IPL | 1 | 43897 | 0.6251 | 0.2183 | 0.7036 |
| IPL | 2 | 40914 | 0.4106 | 0.1322 | 0.8937 |
| Weaker_T20I | 1 | 9611 | 0.6307 | 0.2197 | 0.6863 |
| Weaker_T20I | 2 | 8502 | 0.3632 | 0.1198 | 0.9079 |

### Season-forward (train 2022..t, test t+1; Markov tables refit on training years only)

| train_years | test_year | innings | n | log_loss | brier | auc |
|---|---|---|---|---|---|---|
| 2022-2022 | 2023 | 1 | 15722 | 0.6686 | 0.2332 | 0.6676 |
| 2022-2022 | 2023 | 2 | 14716 | 0.5136 | 0.1608 | 0.8631 |
| 2022-2023 | 2024 | 1 | 22630 | 0.6228 | 0.2172 | 0.7044 |
| 2022-2023 | 2024 | 2 | 20770 | 0.4073 | 0.1349 | 0.8919 |
| 2022-2024 | 2025 | 1 | 23669 | 0.5984 | 0.2067 | 0.7435 |
| 2022-2024 | 2025 | 2 | 21404 | 0.3499 | 0.1127 | 0.9239 |
| 2022-2025 | 2026 | 1 | 16463 | 0.6645 | 0.2336 | 0.6853 |
| 2022-2025 | 2026 | 2 | 14853 | 0.3589 | 0.1164 | 0.9279 |

### Last-over check (chases with ≤6 balls left, out-of-fold; cells with ≥5 matches)

`chosen` = final model; `no_endgame` = same model without the endgame blend. Cells of 20–70 matches have sampling error of roughly ±0.06–0.10.

| balls | need | deliveries | matches | actual | chosen | no_endgame |
|---|---|---|---|---|---|---|
| 1 | 1 | 7 | 7 | 1.0 | 0.8849999904632568 | 0.976 |
| 1 | 2 | 9 | 9 | 0.777999997138977 | 0.7360000014305115 | 0.893 |
| 1 | 3-4 | 15 | 14 | 0.5329999923706055 | 0.42399999499320984 | 0.405 |
| 1 | 5-6 | 16 | 16 | 0.125 | 0.1469999998807907 | 0.044 |
| 1 | 7-9 | 22 | 22 | 0.0 | 0.024000000208616257 | 0.01 |
| 1 | 10-12 | 21 | 21 | 0.0 | 0.0020000000949949026 | 0.004 |
| 1 | 13-18 | 39 | 38 | 0.0 | 0.0010000000474974513 | 0.001 |
| 1 | 19+ | 140 | 134 | 0.0 | 0.0 | 0.0 |
| 2-3 | 1 | 30 | 22 | 1.0 | 0.9660000205039978 | 0.993 |
| 2-3 | 2 | 24 | 21 | 0.9580000042915344 | 0.9020000100135803 | 0.971 |
| 2-3 | 3-4 | 35 | 26 | 0.9139999747276306 | 0.7979999780654907 | 0.902 |
| 2-3 | 5-6 | 40 | 31 | 0.4749999940395355 | 0.5809999704360962 | 0.633 |
| 2-3 | 7-9 | 53 | 37 | 0.20800000429153442 | 0.30000001192092896 | 0.285 |
| 2-3 | 10-12 | 48 | 36 | 0.1459999978542328 | 0.09399999678134918 | 0.056 |
| 2-3 | 13-18 | 102 | 64 | 0.019999999552965164 | 0.01899999938905239 | 0.008 |
| 2-3 | 19+ | 348 | 170 | 0.0 | 0.0010000000474974513 | 0.0 |
| 4-6 | 1 | 38 | 31 | 1.0 | 0.996999979019165 | 0.998 |
| 4-6 | 2 | 37 | 31 | 1.0 | 0.9800000190734863 | 0.993 |
| 4-6 | 3-4 | 67 | 47 | 0.9700000286102295 | 0.9419999718666077 | 0.978 |
| 4-6 | 5-6 | 75 | 53 | 0.9599999785423279 | 0.8970000147819519 | 0.948 |
| 4-6 | 7-9 | 110 | 73 | 0.6819999814033508 | 0.7450000047683716 | 0.801 |
| 4-6 | 10-12 | 92 | 55 | 0.3700000047683716 | 0.48100000619888306 | 0.524 |
| 4-6 | 13-18 | 177 | 87 | 0.22599999606609344 | 0.210999995470047 | 0.176 |
| 4-6 | 19+ | 694 | 242 | 0.013000000268220901 | 0.014000000432133675 | 0.009 |

### Spot states: final model vs Markov chain vs what happened

Final model in an IPL chase of 175 at par 175; Markov chain = state-only ball-outcome model (ties count 0.5); empirical = chase win rate in all kept matches at that balls-left with runs needed ±1 and wickets in hand ±1 (one state per match).

| state | final_model | markov_chain | empirical (need ±1, wkts in hand ±1) |
|---|---|---|---|
| need 2 off 1, 7 wkts in hand | 0.822 | 0.51 | 1.00 (1 matches) |
| need 1 off 1, 7 wkts in hand | 0.953 | 0.826 | 1.00 (1 matches) |
| need 5 off 3, 5 wkts in hand | 0.824 | 0.602 | 0.71 (21 matches) |
| need 9 off 6, 6 wkts in hand | 0.925 | 0.687 | 0.84 (19 matches) |
| need 12 off 6, 6 wkts in hand | 0.823 | 0.457 | 0.50 (10 matches) |
| need 16 off 6, 7 wkts in hand | 0.695 | 0.201 | 0.67 (9 matches) |
| need 30 off 24, 5 wkts in hand | 0.868 | 0.854 | 1.00 (16 matches) |

**Reading this table honestly.** The state-only Markov chain gives wickets in hand little value in the last over, because its last-over outcome rates average deep batting line-ups with tail-enders. Real chases with wickets in hand convert far more often (e.g. 10–15 needed off the last over with 6+ wickets in hand: 7 of 10 won), and the final model agrees with the wicket-matched data. That is why the chain is used as a feature, not as the model. Every endgame cell holds only 5–25 matches, so last-over WP is the least certain part of the model. Phase 3 leverage is computed from WP *changes*, which are less sensitive to a level error than WP itself.


## Checks

- Monotonic spot check (chase 30 off 24, IPL, par 175), WP for 1..10 wickets in hand: 0.32, 0.48, 0.64, 0.77, 0.87, 0.93, 0.96, 0.98, 0.99, 0.99
- Perturbation check on random live states (each change should never lower WP): {'one more wicket in hand': {'states': 29172, 'violations': 0, 'largest_drop': 0.0}, 'one more ball left': {'states': 29172, 'violations': 31, 'largest_drop': 0.0}, 'one more run scored': {'states': 29172, 'violations': 0, 'largest_drop': 0.0}}. The only source of possible violations is the linear blend between the main and endgame models (12–24 balls left); each model is monotone on its own.
- Calibration: `outputs/charts/wp_calibration.png`
- Sanity curves (chosen by rule: the two closest knockout finishes + RR's tied match). Last three chase deliveries shown:
  - `outputs/charts/wp_curve_45513.png` — India v Pakistan, 2025-09-28 (Final) — India Won by 5 wickets
    need 8 off 5, 5 wkts in hand → WP 0.81; need 2 off 4, 5 wkts in hand → WP 0.99; need 1 off 3, 5 wkts in hand → WP 0.99
  - `outputs/charts/wp_curve_15187.png` — Gujarat Titans v Rajasthan Royals, 2022-05-24 (Qualifier) — Gujarat Titans Won by 7 Wickets
    need 16 off 6, 7 wkts in hand → WP 0.74; need 10 off 5, 7 wkts in hand → WP 0.88; need 4 off 4, 7 wkts in hand → WP 0.98
  - `outputs/charts/wp_curve_40725.png` — Delhi Capitals v Rajasthan Royals, 2025-04-16 (League) — Delhi Capitals Tied with Rajasthan Royals (DC won the Super Over)
    need 5 off 3, 7 wkts in hand → WP 0.89; need 3 off 2, 7 wkts in hand → WP 0.89; need 2 off 1, 7 wkts in hand → WP 0.83