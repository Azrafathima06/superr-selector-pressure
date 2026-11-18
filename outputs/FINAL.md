# SupeRR Selector — Round 1: pressure performance (FINAL working file)

_Every number in this file is generated from the pipeline outputs (`python run_all.py`). Charts and tables are referenced by path. Decisions and reasons: `DECISIONS.md` (summarised in the appendix). Lists locked at git tag `final-lists`._

## 0. Summary

**Pressure skill is real. We measured it, tested that it lasts, and ranked the players who have it.**

We modelled the win probability (WP) of every delivery in 843 men's T20 matches (2022–2026; 362 IPL, the rest T20I), measured how much each delivery could swing the match (a leverage index), and defined pressure as the top 25% of leverage within each innings. Each delivery was then scored as actual minus expected outcome, adjusted for match state, venue, competition strength and the quality of the opponent faced. We tested which pressure measures repeat for the same player (split-half over 50 random match splits, season-to-season, earlier-vs-later), tried five pre-specified ways to recover more pressure-specific signal, and rated players with an empirical Bayes model of their run value on high-leverage deliveries, tested out of time (2022–24 → 2025–26). Final top 10 batters: Vaibhav Sooryavanshi, Abhishek Sharma, Finn Allen, Phil Salt, Suryakumar Yadav, Tim David, Heinrich Klaasen, Andre Russell, Travis Head, Priyansh Arya. Final top 10 bowlers: Jasprit Bumrah, Sunil Narine, Matheesha Pathirana, Nathan Ellis, Rashid Khan (Afghanistan, Gujarat Titans), Eshan Malinga, Josh Hazlewood, Bhuvneshwar Kumar, Mujeeb Ur Rahman, Varun Chakravarthy.

**Headline finding.** Pressure-specific variation exists on both sides — a mixed model with a player-specific slope on leverage beats one without it (likelihood-ratio p = 2.69e-08 batters, 6.17e-06 bowlers; slope SD ≈ 0.0667 runs per ball) — but it is too small relative to sampling noise to estimate reliably per player with 2022–26 data. **What repeats under pressure is process** (strike-rate delta, dot suppression, boundary elevation for batters); outcome measures (WP value, wicket security) and bowlers' individual pressure metrics do not. **Core skill is the best predictor of pressure performance**, so the ratings are core skill measured carefully, updated by each player's own pressure record only as far as the data justify, and restricted to players who have demonstrably delivered under pressure.

## 1. Pressure definition

### 1.1 Win probability model

- WP is always the batting side's probability of winning; ties count 0.5 in all WP values and swings.
- **Venue par (leak-free):** League mean for a match = mean first-innings total of the same competition group in the previous 730 days, strictly before the match date (fallback to all groups if <10 such matches; to the overall mean if none). Sources: {'group': 812, 'all_groups': 30, 'global_fallback': 1}. Venue adjustment = mean residual (total − its own league mean) of earlier matches at the venue in the window, shrunk by n/(n+k), k = within-venue variance / between-venue variance = 1216 / 101.4 = **12.0** (estimated from data). History pool: 811 full-length, un-revised first innings. Par totals 97–207 (median 174).
- Par at a stage = par total × f(k), f = average share of the final total reached after k legal balls (f(36) = 0.29, f(90) = 0.71). This is a shape curve and uses no match outcomes.
- **Features (sign-constrained logistic regression; L2 as sklearn C=1 on standardised inputs; signs enforced with L-BFGS-B):**
  - 1st innings: score_vs_par, balls_left, wkts_in_hand, resources, g_FM_T20I, g_Weaker_T20I (signs [1, 0, 1, 1, 0, 0]).
  - 2nd innings main model: zmv, runs_needed, ln_need, rrr_c, chase_z, wb3, wb6, wb12, wb24, wb48, wb120, g_FM_T20I, g_Weaker_T20I (signs [1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 0, 0]).
  - 2nd innings endgame model (fit on ≤18 balls left, blended linearly from 24 to 12 balls left): zmv, dp_logit, runs_needed, ln_need, rrr_c, wb3, wb6, wb12, wb24, wb48, wb120, g_FM_T20I, g_Weaker_T20I.
  - zmv and dp_logit come from a Markov ball-outcome model of the chase (outcome probabilities by over, wickets lost; exact backward induction; expected runs still to come and their s.d. = a data-estimated DLS-like resource table). wbK = wickets in hand × min(balls left, K)/K, so wickets lose value as balls run out.
- **How the model was chosen:** LightGBM could not load (needs libomp; no system libraries installed by agreement); a monotonic gradient-boosted model (scikit-learn HistGradientBoosting) lost to logistic regression on grouped out-of-fold log-loss. A first logistic version gave a side needing 2 off the last ball 99.6%; the Markov features, wicket bases and endgame blend fixed this.
- **Out-of-fold scores (5-fold, grouped by match):**

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

- Calibration: `outputs/charts/wp_calibration.png`. Endgame calibration: `outputs/charts/wp_endgame_calibration.png`. Sanity curves: `outputs/charts/wp_curve_45513.png`, `wp_curve_15187.png`, `wp_curve_40725.png`.
- **Endgame test (2nd innings, ≤30 balls left, out-of-fold; chosen by log-loss on these deliveries):**

| candidate | log_loss_le30 | brier_le30 | log_loss_le12 | log_loss_full_2nd_inns | brier_full_2nd_inns | cells_mean_abs_error (weighted by matches) | cells outside actual 95% CI (of 59) |
|---|---|---|---|---|---|---|---|
| Current (logistic + endgame logistic) | 0.2347 | 0.0724 | 0.2005 | 0.3844 | 0.1244 | 0.031 | 6 |
| Main logistic only | 0.2354 | 0.0727 | 0.2042 | 0.3845 | 0.1244 | 0.035 | 8 |
| Spline logistic + Markov features | 0.2379 | 0.0727 | 0.2131 | 0.3862 | 0.125 | 0.034 | 10 |
| Hybrid (Markov ≤12, blend 24→12) | 0.242 | 0.0741 | 0.2216 | 0.3858 | 0.1247 | 0.047 | 14 |
| Spline logistic | 0.244 | 0.0736 | 0.229 | 0.3874 | 0.1251 | 0.037 | 15 |
| Markov chain (≤30 balls) | 0.2592 | 0.0785 | 0.2216 | 0.3892 | 0.1256 | 0.065 | 24 |

Spot states (IPL chase of 175, par 175; empirical with 95% Wilson interval):

| state | wkts in hand | current | main_only | markov | hybrid | spline | spline_markov | empirical |
|---|---|---|---|---|---|---|---|---|
| need 2 off 1 | 7+ (model at 8) | 0.831 | 0.928 | 0.51 | 0.51 | 0.937 | 0.885 | no matches |
| need 2 off 1 | 3–5 (model at 4) | 0.779 | 0.908 | 0.51 | 0.51 | 0.905 | 0.836 | 0.67 [0.30–0.90] n=6 |
| need 12 off 6 | 7+ (model at 8) | 0.915 | 0.908 | 0.457 | 0.457 | 0.945 | 0.929 | 0.67 [0.21–0.94] n=3 |
| need 12 off 6 | 3–5 (model at 4) | 0.645 | 0.668 | 0.456 | 0.456 | 0.718 | 0.704 | 0.60 [0.39–0.78] n=20 |
| need 16 off 6 | 7+ (model at 8) | 0.771 | 0.688 | 0.202 | 0.202 | 0.686 | 0.716 | 1.00 [0.21–1.00] n=1 |
| need 16 off 6 | 3–5 (model at 4) | 0.361 | 0.31 | 0.202 | 0.202 | 0.245 | 0.315 | 0.35 [0.17–0.59] n=17 |
| need 30 off 24 | 7+ (model at 8) | 0.979 | 0.979 | 0.915 | 0.979 | 0.985 | 0.979 | 0.95 [0.76–0.99] n=20 |
| need 30 off 24 | 3–5 (model at 4) | 0.773 | 0.773 | 0.745 | 0.773 | 0.833 | 0.799 | 0.90 [0.70–0.97] n=20 |

- Season-forward and by-competition-group scores: `outputs/tables/wp_season_forward.csv`, `wp_metrics_by_group.csv`.
- **1st-innings limitation:** The 1st-innings WP model gives a run the same WP value at every stage. Tested before Phase 4: interactions letting (score − par) and wickets in hand vary with balls left — (score − par)·√((120+k)/(balls left+k)) for k = 6/12/24/48, (score − par) × balls-left hinges at 12/30/60/90, wickets × balls-left bases — with sign constraints. Out-of-fold 1st-innings log-loss: current 0.6242; best variant 0.6240 (wickets bases; per-fold changes mixed, three folds worse). Unconstrained, the stage interaction is slightly *negative*: a run late in the innings is not worth more once par is accounted for (an above-par score early also signals conditions and batting strength). Current model kept. Consequence: 1st-innings leverage comes mainly from wicket risk, so 1st-innings middle overs rarely reach H even within innings.

### 1.2 Leverage index (pressure)

- Player-agnostic outcome model p(o | state, context), o ∈ {0,1,2,3,4,6,wicket,wide/no-ball}; context = innings, balls left, wickets, score vs par / required rate vs par, venue par, phase, batting hand × bowling type, pace/spin, competition strength. Gradient boosting, out-of-fold by match:

| model | log_loss |
|---|---|
| Base rates (no features) | 1.6364 |
| Multinomial logistic | 1.5921 |
| Gradient boosting (HGB) | 1.5882 |

- **Centred leverage (used):** LI_raw = Σ_o p(o|s)·|WP(s_o) − Σ_o' p(o'|s)·WP(s_o')|; **LI = LI_raw / mean LI_raw**. The specified form Σ p·|WP(s_o) − WP(s)| is kept as `LI_spec`; it counted the 1st-to-2nd-innings model hand-over jump as pressure. Where the WP model is internally consistent the two coincide:

| group | deliveries | mean_abs_gap | mean_LI_as_specified | mean_LI_used |
|---|---|---|---|---|
| full matches: 2nd innings | 92813 | 0.0014 | 1.1392 | 1.1516 |
| full matches: last ball of 1st innings | 781 | 0.0573 | 4.3867 | 1.2839 |
| full matches: other 1st-innings balls | 100212 | 0.001 | 0.8447 | 0.8574 |
| rain-shortened matches | 5401 | 0.0026 | 1.1903 | 1.1431 |

- **High-leverage set H = top 25% of LI within each innings** (H20 and H33 for robustness). A global cut-off put 83% of H in the 2nd innings and none in 1st-innings middle overs. Baseline L = the rest.
- Where H comes from (share of all deliveries vs share of H):

| innings | phase | deliveries | mean_LI | median_LI | in_H25_global | in_H25_within | share_of_all | share_of_H25_global | share_of_H25_within | pct_of_group_in_H25_global |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | powerplay | 30391 | 1.026 | 1.032 | 4876 | 16635 | 0.157 | 0.101 | 0.343 | 0.16 |
| 1 | middle | 45293 | 0.739 | 0.763 | 0 | 861 | 0.234 | 0.0 | 0.018 | 0.0 |
| 1 | death | 25309 | 0.88 | 0.913 | 3185 | 7753 | 0.131 | 0.066 | 0.16 | 0.126 |
| 2 | powerplay | 30407 | 1.105 | 1.195 | 15764 | 6922 | 0.157 | 0.325 | 0.143 | 0.518 |
| 2 | middle | 44030 | 0.942 | 0.853 | 17088 | 9599 | 0.227 | 0.353 | 0.198 | 0.388 |
| 2 | death | 18376 | 1.731 | 0.616 | 7539 | 6683 | 0.095 | 0.156 | 0.138 | 0.41 |

- **Evidence that leverage separates situations** (outcome rates across LI deciles; chart `outputs/charts/leverage_deciles.png`, histogram `outputs/charts/leverage_histogram.png`):

| innings | decile | deliveries | mean_LI | dismissal_rate | wide_nb_rate | dot_rate | boundary_rate |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 10100 | 0.4169 | 0.0526 | 0.0444 | 0.307 | 0.1608 |
| 1 | 2 | 10099 | 0.6257 | 0.0511 | 0.0362 | 0.2808 | 0.1543 |
| 1 | 3 | 10099 | 0.7214 | 0.0501 | 0.0337 | 0.2695 | 0.1412 |
| 1 | 4 | 10099 | 0.7858 | 0.045 | 0.0352 | 0.2617 | 0.1568 |
| 1 | 5 | 10100 | 0.8445 | 0.0479 | 0.0414 | 0.2746 | 0.1679 |
| 1 | 6 | 10099 | 0.9013 | 0.0487 | 0.0431 | 0.2942 | 0.1851 |
| 1 | 7 | 10099 | 0.9576 | 0.0481 | 0.0489 | 0.3215 | 0.1905 |
| 1 | 8 | 10099 | 1.0168 | 0.0556 | 0.0478 | 0.3217 | 0.2095 |
| 1 | 9 | 10099 | 1.0909 | 0.0572 | 0.0481 | 0.3213 | 0.2322 |
| 1 | 10 | 10100 | 1.2461 | 0.0733 | 0.0513 | 0.272 | 0.2687 |
| 2 | 1 | 9282 | 0.0169 | 0.0726 | 0.0488 | 0.3033 | 0.1907 |
| 2 | 2 | 9281 | 0.1289 | 0.0558 | 0.038 | 0.2976 | 0.1831 |
| 2 | 3 | 9281 | 0.3266 | 0.0528 | 0.039 | 0.3007 | 0.1791 |
| 2 | 4 | 9281 | 0.5747 | 0.0492 | 0.0404 | 0.3161 | 0.1751 |
| 2 | 5 | 9282 | 0.8486 | 0.0436 | 0.0367 | 0.3193 | 0.1824 |
| 2 | 6 | 9281 | 1.1227 | 0.0477 | 0.0384 | 0.3478 | 0.1768 |
| 2 | 7 | 9281 | 1.3645 | 0.044 | 0.0379 | 0.3406 | 0.1786 |
| 2 | 8 | 9281 | 1.5668 | 0.0509 | 0.0356 | 0.3244 | 0.1841 |
| 2 | 9 | 9281 | 1.8218 | 0.05 | 0.0404 | 0.2908 | 0.2035 |
| 2 | 10 | 9282 | 3.7444 | 0.0789 | 0.0642 | 0.2137 | 0.2052 |

- **Six scenarios** (definitions in §2.3) and their share of H:

| scenario | deliveries | share_of_all | share_of_H25_global | share_of_H25_within | pct_of_scenario_in_H25_within | enrichment (H25 within share / overall share) | mean_LI |
|---|---|---|---|---|---|---|---|
| post_wicket | 86223 | 0.445 | 0.414 | 0.4 | 0.225 | 0.9 | 0.994 |
| entry_under_pressure | 15551 | 0.08 | 0.2 | 0.255 | 0.795 | 3.19 | 1.894 |
| death_chase | 18376 | 0.095 | 0.156 | 0.138 | 0.364 | 1.45 | 1.731 |
| death_defence | 18376 | 0.095 | 0.156 | 0.138 | 0.364 | 1.45 | 1.731 |
| dot_squeeze | 5891 | 0.03 | 0.029 | 0.03 | 0.246 | 1.0 | 0.922 |
| pp_vs_set_openers | 7981 | 0.041 | 0.075 | 0.097 | 0.587 | 2.37 | 1.152 |

- Top-20 highest-leverage deliveries: `outputs/tables/leverage_top20.csv` (all are last-ball chases needing 2–4).
- **Rain-shortened matches** (32; recorded or detected revisions, or innings shorter than 20 overs) are excluded from H and L (the WP model was never trained on shortened innings) but kept for conventional statistics.
- Pressure-band check (H25 vs H33 vs H50 vs LI-weighted; split-half reliability): `outputs/tables/pressure_band_check.csv`; H25 kept (no version clearly more reliable).

## 2. Performance measurement

### 2.1 Per-ball value vs expectation

#### 2.1.1 Opposition quality (leave-one-match-out run-value ratings)

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

#### 2.1.2 Opposition-aware expectation models

Same features and folds as the Phase 3 outcome model plus one opposition rating; gradient boosting with 53 trees (median of the Phase 3 held-out choice). Out-of-fold log-loss:

| model | log_loss |
|---|---|
| Phase 3 player-agnostic (context only) | 1.58611 |
| Batter-side: context + bowler rating (excl. match) | 1.58489 |
| Bowler-side: context + batter rating (excl. match) | 1.58395 |

#### 2.1.3 Value per delivery

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

### 2.2 Metrics (all = actual − expected, positive = good, on H and on L)

**Accounting rules.** Rates use legal deliveries (outcome probabilities renormalised over legal outcomes). Wides and no-balls count against the bowler, never the batter (wides are not balls faced; no-balls count as faced only for the v-based batter metrics). Byes and leg byes are not charged to the bowler. Run-outs, retired-out and obstructing are not bowler wickets.

Constants estimated from the data: {'runs_per_class': {'0': np.float64(0.0), '1': np.float64(1.0), '2': np.float64(2.0), '3': np.float64(3.0), '4': np.float64(4.001), '6': np.float64(6.0), 'wicket': np.float64(0.024), 'wide_nb': np.float64(1.329)}, 'byes_per_legal_ball': 0.0237, 'uncharged_runs_per_delivery': 0.0231, 'bowler_credited_share_of_wickets': 0.9456}. Expected runs use the observed mean runs per outcome class (e.g. a wide/no-ball averages 1.33 runs), minus byes/leg byes (batter) or uncharged runs (bowler).

| Side | Metric | Kind | Per-delivery component |
|---|---|---|---|
| Batter | pressure_v | value | mean v (Phase 4) — P on H, B on L |
| Batter | sr_delta | skill | strike rate vs expected (runs per 100 legal balls) |
| Batter | dot_suppression | skill | expected − actual dot-ball rate (legal) |
| Batter | boundary_elevation | skill | actual − expected boundary rate (legal) |
| Batter | wicket_security | skill | expected − actual bowler-credited dismissal rate (legal) |
| Batter | squeeze_response | temperament | mean v on balls faced after 3+ consecutive dots (ranked on H+L pooled) |
| Batter | entry_resilience | temperament | mean v over first 10 balls faced when arriving in H |
| Batter | collapse_resistance | temperament | mean v over innings begun with team WP < 35% |
| Batter | post_wicket_response | temperament | mean v over the 12 deliveries after a wicket |
| Batter | runout_avoidance | temperament | expected − actual run-out dismissal rate while at the crease |
| Batter | attacking_stroke_share | profile | share of attacking strokes (profile only, not for ranking) |
| Bowler | pressure_v | value | mean v (Phase 4) — P on H, B on L |
| Bowler | economy_suppression | skill | expected − actual runs charged per 6 deliveries |
| Bowler | dot_rate | skill | actual − expected dot-ball rate (legal) |
| Bowler | wicket_taking | skill | actual − expected bowler-wicket rate (legal) |
| Bowler | boundary_suppression | skill | expected − actual boundary rate (legal) |
| Bowler | bounce_back | temperament | mean v on the next legal ball after conceding a 4 or 6 |
| Bowler | post_wicket_response | temperament | mean v over the 12 deliveries after a wicket |
| Bowler | execution_error | temperament | expected − actual wide/no-ball/full-toss/beamer rate (drift = H − L) |
| Bowler | yorker_execution | report_only | mean v on yorker / wide-yorker deliveries (report only: label depends on outcome) |
| Bowler | variation | temperament | Shannon entropy of delivery_style on H vs L (Miller–Madow corrected, bits); only bowlers with ≥60 H deliveries |
| Batter | partner_effect_H | temperament (**ON TRIAL**) | partner's mean v on strike under H with this batter at the other end, minus the partner's mean v under H without him; weighted over partners |

**Trigger definitions (no leakage: each trigger is decided before the measured ball).**
- Bounce-back: the first legal ball the same bowler bowls in the same innings after a delivery that went for 4 or 6.
- Squeeze response: deliveries after 3+ consecutive dot balls by the batting side (the measured ball is not part of the streak).
- Entry resilience: a batter's first 10 balls faced when his first ball faced was in H (decided by the leverage of the state, not its outcome).
- Collapse resistance: all balls of a batter's innings when team WP before his first ball faced was < 0.35.
- Post-wicket response: the 12 deliveries after a wicket in the same innings (the wicket ball excluded).
- Execution drift: execution errors = wides, no-balls, full tosses, beamers; component = league rate for the same innings × phase × pace/spin − actual; drift = H − L (positive = fewer extra errors under pressure).
- Yorker execution: mean v on yorker / wide-yorker deliveries; yorker counts are reported per bowler (`yorker_execution_n_H`, `_n_L`).
- Run-out avoidance: both batters at the crease on every delivery; expected = league run-out rate per batter for the same innings × phase.
- Stroke intent: share of attacking strokes among legal balls with a recorded stroke; **profile only, not used for ranking**.

### 2.3 Pressure level, baseline, retention

P = mean v over H; B = mean v over L; **R = (P − B) − population (P − B)** (delivery-weighted; R_raw kept). **Finding — population-wide shift:** batters P_pop = -0.0085, B_pop = -0.0274 (P − B = 0.0189); bowlers P_pop = -0.023, B_pop = -0.0048 (P − B = -0.0182). These shifts are also the size of the expectation models' calibration error on H vs L, so the centring removes it. `R_raw` is kept.

### 2.4 Final rating method

- **Batters — L-only component prior + component-wise H update.** Run value per ball = runs component (runs scored vs expected) + wicket component (dismissals vs expected × wicket cost in runs). Prior from non-H (L) deliveries only, each component shrunk by its own reliability (c_runs 153.4, c_wkt 78.3); prior = a + b_runs·θ_runs + b_wkt·θ_wkt (weighted regression of observed H run value across players). Rating = prior + k·(H mean − prior), k = τ²_dev / (τ²_dev + σ²_H/n_H), τ²_dev = 0.003542 (pressure-specific variance against this prior).
- **Bowlers — bivariate component rating.** Rating = Σ_k (μ̂_k + δ̂_k): μ_k = core skill from ALL deliveries shrunk by its between-player variance; δ_k = (H mean − L mean) − population shift, shrunk by var(δ_k). Runs component: c 317.9, var(δ) 0.00221 [-0.00425, 0.00992]. Wicket component: not estimable → dropped (bowler wicket-taking has no estimable between-player variance, on L or on all deliveries).
- **Why these methods** (out-of-sample prediction of next-season run value on H, folds grouped by player):

| side | rating | oos_r_next_season_rv_H | ci95 | pairs |
|---|---|---|---|---|
| bat | previous (L-only component prior + H update) | 0.184 | [-0.018, 0.366] | 297 |
| bat | bivariate (core skill from all deliveries + shrunk H−L deviation) | 0.103 | [-0.038, 0.246] | 297 |
| bowl | previous (L-only component prior + H update) | 0.019 | [-0.103, 0.157] | 292 |
| bowl | bivariate (core skill from all deliveries + shrunk H−L deviation) | 0.028 | [-0.087, 0.140] | 292 |

- **Rank score = rating − SE** (SE: bootstrap over the player's matches, 2,000 resamples). **Tiers:** a new tier starts when the gap to the previous rank score exceeds the larger of the two adjacent SEs; positions inside a tier are not statistically separable. Recency weighting was tested and not kept (no clear out-of-sample gain).
- All fitted parameters per variant: `outputs/tables/rating_variants.csv`:

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

## 3. Fair comparisons

- **Eligibility:** ≥ 2 × the median number of H deliveries in a regular player-season (≥10 matches): batters 108, bowlers 154 (two regular seasons of pressure use); played in 2025 or 2026.
- **Delivery requirement:** to appear in a final ranked list a player must have demonstrated delivery under pressure — raw run value on H above expectation (> 0) (brief: 'deliver when it matters'). It is a requirement, not an estimate; the noise-aware version (exclude only if raw H + 1.28·SE < 0) is a sensitivity check (§5).
- **Retention safety gate (pooled):** exclude a player only if the 95% interval of centred R — pooled over 5 seeds × 2,000 bootstrap resamples — lies entirely below 0. R is unreliable as a metric (§5), so it is a gate only.
- Eligible after all conditions: 82 batters, 41 bowlers. Removed (volume and activity met): retention 2 batters / 6 bowlers; delivery requirement 36 batters / 39 bowlers (names in §4.2).
- **Sample size and shrinkage:** every player value is shrunk toward the population (empirical Bayes; c = within-player per-ball variance / between-player variance). Opposition ratings: c batters 132, bowlers 564. Rating components as in §2.4.
- **IPL vs T20I:** competition strength group (IPL / full-member T20I / weaker T20I) enters the WP, outcome and expectation models; every value is relative to expectation for that competition. IPL-only rerun in §5.
- **Competition groups (teams):**

| group | team_name |
|---|---|
| Full member | Afghanistan, Australia, Bangladesh, England, India, Ireland, New Zealand, Pakistan, South Africa, Sri Lanka, West Indies, Zimbabwe |
| IPL franchise | Chennai Super Kings, Delhi Capitals, Gujarat Titans, Kolkata Knight Riders, Lucknow Super Giants, Mumbai Indians, Punjab Kings, Rajasthan Royals, Royal Challengers Bengaluru, Sunrisers Hyderabad |
| Weaker (associate / A side) | Afghanistan A, Bangladesh A, Canada, Hong Kong, India A, Japan, Kuwait, Nepal, Netherlands, Oman, Pakistan A, Scotland, Singapore, Sri Lanka A, United Arab Emirates, United States Of America |

- **Data cleaning and exclusions** (sub-sections 3.1–3.3):

#### 3.1 What the run columns mean (verified by reconciliation)

Hypothesis tested: ball total = `runs + extras1_runs + extras_2_runs`. Checked against `result` for all decided matches without recorded revisions: **804 / 809 reconcile** (runs margin exact; wickets: chase total 1–6 above first innings and wickets lost = 10 − margin). Dropping `extras_2_runs` reconciles only ~91.5% of runs-margin matches, so it is real runs.

Non-reconciling matches (all are DLS/rain games whose revision is not recorded in matches.csv, or one data error):

| match_id | result | tot1 | tot2 | wk2 | margin |
|---|---|---|---|---|---|
| 14130 | Lucknow Super Giants Won 36  Runs | 167.0 | 130.0 | 8.0 | 36.0 |
| 20182 | Punjab Kings Won by 7 Runs(D/L Method) | 191.0 | 146.0 | 7.0 | 7.0 |
| 20190 | Lucknow Super Giants Won by 6 wickets | 121.0 | 127.0 | 5.0 | 6.0 |
| 24822 | India Won by 2 Runs (DLS method) | 139.0 | 47.0 | 2.0 | 2.0 |
| 28334 | New Zealand won by 17runs (DLS Method) | 110.0 | 95.0 | 5.0 | 17.0 |

Meaning of each column (from value distributions by `extras_type`):

| extras_type | rows | `runs` | `extras1_runs` | `extras_2_runs` | legal? |
|---|---|---|---|---|---|
| None | normal | batter runs | 0 | 0 | yes |
| wide | | 0 | 1 + runs run (all wides) | 0 | no |
| Overstep (no-ball) | | batter runs | 1 (no-ball penalty) | byes/leg byes off the no-ball | no |
| other | | batter runs | 1 | byes/leg byes | **no** — these are height no-balls (delivery_style Beamer / Full Toss); with them treated as illegal every over except one has ≤6 legal balls (with them legal, 244 overs show 7–9) |
| byes / legbyes | | 0 | bye runs | 0 | yes |
| Penalty | 1 | 0 | 5 penalty runs | 0 | not a delivery |

Definitions used everywhere:
- **Legal delivery**: not wide, not no-ball (Overstep/other), not the penalty row.
- **Ball faced** (batter): legal or no-ball (wides are not faced).
- **Batter runs** = `runs`.
- **Total runs** = `runs + extras1_runs + extras_2_runs`.
- **Bowler runs conceded** = `runs` + `extras1_runs` on wides and no-balls. Byes, leg byes (incl. off no-balls) and penalty runs are not charged.
- **Bowler wicket**: Caught, Bowled, LBW, Stump(ed), Hit Wicket. Not credited: RunOut, Retire Out, Obstructing the field.

Dismissal types in the data:

| dismissal_type | count |
|---|---|
| Caught | 7557 |
| Bowled | 1775 |
| LBW | 709 |
| RunOut | 600 |
| Stump | 214 |
| Hit Wicket | 17 |
| Retire Out | 6 |
| Obstracting the field | 2 |
| Obstructing the field | 1 |

#### 3.2 Exclusions

| exclude_reason | matches |
|---|---|
| abandoned | 19 |
| no_result | 21 |

- Super-over innings (3–4) dropped: 38 balls in 6 matches.
- Kept matches: 843; kept balls (innings 1–2): 199,208.
- **Tied matches** (10): kept for player analysis; win label = undefined, so excluded from WP training.
- **Rain-revised, recorded** (24): kept for player analysis. Match state uses revised balls (the revised fields are in balls, e.g. 114 = 19 overs) and `revised_target`. Excluded from WP training (cleaner labels); still scored by the WP model using the revised balls/target.
- **Rain-shortened but NOT recorded in matches.csv** (8): detected from DLS text, or an innings ending ≥1 over early with ≤8 wickets down and the chase not complete (an innings ending early at 9 down is treated as all out with an absent/retired-hurt batter). Where a non-DLS game was shortened before it began, the shortened length is inferred from the balls bowled (`max_balls_inferred`); for DLS games the revised target is unknown, so 20-over state is used. All kept for player analysis, all excluded from WP training.

| match_id | result | legal1 | wk1 | legal2 | wk2 | tot1 | tot2 | dls_text | max_balls_inferred | max_balls1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 15795 | India Won by 7 Wickets | 72 | 4.0 | 56 | 3.0 | 108.0 | 111.0 | False | True | 72.0 |
| 18532 | Match Tied | 118 | 10.0 | 54 | 4.0 | 160.0 | 75.0 | False | False | 120.0 |
| 20182 | Punjab Kings Won by 7 Runs(D/L Method) | 120 | 5.0 | 96 | 7.0 | 191.0 | 146.0 | True | False | 120.0 |
| 23674 | Bangladesh Won by 6 Wickets(D/L Method) | 102 | 7.0 | 97 | 4.0 | 116.0 | 119.0 | True | False | 120.0 |
| 24822 | India Won by 2 Runs (DLS method) | 120 | 7.0 | 41 | 2.0 | 139.0 | 47.0 | True | False | 120.0 |
| 28334 | New Zealand won by 17runs (DLS Method) | 116 | 10.0 | 88 | 5.0 | 110.0 | 95.0 | True | False | 120.0 |
| 31278 | Kolkata Knight Riders Won by 18 runs | 96 | 7.0 | 96 | 8.0 | 157.0 | 139.0 | False | True | 96.0 |
| 48514 | Sri Lanka Won by 14 runs | 72 | 6.0 | 72 | 8.0 | 160.0 | 146.0 | False | True | 72.0 |

- **One missing ball row** (2 matches; innings 1 legal ball short of full length, not all out, not rain). Kept; excluded from WP training only if they also fail reconciliation.

| match_id | result | legal1 | legal2 | recon_ok |
|---|---|---|---|---|
| 14130 | Lucknow Super Giants Won 36  Runs | 119 | 119 | 0.0 |
| 14164 | Punjab Kings Won by 5 wickets | 119 | 91 | 1.0 |

WP training matches: 800.

#### 3.3 Data problems found

- overs with >6 legal balls: [{'match_id': 41509, 'innings': 1, 'overid': 12, 'legal': 7}]
- overs bowled by >1 bowler (injury/mid-over change): 35
- wickets where dismissed player is neither batter: 1
- duplicate (match, innings, innings_ball_num): 0
- balls whose striker is not in batting XI: 83
- balls whose bowler is not in bowling XI: 30
- bowl_speed missing (0) share: 0.387
- delivery_style missing share: 0.0019
- stroke missing share: 0.0306

- Knockout matches: 31 (small sample).

- **Player identity:** ids are the key everywhere. id 4458 = 'Aneurin Donald' (mapping file's 'AHT Donald' ignored). Same-name different players kept apart by id. Final-list names match the data spelling; the only final name shared with another id is **Rashid Khan** → shown as 'Rashid Khan (Afghanistan, Gujarat Titans)', player_id 4722 (id 20172 is Nepal's Rashid Khan, mapping file only). Basil Hameed (UAE) ids 8638/39444 are probably one person but kept apart (not provable; negligible).

## 4. Evidence for the top 10

### 4.1 Final lists (LOCKED) — ranks 1–10 and next 10

Ratings: runs per 100 high-leverage balls relative to expectation. Core = core skill; pressure dev = the player's pressure-specific adjustment; raw H = unshrunk H mean ± bootstrap SE; k_update = weight on the player's own pressure record.

**Batters**

| rank | tier | player | latest_team | role | rating_runs_per_100_H_balls | se_per_100 | rank_score_per_100 | core_skill_per_100 | pressure_dev_per_100 | raw_H_per_100 | raw_H_se_per_100 | k_update | n_H | seasons_played | R | R_ci95 | strongest_process_on_H |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | Vaibhav Sooryavanshi | India | opener | 52.28 | 8.97 | 43.31 | 51.18 | 1.1 | 59.03 | 22.23 | 0.14 | 256.0 | 2025–2026 (2 seasons) | -0.137 | [-0.522, +0.213] | strike-rate delta +2.62σ; boundary elevation +2.45σ; wicket security -0.26σ |
| 2 | 2 | Abhishek Sharma | India | opener | 35.93 | 5.35 | 30.58 | 36.13 | -0.2 | 35.54 | 10.55 | 0.337 | 801.0 | 2022–2026 (5 seasons) | -0.034 | [-0.190, +0.127] | strike-rate delta +1.41σ; boundary elevation +1.15σ; dot suppression +1.03σ |
| 3 | 3 | Finn Allen | New Zealand | opener | 27.33 | 5.53 | 21.8 | 28.91 | -1.58 | 16.94 | 20.2 | 0.132 | 239.0 | 2022–2026 (5 seasons) | -0.175 | [-0.489, +0.117] | strike-rate delta +1.28σ; boundary elevation +1.09σ; dot suppression +0.98σ |
| 4 | 3 | Phil Salt | England | opener | 25.66 | 4.04 | 21.62 | 22.89 | 2.77 | 35.66 | 12.27 | 0.217 | 436.0 | 2022–2026 (5 seasons) | 0.01 | [-0.190, +0.215] | dot suppression +1.63σ; strike-rate delta +1.04σ; boundary elevation +0.42σ |
| 5 | 3 | Suryakumar Yadav | Mumbai Indians | middle order (4-5) | 24.05 | 3.63 | 20.42 | 24.09 | -0.04 | 23.93 | 11.14 | 0.245 | 512.0 | 2022–2026 (5 seasons) | -0.114 | [-0.291, +0.058] | boundary elevation +0.99σ; strike-rate delta +0.63σ; wicket security -0.47σ |
| 6 | 3 | Tim David | Australia | finisher (6-7) | 24.77 | 5.27 | 19.51 | 21.66 | 3.11 | 39.18 | 12.37 | 0.178 | 340.0 | 2022–2026 (5 seasons) | -0.008 | [-0.206, +0.191] | strike-rate delta +1.56σ; boundary elevation +1.06σ; wicket security +0.82σ |
| 7 | 3 | Heinrich Klaasen | Sunrisers Hyderabad | middle order (4-5) | 19.73 | 3.72 | 16.01 | 15.86 | 3.87 | 51.84 | 12.27 | 0.108 | 190.0 | 2022–2026 (5 seasons) | 0.129 | [-0.101, +0.344] | strike-rate delta +2.17σ; boundary elevation +1.97σ; dot suppression +0.79σ |
| 8 | 3 | Andre Russell | West Indies | finisher (6-7) | 21.11 | 5.93 | 15.18 | 20.53 | 0.58 | 25.79 | 19.15 | 0.11 | 195.0 | 2022–2025 (4 seasons) | -0.129 | [-0.502, +0.184] | boundary elevation +1.39σ; strike-rate delta +1.01σ; wicket security +0.23σ |
| 9 | 3 | Travis Head | Sunrisers Hyderabad | opener | 20.43 | 6.35 | 14.08 | 17.37 | 3.07 | 30.52 | 15.49 | 0.233 | 479.0 | 2023–2026 (4 seasons) | 0.196 | [-0.046, +0.431] | boundary elevation +1.14σ; strike-rate delta +0.98σ; dot suppression +0.89σ |
| 10 | 3 | Priyansh Arya | Punjab Kings | opener | 22.57 | 8.82 | 13.75 | 21.11 | 1.46 | 34.06 | 23.22 | 0.113 | 200.0 | 2025–2026 (2 seasons) | 0.108 | [-0.251, +0.526] | dot suppression +2.49σ; strike-rate delta +2.22σ; boundary elevation +1.48σ |
| 11 | 3 | Josh Inglis | Australia | No.3 | 17.88 | 5.68 | 12.2 | 18.16 | -0.29 | 15.68 | 20.63 | 0.116 | 206.0 | 2022–2026 (5 seasons) | -0.075 | [-0.405, +0.258] | boundary elevation +0.54σ; strike-rate delta +0.41σ; dot suppression -0.42σ |
| 12 | 3 | Glenn Maxwell | Australia | middle order (4-5) | 17.0 | 4.96 | 12.03 | 16.56 | 0.44 | 21.86 | 28.86 | 0.083 | 142.0 | 2022–2025 (4 seasons) | 0.024 | [-0.551, +0.430] | boundary elevation +1.38σ; strike-rate delta +1.10σ; dot suppression +0.52σ |
| 13 | 3 | Nicholas Pooran | Lucknow Super Giants | middle order (4-5) | 15.95 | 4.65 | 11.3 | 15.93 | 0.02 | 16.05 | 15.86 | 0.182 | 350.0 | 2022–2026 (5 seasons) | -0.0 | [-0.226, +0.222] | boundary elevation +0.15σ; wicket security +0.04σ; strike-rate delta -0.05σ |
| 14 | 3 | Romario Shepherd | West Indies | finisher (6-7) | 15.01 | 4.67 | 10.33 | 15.65 | -0.65 | 8.53 | 24.84 | 0.091 | 157.0 | 2022–2026 (5 seasons) | -0.142 | [-0.518, +0.196] | wicket security +0.30σ; boundary elevation +0.17σ; strike-rate delta -0.53σ |
| 15 | 3 | Jos Buttler | England | opener | 13.56 | 3.51 | 10.05 | 12.44 | 1.11 | 15.86 | 8.26 | 0.325 | 759.0 | 2022–2026 (5 seasons) | -0.009 | [-0.138, +0.123] | wicket security +0.39σ; boundary elevation -0.41σ; strike-rate delta -0.45σ |
| 16 | 3 | Mitchell Marsh | Australia | opener | 13.75 | 3.86 | 9.89 | 16.01 | -2.26 | 6.78 | 11.26 | 0.245 | 510.0 | 2022–2026 (5 seasons) | -0.1 | [-0.287, +0.080] | wicket security -0.21σ; dot suppression -0.66σ; boundary elevation -0.76σ |
| 17 | 3 | Harry Brook | England | middle order (4-5) | 14.81 | 5.46 | 9.35 | 13.69 | 1.12 | 29.39 | 30.83 | 0.071 | 121.0 | 2022–2026 (5 seasons) | -0.032 | [-0.451, +0.355] | dot suppression +2.30σ; strike-rate delta +1.29σ; boundary elevation +0.41σ |
| 18 | 3 | Prabhsimran Singh | Punjab Kings | opener | 13.63 | 4.38 | 9.25 | 9.46 | 4.17 | 28.49 | 12.12 | 0.219 | 442.0 | 2022–2026 (5 seasons) | 0.29 | [+0.048, +0.541] | boundary elevation +0.63σ; wicket security +0.26σ; strike-rate delta +0.17σ |
| 19 | 3 | Glenn Phillips | New Zealand | middle order (4-5) | 12.71 | 3.81 | 8.9 | 13.15 | -0.45 | 7.83 | 16.18 | 0.084 | 144.0 | 2022–2026 (5 seasons) | 0.016 | [-0.269, +0.292] | wicket security +1.48σ; dot suppression -1.07σ; strike-rate delta -1.54σ |
| 20 | 3 | Cameron Green | Kolkata Knight Riders | middle order (4-5) | 12.92 | 4.29 | 8.63 | 13.53 | -0.61 | 8.78 | 14.54 | 0.129 | 234.0 | 2022–2026 (5 seasons) | -0.059 | [-0.310, +0.163] | wicket security +0.37σ; dot suppression -0.09σ; strike-rate delta -0.44σ |

**Bowlers**

| rank | tier | player | latest_team | role | rating_runs_per_100_H_balls | se_per_100 | rank_score_per_100 | core_skill_per_100 | pressure_dev_per_100 | raw_H_per_100 | raw_H_se_per_100 | k_update | n_H | seasons_played | R | R_ci95 | strongest_process_on_H |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | Jasprit Bumrah | India | pace, mainly powerplay | 25.27 | 5.05 | 20.22 | 27.58 | -2.3 | 10.7 | 11.46 | 0.207 | 539.0 | 2022–2026 (5 seasons) | -0.165 | [-0.338, +0.007] | boundary suppression +2.00σ; economy suppression +1.46σ; dot rate +1.42σ |
| 2 | 1 | Sunil Narine | Kolkata Knight Riders | spin, mainly middle | 20.86 | 3.64 | 17.21 | 20.66 | 0.19 | 24.05 | 13.67 | 0.108 | 203.0 | 2022–2026 (5 seasons) | 0.046 | [-0.207, +0.316] | dot rate +1.74σ; economy suppression +1.54σ; boundary suppression +1.18σ |
| 3 | 2 | Matheesha Pathirana | Kolkata Knight Riders | pace, mainly death | 16.36 | 4.09 | 12.27 | 16.71 | -0.35 | 19.07 | 10.82 | 0.144 | 331.0 | 2022–2026 (5 seasons) | 0.104 | [-0.104, +0.331] | economy suppression +0.91σ; boundary suppression +0.77σ; wicket-taking +0.41σ |
| 4 | 2 | Nathan Ellis | Australia | pace, mainly death | 15.4 | 3.23 | 12.17 | 14.78 | 0.61 | 22.29 | 9.84 | 0.133 | 344.0 | 2022–2026 (5 seasons) | 0.008 | [-0.203, +0.212] | boundary suppression +1.70σ; economy suppression +1.25σ; dot rate -0.31σ |
| 5 | 2 | Rashid Khan (Afghanistan, Gujarat Titans) | Afghanistan | spin, mainly middle | 12.24 | 3.88 | 8.37 | 12.11 | 0.13 | 10.07 | 10.64 | 0.199 | 424.0 | 2022–2026 (5 seasons) | -0.071 | [-0.225, +0.093] | boundary suppression +0.77σ; economy suppression +0.34σ; wicket-taking -0.13σ |
| 6 | 2 | Eshan Malinga | Sri Lanka | pace, mainly middle | 11.94 | 4.36 | 7.57 | 10.2 | 1.74 | 20.61 | 12.11 | 0.108 | 227.0 | 2024–2026 (3 seasons) | 0.163 | [-0.017, +0.364] | boundary suppression +1.91σ; economy suppression +1.50σ; dot rate +0.07σ |
| 7 | 2 | Josh Hazlewood | Royal Challengers Bengaluru | pace, mainly powerplay | 11.91 | 4.63 | 7.28 | 10.05 | 1.86 | 25.87 | 10.32 | 0.197 | 570.0 | 2022–2026 (5 seasons) | 0.025 | [-0.147, +0.188] | wicket-taking +1.26σ; dot rate +1.06σ; economy suppression +0.79σ |
| 8 | 2 | Bhuvneshwar Kumar | India | pace, mainly powerplay | 9.88 | 3.99 | 5.88 | 11.08 | -1.21 | 11.84 | 9.73 | 0.266 | 771.0 | 2022–2026 (5 seasons) | -0.003 | [-0.159, +0.151] | boundary suppression +0.42σ; economy suppression -0.02σ; wicket-taking -0.37σ |
| 9 | 2 | Mujeeb Ur Rahman | Afghanistan | spin, mainly powerplay | 9.82 | 4.42 | 5.41 | 7.64 | 2.19 | 30.89 | 15.42 | 0.083 | 162.0 | 2022–2026 (5 seasons) | 0.1 | [-0.099, +0.328] | dot rate +2.07σ; economy suppression +2.04σ; boundary suppression +1.20σ |
| 10 | 2 | Varun Chakravarthy | India | spin, mainly middle | 7.61 | 3.68 | 3.92 | 7.14 | 0.47 | 7.98 | 11.24 | 0.175 | 367.0 | 2022–2026 (5 seasons) | -0.116 | [-0.294, +0.059] | dot rate +0.51σ; economy suppression -0.00σ; wicket-taking -0.33σ |
| 11 | 2 | Mohammed Siraj | Gujarat Titans | pace, mainly powerplay | 8.63 | 4.76 | 3.88 | 7.63 | 1.0 | 12.1 | 10.79 | 0.242 | 733.0 | 2022–2026 (5 seasons) | 0.087 | [-0.060, +0.243] | dot rate +0.73σ; economy suppression +0.16σ; wicket-taking -0.18σ |
| 12 | 2 | Mitchell Santner | New Zealand | spin, mainly middle | 7.35 | 4.07 | 3.28 | 5.4 | 1.95 | 27.0 | 19.01 | 0.089 | 157.0 | 2022–2026 (5 seasons) | 0.175 | [-0.087, +0.437] | economy suppression +1.48σ; boundary suppression +0.95σ; dot rate +0.80σ |
| 13 | 2 | Joshua Little | Ireland | pace, mainly powerplay | 7.44 | 4.4 | 3.04 | 4.99 | 2.45 | 15.16 | 11.96 | 0.105 | 258.0 | 2022–2025 (4 seasons) | 0.012 | [-0.237, +0.243] | economy suppression +1.17σ; dot rate +1.06σ; boundary suppression +0.70σ |
| 14 | 2 | Taskin Ahmed | Bangladesh | pace, mainly powerplay | 7.5 | 4.59 | 2.91 | 4.94 | 2.55 | 13.7 | 9.85 | 0.15 | 400.0 | 2022–2026 (5 seasons) | 0.054 | [-0.143, +0.247] | economy suppression +0.68σ; boundary suppression +0.35σ; dot rate -0.04σ |
| 15 | 2 | Lungi Ngidi | Delhi Capitals | pace, mainly powerplay | 7.92 | 5.4 | 2.52 | 6.37 | 1.55 | 11.69 | 13.65 | 0.119 | 297.0 | 2022–2026 (5 seasons) | 0.082 | [-0.138, +0.288] | boundary suppression +0.84σ; dot rate +0.74σ; economy suppression +0.70σ |
| 16 | 2 | Krunal Pandya | Royal Challengers Bengaluru | spin, mainly middle | 6.12 | 4.34 | 1.78 | 4.19 | 1.93 | 22.81 | 14.76 | 0.102 | 194.0 | 2022–2026 (5 seasons) | 0.201 | [-0.051, +0.470] | wicket-taking +1.20σ; economy suppression +1.02σ; dot rate +0.91σ |
| 17 | 2 | Jason Holder | West Indies | pace, mainly middle | 4.59 | 3.65 | 0.95 | 4.88 | -0.29 | 4.07 | 8.41 | 0.242 | 663.0 | 2022–2026 (5 seasons) | 0.003 | [-0.146, +0.157] | boundary suppression +0.10σ; wicket-taking +0.10σ; economy suppression -0.49σ |
| 18 | 2 | Mark Adair | Ireland | pace, mainly powerplay | 3.94 | 3.58 | 0.36 | 2.85 | 1.09 | 3.58 | 10.24 | 0.128 | 365.0 | 2022–2026 (5 seasons) | -0.013 | [-0.222, +0.190] | boundary suppression +0.15σ; economy suppression -0.08σ; dot rate -0.46σ |
| 19 | 2 | Haris Rauf | Pakistan | pace, mainly powerplay | 4.64 | 4.48 | 0.16 | 4.54 | 0.1 | 10.25 | 12.73 | 0.148 | 397.0 | 2022–2025 (4 seasons) | -0.013 | [-0.260, +0.250] | wicket-taking +1.04σ; dot rate -0.04σ; economy suppression -0.32σ |
| 20 | 2 | Shaheen Afridi | Pakistan | pace, mainly powerplay | 5.34 | 5.19 | 0.15 | 6.56 | -1.22 | 8.48 | 16.13 | 0.133 | 319.0 | 2023–2026 (4 seasons) | -0.042 | [-0.283, +0.181] | wicket-taking -0.15σ; boundary suppression -0.69σ; economy suppression -0.71σ |

### 4.2 Who was removed by the eligibility conditions

**Batters**

- **retention gate (pooled 10,000 resamples)** (2): Devon Conway (R -0.229, CI [-0.394, -0.076]); Charith Asalanka (R -0.608, CI [-1.103, -0.208])
- **delivery condition (raw H run value ≤ 0)** (36): Rajat Patidar (raw H -0.30 ± 20.83/100 balls, rating +18.94); Mark Chapman (raw H -0.40 ± 23.54/100 balls, rating +12.69); Johnson Charles (raw H -10.36 ± 23.52/100 balls, rating +12.65); Shivam Dube (raw H -11.18 ± 9.99/100 balls, rating +10.71); Paul Stirling (raw H -16.10 ± 22.12/100 balls, rating +10.37); Quinton de Kock (raw H -4.68 ± 14.22/100 balls, rating +8.12); Tim Robinson (raw H -4.94 ± 18.78/100 balls, rating +7.72); Brian Bennett (raw H -8.96 ± 19.40/100 balls, rating +7.18); Faf du Plessis (raw H -2.23 ± 12.00/100 balls, rating +6.69); Tanzid Hasan (raw H -22.88 ± 15.83/100 balls, rating +6.65); Sherfane Rutherford (raw H -6.76 ± 12.42/100 balls, rating +6.45); William Jacks (raw H -18.16 ± 21.70/100 balls, rating +6.38); Nehal Wadhera (raw H -12.80 ± 13.27/100 balls, rating +6.09); Rachin Ravindra (raw H -6.34 ± 22.85/100 balls, rating +4.49); Venkatesh Iyer (raw H -12.65 ± 23.88/100 balls, rating +4.42); Rahul Tripathi (raw H -3.19 ± 16.99/100 balls, rating +4.31); Dhruv Jurel (raw H -7.48 ± 12.46/100 balls, rating +3.93); Abishek Porel (raw H -9.26 ± 22.13/100 balls, rating +3.62); Liton Das (raw H -6.85 ± 17.07/100 balls, rating +3.57); Aiden Markram (raw H -23.76 ± 14.85/100 balls, rating +3.21); Ayush Badoni (raw H -6.30 ± 15.40/100 balls, rating +2.92); Axar Patel (raw H -8.88 ± 9.91/100 balls, rating +2.68); Deepak Hooda (raw H -19.05 ± 16.23/100 balls, rating +2.64); Angkrish Raghuvanshi (raw H -8.91 ± 17.12/100 balls, rating +2.48); Brandon King (raw H -9.72 ± 14.08/100 balls, rating +2.16); Babar Azam (raw H -10.40 ± 13.89/100 balls, rating +1.74); Hardik Pandya (raw H -1.58 ± 7.99/100 balls, rating +1.56); Vijay Shankar (raw H -10.17 ± 27.20/100 balls, rating +0.07); Kamil Mishara (raw H -15.89 ± 30.12/100 balls, rating -0.30); Rassie van der Dussen (raw H -25.54 ± 25.29/100 balls, rating -1.04); Rahmanullah Gurbaz (raw H -21.80 ± 15.96/100 balls, rating -1.32); Devdutt Padikkal (raw H -0.94 ± 16.79/100 balls, rating -2.05); Mayank Agarwal (raw H -37.01 ± 18.33/100 balls, rating -2.42); Krunal Pandya (raw H -14.08 ± 19.55/100 balls, rating -3.65); Ibrahim Zadran (raw H -13.01 ± 13.34/100 balls, rating -4.50); Tadiwanashe Marumani (raw H -40.26 ± 27.28/100 balls, rating -5.45)
- **borderline (single-seed gate decisions disagree; reported, not excluded)** (0): none

**Bowlers**

- **retention gate (pooled 10,000 resamples)** (6): Deepak Chahar (R -0.239, CI [-0.433, -0.034]); Jofra Archer (R -0.222, CI [-0.412, -0.035]); Hardik Pandya (R -0.227, CI [-0.386, -0.059]); Mustafizur Rahman (R -0.187, CI [-0.348, -0.021]); Azmatullah Omarzai (R -0.293, CI [-0.512, -0.075]); Prince Yadav (R -0.368, CI [-0.635, -0.104])
- **delivery condition (raw H run value ≤ 0)** (39): Blessing Muzarabani (raw H -3.77 ± 18.19/100 balls, rating +10.26); Maheesh Theekshana (raw H -0.48 ± 10.16/100 balls, rating +7.52); Ravichandran Ashwin (raw H -9.95 ± 12.65/100 balls, rating +5.85); Sandeep Sharma (raw H -7.51 ± 10.20/100 balls, rating +4.72); Barry McCarthy (raw H -10.35 ± 11.55/100 balls, rating +3.11); Noor Ahmad (raw H -8.97 ± 12.35/100 balls, rating +2.79); T Natarajan (raw H -0.52 ± 10.64/100 balls, rating +2.66); Harshal Patel (raw H -8.53 ± 8.40/100 balls, rating +2.11); Reece Topley (raw H -21.38 ± 14.86/100 balls, rating +1.71); Ravi Bishnoi (raw H -9.60 ± 8.46/100 balls, rating +0.54); Kuldeep Yadav (raw H -7.55 ± 14.15/100 balls, rating +0.52); Wanindu Hasaranga (raw H -11.77 ± 10.18/100 balls, rating +0.50); Prasidh Krishna (raw H -11.55 ± 17.26/100 balls, rating +0.36); Khaleel Ahmed (raw H -7.44 ± 12.36/100 balls, rating +0.10); Ishant Sharma (raw H -1.48 ± 21.03/100 balls, rating +0.02); Patrick Cummins (raw H -7.32 ± 13.93/100 balls, rating -0.20); FazalHaq Farooqi (raw H -13.32 ± 13.43/100 balls, rating -0.48); Naseem Shah (raw H -5.61 ± 12.67/100 balls, rating -0.72); Sam Curran (raw H -9.45 ± 9.60/100 balls, rating -0.96); Tushar Deshpande (raw H -12.43 ± 10.10/100 balls, rating -1.22); Obed McCoy (raw H -7.06 ± 12.87/100 balls, rating -2.20); Dushmantha Chameera (raw H -19.25 ± 15.41/100 balls, rating -2.32); Kagiso Rabada (raw H -5.52 ± 8.70/100 balls, rating -2.47); Marco Jansen (raw H -15.41 ± 12.03/100 balls, rating -3.06); Mohammed Shami (raw H -3.81 ± 16.85/100 balls, rating -3.36); Mohit Sharma (raw H -30.28 ± 18.67/100 balls, rating -3.53); Adam Milne (raw H -5.41 ± 19.50/100 balls, rating -3.97); Shardul Thakur (raw H -2.60 ± 13.88/100 balls, rating -4.96); Vaibhav Arora (raw H -14.68 ± 15.45/100 balls, rating -5.00); Tanzim Sakib (raw H -14.54 ± 20.17/100 balls, rating -5.26); Cameron Green (raw H -20.62 ± 18.58/100 balls, rating -5.28); Xavier Bartlett (raw H -12.33 ± 15.81/100 balls, rating -5.37); Mitchell Starc (raw H -2.72 ± 17.05/100 balls, rating -5.45); Mukesh Choudhary (raw H -16.11 ± 17.14/100 balls, rating -5.81); Matt Henry (raw H -26.94 ± 15.80/100 balls, rating -6.12); Kyle Jamieson (raw H -24.57 ± 14.94/100 balls, rating -10.08); Romario Shepherd (raw H -16.95 ± 7.93/100 balls, rating -10.63); Anrich Nortje (raw H -36.21 ± 15.04/100 balls, rating -11.80); Andre Russell (raw H -21.61 ± 16.64/100 balls, rating -14.25)
- **borderline (single-seed gate decisions disagree; reported, not excluded)** (0): none

### 4.3 Comparisons with conventional stats and role mix

**Batters**

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

**Bowlers**

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

### 4.4 Pressure Value Gap

Pressure rank = rank by rank score among eligible players. Conventional rank = mean of the two conventional ranks among the same players (bat: average, strike rate; bowl: economy, wickets; all kept matches 2022–26). Gap > 0 = ranked higher on pressure than on conventional stats.

**Batters: final top 20** (82 eligible)

| pressure_rank | player | average | strike_rate | avg_rank | sr_rank | conventional_rank | gap (conventional − pressure rank) |
|---|---|---|---|---|---|---|---|
| 1 | Vaibhav Sooryavanshi | 43.1 | 223.2 | 4.0 | 1.0 | 1.0 | 0 |
| 2 | Abhishek Sharma | 33.3 | 186.8 | 27.0 | 3.0 | 7.0 | 5 |
| 3 | Finn Allen | 30.1 | 176.0 | 39.0 | 6.0 | 16.0 | 13 |
| 4 | Phil Salt | 37.1 | 169.6 | 13.0 | 10.0 | 4.0 | 0 |
| 5 | Suryakumar Yadav | 38.2 | 166.3 | 11.0 | 13.0 | 5.0 | 0 |
| 6 | Tim David | 34.6 | 177.7 | 23.0 | 5.0 | 6.0 | 0 |
| 7 | Heinrich Klaasen | 39.4 | 166.8 | 8.0 | 12.0 | 3.0 | -4 |
| 8 | Andre Russell | 27.3 | 171.1 | 52.0 | 8.0 | 26.0 | 18 |
| 9 | Travis Head | 30.8 | 175.0 | 35.0 | 7.0 | 11.0 | 2 |
| 10 | Priyansh Arya | 25.6 | 189.2 | 62.0 | 2.0 | 33.0 | 23 |
| 11 | Josh Inglis | 29.8 | 162.4 | 40.0 | 16.0 | 24.0 | 13 |
| 12 | Glenn Maxwell | 23.5 | 162.3 | 72.0 | 17.0 | 45.0 | 33 |
| 13 | Nicholas Pooran | 32.7 | 158.6 | 29.0 | 20.0 | 19.0 | 6 |
| 14 | Romario Shepherd | 25.2 | 154.1 | 63.0 | 28.0 | 47.0 | 33 |
| 15 | Jos Buttler | 40.6 | 155.7 | 5.0 | 25.0 | 7.0 | -8 |
| 16 | Mitchell Marsh | 35.5 | 157.4 | 21.0 | 22.0 | 12.0 | -4 |
| 17 | Harry Brook | 35.9 | 164.1 | 19.0 | 15.0 | 9.0 | -8 |
| 18 | Prabhsimran Singh | 29.1 | 159.4 | 43.0 | 19.0 | 29.0 | 11 |
| 19 | Glenn Phillips | 35.1 | 149.8 | 22.0 | 39.0 | 28.0 | 9 |
| 20 | Cameron Green | 38.0 | 152.7 | 12.0 | 33.0 | 16.0 | -4 |

**Batters: largest positive gaps among all eligible players**

| pressure_rank | player | average | strike_rate | avg_rank | sr_rank | conventional_rank | gap (conventional − pressure rank) |
|---|---|---|---|---|---|---|---|
| 12 | Glenn Maxwell | 23.5 | 162.3 | 72.0 | 17.0 | 45.0 | 33 |
| 14 | Romario Shepherd | 25.2 | 154.1 | 63.0 | 28.0 | 47.0 | 33 |
| 52 | Kamindu Mendis | 24.0 | 129.5 | 67.0 | 77.0 | 78.0 | 26 |
| 24 | Jitesh Sharma | 23.9 | 155.4 | 68.0 | 26.0 | 49.0 | 25 |
| 41 | Sikandar Raza | 27.0 | 135.9 | 54.0 | 69.0 | 66.0 | 25 |
| 47 | Kyle Mayers | 23.7 | 141.3 | 70.0 | 61.0 | 71.0 | 24 |
| 10 | Priyansh Arya | 25.6 | 189.2 | 62.0 | 2.0 | 33.0 | 23 |
| 60 | Mohammad Nabi | 22.1 | 130.0 | 74.0 | 76.0 | 82.0 | 22 |
| 55 | Saim Ayub | 22.1 | 137.8 | 74.0 | 66.0 | 76.0 | 21 |
| 31 | Rovman Powell | 27.6 | 146.8 | 50.0 | 48.0 | 52.0 | 21 |

**Bowlers: final top 20** (41 eligible)

| pressure_rank | player | economy | wickets | econ_rank | wkt_rank | conventional_rank | gap (conventional − pressure rank) |
|---|---|---|---|---|---|---|---|
| 1 | Jasprit Bumrah | 7.27 | 85 | 3.0 | 12.0 | 5.0 | 4 |
| 2 | Sunil Narine | 6.84 | 64 | 1.0 | 19.0 | 6.0 | 4 |
| 3 | Matheesha Pathirana | 8.87 | 80 | 26.0 | 13.0 | 22.0 | 19 |
| 4 | Nathan Ellis | 8.15 | 57 | 13.0 | 25.0 | 21.0 | 17 |
| 5 | Rashid Khan (Afghanistan, Gujarat Titans) | 7.43 | 153 | 4.0 | 2.0 | 1.0 | -4 |
| 6 | Eshan Malinga | 9.12 | 54 | 31.0 | 29.0 | 33.0 | 27 |
| 7 | Josh Hazlewood | 8.24 | 97 | 14.0 | 8.0 | 8.0 | 1 |
| 8 | Bhuvneshwar Kumar | 8.29 | 115 | 16.0 | 6.0 | 8.0 | 0 |
| 9 | Mujeeb Ur Rahman | 7.03 | 48 | 2.0 | 30.0 | 13.0 | 4 |
| 10 | Varun Chakravarthy | 7.97 | 134 | 9.0 | 3.0 | 2.0 | -8 |
| 11 | Mohammed Siraj | 8.84 | 86 | 25.0 | 9.0 | 16.0 | 5 |
| 12 | Mitchell Santner | 7.48 | 86 | 5.0 | 9.0 | 4.0 | -8 |
| 13 | Joshua Little | 8.27 | 42 | 15.0 | 33.0 | 26.0 | 13 |
| 14 | Taskin Ahmed | 7.98 | 67 | 10.0 | 18.0 | 11.0 | -3 |
| 15 | Lungi Ngidi | 8.94 | 45 | 28.0 | 31.0 | 31.0 | 16 |
| 16 | Krunal Pandya | 7.81 | 56 | 7.0 | 26.0 | 14.0 | -2 |
| 17 | Jason Holder | 9.0 | 109 | 30.0 | 7.0 | 19.0 | 2 |
| 18 | Mark Adair | 8.72 | 56 | 23.0 | 26.0 | 27.0 | 9 |
| 19 | Haris Rauf | 8.66 | 75 | 22.0 | 15.0 | 19.0 | 0 |
| 20 | Shaheen Afridi | 7.98 | 63 | 10.0 | 20.0 | 12.0 | -8 |

**Bowlers: largest positive gaps among all eligible players**

| pressure_rank | player | economy | wickets | econ_rank | wkt_rank | conventional_rank | gap (conventional − pressure rank) |
|---|---|---|---|---|---|---|---|
| 6 | Eshan Malinga | 9.12 | 54 | 31.0 | 29.0 | 33.0 | 27 |
| 3 | Matheesha Pathirana | 8.87 | 80 | 26.0 | 13.0 | 22.0 | 19 |
| 4 | Nathan Ellis | 8.15 | 57 | 13.0 | 25.0 | 21.0 | 17 |
| 23 | Rasikh Salam | 9.58 | 38 | 34.0 | 37.0 | 39.0 | 16 |
| 15 | Lungi Ngidi | 8.94 | 45 | 28.0 | 31.0 | 31.0 | 16 |
| 13 | Joshua Little | 8.27 | 42 | 15.0 | 33.0 | 26.0 | 13 |
| 21 | Mohsin Khan | 8.41 | 38 | 19.0 | 37.0 | 30.0 | 9 |
| 18 | Mark Adair | 8.72 | 56 | 23.0 | 26.0 | 27.0 | 9 |
| 31 | Nuwan Thushara | 8.92 | 36 | 27.0 | 39.0 | 36.0 | 5 |
| 11 | Mohammed Siraj | 8.84 | 86 | 25.0 | 9.0 | 16.0 | 5 |

### 4.5 Impact above replacement

WP value of one run on H deliveries (mean WP(1 run) − WP(dot), batting side): 0.0143 (check via a four: 0.0143 per run) → about 70.0 runs per win on H deliveries.

**Batters** — replacement: {'min_H': 108, 'replacement_pool': '56 players active in 2025–26 with 54–107 H balls', 'replacement_rating_per_100': 3.37}

| rank | player | rating_per_100 | H_balls_per_season | runs_above_replacement_per_season | wins_above_replacement_per_season |
|---|---|---|---|---|---|
| 1 | Vaibhav Sooryavanshi | 52.28 | 128.0 | 62.6 | 0.894 |
| 2 | Abhishek Sharma | 35.93 | 160.2 | 52.16 | 0.745 |
| 3 | Finn Allen | 27.33 | 47.8 | 11.45 | 0.164 |
| 4 | Phil Salt | 25.66 | 87.2 | 19.44 | 0.278 |
| 5 | Suryakumar Yadav | 24.05 | 102.4 | 21.18 | 0.302 |
| 6 | Tim David | 24.77 | 68.0 | 14.56 | 0.208 |
| 7 | Heinrich Klaasen | 19.73 | 38.0 | 6.22 | 0.089 |
| 8 | Andre Russell | 21.11 | 48.8 | 8.65 | 0.124 |
| 9 | Travis Head | 20.43 | 119.8 | 20.43 | 0.292 |
| 10 | Priyansh Arya | 22.57 | 100.0 | 19.2 | 0.274 |
| 11 | Josh Inglis | 17.88 | 41.2 | 5.98 | 0.085 |
| 12 | Glenn Maxwell | 17.0 | 35.5 | 4.84 | 0.069 |
| 13 | Nicholas Pooran | 15.95 | 70.0 | 8.81 | 0.126 |
| 14 | Romario Shepherd | 15.01 | 31.4 | 3.65 | 0.052 |
| 15 | Jos Buttler | 13.56 | 151.8 | 15.47 | 0.221 |
| 16 | Mitchell Marsh | 13.75 | 102.0 | 10.59 | 0.151 |
| 17 | Harry Brook | 14.81 | 24.2 | 2.77 | 0.04 |
| 18 | Prabhsimran Singh | 13.63 | 88.4 | 9.07 | 0.13 |
| 19 | Glenn Phillips | 12.71 | 28.8 | 2.69 | 0.038 |
| 20 | Cameron Green | 12.92 | 46.8 | 4.47 | 0.064 |

**Bowlers** — replacement: {'min_H': 154, 'replacement_pool': '73 players active in 2025–26 with 77–153 H balls', 'replacement_rating_per_100': -0.43}

| rank | player | rating_per_100 | H_balls_per_season | runs_above_replacement_per_season | wins_above_replacement_per_season |
|---|---|---|---|---|---|
| 1 | Jasprit Bumrah | 25.27 | 107.8 | 27.71 | 0.396 |
| 2 | Sunil Narine | 20.86 | 40.6 | 8.64 | 0.123 |
| 3 | Matheesha Pathirana | 16.36 | 66.2 | 11.12 | 0.159 |
| 4 | Nathan Ellis | 15.4 | 68.8 | 10.89 | 0.155 |
| 5 | Rashid Khan (Afghanistan, Gujarat Titans) | 12.24 | 84.8 | 10.75 | 0.153 |
| 6 | Eshan Malinga | 11.94 | 75.7 | 9.36 | 0.134 |
| 7 | Josh Hazlewood | 11.91 | 114.0 | 14.07 | 0.201 |
| 8 | Bhuvneshwar Kumar | 9.88 | 154.2 | 15.89 | 0.227 |
| 9 | Mujeeb Ur Rahman | 9.82 | 32.4 | 3.32 | 0.047 |
| 10 | Varun Chakravarthy | 7.61 | 73.4 | 5.9 | 0.084 |
| 11 | Mohammed Siraj | 8.63 | 146.6 | 13.29 | 0.19 |
| 12 | Mitchell Santner | 7.35 | 31.4 | 2.44 | 0.035 |
| 13 | Joshua Little | 7.44 | 64.5 | 5.08 | 0.073 |
| 14 | Taskin Ahmed | 7.5 | 80.0 | 6.34 | 0.091 |
| 15 | Lungi Ngidi | 7.92 | 59.4 | 4.96 | 0.071 |
| 16 | Krunal Pandya | 6.12 | 38.8 | 2.54 | 0.036 |
| 17 | Jason Holder | 4.59 | 132.6 | 6.67 | 0.095 |
| 18 | Mark Adair | 3.94 | 73.0 | 3.19 | 0.046 |
| 19 | Haris Rauf | 4.64 | 99.2 | 5.03 | 0.072 |
| 20 | Shaheen Afridi | 5.34 | 79.8 | 4.61 | 0.066 |

Note: about 70 runs per win is the WP model's sensitivity to one run on an average high-leverage ball (1 run ≈ 1.4% WP), not a season-standings conversion.

### 4.6 RR fit

RR's high-leverage deliveries in matches since 2022 (rain-shortened excluded). Value = actual WP change − expected (player-agnostic outcome model), from RR's side; league = all other teams in the same role and scenario. Negative total = WP (wins) RR lost relative to a league-average team over these deliveries. Death defence = 'death overs, 2nd innings' with RR bowling.

| scenario | role | RR_H_deliveries | RR_matches | RR_WP_per_delivery_vs_expectation | league_WP_per_delivery | RR_total_WP_vs_league (wins) |
|---|---|---|---|---|---|---|
| death overs, 2nd innings | RR bowling | 445 | 24 | -0.00578 | -0.00229 | -1.55 |
| entry under pressure | RR bowling | 756 | 52 | 0.0009 | 0.00252 | -1.23 |
| post-wicket (12 deliveries after a wicket) | RR bowling | 1070 | 59 | -0.00014 | 0.00044 | -0.628 |
| death overs, 2nd innings | RR batting | 400 | 20 | 0.00137 | 0.0026 | -0.492 |
| entry under pressure | RR batting | 684 | 51 | -0.00299 | -0.00239 | -0.409 |
| powerplay vs set openers | RR bowling | 264 | 24 | -0.00044 | -0.0008 | 0.093 |
| dot squeeze (after 3+ dots) | RR bowling | 62 | 31 | 0.00171 | -0.00066 | 0.147 |
| powerplay vs set openers | RR batting | 323 | 34 | 0.00211 | 0.00068 | 0.461 |
| dot squeeze (after 3+ dots) | RR batting | 83 | 39 | 0.01103 | -7e-05 | 0.921 |
| post-wicket (12 deliveries after a wicket) | RR batting | 1031 | 60 | 0.00128 | -0.00051 | 1.837 |

Strongest final top-20 players in RR's three costliest scenario × role combinations (raw run value per 100 balls on H deliveries in that scenario, ≥20 deliveries):

| RR's costly scenario | player | final_rank | run_value_per_100_balls_in_scenario_on_H | deliveries |
|---|---|---|---|---|
| death overs, 2nd innings (RR bowling) | Sunil Narine | 2 | 87.3 | 35 |
| death overs, 2nd innings (RR bowling) | Varun Chakravarthy | 10 | 68.7 | 50 |
| death overs, 2nd innings (RR bowling) | Taskin Ahmed | 14 | 65.6 | 38 |
| death overs, 2nd innings (RR bowling) | Josh Hazlewood | 7 | 53.3 | 77 |
| death overs, 2nd innings (RR bowling) | Rashid Khan (Afghanistan, Gujarat Titans) | 5 | 52.6 | 68 |
| entry under pressure (RR bowling) | Mujeeb Ur Rahman | 9 | 71.8 | 48 |
| entry under pressure (RR bowling) | Josh Hazlewood | 7 | 51.6 | 164 |
| entry under pressure (RR bowling) | Matheesha Pathirana | 3 | 46.4 | 95 |
| entry under pressure (RR bowling) | Lungi Ngidi | 15 | 43.5 | 71 |
| entry under pressure (RR bowling) | Sunil Narine | 2 | 41.9 | 83 |
| post-wicket (12 deliveries after a wicket) (RR bowling) | Mujeeb Ur Rahman | 9 | 61.6 | 60 |
| post-wicket (12 deliveries after a wicket) (RR bowling) | Sunil Narine | 2 | 44.8 | 102 |
| post-wicket (12 deliveries after a wicket) (RR bowling) | Varun Chakravarthy | 10 | 43.5 | 149 |
| post-wicket (12 deliveries after a wicket) (RR bowling) | Eshan Malinga | 6 | 38.8 | 94 |
| post-wicket (12 deliveries after a wicket) (RR bowling) | Bhuvneshwar Kumar | 8 | 32.6 | 268 |

Final top-20 players in RR's IPL 2026 squad (competition_team_player_mapping):

| side | final_rank | player |
|---|---|---|
| bat | 1 | Vaibhav Sooryavanshi |

Costliest for RR: **death overs, 2nd innings (RR bowling)**, -1.55 wins vs a league-average team over 445 H deliveries.

### 4.7 Role charts

- Batters: `outputs/charts/role_chart_batters.png` (wicket security vs boundary elevation on H).
- Bowlers: `outputs/charts/role_chart_bowlers.png` (economy suppression vs wicket-taking on H).
- Core skill vs pressure (H vs L): `outputs/charts/core_vs_pressure.png`.

### 4.8 Player cards (all 20)

#### Vaibhav Sooryavanshi — batter #1 (tier 1)

- **Rating:** +52.28 runs per 100 high-leverage balls vs expectation (SE 8.97; rank score +43.31). Core skill +51.18, pressure deviation +1.10.
- **Evidence:** 256 H balls; 2025–2026 (2 seasons); raw H +59.03 ± 22.23; retention R -0.137 [-0.522, +0.213] (pooled bootstrap).
- **Role (data):** opener; median batting position 2; mean entry over 1.1. Latest team: India. **In RR's IPL 2026 squad.**
- **Process on H (σ vs eligible players):** strike-rate delta +2.62σ; boundary elevation +2.45σ; wicket security -0.26σ.
- **Strongest scenario:** entry under pressure (+4.2 per 100 balls, n=76). **Weakest scenario:** post-wicket (12 deliveries after a wicket) (+0.7 per 100 balls, n=54).
- **Impact above replacement:** +62.60 runs / +0.894 wins per season (128 H balls per season).

**Analyst.** Rated mainly on core skill (+51.2); the H record moved the rating by +1.1 (weight 0.14). Tier 1 (rank 1 alone of the top 20): positions within a tier are not statistically separable (adjacent rank scores within one SE).
**Scout (verify on video).** The weakest scenario — post-wicket (12 deliveries after a wicket) (+0.7 per 100 balls, n=54) — and the lowest process signal (wicket security -0.26σ). Check whether the numbers reflect shot/ball execution or match situation.
**Coach (best role / entry point).** Highest H value in innings 1, powerplay (+62.7 per 100 H balls, n=148); current data role: opener.

_All figures from the pipeline outputs (2022–26, rain-shortened matches excluded from pressure sets). Scenario values are raw means over H deliveries in that scenario (not shrunk)._

_(card file: `outputs/cards/bat_01_vaibhav_sooryavanshi.md`)_

#### Abhishek Sharma — batter #2 (tier 2)

- **Rating:** +35.93 runs per 100 high-leverage balls vs expectation (SE 5.35; rank score +30.58). Core skill +36.13, pressure deviation -0.20.
- **Evidence:** 801 H balls; 2022–2026 (5 seasons); raw H +35.54 ± 10.55; retention R -0.034 [-0.190, +0.127] (pooled bootstrap).
- **Role (data):** opener; median batting position 2; mean entry over 1.4. Latest team: India.
- **Process on H (σ vs eligible players):** strike-rate delta +1.41σ; boundary elevation +1.15σ; dot suppression +1.03σ.
- **Strongest scenario:** entry under pressure (+44.6 per 100 balls, n=136). **Weakest scenario:** dot squeeze (after 3+ dots) (-1.5 per 100 balls, n=25).
- **Impact above replacement:** +52.16 runs / +0.745 wins per season (160 H balls per season).

**Analyst.** Rated mainly on core skill (+36.1); the H record moved the rating by -0.2 (weight 0.34). Tier 2 (rank 2 alone of the top 20): positions within a tier are not statistically separable (adjacent rank scores within one SE).
**Scout (verify on video).** The weakest scenario — dot squeeze (after 3+ dots) (-1.5 per 100 balls, n=25) — and the lowest process signal (dot suppression +1.03σ). Check whether the numbers reflect shot/ball execution or match situation.
**Coach (best role / entry point).** Highest H value in innings 2, powerplay (+45.5 per 100 H balls, n=210); current data role: opener.

_All figures from the pipeline outputs (2022–26, rain-shortened matches excluded from pressure sets). Scenario values are raw means over H deliveries in that scenario (not shrunk)._

_(card file: `outputs/cards/bat_02_abhishek_sharma.md`)_

#### Finn Allen — batter #3 (tier 3)

- **Rating:** +27.33 runs per 100 high-leverage balls vs expectation (SE 5.53; rank score +21.80). Core skill +28.91, pressure deviation -1.58.
- **Evidence:** 239 H balls; 2022–2026 (5 seasons); raw H +16.94 ± 20.20; retention R -0.175 [-0.489, +0.117] (pooled bootstrap).
- **Role (data):** opener; median batting position 2; mean entry over 1.0. Latest team: New Zealand.
- **Process on H (σ vs eligible players):** strike-rate delta +1.28σ; boundary elevation +1.09σ; dot suppression +0.98σ.
- **Strongest scenario:** post-wicket (12 deliveries after a wicket) (+18.9 per 100 balls, n=32). **Weakest scenario:** only one scenario with ≥20 H deliveries.
- **Impact above replacement:** +11.45 runs / +0.164 wins per season (48 H balls per season).

**Analyst.** Rated mainly on core skill (+28.9); the H record moved the rating by -1.6 (weight 0.13). Tier 3 (ranks 3–20 of the top 20): positions within a tier are not statistically separable (adjacent rank scores within one SE).
**Scout (verify on video).** The weakest scenario — only one scenario with ≥20 H deliveries — and the lowest process signal (dot suppression +0.98σ). Check whether the numbers reflect shot/ball execution or match situation.
**Coach (best role / entry point).** Highest H value in innings 1, powerplay (+27.0 per 100 H balls, n=211); current data role: opener.

_All figures from the pipeline outputs (2022–26, rain-shortened matches excluded from pressure sets). Scenario values are raw means over H deliveries in that scenario (not shrunk)._

_(card file: `outputs/cards/bat_03_finn_allen.md`)_

#### Phil Salt — batter #4 (tier 3)

- **Rating:** +25.66 runs per 100 high-leverage balls vs expectation (SE 4.04; rank score +21.62). Core skill +22.89, pressure deviation +2.77.
- **Evidence:** 436 H balls; 2022–2026 (5 seasons); raw H +35.66 ± 12.27; retention R +0.010 [-0.190, +0.215] (pooled bootstrap).
- **Role (data):** opener; median batting position 1; mean entry over 1.9. Latest team: England.
- **Process on H (σ vs eligible players):** dot suppression +1.63σ; strike-rate delta +1.04σ; boundary elevation +0.42σ.
- **Strongest scenario:** post-wicket (12 deliveries after a wicket) (+31.5 per 100 balls, n=72). **Weakest scenario:** only one scenario with ≥20 H deliveries.
- **Impact above replacement:** +19.44 runs / +0.278 wins per season (87 H balls per season).

**Analyst.** Rated mainly on core skill (+22.9); the H record moved the rating by +2.8 (weight 0.22). Tier 3 (ranks 3–20 of the top 20): positions within a tier are not statistically separable (adjacent rank scores within one SE).
**Scout (verify on video).** The weakest scenario — only one scenario with ≥20 H deliveries — and the lowest process signal (boundary elevation +0.42σ). Check whether the numbers reflect shot/ball execution or match situation.
**Coach (best role / entry point).** Highest H value in innings 2, powerplay (+90.0 per 100 H balls, n=85); current data role: opener.

_All figures from the pipeline outputs (2022–26, rain-shortened matches excluded from pressure sets). Scenario values are raw means over H deliveries in that scenario (not shrunk)._

_(card file: `outputs/cards/bat_04_phil_salt.md`)_

#### Suryakumar Yadav — batter #5 (tier 3)

- **Rating:** +24.05 runs per 100 high-leverage balls vs expectation (SE 3.63; rank score +20.42). Core skill +24.09, pressure deviation -0.04.
- **Evidence:** 512 H balls; 2022–2026 (5 seasons); raw H +23.93 ± 11.14; retention R -0.114 [-0.291, +0.058] (pooled bootstrap).
- **Role (data):** middle order (4-5); median batting position 4; mean entry over 5.9. Latest team: Mumbai Indians.
- **Process on H (σ vs eligible players):** boundary elevation +0.99σ; strike-rate delta +0.63σ; wicket security -0.47σ.
- **Strongest scenario:** post-wicket (12 deliveries after a wicket) (+11.9 per 100 balls, n=208). **Weakest scenario:** death overs, 2nd innings (-32.1 per 100 balls, n=59).
- **Impact above replacement:** +21.18 runs / +0.302 wins per season (102 H balls per season).

**Analyst.** Rated mainly on core skill (+24.1); the H record moved the rating by -0.0 (weight 0.24). Tier 3 (ranks 3–20 of the top 20): positions within a tier are not statistically separable (adjacent rank scores within one SE).
**Scout (verify on video).** The weakest scenario — death overs, 2nd innings (-32.1 per 100 balls, n=59) — and the lowest process signal (wicket security -0.47σ). Check whether the numbers reflect shot/ball execution or match situation.
**Coach (best role / entry point).** Highest H value in innings 1, death (+100.9 per 100 H balls, n=62); current data role: middle order (4-5).

_All figures from the pipeline outputs (2022–26, rain-shortened matches excluded from pressure sets). Scenario values are raw means over H deliveries in that scenario (not shrunk)._

_(card file: `outputs/cards/bat_05_suryakumar_yadav.md`)_

#### Tim David — batter #6 (tier 3)

- **Rating:** +24.77 runs per 100 high-leverage balls vs expectation (SE 5.27; rank score +19.51). Core skill +21.66, pressure deviation +3.11.
- **Evidence:** 340 H balls; 2022–2026 (5 seasons); raw H +39.18 ± 12.37; retention R -0.008 [-0.206, +0.191] (pooled bootstrap).
- **Role (data):** finisher (6-7); median batting position 6; mean entry over 12.8. Latest team: Australia.
- **Process on H (σ vs eligible players):** strike-rate delta +1.56σ; boundary elevation +1.06σ; wicket security +0.82σ.
- **Strongest scenario:** death overs, 2nd innings (+51.1 per 100 balls, n=138). **Weakest scenario:** entry under pressure (+15.3 per 100 balls, n=152).
- **Impact above replacement:** +14.56 runs / +0.208 wins per season (68 H balls per season).

**Analyst.** Rated mainly on core skill (+21.7); the H record moved the rating by +3.1 (weight 0.18). Tier 3 (ranks 3–20 of the top 20): positions within a tier are not statistically separable (adjacent rank scores within one SE).
**Scout (verify on video).** The weakest scenario — entry under pressure (+15.3 per 100 balls, n=152) — and the lowest process signal (wicket security +0.82σ). Check whether the numbers reflect shot/ball execution or match situation.
**Coach (best role / entry point).** Highest H value in innings 2, death (+51.1 per 100 H balls, n=138); current data role: finisher (6-7).

_All figures from the pipeline outputs (2022–26, rain-shortened matches excluded from pressure sets). Scenario values are raw means over H deliveries in that scenario (not shrunk)._

_(card file: `outputs/cards/bat_06_tim_david.md`)_

#### Heinrich Klaasen — batter #7 (tier 3)

- **Rating:** +19.73 runs per 100 high-leverage balls vs expectation (SE 3.72; rank score +16.01). Core skill +15.86, pressure deviation +3.87.
- **Evidence:** 190 H balls; 2022–2026 (5 seasons); raw H +51.84 ± 12.27; retention R +0.129 [-0.101, +0.344] (pooled bootstrap).
- **Role (data):** middle order (4-5); median batting position 5; mean entry over 9.1. Latest team: Sunrisers Hyderabad.
- **Process on H (σ vs eligible players):** strike-rate delta +2.17σ; boundary elevation +1.97σ; dot suppression +0.79σ.
- **Strongest scenario:** death overs, 2nd innings (+82.4 per 100 balls, n=32). **Weakest scenario:** entry under pressure (+8.4 per 100 balls, n=49).
- **Impact above replacement:** +6.22 runs / +0.089 wins per season (38 H balls per season).

**Analyst.** Rated mainly on core skill (+15.9); the H record moved the rating by +3.9 (weight 0.11). Tier 3 (ranks 3–20 of the top 20): positions within a tier are not statistically separable (adjacent rank scores within one SE).
**Scout (verify on video).** The weakest scenario — entry under pressure (+8.4 per 100 balls, n=49) — and the lowest process signal (dot suppression +0.79σ). Check whether the numbers reflect shot/ball execution or match situation.
**Coach (best role / entry point).** Highest H value in innings 2, death (+82.4 per 100 H balls, n=32); current data role: middle order (4-5).

_All figures from the pipeline outputs (2022–26, rain-shortened matches excluded from pressure sets). Scenario values are raw means over H deliveries in that scenario (not shrunk)._

_(card file: `outputs/cards/bat_07_heinrich_klaasen.md`)_

#### Andre Russell — batter #8 (tier 3)

- **Rating:** +21.11 runs per 100 high-leverage balls vs expectation (SE 5.93; rank score +15.18). Core skill +20.53, pressure deviation +0.58.
- **Evidence:** 195 H balls; 2022–2025 (4 seasons); raw H +25.79 ± 19.15; retention R -0.129 [-0.502, +0.184] (pooled bootstrap).
- **Role (data):** finisher (6-7); median batting position 7; mean entry over 13.3. Latest team: West Indies.
- **Process on H (σ vs eligible players):** boundary elevation +1.39σ; strike-rate delta +1.01σ; wicket security +0.23σ.
- **Strongest scenario:** post-wicket (12 deliveries after a wicket) (-3.9 per 100 balls, n=125). **Weakest scenario:** death overs, 2nd innings (-27.7 per 100 balls, n=46).
- **Impact above replacement:** +8.65 runs / +0.124 wins per season (49 H balls per season).

**Analyst.** Rated mainly on core skill (+20.5); the H record moved the rating by +0.6 (weight 0.11). Tier 3 (ranks 3–20 of the top 20): positions within a tier are not statistically separable (adjacent rank scores within one SE).
**Scout (verify on video).** The weakest scenario — death overs, 2nd innings (-27.7 per 100 balls, n=46) — and the lowest process signal (wicket security +0.23σ). Check whether the numbers reflect shot/ball execution or match situation.
**Coach (best role / entry point).** Highest H value in innings 1, death (+56.7 per 100 H balls, n=107); current data role: finisher (6-7).

_All figures from the pipeline outputs (2022–26, rain-shortened matches excluded from pressure sets). Scenario values are raw means over H deliveries in that scenario (not shrunk)._

_(card file: `outputs/cards/bat_08_andre_russell.md`)_

#### Travis Head — batter #9 (tier 3)

- **Rating:** +20.43 runs per 100 high-leverage balls vs expectation (SE 6.35; rank score +14.08). Core skill +17.37, pressure deviation +3.07.
- **Evidence:** 479 H balls; 2023–2026 (4 seasons); raw H +30.52 ± 15.49; retention R +0.196 [-0.046, +0.431] (pooled bootstrap).
- **Role (data):** opener; median batting position 2; mean entry over 1.0. Latest team: Sunrisers Hyderabad.
- **Process on H (σ vs eligible players):** boundary elevation +1.14σ; strike-rate delta +0.98σ; dot suppression +0.89σ.
- **Strongest scenario:** entry under pressure (+36.9 per 100 balls, n=131). **Weakest scenario:** post-wicket (12 deliveries after a wicket) (-10.3 per 100 balls, n=79).
- **Impact above replacement:** +20.43 runs / +0.292 wins per season (120 H balls per season).

**Analyst.** Rated mainly on core skill (+17.4); the H record moved the rating by +3.1 (weight 0.23). Tier 3 (ranks 3–20 of the top 20): positions within a tier are not statistically separable (adjacent rank scores within one SE).
**Scout (verify on video).** The weakest scenario — post-wicket (12 deliveries after a wicket) (-10.3 per 100 balls, n=79) — and the lowest process signal (dot suppression +0.89σ). Check whether the numbers reflect shot/ball execution or match situation.
**Coach (best role / entry point).** Highest H value in innings 1, powerplay (+39.4 per 100 H balls, n=369); current data role: opener.

_All figures from the pipeline outputs (2022–26, rain-shortened matches excluded from pressure sets). Scenario values are raw means over H deliveries in that scenario (not shrunk)._

_(card file: `outputs/cards/bat_09_travis_head.md`)_

#### Priyansh Arya — batter #10 (tier 3)

- **Rating:** +22.57 runs per 100 high-leverage balls vs expectation (SE 8.82; rank score +13.75). Core skill +21.11, pressure deviation +1.46.
- **Evidence:** 200 H balls; 2025–2026 (2 seasons); raw H +34.06 ± 23.22; retention R +0.108 [-0.251, +0.526] (pooled bootstrap).
- **Role (data):** opener; median batting position 1; mean entry over 1.0. Latest team: Punjab Kings.
- **Process on H (σ vs eligible players):** dot suppression +2.49σ; strike-rate delta +2.22σ; boundary elevation +1.48σ.
- **Strongest scenario:** post-wicket (12 deliveries after a wicket) (+182.5 per 100 balls, n=21). **Weakest scenario:** entry under pressure (+33.1 per 100 balls, n=29).
- **Impact above replacement:** +19.20 runs / +0.274 wins per season (100 H balls per season).

**Analyst.** Rated mainly on core skill (+21.1); the H record moved the rating by +1.5 (weight 0.11). Tier 3 (ranks 3–20 of the top 20): positions within a tier are not statistically separable (adjacent rank scores within one SE).
**Scout (verify on video).** The weakest scenario — entry under pressure (+33.1 per 100 balls, n=29) — and the lowest process signal (boundary elevation +1.48σ). Check whether the numbers reflect shot/ball execution or match situation.
**Coach (best role / entry point).** Highest H value in innings 1, powerplay (+39.9 per 100 H balls, n=137); current data role: opener.

_All figures from the pipeline outputs (2022–26, rain-shortened matches excluded from pressure sets). Scenario values are raw means over H deliveries in that scenario (not shrunk)._

_(card file: `outputs/cards/bat_10_priyansh_arya.md`)_

#### Jasprit Bumrah — bowler #1 (tier 1)

- **Rating:** +25.27 runs per 100 high-leverage balls vs expectation (SE 5.05; rank score +20.22). Core skill +27.58, pressure deviation -2.30.
- **Evidence:** 539 H balls; 2022–2026 (5 seasons); raw H +10.70 ± 11.46; retention R -0.165 [-0.338, +0.007] (pooled bootstrap).
- **Role (data):** pace, mainly powerplay; overs share PP/middle/death 44%/24%/32%. Latest team: India.
- **Process on H (σ vs eligible players):** boundary suppression +2.00σ; economy suppression +1.46σ; dot rate +1.42σ.
- **Strongest scenario:** powerplay vs set openers (+24.3 per 100 balls, n=98). **Weakest scenario:** death overs, 2nd innings (-0.9 per 100 balls, n=57).
- **Impact above replacement:** +27.71 runs / +0.396 wins per season (108 H balls per season).

**Analyst.** Rated mainly on core skill (+27.6); the H record moved the rating by -2.3 (weight 0.21). Tier 1 (ranks 1–2 of the top 20): positions within a tier are not statistically separable (adjacent rank scores within one SE).
**Scout (verify on video).** The weakest scenario — death overs, 2nd innings (-0.9 per 100 balls, n=57) — and the lowest process signal (dot rate +1.42σ). Check whether the numbers reflect shot/ball execution or match situation.
**Coach (best role / entry point).** Highest H value in innings 1, death (+54.5 per 100 H balls, n=108); current data role: pace, mainly powerplay.

_All figures from the pipeline outputs (2022–26, rain-shortened matches excluded from pressure sets). Scenario values are raw means over H deliveries in that scenario (not shrunk)._

_(card file: `outputs/cards/bowl_01_jasprit_bumrah.md`)_

#### Sunil Narine — bowler #2 (tier 1)

- **Rating:** +20.86 runs per 100 high-leverage balls vs expectation (SE 3.64; rank score +17.21). Core skill +20.66, pressure deviation +0.19.
- **Evidence:** 203 H balls; 2022–2026 (5 seasons); raw H +24.05 ± 13.67; retention R +0.046 [-0.207, +0.316] (pooled bootstrap).
- **Role (data):** spin, mainly middle; overs share PP/middle/death 14%/72%/14%. Latest team: Kolkata Knight Riders.
- **Process on H (σ vs eligible players):** dot rate +1.74σ; economy suppression +1.54σ; boundary suppression +1.18σ.
- **Strongest scenario:** powerplay vs set openers (+95.8 per 100 balls, n=29). **Weakest scenario:** post-wicket (12 deliveries after a wicket) (+44.8 per 100 balls, n=102).
- **Impact above replacement:** +8.64 runs / +0.123 wins per season (41 H balls per season).

**Analyst.** Rated mainly on core skill (+20.7); the H record moved the rating by +0.2 (weight 0.11). Tier 1 (ranks 1–2 of the top 20): positions within a tier are not statistically separable (adjacent rank scores within one SE).
**Scout (verify on video).** The weakest scenario — post-wicket (12 deliveries after a wicket) (+44.8 per 100 balls, n=102) — and the lowest process signal (boundary suppression +1.18σ). Check whether the numbers reflect shot/ball execution or match situation.
**Coach (best role / entry point).** Highest H value in innings 2, death (+87.3 per 100 H balls, n=35); current data role: spin, mainly middle.

_All figures from the pipeline outputs (2022–26, rain-shortened matches excluded from pressure sets). Scenario values are raw means over H deliveries in that scenario (not shrunk)._

_(card file: `outputs/cards/bowl_02_sunil_narine.md`)_

#### Matheesha Pathirana — bowler #3 (tier 2)

- **Rating:** +16.36 runs per 100 high-leverage balls vs expectation (SE 4.09; rank score +12.27). Core skill +16.71, pressure deviation -0.35.
- **Evidence:** 331 H balls; 2022–2026 (5 seasons); raw H +19.07 ± 10.82; retention R +0.104 [-0.104, +0.331] (pooled bootstrap).
- **Role (data):** pace, mainly death; overs share PP/middle/death 7%/46%/47%. Latest team: Kolkata Knight Riders.
- **Process on H (σ vs eligible players):** economy suppression +0.91σ; boundary suppression +0.77σ; wicket-taking +0.41σ.
- **Strongest scenario:** post-wicket (12 deliveries after a wicket) (+31.3 per 100 balls, n=199). **Weakest scenario:** death overs, 2nd innings (+10.0 per 100 balls, n=88).
- **Impact above replacement:** +11.12 runs / +0.159 wins per season (66 H balls per season).

**Analyst.** Rated mainly on core skill (+16.7); the H record moved the rating by -0.3 (weight 0.14). Tier 2 (ranks 3–20 of the top 20): positions within a tier are not statistically separable (adjacent rank scores within one SE).
**Scout (verify on video).** The weakest scenario — death overs, 2nd innings (+10.0 per 100 balls, n=88) — and the lowest process signal (wicket-taking +0.41σ). Check whether the numbers reflect shot/ball execution or match situation.
**Coach (best role / entry point).** Highest H value in innings 1, death (+32.0 per 100 H balls, n=138); current data role: pace, mainly death.

_All figures from the pipeline outputs (2022–26, rain-shortened matches excluded from pressure sets). Scenario values are raw means over H deliveries in that scenario (not shrunk)._

_(card file: `outputs/cards/bowl_03_matheesha_pathirana.md`)_

#### Nathan Ellis — bowler #4 (tier 2)

- **Rating:** +15.40 runs per 100 high-leverage balls vs expectation (SE 3.23; rank score +12.17). Core skill +14.78, pressure deviation +0.61.
- **Evidence:** 344 H balls; 2022–2026 (5 seasons); raw H +22.29 ± 9.84; retention R +0.008 [-0.203, +0.212] (pooled bootstrap).
- **Role (data):** pace, mainly death; overs share PP/middle/death 28%/33%/39%. Latest team: Australia.
- **Process on H (σ vs eligible players):** boundary suppression +1.70σ; economy suppression +1.25σ; dot rate -0.31σ.
- **Strongest scenario:** powerplay vs set openers (+22.0 per 100 balls, n=39). **Weakest scenario:** death overs, 2nd innings (-4.3 per 100 balls, n=65).
- **Impact above replacement:** +10.89 runs / +0.155 wins per season (69 H balls per season).

**Analyst.** Rated mainly on core skill (+14.8); the H record moved the rating by +0.6 (weight 0.13). Tier 2 (ranks 3–20 of the top 20): positions within a tier are not statistically separable (adjacent rank scores within one SE).
**Scout (verify on video).** The weakest scenario — death overs, 2nd innings (-4.3 per 100 balls, n=65) — and the lowest process signal (dot rate -0.31σ). Check whether the numbers reflect shot/ball execution or match situation.
**Coach (best role / entry point).** Highest H value in innings 2, powerplay (+65.8 per 100 H balls, n=29); current data role: pace, mainly death.

_All figures from the pipeline outputs (2022–26, rain-shortened matches excluded from pressure sets). Scenario values are raw means over H deliveries in that scenario (not shrunk)._

_(card file: `outputs/cards/bowl_04_nathan_ellis.md`)_

#### Rashid Khan (Afghanistan, Gujarat Titans) — bowler #5 (tier 2)

- **Rating:** +12.24 runs per 100 high-leverage balls vs expectation (SE 3.88; rank score +8.37). Core skill +12.11, pressure deviation +0.13.
- **Evidence:** 424 H balls; 2022–2026 (5 seasons); raw H +10.07 ± 10.64; retention R -0.071 [-0.225, +0.093] (pooled bootstrap).
- **Role (data):** spin, mainly middle; overs share PP/middle/death 10%/71%/19%. Latest team: Afghanistan.
- **Process on H (σ vs eligible players):** boundary suppression +0.77σ; economy suppression +0.34σ; wicket-taking -0.13σ.
- **Strongest scenario:** death overs, 2nd innings (+52.6 per 100 balls, n=68). **Weakest scenario:** powerplay vs set openers (-4.2 per 100 balls, n=35).
- **Impact above replacement:** +10.75 runs / +0.153 wins per season (85 H balls per season).

**Analyst.** Rated mainly on core skill (+12.1); the H record moved the rating by +0.1 (weight 0.20). Tier 2 (ranks 3–20 of the top 20): positions within a tier are not statistically separable (adjacent rank scores within one SE).
**Scout (verify on video).** The weakest scenario — powerplay vs set openers (-4.2 per 100 balls, n=35) — and the lowest process signal (wicket-taking -0.13σ). Check whether the numbers reflect shot/ball execution or match situation.
**Coach (best role / entry point).** Highest H value in innings 2, death (+52.6 per 100 H balls, n=68); current data role: spin, mainly middle.

_All figures from the pipeline outputs (2022–26, rain-shortened matches excluded from pressure sets). Scenario values are raw means over H deliveries in that scenario (not shrunk)._

_(card file: `outputs/cards/bowl_05_rashid_khan_afghanistan_gujarat_titans.md`)_

#### Eshan Malinga — bowler #6 (tier 2)

- **Rating:** +11.94 runs per 100 high-leverage balls vs expectation (SE 4.36; rank score +7.57). Core skill +10.20, pressure deviation +1.74.
- **Evidence:** 227 H balls; 2024–2026 (3 seasons); raw H +20.61 ± 12.11; retention R +0.163 [-0.017, +0.364] (pooled bootstrap).
- **Role (data):** pace, mainly middle; overs share PP/middle/death 33%/36%/32%. Latest team: Sri Lanka.
- **Process on H (σ vs eligible players):** boundary suppression +1.91σ; economy suppression +1.50σ; dot rate +0.07σ.
- **Strongest scenario:** post-wicket (12 deliveries after a wicket) (+38.8 per 100 balls, n=94). **Weakest scenario:** powerplay vs set openers (+14.4 per 100 balls, n=35).
- **Impact above replacement:** +9.36 runs / +0.134 wins per season (76 H balls per season).

**Analyst.** Rated mainly on core skill (+10.2); the H record moved the rating by +1.7 (weight 0.11). Tier 2 (ranks 3–20 of the top 20): positions within a tier are not statistically separable (adjacent rank scores within one SE).
**Scout (verify on video).** The weakest scenario — powerplay vs set openers (+14.4 per 100 balls, n=35) — and the lowest process signal (dot rate +0.07σ). Check whether the numbers reflect shot/ball execution or match situation.
**Coach (best role / entry point).** Highest H value in innings 2, middle (+55.7 per 100 H balls, n=20); current data role: pace, mainly middle.

_All figures from the pipeline outputs (2022–26, rain-shortened matches excluded from pressure sets). Scenario values are raw means over H deliveries in that scenario (not shrunk)._

_(card file: `outputs/cards/bowl_06_eshan_malinga.md`)_

#### Josh Hazlewood — bowler #7 (tier 2)

- **Rating:** +11.91 runs per 100 high-leverage balls vs expectation (SE 4.63; rank score +7.28). Core skill +10.05, pressure deviation +1.86.
- **Evidence:** 570 H balls; 2022–2026 (5 seasons); raw H +25.87 ± 10.32; retention R +0.025 [-0.147, +0.188] (pooled bootstrap).
- **Role (data):** pace, mainly powerplay; overs share PP/middle/death 53%/19%/28%. Latest team: Royal Challengers Bengaluru.
- **Process on H (σ vs eligible players):** wicket-taking +1.26σ; dot rate +1.06σ; economy suppression +0.79σ.
- **Strongest scenario:** death overs, 2nd innings (+53.3 per 100 balls, n=77). **Weakest scenario:** post-wicket (12 deliveries after a wicket) (+20.0 per 100 balls, n=236).
- **Impact above replacement:** +14.07 runs / +0.201 wins per season (114 H balls per season).

**Analyst.** Rated mainly on core skill (+10.1); the H record moved the rating by +1.9 (weight 0.20). Tier 2 (ranks 3–20 of the top 20): positions within a tier are not statistically separable (adjacent rank scores within one SE).
**Scout (verify on video).** The weakest scenario — post-wicket (12 deliveries after a wicket) (+20.0 per 100 balls, n=236) — and the lowest process signal (economy suppression +0.79σ). Check whether the numbers reflect shot/ball execution or match situation.
**Coach (best role / entry point).** Highest H value in innings 2, death (+53.3 per 100 H balls, n=77); current data role: pace, mainly powerplay.

_All figures from the pipeline outputs (2022–26, rain-shortened matches excluded from pressure sets). Scenario values are raw means over H deliveries in that scenario (not shrunk)._

_(card file: `outputs/cards/bowl_07_josh_hazlewood.md`)_

#### Bhuvneshwar Kumar — bowler #8 (tier 2)

- **Rating:** +9.88 runs per 100 high-leverage balls vs expectation (SE 3.99; rank score +5.88). Core skill +11.08, pressure deviation -1.21.
- **Evidence:** 771 H balls; 2022–2026 (5 seasons); raw H +11.84 ± 9.73; retention R -0.003 [-0.159, +0.151] (pooled bootstrap).
- **Role (data):** pace, mainly powerplay; overs share PP/middle/death 60%/8%/32%. Latest team: India.
- **Process on H (σ vs eligible players):** boundary suppression +0.42σ; economy suppression -0.02σ; wicket-taking -0.37σ.
- **Strongest scenario:** post-wicket (12 deliveries after a wicket) (+32.6 per 100 balls, n=268). **Weakest scenario:** death overs, 2nd innings (-7.3 per 100 balls, n=140).
- **Impact above replacement:** +15.89 runs / +0.227 wins per season (154 H balls per season).

**Analyst.** Rated mainly on core skill (+11.1); the H record moved the rating by -1.2 (weight 0.27). Tier 2 (ranks 3–20 of the top 20): positions within a tier are not statistically separable (adjacent rank scores within one SE).
**Scout (verify on video).** The weakest scenario — death overs, 2nd innings (-7.3 per 100 balls, n=140) — and the lowest process signal (wicket-taking -0.37σ). Check whether the numbers reflect shot/ball execution or match situation.
**Coach (best role / entry point).** Highest H value in innings 1, death (+23.2 per 100 H balls, n=138); current data role: pace, mainly powerplay.

_All figures from the pipeline outputs (2022–26, rain-shortened matches excluded from pressure sets). Scenario values are raw means over H deliveries in that scenario (not shrunk)._

_(card file: `outputs/cards/bowl_08_bhuvneshwar_kumar.md`)_

#### Mujeeb Ur Rahman — bowler #9 (tier 2)

- **Rating:** +9.82 runs per 100 high-leverage balls vs expectation (SE 4.42; rank score +5.41). Core skill +7.64, pressure deviation +2.19.
- **Evidence:** 162 H balls; 2022–2026 (5 seasons); raw H +30.89 ± 15.42; retention R +0.100 [-0.099, +0.328] (pooled bootstrap).
- **Role (data):** spin, mainly powerplay; overs share PP/middle/death 51%/40%/9%. Latest team: Afghanistan.
- **Process on H (σ vs eligible players):** dot rate +2.07σ; economy suppression +2.04σ; boundary suppression +1.20σ.
- **Strongest scenario:** post-wicket (12 deliveries after a wicket) (+61.6 per 100 balls, n=60). **Weakest scenario:** powerplay vs set openers (-34.3 per 100 balls, n=21).
- **Impact above replacement:** +3.32 runs / +0.047 wins per season (32 H balls per season).

**Analyst.** Rated mainly on core skill (+7.6); the H record moved the rating by +2.2 (weight 0.08). Tier 2 (ranks 3–20 of the top 20): positions within a tier are not statistically separable (adjacent rank scores within one SE).
**Scout (verify on video).** The weakest scenario — powerplay vs set openers (-34.3 per 100 balls, n=21) — and the lowest process signal (boundary suppression +1.20σ). Check whether the numbers reflect shot/ball execution or match situation.
**Coach (best role / entry point).** Highest H value in innings 2, powerplay (+49.7 per 100 H balls, n=62); current data role: spin, mainly powerplay.

_All figures from the pipeline outputs (2022–26, rain-shortened matches excluded from pressure sets). Scenario values are raw means over H deliveries in that scenario (not shrunk)._

_(card file: `outputs/cards/bowl_09_mujeeb_ur_rahman.md`)_

#### Varun Chakravarthy — bowler #10 (tier 2)

- **Rating:** +7.61 runs per 100 high-leverage balls vs expectation (SE 3.68; rank score +3.92). Core skill +7.14, pressure deviation +0.47.
- **Evidence:** 367 H balls; 2022–2026 (5 seasons); raw H +7.98 ± 11.24; retention R -0.116 [-0.294, +0.059] (pooled bootstrap).
- **Role (data):** spin, mainly middle; overs share PP/middle/death 16%/69%/15%. Latest team: India.
- **Process on H (σ vs eligible players):** dot rate +0.51σ; economy suppression -0.00σ; wicket-taking -0.33σ.
- **Strongest scenario:** death overs, 2nd innings (+68.7 per 100 balls, n=50). **Weakest scenario:** powerplay vs set openers (-27.8 per 100 balls, n=43).
- **Impact above replacement:** +5.90 runs / +0.084 wins per season (73 H balls per season).

**Analyst.** Rated mainly on core skill (+7.1); the H record moved the rating by +0.5 (weight 0.17). Tier 2 (ranks 3–20 of the top 20): positions within a tier are not statistically separable (adjacent rank scores within one SE).
**Scout (verify on video).** The weakest scenario — powerplay vs set openers (-27.8 per 100 balls, n=43) — and the lowest process signal (wicket-taking -0.33σ). Check whether the numbers reflect shot/ball execution or match situation.
**Coach (best role / entry point).** Highest H value in innings 2, death (+68.7 per 100 H balls, n=50); current data role: spin, mainly middle.

_All figures from the pipeline outputs (2022–26, rain-shortened matches excluded from pressure sets). Scenario values are raw means over H deliveries in that scenario (not shrunk)._

_(card file: `outputs/cards/bowl_10_varun_chakravarthy.md`)_

### 4.9 Vignesh Puthur

| role | in_data | player_id | matches | seasons | deliveries | H_deliveries | min_H_for_eligibility | eligible | rating_per_100 (model, shrunk) | core_skill_per_100 | raw_H_per_100 | raw_H_se_per_100 (simple) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bat | False |  |  |  |  |  |  |  |  |  |  |  |
| bowler |  | 48557.0 | 5.0 | 2025.0 | 73.0 | 4.0 | 154.0 | False | -4.09 | -3.97 | -77.7 | 131.91 |

In the data as a bowler only: 5 matches in 2025.0, 73 deliveries, 4 on high-leverage deliveries (eligibility needs 154). Not eligible; the model rating (-4.09 per 100 H balls) is almost entirely the population prior. The raw H value (-77.7 ± 131.91) is too noisy to support any claim. He is in RR's IPL 2026 squad.

## 5. Reliability and limitations

### 5.1 Trust table (split-half reliability, mean of 50 random match splits, Spearman–Brown; season-to-season)

Chart: `outputs/charts/trust_table.png`.

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

### 5.2 Phase 6 / 6B findings

#### 5.2.1 Headline finding
**What repeats under pressure is process, not outcome.** Split-half reliability (mean of 50 random match splits, Spearman–Brown, 95% bootstrap interval), H = top 25% LI within innings:
- Repeats (batters): strike-rate delta 0.43 [0.26, 0.55], dot suppression 0.42 [0.26, 0.54], boundary elevation 0.45 [0.29, 0.58]; process composite on H 0.41 [0.23, 0.52]; run-out avoidance 0.23 [0.05, 0.37].
- Does not repeat (batters): WP value P 0.02 [−0.49, 0.31], wicket security 0.19 [−0.11, 0.36], retention R 0.06 [−0.35, 0.31], entry resilience, collapse resistance, post-wicket response, squeeze (pooled), partner effect.
- Bowlers: no H metric repeats individually — P −0.14, economy −0.12, dots 0.06, wickets 0.14, boundaries −0.11, bounce-back, post-wicket, execution drift, retention R (all intervals span 0). Only variation (entropy H − L) passes, 0.22 [0.06, 0.38], and it predicts next-season performance *negatively* (r −0.13 [−0.22, −0.03]).
- True variance of pressure-specific run value (H beyond what L predicts): batters 0.0029 runs²/ball [−0.012, 0.014] (26% of true H variance, interval spans 0); bowlers ≈ 0 (true H variance itself estimated negative). Phase 7's empirical Bayes H update is therefore small for batters and ≈ 0 for bowlers.
- Out of sample (next season, grouped by player), baseline skill alone predicts the pressure target better than weighted pressure metrics: batters r 0.16 [0.01, 0.30] vs ridge 0.07 / equal weights 0.10; bowlers 0.07 [−0.02, 0.18] vs ridge −0.12 / equal −0.13.

#### 5.2.2 Headline findings
- **Pressure-specific variation exists on both sides** (mixed-model LRT p = 2.7e-8 batters, 6.2e-6 bowlers; slope SD ≈ 0.065 runs/ball), **but is too small relative to sampling noise to estimate reliably per player with 2022–26 data. Core skill is the best predictor of pressure performance.**
- **Death pace specialists** (method E within H) show the strongest pressure signal (reliability: P 0.47, economy 0.40, boundary suppression 0.53; n ≈ 45–51 bowlers) but fail the pre-specified rule; reported as a lead for further work, **not used in the ranking**.
- **Role-group pooling raised reliability without improving prediction**: the gain came from shared role means (both split halves share the same role mean), not from player-level signal.

Pressure-specific deviation variance (Phase 6, runs² per ball):

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

Signal recovery — every pre-specified method (pass = reliability interval above 0 AND clear out-of-sample gain):

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

### 5.3 Out-of-time validation (rating stage rebuilt on 2022–24, scored on 2025–26)

Chart: `outputs/charts/out_of_time_validation.png`. Per-ball values come from league-level models fitted out-of-fold over all seasons; the player-level rating stage uses no 2025–26 data.

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

| side | top20_2022_24 | top20_with_2025_26_H_data | top20_above_average_on_H_2025_26 | share | share_ci95_wilson | share_for_random_eligible_player | average_used (runs per H ball, eligible test players) | top20_above_expectation_(>0) |
|---|---|---|---|---|---|---|---|---|
| bat | 20 | 16 | 10 | 0.625 | [0.39, 0.82] | 0.557 | 0.1001 | 12 |
| bowl | 20 | 18 | 13 | 0.722 | [0.49, 0.88] | 0.58 | -0.079 | 11 |

### 5.4 Knockout check (31 matches — small sample)

| side | matches | group | players | H_deliveries | n_matches | run_value_per_100_H_balls | ci95 (bootstrap over matches) |
|---|---|---|---|---|---|---|---|
| bat | knockout matches | final top 20 | 13 | 197 | 16 | -30.41 | [-101.10, 21.27] |
| bat | knockout matches | all other players | 135 | 1541 | 29 | 1.76 | [-11.95, 13.15] |
| bat | all other matches | final top 20 | 20 | 6516 | 411 | 27.5 | [20.74, 34.25] |
| bat | all other matches | all other players | 686 | 37816 | 762 | -2.0 | [-4.44, 0.29] |
| bowl | knockout matches | final top 20 | 12 | 337 | 19 | 17.23 | [-9.09, 46.33] |
| bowl | knockout matches | all other players | 104 | 1469 | 30 | -0.65 | [-14.92, 16.95] |
| bowl | all other matches | final top 20 | 20 | 7384 | 478 | 13.65 | [8.42, 19.11] |
| bowl | all other matches | all other players | 506 | 39263 | 764 | -5.32 | [-7.61, -2.86] |

### 5.5 Face validity (data reasons only)

| player | side | status | reason |
|---|---|---|---|
| Rinku Singh | bat | eligible, rank 38 of 82 | rating +9.2 (core +7.0, pressure dev +2.2); rank score +5.5; raw H +24.0 ± 13.0 over 229 H balls |
| Hardik Pandya | bat | not eligible | raw H run value -1.6 per 100 balls ≤ 0 (delivery requirement). Rating +1.6 (core +2.3, pressure dev -0.7); raw H -1.6 |
| Virat Kohli | bat | eligible, rank 46 of 82 | rating +6.4 (core +5.1, pressure dev +1.4); rank score +3.7; raw H +9.3 ± 7.3 over 745 H balls |
| Mahendra Singh Dhoni | bat | eligible, rank 76 of 82 | rating +2.8 (core +1.8, pressure dev +1.0); rank score -2.5; raw H +14.6 ± 21.8 over 139 H balls |
| Arshdeep Singh | bowl | eligible, rank 27 of 41 | rating +2.2 (core +4.2, pressure dev -2.0); rank score -1.6; raw H +2.1 ± 7.3 over 1118 H balls |
| Kuldeep Yadav | bowl | not eligible | raw H run value -7.6 per 100 balls ≤ 0 (delivery requirement). Rating +0.5 (core +1.8, pressure dev -1.3); raw H -7.6 |
| Hardik Pandya | bowl | not eligible | retention interval entirely below 0 [-0.386, -0.059]. Rating -4.8 (core -0.8, pressure dev -4.0); raw H -29.9 |

### 5.6 Sensitivity (top-10 overlap with the final list)

**Batters**

| variant | top10_overlap_with_primary | only_in_primary | only_in_variant |
|---|---|---|---|
| H20 | 9 | Andre Russell | Glenn Maxwell |
| H33 | 8 | Travis Head, Priyansh Arya | Rajat Patidar, Muhammad Waseem |
| threshold −25% | 10 |  |  |
| threshold +25% | 10 |  |  |
| IPL only | 7 | Finn Allen, Tim David, Andre Russell | Nicholas Pooran, Sunil Narine, Liam Livingstone |
| noise-aware delivery condition | 10 |  |  |
| bivariate method | 8 | Travis Head, Priyansh Arya | Shreyas Iyer, Reeza Hendricks |

**Bowlers**

| variant | top10_overlap_with_primary | only_in_primary | only_in_variant |
|---|---|---|---|
| H20 | 7 | Bhuvneshwar Kumar, Mujeeb Ur Rahman, Varun Chakravarthy | Mohammed Siraj, Joshua Little, Taskin Ahmed |
| H33 | 7 | Bhuvneshwar Kumar, Mujeeb Ur Rahman, Varun Chakravarthy | Joshua Little, Mohammad Wasim, Taskin Ahmed |
| threshold −25% | 9 | Varun Chakravarthy | Abrar Ahmed |
| threshold +25% | 9 | Mujeeb Ur Rahman | Mohammed Siraj |
| IPL only | 6 | Nathan Ellis, Eshan Malinga, Mujeeb Ur Rahman, Varun Chakravarthy | Mohammed Siraj, Krunal Pandya, Arshdeep Singh, Avesh Khan |
| noise-aware delivery condition | 9 | Varun Chakravarthy | Blessing Muzarabani |
| L-only (previous) method | 7 | Eshan Malinga, Josh Hazlewood, Mujeeb Ur Rahman | Mohsin Khan, Shaheen Afridi, Arshdeep Singh |

Rating-method sensitivity: the bivariate batter rating (tested alternative) is in `outputs/tables/appendix_bivariate_batters.csv`.

### 5.7 Tiers

Batters: tier 1 = rank 1 (Vaibhav Sooryavanshi); tier 2 = rank 2; ranks 3–20 form one tier. Bowlers: tier 1 = ranks 1–2; ranks 3–20 form one tier. Ordering inside a tier follows rank score but is not statistically supported.

### 5.8 Traps and leads

- **Yorker-label trap:** delivery_style is recorded after the ball; a missed yorker tends to be logged as a full toss (yorkers 0.84 runs per ball vs full tosses 1.87), so 'yorker execution' measures the label, not the skill. Reported, never ranked.
- **Death-pace lead:** death pace specialists (pace, ≥30% of overs at the death; 45–51 bowlers) show the strongest pressure signal within H (reliability: WP value 0.47, economy 0.40, boundary suppression 0.53) but fail the pre-specified rule; a lead for further work, not used in the ranking.
- **Role pooling:** raised reliability without improving prediction — the gain came from shared role means, not player-level signal.

### 5.9 Limitations

- No ball-tracking: no line/length, no false-shot or beaten-bat data; quality of contact cannot be measured.
- bowl_speed is missing (0) on 38.7% of deliveries; speed is not used.
- Knockout sample is 31 matches; knockout results are descriptive only.
- 32 rain-shortened matches are excluded from pressure sets; 8 rain revisions were not recorded in matches.csv and were detected from the data.
- 1st-innings WP is inherently less certain (out-of-fold AUC 0.7048); 1st-innings leverage comes mainly from wicket risk and middle overs rarely reach H.
- Last-over WP rests on cells of 5–25 matches; it is the least certain part of the WP model.
- Per-player pressure-specific skill is not reliably estimable with 2022–26 data; ratings lean on core skill by design.
- Bowler wicket-taking has no estimable between-player variance; bowler ratings carry no wicket value beyond what economy reflects.
- WP value P and retention R are not repeatable per player; R is used only as a safety gate.
- Out of time, our rating did not beat a plain career average (batters r 0.149 vs 0.199; bowlers 0.419 vs 0.455); intervals overlap.
- Per-ball values use league-level models fitted out-of-fold over all seasons (no player's own match informs its value, but model parameters see all years).
- The delivery requirement (raw H > 0) is noisy for players near 0 (e.g. Rajat Patidar −0.30, Maheesh Theekshana −0.48 per 100 H balls).
- Player identity: Basil Hameed (UAE) may be one person under two ids; negligible sample.

### 5.10 Picks we are least sure of

| side | rank | player | why less sure |
|---|---|---|---|
| Batters | 1 | Vaibhav Sooryavanshi | only 2 seasons of data |
| Batters | 3 | Finn Allen | raw H +16.9 smaller than its SE 20.2 |
| Batters | 8 | Andre Russell | drops out of the top 10 in 2 sensitivity variants |
| Batters | 9 | Travis Head | drops out of the top 10 in 2 sensitivity variants |
| Batters | 10 | Priyansh Arya | drops out of the top 10 in 2 sensitivity variants; only 2 seasons of data |
| Bowlers | 1 | Jasprit Bumrah | raw H +10.7 smaller than its SE 11.5; retention interval upper end +0.007 (just passes the gate) |
| Bowlers | 5 | Rashid Khan (Afghanistan, Gujarat Titans) | raw H +10.1 smaller than its SE 10.6 |
| Bowlers | 6 | Eshan Malinga | drops out of the top 10 in 2 sensitivity variants |
| Bowlers | 8 | Bhuvneshwar Kumar | drops out of the top 10 in 2 sensitivity variants |
| Bowlers | 9 | Mujeeb Ur Rahman | drops out of the top 10 in 5 sensitivity variants |
| Bowlers | 10 | Varun Chakravarthy | drops out of the top 10 in 5 sensitivity variants; raw H +8.0 smaller than its SE 11.2 |

Rules used: drops out of the top 10 in ≥2 sensitivity variants; ≤2 seasons; raw H smaller than its own SE; retention interval upper end < 0.02.

## 6. Appendix

### 6.1 Thresholds and parameters

| area | parameter | value | reason |
|---|---|---|---|
| Data | Total runs | runs + extras1_runs + extras_2_runs | reconciles 804/809 results |
| Data | No-balls | extras_type Overstep and 'other' | 'other' legal → 244 overs of 7–9 balls |
| WP | Venue par window | 730 days, shrink k estimated (12.0) | leak-free |
| WP | CV folds | 5, grouped by match | no leakage |
| WP | Endgame model | fit on ≤18 balls left; blend 24→12 | chosen on ≤30-ball log-loss |
| Leverage | H | top 25% LI within innings (H20, H33 robustness) | global cut-off skewed to 2nd innings |
| Leverage | Rain-shortened | excluded from H/L (switch EXCLUDE_SHORTENED_FROM_PRESSURE) | WP not trained on them |
| Scenarios | Post-wicket / entry / squeeze | 12 deliveries; first 10 balls faced when arriving in H; after 3+ dots | brief |
| Scenarios | Collapse | team WP < 0.35 at the batter's first ball | brief |
| Metrics | Report minimums | {'skill': 100, 'temperament': 30, 'yorker': 20, 'variation': 60, 'partner': 100} | Phase 5 summaries |
| Phase 6 | Reliability | mean SB over 50 random match splits; 200 bootstrap | single splits unstable |
| Phase 6B | Mixed-model interval | parametric bootstrap, 100 resamples | profile likelihood not computable reliably |
| Rating | Bootstrap | 2000 resamples for SE; retention pooled over 5 seeds | stable gates |
| Rating | Eligibility | bat ≥108, bowl ≥154 H balls; active 2025/26 | 2 × median regular season |
| Rating | Delivery requirement | raw H run value > 0 | brief |
| Rating | Retention gate | pooled R 95% interval entirely < 0 → exclude | safety gate |
| Rating | Tiers | gap > larger adjacent SE → new tier | separability |
| Rating | Recency | λ ∈ [1.0, 0.8, 0.6, 0.4] tested; not kept | no clear gain |
| Validation | Out-of-time | 2022–24 → 2025–26; ≥30 test H balls | brief |
| General | Seed | 42; OMP threads 2 | reproducibility |

### 6.2 Decisions log (DECISIONS.md, verbatim)

### Decisions log

Every analysis decision, with its reason. Re-read at the start of each phase; append new decisions; never reverse a
logged decision without the project owner's approval. Status: **approved** = confirmed by the owner;
**proposed** = awaiting approval; **superseded by X** = replaced by a later decision (the final method is P7F-1/P7F-2, P7F-3, P7G-1).

#### Environment
| # | Decision | Reason | Status |
|---|---|---|---|
| E1 | `.venv` built from system Python 3.9.6 (`/usr/bin/python3`), never the FSL Python | reproducibility, independent of other installs | approved |
| E2 | LightGBM not used; scikit-learn HistGradientBoosting instead; no system libraries installed | LightGBM needs libomp, not available | approved |
| E3 | Seeds fixed (42), n_jobs = 2, OMP_NUM_THREADS = 2; one command `python run_all.py` rebuilds everything | reproducibility, MacBook Air limits | approved |
| E4 | Git: `.venv/`, `data_processed/` ignored; raw CSVs committed read-only; commit after every phase | version control | approved |

#### Data (Phase 1)
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

#### Win probability (Phase 2)
| # | Decision | Reason | Status |
|---|---|---|---|
| W1 | Leak-free venue par: same-group league mean over previous 730 days + venue residual shrunk by n/(n+k), k estimated (12.0) | brief; no future data | approved |
| W2 | 1st innings: sign-constrained logistic; 2nd innings: sign-constrained logistic + endgame logistic blended 24→12 balls left | beat monotonic HGB, spline logistic, Markov chain and hybrid on out-of-fold log-loss/Brier for ≤30 balls left | approved |
| W3 | 5-fold CV grouped by match; non-training matches scored by the all-data model | no leakage | approved |
| W4 | Ties = 0.5 in all WP values and swings | brief | approved |
| W5 | End of 1st innings → 1 − WP(start of chase, target = score + 1, chase length = 1st-innings length) | later rain revisions are unknowable at the break | approved |
| W6 | 1st-innings stage interactions (score−par × balls left, wickets × balls left) tested and **rejected** (0.6242 → 0.6240, folds mixed; unconstrained interaction slightly negative). Limitation noted | keep only if out-of-fold log-loss improves | approved |

#### Leverage (Phase 3)
| # | Decision | Reason | Status |
|---|---|---|---|
| L1 | Centred LI: LI_raw = Σ p(o)·\|WP(s_o) − Σ p(o')·WP(s_o')\|; specified version kept as `LI_spec` (appendix) | removes the 1st→2nd-innings model hand-over jump; identical elsewhere (Spearman 0.996) | approved |
| L2 | Primary H = top 25% LI **within each innings**; H20/H33 within innings for robustness; L = the rest | global cut-off put 83% of H in the 2nd innings and none in 1st-innings middle overs | approved |
| L3 | Rain-shortened matches (32) excluded from H and L; kept for conventional statistics | WP model never trained on shortened innings | approved |
| L4 | Outcome model: gradient boosting (8 classes), player-agnostic, out-of-fold by match | lower log-loss than multinomial logistic | approved |

#### Value and metrics (Phases 4–5)
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

#### Phase 6
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

#### Headline finding (Phase 6)
**What repeats under pressure is process, not outcome.** Split-half reliability (mean of 50 random match splits, Spearman–Brown, 95% bootstrap interval), H = top 25% LI within innings:
- Repeats (batters): strike-rate delta 0.43 [0.26, 0.55], dot suppression 0.42 [0.26, 0.54], boundary elevation 0.45 [0.29, 0.58]; process composite on H 0.41 [0.23, 0.52]; run-out avoidance 0.23 [0.05, 0.37].
- Does not repeat (batters): WP value P 0.02 [−0.49, 0.31], wicket security 0.19 [−0.11, 0.36], retention R 0.06 [−0.35, 0.31], entry resilience, collapse resistance, post-wicket response, squeeze (pooled), partner effect.
- Bowlers: no H metric repeats individually — P −0.14, economy −0.12, dots 0.06, wickets 0.14, boundaries −0.11, bounce-back, post-wicket, execution drift, retention R (all intervals span 0). Only variation (entropy H − L) passes, 0.22 [0.06, 0.38], and it predicts next-season performance *negatively* (r −0.13 [−0.22, −0.03]).
- True variance of pressure-specific run value (H beyond what L predicts): batters 0.0029 runs²/ball [−0.012, 0.014] (26% of true H variance, interval spans 0); bowlers ≈ 0 (true H variance itself estimated negative). Phase 7's empirical Bayes H update is therefore small for batters and ≈ 0 for bowlers.
- Out of sample (next season, grouped by player), baseline skill alone predicts the pressure target better than weighted pressure metrics: batters r 0.16 [0.01, 0.30] vs ridge 0.07 / equal weights 0.10; bowlers 0.07 [−0.02, 0.18] vs ridge −0.12 / equal −0.13.

#### Phase 6B — signal recovery (pre-specified; no other methods added)
| # | Decision | Reason | Status |
|---|---|---|---|
| P6B-0 | Methods A–E fixed in advance. Pass = random-split reliability 95% interval above 0 AND bootstrap 95% interval of the out-of-sample gain (weighted r, next-season run value on H, folds grouped by player, over the best baseline) above 0 | owner's stopping rule | approved |
| P6B-1 | Mixed-model slope-variance interval by parametric bootstrap (100 resamples), not profile likelihood: statsmodels' parameter-fixing did not hold the slope variance fixed in either parameterisation. Split-half/season slopes = exact BLUPs from sufficient statistics with variance components fixed at full-data values | a correct profile could not be computed; bootstrap is the allowed alternative | approved (method listed by owner) |
| P6B-2 | **Result: no method passes on either side.** Best baseline = all-delivery run value on both sides (LI-weighted was worse: batters r 0.076 vs 0.140; bowlers −0.141 vs −0.057) | see outputs/06b_signal_recovery.md | recorded |
| P6B-3 | **Pressure-specific bowler skill not detectable in this data.** | stopping rule | recorded |
| P6B-4 | Phase 7 rating = empirical Bayes estimate of H performance (run value per ball on H): prior from all-delivery run value (best baseline), H update in proportion to the estimated pressure-specific deviation variance (Phase 6: batters 0.0029 runs²/ball, interval spans 0; bowlers ≈ 0). No validated pressure-specific component (slope, episode rate) is added | owner's 6B instruction | superseded by P7F-1 (batters) and P7F-2 (bowlers); 'no validated pressure component added' still holds |
| P6B-5 | Retention R stays a condition (exclude if R 95% interval entirely below 0) | P6-5 | approved |

#### Headline findings after Phase 6B (owner-approved wording)
- **Pressure-specific variation exists on both sides** (mixed-model LRT p = 2.7e-8 batters, 6.2e-6 bowlers; slope SD ≈ 0.065 runs/ball), **but is too small relative to sampling noise to estimate reliably per player with 2022–26 data. Core skill is the best predictor of pressure performance.**
- **Death pace specialists** (method E within H) show the strongest pressure signal (reliability: P 0.47, economy 0.40, boundary suppression 0.53; n ≈ 45–51 bowlers) but fail the pre-specified rule; reported as a lead for further work, **not used in the ranking**.
- **Role-group pooling raised reliability without improving prediction**: the gain came from shared role means (both split halves share the same role mean), not from player-level signal.

#### Phase 7 — rating and ranking
| # | Decision | Reason | Status |
|---|---|---|---|
| P7-1 | Bowler prior = **all-delivery economy suppression** (runs vs expected, no wicket cost). Out-of-sample r for next-season run value on H: run value −0.057 [−0.164, 0.079], economy 0.033 [−0.081, 0.153], process composite −0.027 [−0.135, 0.101]; all intervals overlap, so the most reliable was chosen (economy: split-half 0.25 [0.09, 0.39], season 0.19 [0.08, 0.28]). Note: no bowler baseline predicts next-season H run value | owner's rule P7.1 | superseded by P7F-2 (bowlers: bivariate component rating; wicket component dropped per P7B-2) |
| P7-2 | Batter prior = all-delivery run value | owner's rule | superseded by P7F-1 (batters: L-only component prior) |
| P7-3 | Recency weighting tested (λ = 0.8/0.6/0.4 vs 1) and **not kept**: batters best λ 0.4 gain +0.017 [−0.033, 0.063]; bowlers best λ 0.8 gain −0.002 [−0.015, 0.010] | keep only if out-of-sample gain is clearly positive | superseded by P7R-4 (re-tested on the component prior; still not kept) |
| P7-4 | Rating = prior (a + b × shrunk baseline, a/b by weighted regression of observed H run value on the baseline) + k × (H mean − prior), k = τ²_dev / (τ²_dev + σ²_H / n_H); τ²_dev batters 0.00288, bowlers 0 (estimate −0.0017, clipped) → bowler ratings are the prior | P6B-4 | superseded by P7F-1 (batters) and P7F-2 (bowlers) |
| P7-5 | SE = bootstrap over the player's matches (200); rank score = rating − SE | brief | superseded by P7F-4 (200 → 2,000 resamples for SEs) and P7G-1 (retention gate on pooled 10,000 resamples); rank score = rating − SE unchanged |
| P7-6 | Eligibility: ≥ 2 × median H deliveries in regular player-seasons (≥10 matches): batters 108 (median 54), bowlers 154 (median 77); played in 2025 or 2026; not excluded by retention (R 95% interval entirely below 0; R unreliable, gate only). Eligible: 118 batters, 80 bowlers; retention excluded 2 batters, 6 bowlers | owner's rule P7.4 | partly superseded: thresholds (108 / 154) and 2025–26 activity rule kept; gates superseded by P7F-3 (delivery requirement) and P7G-1 (pooled retention gate) |
| P7-7 | Duplicate data names get a team tag; all other names exactly as in the data | P6 Part 1 | recorded |

#### Phase 7 review fixes (owner instruction)
| # | Decision | Reason | Status |
|---|---|---|---|
| P7R-1 | Prior uses only non-H (L) deliveries; pressure-specific variance re-estimated with that prior (batters 0.0035 [−0.0055, 0.0123] → used; bowlers −0.0022 [−0.0078, 0.0038] → 0). Supersedes P7-4's all-delivery prior | no double counting of H deliveries | final for batters (P7F-1); superseded for bowlers by P7F-2 |
| P7R-2 | Component-wise prior: runs component + wicket component (wicket cost from the Phase 2 resource table), each shrunk with its own reliability (batters c_runs 153, c_wkt 78; reliability at median L sample 0.22 / 0.36). Bowlers: the wicket component has no estimable between-player variance on L (reliability 0) → it is a constant and is dropped (coefficient 0); bowler prior = economy component on L. Out of sample vs the reviewed rating: batters +0.051 [−0.022, 0.103], bowlers −0.012 [−0.069, 0.048] → not clearly worse, component version kept on both sides. Supersedes P7-1 for the rating | owner's rule: keep unless clearly worse | final for batters (P7F-1); superseded for bowlers by P7F-2 (bivariate; wicket component dropped per P7B-2) |
| P7R-3 | Delivery condition: eligible only if own raw run value on H > 0 (in addition to volume, 2025/26 activity and the retention gate). Removed 36 batters and 38 bowlers (names in outputs/07_rating.md) | brief: "deliver when it matters"; ranking supported by data | superseded by P7B-3, then reinstated as P7F-3 (same rule, framed as a requirement) |
| P7R-4 | Recency on the component prior re-tested: not kept (batters +0.015 [−0.034, 0.056], bowlers −0.001 [−0.013, 0.014]) | P7-3 rule | recorded |
| P7R-5 | Numerical safeguard: a prior component whose between-player variance is at the estimation floor is dropped rather than entering a collinear regression (first run produced coefficients of ~10^8 for the bowler wicket component) | numerical stability; no effect on any estimable component | recorded |

#### Phase 7 second review (owner instruction)
| # | Decision | Reason | Status |
|---|---|---|---|
| P7B-1 | Bivariate, component-wise empirical Bayes replaces the L-only prior (P7R-1): H performance = core skill (all deliveries, shrunk by its between-player variance) + pressure-specific deviation ((H − L) − population shift, shrunk by var(δ)). Batters: c_runs 157, c_wkt 106; var(δ) runs 0.0048 [−0.0030, 0.0114] (used), wicket −0.0004 (→ 0). Bowlers: c_runs 318; var(δ) runs 0.0022 [−0.0043, 0.0099] (used, so bowlers now get a small H update) | uses pressure deliveries as evidence of core skill without double counting | final for bowlers (P7F-2); superseded for batters by P7F-1 (bivariate kept as a tested alternative) |
| P7B-2 | Bowler wicket component re-estimated on ALL deliveries: between-player variance −0.0004 → not estimable → dropped (stated in report) | owner's rule | recorded |
| P7B-3 | Noise-aware delivery condition replaces raw H > 0 (P7R-3): exclude only if raw H + 1.28·SE < 0 (SE = bootstrap over matches). Removed 5 batters, 9 bowlers | owner's rule | superseded by P7F-3 (noise-aware version kept as a sensitivity check) |
| P7B-4 | Tiers: new tier when the gap to the previous rank score exceeds the larger adjacent SE; order still by rank score | owner's rule | approved |
| P7B-5 | Re-validation (next-season run value on H): batters bivariate 0.103 vs previous 0.184, difference −0.082 [−0.144, +0.011]; bowlers 0.028 vs 0.019, +0.008 [−0.073, +0.080]. **Closed by P7F-1/P7F-2** | | closed |

#### Phase 7 final (owner decisions; rule: out-of-sample prediction first, then use of evidence)
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

#### LOCKED final lists (git tag `final-lists`)
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

#### Phases 8–9
| # | Decision | Reason | Status |
|---|---|---|---|
| P8-1 | Out-of-time test rebuilds the Phase 7 rating stage on 2022–24 (activity window 2023–24); per-ball values come from league-level models fitted out-of-fold over all seasons (stated as a limitation) | full model retraining per window not implemented | recorded |
| P9-1 | Runs-to-wins conversion = WP value of one run on an average H delivery (WP model): 0.0143 → ≈70 runs per win; replacement = median rating of active players with ½·min to min H balls | data-derived | recorded |
| FIX-1 | Reproducibility bug (report text only, no effect on lists or numbers): owner-approved Phase 6/6B headline text and the Phase 6 band-check section were inserted into reports by hand and lost on a clean rebuild. Now generated by code; full rebuild verified (all phase checks pass, lists byte-identical to the locked copies) | genuine bug, fixed without changing results | recorded |


### 6.3 Charts and tables

- `outputs/charts/core_vs_pressure.png`
- `outputs/charts/leverage_deciles.png`
- `outputs/charts/leverage_histogram.png`
- `outputs/charts/leverage_share_by_innings_phase.png`
- `outputs/charts/out_of_time_validation.png`
- `outputs/charts/repeatability_batters.png`
- `outputs/charts/repeatability_bowlers.png`
- `outputs/charts/repeatability_targets.png`
- `outputs/charts/role_chart_batters.png`
- `outputs/charts/role_chart_bowlers.png`
- `outputs/charts/trust_table.png`
- `outputs/charts/value_P_vs_B_batters.png`
- `outputs/charts/value_P_vs_B_bowlers.png`
- `outputs/charts/wp_calibration.png`
- `outputs/charts/wp_curve_15187.png`
- `outputs/charts/wp_curve_40725.png`
- `outputs/charts/wp_curve_45513.png`
- `outputs/charts/wp_endgame_calibration.png`

- `outputs/tables/appendix_bivariate_batters.csv`
- `outputs/tables/borderline_bat.csv`
- `outputs/tables/borderline_bowl.csv`
- `outputs/tables/checks_phase1.csv`
- `outputs/tables/checks_phase2.csv`
- `outputs/tables/checks_phase3.csv`
- `outputs/tables/checks_phase4.csv`
- `outputs/tables/checks_phase5.csv`
- `outputs/tables/checks_phase6.csv`
- `outputs/tables/checks_phase7.csv`
- `outputs/tables/checks_phase8.csv`
- `outputs/tables/checks_phase9.csv`
- `outputs/tables/face_validity.csv`
- `outputs/tables/impact_above_replacement_bat.csv`
- `outputs/tables/impact_above_replacement_bowl.csv`
- `outputs/tables/knockout_check.csv`
- `outputs/tables/leverage_close_games.csv`
- `outputs/tables/leverage_consistency.csv`
- `outputs/tables/leverage_deciles.csv`
- `outputs/tables/leverage_scenarios.csv`
- `outputs/tables/leverage_share_by_innings_phase.csv`
- `outputs/tables/leverage_top20.csv`
- `outputs/tables/locked_top10_batters.csv`
- `outputs/tables/locked_top10_bowlers.csv`
- `outputs/tables/metric_repeatability.csv`
- `outputs/tables/metric_weights.csv`
- `outputs/tables/metrics_batters.csv`
- `outputs/tables/metrics_bowlers.csv`
- `outputs/tables/out_of_time_top20_share.csv`
- `outputs/tables/out_of_time_validation.csv`
- `outputs/tables/outcome_model_metrics.csv`
- `outputs/tables/phase6_feature_z_means.csv`
- `outputs/tables/player_pressure_batters.csv`
- `outputs/tables/player_pressure_bowlers.csv`
- `outputs/tables/pressure_band_check.csv`
- `outputs/tables/pressure_band_concentration.csv`
- `outputs/tables/pressure_deviation_variance.csv`
- `outputs/tables/pressure_value_gap_bat.csv`
- `outputs/tables/pressure_value_gap_bowl.csv`
- `outputs/tables/rating_method_oos.csv`
- `outputs/tables/rating_variants.csv`
- `outputs/tables/ratings_all_bat.csv`
- `outputs/tables/ratings_all_bowl.csv`
- `outputs/tables/reconciliation.csv`
- `outputs/tables/roles_batters.csv`
- `outputs/tables/roles_bowlers.csv`
- `outputs/tables/rr_fit_best_players.csv`
- `outputs/tables/rr_fit_scenarios.csv`
- `outputs/tables/signal_recovery_summary.csv`
- `outputs/tables/team_strength_groups.csv`
- `outputs/tables/top10_batters.csv`
- `outputs/tables/top10_bowlers.csv`
- `outputs/tables/vignesh_puthur.csv`
- `outputs/tables/weights_comparison.csv`
- `outputs/tables/wp_endgame_candidates.csv`
- `outputs/tables/wp_endgame_cells.csv`
- `outputs/tables/wp_endgame_spots.csv`
- `outputs/tables/wp_last_over_check.csv`
- `outputs/tables/wp_metrics.csv`
- `outputs/tables/wp_metrics_by_group.csv`
- `outputs/tables/wp_season_forward.csv`
- `outputs/tables/wp_spot_states.csv`

### 6.4 How to reproduce

```bash
/usr/bin/python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_all.py
```
Phase reports: `outputs/01_data_audit.md` … `outputs/09_rr_outputs.md`. Checks run after every phase (`src/checks.py`).