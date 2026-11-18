# Phase 3 — Pressure: the leverage index

## Outcome model p(o | state, context)

Outcomes: 0, 1, 2, 3, 4 (incl. rare 5s), 6, wicket, wide/no-ball. Player-agnostic features: innings, balls left, wickets down, legal balls bowled, score vs venue par at this stage (1st inns), runs needed / required rate vs par (2nd inns), venue par, phase, batting hand × bowling type, pace/spin, competition strength. Scored out-of-fold (5 folds grouped by match; matches outside WP training get a seeded random fold).

| model | log_loss |
|---|---|
| Base rates (no features) | 1.6364 |
| Multinomial logistic | 1.5921 |
| Gradient boosting (HGB) | 1.5882 |

Chosen: **gradient boosting** (lower out-of-fold log-loss). Trees per fold (chosen on held-out matches): {0: 52, 1: 68, 2: 53, 3: 59, 4: 49}.

Overall calibration (out-of-fold):

| outcome | actual_rate | mean_predicted |
|---|---|---|
| 0 | 0.2858 | 0.2857 |
| 1 | 0.3709 | 0.3708 |
| 2 | 0.0627 | 0.0625 |
| 3 | 0.0033 | 0.0032 |
| 4 | 0.1174 | 0.1175 |
| 6 | 0.0635 | 0.0636 |
| wicket | 0.0539 | 0.054 |
| wide_nb | 0.0426 | 0.0425 |

## Leverage index

For each delivery, the state after each outcome s_o is built exactly (runs: score + r and one legal ball; wicket: one more wicket and one legal ball; wide/no-ball: score + 1, no legal ball). WP(s_o) comes from the Phase 2 model with the match's own out-of-fold WP models; terminal states are exact and **ties count 0.5**; the end of the 1st innings becomes 1 − WP(start of the chase).

**Deviation from the specified formula (stated plainly).** As specified, LI_raw = Σ_o p(o|s)·|WP(s_o) − WP(s)|. This assumes the WP model is internally consistent (WP(s) = E[WP(s_o)]). It is, to ~0.001, everywhere except the last ball of the 1st innings, where the 1st-innings model hands over to the chase model and the two disagree. There the specified formula counts the hand-over jump as pressure regardless of the outcome. The LI used here measures each outcome's swing around the expected next WP instead: **LI_raw = Σ_o p(o|s)·|WP(s_o) − Σ_o' p(o'|s)·WP(s_o')|**. Where the model is consistent the two are the same; the specified version is kept as `LI_spec`.

| group | deliveries | mean_abs_gap | mean_LI_as_specified | mean_LI_used |
|---|---|---|---|---|
| full matches: 2nd innings | 92813 | 0.0014 | 1.13919997215271 | 1.1516000032424927 |
| full matches: last ball of 1st innings | 781 | 0.0573 | 4.38670015335083 | 1.2839000225067139 |
| full matches: other 1st-innings balls | 100212 | 0.001 | 0.8446999788284302 | 0.8574000000953674 |
| rain-shortened matches | 5401 | 0.0026 | 1.1902999877929688 | 1.1431000232696533 |

Agreement on included deliveries: {'spearman_LI_vs_specified': 0.9958, 'H25_overlap': 0.9758}.

**Rain-shortened matches excluded from pressure sets** (32 matches, 5,401 deliveries: recorded or detected revisions, or innings shorter than 20 overs). The WP model never trained on shortened innings and extrapolates badly there (four of the original top-20 deliveries were last balls of 5–12-over innings). Switch: `EXCLUDE_SHORTENED_FROM_PRESSURE` in `src/config.py`.

**LI = LI_raw / mean LI_raw** over the 193,806 included deliveries. H25 global cut-off: LI ≥ 1.163.

Histogram: `outputs/charts/leverage_histogram.png`.

## Check 1 — where high leverage comes from

**Decision (after review): the primary high-leverage set H is the top 25% of LI within each innings** (`H25_within`); top 20% and 33% within innings are kept for robustness. With a global cut-off, 83% of H25 came from the 2nd innings and 1st-innings middle overs contributed none (share of H by innings, global: {'H20': {1: 0.063, 2: 0.937}, 'H25': {1: 0.166, 2: 0.834}, 'H33': {1: 0.314, 2: 0.686}}). A global cut-off would leave players who mostly bat or bowl first with almost no pressure sample. Baseline L = included deliveries not in H. Rain-shortened matches are in neither H nor L; they remain in the data for conventional statistics (Phase 9).

**1st-innings limitation.** The 1st-innings WP model gives a run the same WP value at every stage. Tested before Phase 4: interactions letting (score − par) and wickets in hand vary with balls left — (score − par)·√((120+k)/(balls left+k)) for k = 6/12/24/48, (score − par) × balls-left hinges at 12/30/60/90, wickets × balls-left bases — with sign constraints. Out-of-fold 1st-innings log-loss: current 0.6242; best variant 0.6240 (wickets bases; per-fold changes mixed, three folds worse). Unconstrained, the stage interaction is slightly *negative*: a run late in the innings is not worth more once par is accounted for (an above-par score early also signals conditions and batting strength). Current model kept. Consequence: 1st-innings leverage comes mainly from wicket risk, so 1st-innings middle overs rarely reach H even within innings.

| innings | phase | deliveries | mean_LI | median_LI | in_H25_global | in_H25_within | share_of_all | share_of_H25_global | share_of_H25_within | pct_of_group_in_H25_global |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | powerplay | 30391 | 1.0260000228881836 | 1.031999945640564 | 4876 | 16635 | 0.157 | 0.101 | 0.343 | 0.16 |
| 1 | middle | 45293 | 0.7390000224113464 | 0.7630000114440918 | 0 | 861 | 0.234 | 0.0 | 0.018 | 0.0 |
| 1 | death | 25309 | 0.8799999952316284 | 0.9129999876022339 | 3185 | 7753 | 0.131 | 0.066 | 0.16 | 0.126 |
| 2 | powerplay | 30407 | 1.1050000190734863 | 1.1950000524520874 | 15764 | 6922 | 0.157 | 0.325 | 0.143 | 0.518 |
| 2 | middle | 44030 | 0.9419999718666077 | 0.8529999852180481 | 17088 | 9599 | 0.227 | 0.353 | 0.198 | 0.388 |
| 2 | death | 18376 | 1.7309999465942383 | 0.6159999966621399 | 7539 | 6683 | 0.095 | 0.156 | 0.138 | 0.41 |

`outputs/charts/leverage_share_by_innings_phase.png`

### 1st-innings death overs in close games

Close game = tie, won by ≤10 runs, or chase completed in the final over.

| group | deliveries | matches | mean_LI | median_LI | p90_LI | pct_in_H25_global | pct_in_H25_within |
|---|---|---|---|---|---|---|---|
| 1st inns death, close game | 7965 | 250 | 0.9419999718666077 | 0.9549999833106995 | 1.235 | 0.156 | 0.366 |
| 1st inns death, other games | 17344 | 555 | 0.8519999980926514 | 0.890999972820282 | 1.182 | 0.112 | 0.279 |
| 2nd inns death, close game | 7477 | 250 | 3.2219998836517334 | 2.5820000171661377 | 6.589 | 0.716 | 0.657 |
| 2nd inns death, other games | 10899 | 483 | 0.7070000171661377 | 0.10999999940395355 | 2.429 | 0.2 | 0.162 |
| other deliveries | 150121 | 811 | 0.9309999942779541 | 0.8920000195503235 | 1.601 | 0.251 | 0.227 |

## Check 2 — the 20 highest-leverage deliveries

| match | innings | batting | over.ball | situation | striker_name | bowler_name | wp_before | LI | what_happened | wp_after |
|---|---|---|---|---|---|---|---|---|---|---|
| New Zealand v Australia, 2024-02-21 | 2 | Australia | 20.7 | need 4 off 1, 6 wkts in hand | Tim David | Tim Southee | 0.453000009059906 | 33.83000183105469 | 4 run(s) | 1.0 |
| Rajasthan Royals v Lucknow Super Giants, 2025-04-19 | 2 | Rajasthan Royals | 20.6 | need 4 off 1, 5 wkts in hand | Shubham Dubey | Avesh Khan | 0.4580000042915344 | 33.810001373291016 | 1 run(s) | 0.0 |
| Gujarat Titans v Sunrisers Hyderabad, 2022-04-27 | 2 | Gujarat Titans | 20.6 | need 3 off 1, 5 wkts in hand | Rashid Khan | Marco Jansen | 0.671999990940094 | 32.290000915527344 | 6 run(s) | 1.0 |
| Royal Challengers Bengaluru v Chennai Super Kings, 2025-05-03 | 2 | Chennai Super Kings | 20.7 | need 4 off 1, 5 wkts in hand | Shivam Dube | Yash Dayal | 0.48399999737739563 | 32.279998779296875 | 1 run(s) | 0.0 |
| Kolkata Knight Riders v Rajasthan Royals, 2025-05-04 | 2 | Rajasthan Royals | 20.6 | need 3 off 1, 3 wkts in hand | Shubham Dubey | Vaibhav Arora | 0.6050000190734863 | 31.139999389648438 | WICKET (RunOut) | 0.0 |
| India v Afghanistan, 2024-01-17 | 2 | Afghanistan | 20.8 | need 3 off 1, 4 wkts in hand | Gulbadin Naib | Mukesh Kumar | 0.527999997138977 | 30.600000381469727 | 2 run(s) | 0.5 |
| Kolkata Knight Riders v Lucknow Super Giants, 2022-05-18 | 2 | Kolkata Knight Riders | 20.6 | need 3 off 1, 3 wkts in hand | Umesh Yadav | Marcus Stoinis | 0.6230000257492065 | 30.299999237060547 | WICKET (Bowled) | 0.0 |
| India v Sri Lanka, 2025-09-26 | 2 | Sri Lanka | 20.6 | need 3 off 1, 5 wkts in hand | Dasun Shanaka | Harshit Rana | 0.5460000038146973 | 30.18000030517578 | 2 run(s) | 0.5 |
| Chennai Super Kings v Punjab Kings, 2023-04-30 | 2 | Punjab Kings | 20.6 | need 3 off 1, 4 wkts in hand | Sikandar Raza | Matheesha Pathirana | 0.4020000100135803 | 29.34000015258789 | 3 run(s) | 1.0 |
| Sri Lanka v Zimbabwe, 2024-01-14 | 2 | Sri Lanka | 20.6 | need 2 off 1, 3 wkts in hand | Dushmantha Chameera | Blessing Muzarabani | 0.7699999809265137 | 28.8799991607666 | 2 run(s) | 1.0 |
| West Indies v Pakistan, 2025-08-02 | 2 | West Indies | 20.7 | need 3 off 1, 2 wkts in hand | Jason Holder | Shaheen Afridi | 0.5080000162124634 | 28.729999542236328 | 4 run(s) | 1.0 |
| Rajasthan Royals v Sunrisers Hyderabad, 2023-05-07 | 2 | Sunrisers Hyderabad | 20.7 | need 4 off 1, 4 wkts in hand | Abdul Samad | Sandeep Sharma | 0.23999999463558197 | 28.719999313354492 | 6 run(s) | 1.0 |
| Kolkata Knight Riders v Royal Challengers Bengaluru, 2024-04-21 | 2 | Royal Challengers Bengaluru | 20.6 | need 3 off 1, 1 wkts in hand | Lockie Ferguson | Mitchell Starc | 0.5410000085830688 | 28.520000457763672 | WICKET (RunOut) | 0.0 |
| Rajasthan Royals v Gujarat Titans, 2024-04-10 | 2 | Gujarat Titans | 20.6 | need 2 off 1, 3 wkts in hand | Rashid Khan | Avesh Khan | 0.7429999709129333 | 28.040000915527344 | 4 run(s) | 1.0 |
| Mumbai Indians v Chennai Super Kings, 2022-04-21 | 2 | Chennai Super Kings | 20.6 | need 4 off 1, 3 wkts in hand | Mahendra Singh Dhoni | Jaydev Unadkat | 0.3409999907016754 | 27.860000610351562 | 4 run(s) | 1.0 |
| Sunrisers Hyderabad v Rajasthan Royals, 2024-05-02 | 2 | Rajasthan Royals | 20.6 | need 2 off 1, 4 wkts in hand | Rovman Powell | Bhuvneshwar Kumar | 0.8510000109672546 | 27.700000762939453 | WICKET (LBW) | 0.0 |
| India v Australia, 2023-11-28 | 2 | Australia | 20.6 | need 2 off 1, 5 wkts in hand | Glenn Maxwell | Prasidh Krishna | 0.6380000114440918 | 27.540000915527344 | 4 run(s) | 1.0 |
| Bangladesh A v Pakistan A, 2025-11-23 | 2 | Bangladesh A | 20.6 | need 2 off 1, 1 wkts in hand | Abdul Saqlain | Ahmed Daniyal | 0.3959999978542328 | 27.540000915527344 | 1 run(s) | 0.5 |
| Delhi Capitals v Gujarat Titans, 2026-04-08 | 2 | Delhi Capitals | 20.6 | need 2 off 1, 3 wkts in hand | David Miller | Prasidh Krishna | 0.75 | 27.469999313354492 | WICKET (RunOut) | 0.0 |
| Sri Lanka v India, 2024-07-30 | 2 | Sri Lanka | 20.6 | need 3 off 1, 2 wkts in hand | Chamindu Wickramasinghe | Suryakumar Yadav | 0.47600001096725464 | 27.420000076293945 | 2 run(s) | 0.5 |

## Validation — outcome rates across LI deciles (deciles within each innings)

`outputs/charts/leverage_deciles.png`

| innings | decile | deliveries | mean_LI | dismissal_rate | wide_nb_rate | dot_rate | boundary_rate |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 10100 | 0.41690000891685486 | 0.0526 | 0.0444 | 0.307 | 0.1608 |
| 1 | 2 | 10099 | 0.6256999969482422 | 0.0511 | 0.0362 | 0.2808 | 0.1543 |
| 1 | 3 | 10099 | 0.7214000225067139 | 0.0501 | 0.0337 | 0.2695 | 0.1412 |
| 1 | 4 | 10099 | 0.7857999801635742 | 0.045 | 0.0352 | 0.2617 | 0.1568 |
| 1 | 5 | 10100 | 0.8445000052452087 | 0.0479 | 0.0414 | 0.2746 | 0.1679 |
| 1 | 6 | 10099 | 0.9013000130653381 | 0.0487 | 0.0431 | 0.2942 | 0.1851 |
| 1 | 7 | 10099 | 0.9575999975204468 | 0.0481 | 0.0489 | 0.3215 | 0.1905 |
| 1 | 8 | 10099 | 1.016800045967102 | 0.0556 | 0.0478 | 0.3217 | 0.2095 |
| 1 | 9 | 10099 | 1.09089994430542 | 0.0572 | 0.0481 | 0.3213 | 0.2322 |
| 1 | 10 | 10100 | 1.2460999488830566 | 0.0733 | 0.0513 | 0.272 | 0.2687 |
| 2 | 1 | 9282 | 0.016899999231100082 | 0.0726 | 0.0488 | 0.3033 | 0.1907 |
| 2 | 2 | 9281 | 0.12890000641345978 | 0.0558 | 0.038 | 0.2976 | 0.1831 |
| 2 | 3 | 9281 | 0.32659998536109924 | 0.0528 | 0.039 | 0.3007 | 0.1791 |
| 2 | 4 | 9281 | 0.5746999979019165 | 0.0492 | 0.0404 | 0.3161 | 0.1751 |
| 2 | 5 | 9282 | 0.8485999703407288 | 0.0436 | 0.0367 | 0.3193 | 0.1824 |
| 2 | 6 | 9281 | 1.1226999759674072 | 0.0477 | 0.0384 | 0.3478 | 0.1768 |
| 2 | 7 | 9281 | 1.3645000457763672 | 0.044 | 0.0379 | 0.3406 | 0.1786 |
| 2 | 8 | 9281 | 1.5667999982833862 | 0.0509 | 0.0356 | 0.3244 | 0.1841 |
| 2 | 9 | 9281 | 1.8217999935150146 | 0.05 | 0.0404 | 0.2908 | 0.2035 |
| 2 | 10 | 9282 | 3.7444000244140625 | 0.0789 | 0.0642 | 0.2137 | 0.2052 |

## Validation — six pressure scenarios

Definitions: post-wicket = the 12 deliveries after a wicket in the same innings; entry under pressure = a batter's first 10 balls faced when his first ball faced was in H25 (within innings); death chase = 2nd-innings overs 16–20 (batting view); death defence = the same deliveries from the bowling side (identical set); dot squeeze = deliveries after 3+ consecutive dot balls by the batting side; powerplay vs set openers = powerplay, no wicket down, striker has already faced ≥10 balls.

| scenario | deliveries | share_of_all | share_of_H25_global | share_of_H25_within | pct_of_scenario_in_H25_within | enrichment (H25 within share / overall share) | mean_LI |
|---|---|---|---|---|---|---|---|
| post_wicket | 86223 | 0.445 | 0.414 | 0.4 | 0.225 | 0.9 | 0.994 |
| entry_under_pressure | 15551 | 0.08 | 0.2 | 0.255 | 0.795 | 3.19 | 1.894 |
| death_chase | 18376 | 0.095 | 0.156 | 0.138 | 0.364 | 1.45 | 1.731 |
| death_defence | 18376 | 0.095 | 0.156 | 0.138 | 0.364 | 1.45 | 1.731 |
| dot_squeeze | 5891 | 0.03 | 0.029 | 0.03 | 0.246 | 1.0 | 0.922 |
| pp_vs_set_openers | 7981 | 0.041 | 0.075 | 0.097 | 0.587 | 2.37 | 1.152 |