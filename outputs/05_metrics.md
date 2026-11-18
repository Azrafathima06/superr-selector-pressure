# Phase 5 — Pressure metrics

Every metric is a mean of per-delivery components **actual − expected**, signed so **positive = good**, computed on H (`H25_within`: top 25% of LI within each innings) and on L (the rest). Expected values come from the Phase 4 opposition-aware outcome models (player-agnostic context model + opposition rating, out-of-fold by match): batters use the model with the bowler's rating, bowlers the model with the batter's rating. Rain-shortened matches are in neither H nor L. No one is ranked in this phase.

## Definitions

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

## Retention, centred

R = (P − B) − population (P − B), delivery-weighted. **Finding — population-wide shift:** batters P_pop = -0.0085, B_pop = -0.0274 (P − B = 0.0189); bowlers P_pop = -0.023, B_pop = -0.0048 (P − B = -0.0182). These shifts are also the size of the expectation models' calibration error on H vs L, so the centring removes it. `R_raw` is kept.

## Summary per metric (career values; eligible = enough H deliveries for that metric)

`players` = with at least `min_n_H` deliveries on H; `active` = also played in 2025 or 2026. pop_H / pop_L = delivery-weighted population mean of the component.

| side | metric | kind | min_n_H | players | active_2025_26 | H_median | H_p5 | H_p95 | H_min | H_max | L_median | pop_H | pop_L |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bat | pressure_v | value | 100 | 136 | 125 | 0.03009999915957451 | -0.1643 | 0.2095 | -0.5927000045776367 | 0.32199999690055847 | 0.003599999938160181 | -0.0085 | -0.0274 |
| bat | sr_delta | skill | 100 | 136 | 125 | 6.91480016708374 | -24.3575 | 35.1429 | -41.41170120239258 | 53.81019973754883 | 0.6062999963760376 | 2.5346 | -0.874 |
| bat | dot_suppression | skill | 100 | 136 | 125 | 0.010599999688565731 | -0.0617 | 0.0665 | -0.12950000166893005 | 0.11209999769926071 | 0.0031999999191612005 | 0.0084 | -0.0027 |
| bat | boundary_elevation | skill | 100 | 136 | 125 | 0.010300000198185444 | -0.0513 | 0.0767 | -0.08799999952316284 | 0.12099999934434891 | 0.0013000000035390258 | 0.0018 | -0.0037 |
| bat | wicket_security | skill | 100 | 136 | 125 | 0.0024999999441206455 | -0.0277 | 0.0305 | -0.0649000033736229 | 0.0568000003695488 | 0.003599999938160181 | -0.0002 | 0.0003 |
| bat | squeeze_response | temperament | 30 | 3 | 3 | 0.16979999840259552 | -0.0667 | 0.5761 | -0.09300000220537186 | 0.6212000250816345 | 0.1362999975681305 | -0.0121 | -0.0767 |
| bat | entry_resilience | temperament | 30 | 135 | 124 | -0.053199999034404755 | -0.4945 | 0.224 | -0.8170999884605408 | 0.44040000438690186 | 0.07240000367164612 | -0.1126 | 0.0044 |
| bat | collapse_resistance | temperament | 30 | 107 | 100 | 0.10379999876022339 | -0.3647 | 0.4581 | -0.65420001745224 | 0.6560999751091003 | 0.025499999523162842 | 0.0353 | -0.0407 |
| bat | post_wicket_response | temperament | 30 | 184 | 168 | -0.03759999945759773 | -0.4721 | 0.2262 | -0.9221000075340271 | 0.4496000111103058 | -0.04960000142455101 | -0.0639 | -0.0836 |
| bat | runout_avoidance | temperament | 30 | 391 | 306 | 0.0007999999797903001 | -0.0202 | 0.0038 | -0.06040000170469284 | 0.0044999998062849045 | 0.0007999999797903001 | -0.0008 | 0.0003 |
| bat | attacking_stroke_share | profile | 100 | 134 | 123 | 0.6679999828338623 | 0.5365 | 0.7731 | 0.4724000096321106 | 0.8299999833106995 | 0.6262999773025513 | 0.6638 | 0.6246 |
| bat | squeeze_response_pooled (H+L) | temperament | 30 | 46 | 42 | 0.00430000014603138 | -0.5278 | 0.4472 | -0.7789999842643738 | 0.5846999883651733 |  |  |  |
| bat | partner_effect_H (ON TRIAL) | temperament | 100 | 135 | 125 | 0.024 | -0.2294 | 0.2829 | -0.3154 | 0.3311 |  |  |  |
| bowl | pressure_v | value | 100 | 148 | 135 | -0.027799999341368675 | -0.2047 | 0.1773 | -0.27489998936653137 | 0.3147999942302704 | 0.003599999938160181 | -0.023 | -0.0048 |
| bowl | economy_suppression | skill | 100 | 148 | 135 | -0.1363999992609024 | -1.5263 | 1.3511 | -2.3118999004364014 | 1.761299967765808 | 0.18000000715255737 | -0.1454 | 0.0439 |
| bowl | dot_rate | skill | 100 | 141 | 129 | -0.006800000090152025 | -0.0582 | 0.0599 | -0.10279999673366547 | 0.11060000211000443 | 0.007499999832361937 | -0.0077 | 0.0026 |
| bowl | wicket_taking | skill | 100 | 141 | 129 | -0.0007999999797903001 | -0.0258 | 0.0313 | -0.05400000140070915 | 0.04560000076889992 | -0.0006000000284984708 | -0.0001 | -0.0004 |
| bowl | boundary_suppression | skill | 100 | 141 | 129 | 0.0005000000237487257 | -0.0523 | 0.047 | -0.08500000089406967 | 0.06989999860525131 | 0.00839999970048666 | -0.0018 | 0.0034 |
| bowl | bounce_back | temperament | 30 | 94 | 89 | 0.011599999852478504 | -0.4163 | 0.3369 | -0.632099986076355 | 0.44839999079704285 | 0.010900000110268593 | -0.031 | -0.0397 |
| bowl | post_wicket_response | temperament | 30 | 182 | 164 | 0.019300000742077827 | -0.3078 | 0.4097 | -0.5368000268936157 | 1.0520999431610107 | 0.041099999099969864 | 0.0283 | 0.0493 |
| bowl | execution_error | temperament | 30 | 272 | 223 | -0.003700000001117587 | -0.0899 | 0.059 | -0.20350000262260437 | 0.0828000009059906 | 0.006899999920278788 | -0.011 | 0.0037 |
| bowl | yorker_execution | report_only | 20 | 27 | 27 | 0.27079999446868896 | 0.0087 | 0.5661 | -0.09059999883174896 | 0.9035000205039978 | 0.3091000020503998 | 0.2807 | 0.3316 |
| bowl | variation_diff (entropy H − L, bits) | temperament | 60 | 203 | 181 | 0.0257 | -0.3667 | 0.3392 | -0.8915 | 0.6729 |  |  |  |
| bowl | execution_drift (H − L of execution_error) | temperament | 100 | 148 | 135 | -0.012500000186264515 | -0.0581 | 0.0275 | -0.10080000013113022 | 0.04989999905228615 |  |  |  |

## P vs B and reliability of P and B

- **Batters**: {'players (n_H≥100, n_L≥100)': 136, 'r_raw': np.float64(0.168), 'r_true_moments (sampling variance removed)': np.float64(0.671), 'r_disattenuated_by_split_half': np.float64(0.453), 'split_half_P': {'players': 124, 'half_r': np.float64(0.199), 'spearman_brown': np.float64(0.332)}, 'split_half_B': {'players': 230, 'half_r': np.float64(0.26), 'spearman_brown': np.float64(0.413)}, 'P_noise_share': 0.859, 'B_noise_share': 0.555}
- **Bowlers**: {'players (n_H≥100, n_L≥100)': 147, 'r_raw': np.float64(0.017), 'r_true_moments (sampling variance removed)': 'P variance ≤ noise', 'r_disattenuated_by_split_half': 'split-half reliability ≤ 0', 'split_half_P': {'players': 125, 'half_r': np.float64(-0.067), 'spearman_brown': np.float64(-0.143)}, 'split_half_B': {'players': 275, 'half_r': np.float64(0.173), 'spearman_brown': np.float64(0.294)}, 'P_noise_share': 1.014, 'B_noise_share': 0.677}

noise share = mean sampling variance / observed between-player variance (1 = all noise).

## Population means per component (sanity)

| side | metric | deliveries_H | deliveries_L | pop_mean_H | pop_mean_L | sd_per_delivery |
|---|---|---|---|---|---|---|
| bat | pressure_v | 46340 | 140037 | -0.0085 | -0.0274 | 1.764 |
| bat | sr_delta | 46070 | 139500 | 2.5346 | -0.874 | 171.872 |
| bat | dot_suppression | 46070 | 139500 | 0.0084 | -0.0027 | 0.447 |
| bat | boundary_elevation | 46070 | 139500 | 0.0018 | -0.0037 | 0.385 |
| bat | wicket_security | 46070 | 139500 | -0.0002 | 0.0003 | 0.223 |
| bat | squeeze_response | 1403 | 4324 | -0.0121 | -0.0767 | 1.834 |
| bat | entry_resilience | 12366 | 3185 | -0.1126 | 0.0044 | 1.578 |
| bat | collapse_resistance | 9615 | 50714 | 0.0353 | -0.0407 | 1.731 |
| bat | post_wicket_response | 18496 | 64499 | -0.0639 | -0.0836 | 1.648 |
| bat | runout_avoidance | 96906 | 290706 | -0.0008 | 0.0003 | 0.039 |
| bat | attacking_stroke_share | 45544 | 137570 | 0.6638 | 0.6246 | 0.482 |
| bowl | pressure_v | 48453 | 145353 | -0.023 | -0.0048 | 1.743 |
| bowl | economy_suppression | 48453 | 145353 | -0.1454 | 0.0439 | 10.169 |
| bowl | dot_rate | 46070 | 139500 | -0.0077 | 0.0026 | 0.447 |
| bowl | wicket_taking | 46070 | 139500 | -0.0001 | -0.0004 | 0.223 |
| bowl | boundary_suppression | 46070 | 139500 | -0.0018 | 0.0034 | 0.384 |
| bowl | bounce_back | 9681 | 22828 | -0.031 | -0.0397 | 1.782 |
| bowl | post_wicket_response | 19369 | 66854 | 0.0283 | 0.0493 | 1.632 |
| bowl | execution_error | 48453 | 145353 | -0.011 | 0.0037 | 0.273 |
| bowl | yorker_execution | 1951 | 3494 | 0.2807 | 0.3316 | 1.191 |

## Caveats found in this phase

1. **Yorker labels depend on the outcome.** `delivery_style` is recorded after the ball: a missed yorker tends to be logged as a full toss. Yorkers average 0.84 runs per ball against 1.87 for full tosses, and population mean v on yorkers is strongly positive on both H and L. Yorker execution therefore partly measures the label, not the skill, and most bowlers have under 10 yorkers on H. Recommend: report only, not for ranking.
2. **Squeeze response is too sparse on H.** The dot-squeeze trigger covers 3% of deliveries, and only a handful of batters reach 30 trigger deliveries on H. Recommend: use H + L trigger deliveries pooled, or drop.
3. **Entry resilience carries a new-batter effect.** Population mean v on entry deliveries is clearly negative, because the outcome model does not know the batter has just arrived. It is the same for everyone, so comparisons between players hold, but the level is not 'underperformance under pressure'.
4. **Skill components are not centred on H.** The outcome models are calibrated overall, but on H batting sides beat expectation slightly (e.g. strike-rate delta population mean > 0 on H, < 0 on L). Phase 7 standardises each metric across players, which removes this level shift.
5. **Bowlers' pressure level P shows no split-half reliability**, and its sampling noise is as large as the spread between bowlers. Batters' P is weakly reliable. Phase 6 will test every metric; process metrics are expected to do better than WP value.

## Tables

`outputs/tables/metrics_batters.csv`, `outputs/tables/metrics_bowlers.csv`: one row per player × season plus a `career` row; for every metric `<metric>_H`, `<metric>_L`, `<metric>_n_H`, `<metric>_n_L`, `<metric>_matches_H`, `<metric>_matches_L`; P, B, R (centred), R_raw. Per-delivery components for Phase 6: `data_processed/metric_components.parquet`.