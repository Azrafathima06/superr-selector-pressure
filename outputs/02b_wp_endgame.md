# Phase 2b — Endgame study (2nd innings, ≤30 legal balls left)

All candidates are complete 2nd-innings WP models scored **out-of-fold** with the same match folds as Phase 2 (every delivery scored by models and Markov tables built without its match). Selection rule, fixed before looking at results: lowest out-of-fold log-loss on deliveries with ≤30 balls left; Brier also reported.

## Candidates

- **current** — Phase 2 model: main sign-constrained logistic + endgame logistic (fit on ≤18 balls), blended 24→12.
- **main_only** — main logistic without the endgame model (reference).
- **markov** — Markov chain: ball outcomes 0/1/2/3/4/6/wicket/wide-or-no-ball with probabilities by over (i.e. balls left; the phase is implied by the over), wickets lost and competition strength. Strength tables are shrunk toward the pooled over × wickets table with pseudo-count α_s = 3000 chosen by grouped-CV outcome log-loss {30: np.float64(1.60779), 100: np.float64(1.60345), 300: np.float64(1.60099), 1000: np.float64(1.60003), 3000: np.float64(1.60001)}. Wides/no-balls add a run without using a ball; level scores at the end = tie = 0.5. Exact chase probability by backward induction. Used for ≤30 balls left; main model above.
- **hybrid** — Markov for the last 12 balls, linear blend from 24 to 12 with the main model.
- **spline** — one sign-constrained logistic for the whole chase with a richer form: log(1+runs needed), min(runs needed, k) at knots [30.0, 60.0, 101.0, 143.0], log(1+balls left), hinge bases max(0, log RRR − k) at knots [0.3921, 1.7332, 2.0354, 2.2648, 2.4849, 2.7664] (quantiles of the feature, no outcomes used), each hinge × wickets lost/10, wickets × min(balls left, K)/K for K in 3, 6, 12, 24, 48, 120, required rate vs venue par, competition dummies. Every basis is monotone in the state and its sign is enforced, so monotonic sense holds by construction.
- **spline_markov** — spline + logit of the strength-aware Markov chain + expected-runs z-score (both signs +).

## All candidates (out-of-fold)

| candidate | log_loss_le30 | brier_le30 | log_loss_le12 | log_loss_full_2nd_inns | brier_full_2nd_inns | cells_mean_abs_error (weighted by matches) | cells outside actual 95% CI (of 59) | violations: +1 wicket in hand | violations: +1 ball left | violations: +1 run scored |
|---|---|---|---|---|---|---|---|---|---|---|
| Current (logistic + endgame logistic) | 0.2347 | 0.0724 | 0.2005 | 0.3844 | 0.1244 | 0.031 | 6 | 1 (max drop 0.000) | 125 (max drop 0.000) | 0 (max drop 0.000) |
| Main logistic only | 0.2354 | 0.0727 | 0.2042 | 0.3845 | 0.1244 | 0.035 | 8 | 0 (max drop 0.000) | 0 (max drop 0.000) | 0 (max drop 0.000) |
| Spline logistic + Markov features | 0.2379 | 0.0727 | 0.2131 | 0.3862 | 0.125 | 0.034 | 10 | 0 (max drop 0.000) | 0 (max drop 0.000) | 0 (max drop 0.000) |
| Hybrid (Markov ≤12, blend 24→12) | 0.242 | 0.0741 | 0.2216 | 0.3858 | 0.1247 | 0.047 | 14 | 0 (max drop 0.000) | 266 (max drop 0.002) | 0 (max drop 0.000) |
| Spline logistic | 0.244 | 0.0736 | 0.229 | 0.3874 | 0.1251 | 0.037 | 15 | 0 (max drop 0.000) | 0 (max drop 0.000) | 0 (max drop 0.000) |
| Markov chain (≤30 balls) | 0.2592 | 0.0785 | 0.2216 | 0.3892 | 0.1256 | 0.065 | 24 | 0 (max drop 0.000) | 176 (max drop 0.321) | 0 (max drop 0.000) |

**Chosen: Current (logistic + endgame logistic)** (lowest log-loss on ≤30 balls left). It is now the production 2nd-innings WP; all WP values were re-scored.

Monotonicity violations are counted on 15,000 sampled endgame states with fold −1 models; a violation is a WP drop > 0.000001 when the state improves by one wicket in hand, one ball left or one run.

## Final production model, full innings (out-of-fold)

| innings | n | log_loss | brier | auc |
|---|---|---|---|---|
| 1 | 99603 | 0.6242 | 0.2178 | 0.7048 |
| 2 | 91459 | 0.3844 | 0.1244 | 0.906 |

## Phase 2 comparison table (full innings, from Phase 2)

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

## Spot states (model: IPL chase of 175 at par 175; empirical: all kept matches, one state per match, same balls left, runs needed within ±0 / ±1 / ±1 / ±2, wickets in hand in the band; 95% Wilson interval)

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

## Pooled endgame calibration

`outputs/charts/wp_endgame_calibration.png` — one panel per candidate; each point is a cell (balls left × required-rate band × wickets in hand) with ≥10 matches; vertical bars = 95% Wilson interval of the actual win rate using the number of matches (deliveries within one match share one result, so they are not independent). Full cell table: `outputs/tables/wp_endgame_cells.csv` (all cells, including those with <10 matches).

| balls_left | rrr_band | wkts_in_hand | matches | actual | ci_lo | ci_hi | current | main_only | markov | hybrid | spline | spline_markov |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1–6 | <6 | 1–3 | 22 | 0.923 | 0.74 | 0.981 | 0.922 | 0.98 | 0.914 | 0.914 | 0.993 | 0.977 |
| 1–6 | <6 | 4–6 | 59 | 1.0 | 0.939 | 1.0 | 0.98 | 0.991 | 0.958 | 0.958 | 0.998 | 0.993 |
| 1–6 | <6 | 7–10 | 17 | 1.0 | 0.816 | 1.0 | 0.995 | 0.997 | 0.963 | 0.963 | 0.999 | 0.998 |
| 1–6 | 6–9 | 1–3 | 25 | 0.93 | 0.764 | 0.982 | 0.77 | 0.901 | 0.725 | 0.725 | 0.953 | 0.894 |
| 1–6 | 6–9 | 4–6 | 47 | 0.989 | 0.905 | 0.999 | 0.915 | 0.95 | 0.787 | 0.787 | 0.983 | 0.96 |
| 1–6 | 9–12 | 1–3 | 29 | 0.442 | 0.279 | 0.619 | 0.524 | 0.662 | 0.517 | 0.517 | 0.72 | 0.648 |
| 1–6 | 9–12 | 4–6 | 36 | 0.741 | 0.58 | 0.856 | 0.819 | 0.863 | 0.616 | 0.616 | 0.923 | 0.884 |
| 1–6 | 12–15 | 1–3 | 36 | 0.328 | 0.198 | 0.491 | 0.375 | 0.464 | 0.355 | 0.355 | 0.451 | 0.43 |
| 1–6 | 12–15 | 4–6 | 31 | 0.447 | 0.287 | 0.618 | 0.648 | 0.69 | 0.437 | 0.437 | 0.73 | 0.703 |
| 1–6 | 15–18 | 1–3 | 32 | 0.158 | 0.07 | 0.319 | 0.236 | 0.258 | 0.215 | 0.215 | 0.205 | 0.229 |
| 1–6 | 15–18 | 4–6 | 26 | 0.389 | 0.228 | 0.579 | 0.454 | 0.402 | 0.246 | 0.246 | 0.359 | 0.386 |
| 1–6 | 18+ | 1–3 | 232 | 0.016 | 0.006 | 0.041 | 0.019 | 0.015 | 0.015 | 0.015 | 0.007 | 0.009 |
| 1–6 | 18+ | 4–6 | 95 | 0.075 | 0.037 | 0.146 | 0.065 | 0.042 | 0.033 | 0.033 | 0.023 | 0.033 |
| 7–12 | <6 | 1–3 | 15 | 0.755 | 0.502 | 0.904 | 0.916 | 0.965 | 0.914 | 0.914 | 0.984 | 0.965 |
| 7–12 | <6 | 4–6 | 97 | 1.0 | 0.962 | 1.0 | 0.987 | 0.99 | 0.981 | 0.981 | 0.996 | 0.992 |
| 7–12 | <6 | 7–10 | 46 | 1.0 | 0.923 | 1.0 | 0.997 | 0.997 | 0.986 | 0.986 | 0.999 | 0.997 |
| 7–12 | 6–9 | 1–3 | 25 | 0.62 | 0.426 | 0.782 | 0.628 | 0.758 | 0.679 | 0.679 | 0.846 | 0.766 |
| 7–12 | 6–9 | 4–6 | 69 | 0.975 | 0.907 | 0.994 | 0.892 | 0.913 | 0.831 | 0.831 | 0.953 | 0.923 |
| 7–12 | 6–9 | 7–10 | 24 | 0.976 | 0.823 | 0.997 | 0.979 | 0.981 | 0.859 | 0.859 | 0.992 | 0.984 |
| 7–12 | 9–12 | 1–3 | 34 | 0.304 | 0.176 | 0.472 | 0.387 | 0.493 | 0.425 | 0.425 | 0.545 | 0.501 |
| 7–12 | 9–12 | 4–6 | 62 | 0.751 | 0.631 | 0.842 | 0.768 | 0.779 | 0.564 | 0.564 | 0.819 | 0.795 |
| 7–12 | 9–12 | 7–10 | 11 | 0.895 | 0.606 | 0.979 | 0.93 | 0.922 | 0.614 | 0.614 | 0.947 | 0.927 |
| 7–12 | 12–15 | 1–3 | 43 | 0.147 | 0.071 | 0.281 | 0.161 | 0.194 | 0.182 | 0.182 | 0.176 | 0.189 |
| 7–12 | 12–15 | 4–6 | 52 | 0.397 | 0.276 | 0.533 | 0.479 | 0.458 | 0.27 | 0.27 | 0.438 | 0.458 |
| 7–12 | 15–18 | 1–3 | 50 | 0.066 | 0.024 | 0.17 | 0.07 | 0.076 | 0.061 | 0.061 | 0.051 | 0.068 |
| 7–12 | 15–18 | 4–6 | 58 | 0.29 | 0.189 | 0.417 | 0.27 | 0.229 | 0.107 | 0.107 | 0.175 | 0.214 |
| 7–12 | 18+ | 1–3 | 192 | 0.011 | 0.003 | 0.038 | 0.007 | 0.007 | 0.004 | 0.004 | 0.003 | 0.005 |
| 7–12 | 18+ | 4–6 | 111 | 0.049 | 0.022 | 0.106 | 0.034 | 0.027 | 0.008 | 0.008 | 0.015 | 0.021 |
| 13–18 | <6 | 4–6 | 83 | 1.0 | 0.956 | 1.0 | 0.985 | 0.985 | 0.988 | 0.987 | 0.992 | 0.988 |
| 13–18 | <6 | 7–10 | 74 | 1.0 | 0.951 | 1.0 | 0.998 | 0.997 | 0.995 | 0.995 | 0.999 | 0.998 |
| 13–18 | 6–9 | 1–3 | 20 | 0.493 | 0.294 | 0.695 | 0.567 | 0.666 | 0.653 | 0.652 | 0.758 | 0.679 |
| 13–18 | 6–9 | 4–6 | 90 | 0.954 | 0.89 | 0.982 | 0.88 | 0.886 | 0.857 | 0.864 | 0.922 | 0.899 |
| 13–18 | 6–9 | 7–10 | 53 | 0.937 | 0.838 | 0.977 | 0.97 | 0.97 | 0.878 | 0.906 | 0.98 | 0.971 |
| 13–18 | 9–12 | 1–3 | 21 | 0.027 | 0.003 | 0.198 | 0.247 | 0.292 | 0.308 | 0.302 | 0.313 | 0.305 |
| 13–18 | 9–12 | 4–6 | 84 | 0.73 | 0.627 | 0.813 | 0.67 | 0.669 | 0.531 | 0.568 | 0.693 | 0.685 |
| 13–18 | 9–12 | 7–10 | 27 | 0.852 | 0.675 | 0.941 | 0.9 | 0.894 | 0.595 | 0.697 | 0.907 | 0.891 |
| 13–18 | 12–15 | 1–3 | 54 | 0.113 | 0.053 | 0.225 | 0.088 | 0.104 | 0.095 | 0.097 | 0.089 | 0.106 |
| 13–18 | 12–15 | 4–6 | 90 | 0.297 | 0.213 | 0.399 | 0.364 | 0.355 | 0.197 | 0.246 | 0.325 | 0.356 |
| 13–18 | 15–18 | 1–3 | 67 | 0.042 | 0.014 | 0.12 | 0.031 | 0.036 | 0.02 | 0.025 | 0.024 | 0.034 |
| 13–18 | 15–18 | 4–6 | 69 | 0.206 | 0.127 | 0.316 | 0.142 | 0.138 | 0.042 | 0.068 | 0.1 | 0.126 |
| 13–18 | 18+ | 1–3 | 135 | 0.005 | 0.001 | 0.036 | 0.003 | 0.004 | 0.001 | 0.001 | 0.002 | 0.003 |
| 13–18 | 18+ | 4–6 | 100 | 0.02 | 0.005 | 0.07 | 0.014 | 0.016 | 0.002 | 0.006 | 0.008 | 0.011 |
| 19–30 | <6 | 4–6 | 75 | 0.998 | 0.948 | 1.0 | 0.977 | 0.977 | 0.989 | 0.978 | 0.985 | 0.98 |
| 19–30 | <6 | 7–10 | 117 | 1.0 | 0.968 | 1.0 | 0.995 | 0.995 | 0.998 | 0.995 | 0.997 | 0.996 |
| 19–30 | 6–9 | 1–3 | 10 | 0.284 | 0.099 | 0.589 | 0.506 | 0.524 | 0.545 | 0.525 | 0.604 | 0.533 |
| 19–30 | 6–9 | 4–6 | 97 | 0.924 | 0.854 | 0.962 | 0.822 | 0.822 | 0.828 | 0.821 | 0.858 | 0.838 |
| 19–30 | 6–9 | 7–10 | 107 | 0.952 | 0.894 | 0.979 | 0.962 | 0.962 | 0.909 | 0.955 | 0.97 | 0.962 |
| 19–30 | 9–12 | 1–3 | 33 | 0.062 | 0.018 | 0.199 | 0.178 | 0.185 | 0.194 | 0.187 | 0.193 | 0.196 |
| 19–30 | 9–12 | 4–6 | 154 | 0.509 | 0.43 | 0.586 | 0.488 | 0.488 | 0.422 | 0.48 | 0.494 | 0.502 |
| 19–30 | 9–12 | 7–10 | 67 | 0.785 | 0.673 | 0.867 | 0.827 | 0.827 | 0.574 | 0.808 | 0.831 | 0.822 |
| 19–30 | 12–15 | 1–3 | 70 | 0.03 | 0.009 | 0.101 | 0.05 | 0.052 | 0.036 | 0.05 | 0.045 | 0.053 |
| 19–30 | 12–15 | 4–6 | 165 | 0.19 | 0.138 | 0.257 | 0.208 | 0.209 | 0.105 | 0.198 | 0.183 | 0.206 |
| 19–30 | 12–15 | 7–10 | 36 | 0.452 | 0.302 | 0.612 | 0.488 | 0.487 | 0.129 | 0.465 | 0.439 | 0.451 |
| 19–30 | 15–18 | 1–3 | 92 | 0.005 | 0.0 | 0.049 | 0.011 | 0.012 | 0.003 | 0.011 | 0.008 | 0.01 |
| 19–30 | 15–18 | 4–6 | 121 | 0.06 | 0.03 | 0.117 | 0.058 | 0.059 | 0.011 | 0.053 | 0.042 | 0.05 |
| 19–30 | 15–18 | 7–10 | 12 | 0.281 | 0.106 | 0.561 | 0.306 | 0.309 | 0.016 | 0.286 | 0.241 | 0.235 |
| 19–30 | 18+ | 1–3 | 104 | 0.0 | 0.0 | 0.036 | 0.001 | 0.002 | 0.0 | 0.001 | 0.001 | 0.001 |
| 19–30 | 18+ | 4–6 | 90 | 0.003 | 0.0 | 0.047 | 0.01 | 0.011 | 0.0 | 0.009 | 0.006 | 0.007 |
| 19–30 | 18+ | 7–10 | 11 | 0.017 | 0.001 | 0.283 | 0.046 | 0.05 | 0.0 | 0.041 | 0.032 | 0.029 |