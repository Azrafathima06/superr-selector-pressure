# Phase 9 — RR layer and outputs (lists LOCKED)

Phase 9: RR layer and outputs (lists LOCKED).

1. RR fit: RR's H deliveries since 2022, WP gained/lost vs expectation by scenario and role, compared with the league
   average for the same scenario and role; which top-20 players are strongest in RR's costliest scenarios; RR 2026 squad flags.
2. Pressure Value Gap: pressure rank vs conventional rank.
3. Impact above replacement: replacement = median rating of active players just below the H-delivery threshold;
   (rating − replacement) × H balls per season, converted to wins with the WP model's WP value of one run on H.
4. Role charts. 5. Player cards (top 10 per side). 6. README is written separately.

## 1. RR fit

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

## 2. Pressure Value Gap

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

## 3. Impact above replacement

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

## 4. Role charts

- `outputs/charts/role_chart_batters.png` (wicket security vs boundary elevation on H)
- `outputs/charts/role_chart_bowlers.png` (economy suppression vs wicket-taking on H)

## 5. Player cards

- `outputs/cards/bat_01_vaibhav_sooryavanshi.md`
- `outputs/cards/bat_02_abhishek_sharma.md`
- `outputs/cards/bat_03_finn_allen.md`
- `outputs/cards/bat_04_phil_salt.md`
- `outputs/cards/bat_05_suryakumar_yadav.md`
- `outputs/cards/bat_06_tim_david.md`
- `outputs/cards/bat_07_heinrich_klaasen.md`
- `outputs/cards/bat_08_andre_russell.md`
- `outputs/cards/bat_09_travis_head.md`
- `outputs/cards/bat_10_priyansh_arya.md`
- `outputs/cards/bowl_01_jasprit_bumrah.md`
- `outputs/cards/bowl_02_sunil_narine.md`
- `outputs/cards/bowl_03_matheesha_pathirana.md`
- `outputs/cards/bowl_04_nathan_ellis.md`
- `outputs/cards/bowl_05_rashid_khan_afghanistan_gujarat_titans.md`
- `outputs/cards/bowl_06_eshan_malinga.md`
- `outputs/cards/bowl_07_josh_hazlewood.md`
- `outputs/cards/bowl_08_bhuvneshwar_kumar.md`
- `outputs/cards/bowl_09_mujeeb_ur_rahman.md`
- `outputs/cards/bowl_10_varun_chakravarthy.md`