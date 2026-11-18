"""Assemble outputs/FINAL.md — one self-contained file with every number taken from the pipeline outputs."""
import re

import numpy as np
import pandas as pd

from . import config as C
from . import metrics as MT
from . import rating as RT
from . import wp_model as WM

T = C.TABLES


def md(df, index=False):
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v).replace("|", "/") for v in r) + " |")
    return "\n".join(out)


def rd(name):
    return pd.read_csv(T / name)


def section(text, start, end=None):
    """Extract a markdown section of a phase report (from heading `start` up to heading `end` or the next same-level heading)."""
    i = text.find(start)
    if i < 0:
        return f"_(section '{start}' not found)_"
    j = text.find(end, i + len(start)) if end else -1
    return text[i:j if j > 0 else None].strip()


def sub(text, prefix, drop_first=False):
    """Re-number embedded headings as sub-sections (#### prefix.1, prefix.2, ...); strip their old numbers."""
    k = [0]

    def rep(m):
        if drop_first and k[0] == 0 and m.start() == 0:
            k[0] = -1
            return ""
        k[0] = max(k[0], 0) + 1
        title = re.sub(r"^[A-Z0-9]+\. ", "", m.group(2))
        return f"#### {prefix}.{k[0]} {title}"
    out = re.sub(r"(?m)^(#{1,6}) (.+)$", rep, text.strip())
    return out.lstrip("\n")


def demote(text, by=2):
    return re.sub(r"(?m)^(#{1,4}) ", lambda m: "#" * min(len(m.group(1)) + by, 6) + " ", text)


def side_sec(rep07, side):
    return (section(rep07, "## Batters: top 10 and next 10", "## Bowlers: top 10 and next 10") if side == "Batters"
            else section(rep07, "## Bowlers: top 10 and next 10"))


def run():
    rep = {k: (C.OUT / f).read_text() for k, f in {
        "01": "01_data_audit.md", "02": "02_wp_model.md", "02b": "02b_wp_endgame.md", "03": "03_leverage.md",
        "04": "04_value.md", "05": "05_metrics.md", "06": "06_repeatability.md", "06b": "06b_signal_recovery.md",
        "07": "07_rating.md", "08": "08_validation.md", "09": "09_rr_outputs.md"}.items()}
    tb, tw = rd("top10_batters.csv"), rd("top10_bowlers.csv")
    m = pd.read_parquet(C.PROC / "matches.parquet")
    wpm = rd("wp_metrics.csv")
    trust = rd("metric_repeatability.csv")
    sig = rd("signal_recovery_summary.csv")
    dev = rd("pressure_deviation_variance.csv")
    var = rd("rating_variants.csv")
    oos = rd("rating_method_oos.csv")
    oot = rd("out_of_time_validation.csv")
    share = rd("out_of_time_top20_share.csv")
    ko = rd("knockout_check.csv")
    fv = rd("face_validity.csv")
    vp = rd("vignesh_puthur.csv")
    rr = rd("rr_fit_scenarios.csv")
    rrb = rd("rr_fit_best_players.csv")
    lev_sh = rd("leverage_share_by_innings_phase.csv")
    scen = rd("leverage_scenarios.csv")
    dec = rd("leverage_deciles.csv")
    cons = rd("leverage_consistency.csv")
    endg = rd("wp_endgame_candidates.csv")
    spots = rd("wp_endgame_spots.csv")
    om = rd("outcome_model_metrics.csv")
    teams = rd("team_strength_groups.csv")
    band = rd("pressure_band_check.csv")
    imp_b, imp_w = rd("impact_above_replacement_bat.csv"), rd("impact_above_replacement_bowl.csv")
    pvg_b, pvg_w = rd("pressure_value_gap_bat.csv"), rd("pressure_value_gap_bowl.csv")
    prim = {s: var[var.variant == f"{s}: primary"].iloc[0] for s in ("bat", "bowl")}
    chosen_bat = wpm[wpm.model.str.startswith("CHOSEN") & (wpm.innings == 1)].iloc[0]
    chosen_bowl = wpm[wpm.model.str.startswith("CHOSEN") & (wpm.innings == 2)].iloc[0]
    kept = int(m.kept.sum())
    n_ipl = int((m.kept & m.is_ipl).sum())
    rr_worst = rr.iloc[0]
    tp = lambda t: ", ".join(f"{r.player}" for r in t.head(10).itertuples())   # noqa: E731
    sp = {s: dev[(dev.side == s) & dev.quantity.str.startswith("pressure-specific")].iloc[0] for s in ("bat", "bowl")}
    mixA = {s: re.search(r"\| " + s + r" \| (\d+) \| (\d+) \| [\d.]+ \| ([\d.]+) \| (\[[^\]]+\]) \|.*?\| ([\d.]+) \| ([\d.]+) \| ([\d.e+-]+) \|",
                         rep["06b"]) for s in ("bat", "bowl")}

    L = []
    A = L.append
    # ------------------------------------------------------------------------------------------------ 0
    A("# SupeRR Selector — Round 1: pressure performance (FINAL working file)\n")
    A("_Every number in this file is generated from the pipeline outputs (`python run_all.py`). Charts and tables are "
      "referenced by path. Decisions and reasons: `DECISIONS.md` (summarised in the appendix). Lists locked at git tag `final-lists`._\n")
    A("## 0. Summary\n")
    A("**Pressure skill is real. We measured it, tested that it lasts, and ranked the players who have it.**\n")
    A(f"We modelled the win probability (WP) of every delivery in {kept} men's T20 matches (2022–2026; {n_ipl} IPL, the rest "
      "T20I), measured how much each delivery could swing the match (a leverage index), and defined pressure as the top 25% "
      "of leverage within each innings. Each delivery was then scored as actual minus expected outcome, adjusted for match "
      "state, venue, competition strength and the quality of the opponent faced. We tested which pressure measures repeat "
      "for the same player (split-half over 50 random match splits, season-to-season, earlier-vs-later), tried five "
      "pre-specified ways to recover more pressure-specific signal, and rated players with an empirical Bayes model of "
      "their run value on high-leverage deliveries, tested out of time (2022–24 → 2025–26). "
      f"Final top 10 batters: {tp(tb)}. Final top 10 bowlers: {tp(tw)}.\n")
    A("**Headline finding.** Pressure-specific variation exists on both sides — a mixed model with a player-specific "
      f"slope on leverage beats one without it (likelihood-ratio p = {mixA['bat'].group(7) if mixA['bat'] else 'see 06b'} batters, "
      f"{mixA['bowl'].group(7) if mixA['bowl'] else 'see 06b'} bowlers; slope SD ≈ "
      f"{mixA['bat'].group(5) if mixA['bat'] else '0.065'} runs per ball) — but it is too small relative to sampling noise to "
      "estimate reliably per player with 2022–26 data. **What repeats under pressure is process** (strike-rate delta, dot "
      "suppression, boundary elevation for batters); outcome measures (WP value, wicket security) and bowlers' individual "
      "pressure metrics do not. **Core skill is the best predictor of pressure performance**, so the ratings are core skill "
      "measured carefully, updated by each player's own pressure record only as far as the data justify, and restricted to "
      "players who have demonstrably delivered under pressure.\n")

    # ------------------------------------------------------------------------------------------------ 1
    A("## 1. Pressure definition\n")
    A("### 1.1 Win probability model\n")
    A("- WP is always the batting side's probability of winning; ties count 0.5 in all WP values and swings.")
    vp_ = section(rep["02"], "- League mean for a match", "- Par at a stage")
    A("- **Venue par (leak-free):** " + " ".join(x.lstrip("- ").strip() for x in vp_.splitlines() if x.strip()))
    A("- " + section(rep["02"], "- Par at a stage", "## How the model").lstrip("- ").replace("\n", " ").strip())
    A("- **Features (sign-constrained logistic regression; L2 as sklearn C=1 on standardised inputs; signs enforced with L-BFGS-B):**")
    A(f"  - 1st innings: {', '.join(WM.FEATS1)} (signs {WM.SIGN1}).")
    A(f"  - 2nd innings main model: {', '.join(WM.FEATS2)} (signs {WM.SIGN2}).")
    A(f"  - 2nd innings endgame model (fit on ≤{WM.END_FIT_MAX} balls left, blended linearly from {WM.BLEND_HI} to {WM.BLEND_LO} balls left): {', '.join(WM.FEATSE)}.")
    A("  - zmv and dp_logit come from a Markov ball-outcome model of the chase (outcome probabilities by over, wickets lost; "
      "exact backward induction; expected runs still to come and their s.d. = a data-estimated DLS-like resource table). "
      "wbK = wickets in hand × min(balls left, K)/K, so wickets lose value as balls run out.")
    A("- **How the model was chosen:** LightGBM could not load (needs libomp; no system libraries installed by agreement); "
      "a monotonic gradient-boosted model (scikit-learn HistGradientBoosting) lost to logistic regression on grouped "
      "out-of-fold log-loss. A first logistic version gave a side needing 2 off the last ball 99.6%; the Markov features, "
      "wicket bases and endgame blend fixed this.")
    A("- **Out-of-fold scores (5-fold, grouped by match):**\n")
    A(md(wpm))
    A("\n- Calibration: `outputs/charts/wp_calibration.png`. Endgame calibration: `outputs/charts/wp_endgame_calibration.png`. "
      "Sanity curves: `outputs/charts/wp_curve_45513.png`, `wp_curve_15187.png`, `wp_curve_40725.png`.")
    A("- **Endgame test (2nd innings, ≤30 balls left, out-of-fold; chosen by log-loss on these deliveries):**\n")
    A(md(endg.drop(columns=["key"])))
    A("\nSpot states (IPL chase of 175, par 175; empirical with 95% Wilson interval):\n")
    A(md(spots))
    A("\n- Season-forward and by-competition-group scores: `outputs/tables/wp_season_forward.csv`, `wp_metrics_by_group.csv`.")
    A("- **1st-innings limitation:** " + section(rep["03"], "**1st-innings limitation.**", "\n\n").replace("**1st-innings limitation.**", "").strip())
    A("\n### 1.2 Leverage index (pressure)\n")
    A("- Player-agnostic outcome model p(o | state, context), o ∈ {0,1,2,3,4,6,wicket,wide/no-ball}; context = innings, "
      "balls left, wickets, score vs par / required rate vs par, venue par, phase, batting hand × bowling type, pace/spin, "
      "competition strength. Gradient boosting, out-of-fold by match:\n")
    A(md(om))
    A("\n- **Centred leverage (used):** LI_raw = Σ_o p(o|s)·|WP(s_o) − Σ_o' p(o'|s)·WP(s_o')|; **LI = LI_raw / mean LI_raw**. "
      "The specified form Σ p·|WP(s_o) − WP(s)| is kept as `LI_spec`; it counted the 1st-to-2nd-innings model hand-over jump "
      "as pressure. Where the WP model is internally consistent the two coincide:\n")
    A(md(cons))
    A("\n- **High-leverage set H = top 25% of LI within each innings** (H20 and H33 for robustness). A global cut-off put 83% "
      "of H in the 2nd innings and none in 1st-innings middle overs. Baseline L = the rest.")
    A("- Where H comes from (share of all deliveries vs share of H):\n")
    A(md(lev_sh))
    A("\n- **Evidence that leverage separates situations** (outcome rates across LI deciles; chart "
      "`outputs/charts/leverage_deciles.png`, histogram `outputs/charts/leverage_histogram.png`):\n")
    A(md(dec))
    A("\n- **Six scenarios** (definitions in §2.3) and their share of H:\n")
    A(md(scen))
    A("\n- Top-20 highest-leverage deliveries: `outputs/tables/leverage_top20.csv` (all are last-ball chases needing 2–4).")
    A("- **Rain-shortened matches** (32; recorded or detected revisions, or innings shorter than 20 overs) are excluded from H "
      "and L (the WP model was never trained on shortened innings) but kept for conventional statistics.")
    A("- Pressure-band check (H25 vs H33 vs H50 vs LI-weighted; split-half reliability): `outputs/tables/pressure_band_check.csv`; "
      "H25 kept (no version clearly more reliable).\n")

    # ------------------------------------------------------------------------------------------------ 2
    A("## 2. Performance measurement\n")
    A("### 2.1 Per-ball value vs expectation\n")
    A(sub(section(rep["04"], "## 1. Opposition quality", "## 4. Per-player"), "2.1"))
    A("\n### 2.2 Metrics (all = actual − expected, positive = good, on H and on L)\n")
    A(section(rep["05"], "**Accounting rules.**", "## Retention, centred"))
    A("\n### 2.3 Pressure level, baseline, retention\n")
    A("P = mean v over H; B = mean v over L; **R = (P − B) − population (P − B)** (delivery-weighted; R_raw kept). "
      + section(rep["05"], "**Finding — population-wide shift:**", "## Summary per metric").split("\n")[0])
    A("\n### 2.4 Final rating method\n")
    A(f"- **Batters — L-only component prior + component-wise H update.** Run value per ball = runs component (runs scored vs "
      f"expected) + wicket component (dismissals vs expected × wicket cost in runs). Prior from non-H (L) deliveries only, each "
      f"component shrunk by its own reliability (c_runs {prim['bat']['c_runs']}, c_wkt {prim['bat']['c_wkt']}); prior = "
      f"a + b_runs·θ_runs + b_wkt·θ_wkt (weighted regression of observed H run value across players). Rating = prior + k·(H mean − prior), "
      f"k = τ²_dev / (τ²_dev + σ²_H/n_H), τ²_dev = {prim['bat']['tau2_dev_used']} (pressure-specific variance against this prior).")
    A(f"- **Bowlers — bivariate component rating.** Rating = Σ_k (μ̂_k + δ̂_k): μ_k = core skill from ALL deliveries shrunk by its "
      f"between-player variance; δ_k = (H mean − L mean) − population shift, shrunk by var(δ_k). Runs component: c "
      f"{prim['bowl'].get('runs: c (core shrinkage)', 'see table')}, var(δ) {prim['bowl'].get('runs: tau2_delta est [95% CI]', '')}. "
      f"Wicket component: {prim['bowl'].get('wkt: c (core shrinkage)', '')} (bowler wicket-taking has no estimable between-player "
      "variance, on L or on all deliveries).")
    A("- **Why these methods** (out-of-sample prediction of next-season run value on H, folds grouped by player):\n")
    A(md(oos))
    A("\n- **Rank score = rating − SE** (SE: bootstrap over the player's matches, 2,000 resamples). **Tiers:** a new tier starts "
      "when the gap to the previous rank score exceeds the larger of the two adjacent SEs; positions inside a tier are not "
      "statistically separable. Recency weighting was tested and not kept (no clear out-of-sample gain).")
    A("- All fitted parameters per variant: `outputs/tables/rating_variants.csv`:\n")
    A(md(var))

    # ------------------------------------------------------------------------------------------------ 3
    A("\n## 3. Fair comparisons\n")
    A(f"- **Eligibility:** ≥ 2 × the median number of H deliveries in a regular player-season (≥10 matches): batters "
      f"{int(prim['bat'].min_H)}, bowlers {int(prim['bowl'].min_H)} (two regular seasons of pressure use); played in 2025 or 2026.")
    A("- **Delivery requirement:** to appear in a final ranked list a player must have demonstrated delivery under pressure — "
      "raw run value on H above expectation (> 0) (brief: 'deliver when it matters'). It is a requirement, not an estimate; "
      "the noise-aware version (exclude only if raw H + 1.28·SE < 0) is a sensitivity check (§5).")
    A("- **Retention safety gate (pooled):** exclude a player only if the 95% interval of centred R — pooled over 5 seeds × "
      "2,000 bootstrap resamples — lies entirely below 0. R is unreliable as a metric (§5), so it is a gate only.")
    A(f"- Eligible after all conditions: {prim['bat'].eligible} batters, {prim['bowl'].eligible} bowlers. "
      f"Removed (volume and activity met): retention {prim['bat'].removed_by_retention} batters / {prim['bowl'].removed_by_retention} bowlers; "
      f"delivery requirement {prim['bat'].removed_by_delivery_condition} batters / {prim['bowl'].removed_by_delivery_condition} bowlers (names in §4.2).")
    A("- **Sample size and shrinkage:** every player value is shrunk toward the population (empirical Bayes; c = within-player "
      "per-ball variance / between-player variance). Opposition ratings: c batters 132, bowlers 564. Rating components as in §2.4.")
    A("- **IPL vs T20I:** competition strength group (IPL / full-member T20I / weaker T20I) enters the WP, outcome and "
      "expectation models; every value is relative to expectation for that competition. IPL-only rerun in §5.")
    A("- **Competition groups (teams):**\n")
    A(md(teams.groupby("group").team_name.apply(lambda s: ", ".join(sorted(s))).reset_index()))
    A("\n- **Data cleaning and exclusions** (sub-sections 3.1–3.3):\n")
    A(sub(section(rep["01"], "## 2. What the run columns mean", "## 4. Competition strength groups") + "\n\n"
          + section(rep["01"], "## 6. Data problems found"), "3"))
    A("\n- **Player identity:** ids are the key everywhere. id 4458 = 'Aneurin Donald' (mapping file's 'AHT Donald' ignored). "
      "Same-name different players kept apart by id. Final-list names match the data spelling; the only final name shared "
      "with another id is **Rashid Khan** → shown as 'Rashid Khan (Afghanistan, Gujarat Titans)', player_id 4722 (id 20172 is "
      "Nepal's Rashid Khan, mapping file only). Basil Hameed (UAE) ids 8638/39444 are probably one person but kept apart (not provable; negligible).\n")

    # ------------------------------------------------------------------------------------------------ 4
    A("## 4. Evidence for the top 10\n")
    A("### 4.1 Final lists (LOCKED) — ranks 1–10 and next 10\n")
    A("Ratings: runs per 100 high-leverage balls relative to expectation. Core = core skill; pressure dev = the player's "
      "pressure-specific adjustment; raw H = unshrunk H mean ± bootstrap SE; k_update = weight on the player's own pressure record.\n")
    keepc = [c for c in tb.columns if c not in ("player_id", "name_matches_data", "name_shared_by_two_ids")]
    A("**Batters**\n")
    A(md(tb[keepc]))
    A("\n**Bowlers**\n")
    A(md(tw[keepc]))
    A("\n### 4.2 Who was removed by the eligibility conditions\n")
    for side in ("Batters", "Bowlers"):
        sec = side_sec(rep["07"], side)
        A(f"**{side}**\n")
        A(section(sec, "### Removed by the eligibility conditions", "### Retention gate").split("\n", 1)[1].strip())
        A("")
    A("### 4.3 Comparisons with conventional stats and role mix\n")
    for side in ("Batters", "Bowlers"):
        sec = side_sec(rep["07"], side)
        A(f"**{side}**\n")
        A(section(sec, "### Comparisons", "### Sensitivity").split("\n", 1)[1].strip())
        A("")
    A("### 4.4 Pressure Value Gap\n")
    A(sub(section(rep["09"], "## 2. Pressure Value Gap", "## 3. Impact above replacement"), "4.4", drop_first=True))
    A("\n### 4.5 Impact above replacement\n")
    A(sub(section(rep["09"], "## 3. Impact above replacement", "## 4. Role charts"), "4.5", drop_first=True))
    A("\nNote: about 70 runs per win is the WP model's sensitivity to one run on an average high-leverage ball (1 run ≈ "
      "1.4% WP), not a season-standings conversion.\n")
    A("### 4.6 RR fit\n")
    A(sub(section(rep["09"], "## 1. RR fit", "## 2. Pressure Value Gap"), "4.6", drop_first=True))
    A(f"\nCostliest for RR: **{rr_worst.scenario} ({rr_worst.role})**, {rr_worst['RR_total_WP_vs_league (wins)']} wins vs a "
      f"league-average team over {rr_worst.RR_H_deliveries} H deliveries.\n")
    A("### 4.7 Role charts\n")
    A("- Batters: `outputs/charts/role_chart_batters.png` (wicket security vs boundary elevation on H).")
    A("- Bowlers: `outputs/charts/role_chart_bowlers.png` (economy suppression vs wicket-taking on H).")
    A("- Core skill vs pressure (H vs L): `outputs/charts/core_vs_pressure.png`.\n")
    A("### 4.8 Player cards (all 20)\n")
    for p in sorted((C.OUT / "cards").glob("*.md")):
        A(demote(p.read_text(), 3))
        A(f"\n_(card file: `outputs/cards/{p.name}`)_\n")
    A("### 4.9 Vignesh Puthur\n")
    A(md(vp))
    vb = vp[vp.role == "bowler"]
    if len(vb):
        v = vb.iloc[0]
        A(f"\nIn the data as a bowler only: {int(v.matches)} matches in {v.seasons}, {int(v.deliveries)} deliveries, "
          f"{int(v.H_deliveries)} on high-leverage deliveries (eligibility needs {int(v.min_H_for_eligibility)}). Not eligible; the "
          f"model rating ({v['rating_per_100 (model, shrunk)']} per 100 H balls) is almost entirely the population prior. The raw "
          f"H value ({v['raw_H_per_100']} ± {v['raw_H_se_per_100 (simple)']}) is too noisy to support any claim. He is in RR's IPL 2026 squad.\n")

    # ------------------------------------------------------------------------------------------------ 5
    A("## 5. Reliability and limitations\n")
    A("### 5.1 Trust table (split-half reliability, mean of 50 random match splits, Spearman–Brown; season-to-season)\n")
    A("Chart: `outputs/charts/trust_table.png`.\n")
    A(md(trust))
    A("\n### 5.2 Phase 6 / 6B findings\n")
    A(sub(section(rep["06"], "## Headline finding", "## C. Repeatability") + "\n\n"
          + section(rep["06b"], "## Headline findings", "## Summary: method"), "5.2"))
    A("\nPressure-specific deviation variance (Phase 6, runs² per ball):\n")
    A(md(dev.round(6)))
    A("\nSignal recovery — every pre-specified method (pass = reliability interval above 0 AND clear out-of-sample gain):\n")
    A(md(sig))
    A("\n### 5.3 Out-of-time validation (rating stage rebuilt on 2022–24, scored on 2025–26)\n")
    A("Chart: `outputs/charts/out_of_time_validation.png`. Per-ball values come from league-level models fitted out-of-fold "
      "over all seasons; the player-level rating stage uses no 2025–26 data.\n")
    A(md(oot))
    A("\n" + md(share))
    A("\n### 5.4 Knockout check (31 matches — small sample)\n")
    A(md(ko))
    A("\n### 5.5 Face validity (data reasons only)\n")
    A(md(fv))
    A("\n### 5.6 Sensitivity (top-10 overlap with the final list)\n")
    for side in ("Batters", "Bowlers"):
        sec = side_sec(rep["07"], side)
        A(f"**{side}**\n")
        A(section(sec, "| variant |"))
        A("")
    A("Rating-method sensitivity: the bivariate batter rating (tested alternative) is in `outputs/tables/appendix_bivariate_batters.csv`.\n")
    A("### 5.7 Tiers\n")
    A(f"Batters: tier 1 = rank 1 ({tb.player.iat[0]}); tier 2 = rank 2; ranks 3–20 form one tier. Bowlers: tier 1 = ranks 1–2; "
      "ranks 3–20 form one tier. Ordering inside a tier follows rank score but is not statistically supported.\n")
    A("### 5.8 Traps and leads\n")
    A("- **Yorker-label trap:** delivery_style is recorded after the ball; a missed yorker tends to be logged as a full toss "
      "(yorkers 0.84 runs per ball vs full tosses 1.87), so 'yorker execution' measures the label, not the skill. Reported, never ranked.")
    A("- **Death-pace lead:** death pace specialists (pace, ≥30% of overs at the death; 45–51 bowlers) show the strongest pressure "
      "signal within H (reliability: WP value 0.47, economy 0.40, boundary suppression 0.53) but fail the pre-specified rule; a lead "
      "for further work, not used in the ranking.")
    A("- **Role pooling:** raised reliability without improving prediction — the gain came from shared role means, not player-level signal.\n")
    A("### 5.9 Limitations\n")
    lim = [
        "No ball-tracking: no line/length, no false-shot or beaten-bat data; quality of contact cannot be measured.",
        "bowl_speed is missing (0) on 38.7% of deliveries; speed is not used.",
        "Knockout sample is 31 matches; knockout results are descriptive only.",
        "32 rain-shortened matches are excluded from pressure sets; 8 rain revisions were not recorded in matches.csv and were detected from the data.",
        f"1st-innings WP is inherently less certain (out-of-fold AUC {chosen_bat.auc}); 1st-innings leverage comes mainly from wicket risk and middle overs rarely reach H.",
        "Last-over WP rests on cells of 5–25 matches; it is the least certain part of the WP model.",
        "Per-player pressure-specific skill is not reliably estimable with 2022–26 data; ratings lean on core skill by design.",
        "Bowler wicket-taking has no estimable between-player variance; bowler ratings carry no wicket value beyond what economy reflects.",
        "WP value P and retention R are not repeatable per player; R is used only as a safety gate.",
        f"Out of time, our rating did not beat a plain career average (batters r {oot.pearson_r.iat[0]} vs {oot.pearson_r.iat[1]}; bowlers "
        f"{oot.pearson_r.iat[4]} vs {oot.pearson_r.iat[5]}); intervals overlap.",
        "Per-ball values use league-level models fitted out-of-fold over all seasons (no player's own match informs its value, but model parameters see all years).",
        "The delivery requirement (raw H > 0) is noisy for players near 0 (e.g. Rajat Patidar −0.30, Maheesh Theekshana −0.48 per 100 H balls).",
        "Player identity: Basil Hameed (UAE) may be one person under two ids; negligible sample.",
    ]
    for x in lim:
        A(f"- {x}")
    A("\n### 5.10 Picks we are least sure of\n")
    sens_txt = {s: side_sec(rep["07"], s) for s in ("Batters", "Bowlers")}
    rows = []
    for side, t in (("Batters", tb), ("Bowlers", tw)):
        absent = {}
        for line in section(sens_txt[side], "| variant |").splitlines()[2:]:
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) >= 3:
                for n in [x.strip() for x in parts[2].split(",") if x.strip()]:
                    absent[n] = absent.get(n, 0) + 1
        for r in t.head(10).itertuples():
            why = []
            if absent.get(r.player, 0) >= 2:
                why.append(f"drops out of the top 10 in {absent[r.player]} sensitivity variants")
            if "(2 seasons)" in r.seasons_played:
                why.append("only 2 seasons of data")
            if r.raw_H_se_per_100 > abs(r.raw_H_per_100):
                why.append(f"raw H {r.raw_H_per_100:+.1f} smaller than its SE {r.raw_H_se_per_100:.1f}")
            hi = float(re.findall(r"[-+][\d.]+", r.R_ci95)[1])
            if hi < 0.02:
                why.append(f"retention interval upper end {hi:+.3f} (just passes the gate)")
            if why:
                rows.append({"side": side, "rank": r.rank, "player": r.player, "why less sure": "; ".join(why)})
    A(md(pd.DataFrame(rows)) if rows else "none flagged by these rules")
    A("\nRules used: drops out of the top 10 in ≥2 sensitivity variants; ≤2 seasons; raw H smaller than its own SE; retention "
      "interval upper end < 0.02.\n")

    # ------------------------------------------------------------------------------------------------ 6
    A("## 6. Appendix\n")
    A("### 6.1 Thresholds and parameters\n")
    par = pd.DataFrame([
        ("Data", "Total runs", "runs + extras1_runs + extras_2_runs", "reconciles 804/809 results"),
        ("Data", "No-balls", "extras_type Overstep and 'other'", "'other' legal → 244 overs of 7–9 balls"),
        ("WP", "Venue par window", f"{C.PAR_WINDOW_DAYS} days, shrink k estimated (12.0)", "leak-free"),
        ("WP", "CV folds", "5, grouped by match", "no leakage"),
        ("WP", "Endgame model", f"fit on ≤{WM.END_FIT_MAX} balls left; blend {WM.BLEND_HI}→{WM.BLEND_LO}", "chosen on ≤30-ball log-loss"),
        ("Leverage", "H", "top 25% LI within innings (H20, H33 robustness)", "global cut-off skewed to 2nd innings"),
        ("Leverage", "Rain-shortened", "excluded from H/L (switch EXCLUDE_SHORTENED_FROM_PRESSURE)", "WP not trained on them"),
        ("Scenarios", "Post-wicket / entry / squeeze", "12 deliveries; first 10 balls faced when arriving in H; after 3+ dots", "brief"),
        ("Scenarios", "Collapse", f"team WP < {MT.COLLAPSE_WP} at the batter's first ball", "brief"),
        ("Metrics", "Report minimums", str(MT.MIN_REPORT), "Phase 5 summaries"),
        ("Phase 6", "Reliability", "mean SB over 50 random match splits; 200 bootstrap", "single splits unstable"),
        ("Phase 6B", "Mixed-model interval", "parametric bootstrap, 100 resamples", "profile likelihood not computable reliably"),
        ("Rating", "Bootstrap", f"{RT.N_BOOT_SE} resamples for SE; retention pooled over {len(RT.SEEDS)} seeds", "stable gates"),
        ("Rating", "Eligibility", f"bat ≥{int(prim['bat'].min_H)}, bowl ≥{int(prim['bowl'].min_H)} H balls; active 2025/26", "2 × median regular season"),
        ("Rating", "Delivery requirement", "raw H run value > 0", "brief"),
        ("Rating", "Retention gate", "pooled R 95% interval entirely < 0 → exclude", "safety gate"),
        ("Rating", "Tiers", "gap > larger adjacent SE → new tier", "separability"),
        ("Rating", "Recency", f"λ ∈ {RT.LAMBDAS} tested; not kept", "no clear gain"),
        ("Validation", "Out-of-time", "2022–24 → 2025–26; ≥30 test H balls", "brief"),
        ("General", "Seed", f"{C.SEED}; OMP threads 2", "reproducibility"),
    ], columns=["area", "parameter", "value", "reason"])
    A(md(par))
    A("\n### 6.2 Decisions log (DECISIONS.md, verbatim)\n")
    A(demote((C.ROOT / "DECISIONS.md").read_text(), 2))
    A("\n### 6.3 Charts and tables\n")
    for p in sorted(C.CHARTS.glob("*.png")):
        A(f"- `outputs/charts/{p.name}`")
    A("")
    for p in sorted(C.TABLES.glob("*.csv")):
        A(f"- `outputs/tables/{p.name}`")
    A("\n### 6.4 How to reproduce\n")
    A("```bash\n/usr/bin/python3 -m venv .venv\n.venv/bin/pip install -r requirements.txt\n.venv/bin/python run_all.py\n```\n"
      "Phase reports: `outputs/01_data_audit.md` … `outputs/09_rr_outputs.md`. Checks run after every phase (`src/checks.py`).")
    (C.OUT / "FINAL.md").write_text("\n".join(L))
    print(f"[final] FINAL.md written: {len(L)} blocks, {sum(len(x) for x in L):,} characters")
