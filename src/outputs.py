"""Phase 9: RR layer and outputs (lists LOCKED).

1. RR fit: RR's H deliveries since 2022, WP gained/lost vs expectation by scenario and role, compared with the league
   average for the same scenario and role; which top-20 players are strongest in RR's costliest scenarios; RR 2026 squad flags.
2. Pressure Value Gap: pressure rank vs conventional rank.
3. Impact above replacement: replacement = median rating of active players just below the H-delivery threshold;
   (rating − replacement) × H balls per season, converted to wins with the WP model's WP value of one run on H.
4. Role charts. 5. Player cards (top 10 per side). 6. README is written separately.
"""
import re

import numpy as np
import pandas as pd

from . import config as C
from . import rating as RT
from .plotstyle import plt, save, SERIES, TEXT2

KEY = ["match_id", "innings", "innings_ball_num"]
O = ["0", "1", "2", "3", "4", "6", "wicket", "wide_nb"]
SCEN = {"post_wicket": "post-wicket (12 deliveries after a wicket)", "entry_under_pressure": "entry under pressure",
        "death_chase": "death overs, 2nd innings", "dot_squeeze": "dot squeeze (after 3+ dots)",
        "pp_vs_set_openers": "powerplay vs set openers"}
MIN_SC = 20


def _md(df):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join(out)


def deliveries():
    lv = pd.read_parquet(C.PROC / "leverage.parquet")
    b = pd.read_parquet(C.PROC / "balls.parquet")[KEY + ["batting_team_id", "bowling_team_id", "striker_id", "bowler_id",
                                                        "phase", "overid", "year"]]
    wp = pd.read_parquet(C.PROC / "wp_preds.parquet")[KEY + ["wp_after"]]
    d = lv.merge(b, on=KEY).merge(wp, on=KEY)
    P = d[[f"p_{o}" for o in O]].values.astype(float)
    P /= P.sum(1, keepdims=True)
    d["value_bat_team"] = d.wp_after.values - (P * d[[f"wp_o_{o}" for o in O]].values).sum(1)   # vs player-agnostic expectation
    d["H"] = d[C.PRIMARY_H].astype(bool)
    return d


def scenario_masks(d):
    return {"post_wicket": d.sc_post_wicket.values, "entry_under_pressure": d.sc_entry_under_pressure.values,
            "death_chase": d.sc_death_chase.values, "dot_squeeze": d.sc_dot_squeeze.values,
            "pp_vs_set_openers": d.sc_pp_vs_set_openers.values}


# ------------------------------------------------------------------------------------------------ 1. RR fit
def rr_fit(d, comp, shown):
    H = d[d.H]
    masks = scenario_masks(H)
    rows = []
    for sc, lab in SCEN.items():
        m = masks[sc]
        for role, team_col, sign in (("RR batting", "batting_team_id", 1), ("RR bowling", "bowling_team_id", -1)):
            x = H[m]
            v = sign * x.value_bat_team
            rr = x[team_col] == C.RR_TEAM_ID
            n = int(rr.sum())
            if n == 0:
                continue
            rr_mean, lg_mean = float(v[rr].mean()), float(v[x.match_id.isin(x.match_id) & (x[team_col] != C.RR_TEAM_ID)].mean())
            rows.append({"scenario": lab, "role": role, "RR_H_deliveries": n, "RR_matches": int(x[rr].match_id.nunique()),
                         "RR_WP_per_delivery_vs_expectation": round(rr_mean, 5), "league_WP_per_delivery": round(lg_mean, 5),
                         "RR_total_WP_vs_league (wins)": round(n * (rr_mean - lg_mean), 3)})
    t = pd.DataFrame(rows).sort_values("RR_total_WP_vs_league (wins)").reset_index(drop=True)
    # strongest top-20 players in the 3 costliest scenario x role combinations
    best = []
    for _, r in t.head(3).iterrows():
        sc = [k for k, v in SCEN.items() if v == r.scenario][0]
        side = "bat" if r.role == "RR batting" else "bowl"
        top20 = pd.read_csv(C.TABLES / ("top10_batters.csv" if side == "bat" else "top10_bowlers.csv"))
        flags = d[KEY + [f"sc_{sc}"] + [C.PRIMARY_H]].rename(columns={f"sc_{sc}": "sc"})
        cv = comp[(comp.side == side) & (comp.metric == "rv") & comp.player_id.isin(top20.player_id)].merge(flags, on=KEY)
        cv = cv[cv.sc.astype(bool) & cv.H]
        g = cv.groupby("player_id").val.agg(["mean", "size"])
        g = g[g["size"] >= MIN_SC].sort_values("mean", ascending=False).head(5)
        for pid, rr_ in g.iterrows():
            best.append({"RR's costly scenario": f"{r.scenario} ({r.role})", "player": shown[side][pid],
                         "final_rank": int(top20.loc[top20.player_id == pid, "rank"].iat[0]),
                         "run_value_per_100_balls_in_scenario_on_H": round(100 * rr_["mean"], 1), "deliveries": int(rr_["size"])})
    return t, pd.DataFrame(best)


def rr_squad_2026():
    mp = pd.read_csv(C.RAW / "competition_team_player_mapping.csv")
    return set(mp[(mp.comp_name == "IPL 2026") & (mp.team_id == C.RR_TEAM_ID)].player_id)


# ------------------------------------------------------------------------------------------------ 2. Pressure Value Gap
def pvg(side, shown):
    r = pd.read_csv(C.TABLES / f"ratings_all_{side}.csv")
    e = r[r.eligible].sort_values("rank_score", ascending=False).reset_index(drop=True)
    e["pressure_rank"] = np.arange(1, len(e) + 1)
    cv = RT.conventional(side).reindex(e.player_id)
    if side == "bat":
        e["average"], e["strike_rate"] = cv.average.values.round(1), cv.strike_rate.values.round(1)
        e["avg_rank"] = e.average.rank(ascending=False, method="min")
        e["sr_rank"] = e.strike_rate.rank(ascending=False, method="min")
        e["conventional_rank"] = ((e.avg_rank + e.sr_rank) / 2).rank(method="min")
        cols = ["average", "strike_rate", "avg_rank", "sr_rank"]
    else:
        e["economy"], e["wickets"] = cv.economy.values.round(2), cv.wickets.values.astype(int)
        e["econ_rank"] = e.economy.rank(ascending=True, method="min")
        e["wkt_rank"] = e.wickets.rank(ascending=False, method="min")
        e["conventional_rank"] = ((e.econ_rank + e.wkt_rank) / 2).rank(method="min")
        cols = ["economy", "wickets", "econ_rank", "wkt_rank"]
    e["gap (conventional − pressure rank)"] = (e.conventional_rank - e.pressure_rank).astype(int)
    e["player"] = e.player_id.map(shown)
    t = e[["pressure_rank", "player"] + cols + ["conventional_rank", "gap (conventional − pressure rank)", "player_id"]]
    return t, len(e)


# ------------------------------------------------------------------------------------------------ 3. Impact above replacement
def wp_per_run(d):
    H = d[d.H]
    return float((H.wp_o_1 - H.wp_o_0).mean()), float((H.wp_o_4 - H.wp_o_0).mean() / 4)


def impact(side, shown, wpr):
    r = pd.read_csv(C.TABLES / f"ratings_all_{side}.csv")
    min_h = int(pd.read_csv(C.TABLES / "rating_variants.csv").query("variant == @side + ': primary'").min_H.iat[0])
    just_below = r[(r.n_H >= min_h / 2) & (r.n_H < min_h) & (r.last_season >= 2025)]
    repl = float(just_below.rating.median())
    top = pd.read_csv(C.TABLES / ("top10_batters.csv" if side == "bat" else "top10_bowlers.csv"))
    x = r.set_index("player_id").loc[top.player_id]
    hpb = x.n_H / x.seasons
    runs = (x.rating - repl) * hpb
    out = pd.DataFrame({"rank": top["rank"].values, "player": top.player.values, "rating_per_100": (100 * x.rating).round(2).values,
                        "H_balls_per_season": hpb.round(1).values, "runs_above_replacement_per_season": runs.round(2).values,
                        "wins_above_replacement_per_season": (runs * wpr).round(3).values})
    info = {"min_H": min_h, "replacement_pool": f"{len(just_below)} players active in 2025–26 with {min_h // 2}–{min_h - 1} H balls",
            "replacement_rating_per_100": round(100 * repl, 2)}
    return out, info


# ------------------------------------------------------------------------------------------------ 4. Role charts
def role_charts(shown):
    for side, (xm, ym, xl, yl, fn) in {
        "bat": ("boundary_elevation_H", "wicket_security_H", "Boundary elevation on H (actual − expected boundary rate)",
                "Wicket security on H (expected − actual dismissal rate)", "role_chart_batters.png"),
        "bowl": ("wicket_taking_H", "economy_suppression_H", "Wicket-taking on H (actual − expected bowler-wicket rate)",
                 "Economy suppression on H (runs per 6 deliveries vs expected)", "role_chart_bowlers.png")}.items():
        m = pd.read_csv(C.TABLES / ("metrics_batters.csv" if side == "bat" else "metrics_bowlers.csv"))
        m = m[m.season == "career"]
        r = pd.read_csv(C.TABLES / f"ratings_all_{side}.csv")
        e = m[m.player_id.isin(r.player_id[r.eligible])]
        top = pd.read_csv(C.TABLES / ("top10_batters.csv" if side == "bat" else "top10_bowlers.csv")).head(10)
        fig, ax = plt.subplots(figsize=(8, 6.2))
        ax.scatter(e[xm], e[ym], s=16, color=SERIES[0], alpha=0.45, edgecolor="none", label="eligible players")
        tt = e[e.player_id.isin(top.player_id)]
        ax.scatter(tt[xm], tt[ym], s=44, color=SERIES[1], edgecolor="#fcfcfb", linewidth=1.2, zorder=3, label="final top 10")
        for _, rr in tt.iterrows():
            ax.annotate(shown[side][rr.player_id].split(" (")[0], (rr[xm], rr[ym]), fontsize=7.5, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color=TEXT2, lw=0.8)
        ax.axvline(0, color=TEXT2, lw=0.8)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(("Batters" if side == "bat" else "Bowlers") + " — role map on high-leverage deliveries (career)", fontsize=10.5)
        ax.legend(loc="lower left", fontsize=8.5)
        save(fig, C.CHARTS / fn)


# ------------------------------------------------------------------------------------------------ 5. Player cards
def player_scenarios(d, comp, side, pid):
    flags = d[KEY + [f"sc_{k}" for k in SCEN] + ["phase"]]
    x = comp[(comp.side == side) & (comp.metric == "rv") & (comp.player_id == pid) & comp.H].merge(flags, on=KEY, suffixes=("", "_d"))
    sc = []
    for k, lab in SCEN.items():
        if side == "bowl" and k in ("entry_under_pressure", "dot_squeeze"):
            continue
        if side == "bat" and k == "pp_vs_set_openers":
            continue
        z = x[x[f"sc_{k}"].astype(bool)]
        if len(z) >= MIN_SC:
            sc.append((lab, 100 * z.val.mean(), len(z)))
    ph = []
    for (inn, p), z in x.groupby(["innings", "phase"], observed=True):
        if len(z) >= MIN_SC:
            ph.append((f"innings {inn}, {p}", 100 * z.val.mean(), len(z)))
    return sorted(sc, key=lambda t: -t[1]), sorted(ph, key=lambda t: -t[1])


def cards(d, comp, shown, squad, impacts):
    import os
    os.makedirs(C.OUT / "cards", exist_ok=True)
    roles_b = pd.read_csv(C.TABLES / "roles_batters.csv").set_index("player_id")
    roles_w = pd.read_csv(C.TABLES / "roles_bowlers.csv").set_index("player_id")
    paths = []
    for side in ("bat", "bowl"):
        top_all = pd.read_csv(C.TABLES / ("top10_batters.csv" if side == "bat" else "top10_bowlers.csv"))
        top = top_all.head(10)
        r = pd.read_csv(C.TABLES / f"ratings_all_{side}.csv").set_index("player_id")
        imp = impacts[side].set_index("player")
        for _, t in top.iterrows():
            pid = t.player_id
            rr = r.loc[pid]
            sc, ph = player_scenarios(d, comp, side, pid)
            proc = [s.strip() for s in str(t.strongest_process_on_H).split(";")]
            weakest_proc = proc[-1] if proc else ""
            strong = f"{sc[0][0]} ({sc[0][1]:+.1f} per 100 balls, n={sc[0][2]})" if sc else "no scenario with ≥20 H deliveries"
            weak = f"{sc[-1][0]} ({sc[-1][1]:+.1f} per 100 balls, n={sc[-1][2]})" if len(sc) > 1 else "only one scenario with ≥20 H deliveries"
            best_ph = f"{ph[0][0]} ({ph[0][1]:+.1f} per 100 H balls, n={ph[0][2]})" if ph else "no phase with ≥20 H deliveries"
            role_line = (f"{roles_b.loc[pid, 'role']}; median batting position {roles_b.loc[pid, 'median_position']:.0f}; "
                         f"mean entry over {roles_b.loc[pid, 'mean_entry_over']:.1f}" if side == "bat" else
                         f"{roles_w.loc[pid, 'role']}; overs share PP/middle/death {roles_w.loc[pid, 'share_powerplay']:.0%}/"
                         f"{roles_w.loc[pid, 'share_middle']:.0%}/{roles_w.loc[pid, 'share_death']:.0%}")
            ia = imp.loc[t.player]
            L = [f"# {t.player} — {'batter' if side == 'bat' else 'bowler'} #{t['rank']} (tier {t.tier})\n",
                 f"- **Rating:** {t.rating_runs_per_100_H_balls:+.2f} runs per 100 high-leverage balls vs expectation (SE {t.se_per_100:.2f}; "
                 f"rank score {t.rank_score_per_100:+.2f}). Core skill {t.core_skill_per_100:+.2f}, pressure deviation {t.pressure_dev_per_100:+.2f}.",
                 f"- **Evidence:** {int(t.n_H)} H balls; {t.seasons_played}; raw H {t.raw_H_per_100:+.2f} ± {t.raw_H_se_per_100:.2f}; "
                 f"retention R {t.R:+.3f} {t.R_ci95} (pooled bootstrap).",
                 f"- **Role (data):** {role_line}. Latest team: {t.latest_team}." + (" **In RR's IPL 2026 squad.**" if pid in squad else ""),
                 f"- **Process on H (σ vs eligible players):** {t.strongest_process_on_H}.",
                 f"- **Strongest scenario:** {strong}. **Weakest scenario:** {weak}.",
                 f"- **Impact above replacement:** {ia.runs_above_replacement_per_season:+.2f} runs / {ia.wins_above_replacement_per_season:+.3f} wins per season "
                 f"({ia.H_balls_per_season:.0f} H balls per season).",
                 "", "**Analyst.** " + (f"Rated mainly on core skill ({t.core_skill_per_100:+.1f}); the H record moved the rating by "
                                         f"{t.pressure_dev_per_100:+.1f} (weight {t.k_update:.2f}). " + tier_text(top_all, t)),
                 "**Scout (verify on video).** " + (f"The weakest scenario — {weak} — and the lowest process signal ({weak_proc_text(weakest_proc)}). "
                                                    "Check whether the numbers reflect shot/ball execution or match situation."),
                 "**Coach (best role / entry point).** " + f"Highest H value in {best_ph}; current data role: {role_line.split(';')[0]}.",
                 "", "_All figures from the pipeline outputs (2022–26, rain-shortened matches excluded from pressure sets). "
                     "Scenario values are raw means over H deliveries in that scenario (not shrunk)._"]
            slug = re.sub(r"[^a-z0-9]+", "_", t.player.lower()).strip("_")
            path = C.OUT / "cards" / f"{side}_{int(t['rank']):02d}_{slug}.md"
            path.write_text("\n".join(L))
            paths.append(str(path.relative_to(C.ROOT)))
    return paths


def tier_text(top_all, t):
    same = top_all[top_all.tier == t.tier]["rank"]
    span = f"ranks {same.min()}–{same.max()}" if len(same) > 1 else f"rank {same.min()} alone"
    return (f"Tier {t.tier} ({span} of the top 20): positions within a tier are not statistically separable "
            "(adjacent rank scores within one SE).")


def weak_proc_text(s):
    return s if s else "n/a"


def run():
    d = deliveries()
    comp = RT.load()
    shown = {s: RT.names_and_teams(s, RT.global_duplicate_names()[0])[0] for s in ("bat", "bowl")}
    squad = rr_squad_2026()
    rr, best = rr_fit(d, comp, shown)
    rr.to_csv(C.TABLES / "rr_fit_scenarios.csv", index=False)
    best.to_csv(C.TABLES / "rr_fit_best_players.csv", index=False)
    in_squad = []
    for side in ("bat", "bowl"):
        top = pd.read_csv(C.TABLES / ("top10_batters.csv" if side == "bat" else "top10_bowlers.csv"))
        for _, t in top[top.player_id.isin(squad)].iterrows():
            in_squad.append({"side": side, "final_rank": int(t["rank"]), "player": t.player})
    in_squad = pd.DataFrame(in_squad, columns=["side", "final_rank", "player"])
    pv, imps, infos = {}, {}, {}
    w1, w4 = wp_per_run(d)
    for side in ("bat", "bowl"):
        t, n = pvg(side, shown[side])
        t.to_csv(C.TABLES / f"pressure_value_gap_{side}.csv", index=False)
        pv[side] = (t, n)
        imps[side], infos[side] = impact(side, shown[side], w1)
        imps[side].to_csv(C.TABLES / f"impact_above_replacement_{side}.csv", index=False)
    role_charts(shown)
    paths = cards(d, comp, shown, squad, imps)
    L = ["# Phase 9 — RR layer and outputs (lists LOCKED)\n", __doc__.strip(), "",
         "## 1. RR fit\n",
         "RR's high-leverage deliveries in matches since 2022 (rain-shortened excluded). Value = actual WP change − expected "
         "(player-agnostic outcome model), from RR's side; league = all other teams in the same role and scenario. Negative "
         "total = WP (wins) RR lost relative to a league-average team over these deliveries. Death defence = 'death overs, "
         "2nd innings' with RR bowling.\n", _md(rr),
         "\nStrongest final top-20 players in RR's three costliest scenario × role combinations (raw run value per 100 balls on "
         f"H deliveries in that scenario, ≥{MIN_SC} deliveries):\n", _md(best),
         "\nFinal top-20 players in RR's IPL 2026 squad (competition_team_player_mapping):\n", _md(in_squad) if len(in_squad) else "none",
         "\n## 2. Pressure Value Gap\n",
         "Pressure rank = rank by rank score among eligible players. Conventional rank = mean of the two conventional ranks "
         "among the same players (bat: average, strike rate; bowl: economy, wickets; all kept matches 2022–26). "
         "Gap > 0 = ranked higher on pressure than on conventional stats.\n"]
    for side, (t, n) in pv.items():
        nm = "Batters" if side == "bat" else "Bowlers"
        L += [f"**{nm}: final top 20** ({n} eligible)\n", _md(t.head(20).drop(columns=["player_id"])),
              f"\n**{nm}: largest positive gaps among all eligible players**\n",
              _md(t.sort_values("gap (conventional − pressure rank)", ascending=False).head(10).drop(columns=["player_id"])), ""]
    L += ["## 3. Impact above replacement\n",
          f"WP value of one run on H deliveries (mean WP(1 run) − WP(dot), batting side): {w1:.4f} (check via a four: {w4:.4f} per run) "
          f"→ about {1 / w1:.1f} runs per win on H deliveries.\n"]
    for side in ("bat", "bowl"):
        L += [f"**{'Batters' if side == 'bat' else 'Bowlers'}** — replacement: {infos[side]}\n", _md(imps[side]), ""]
    L += ["## 4. Role charts\n", "- `outputs/charts/role_chart_batters.png` (wicket security vs boundary elevation on H)",
          "- `outputs/charts/role_chart_bowlers.png` (economy suppression vs wicket-taking on H)", "",
          "## 5. Player cards\n"] + [f"- `{p}`" for p in paths]
    (C.OUT / "09_rr_outputs.md").write_text("\n".join(L))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(rr.to_string(index=False))
    print(best.to_string(index=False))
    print(in_squad.to_string(index=False))
    for side in ("bat", "bowl"):
        print(pv[side][0].head(20).to_string(index=False))
        print(pv[side][0].sort_values("gap (conventional − pressure rank)", ascending=False).head(8).to_string(index=False))
        print(infos[side])
        print(imps[side].to_string(index=False))
    print("WP per run on H", w1, w4, "runs per win", 1 / w1)
    print(len(paths), "cards")
