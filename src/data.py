"""Phase 1: load, reconcile, clean, rebuild match state, tag competition strength, write audit.

Outputs
  data_processed/balls.parquet    one row per ball (innings 1-2 of kept matches) with match state
  data_processed/matches.parquet  one row per match with flags, strength group, labels
  outputs/tables/team_strength_groups.csv, outputs/tables/reconciliation.csv
  outputs/01_data_audit.md
"""
import re
import numpy as np
import pandas as pd

from . import config as C

BALL_DTYPES = {
    "match_id": "int32", "innings": "int8", "overid": "int8", "over_ball_num": "int8",
    "innings_ball_num": "int16", "batting_team_id": "int32", "bowling_team_id": "int32",
    "striker_id": "int32", "non_striker_id": "int32", "bowler_id": "int32",
    "striker_name": "category", "non_striker_name": "category", "bowler_name": "category",
    "batting_type": "category", "bowling_type": "category", "batting_order": "int8",
    "runs": "int8", "dismissal_type_id": "float32", "dismissal_type": "category",
    "dismissed_player_id": "float64", "extras_type": "category", "extras1_runs": "int8",
    "extras_2_runs": "int8", "pace_or_spin": "category", "bowl_speed": "float32",
    "stroke": "category", "delivery_style": "category", "over_or_around": "category",
    "on_or_off": "category", "delivery_ist_time": "string",
}


def load_raw():
    balls = pd.read_csv(C.RAW / "ball_by_ball.csv", dtype=BALL_DTYPES)
    for c in ("is_four", "is_six", "is_wicket"):
        balls[c] = balls[c].astype(bool)
    matches = pd.read_csv(C.RAW / "matches.csv", parse_dates=["match_date"])
    comps = pd.read_csv(C.RAW / "competitions.csv")
    xi = pd.read_csv(C.RAW / "playing_eleven.csv")
    return balls, matches, comps, xi


# ---------------------------------------------------------------- result parsing
_RUNS = re.compile(r"won\s*(?:by)?\s*(\d+)\s*run", re.I)
_WKTS = re.compile(r"won\s*(?:by)?\s*(\d+)\s*(?:wicket|wkt)", re.I)


def parse_result(s):
    s = str(s)
    low = s.lower()
    if "no result" in low:
        return "no_result", np.nan
    if "abandon" in low:
        return "abandoned", np.nan
    if "tied" in low or "super over" in low or "eliminator" in low:
        return "tie", np.nan
    m = _RUNS.search(s)
    if m:
        return "runs", int(m.group(1))
    m = _WKTS.search(s)
    if m:
        return "wkts", int(m.group(1))
    return "unparsed", np.nan


# ---------------------------------------------------------------- per-ball definitions
def add_ball_definitions(b):
    et = b["extras_type"].astype("string").fillna("None")
    b["is_wide"] = et.eq(C.WIDE)
    b["is_noball"] = et.isin(C.NOBALL_TYPES)
    b["is_penalty"] = et.isin(C.NOT_A_DELIVERY)
    b["is_bye"] = et.isin(C.BYE_TYPES)
    b["legal"] = ~(b.is_wide | b.is_noball | b.is_penalty)
    b["ball_faced"] = b.legal | b.is_noball          # wides are not balls faced
    b["bowled_ball"] = ~b.is_penalty                  # a delivery by the bowler
    b["batter_runs"] = b["runs"].astype("int8")
    b["total_runs"] = (b.runs + b.extras1_runs + b.extras_2_runs).astype("int8")
    # Bowler is charged: bat runs + wide runs + 1 no-ball penalty. Not byes/leg byes
    # (incl. byes off a no-ball in extras_2_runs) and not penalty runs.
    charged_extra = np.where(b.is_wide | b.is_noball, b.extras1_runs, 0)
    b["bowler_runs"] = (b.runs + charged_extra).astype("int8")
    dt = b["dismissal_type"].astype("string")
    b["bowler_wicket"] = b.is_wicket & dt.isin(C.BOWLER_WICKETS).fillna(False)
    b["bowl_speed"] = b["bowl_speed"].where(b.bowl_speed > 0)
    return b


# ---------------------------------------------------------------- match table
def build_matches(b, matches, comps):
    m = matches.merge(comps[["comp_id", "comp_name", "match_format", "season"]], on="comp_id", how="left")
    m["is_ipl"] = m.comp_name.str.startswith("IPL")
    m["year"] = m.match_date.dt.year.astype("int16")
    p = m.result.map(parse_result)
    m["result_kind"] = p.str[0]
    m["margin"] = p.str[1]

    # strength group
    home_fm = m.home_team.isin(C.FULL_MEMBERS)
    away_fm = m.away_team.isin(C.FULL_MEMBERS)
    m["strength"] = np.select([m.is_ipl, home_fm & away_fm], ["IPL", "FM_T20I"], "Weaker_T20I")

    # innings totals from balls (innings 1-2)
    bb = b[b.innings <= 2]
    agg = bb.groupby(["match_id", "innings"]).agg(
        tot=("total_runs", "sum"), wk=("is_wicket", "sum"), legal=("legal", "sum"),
        bat_team=("batting_team_id", "first"))
    agg = agg.unstack("innings")
    agg.columns = [f"{a}{i}" for a, i in agg.columns]
    m = m.merge(agg, left_on="match_id", right_index=True, how="left")
    m["has_balls"] = m.tot1.notna()
    m["has_super_over"] = m.match_id.isin(b.loc[b.innings >= 3, "match_id"].unique())

    # max balls per innings (revised fields are in balls, e.g. 114 = 19 overs)
    m["max_balls1"] = m.first_inn_revised_overs.fillna(120)
    m["max_balls2"] = m.second_inn_revised_overs.fillna(m.first_inn_revised_overs).fillna(120)
    m["target"] = m.revised_target.fillna(m.tot1 + 1)
    m["revised_recorded"] = m[["first_inn_revised_overs", "second_inn_revised_overs", "revised_target"]].notna().any(axis=1)

    # reconciliation checks
    m["recon_ok"] = np.nan
    runs = m.result_kind.eq("runs") & ~m.revised_recorded & m.has_balls
    m.loc[runs, "recon_ok"] = ((m.tot1 - m.tot2) == m.margin)[runs].astype(float)
    wk = m.result_kind.eq("wkts") & ~m.revised_recorded & m.has_balls
    m.loc[wk, "recon_ok"] = ((m.tot2 >= m.tot1 + 1) & (m.tot2 - m.tot1 <= 6) & ((10 - m.wk2) == m.margin))[wk].astype(float)
    m["dls_text"] = m.result.str.contains(r"D/?LS?|DLS", case=False, regex=True)

    # Unrecorded rain truncation: an innings at least one over short that was not (effectively) all out.
    # 9 wickets + innings over early = all out with an absent / retired-hurt batter, not rain.
    # 1 ball short = a missing ball row (data gap), not rain.
    trunc1 = (m.legal1 <= m.max_balls1 - 6) & (m.wk1 < 9)
    trunc2 = (m.legal2 <= m.max_balls2 - 6) & (m.wk2 < 9) & (m.tot2 < m.target)
    decided = m.result_kind.isin(["runs", "wkts", "tie"])
    m["revised_unrecorded"] = decided & ~m.revised_recorded & m.has_balls & (trunc1 | trunc2 | m.dls_text)
    # Where a non-DLS game was shortened before it began (both innings equal length), infer that length.
    infer = m.revised_unrecorded & ~m.dls_text & trunc1 & (m.legal1 == m.legal2.where(m.wk2 < 9, m.legal1))
    infer |= m.revised_unrecorded & ~m.dls_text & trunc1 & (m.result_kind == "wkts")
    m["max_balls_inferred"] = infer
    m.loc[infer, "max_balls1"] = m.loc[infer, "legal1"]
    m.loc[infer, "max_balls2"] = m.loc[infer, "legal1"]
    short_by_one = ((m.legal1 == m.max_balls1 - 1) & (m.wk1 < 10)) | ((m.legal2 == m.max_balls2 - 1) & (m.wk2 < 10) & (m.tot2 < m.target))
    m["data_gap_one_ball"] = m.has_balls & short_by_one & ~m.revised_unrecorded

    # exclusions and training eligibility
    m["exclude_reason"] = np.select(
        [m.result_kind.eq("no_result"), m.result_kind.eq("abandoned"), ~m.has_balls,
         m.tot2.isna()],
        ["no_result", "abandoned", "no_ball_data", "missing_2nd_innings"], "")
    m["kept"] = m.exclude_reason.eq("")
    m["is_tie"] = m.result_kind.eq("tie")
    m["wp_train_ok"] = m.kept & ~m.is_tie & ~m.revised_recorded & ~m.revised_unrecorded & (m.recon_ok != 0)
    return m


def team_groups(m):
    rows = []
    for side in ("home", "away"):
        rows.append(m[[f"{side}_team_id", f"{side}_team", "is_ipl"]].set_axis(["team_id", "team_name", "is_ipl"], axis=1))
    t = pd.concat(rows).drop_duplicates(["team_id", "team_name"])
    t["group"] = np.where(t.is_ipl, "IPL franchise",
                  np.where(t.team_name.isin(C.FULL_MEMBERS), "Full member", "Weaker (associate / A side)"))
    t = t.groupby(["team_id", "team_name", "group"], as_index=False).size().drop(columns="size")
    return t.sort_values(["group", "team_name"]).reset_index(drop=True)


# ---------------------------------------------------------------- match state
def phase_of(overid):
    return pd.Categorical(np.select([overid <= 6, overid <= 15], ["powerplay", "middle"], "death"),
                          categories=["powerplay", "middle", "death"])


def add_state(b, m):
    b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
    g = b.groupby(["match_id", "innings"], sort=False)
    b["score_before"] = (g.total_runs.cumsum() - b.total_runs).astype("int16")
    b["wkts_before"] = (g.is_wicket.cumsum() - b.is_wicket).astype("int8")
    b["legal_before"] = (g.legal.cumsum() - b.legal).astype("int16")
    b["score_after"] = (b.score_before + b.total_runs).astype("int16")
    b["wkts_after"] = (b.wkts_before + b.is_wicket).astype("int8")
    b["legal_after"] = (b.legal_before + b.legal).astype("int16")

    mm = m.set_index("match_id")
    inn1 = b.innings.eq(1)
    max_balls = np.where(inn1, b.match_id.map(mm.max_balls1), b.match_id.map(mm.max_balls2))
    b["max_balls"] = max_balls.astype("int16")
    b["balls_left"] = (b.max_balls - b.legal_before).clip(lower=0).astype("int16")
    tgt = b.match_id.map(mm.target)
    b["target"] = np.where(inn1, np.nan, tgt).astype("float32")
    b["runs_needed"] = (b.target - b.score_before).astype("float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        b["rrr"] = np.where(b.balls_left > 0, b.runs_needed * 6 / b.balls_left, np.nan).astype("float32")
        b["crr"] = np.where(b.legal_before > 0, b.score_before * 6 / b.legal_before, np.nan).astype("float32")
    b["phase"] = phase_of(b.overid.values)

    for col in ("year", "strength", "venue_id", "match_date", "is_ipl", "knock_out", "is_tie",
                "wp_train_ok", "revised_recorded", "revised_unrecorded"):
        b[col] = b.match_id.map(mm[col])
    b["strength"] = b.strength.astype("category")
    win = b.match_id.map(mm.win_team_id)
    b["bat_win"] = np.where(b.is_tie | win.isna(), np.nan, (win == b.batting_team_id).astype(float)).astype("float32")
    return b


# ---------------------------------------------------------------- audit
def _md(df, index=False):
    """Markdown table without the optional `tabulate` dependency."""
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        rows.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(rows)


def data_problems(b_all, b, m, xi):
    probs = {}
    over_legal = b.groupby(["match_id", "innings", "overid"]).legal.sum()
    probs["overs with >6 legal balls"] = over_legal[over_legal > 6].reset_index().to_dict("records")
    probs["overs bowled by >1 bowler (injury/mid-over change)"] = int(
        (b.groupby(["match_id", "innings", "overid"]).bowler_id.nunique() > 1).sum())
    w = b[b.is_wicket]
    probs["wickets where dismissed player is neither batter"] = int(
        ((w.dismissed_player_id != w.striker_id) & (w.dismissed_player_id != w.non_striker_id)).sum())
    probs["duplicate (match, innings, innings_ball_num)"] = int(b_all.duplicated(["match_id", "innings", "innings_ball_num"]).sum())
    # striker / bowler in the playing XI of the right team
    keys = set(zip(xi.match_id, xi.team_id, xi.player_id))
    s_ok = pd.Series([k in keys for k in zip(b.match_id, b.batting_team_id, b.striker_id)])
    bw_ok = pd.Series([k in keys for k in zip(b.match_id, b.bowling_team_id, b.bowler_id)])
    probs["balls whose striker is not in batting XI"] = int((~s_ok).sum())
    probs["balls whose bowler is not in bowling XI"] = int((~bw_ok).sum())
    probs["bowl_speed missing (0) share"] = round(float(b.bowl_speed.isna().mean()), 3)
    probs["delivery_style missing share"] = round(float(b.delivery_style.isna().mean()), 4)
    probs["stroke missing share"] = round(float(b.stroke.isna().mean()), 4)
    return probs


def write_audit(b_all, b, m, teams, probs):
    L = []
    L.append("# Phase 1 — Data audit\n")
    L.append("Generated by `src/data.py`. Raw CSVs are read-only; nothing in them was modified.\n")

    L.append("## 1. Counts\n")
    L.append(f"- Ball rows in file: {len(b_all):,} (innings 1–2: {(b_all.innings <= 2).sum():,}; super-over innings 3–4: {(b_all.innings >= 3).sum():,})")
    L.append(f"- Matches: {len(m):,}; with ball data: {int(m.has_balls.sum()):,}")
    ct = m.groupby(["year", "strength"]).size().unstack(fill_value=0)
    L.append("\nMatches by calendar year and competition group:\n")
    L.append(_md(ct, index=True))
    comp_t20 = m[m.match_format == "T20"].groupby("comp_name").size().rename("matches").reset_index()
    L.append("\n**Note — `match_format = T20` is not only IPL.** It also contains two A-team tournaments, so IPL is "
             "identified by competition name (`IPL 20xx`):\n")
    L.append(_md(comp_t20))
    L.append(f"\nIPL matches: {int(m.is_ipl.sum())} (the brief said 401; 401 is all `T20`-format matches including the A-team events). "
             f"RR (team_id 3) matches: {int(((m.home_team_id == 3) | (m.away_team_id == 3)).sum())}.\n")

    L.append("## 2. What the run columns mean (verified by reconciliation)\n")
    rk = m[m.result_kind.isin(["runs", "wkts"]) & ~m.revised_recorded & m.has_balls]
    ok = rk.recon_ok.astype(bool)
    L.append(f"Hypothesis tested: ball total = `runs + extras1_runs + extras_2_runs`. Checked against `result` for all "
             f"decided matches without recorded revisions: **{ok.sum()} / {len(ok)} reconcile** "
             f"(runs margin exact; wickets: chase total 1–6 above first innings and wickets lost = 10 − margin). "
             f"Dropping `extras_2_runs` reconciles only ~91.5% of runs-margin matches, so it is real runs.\n")
    bad = rk[~ok][["match_id", "result", "tot1", "tot2", "wk2", "margin"]]
    L.append("Non-reconciling matches (all are DLS/rain games whose revision is not recorded in matches.csv, or one data error):\n")
    L.append(_md(bad))
    L.append("""
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
""")
    L.append("Dismissal types in the data:\n")
    L.append(_md(b_all.dismissal_type.value_counts().rename_axis("dismissal_type").reset_index()))

    L.append("\n## 3. Exclusions\n")
    ex = m[~m.kept].groupby("exclude_reason").size().rename("matches").reset_index()
    L.append(_md(ex))
    L.append(f"\n- Super-over innings (3–4) dropped: {(b_all.innings >= 3).sum()} balls in {int(m.has_super_over.sum())} matches.")
    L.append(f"- Kept matches: {int(m.kept.sum())}; kept balls (innings 1–2): {len(b):,}.")
    L.append(f"- **Tied matches** ({int((m.kept & m.is_tie).sum())}): kept for player analysis; win label = undefined, so excluded from WP training.")
    L.append(f"- **Rain-revised, recorded** ({int((m.kept & m.revised_recorded).sum())}): kept for player analysis. Match state uses revised "
             "balls (the revised fields are in balls, e.g. 114 = 19 overs) and `revised_target`. Excluded from WP training (cleaner labels); "
             "still scored by the WP model using the revised balls/target.")
    unrec = m[m.kept & m.revised_unrecorded][["match_id", "result", "legal1", "wk1", "legal2", "wk2", "tot1", "tot2",
                                               "dls_text", "max_balls_inferred", "max_balls1"]]
    L.append(f"- **Rain-shortened but NOT recorded in matches.csv** ({len(unrec)}): detected from DLS text, or an innings ending ≥1 over early "
             "with ≤8 wickets down and the chase not complete (an innings ending early at 9 down is treated as all out with an absent/retired-hurt "
             "batter). Where a non-DLS game was shortened before it began, the shortened length is inferred from the balls bowled "
             "(`max_balls_inferred`); for DLS games the revised target is unknown, so 20-over state is used. All kept for player "
             "analysis, all excluded from WP training.\n")
    L.append(_md(unrec))
    gap = m[m.kept & m.data_gap_one_ball][["match_id", "result", "legal1", "legal2", "recon_ok"]]
    L.append(f"\n- **One missing ball row** ({len(gap)} matches; innings 1 legal ball short of full length, not all out, not rain). "
             "Kept; excluded from WP training only if they also fail reconciliation.\n")
    L.append(_md(gap))
    L.append(f"\nWP training matches: {int(m.wp_train_ok.sum())}.\n")

    L.append("## 4. Competition strength groups\n")
    L.append("Match-level: **IPL** (competition name IPL 20xx); **FM_T20I** = both teams ICC full members; "
             "**Weaker_T20I** = any associate or A side involved (incl. the two A-team Asia Cup events). "
             "Full list in `outputs/tables/team_strength_groups.csv`.\n")
    for grp, d in teams.groupby("group"):
        L.append(f"- **{grp}** ({len(d)}): " + ", ".join(sorted(d.team_name)))
    L.append("\nMatches per group (kept): " + ", ".join(f"{k} {v}" for k, v in m[m.kept].strength.value_counts().items()))

    L.append("\n## 5. Match state\n")
    L.append("Per ball (state *before* the ball): score, wickets, legal balls bowled, legal balls remaining (vs revised balls), "
             "target (revised if recorded, else first-innings total + 1), runs needed, required rate, current rate, "
             "phase by over: powerplay 1–6, middle 7–15, death 16–20 (by over number even in shortened matches).\n")

    L.append("## 6. Data problems found\n")
    for k, v in probs.items():
        L.append(f"- {k}: {v}")
    L.append("\n- Knockout matches: " + str(int(m.knock_out.sum())) + " (small sample).")
    (C.OUT / "01_data_audit.md").write_text("\n".join(L))


# ---------------------------------------------------------------- main
def run():
    b_all, matches, comps, xi = load_raw()
    b_all = add_ball_definitions(b_all)
    m = build_matches(b_all, matches, comps)
    teams = team_groups(m)
    teams.to_csv(C.TABLES / "team_strength_groups.csv", index=False)
    m[["match_id", "result", "result_kind", "margin", "tot1", "tot2", "wk1", "wk2", "recon_ok",
       "revised_recorded", "revised_unrecorded", "exclude_reason"]].to_csv(C.TABLES / "reconciliation.csv", index=False)

    b = b_all[(b_all.innings <= 2) & b_all.match_id.isin(m.loc[m.kept, "match_id"])].copy()
    b = add_state(b, m)
    probs = data_problems(b_all, b, m, xi)
    write_audit(b_all, b, m, teams, probs)

    b.drop(columns=["delivery_ist_time", "dismissal_type_id"]).to_parquet(C.PROC / "balls.parquet", index=False)
    m.to_parquet(C.PROC / "matches.parquet", index=False)

    print(f"[phase1] kept {int(m.kept.sum())} matches, {len(b):,} balls; "
          f"WP-train matches {int(m.wp_train_ok.sum())}; reconciliation "
          f"{int(m.recon_ok.fillna(0).sum())}/{int(m.recon_ok.notna().sum())}")
    print(teams.groupby("group").team_name.apply(lambda s: ", ".join(sorted(s))).to_string())
    return b, m
