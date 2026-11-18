"""Phase 5: pressure metrics per player (H and L), per season and career.

Every metric is built from a per-delivery component = actual − expected, signed so positive = good.
Expected values come from the Phase 4 opposition-aware outcome models (batter side: context + bowler rating;
bowler side: context + batter rating), which are the player-agnostic outcome model plus opposition quality,
scored out-of-fold by match. Rates use legal deliveries (class probabilities renormalised over legal outcomes).

The long component table (data_processed/metric_components.parquet) keeps one row per (delivery, player, metric)
so Phase 6 can split by match or season freely.
"""
import numpy as np
import pandas as pd

from . import config as C
from . import leverage as LV
from .plotstyle import plt, save, SERIES, TEXT2

O = LV.OUTCOMES                                         # 0 1 2 3 4 6 wicket wide_nb
KEY = ["match_id", "innings", "innings_ball_num"]
YORKERS = {"Yorker", "Wide Yorker"}
EXEC_ERRORS = {"Full Toss", "Beamer"}                   # plus wides and no-balls
ATTACK = {"Slog", "Slog Sweep", "Pull shot", "Hook Shot", "Paddle Scoop", "Reverse Paddle Scoop", "Reverse Sweep",
          "Switch", "UpperCut", "Inside out", "PickUp", "Sweep Shot", "Paddle Sweep", "Straight Drive", "Cover Drive",
          "Off Drive", "On drive", "Square drive", "Square Cut", "Late Cut"}
DEFEND = {"Frontfoot defensive", "Backfoot defensive", "Defense", "Push", "Intentional Padding"}
# everything else (Worked, Flick, Leg Glance, Glance, Steer) = rotation
COLLAPSE_WP = 0.35
MIN_REPORT = {"skill": 100, "temperament": 30, "yorker": 20, "variation": 60, "partner": 100}

# LOCKED candidate list for Phases 6-7 (DECISIONS.md M1-M7). Column used for ranking in brackets.
CANDIDATES = {
    "bat": ["pressure_v", "sr_delta", "dot_suppression", "boundary_elevation", "wicket_security",
            "squeeze_response[pooled]", "entry_resilience", "collapse_resistance", "post_wicket_response",
            "runout_avoidance", "partner_effect[on trial]"],
    "bowl": ["pressure_v", "economy_suppression", "dot_rate", "wicket_taking", "boundary_suppression",
             "bounce_back", "post_wicket_response", "execution_drift", "variation"],
}
REPORT_ONLY = {"bat": ["attacking_stroke_share"], "bowl": ["yorker_execution"]}

BAT_METRICS = {  # name: (kind, description)
    "pressure_v": ("value", "mean v (Phase 4) — P on H, B on L"),
    "sr_delta": ("skill", "strike rate vs expected (runs per 100 legal balls)"),
    "dot_suppression": ("skill", "expected − actual dot-ball rate (legal)"),
    "boundary_elevation": ("skill", "actual − expected boundary rate (legal)"),
    "wicket_security": ("skill", "expected − actual bowler-credited dismissal rate (legal)"),
    "squeeze_response": ("temperament", "mean v on balls faced after 3+ consecutive dots (ranked on H+L pooled)"),
    "entry_resilience": ("temperament", "mean v over first 10 balls faced when arriving in H"),
    "collapse_resistance": ("temperament", "mean v over innings begun with team WP < 35%"),
    "post_wicket_response": ("temperament", "mean v over the 12 deliveries after a wicket"),
    "runout_avoidance": ("temperament", "expected − actual run-out dismissal rate while at the crease"),
    "attacking_stroke_share": ("profile", "share of attacking strokes (profile only, not for ranking)"),
}
BOWL_METRICS = {
    "pressure_v": ("value", "mean v (Phase 4) — P on H, B on L"),
    "economy_suppression": ("skill", "expected − actual runs charged per 6 deliveries"),
    "dot_rate": ("skill", "actual − expected dot-ball rate (legal)"),
    "wicket_taking": ("skill", "actual − expected bowler-wicket rate (legal)"),
    "boundary_suppression": ("skill", "expected − actual boundary rate (legal)"),
    "bounce_back": ("temperament", "mean v on the next legal ball after conceding a 4 or 6"),
    "post_wicket_response": ("temperament", "mean v over the 12 deliveries after a wicket"),
    "execution_error": ("temperament", "expected − actual wide/no-ball/full-toss/beamer rate (drift = H − L)"),
    "yorker_execution": ("report_only", "mean v on yorker / wide-yorker deliveries (report only: label depends on outcome)"),
}


def load():
    b = pd.read_parquet(C.PROC / "balls.parquet")
    lev = pd.read_parquet(C.PROC / "leverage.parquet")
    val = pd.read_parquet(C.PROC / "values.parquet")
    wp = pd.read_parquet(C.PROC / "wp_preds.parquet")[KEY + ["wp_before"]]
    d = lev[KEY + ["LI", C.PRIMARY_H, "H20_within", "H33_within", "sc_post_wicket", "sc_entry_under_pressure",
                   "sc_dot_squeeze"]].merge(b, on=KEY).merge(val, on=KEY).merge(wp, on=KEY)
    d = d.sort_values(["match_date", "match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
    d["H"] = d[C.PRIMARY_H].astype(bool)
    return d, b


def probs(d, side):
    P = d[[f"p{side}_{o}" for o in O]].values.astype(float)
    P = P / P.sum(1, keepdims=True)
    PL = P[:, :7] / P[:, :7].sum(1, keepdims=True)      # legal outcomes only
    return P, PL


def components(d):
    """Return the long component table and per-delivery helper columns."""
    cls = LV.outcome_class(d)
    runs_emp = pd.Series(d.total_runs.values).groupby(cls).mean().reindex(range(8)).fillna(0).values
    legal = d.legal.values
    k_bat = float((d.total_runs - d.batter_runs)[legal].mean())        # byes/leg byes per legal ball
    k_bowl = float((d.total_runs - d.bowler_runs).mean())               # runs not charged to the bowler
    wk = d.is_wicket.values & legal
    q = float(d.bowler_wicket.values[wk].mean())                        # share of legal wickets credited to bowler
    consts = {"runs_per_class": dict(zip(O, runs_emp.round(3))), "byes_per_legal_ball": round(k_bat, 4),
              "uncharged_runs_per_delivery": round(k_bowl, 4), "bowler_credited_share_of_wickets": round(q, 4)}

    dot = legal & (cls == 0)
    bnd = legal & (d.is_four.values | d.is_six.values)
    bw = d.bowler_wicket.values & legal
    striker_bw = bw & (d.dismissed_player_id.values == d.striker_id.values)

    parts = []

    def add(side, pid, mask, metric, val):
        m = np.asarray(mask, bool)
        parts.append(pd.DataFrame({"side": side, "player_id": np.asarray(pid)[m].astype("int32"),
                                   "match_id": d.match_id.values[m].astype("int32"), "year": d.year.values[m].astype("int16"),
                                   "match_date": d.match_date.values[m], "innings": d.innings.values[m].astype("int8"),
                                   "innings_ball_num": d.innings_ball_num.values[m].astype("int16"),
                                   "overid": d.overid.values[m].astype("int8"),
                                   "LI": d.LI.values[m].astype("float32"),
                                   "H": d.H.values[m], "metric": metric, "val": np.asarray(val, float)[m].astype("float32")}))

    # ---------------- batters (balls faced = legal or no-ball; rates on legal)
    _, PLb = probs(d, "bat")
    faced = d.ball_faced.values & ~d.is_wide.values
    sid = d.striker_id.values
    add("bat", sid, faced, "pressure_v", d.v_bat.values)
    exp_runs_bat = PLb @ runs_emp[:7] - k_bat
    add("bat", sid, legal, "sr_delta", 100 * (d.batter_runs.values - exp_runs_bat))
    add("bat", sid, legal, "dot_suppression", PLb[:, 0] - dot)
    add("bat", sid, legal, "boundary_elevation", bnd - (PLb[:, 4] + PLb[:, 5]))
    add("bat", sid, legal, "wicket_security", q * PLb[:, 6] - striker_bw)
    add("bat", sid, faced & d.sc_dot_squeeze.values, "squeeze_response", d.v_bat.values)
    add("bat", sid, faced & d.sc_entry_under_pressure.values, "entry_resilience", d.v_bat.values)
    # collapse resistance: the batter's innings began (first ball faced) with team WP < 35%
    g = d[faced].groupby(["match_id", "innings", "striker_id"])
    start_wp = g.wp_before.transform("first")
    col = np.zeros(len(d), bool)
    col[np.where(faced)[0]] = (start_wp.values < COLLAPSE_WP)
    add("bat", sid, faced & col, "collapse_resistance", d.v_bat.values)
    add("bat", sid, faced & d.sc_post_wicket.values, "post_wicket_response", d.v_bat.values)
    # run-out avoidance: both batters at the crease on every delivery; expected = league rate per innings x phase
    ro_out = d.is_wicket.values & (d.dismissal_type.astype(str).values == "RunOut")
    cell = d.innings.astype(str) + "|" + d.phase.astype(str)
    for who in ("striker_id", "non_striker_id"):
        mine = ro_out & (d.dismissed_player_id.values == d[who].values)
        exp = pd.Series(np.r_[ro_out & (d.dismissed_player_id.values == d.striker_id.values)].astype(float) +
                        (ro_out & (d.dismissed_player_id.values == d.non_striker_id.values)).astype(float)).groupby(cell.values).transform("mean").values / 2
        add("bat", d[who].values, np.ones(len(d), bool), "runout_avoidance", exp - mine)
    stroke = d.stroke.astype(str).values
    known = d.stroke.notna().values & legal
    add("bat", sid, known, "attacking_stroke_share", np.isin(stroke, list(ATTACK)).astype(float))
    cost = d.wkt_cost_runs.values.astype(float)
    # TARGET measure (P6-2), not a candidate: run value per legal ball faced, runs above expected + wicket value
    add("bat", sid, legal, "rv", (d.batter_runs.values - exp_runs_bat) - cost * (striker_bw - q * PLb[:, 6]))
    # rv split into components (Phase 7 component-wise rating): rv = rv_runs + rv_wkt
    add("bat", sid, legal, "rv_runs", d.batter_runs.values - exp_runs_bat)
    add("bat", sid, legal, "rv_wkt", -cost * (striker_bw - q * PLb[:, 6]))

    # ---------------- bowlers (all deliveries; rates on legal)
    Pw, PLw = probs(d, "bowl")
    bid = d.bowler_id.values
    alld = np.ones(len(d), bool)
    add("bowl", bid, alld, "pressure_v", d.v_bowl.values)
    add("bowl", bid, alld, "economy_suppression", 6 * ((Pw @ runs_emp - k_bowl) - d.bowler_runs.values))
    add("bowl", bid, legal, "dot_rate", dot - PLw[:, 0])
    add("bowl", bid, legal, "wicket_taking", bw - q * PLw[:, 6])
    add("bowl", bid, legal, "boundary_suppression", (PLw[:, 4] + PLw[:, 5]) - bnd)
    # bounce-back: first legal ball by the same bowler (same innings) after a delivery that went for 4 or 6
    o = d.sort_values(["match_id", "innings", "bowler_id", "innings_ball_num"])
    pos = o.groupby(["match_id", "innings", "bowler_id"]).cumcount().values
    bpos = pd.Series(np.where(o.is_four.values | o.is_six.values, pos, np.nan), index=o.index)
    lpos = pd.Series(np.where(o.legal.values, pos, np.nan), index=o.index)
    grp = [o.match_id, o.innings, o.bowler_id]
    prev_b = bpos.groupby(grp).shift(1).groupby(grp).ffill()
    prev_l = lpos.groupby(grp).shift(1).groupby(grp).ffill()
    trig = o.legal.values & prev_b.notna().values & (prev_b.fillna(-1).values >= prev_l.fillna(-1).values)
    bb = np.zeros(len(d), bool)
    bb[o.index.values[trig]] = True
    add("bowl", bid, bb, "bounce_back", d.v_bowl.values)
    add("bowl", bid, d.sc_post_wicket.values, "post_wicket_response", d.v_bowl.values)
    # execution error vs a context expectation (league rate by innings x phase x pace/spin)
    style = d.delivery_style.astype(str).values
    err = (~legal & ~d.is_penalty.values) | np.isin(style, list(EXEC_ERRORS))
    ecell = cell + "|" + d.pace_or_spin.astype(str)
    exp_err = pd.Series(err.astype(float)).groupby(ecell.values).transform("mean").values
    add("bowl", bid, alld, "execution_error", exp_err - err)
    # TARGET measure (P6-2): bowler run value per delivery, runs saved vs expected + wicket value
    add("bowl", bid, alld, "rv", ((Pw @ runs_emp - k_bowl) - d.bowler_runs.values)
        + cost * (d.bowler_wicket.values - q * Pw[:, 6]))
    add("bowl", bid, alld, "rv_runs", (Pw @ runs_emp - k_bowl) - d.bowler_runs.values)
    add("bowl", bid, alld, "rv_wkt", cost * (d.bowler_wicket.values - q * Pw[:, 6]))
    yk = np.isin(style, list(YORKERS))
    add("bowl", bid, yk, "yorker_execution", d.v_bowl.values)

    comp = pd.concat(parts, ignore_index=True)
    comp["side"] = comp.side.astype("category")
    comp["metric"] = comp.metric.astype("category")
    return comp, consts, err


def aggregate(comp, by):
    g = comp.groupby(["side", "player_id", "metric"] + by + ["H"], observed=True)
    a = g.agg(value=("val", "mean"), n=("val", "size"), matches=("match_id", "nunique")).reset_index()
    a["set"] = np.where(a.H, "H", "L")
    w = a.pivot_table(index=["side", "player_id"] + by, columns=["metric", "set"], values=["value", "n", "matches"],
                      observed=True)
    w.columns = [f"{m}_{s}" if v == "value" else f"{m}_{v}_{s}" for v, m, s in w.columns]
    return w.reset_index()


def entropy_table(d, by):
    """Shannon entropy (bits, Miller–Madow corrected) of delivery_style on H and on L per bowler."""
    x = d[d.delivery_style.notna()].assign(style=lambda t: t.delivery_style.astype(str))
    rows = []
    for keys, t in x.groupby(["bowler_id"] + by):
        rec = dict(zip(["bowler_id"] + by, keys if isinstance(keys, tuple) else (keys,)))
        for s, tt in (("H", t[t.H]), ("L", t[~t.H])):
            n = len(tt)
            if n == 0:
                rec[f"variation_entropy_{s}"], rec[f"variation_n_{s}"] = np.nan, 0
                continue
            p = tt["style"].value_counts().values / n
            k = (p > 0).sum()
            rec[f"variation_entropy_{s}"] = float(-(p * np.log2(p)).sum() + (k - 1) / (2 * n * np.log(2)))
            rec[f"variation_n_{s}"] = n
        rows.append(rec)
    e = pd.DataFrame(rows)
    ok = e.variation_n_H >= MIN_REPORT["variation"]
    e["variation_diff"] = np.where(ok, e.variation_entropy_H - e.variation_entropy_L, np.nan)
    return e.rename(columns={"bowler_id": "player_id"})


def partner_table(d, by):
    """ON TRIAL. Partner's mean v (on strike, H) with this batter as non-striker vs without him."""
    x = d[d.H & d.ball_faced & ~d.is_wide][["striker_id", "non_striker_id", "v_bat"] + by]
    tot = x.groupby(["striker_id"] + by).v_bat.agg(["sum", "size"])
    pair = x.groupby(["striker_id", "non_striker_id"] + by).v_bat.agg(["sum", "size"]).reset_index()
    pair = pair.merge(tot.rename(columns={"sum": "tsum", "size": "tn"}).reset_index(), on=["striker_id"] + by)
    pair = pair[pair.tn > pair["size"]]
    pair["with"] = pair["sum"] / pair["size"]
    pair["without"] = (pair.tsum - pair["sum"]) / (pair.tn - pair["size"])
    pair["diff_w"] = (pair["with"] - pair["without"]) * pair["size"]
    out = pair.groupby(["non_striker_id"] + by).agg(partner_effect_H=("diff_w", "sum"), partner_n_H=("size", "sum"),
                                                   partner_partners=("striker_id", "nunique")).reset_index()
    out["partner_effect_H"] = out.partner_effect_H / out.partner_n_H
    return out.rename(columns={"non_striker_id": "player_id"})


def split_half(comp, side, metric, min_n):
    """Odd vs even matches (by date order within player); Pearson across players; Spearman–Brown."""
    c = comp[(comp.side == side) & (comp.metric == metric)]
    out = {}
    for s, h in (("P", True), ("B", False)):
        x = c[c.H == h]
        mo = x.groupby("player_id").match_id.transform(lambda m: m.rank(method="dense").astype(int) % 2)
        g = x.assign(half=mo.values).groupby(["player_id", "half"]).val.agg(["mean", "size"]).unstack()
        g = g[(g[("size", 0)] >= min_n / 2) & (g[("size", 1)] >= min_n / 2)]
        r = np.corrcoef(g[("mean", 0)], g[("mean", 1)])[0, 1]
        out[s] = {"players": len(g), "half_r": round(r, 3), "spearman_brown": round(2 * r / (1 + r), 3)}
    return out


def _md(df):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join(out)


def run():
    d, b_all = load()
    comp, consts, _ = components(d)
    comp.to_parquet(C.PROC / "metric_components.parquet", index=False)

    # population shift (delivery-weighted) of P − B for the retention centring
    shift = {}
    for side in ("bat", "bowl"):
        pv = comp[(comp.side == side) & (comp.metric == "pressure_v")]
        shift[side] = {"P_pop": round(float(pv.val[pv.H].mean()), 4), "B_pop": round(float(pv.val[~pv.H].mean()), 4)}
        shift[side]["P_minus_B"] = round(shift[side]["P_pop"] - shift[side]["B_pop"], 4)

    names = {"bat": d.groupby("striker_id").striker_name.agg(lambda s: str(s.iloc[0])),
             "bowl": d.groupby("bowler_id").bowler_name.agg(lambda s: str(s.iloc[0]))}
    nonstr = d.groupby("non_striker_id").non_striker_name.agg(lambda s: str(s.iloc[0]))
    names["bat"] = names["bat"].combine_first(nonstr)

    tables = {}
    for by, label in (([], "career"), (["year"], "season")):
        w = aggregate(comp, by)
        if not by:
            w["year"] = "career"
        tables[label] = w
    ent = {lab: entropy_table(d, by) for by, lab in (([], "career"), (["year"], "season"))}
    par = {lab: partner_table(d, by) for by, lab in (([], "career"), (["year"], "season"))}

    out = {}
    for side, metrics, fname in (("bat", BAT_METRICS, "metrics_batters.csv"), ("bowl", BOWL_METRICS, "metrics_bowlers.csv")):
        frames = []
        for lab in ("career", "season"):
            w = tables[lab][tables[lab].side == side].drop(columns="side").copy()
            by = ["year"] if lab == "season" else []
            if side == "bowl":
                e = ent[lab].copy()
                w = w.merge(e, on=["player_id"] + by, how="left")
            else:
                pt = par[lab].copy()
                w = w.merge(pt, on=["player_id"] + by, how="left")
            if lab == "career":
                w["year"] = "career"
            w["year"] = w.year.astype(str)
            frames.append(w)
        t = pd.concat(frames, ignore_index=True)
        t.insert(1, "player", t.player_id.map(names[side]))
        t = t.rename(columns={"year": "season", "pressure_v_H": "P", "pressure_v_L": "B"})
        t["R_raw"] = t.P - t.B
        t["R"] = t.R_raw - shift[side]["P_minus_B"]                       # centred on the population shift
        if side == "bat":                                                  # M2: squeeze response pooled over H and L
            sq = comp[(comp.side == "bat") & (comp.metric == "squeeze_response")]
            pooled = pd.concat([sq.groupby("player_id").val.agg(["mean", "size"]).assign(season="career"),
                                sq.groupby(["player_id", "year"]).val.agg(["mean", "size"]).reset_index("year")
                                  .assign(season=lambda x: x.year.astype(str)).drop(columns="year")]).reset_index()
            pooled.columns = ["player_id", "squeeze_response_pooled", "squeeze_response_n_pooled", "season"]
            t = t.merge(pooled, on=["player_id", "season"], how="left")
        if side == "bowl":
            t["execution_drift"] = t.execution_error_H - t.execution_error_L
        # convenience columns
        t["n_H"] = t.get("pressure_v_n_H")
        t["n_L"] = t.get("pressure_v_n_L")
        t["matches"] = t[["pressure_v_matches_H", "pressure_v_matches_L"]].max(axis=1)
        t = t.dropna(axis=1, how="all")                                   # other side's metrics
        mcols = sorted(c for c in t.columns if c not in ("player_id", "player", "season"))
        t = t[["player_id", "player", "season"] + mcols]
        lead = ["player_id", "player", "season", "n_H", "n_L", "matches", "P", "B", "R", "R_raw"]
        t = t[lead + [c for c in t.columns if c not in lead]]
        num = t.select_dtypes("number").columns
        t[num] = t[num].round(5)
        t.to_csv(C.TABLES / fname, index=False)
        out[side] = t

    summary = summarise(out, comp, shift)
    write_report(summary, consts, shift, out)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 100)
    print(consts)
    print(shift)
    print(summary["metrics"].to_string(index=False))
    print(summary["pb"])
    print(summary["checks"].to_string(index=False))


def summarise(out, comp, shift):
    rows = []
    for side, metrics in (("bat", BAT_METRICS), ("bowl", BOWL_METRICS)):
        t = out[side][out[side].season == "career"]
        active = out[side][(out[side].season.isin(["2025", "2026"]))].player_id.unique()
        for m, (kind, desc) in metrics.items():
            vH = "P" if m == "pressure_v" else f"{m}_H"
            vL = "B" if m == "pressure_v" else f"{m}_L"
            nH = f"{m}_n_H"
            minn = MIN_REPORT["skill"] if kind in ("skill", "value", "profile") else MIN_REPORT["yorker"] if m == "yorker_execution" else MIN_REPORT["temperament"]
            e = t[t[nH] >= minn]
            rows.append({"side": side, "metric": m, "kind": kind, "min_n_H": minn, "players": len(e),
                         "active_2025_26": int(e.player_id.isin(active).sum()),
                         "H_median": round(e[vH].median(), 4), "H_p5": round(e[vH].quantile(.05), 4),
                         "H_p95": round(e[vH].quantile(.95), 4), "H_min": round(e[vH].min(), 4), "H_max": round(e[vH].max(), 4),
                         "L_median": round(e[vL].median(), 4),
                         "pop_H": round(float(comp[(comp.side == side) & (comp.metric == m) & comp.H].val.mean()), 4),
                         "pop_L": round(float(comp[(comp.side == side) & (comp.metric == m) & ~comp.H].val.mean()), 4)})
        if side == "bowl":
            e = t[t.variation_n_H >= MIN_REPORT["variation"]]
            rows.append({"side": side, "metric": "variation_diff (entropy H − L, bits)", "kind": "temperament",
                         "min_n_H": MIN_REPORT["variation"], "players": len(e), "active_2025_26": int(e.player_id.isin(active).sum()),
                         "H_median": round(e.variation_diff.median(), 4), "H_p5": round(e.variation_diff.quantile(.05), 4),
                         "H_p95": round(e.variation_diff.quantile(.95), 4), "H_min": round(e.variation_diff.min(), 4),
                         "H_max": round(e.variation_diff.max(), 4), "L_median": np.nan, "pop_H": np.nan, "pop_L": np.nan})
            e = t[t.execution_error_n_H >= MIN_REPORT["skill"]]
            rows.append({"side": side, "metric": "execution_drift (H − L of execution_error)", "kind": "temperament",
                         "min_n_H": MIN_REPORT["skill"], "players": len(e), "active_2025_26": int(e.player_id.isin(active).sum()),
                         "H_median": round(e.execution_drift.median(), 4), "H_p5": round(e.execution_drift.quantile(.05), 4),
                         "H_p95": round(e.execution_drift.quantile(.95), 4), "H_min": round(e.execution_drift.min(), 4),
                         "H_max": round(e.execution_drift.max(), 4), "L_median": np.nan, "pop_H": np.nan, "pop_L": np.nan})
        else:
            e = t[t.squeeze_response_n_pooled >= MIN_REPORT["temperament"]]
            rows.append({"side": side, "metric": "squeeze_response_pooled (H+L)", "kind": "temperament",
                         "min_n_H": MIN_REPORT["temperament"], "players": len(e), "active_2025_26": int(e.player_id.isin(active).sum()),
                         "H_median": round(e.squeeze_response_pooled.median(), 4), "H_p5": round(e.squeeze_response_pooled.quantile(.05), 4),
                         "H_p95": round(e.squeeze_response_pooled.quantile(.95), 4), "H_min": round(e.squeeze_response_pooled.min(), 4),
                         "H_max": round(e.squeeze_response_pooled.max(), 4), "L_median": np.nan, "pop_H": np.nan, "pop_L": np.nan})
            e = t[t.partner_n_H >= MIN_REPORT["partner"]]
            rows.append({"side": side, "metric": "partner_effect_H (ON TRIAL)", "kind": "temperament",
                         "min_n_H": MIN_REPORT["partner"], "players": len(e), "active_2025_26": int(e.player_id.isin(active).sum()),
                         "H_median": round(e.partner_effect_H.median(), 4), "H_p5": round(e.partner_effect_H.quantile(.05), 4),
                         "H_p95": round(e.partner_effect_H.quantile(.95), 4), "H_min": round(e.partner_effect_H.min(), 4),
                         "H_max": round(e.partner_effect_H.max(), 4), "L_median": np.nan, "pop_H": np.nan, "pop_L": np.nan})
    met = pd.DataFrame(rows)

    # P–B correlation (raw and corrected for sampling noise) and split-half reliability of P and B
    pb = {}
    for side in ("bat", "bowl"):
        t = out[side][out[side].season == "career"]
        e = t[(t.n_H >= 100) & (t.n_L >= 100)].copy()
        c = comp[(comp.side == side) & (comp.metric == "pressure_v")]
        se = c.groupby(["player_id", "H"]).val.agg(lambda s: s.var(ddof=1) / len(s)).unstack()
        e = e.merge(se.rename(columns={True: "seP2", False: "seB2"}), left_on="player_id", right_index=True)
        r_raw = np.corrcoef(e.P, e.B)[0, 1]
        cov = np.cov(e.P, e.B)[0, 1]
        vP = max(e.P.var() - e.seP2.mean(), 1e-12)
        vB = max(e.B.var() - e.seB2.mean(), 1e-12)
        sh = split_half(comp, side, "pressure_v", 100)
        rel = {"P_noise_share": round(float(e.seP2.mean() / e.P.var()), 3), "B_noise_share": round(float(e.seB2.mean() / e.B.var()), 3)}
        disatt = r_raw / np.sqrt(max(sh["P"]["spearman_brown"], 1e-6) * max(sh["B"]["spearman_brown"], 1e-6)) \
            if sh["P"]["spearman_brown"] > 0 and sh["B"]["spearman_brown"] > 0 else np.nan
        pb[side] = {"players (n_H≥100, n_L≥100)": len(e), "r_raw": round(r_raw, 3),
                    "r_true_moments (sampling variance removed)": round(cov / np.sqrt(vP * vB), 3) if e.P.var() > e.seP2.mean() else "P variance ≤ noise",
                    "r_disattenuated_by_split_half": round(disatt, 3) if not np.isnan(disatt) else "split-half reliability ≤ 0",
                    "split_half_P": sh["P"], "split_half_B": sh["B"], **rel}

    # suspicious-pattern checks
    checks = []
    for side, metrics in (("bat", BAT_METRICS), ("bowl", BOWL_METRICS)):
        for m in metrics:
            c = comp[(comp.side == side) & (comp.metric == m)]
            checks.append({"side": side, "metric": m, "deliveries_H": int(c.H.sum()), "deliveries_L": int((~c.H).sum()),
                           "pop_mean_H": round(float(c.val[c.H].mean()), 4), "pop_mean_L": round(float(c.val[~c.H].mean()), 4),
                           "sd_per_delivery": round(float(c.val.std()), 3)})
    return {"metrics": met, "pb": pb, "checks": pd.DataFrame(checks)}


def write_report(summary, consts, shift, out):
    L = ["# Phase 5 — Pressure metrics\n",
         "Every metric is a mean of per-delivery components **actual − expected**, signed so **positive = good**, computed on "
         f"H (`{C.PRIMARY_H}`: top 25% of LI within each innings) and on L (the rest). Expected values come from the Phase 4 "
         "opposition-aware outcome models (player-agnostic context model + opposition rating, out-of-fold by match): batters "
         "use the model with the bowler's rating, bowlers the model with the batter's rating. Rain-shortened matches are in "
         "neither H nor L. No one is ranked in this phase.\n",
         "## Definitions\n",
         "**Accounting rules.** Rates use legal deliveries (outcome probabilities renormalised over legal outcomes). Wides and "
         "no-balls count against the bowler, never the batter (wides are not balls faced; no-balls count as faced only for the "
         "v-based batter metrics). Byes and leg byes are not charged to the bowler. Run-outs, retired-out and obstructing are "
         "not bowler wickets.\n",
         f"Constants estimated from the data: {consts}. Expected runs use the observed mean runs per outcome class "
         "(e.g. a wide/no-ball averages 1.33 runs), minus byes/leg byes (batter) or uncharged runs (bowler).\n",
         "| Side | Metric | Kind | Per-delivery component |", "|---|---|---|---|"]
    for side, metrics in (("Batter", BAT_METRICS), ("Bowler", BOWL_METRICS)):
        for m, (k, desc) in metrics.items():
            L.append(f"| {side} | {m} | {k} | {desc} |")
    L += ["| Bowler | variation | temperament | Shannon entropy of delivery_style on H vs L (Miller–Madow corrected, bits); only bowlers with ≥60 H deliveries |",
          "| Batter | partner_effect_H | temperament (**ON TRIAL**) | partner's mean v on strike under H with this batter at the other end, minus the partner's mean v under H without him; weighted over partners |",
          "",
          "**Trigger definitions (no leakage: each trigger is decided before the measured ball).**",
          "- Bounce-back: the first legal ball the same bowler bowls in the same innings after a delivery that went for 4 or 6.",
          "- Squeeze response: deliveries after 3+ consecutive dot balls by the batting side (the measured ball is not part of the streak).",
          "- Entry resilience: a batter's first 10 balls faced when his first ball faced was in H (decided by the leverage of the state, not its outcome).",
          f"- Collapse resistance: all balls of a batter's innings when team WP before his first ball faced was < {COLLAPSE_WP}.",
          "- Post-wicket response: the 12 deliveries after a wicket in the same innings (the wicket ball excluded).",
          "- Execution drift: execution errors = wides, no-balls, full tosses, beamers; component = league rate for the same innings × phase × pace/spin − actual; drift = H − L (positive = fewer extra errors under pressure).",
          "- Yorker execution: mean v on yorker / wide-yorker deliveries; yorker counts are reported per bowler (`yorker_execution_n_H`, `_n_L`).",
          "- Run-out avoidance: both batters at the crease on every delivery; expected = league run-out rate per batter for the same innings × phase.",
          "- Stroke intent: share of attacking strokes among legal balls with a recorded stroke; **profile only, not used for ranking**.",
          "",
          "## Retention, centred\n",
          "R = (P − B) − population (P − B), delivery-weighted. **Finding — population-wide shift:** "
          f"batters P_pop = {shift['bat']['P_pop']}, B_pop = {shift['bat']['B_pop']} (P − B = {shift['bat']['P_minus_B']}); "
          f"bowlers P_pop = {shift['bowl']['P_pop']}, B_pop = {shift['bowl']['B_pop']} (P − B = {shift['bowl']['P_minus_B']}). "
          "These shifts are also the size of the expectation models' calibration error on H vs L, so the centring removes it. "
          "`R_raw` is kept.\n",
          "## Summary per metric (career values; eligible = enough H deliveries for that metric)\n",
          "`players` = with at least `min_n_H` deliveries on H; `active` = also played in 2025 or 2026. "
          "pop_H / pop_L = delivery-weighted population mean of the component.\n",
          _md(summary["metrics"]),
          "\n## P vs B and reliability of P and B\n"]
    for side, v in summary["pb"].items():
        L.append(f"- **{'Batters' if side == 'bat' else 'Bowlers'}**: {v}")
    L += ["\nnoise share = mean sampling variance / observed between-player variance (1 = all noise).\n",
          "## Population means per component (sanity)\n", _md(summary["checks"]),
          "\n## Caveats found in this phase\n",
          "1. **Yorker labels depend on the outcome.** `delivery_style` is recorded after the ball: a missed yorker tends to be "
          "logged as a full toss. Yorkers average 0.84 runs per ball against 1.87 for full tosses, and population mean v on "
          "yorkers is strongly positive on both H and L. Yorker execution therefore partly measures the label, not the skill, "
          "and most bowlers have under 10 yorkers on H. Recommend: report only, not for ranking.",
          "2. **Squeeze response is too sparse on H.** The dot-squeeze trigger covers 3% of deliveries, and only a handful of "
          "batters reach 30 trigger deliveries on H. Recommend: use H + L trigger deliveries pooled, or drop.",
          "3. **Entry resilience carries a new-batter effect.** Population mean v on entry deliveries is clearly negative, "
          "because the outcome model does not know the batter has just arrived. It is the same for everyone, so comparisons "
          "between players hold, but the level is not 'underperformance under pressure'.",
          "4. **Skill components are not centred on H.** The outcome models are calibrated overall, but on H batting sides beat "
          "expectation slightly (e.g. strike-rate delta population mean > 0 on H, < 0 on L). Phase 7 standardises each metric "
          "across players, which removes this level shift.",
          "5. **Bowlers' pressure level P shows no split-half reliability**, and its sampling noise is as large as the spread "
          "between bowlers. Batters' P is weakly reliable. Phase 6 will test every metric; process metrics are expected to do "
          "better than WP value.",
          "\n## Tables\n",
          "`outputs/tables/metrics_batters.csv`, `outputs/tables/metrics_bowlers.csv`: one row per player × season plus a "
          "`career` row; for every metric `<metric>_H`, `<metric>_L`, `<metric>_n_H`, `<metric>_n_L`, `<metric>_matches_H`, "
          "`<metric>_matches_L`; P, B, R (centred), R_raw. Per-delivery components for Phase 6: "
          "`data_processed/metric_components.parquet`."]
    (C.OUT / "05_metrics.md").write_text("\n".join(L))
