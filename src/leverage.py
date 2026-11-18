"""Phase 3: pressure = leverage index (LI).

1. Player-agnostic outcome model p(o | state, context), o in {0,1,2,3,4,6,wicket,wide/no-ball}.
   Context: phase, venue par, batting hand x bowling type, pace/spin, competition strength (+ match state).
   Scored out-of-fold by match (every match scored by a model that never saw it).
2. LI_raw = sum_o p(o|s) * |WP(s_o) - WP(s)|, WP of the batting side (Phase 2 model; ties = 0.5).
   LI = LI_raw / mean(LI_raw).
3. High-leverage sets H at top 20/25/33%, cut-offs global and within innings; baseline L = the rest.
4. Validation: outcome rates across LI deciles; six pressure scenarios' share of H.

Outputs: data_processed/leverage.parquet, outputs/03_leverage.md, outputs/tables/*, outputs/charts/*
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from . import config as C
from . import wp_model as W
from .plotstyle import plt, save, SERIES, TEXT2

OUTCOMES = ["0", "1", "2", "3", "4", "6", "wicket", "wide_nb"]
RUNS = [0, 1, 2, 3, 4, 6]
QUANTS = {"H20": 0.80, "H25": 0.75, "H33": 2 / 3}
NUM = ["innings", "balls_left", "wkts_before", "legal_before", "score_vs_par", "runs_needed_c", "rrr_minus_par_c",
       "par_total"]
CAT = ["phase", "hand_x_bowltype", "pace_or_spin", "strength"]
HGB = dict(learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=500, l2_regularization=1.0,
           early_stopping=False, random_state=C.SEED)


# ================================================================= outcome model
def outcome_class(b):
    tr = b.total_runs.values.astype(int)
    legal_cls = np.select([tr <= 0, tr == 1, tr == 2, tr == 3, tr <= 5], [0, 1, 2, 3, 4], 5)
    return np.where(~b.legal.values, 7, np.where(b.is_wicket.values, 6, legal_cls)).astype(int)


def context_frame(b, X):
    """Player-agnostic features. X = Phase 2 state features for the same rows."""
    inn2 = b.innings.values == 2
    f = pd.DataFrame({
        "innings": b.innings.values, "balls_left": X.balls_left.values, "wkts_before": b.wkts_before.values,
        "legal_before": b.legal_before.values,
        "score_vs_par": np.where(inn2, 0.0, X.score_vs_par.values),
        "runs_needed_c": np.where(inn2, np.clip(X.runs_needed.values, 0, 300), 0.0),
        "rrr_minus_par_c": np.where(inn2, X.rrr_c.values, 0.0),
        "par_total": b.par_total.values,
        "phase": b.phase.astype(str).values,
        "hand_x_bowltype": (b.batting_type.astype(str) + "|" + b.bowling_type.astype(str)).values,
        "pace_or_spin": b.pace_or_spin.astype(str).values,
        "strength": b.strength.astype(str).values})
    return f


def encode_hgb(f, cats):
    Z = f[NUM].copy()
    for c in CAT:
        Z[c] = pd.Categorical(f[c], categories=cats[c]).codes
    return Z.values, list(range(len(NUM), len(NUM) + len(CAT)))


def fit_hgb(Xtr, ytr, gtr, catidx, rng):
    ug = np.unique(gtr)
    hm = np.isin(gtr, rng.choice(ug, size=len(ug) // 5, replace=False))
    probe = HistGradientBoostingClassifier(max_iter=400, categorical_features=catidx, **HGB).fit(Xtr[~hm], ytr[~hm])
    losses = [log_loss(ytr[hm], p, labels=range(8)) for p in probe.staged_predict_proba(Xtr[hm])]
    n = int(np.argmin(losses)) + 1
    return HistGradientBoostingClassifier(max_iter=n, categorical_features=catidx, **HGB).fit(Xtr, ytr), n


def logit_pipe():
    pre = ColumnTransformer([("num", StandardScaler(), NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=200), CAT)])
    return make_pipeline(pre, LogisticRegression(max_iter=2000, C=1.0))


# ================================================================= WP after each outcome
def wp_after_outcomes(b, models):
    s = W.states_before(b)
    out = np.empty((len(b), 8), dtype=np.float32)
    for k, name in enumerate(OUTCOMES):
        t = s.copy()
        if k < 6:
            t["score"], t["legal"] = t.score + RUNS[k], t.legal + 1
        elif name == "wicket":
            t["wkts"], t["legal"] = t.wkts + 1, t.legal + 1
        else:
            t["score"] = t.score + 1
        out[:, k] = W.wp_state(t, models)
    return out


# ================================================================= scenarios
def scenarios(b):
    """Six pressure scenarios (boolean per delivery). Deliveries in innings order."""
    b = b.sort_values(["match_id", "innings", "innings_ball_num"])
    g = b.groupby(["match_id", "innings"], sort=False)
    sc = pd.DataFrame(index=b.index)
    # post-wicket window: the 12 deliveries after a wicket (same innings)
    wk_idx = g.is_wicket.cumsum() - b.is_wicket                       # wickets before this ball
    pos = g.cumcount()
    last_wk_pos = pd.Series(np.where(b.is_wicket, pos, np.nan), index=b.index).groupby(
        [b.match_id, b.innings]).ffill().groupby([b.match_id, b.innings]).shift(1)
    since = pos - last_wk_pos
    sc["post_wicket"] = (since >= 1) & (since <= 12) & (wk_idx > 0)
    # entry under pressure: first 10 balls faced by a batter whose first ball faced was in H (filled later)
    faced = b.ball_faced
    nth = b.assign(f=faced.astype(int)).groupby(["match_id", "innings", "striker_id"]).f.cumsum()
    sc["_nth_faced"] = np.where(faced, nth, np.nan)
    # death chase / death defence (same 2nd-innings death deliveries, batting vs bowling view)
    sc["death_chase"] = (b.innings == 2) & (b.phase.astype(str) == "death")
    sc["death_defence"] = sc.death_chase
    # dot squeeze: deliveries following 3+ consecutive dot balls by the batting side
    dot = (b.legal & (b.total_runs == 0) & ~b.is_wicket).astype(int)
    brk = (dot == 0).astype(int).groupby([b.match_id, b.innings]).cumsum()
    run = dot.groupby([b.match_id, b.innings, brk]).cumsum()
    prev_run = run.groupby([b.match_id, b.innings]).shift(1).fillna(0)
    sc["dot_squeeze"] = prev_run >= 3
    # powerplay vs set openers: powerplay, no wicket down, striker has already faced >= 10 balls
    sc["pp_vs_set_openers"] = (b.phase.astype(str) == "powerplay") & (b.wkts_before == 0) & (nth - faced.astype(int) >= 10)
    return sc


def add_entry(b, sc, H):
    """Entry under pressure: a batter's first 10 balls faced when his first ball faced was in H."""
    nth = sc["_nth_faced"]
    first = b[nth == 1][["match_id", "innings", "striker_id"]].assign(entry_H=H[nth == 1].values)
    key = b[["match_id", "innings", "striker_id"]].merge(first, how="left", on=["match_id", "innings", "striker_id"])
    return (key.entry_H.astype("boolean").fillna(False).to_numpy(dtype=bool)) & (nth.values <= 10)


# ================================================================= charts
def li_hist(li, path, cut25):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.logspace(np.log10(max(li[li > 0].min(), 1e-3)), np.log10(li.max()), 70)
    ax.hist(li[li > 0], bins=bins, color=SERIES[0], edgecolor="#fcfcfb", linewidth=0.6)
    ax.axvline(1, color=TEXT2, lw=1, ls=":")
    ax.axvline(cut25, color=SERIES[1], lw=2)
    ax.text(cut25 * 1.08, ax.get_ylim()[1] * 0.92, f"H25 cut-off = {cut25:.2f}", color=TEXT2, fontsize=9)
    ax.text(1.08, ax.get_ylim()[1] * 0.80, "mean = 1", color=TEXT2, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Leverage index (log scale)")
    ax.set_ylabel("Deliveries")
    ax.set_title("Distribution of the leverage index (all deliveries, both innings)")
    save(fig, path)


def decile_chart(dec, path):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharex=True)
    for ax, (col, lab) in zip(axes, [("dismissal_rate", "Dismissal rate"), ("wide_nb_rate", "Wide / no-ball rate"),
                                     ("dot_rate", "Dot-ball rate (legal balls)"), ("boundary_rate", "Boundary rate (legal balls)")]):
        for i, inn in enumerate([1, 2]):
            d = dec[dec.innings == inn]
            ax.plot(d.decile, d[col], marker="o", ms=4, color=SERIES[i], label=f"{inn}{'st' if inn == 1 else 'nd'} innings")
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("LI decile (1 = lowest)")
        ax.set_xticks(range(1, 11))
    axes[0].legend(loc="upper left")
    fig.suptitle("Outcome rates across leverage deciles (deciles within each innings)", fontsize=11)
    save(fig, path)


def share_chart(tab, path):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    labels = [f"{r.innings}-{r.phase}" for r in tab.itertuples()]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, tab.share_of_all, width=0.38, color=SERIES[0], label="share of all deliveries")
    ax.bar(x + 0.2, tab.share_of_H25_within, width=0.38, color=SERIES[1], label="share of H25 (within-innings cut-off, primary)")
    ax.set_xticks(x, ["Inns " + l_.replace("-", " · ") for l_ in labels], fontsize=9)
    ax.set_ylabel("Share")
    ax.set_title("Where high-leverage deliveries come from")
    ax.legend(loc="upper left")
    save(fig, path)


def _md(df):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


# ================================================================= main
def run():
    rng = np.random.default_rng(C.SEED)
    b = pd.read_parquet(C.PROC / "balls.parquet")
    p = pd.read_parquet(C.PROC / "wp_preds.parquet")
    m = pd.read_parquet(C.PROC / "matches.parquet")
    b = b.merge(p, on=["match_id", "innings", "innings_ball_num"])
    b = b[~b.is_penalty].reset_index(drop=True)               # the 5-run penalty row is not a delivery
    models = joblib.load(C.PROC / "wp_models.joblib")

    # outcome-model folds: WP folds where they exist, otherwise a seeded random fold (every match out-of-fold)
    mids = b.match_id.unique()
    extra = {mid: int(f) for mid, f in zip(mids, rng.integers(0, W.N_FOLDS, len(mids)))}
    b["ofold"] = np.where(b.fold.values >= 0, b.fold.values, b.match_id.map(extra).values)

    X = W.build_features(W.states_before(b), models)
    f = context_frame(b, X)
    y = outcome_class(b)
    cats = {c: sorted(f[c].unique()) for c in CAT}
    Xh, catidx = encode_hgb(f, cats)

    # ---- out-of-fold outcome probabilities: HGB vs multinomial logistic
    P = {"hgb": np.zeros((len(b), 8)), "logit": np.zeros((len(b), 8))}
    trees = {}
    for fo in range(W.N_FOLDS):
        tr, te = b.ofold.values != fo, b.ofold.values == fo
        h, trees[fo] = fit_hgb(Xh[tr], y[tr], b.match_id.values[tr], catidx, rng)
        P["hgb"][te] = h.predict_proba(Xh[te])
        lp = logit_pipe().fit(f[tr], y[tr])
        P["logit"][te] = lp.predict_proba(f[te])
        print(f"[phase3] outcome fold {fo}: trees {trees[fo]}")
    base_rate = np.bincount(y, minlength=8) / len(y)
    om = pd.DataFrame([
        {"model": "Base rates (no features)", "log_loss": round(log_loss(y, np.tile(base_rate, (len(y), 1)), labels=range(8)), 4)},
        {"model": "Multinomial logistic", "log_loss": round(log_loss(y, P["logit"], labels=range(8)), 4)},
        {"model": "Gradient boosting (HGB)", "log_loss": round(log_loss(y, P["hgb"], labels=range(8)), 4)}])
    chosen = "hgb" if om.log_loss.iloc[2] <= om.log_loss.iloc[1] else "logit"
    po = P[chosen]
    calib = pd.DataFrame({"outcome": OUTCOMES, "actual_rate": np.bincount(y, minlength=8) / len(y),
                          "mean_predicted": po.mean(0)}).round(4)
    om.to_csv(C.TABLES / "outcome_model_metrics.csv", index=False)

    # ---- WP after each outcome and leverage
    wp_o = wp_after_outcomes(b, models)
    wp_s = b.wp_before.values.astype(float)
    exp_next = (po * wp_o).sum(1)                              # E_o[WP(s_o)]
    li_raw_spec = (po * np.abs(wp_o - wp_s[:, None])).sum(1)    # formula as specified: swing vs model WP(s)
    li_raw = (po * np.abs(wp_o - exp_next[:, None])).sum(1)     # used: swing vs expected next WP (removes model jumps)
    short = (b.revised_recorded.to_numpy(bool) | b.revised_unrecorded.to_numpy(bool) |
             (b.max_balls.values < 120) | (b.max_balls2.values < 120))
    valid = ~short if C.EXCLUDE_SHORTENED_FROM_PRESSURE else np.ones(len(b), bool)
    li = li_raw / li_raw[valid].mean()
    li_spec = li_raw_spec / li_raw_spec[valid].mean()
    b["LI_raw"], b["LI"], b["LI_spec"] = li_raw.astype("float32"), li.astype("float32"), li_spec.astype("float32")
    b["wp_gap"] = np.abs(exp_next - wp_s)
    b["li_valid"] = valid

    # consistency diagnostics (all deliveries, before exclusions)
    grp = np.select([short, (b.innings.values == 1) & (b.balls_left.values == 1), b.innings.values == 1],
                    ["rain-shortened matches", "full matches: last ball of 1st innings", "full matches: other 1st-innings balls"],
                    "full matches: 2nd innings")
    cons = (b.assign(group=grp).groupby("group")
              .agg(deliveries=("LI", "size"), mean_abs_gap=("wp_gap", "mean"), mean_LI_as_specified=("LI_spec", "mean"),
                   mean_LI_used=("LI", "mean")).round(4).reset_index())
    lv, ls_ = li[valid], li_spec[valid]
    agree = {"spearman_LI_vs_specified": round(float(pd.Series(lv).corr(pd.Series(ls_), method="spearman")), 4),
             "H25_overlap": round(float(((lv >= np.quantile(lv, .75)) & (ls_ >= np.quantile(ls_, .75))).sum()
                                        / (lv >= np.quantile(lv, .75)).sum()), 4)}
    cons.to_csv(C.TABLES / "leverage_consistency.csv", index=False)
    excluded = {"matches": int(b.loc[~valid, "match_id"].nunique()), "deliveries": int((~valid).sum())}

    b = b[valid].copy()
    li = b.LI.values.astype(float)
    for k, q in QUANTS.items():
        b[f"{k}_global"] = li >= np.quantile(li, q)
        cut = b.groupby("innings").LI.transform(lambda s: s.quantile(q))
        b[f"{k}_within"] = b.LI >= cut
    cut25 = float(np.quantile(li, 0.75))
    po, wp_o = po[valid], wp_o[valid]

    # ---- check 1: where does H come from?
    b["phase_s"] = b.phase.astype(str)
    shares = (b.groupby(["innings", "phase_s"], observed=True)
                .agg(deliveries=("LI", "size"), mean_LI=("LI", "mean"), median_LI=("LI", "median"),
                     in_H25_global=("H25_global", "sum"), in_H25_within=("H25_within", "sum")).reset_index()
                .rename(columns={"phase_s": "phase"}))
    order = {"powerplay": 0, "middle": 1, "death": 2}
    shares = shares.sort_values(["innings", "phase"], key=lambda s: s.map(order) if s.name == "phase" else s)
    shares["share_of_all"] = (shares.deliveries / shares.deliveries.sum()).round(3)
    shares["share_of_H25_global"] = (shares.in_H25_global / shares.in_H25_global.sum()).round(3)
    shares["share_of_H25_within"] = (shares.in_H25_within / shares.in_H25_within.sum()).round(3)
    shares["pct_of_group_in_H25_global"] = (shares.in_H25_global / shares.deliveries).round(3)
    shares[["mean_LI", "median_LI"]] = shares[["mean_LI", "median_LI"]].round(3)
    shares.to_csv(C.TABLES / "leverage_share_by_innings_phase.csv", index=False)
    inn_share = {k: b.groupby("innings")[f"{k}_global"].sum().div(b[f"{k}_global"].sum()).round(3).to_dict() for k in QUANTS}

    # 1st-innings death overs in close vs other games
    mm = m.set_index("match_id")
    close = (mm.is_tie | (mm.result_kind.eq("runs") & (mm.margin <= 10)) |
             (mm.result_kind.eq("wkts") & (mm.legal2 >= mm.max_balls2 - 6)))
    b["close_game"] = b.match_id.map(close).fillna(False).astype(bool)
    d1 = b[(b.innings == 1) & (b.phase_s == "death")]
    close_tab = (b.assign(group=np.select(
        [(b.innings == 1) & (b.phase_s == "death") & b.close_game, (b.innings == 1) & (b.phase_s == "death"),
         (b.innings == 2) & (b.phase_s == "death") & b.close_game, (b.innings == 2) & (b.phase_s == "death")],
        ["1st inns death, close game", "1st inns death, other games", "2nd inns death, close game", "2nd inns death, other games"],
        "other deliveries"))
        .groupby("group").agg(deliveries=("LI", "size"), matches=("match_id", "nunique"), mean_LI=("LI", "mean"),
                              median_LI=("LI", "median"), p90_LI=("LI", lambda s: s.quantile(0.9)),
                              pct_in_H25_global=("H25_global", "mean"), pct_in_H25_within=("H25_within", "mean"))
        .round(3).reset_index())
    close_tab.to_csv(C.TABLES / "leverage_close_games.csv", index=False)

    # ---- check 2: top 20 deliveries
    teams = pd.concat([m[["home_team_id", "home_team"]].set_axis(["id", "name"], axis=1),
                       m[["away_team_id", "away_team"]].set_axis(["id", "name"], axis=1)]).drop_duplicates("id").set_index("id").name
    top = b.nlargest(20, "LI").copy()
    top["match"] = top.match_id.map(lambda i: f"{mm.home_team[i]} v {mm.away_team[i]}, {mm.match_date[i].date()}")
    top["batting"] = top.batting_team_id.map(teams)
    top["over.ball"] = top.overid.astype(str) + "." + top.over_ball_num.astype(str)
    top["situation"] = np.where(top.innings == 2,
                                "need " + top.runs_needed.fillna(0).astype(int).astype(str) + " off " + top.balls_left.astype(str)
                                + ", " + (10 - top.wkts_before).astype(str) + " wkts in hand",
                                top.score_before.astype(str) + "/" + top.wkts_before.astype(str) + ", " + top.balls_left.astype(str) + " balls left")
    top["what_happened"] = np.where(top.is_wicket, "WICKET (" + top.dismissal_type.astype(str) + ")",
                                    np.where(top.legal, top.total_runs.astype(str) + " run(s)",
                                             top.extras_type.astype(str) + " +" + top.total_runs.astype(str)))
    top_out = top[["match", "innings", "batting", "over.ball", "situation", "striker_name", "bowler_name",
                   "wp_before", "LI", "what_happened", "wp_after"]].round({"wp_before": 3, "wp_after": 3, "LI": 2})
    top_out.to_csv(C.TABLES / "leverage_top20.csv", index=False)

    # ---- validation: outcome rates across LI deciles (within innings)
    b["decile"] = b.groupby("innings").LI.transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
    lg = b.legal
    dec = (b.assign(dis=b.is_wicket, wnb=~b.legal & ~b.is_penalty,
                    dot=np.where(lg, (b.total_runs == 0) & ~b.is_wicket, np.nan),
                    bnd=np.where(lg, b.is_four | b.is_six, np.nan))
             .groupby(["innings", "decile"]).agg(deliveries=("LI", "size"), mean_LI=("LI", "mean"),
                                                 dismissal_rate=("dis", "mean"), wide_nb_rate=("wnb", "mean"),
                                                 dot_rate=("dot", "mean"), boundary_rate=("bnd", "mean"))
             .reset_index().round(4))
    dec.to_csv(C.TABLES / "leverage_deciles.csv", index=False)

    # ---- validation: six scenarios
    sc = scenarios(b)
    sc = sc.reindex(b.index)
    sc["entry_under_pressure"] = add_entry(b, sc, b[C.PRIMARY_H])
    scen_rows = []
    for name in ["post_wicket", "entry_under_pressure", "death_chase", "death_defence", "dot_squeeze", "pp_vs_set_openers"]:
        s_ = sc[name].values.astype(bool)
        row = {"scenario": name, "deliveries": int(s_.sum()), "share_of_all": round(s_.mean(), 3)}
        for k in QUANTS:
            for v in ("global", "within"):
                h = b[f"{k}_{v}"].values
                if k == "H25":
                    row[f"share_of_{k}_{v}"] = round(s_[h].mean(), 3)
        row["pct_of_scenario_in_H25_within"] = round(b.H25_within.values[s_].mean(), 3)
        row["enrichment (H25 within share / overall share)"] = round(row["share_of_H25_within"] / max(row["share_of_all"], 1e-9), 2)
        row["mean_LI"] = round(float(b.LI.values[s_].mean()), 3)
        scen_rows.append(row)
    scen = pd.DataFrame(scen_rows)
    scen.to_csv(C.TABLES / "leverage_scenarios.csv", index=False)

    # ---- charts
    li_hist(li, C.CHARTS / "leverage_histogram.png", cut25)
    decile_chart(dec, C.CHARTS / "leverage_deciles.png")
    share_chart(shares, C.CHARTS / "leverage_share_by_innings_phase.png")

    # ---- persist
    keep = ["match_id", "innings", "innings_ball_num", "LI_raw", "LI", "LI_spec"] + [f"{k}_{v}" for k in QUANTS for v in ("global", "within")]
    out = b[keep].copy()
    for k, name in enumerate(OUTCOMES):
        out[f"p_{name}"] = po[:, k].astype("float32")
        out[f"wp_o_{name}"] = wp_o[:, k]
    out["ofold"] = b.ofold.values.astype("int8")
    for c in ["post_wicket", "entry_under_pressure", "death_chase", "dot_squeeze", "pp_vs_set_openers"]:
        out[f"sc_{c}"] = sc[c].values.astype(bool)
    out.to_parquet(C.PROC / "leverage.parquet", index=False)
    joblib.dump({"chosen": chosen, "trees": trees, "cats": cats}, C.PROC / "outcome_model_info.joblib")

    write_report(om, chosen, calib, trees, shares, inn_share, close_tab, top_out, dec, scen, cut25, b, cons, agree, excluded)
    print(cons.to_string(index=False)); print(agree, excluded)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(om.to_string(index=False))
    print("H share by innings (global cut-off):", inn_share)
    print(shares.to_string(index=False))
    print(close_tab.to_string(index=False))
    print(scen.to_string(index=False))


def write_report(om, chosen, calib, trees, shares, inn_share, close_tab, top_out, dec, scen, cut25, b, cons, agree, excluded):
    L = ["# Phase 3 — Pressure: the leverage index\n",
         "## Outcome model p(o | state, context)\n",
         "Outcomes: 0, 1, 2, 3, 4 (incl. rare 5s), 6, wicket, wide/no-ball. Player-agnostic features: innings, balls left, "
         "wickets down, legal balls bowled, score vs venue par at this stage (1st inns), runs needed / required rate vs par "
         "(2nd inns), venue par, phase, batting hand × bowling type, pace/spin, competition strength. Scored out-of-fold "
         "(5 folds grouped by match; matches outside WP training get a seeded random fold).\n",
         _md(om),
         f"\nChosen: **{'gradient boosting' if chosen == 'hgb' else 'multinomial logistic'}** (lower out-of-fold log-loss). "
         f"Trees per fold (chosen on held-out matches): {trees}.\n",
         "Overall calibration (out-of-fold):\n", _md(calib),
         "\n## Leverage index\n",
         "For each delivery, the state after each outcome s_o is built exactly (runs: score + r and one legal ball; wicket: one "
         "more wicket and one legal ball; wide/no-ball: score + 1, no legal ball). WP(s_o) comes from the Phase 2 model with "
         "the match's own out-of-fold WP models; terminal states are exact and **ties count 0.5**; the end of the 1st innings "
         "becomes 1 − WP(start of the chase).\n",
         "**Deviation from the specified formula (stated plainly).** As specified, LI_raw = Σ_o p(o|s)·|WP(s_o) − WP(s)|. "
         "This assumes the WP model is internally consistent (WP(s) = E[WP(s_o)]). It is, to ~0.001, everywhere except the last "
         "ball of the 1st innings, where the 1st-innings model hands over to the chase model and the two disagree. There the "
         "specified formula counts the hand-over jump as pressure regardless of the outcome. The LI used here measures each "
         "outcome's swing around the expected next WP instead: **LI_raw = Σ_o p(o|s)·|WP(s_o) − Σ_o' p(o'|s)·WP(s_o')|**. "
         "Where the model is consistent the two are the same; the specified version is kept as `LI_spec`.\n",
         _md(cons),
         f"\nAgreement on included deliveries: {agree}.\n",
         f"**Rain-shortened matches excluded from pressure sets** ({excluded['matches']} matches, {excluded['deliveries']:,} "
         "deliveries: recorded or detected revisions, or innings shorter than 20 overs). The WP model never trained on shortened "
         "innings and extrapolates badly there (four of the original top-20 deliveries were last balls of 5–12-over innings). "
         "Switch: `EXCLUDE_SHORTENED_FROM_PRESSURE` in `src/config.py`.\n",
         f"**LI = LI_raw / mean LI_raw** over the {len(b):,} included deliveries. H25 global cut-off: LI ≥ {cut25:.3f}.\n",
         "Histogram: `outputs/charts/leverage_histogram.png`.\n",
         "## Check 1 — where high leverage comes from\n",
         "**Decision (after review): the primary high-leverage set H is the top 25% of LI within each innings** "
         "(`H25_within`); top 20% and 33% within innings are kept for robustness. With a global cut-off, 83% of H25 came "
         f"from the 2nd innings and 1st-innings middle overs contributed none (share of H by innings, global: {inn_share}). "
         "A global cut-off would leave players who mostly bat or bowl first with almost no pressure sample. "
         "Baseline L = included deliveries not in H. Rain-shortened matches are in neither H nor L; they remain in the "
         "data for conventional statistics (Phase 9).\n",
         "**1st-innings limitation.** The 1st-innings WP model gives a run the same WP value at every stage. Tested before "
         "Phase 4: interactions letting (score − par) and wickets in hand vary with balls left — (score − par)·√((120+k)/(balls "
         "left+k)) for k = 6/12/24/48, (score − par) × balls-left hinges at 12/30/60/90, wickets × balls-left bases — with "
         "sign constraints. Out-of-fold 1st-innings log-loss: current 0.6242; best variant 0.6240 (wickets bases; per-fold "
         "changes mixed, three folds worse). Unconstrained, the stage interaction is slightly *negative*: a run late in the "
         "innings is not worth more once par is accounted for (an above-par score early also signals conditions and batting "
         "strength). Current model kept. Consequence: 1st-innings leverage comes mainly from wicket risk, so 1st-innings "
         "middle overs rarely reach H even within innings.\n",
         _md(shares),
         "\n`outputs/charts/leverage_share_by_innings_phase.png`\n",
         "### 1st-innings death overs in close games\n",
         "Close game = tie, won by ≤10 runs, or chase completed in the final over.\n", _md(close_tab),
         "\n## Check 2 — the 20 highest-leverage deliveries\n", _md(top_out),
         "\n## Validation — outcome rates across LI deciles (deciles within each innings)\n",
         "`outputs/charts/leverage_deciles.png`\n", _md(dec),
         "\n## Validation — six pressure scenarios\n",
         "Definitions: post-wicket = the 12 deliveries after a wicket in the same innings; entry under pressure = a batter's "
         "first 10 balls faced when his first ball faced was in H25 (within innings); death chase = 2nd-innings overs 16–20 (batting "
         "view); death defence = the same deliveries from the bowling side (identical set); dot squeeze = deliveries after 3+ "
         "consecutive dot balls by the batting side; powerplay vs set openers = powerplay, no wicket down, striker has already "
         "faced ≥10 balls.\n", _md(scen)]
    (C.OUT / "03_leverage.md").write_text("\n".join(L))
