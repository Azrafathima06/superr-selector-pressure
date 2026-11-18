"""Phase 4: per-ball performance value.

1. Opposition quality. Run value of a delivery (batting side's view):
       RV = (runs − E[runs | context]) − wicket_cost · (wicket − p_wicket)
   E[·] from the player-agnostic Phase 3 outcome model (out-of-fold). wicket_cost = expected runs still to come
   with one more wicket down (Phase 2 Markov resource table for the match's own fold). Batters are rated on
   deliveries they faced (no wides), bowlers on everything they bowled (−RV). Ratings are empirical-Bayes shrunk
   toward 0 (league average), c = within-player variance / between-player variance (method of moments), and every
   delivery uses ratings computed WITHOUT its own match.
2. Opposition-aware expectation models (out-of-fold by match):
       batter side: p_bat(o | state, context, bowler rating)
       bowler side: p_bowl(o | state, context, batter rating)
3. Value = actual WP change − Σ_o p(o|s,x)·[WP(s_o) − WP(s)]  (batting side's WP; ties 0.5)
   v = Value / LI_raw; bowlers use −Value.
4. Per player: P = mean v over H (primary: top 25% LI within innings), B = mean v over L, R = P − B.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

from . import config as C
from . import wp_model as W
from . import leverage as LV
from .plotstyle import plt, save, SERIES, TEXT2

RUNS_O = np.array([0, 1, 2, 3, 4, 6, 0, 1], dtype=float)


def shrink_constant(df, val, who):
    """c = sigma^2_within / tau^2_between from per-delivery values grouped by player (method of moments)."""
    g = df.groupby(who)[val]
    n, mean = g.size(), g.mean()
    sigma2 = df[val].var()
    tau2 = max(np.average((mean - df[val].mean()) ** 2, weights=n) - sigma2 * len(n) / n.sum(), 1e-12)
    return sigma2 / tau2, sigma2, tau2


def loo_rating(df, val, who, c):
    """Shrunk mean of `val` for player `who`, excluding the delivery's own match."""
    S = df.groupby(who)[val].transform("sum")
    N = df.groupby(who)[val].transform("size")
    s = df.groupby([who, "match_id"])[val].transform("sum")
    m = df.groupby([who, "match_id"])[val].transform("size")
    return ((S - s) / (N - m + c)).astype("float32")


def fit_predict_oof(Xh, y, folds, catidx, n_trees):
    P = np.zeros((len(y), 8))
    for fo in range(W.N_FOLDS):
        tr, te = folds != fo, folds == fo
        h = HistGradientBoostingClassifier(max_iter=n_trees, categorical_features=catidx, **LV.HGB).fit(Xh[tr], y[tr])
        P[te] = h.predict_proba(Xh[te])
    return P


def player_table(d, who, name_col, vcol):
    sets = {"H25": C.PRIMARY_H, "H20": "H20_within", "H33": "H33_within"}
    g = d.groupby(who)
    out = pd.DataFrame({"player": g[name_col].agg(lambda s: s.astype(str).mode().iat[0]),
                        "deliveries": g.size(), "matches": g.match_id.nunique(),
                        "seasons": g.year.nunique(), "last_season": g.year.max()})
    for lab, col in sets.items():
        h = d[col].values.astype(bool)
        gh, gl = d[h].groupby(who)[vcol], d[~h].groupby(who)[vcol]
        out[f"n_{lab}"] = gh.size()
        out[f"P_{lab}"] = gh.mean()
        if lab == "H25":
            out["n_L"] = gl.size()
            out["B"] = gl.mean()
            out["se_P"] = gh.std() / np.sqrt(gh.size())
            out["se_B"] = gl.std() / np.sqrt(gl.size())
        out[f"R_{lab}"] = out[f"P_{lab}"] - d[~d[col].values.astype(bool)].groupby(who)[vcol].mean()
    out = out.rename(columns={"P_H25": "P", "R_H25": "R"})
    out["se_R"] = np.sqrt(out.se_P ** 2 + out.se_B ** 2)
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(4)
    return out.reset_index().rename(columns={who: "player_id"}).fillna({"n_H25": 0, "n_H20": 0, "n_H33": 0})


def scatter_PB(t, title, path, min_h):
    d = t[t.n_H25 >= min_h]
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.axhline(0, color=TEXT2, lw=0.8)
    ax.axvline(0, color=TEXT2, lw=0.8)
    ax.scatter(d.B, d.P, s=np.clip(d.n_H25 / 8, 8, 80), color=SERIES[0], alpha=0.55, edgecolor="#fcfcfb", linewidth=0.6)
    r = d[["B", "P"]].corr().iloc[0, 1]
    ax.set_xlabel("Baseline B (mean v on low-leverage deliveries)")
    ax.set_ylabel("Pressure level P (mean v on H)")
    ax.set_title(f"{title} — {len(d)} players with ≥{min_h} H deliveries; r = {r:.2f}", fontsize=10)
    save(fig, path)
    return r


def _md(df):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def run():
    b = pd.read_parquet(C.PROC / "balls.parquet")
    lev = pd.read_parquet(C.PROC / "leverage.parquet")
    p = pd.read_parquet(C.PROC / "wp_preds.parquet")
    models = joblib.load(C.PROC / "wp_models.joblib")
    info = joblib.load(C.PROC / "outcome_model_info.joblib")
    key = ["match_id", "innings", "innings_ball_num"]
    d = lev.merge(b, on=key).merge(p, on=key)          # included deliveries only (rain-shortened excluded)
    O = LV.OUTCOMES
    po = d[[f"p_{o}" for o in O]].values.astype(float)
    po = po / po.sum(1, keepdims=True)                 # stored as float32; renormalise
    wpo = d[[f"wp_o_{o}" for o in O]].values.astype(float)
    y = LV.outcome_class(d)

    # ---- 1. run value and leave-one-match-out ratings
    exp_runs = po @ RUNS_O
    bl = d.balls_left.values.astype(int)
    wl = d.wkts_before.values.astype(int)
    cost = np.empty(len(d))
    for fo in np.unique(d.fold.values):
        sel = d.fold.values == fo
        M = models["aux"][fo]["M"]
        bb = np.clip(bl[sel] - 1, 0, 120)
        cost[sel] = M[bb, np.clip(wl[sel], 0, 9)] - M[bb, np.clip(wl[sel] + 1, 0, 10)]
    d["wkt_cost_runs"] = cost
    d["RV"] = (d.total_runs.values - exp_runs) - cost * (d.is_wicket.values - po[:, 6])
    bat_rows = ~d.is_wide.values
    c_bat, s2b, t2b = shrink_constant(d[bat_rows], "RV", "striker_id")
    dn = d.assign(RVn=-d.RV)
    c_bowl, s2w, t2w = shrink_constant(dn, "RVn", "bowler_id")
    d["bat_rating"] = np.nan
    d.loc[bat_rows, "bat_rating"] = loo_rating(d[bat_rows], "RV", "striker_id", c_bat).values
    # wides: the striker's rating from his other deliveries in other matches (same leave-one-match-out value)
    d["bat_rating"] = d.groupby(["striker_id", "match_id"]).bat_rating.transform(lambda s: s.fillna(s.mean()))
    d["bat_rating"] = d.bat_rating.fillna(0.0)
    d["bowl_rating"] = loo_rating(dn, "RVn", "bowler_id", c_bowl).values
    ratings_info = {"batters": {"c": round(c_bat, 1), "sigma2": round(s2b, 3), "tau2": round(t2b, 5)},
                    "bowlers": {"c": round(c_bowl, 1), "sigma2": round(s2w, 3), "tau2": round(t2w, 5)}}

    # ---- 2. opposition-aware expectation models (out-of-fold, same folds as Phase 3)
    X = W.build_features(W.states_before(d), models)
    f = LV.context_frame(d, X)
    Xh, catidx = LV.encode_hgb(f, info["cats"])
    n_trees = int(np.median(list(info["trees"].values())))
    folds = d.ofold.values
    P_bat = fit_predict_oof(np.c_[Xh, d.bowl_rating.values], y, folds, catidx, n_trees)
    P_bowl = fit_predict_oof(np.c_[Xh, d.bat_rating.values], y, folds, catidx, n_trees)
    om = pd.DataFrame([
        {"model": "Phase 3 player-agnostic (context only)", "log_loss": round(log_loss(y, po, labels=range(8)), 5)},
        {"model": "Batter-side: context + bowler rating (excl. match)", "log_loss": round(log_loss(y, P_bat, labels=range(8)), 5)},
        {"model": "Bowler-side: context + batter rating (excl. match)", "log_loss": round(log_loss(y, P_bowl, labels=range(8)), 5)}])

    # ---- 3. value per delivery
    wp_s = d.wp_before.values.astype(float)
    actual = d.wp_after.values.astype(float) - wp_s
    exp_bat = (P_bat * wpo).sum(1) - wp_s
    exp_bowl = (P_bowl * wpo).sum(1) - wp_s
    d["value_bat"] = actual - exp_bat
    d["value_bowl"] = -(actual - exp_bowl)
    li_raw = d.LI_raw.values.astype(float)
    d["v_bat"] = d.value_bat / li_raw
    d["v_bowl"] = d.value_bowl / li_raw
    vdist = pd.DataFrame({k: d[k].describe(percentiles=[.001, .01, .25, .5, .75, .99, .999]) for k in ["v_bat", "v_bowl"]}).T.round(3)
    vdist.columns = [str(c) for c in vdist.columns]
    # how extreme v arises: tiny LI_raw (decided games) and outcomes off the representative transitions
    lo = li_raw < np.quantile(li_raw, 0.01)
    extreme = pd.DataFrame([{"group": g, "deliveries": int(m_.sum()), "mean_abs_v_bat": round(float(np.abs(d.v_bat[m_]).mean()), 3),
                             "p99_abs_v_bat": round(float(np.abs(d.v_bat[m_]).quantile(.99)), 3)}
                            for g, m_ in [("LI_raw in lowest 1%", lo), ("other deliveries", ~lo)]])

    # ---- 4. per-player P, B, R
    bat_t = player_table(d[bat_rows], "striker_id", "striker_name", "v_bat")
    bowl_t = player_table(d, "bowler_id", "bowler_name", "v_bowl")
    bat_t.to_csv(C.TABLES / "player_pressure_batters.csv", index=False)
    bowl_t.to_csv(C.TABLES / "player_pressure_bowlers.csv", index=False)

    # sanity: average v on H and on L across all deliveries (should be ≈ 0 if expectations are calibrated)
    H = d[C.PRIMARY_H].values.astype(bool)
    sanity = pd.DataFrame([{"side": s_, "mean_v_H": round(float(d.loc[H & rows, col].mean()), 4),
                            "mean_v_L": round(float(d.loc[~H & rows, col].mean()), 4),
                            "mean_Value_H": round(float(d.loc[H & rows, vc].mean()), 5),
                            "mean_Value_L": round(float(d.loc[~H & rows, vc].mean()), 5)}
                           for s_, col, vc, rows in [("batters", "v_bat", "value_bat", bat_rows),
                                                     ("bowlers", "v_bowl", "value_bowl", np.ones(len(d), bool))]])
    # sample-size picture (for Phase 7 eligibility)
    dist = pd.DataFrame({
        "batters": bat_t.n_H25.describe(percentiles=[.25, .5, .75, .9]),
        "bowlers": bowl_t.n_H25.describe(percentiles=[.25, .5, .75, .9])}).T.round(1)
    dist.columns = [str(c) for c in dist.columns]
    min_h = 100
    r_bat = scatter_PB(bat_t, "Batters", C.CHARTS / "value_P_vs_B_batters.png", min_h)
    r_bowl = scatter_PB(bowl_t, "Bowlers", C.CHARTS / "value_P_vs_B_bowlers.png", min_h)

    # top opposition ratings (sanity, ≥500 deliveries)
    rb = d[bat_rows].groupby("striker_id").agg(player=("striker_name", lambda s: str(s.iloc[0])), n=("RV", "size"), RV=("RV", "sum"))
    rb["rating"] = rb.RV / (rb.n + c_bat)
    rw = dn.groupby("bowler_id").agg(player=("bowler_name", lambda s: str(s.iloc[0])), n=("RVn", "size"), RV=("RVn", "sum"))
    rw["rating"] = rw.RV / (rw.n + c_bowl)
    top_bat = rb[rb.n >= 500].nlargest(10, "rating")[["player", "n", "rating"]].round(4)
    top_bowl = rw[rw.n >= 500].nlargest(10, "rating")[["player", "n", "rating"]].round(4)

    # ---- persist
    keep = key + ["RV", "wkt_cost_runs", "bat_rating", "bowl_rating", "value_bat", "value_bowl", "v_bat", "v_bowl"]
    out = d[keep].copy()
    for k, o in enumerate(O):
        out[f"pbat_{o}"] = P_bat[:, k].astype("float32")
        out[f"pbowl_{o}"] = P_bowl[:, k].astype("float32")
    out.to_parquet(C.PROC / "values.parquet", index=False)

    write_report(ratings_info, om, n_trees, vdist, extreme, sanity, dist, r_bat, r_bowl, min_h, top_bat, top_bowl,
                 bat_t, bowl_t, len(d))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(ratings_info)
    print(om.to_string(index=False))
    print(vdist.to_string())
    print(extreme.to_string(index=False))
    print(sanity.to_string(index=False))
    print(dist.to_string())
    print(f"P vs B correlation (≥{min_h} H deliveries): batters {r_bat:.2f}, bowlers {r_bowl:.2f}")
    print(top_bat.to_string(index=False))
    print(top_bowl.to_string(index=False))


def write_report(ratings_info, om, n_trees, vdist, extreme, sanity, dist, r_bat, r_bowl, min_h, top_bat, top_bowl,
                 bat_t, bowl_t, n):
    L = ["# Phase 4 — Per-ball performance value\n",
         f"Deliveries: {n:,} (rain-shortened matches excluded, as decided in Phase 3). WP is the batting side's; ties = 0.5.\n",
         "## 1. Opposition quality (leave-one-match-out run-value ratings)\n",
         "Run value of a delivery, batting side's view: **RV = (runs − E[runs]) − wicket_cost × (wicket − p_wicket)**. "
         "E[runs] and p_wicket come from the Phase 3 player-agnostic outcome model (out-of-fold). wicket_cost = expected runs "
         "still to come at this state minus the same with one more wicket down, from the Phase 2 Markov resource table "
         "(match's own fold). Batter rating: shrunk mean RV over deliveries faced (wides excluded). Bowler rating: shrunk "
         "mean −RV over everything bowled. Shrinkage toward 0 (league average) with c = σ²_within / τ²_between "
         f"(method of moments): {ratings_info}. **Every delivery uses ratings computed without its own match** (later "
         "matches are used; this affects only the opposition adjustment).\n",
         "Highest-rated (≥500 deliveries, all-data ratings, sanity check only):\n",
         "Batters:\n", _md(top_bat), "\nBowlers:\n", _md(top_bowl),
         "\n## 2. Opposition-aware expectation models\n",
         f"Same features and folds as the Phase 3 outcome model plus one opposition rating; gradient boosting with "
         f"{n_trees} trees (median of the Phase 3 held-out choice). Out-of-fold log-loss:\n", _md(om),
         "\n## 3. Value per delivery\n",
         "Value = actual WP change − Σ_o p(o|s,x)·[WP(s_o) − WP(s)] = WP_after − Σ_o p(o|s,x)·WP(s_o). Batters use the "
         "batter-side model (opposition = bowler), bowlers the bowler-side model (opposition = batter) and −Value. "
         "**v = Value / LI_raw** (LI_raw = centred Phase 3 leverage).\n",
         "Distribution of v:\n", _md(vdist.reset_index().rename(columns={"index": "series"})),
         "\nWhere extreme v comes from:\n", _md(extreme),
         "\nCalibration check (mean over all deliveries should be ≈ 0):\n", _md(sanity),
         "\n## 4. Per-player pressure level, baseline and retention\n",
         f"P = mean v over H (primary `{C.PRIMARY_H}`), B = mean v over L, R = P − B. Also P/R on H20 and H33 within "
         "innings. Batters: deliveries faced as striker (wides excluded). Bowlers: all deliveries bowled. Simple standard "
         "errors here; Phase 7 bootstraps over matches. Tables: `outputs/tables/player_pressure_batters.csv`, "
         "`outputs/tables/player_pressure_bowlers.csv`.\n",
         f"Players: {len(bat_t)} batters, {len(bowl_t)} bowlers. High-leverage deliveries per player:\n",
         _md(dist.reset_index().rename(columns={"index": "role"})),
         f"\nCorrelation of P with B across players with ≥{min_h} H deliveries: batters {r_bat:.2f}, bowlers {r_bowl:.2f} "
         "(`outputs/charts/value_P_vs_B_batters.png`, `..._bowlers.png`).\n",
         "No player is ranked in this phase."]
    (C.OUT / "04_value.md").write_text("\n".join(L))
