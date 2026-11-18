"""Phase 7: rating and ranking (DECISIONS.md P6B-4, P7-*, P7R-*, P7B-*).

Bivariate, component-wise empirical Bayes (second review). For each component k (runs; wicket = dismissals vs expected
x wicket cost from the Phase 2 resource table):
  H performance = core skill mu_k + pressure-specific deviation delta_k
  mu_k   : mean over ALL deliveries, shrunk toward the population by its own between-player variance
           (c_k = within-player per-ball variance / between-player variance, method of moments)
  delta_k: d = (H mean − L mean) − population (H − L) shift, shrunk by var(delta_k) estimated from data
           (var(d) − mean sampling variance; 0 if not positive -> delta_hat = 0)
  rating = sum_k (mu_hat_k + delta_hat_k). A component whose between-player variance is not estimable is a constant
  (dropped). Pressure deliveries count as evidence of core skill once; the H−L contrast carries only the deviation.
SE = bootstrap over the player's matches (200; hyper-parameters fixed). Rank score = rating − SE.
Eligibility: ≥ 2 × median H deliveries of regular player-seasons (≥10 matches); played in 2025 or 2026; retention gate
(exclude if centred R 95% interval entirely below 0); noise-aware delivery condition: exclude only if raw run value on H is
clearly below expectation (raw H + 1.28·SE < 0, i.e. the 80% interval lies below 0).
Tiers: walking down the ordered list, a new tier starts when the gap to the previous player's rank score exceeds the larger
of the two players' SEs.
"""
import numpy as np
import pandas as pd

from . import config as C
from . import signal_recovery as SR

N_BOOT = 200          # hyper-parameter intervals
N_BOOT_SE = 2000      # player SEs, raw-H SE and retention intervals (P7F-4)
SEEDS = [0, 1, 2, 3, 4]
METHOD = {"bat": "component", "bowl": "bivariate"}   # P7F-1, P7F-2
LAMBDAS = [1.0, 0.8, 0.6, 0.4]
PROC_H = {"bat": ["sr_delta", "dot_suppression", "boundary_elevation", "wicket_security"],
          "bowl": ["dot_rate", "boundary_suppression", "economy_suppression", "wicket_taking"]}
NICE = {"sr_delta": "strike-rate delta", "dot_suppression": "dot suppression", "boundary_elevation": "boundary elevation",
        "wicket_security": "wicket security", "dot_rate": "dot rate", "boundary_suppression": "boundary suppression",
        "economy_suppression": "economy suppression", "wicket_taking": "wicket-taking"}
# previously reviewed method (kept for comparison): prior from ALL deliveries of one baseline metric
CURRENT_BASE = {"bat": "rv", "bowl": "economy_suppression"}
CURRENT_TAU = {"bat": 0.002878, "bowl": 0.0}
COMPS = ("rv_runs", "rv_wkt")


def load():
    comp = pd.read_parquet(C.PROC / "metric_components.parquet")
    m = pd.read_parquet(C.PROC / "matches.parquet")[["match_id", "strength"]]
    comp = comp.merge(m, on="match_id", how="left")
    lv = pd.read_parquet(C.PROC / "leverage.parquet")[["match_id", "innings", "innings_ball_num", "H20_within", "H33_within"]]
    comp = comp.merge(lv, on=["match_id", "innings", "innings_ball_num"], how="left")
    comp["H"] = comp.H.astype(bool)
    return comp


# ------------------------------------------------------------------------------------------------ sufficient statistics
def stats(comp, side, keys):
    """Per key (player, or player x match, or player x year): sums needed by both rating methods."""
    c = comp[comp.side == side]
    out = []

    def add(rows, name):
        g = rows.groupby(keys).val
        out.append(pd.DataFrame({f"s_{name}": g.sum(), f"n_{name}": g.size()}))
    for k in COMPS:
        add(c[(c.metric == k) & ~c.H], f"{k}_L")
        add(c[(c.metric == k) & c.H], f"{k}_H")
        add(c[c.metric == k], f"{k}_all")
    add(c[(c.metric == "rv") & c.H], "rvH")
    add(c[c.metric == CURRENT_BASE[side]], "base_all")
    pv = c[c.metric == "pressure_v"]
    add(pv[pv.H], "pvH")
    add(pv[~pv.H], "pvL")
    return pd.concat(out, axis=1).fillna(0.0)


def means(st):
    with np.errstate(invalid="ignore", divide="ignore"):
        return {k[2:]: st[k] / st["n_" + k[2:]] for k in st.columns if k.startswith("s_")}


def shrink_params(rows):
    """mu and c = sigma2/tau2 for player means of `rows` (method of moments)."""
    g = rows.groupby("player_id").val.agg(["mean", "size"])
    s2 = float(rows.val.var())
    mu = np.average(g["mean"], weights=g["size"])
    tau2 = max(np.average((g["mean"] - mu) ** 2, weights=g["size"]) - s2 * len(g) / g["size"].sum(), 1e-12)
    return {"mu": mu, "c": s2 / tau2, "s2": s2, "tau2": tau2}


def thetas(st, mn, P):
    if P["method"] == "component":                 # only components with estimable between-player variance
        return [P["comps"][k]["mu"] + st[f"n_{k}_L"] / (st[f"n_{k}_L"] + P["comps"][k]["c"]) *
                (mn[f"{k}_L"].fillna(P["comps"][k]["mu"]) - P["comps"][k]["mu"]) for k in P["active"]]
    sp = P["base"]
    return [sp["mu"] + st.n_base_all / (st.n_base_all + sp["c"]) * (mn["base_all"].fillna(sp["mu"]) - sp["mu"])]


def fit(comp, side, method, rng):
    """Hyper-parameters from career data."""
    c = comp[comp.side == side]
    st = stats(comp, side, ["player_id"])
    mn = means(st)
    s2H = float(c[(c.metric == "rv") & c.H].val.var())
    P = {"method": method, "s2H": s2H}
    if method == "component":
        P["comps"] = {k: shrink_params(c[(c.metric == k) & ~c.H]) for k in COMPS}
        # a component whose between-player variance is not estimable (tau2 at the floor) shrinks every player to the
        # same value: it is a constant, so it is dropped (coefficient 0) instead of entering a collinear regression
        P["active"] = [k for k in COMPS if P["comps"][k]["tau2"] > 1e-9]
        nL = st["n_rv_runs_L"].values
    else:
        P["base"] = shrink_params(c[c.metric == CURRENT_BASE[side]])
        nL = st.n_base_all.values
    X = np.column_stack([t.values for t in thetas(st, mn, P)])
    xH, nH = mn["rvH"].values, st.n_rvH.values
    ok = (nH >= 100) & (nL >= 100)
    tdev = 0.0 if method == "component" else CURRENT_TAU[side]
    est = tdev
    for _ in range(3):                                       # regression weights depend on tau2_dev: iterate
        w = 1 / (s2H / nH[ok] + tdev + 1e-6)
        A = np.c_[np.ones(ok.sum()), X[ok]]
        coef = np.linalg.lstsq(A * np.sqrt(w)[:, None], xH[ok] * np.sqrt(w), rcond=None)[0]
        if method != "component":
            break
        resid = xH[ok] - A @ coef
        postv = sum(coef[1 + i] ** 2 * (P["comps"][k]["tau2"] * P["comps"][k]["c"] / (st[f"n_{k}_L"].values[ok] + P["comps"][k]["c"]))
                    for i, k in enumerate(P["active"]))
        est = np.var(resid, ddof=len(coef)) - np.mean(s2H / nH[ok]) - np.mean(postv)
        tdev = max(est, 0.0)
    P["coef"], P["tau2_dev"], P["tau2_dev_estimate"] = coef, tdev, est
    if method == "component":
        sv = s2H / nH[ok]
        bt = []
        for _ in range(N_BOOT):
            i = rng.integers(0, ok.sum(), ok.sum())
            bt.append(np.var(resid[i], ddof=len(coef)) - sv[i].mean() - postv[i].mean())
        P["tau2_dev_ci"] = tuple(np.percentile(bt, [2.5, 97.5]))
        P["players_for_tau"] = int(ok.sum())
        P["reliability_at_median_nL"] = {}
        for k in COMPS:
            if P["comps"][k]["tau2"] <= 1e-9:
                P["reliability_at_median_nL"][k] = 0.0
                continue
            nmed = float(np.median(st[f"n_{k}_L"][st[f"n_{k}_L"] > 0]))
            P["reliability_at_median_nL"][k] = round(nmed / (nmed + P["comps"][k]["c"]), 3)
    pv = c[c.metric == "pressure_v"]
    P["pop_shift"] = float(pv.val[pv.H].mean() - pv.val[~pv.H].mean())
    return P


def rate(st, P):
    """Ratings for every row of a statistics table (any grouping)."""
    mn = means(st)
    th = thetas(st, mn, P)
    prior = P["coef"][0] + sum(P["coef"][1 + i] * t for i, t in enumerate(th))
    nH = st.n_rvH
    k = P["tau2_dev"] / (P["tau2_dev"] + P["s2H"] / nH.clip(lower=1))
    xH = mn["rvH"].fillna(prior)
    rating = prior + k * (xH - prior)
    R = (mn["pvH"] - mn["pvL"]) - P["pop_shift"]
    return pd.DataFrame({"rating": rating, "prior": prior, "pressure_dev": rating - prior, "k_update": k.where(nH > 0, 0.0), "H_mean_raw": mn["rvH"],
                         "n_H": nH, "R": R})


def fit_biv(comp, side, rng):
    c = comp[comp.side == side]
    st = stats(comp, side, ["player_id"])
    mn = means(st)
    P = {"method": "bivariate", "comps": {}}
    for k in COMPS:
        rows = c[c.metric == k]
        sp = shrink_params(rows)
        sp["active"] = sp["tau2"] > 1e-9
        H, L = rows[rows.H].val, rows[~rows.H].val
        sp["d0"] = float(H.mean() - L.mean())                       # population (H − L) shift
        sp["s2H"], sp["s2L"] = float(H.var()), float(L.var())
        d = mn[f"{k}_H"] - mn[f"{k}_L"] - sp["d0"]
        sv = sp["s2H"] / st[f"n_{k}_H"].clip(lower=1) + sp["s2L"] / st[f"n_{k}_L"].clip(lower=1)
        ok = ((st[f"n_{k}_H"] >= 100) & (st[f"n_{k}_L"] >= 100)).values
        dd, vv = d.values[ok], sv.values[ok]
        est = float(np.var(dd, ddof=1) - vv.mean())
        bt = []
        for _ in range(N_BOOT):
            i = rng.integers(0, ok.sum(), ok.sum())
            bt.append(np.var(dd[i], ddof=1) - vv[i].mean())
        sp.update({"tau2_delta_estimate": est, "tau2_delta": max(est, 0.0), "tau2_delta_ci": tuple(np.percentile(bt, [2.5, 97.5])),
                   "players_for_delta": int(ok.sum())})
        P["comps"][k] = sp
    pv = c[c.metric == "pressure_v"]
    P["pop_shift"] = float(pv.val[pv.H].mean() - pv.val[~pv.H].mean())
    return P


def rate_biv(st, P):
    mn = means(st)
    core, pdev, kw = 0.0, 0.0, None
    for k, sp in P["comps"].items():
        n = st[f"n_{k}_all"]
        mu = sp["mu"] + (n / (n + sp["c"]) * (mn[f"{k}_all"].fillna(sp["mu"]) - sp["mu"]) if sp["active"] else 0.0)
        sv = sp["s2H"] / st[f"n_{k}_H"].clip(lower=1) + sp["s2L"] / st[f"n_{k}_L"].clip(lower=1)
        w = sp["tau2_delta"] / (sp["tau2_delta"] + sv)
        has = (st[f"n_{k}_H"] > 0) & (st[f"n_{k}_L"] > 0)
        dlt = (w * (mn[f"{k}_H"] - mn[f"{k}_L"] - sp["d0"])).where(has, 0.0).fillna(0.0)
        core = core + mu
        pdev = pdev + dlt
        if k == "rv_runs":
            kw = w.where(has, 0.0)
    R_ = (mn["pvH"] - mn["pvL"]) - P["pop_shift"]
    return pd.DataFrame({"rating": core + pdev, "prior": core, "pressure_dev": pdev, "k_update": kw,
                         "H_mean_raw": mn["rvH"], "n_H": st.n_rvH, "R": R_})


def rate_any(st, P):
    return rate_biv(st, P) if P["method"] == "bivariate" else rate(st, P)


def fit_any(comp, side, method, rng):
    return fit_biv(comp, side, rng) if method == "bivariate" else fit(comp, side, method, rng)


def build(comp, side, Hcol, min_h, rngs, label, method="bivariate", delivery="strict", active_from=2025):
    """rngs: one generator per seed. SE and raw-H SE from seed 0 (2,000 resamples); the retention gate uses the pooled
    resamples of all seeds (5 x 2,000 = 10,000) (P7G-1). Per-seed retention intervals are kept for reporting."""
    c = comp.assign(H=comp[Hcol].astype(bool))
    P = fit_any(c, side, method, SR.sub_rng(side, "E"))
    st = stats(c, side, ["player_id"])
    t = rate_any(st, P)
    yrs = c[(c.side == side) & (c.metric == "rv")].groupby("player_id").year.agg(seasons="nunique", last_season="max",
                                                                                  first_season="min")
    t = t.join(yrs)
    cand = t.index[(t.n_H >= min_h) & (t.last_season >= active_from)]
    pm = stats(c[c.player_id.isin(cand)], side, ["player_id", "match_id"])
    se, rlo, rhi, hse, per_seed = {}, {}, {}, {}, {}
    for p in cand:                                        # bootstrap over the player's matches
        x = pm.loc[p]
        Rs = []
        for i, rng in enumerate(rngs):
            idx = rng.integers(0, len(x), (N_BOOT_SE, len(x)))
            B = pd.DataFrame({col: x[col].values[idx].sum(1) for col in x.columns})
            rb = rate_any(B, P)
            if i == 0:
                se[p] = float(np.nanstd(rb.rating, ddof=1))
                hse[p] = float(np.nanstd(rb.H_mean_raw, ddof=1))
            Rs.append(rb.R.values)
            per_seed[(p, i)] = tuple(np.nanpercentile(rb.R.values, [2.5, 97.5]))
        rlo[p], rhi[p] = np.nanpercentile(np.concatenate(Rs), [2.5, 97.5])   # pooled interval -> gate
    t["se"] = pd.Series(se)
    t["H_raw_se"] = pd.Series(hse)
    t["R_lo"], t["R_hi"] = pd.Series(rlo), pd.Series(rhi)
    for i in range(len(rngs)):
        t[f"R_lo_seed{i}"] = pd.Series({p: per_seed[(p, i)][0] for p in cand})
        t[f"R_hi_seed{i}"] = pd.Series({p: per_seed[(p, i)][1] for p in cand})
    t["rank_score"] = t.rating - t.se
    t["eligible_volume"] = t.n_H >= min_h
    t["eligible_active"] = t.last_season >= active_from
    t["excluded_by_retention"] = t.R_hi < 0
    t["delivers_strict"] = t.H_mean_raw > 0                              # demonstrated delivery under pressure (P7F-3)
    t["delivers_noise_aware"] = ~(t.H_mean_raw + 1.28 * t.H_raw_se < 0)  # sensitivity: excluded only if clearly below
    t["delivers_on_H"] = t.delivers_strict if delivery == "strict" else t.delivers_noise_aware
    t["eligible"] = t.eligible_volume & t.eligible_active & ~t.excluded_by_retention & t.delivers_on_H
    t = t.rename_axis("player_id").reset_index()
    pre = t.eligible_volume & t.eligible_active
    if method == "bivariate":
        info = {"variant": label, "side": side, "method": method, "H": Hcol, "min_H": min_h,
                "tau2_dev_used": round(sum(sp["tau2_delta"] for sp in P["comps"].values()), 6)}
        for k, sp in P["comps"].items():
            nm_ = k.replace("rv_", "")
            info[f"{nm_}: c (core shrinkage)"] = round(sp["c"], 1) if sp["active"] else "not estimable → dropped"
            info[f"{nm_}: tau2_core"] = round(sp["tau2"], 6)
            info[f"{nm_}: tau2_delta est [95% CI]"] = "{:.5f} [{:.5f}, {:.5f}]".format(sp["tau2_delta_estimate"], *sp["tau2_delta_ci"])
        info.update({"volume_and_active": int(pre.sum()), "removed_by_retention": int((pre & t.excluded_by_retention).sum()),
                     "removed_by_delivery_condition": int((pre & ~t.excluded_by_retention & ~t.delivers_on_H).sum()),
                     "eligible": int(t.eligible.sum())})
        return t, info, P
    info = {"variant": label, "side": side, "method": method, "H": Hcol, "min_H": min_h,
            "tau2_dev_used": round(P["tau2_dev"], 6), "tau2_dev_estimate": round(float(P["tau2_dev_estimate"]), 6),
            "tau2_dev_ci95": "[{:.5f}, {:.5f}]".format(*P["tau2_dev_ci"]) if "tau2_dev_ci" in P else "",
            "prior_coef": dict(zip(["intercept"] + P.get("active", [CURRENT_BASE[side]]), np.round(P["coef"], 4).tolist())),
            "dropped_components": [k for k in COMPS if k not in P.get("active", COMPS)],
            "c_runs": round(P["comps"]["rv_runs"]["c"], 1) if "comps" in P else None,
            "c_wkt": (round(P["comps"]["rv_wkt"]["c"], 1) if P["comps"]["rv_wkt"]["tau2"] > 1e-9 else "not estimable (between-player variance ≈ 0)") if "comps" in P else None,
            "reliability_at_median_nL": P.get("reliability_at_median_nL"),
            "volume_and_active": int(pre.sum()), "removed_by_retention": int((pre & t.excluded_by_retention).sum()),
            "removed_by_delivery_condition": int((pre & ~t.excluded_by_retention & ~t.delivers_on_H).sum()),
            "eligible": int(t.eligible.sum())}
    return t, info, P


# ------------------------------------------------------------------------------------------------ out-of-sample comparison
def oos_compare(comp, side, P_cur, P_cmp):
    oos = SR.OOS(comp, side, SR.sub_rng(side, "OOS"))
    st = stats(comp, side, ["player_id", "year"])
    y, w = oos.df.y.values, oos.df.w.values
    preds, rows = {}, []
    for lab, P in (("previous (L-only component prior + H update)", P_cur),
                   ("bivariate (core skill from all deliveries + shrunk H−L deviation)", P_cmp)):
        p = oos.oof(oos.feature(rate_any(st, P).rating)[:, None])
        preds[lab] = p
        bt = [SR.wcorr(p[i], y[i], w[i]) for i in oos.boot]
        lo, hi = np.nanpercentile(bt, [2.5, 97.5])
        rows.append({"side": side, "rating": lab, "oos_r_next_season_rv_H": round(SR.wcorr(p, y, w), 3),
                     "ci95": f"[{lo:.3f}, {hi:.3f}]", "pairs": len(y)})
    a, b = list(preds.values())
    d = [SR.wcorr(b[i], y[i], w[i]) - SR.wcorr(a[i], y[i], w[i]) for i in oos.boot]
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return pd.DataFrame(rows), {"side": side, "bivariate_minus_previous_r": round(SR.wcorr(b, y, w) - SR.wcorr(a, y, w), 3),
                                "ci95": f"[{lo:.3f}, {hi:.3f}]", "clearly_worse": bool(hi < 0)}


# ------------------------------------------------------------------------------------------------ recency (prior) test
def _shrunk_mean(r, years_upto, lam):
    r = r[r.year <= years_upto]
    w = lam ** (years_upto - r.year.values).astype(float)
    t = r.assign(w=w, wv=w * r.val.values, w2=w ** 2).groupby("player_id").agg(sw=("w", "sum"), swv=("wv", "sum"), sw2=("w2", "sum"))
    t["x"], t["neff"] = t.swv / t.sw, t.sw ** 2 / t.sw2
    s2 = float(r.val.var())
    mu = np.average(t.x, weights=t.neff)
    tau2 = max(np.average((t.x - mu) ** 2, weights=t.neff) - s2 * len(t) / t.neff.sum(), 1e-12)
    return mu + t.neff / (t.neff + s2 / tau2) * (t.x - mu)


def recency_test(comp, side):
    """Recency weights on the L-only component prior (sum of the two shrunk components), cumulative history ≤ t."""
    oos = SR.OOS(comp, side, SR.sub_rng(side, "OOS"))
    y, w = oos.df.y.values, oos.df.w.values
    res, preds = [], {}
    for lam in LAMBDAS:
        vals = {}
        for t in sorted(oos.df.year.unique()):
            parts = [_shrunk_mean(comp[(comp.side == side) & (comp.metric == k)], t, lam) for k in COMPS]   # core skill: all deliveries
            vals[t] = parts[0].add(parts[1], fill_value=0)
        s = pd.concat(vals, names=["year", "player_id"]).swaplevel().sort_index()
        p = oos.oof(oos.feature(s)[:, None])
        preds[lam] = p
        bt = [SR.wcorr(p[i], y[i], w[i]) for i in oos.boot]
        lo, hi = np.nanpercentile(bt, [2.5, 97.5])
        res.append({"lambda": lam, "oos_r": round(SR.wcorr(p, y, w), 3), "ci95": f"[{lo:.3f}, {hi:.3f}]"})
    best = max([l_ for l_ in LAMBDAS if l_ < 1], key=lambda l_: [x["oos_r"] for x in res if x["lambda"] == l_][0])
    g = [SR.wcorr(preds[best][i], y[i], w[i]) - SR.wcorr(preds[1.0][i], y[i], w[i]) for i in oos.boot]
    lo, hi = np.nanpercentile(g, [2.5, 97.5])
    return pd.DataFrame(res), {"best_lambda": best, "gain_vs_no_recency": round(SR.wcorr(preds[best], y, w) - SR.wcorr(preds[1.0], y, w), 3),
                               "gain_ci95": f"[{lo:.3f}, {hi:.3f}]", "kept": bool(lo > 0)}


# ------------------------------------------------------------------------------------------------ presentation helpers
def names_and_teams(side, dup_names=()):
    b = pd.read_parquet(C.PROC / "balls.parquet")
    m = pd.read_parquet(C.PROC / "matches.parquet")
    tn = pd.concat([m[["home_team_id", "home_team"]].set_axis(["id", "n"], axis=1),
                    m[["away_team_id", "away_team"]].set_axis(["id", "n"], axis=1)]).drop_duplicates("id").set_index("id").n
    pid, nm, team = ("striker_id", "striker_name", "batting_team_id") if side == "bat" else ("bowler_id", "bowler_name", "bowling_team_id")
    x = b.sort_values("match_date")
    names = x.groupby(pid)[nm].agg(lambda s: str(s.iloc[-1])).astype(str)
    last_team = x.groupby(pid)[team].agg(lambda s: s.iloc[-1]).map(tn).astype(str)
    teams = x.groupby(pid)[team].agg(lambda s: ", ".join(sorted(set(tn[i] for i in s.unique())))).astype(str)
    dup = names[names.duplicated(keep=False) | names.isin(set(dup_names))]     # duplicates anywhere in the data (D11)
    shown = names.copy()
    shown[dup.index] = dup + " (" + teams[dup.index] + ")"
    return shown, last_team


def conventional(side):
    b = pd.read_parquet(C.PROC / "balls.parquet")              # all kept matches incl. rain-shortened (conventional stats)
    b = b[~b.is_penalty]
    if side == "bat":
        f = b[b.ball_faced & ~b.is_wide]
        t = pd.DataFrame({"runs": f.groupby("striker_id").batter_runs.sum(), "balls": f.groupby("striker_id").size()})
        outs = b[b.is_wicket].groupby("dismissed_player_id").size()
        outs.index = outs.index.astype(int)
        t = t.join(outs.rename("outs")).fillna({"outs": 0})
        t["average"] = t.runs / t.outs.replace(0, np.nan)
        t["strike_rate"] = 100 * t.runs / t.balls
        return t
    lg = b[b.legal]
    t = pd.DataFrame({"runs": b.groupby("bowler_id").bowler_runs.sum(), "legal_balls": lg.groupby("bowler_id").size(),
                      "wickets": b.groupby("bowler_id").bowler_wicket.sum()})
    t["economy"] = 6 * t.runs / t.legal_balls
    return t


def process_strengths(comp, side, players):
    c = comp[(comp.side == side) & comp.metric.isin(PROC_H[side]) & comp.H]
    m = c.groupby(["player_id", "metric"]).val.mean().unstack()
    m = m.loc[m.index.isin(players)]
    z = (m - m.mean()) / m.std()
    return z.apply(lambda row: "; ".join(f"{NICE[k]} {v:+.2f}σ" for k, v in row.dropna().sort_values(ascending=False).head(3).items()), axis=1)


def min_h_rule(comp, side):
    x = comp[(comp.side == side) & (comp.metric == "rv")]
    ps = x.groupby(["player_id", "year"]).agg(m=("match_id", "nunique"), h=("H", "sum"))
    med = float(ps[ps.m >= 10].h.median())
    return int(round(2 * med)), med, int((ps.m >= 10).sum())


def _md(df):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ main
def seed_rng(side, s):
    return np.random.default_rng([C.SEED, ("bat", "bowl").index(side), 70 + s])


def global_duplicate_names():
    """Names that map to more than one player_id anywhere in the data (ball file roles + mapping file)."""
    b = pd.read_csv(C.RAW / "ball_by_ball.csv", usecols=["striker_id", "striker_name", "non_striker_id", "non_striker_name",
                                                       "bowler_id", "bowler_name"])
    mp = pd.read_csv(C.RAW / "competition_team_player_mapping.csv", usecols=["player_id", "player_name"])
    pairs = pd.concat([b[[i, n]].set_axis(["id", "name"], axis=1) for i, n in
                       (("striker_id", "striker_name"), ("non_striker_id", "non_striker_name"), ("bowler_id", "bowler_name"))]
                      + [mp.set_axis(["id", "name"], axis=1)]).drop_duplicates()
    ids = pairs.groupby("name").id.nunique()
    return set(ids[ids > 1].index), pairs


def run():
    comp = load()
    dup_names, name_pairs = global_duplicate_names()
    tops, infos, oos_rows, oos_diff, rec = {}, [], [], [], {}
    for side in ("bat", "bowl"):
        min_h, med, nreg = min_h_rule(comp, side)
        shown, last_team = names_and_teams(side, dup_names)
        roles = pd.read_csv(C.TABLES / ("roles_batters.csv" if side == "bat" else "roles_bowlers.csv")).set_index("player_id").role
        P_cur = fit(comp, side, "component", SR.sub_rng(side, "E"))
        P_cmp = fit_biv(comp, side, SR.sub_rng(side, "E"))
        o, dif = oos_compare(comp, side, P_cur, P_cmp)
        oos_rows.append(o)
        oos_diff.append(dif)
        method = METHOD[side]
        other = "bivariate" if method == "component" else "component"
        rec[side] = recency_test(comp, side)

        # ---- primary build: retention gate on the pooled 10,000 resamples (5 seeds x 2,000)
        rngs = [seed_rng(side, s_) for s_ in SEEDS]
        full, info, P = build(comp, side, "H", min_h, rngs, f"{side}: primary", method)
        infos.append(info)
        fi = full.set_index("player_id")
        # borderline = retention decision would differ between single seeds (Monte Carlo error at the interval edge):
        # reported, not excluded (P7G-1)
        hi_cols = [f"R_hi_seed{i}" for i in range(len(SEEDS))]
        cand_ = full[full.eligible_volume & full.eligible_active].copy()
        below = (cand_[hi_cols] < 0)
        flip = below.any(axis=1) & ~below.all(axis=1)
        bl_rows = []
        for _, r_ in cand_[flip].iterrows():
            bl_rows.append({"player": shown[r_.player_id], "player_id": int(r_.player_id),
                            **{f"seed {i}": f"[{r_[f'R_lo_seed{i}']:+.3f}, {r_[f'R_hi_seed{i}']:+.3f}]" for i in range(len(SEEDS))},
                            "pooled (10,000)": f"[{r_.R_lo:+.3f}, {r_.R_hi:+.3f}]",
                            "gate decision (pooled)": "excluded" if r_.R_hi < 0 else "passes",
                            "rating_per_100": round(100 * r_.rating, 2)})
        borderline_df = pd.DataFrame(bl_rows)
        borderline = []                                                    # not excluded for flipping
        seed_t10 = "n/a (single pooled gate)"

        # ---- sensitivity variants (seed 0)
        variants = {"H20": ("H20_within", min_h, False, method, "strict"), "H33": ("H33_within", min_h, False, method, "strict"),
                    "threshold −25%": ("H", int(round(min_h * 0.75)), False, method, "strict"),
                    "threshold +25%": ("H", int(round(min_h * 1.25)), False, method, "strict"),
                    "IPL only": ("H", None, True, method, "strict"),
                    "noise-aware delivery condition": ("H", min_h, False, method, "noise"),
                    (f"{'bivariate' if side == 'bat' else 'L-only (previous)'} method"): ("H", min_h, False, other, "strict")}
        e = full[full.eligible & ~full.player_id.isin(borderline)].sort_values("rank_score", ascending=False).reset_index(drop=True)
        ours = e.player_id.head(10).tolist()
        lists, sens_rows, appendix = {}, [], None
        for vname, (Hcol, mh, ipl, meth, deliv) in variants.items():
            cc = comp[comp.strength == "IPL"] if ipl else comp
            if mh is None:
                mh = min_h_rule(cc, side)[0]
            t, info, P = build(cc, side, Hcol, mh, rngs, f"{side}: {vname}", meth, deliv)
            infos.append(info)
            ev = t[t.eligible & ~t.player_id.isin(borderline)].sort_values("rank_score", ascending=False)
            lv = ev.player_id.head(10).tolist()
            lists[vname] = lv
            sens_rows.append({"variant": vname, "top10_overlap_with_primary": len(set(ours) & set(lv)),
                              "only_in_primary": ", ".join(shown[p] for p in ours if p not in lv),
                              "only_in_variant": ", ".join(shown[p] for p in lv if p not in ours)})
            if vname.startswith("bivariate"):
                appendix = ev.head(20).assign(player=lambda x: x.player_id.map(shown))
        if appendix is not None:
            appendix[["player", "rating", "se", "rank_score", "prior", "pressure_dev", "H_mean_raw", "n_H"]].assign(
                **{c_: lambda x, c_=c_: (100 * x[c_]).round(2) for c_ in ["rating", "se", "rank_score", "prior", "pressure_dev", "H_mean_raw"]}
            ).to_csv(C.TABLES / "appendix_bivariate_batters.csv", index=False)

        pre = full[full.eligible_volume & full.eligible_active]
        removed = {
            "retention gate (pooled 10,000 resamples)": [f"{shown[p]} (R {r:+.3f}, CI [{lo:+.3f}, {hi:+.3f}])" for p, r, lo, hi in
                                        pre[pre.excluded_by_retention][["player_id", "R", "R_lo", "R_hi"]].itertuples(index=False)],
            "delivery condition (raw H run value ≤ 0)": [
                f"{shown[p]} (raw H {100 * x:+.2f} ± {100 * se_:.2f}/100 balls, rating {100 * r:+.2f})" for p, x, se_, r in
                pre[~pre.excluded_by_retention & ~pre.delivers_on_H].sort_values("rating", ascending=False)[["player_id", "H_mean_raw", "H_raw_se", "rating"]].itertuples(index=False)],
            "borderline (single-seed gate decisions disagree; reported, not excluded)": [r_["player"] + f" (pooled {r_['pooled (10,000)']}, {r_['gate decision (pooled)']})" for r_ in bl_rows]}
        gap = e.rank_score.shift(1) - e.rank_score
        tol = np.maximum(e.se.shift(1), e.se)
        e["tier"] = (gap > tol).cumsum() + 1                                # new tier when gap > larger adjacent SE
        strengths = process_strengths(comp, side, e.player_id.values)
        top = e.head(20).copy()
        top.insert(0, "rank", np.arange(1, len(top) + 1))
        top.insert(1, "player", top.player_id.map(shown))
        top["latest_team"] = top.player_id.map(last_team)
        top["role"] = top.player_id.map(roles)
        top["strongest_process_on_H"] = top.player_id.map(strengths)
        top["R_ci95"] = [f"[{a_:+.3f}, {b_:+.3f}]" for a_, b_ in zip(top.R_lo, top.R_hi)]
        for col, src in (("rating_runs_per_100_H_balls", "rating"), ("se_per_100", "se"), ("rank_score_per_100", "rank_score"),
                         ("core_skill_per_100", "prior"), ("pressure_dev_per_100", "pressure_dev"),
                         ("raw_H_per_100", "H_mean_raw"), ("raw_H_se_per_100", "H_raw_se")):
            top[col] = (100 * top[src]).round(2)
        top["seasons_played"] = [f"{a_}–{b_} ({n_} seasons)" for a_, b_, n_ in zip(top.first_season, top.last_season, top.seasons)]
        # name check: exact data spelling; flag names shared by two player_ids anywhere in the data
        raw_ok = [((name_pairs.id == p) & (name_pairs.name == shown[p].split(" (")[0])).any() for p in top.player_id]
        top["name_matches_data"] = raw_ok
        top["name_shared_by_two_ids"] = [shown[p].split(" (")[0] in dup_names for p in top.player_id]
        cols = ["rank", "tier", "player", "latest_team", "role", "rating_runs_per_100_H_balls", "se_per_100", "rank_score_per_100",
                "core_skill_per_100", "pressure_dev_per_100", "raw_H_per_100", "raw_H_se_per_100", "k_update", "n_H",
                "seasons_played", "R", "R_ci95", "strongest_process_on_H", "name_matches_data", "name_shared_by_two_ids", "player_id"]
        top = top[cols].round({"k_update": 3, "R": 3})
        top.to_csv(C.TABLES / ("top10_batters.csv" if side == "bat" else "top10_bowlers.csv"), index=False)
        full.assign(player=full.player_id.map(shown), borderline=full.player_id.isin(borderline)).to_csv(
            C.TABLES / f"ratings_all_{side}.csv", index=False)
        borderline_df.to_csv(C.TABLES / f"borderline_{side}.csv", index=False)

        # comparisons
        elig = e.player_id.values
        plain = comp[(comp.side == side) & (comp.metric == "rv")].groupby("player_id").val.mean().reindex(elig).sort_values(ascending=False).index[:10]
        rawH = e.set_index("player_id").H_mean_raw.sort_values(ascending=False).index[:10]
        cv = conventional(side).reindex(elig)
        conv_lists = ({"batting average": cv.average.sort_values(ascending=False).index[:10],
                       "strike rate": cv.strike_rate.sort_values(ascending=False).index[:10]} if side == "bat" else
                      {"economy (lowest)": cv.economy.sort_values().index[:10], "wickets (most)": cv.wickets.sort_values(ascending=False).index[:10]})
        cmp_rows = [{"comparison": "plain all-delivery run value", "overlap": len(set(ours) & set(plain))},
                    {"comparison": "raw run value on H", "overlap": len(set(ours) & set(rawH))}] + \
                   [{"comparison": k, "overlap": len(set(ours) & set(v))} for k, v in conv_lists.items()]
        ee = e.assign(role=e.player_id.map(roles))
        g = ee.groupby("role").rating
        eta2 = float(((g.transform("mean") - ee.rating.mean()) ** 2).sum() / ((ee.rating - ee.rating.mean()) ** 2).sum())
        role_means = (100 * g.mean()).round(2).rename("mean_rating_per_100").reset_index().merge(g.size().rename("eligible").reset_index())

        # changes vs the last reviewed version
        prev_path = C.OUT / "history" / ("prev4_top_bat.csv" if side == "bat" else "prev4_top_bowl.csv")
        prev = pd.read_csv(prev_path) if prev_path.exists() else pd.DataFrame({"player_id": ours})   # snapshot of the last reviewed list
        p10 = prev.player_id.head(10).tolist()
        left, entered = {}, {}
        for p in [q for q in p10 if q not in ours]:
            r = fi.loc[p]
            if not r.delivers_on_H:
                why = f"removed by the delivery condition (raw H {100 * r.H_mean_raw:+.2f} per 100 balls ≤ 0)"
            elif r.excluded_by_retention:
                why = "removed by the retention gate"
            else:
                nr = int(np.where(e.player_id.values == p)[0][0]) + 1
                why = (f"now rank {nr}: rating {100 * r.rating:+.2f} (core {100 * r.prior:+.2f}, pressure dev {100 * r.pressure_dev:+.2f}), "
                       f"rank score {100 * r.rank_score:+.2f}")
            left[shown[p]] = why
        for p in [q for q in ours if q not in p10]:
            r = fi.loc[p]
            old = prev.player_id.tolist()
            was = f"was rank {old.index(p) + 1}" if p in old else ("was excluded as borderline under the seed-flip rule (now passes the pooled retention gate)"
                                                                   if not prev.player_id.isin([p]).any() and fi.loc[p].R_lo < 0 <= fi.loc[p].R_hi else "was outside the last top 20")
            entered[shown[p]] = (f"{was}; now rating {100 * r.rating:+.2f} (core {100 * r.prior:+.2f}, pressure dev "
                                 f"{100 * r.pressure_dev:+.2f}, raw H {100 * r.H_mean_raw:+.2f}), rank score {100 * r.rank_score:+.2f}")
        tops[side] = {"top": top, "cmp": pd.DataFrame(cmp_rows), "eta2": eta2, "role_means": role_means,
                      "role_mix": top.role.value_counts().rename_axis("role").reset_index(name="in_top20"),
                      "sens": pd.DataFrame(sens_rows), "min_h": (min_h, med, nreg), "n_eligible": len(e), "method": method,
                      "conv_names": {k: [shown.get(i, i) for i in v] for k, v in conv_lists.items()},
                      "left": left, "entered": entered, "removed": removed, "borderline": borderline_df, "seed_t10": seed_t10}
    infos = pd.DataFrame(infos)
    infos.to_csv(C.TABLES / "rating_variants.csv", index=False)
    oos_rows = pd.concat(oos_rows)
    oos_rows.to_csv(C.TABLES / "rating_method_oos.csv", index=False)
    write_report(tops, infos, oos_rows, pd.DataFrame(oos_diff), rec)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 25)
    pd.set_option("display.max_colwidth", 90)
    print(oos_rows.to_string(index=False))
    print(pd.DataFrame(oos_diff).to_string(index=False))
    print(infos.to_string(index=False))
    for side in ("bat", "bowl"):
        d = tops[side]
        print(d["top"].drop(columns=["player_id"]).to_string(index=False))
        print(d["cmp"].to_string(index=False))
        print(d["sens"].to_string(index=False))
        print("REMOVED:", d["removed"])
        print("LEFT:", d["left"])
        print("ENTERED:", d["entered"])
        print("eta2", round(d["eta2"], 3), d["role_mix"].to_dict("records"))
        print("conv", d["conv_names"])
    print({s: rec[s][1] for s in rec})


def write_report(tops, infos, oos_rows, oos_diff, rec):
    L = ["# Phase 7 — Final rating and ranking\n", __doc__.strip(), "",
         "## Final method (owner decision, Phase 7 rule: out-of-sample prediction first, then use of evidence)\n",
         "- **Batters: L-only component prior + component-wise H update** (prediction r 0.184 vs 0.103 for bivariate). The "
         "bivariate batter rating is kept as a tested alternative (`outputs/tables/appendix_bivariate_batters.csv`).",
         "- **Bowlers: bivariate component rating** (prediction equivalent, 0.028 vs 0.019; uses all deliveries as evidence of core skill).",
         "- **Delivery condition: a requirement, not an estimate.** To appear in a final ranked list a player must have "
         "demonstrated delivery under pressure — raw run value on H above expectation (> 0) (brief: 'deliver when it matters'). "
         "The noise-aware version (exclude only if raw H + 1.28·SE < 0) is reported as a sensitivity check.",
         "- **Retention gate on pooled resamples:** SEs and raw-H SEs use 2,000 bootstrap resamples (fixed seed 0); the retention "
         "interval pools 5 seeds x 2,000 = 10,000 resamples and the gate is applied once (exclude only if the 95% interval is "
         "entirely below 0). Players whose single-seed decisions would disagree are reported as borderline, not excluded.\n",
         "## Changes after the second Phase 7 review (owner instruction)\n",
         "1. Bivariate empirical Bayes replaces the L-only prior: core skill from all deliveries + shrunk pressure-specific "
         "deviation from (H − L), component-wise.",
         "2. Bowler wicket component re-estimated on all deliveries.",
         "3. Noise-aware delivery condition: excluded only if raw H + 1.28·SE < 0.",
         "4. Tiers mark positions that are not statistically separable.\n",
         "## Out-of-sample: bivariate rating vs the previous (L-only prior) rating\n",
         "Prediction of next-season run value on H (season-t ratings, hyper-parameters from career data; folds grouped by "
         "player).\n",
         _md(oos_rows), "", _md(oos_diff), "",
         "## Recency weighting on core skill (all deliveries, both components)\n"]
    for side, (tab, info) in rec.items():
        L += [f"**{'Batters' if side == 'bat' else 'Bowlers'}**", _md(tab), f"\n{info}\n"]
    L += ["## Fitted parameters and eligibility counts per variant\n", _md(infos), ""]
    for side, d in tops.items():
        nm = "Batters" if side == "bat" else "Bowlers"
        mh, med, nreg = d["min_h"]
        L += [f"\n## {nm}: top 10 and next 10 (method: {d['method']})\n",
              f"Eligibility: ≥ {mh} H deliveries (2 × median {med:.0f} in {nreg} regular player-seasons), played in 2025 or "
              f"2026, retention gate, noise-aware delivery condition. Eligible: {d['n_eligible']}. Ratings: runs per 100 H balls vs "
              "expectation = core skill + pressure deviation; `k_update` = shrinkage weight on the player's own H − L "
              "deviation (runs component). Tier: positions in one tier are within one SE of their neighbours (not separable).\n",
              _md(d["top"].drop(columns=["player_id"])),
              "\n### Removed by the eligibility conditions (had enough H deliveries and were active)\n"]
        for k, v in d["removed"].items():
            L.append(f"- **{k}** ({len(v)}): " + ("; ".join(v) if v else "none"))
        L += ["\n### Retention gate: pooled bootstrap (5 seeds x 2,000 = 10,000 resamples)\n",
              "Borderline = players whose single-seed retention decisions disagree (Monte Carlo error at the interval edge). "
              "They are reported, not excluded; the gate uses the pooled interval (P7G-1).\n",
              (_md(d["borderline"]) if len(d["borderline"]) else "No borderline players.")]
        L += ["\n### Changes in the top 10 vs the previous version\n",
              "- Left: " + ("; ".join(f"**{k}** — {v}" for k, v in d["left"].items()) or "none"),
              "- Entered: " + ("; ".join(f"**{k}** — {v}" for k, v in d["entered"].items()) or "none"),
              "\n### Comparisons (overlap with our top 10)\n", _md(d["cmp"]),
              f"\nConventional top 10s: {d['conv_names']}\n",
              "Role mix of the top 20:\n", _md(d["role_mix"]),
              f"\nRating vs role (eligible players): η² = {d['eta2']:.3f}\n", _md(d["role_means"]),
              "\n### Sensitivity — top-10 overlap with the primary list\n", _md(d["sens"])]
    (C.OUT / "07_rating.md").write_text("\n".join(L))
