"""Phase 2: leak-free venue par + win-probability (WP) model.

WP is always for the BATTING side of the ball.

1st innings: sign-constrained logistic regression.
2nd innings: sign-constrained logistic regression (main) blended with a sign-constrained endgame
logistic for the last overs. Both use features derived from a Markov ball-outcome model of the chase
(WASP-style): the exact chase win probability by backward induction, and the mean / s.d. of runs
still to come given (balls left, wickets lost).

Every feature is monotone in the state, and its coefficient sign is enforced (L-BFGS-B bounds), so WP
cannot fall when wickets in hand or balls left rise, or when runs needed fall (checked by perturbation;
the endgame blend is the only place this could fail, and the check reports it).

Leakage: 5-fold cross-validation grouped by match. Every ball of a training match is scored with
models AND Markov tables built without that match. Matches not used for training (ties, rain-revised,
non-reconciling) are scored with the all-training-data model.

Public API reused by later phases:
  joblib.load(data_processed/wp_models.joblib) -> models
  wp_state(df, models) -> WP of batting side for arbitrary states (terminal states exact)
  df columns: innings, score, wkts, legal, max_balls, max_balls2, target, par_total, strength, fold
"""
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .plotstyle import plt, save, SERIES, TEXT2

N_FOLDS = 5
FOLDS_ALL = list(range(N_FOLDS)) + [-1]          # -1 = model trained on all training matches
STRENGTHS = ["IPL", "FM_T20I", "Weaker_T20I"]
G = ["g_FM_T20I", "g_Weaker_T20I"]
WB_K = [3, 6, 12, 24, 48, 120]
WB = [f"wb{k}" for k in WB_K]                    # wickets in hand x min(balls left, k)/k

FEATS1 = ["score_vs_par", "balls_left", "wkts_in_hand", "resources"] + G
SIGN1 = [1, 0, 1, 1, 0, 0]
FEATS2 = ["zmv", "runs_needed", "ln_need", "rrr_c", "chase_z"] + WB + G
SIGN2 = [1, -1, -1, -1, 1] + [1] * len(WB) + [0, 0]
FEATSE = ["zmv", "dp_logit", "runs_needed", "ln_need", "rrr_c"] + WB + G
SIGNE = [1, 1, -1, -1, -1] + [1] * len(WB) + [0, 0]
END_FIT_MAX = 18            # endgame model trained on chase balls with <= 18 balls left (starting value)
BLEND_LO, BLEND_HI = 12, 24  # endgame weight 1 at <=12 balls left, 0 at >=24, linear between (starting values)

# comparison model (monotonic gradient boosting): raw features
HFEATS1, HMONO1 = ["score_vs_par", "balls_left", "wkts_in_hand", "strength_code"], [1, 0, 1, 0]
HFEATS2, HMONO2 = ["runs_needed", "balls_left", "wkts_in_hand", "rrr_minus_par", "strength_code"], [-1, 1, 1, -1, 0]
HGB_PARAMS = dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=3000, l2_regularization=1.0,
                  early_stopping=False, random_state=C.SEED)

# Markov outcome model
OUT_RUNS = np.array([0, 1, 2, 3, 4, 6])        # legal non-wicket outcomes; class 6 = wicket, 7 = wide/no-ball
N_OUT = 8
R_MAX = 400
ALPHAS = [10, 30, 100, 300, 1000]


# ================================================================= constrained logistic
class SignedLogit:
    """L2 logistic regression (penalty as sklearn C=1 on standardised inputs) with coefficient sign bounds."""

    def __init__(self, signs, l2=1.0):
        self.signs, self.l2 = list(signs), l2

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-12
        Z1 = np.c_[np.ones(len(X)), (X - self.mu_) / self.sd_]
        y = np.asarray(y, float)

        def f(w):
            z = Z1 @ w
            p = 1 / (1 + np.exp(-z))
            g = Z1.T @ (p - y)
            g[1:] += self.l2 * w[1:]
            return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * self.l2 * np.sum(w[1:] ** 2), g

        bounds = [(None, None)] + [(0, None) if s > 0 else (None, 0) if s < 0 else (None, None) for s in self.signs]
        self.w_ = minimize(f, np.zeros(Z1.shape[1]), jac=True, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 3000}).x
        self.coef_raw_ = self.w_[1:] / self.sd_
        return self

    def predict_proba(self, X):
        Z = (np.asarray(X, float) - self.mu_) / self.sd_
        p = 1 / (1 + np.exp(-(self.w_[0] + Z @ self.w_[1:])))
        return np.c_[1 - p, p]


# ================================================================= venue par
def venue_par(b, m):
    """Leak-free par first-innings total per match.

    league_i = mean 1st-inn total of same-group matches in the previous PAR_WINDOW_DAYS (strictly earlier dates)
    par_i    = league_i + n_v/(n_v + k) * mean(prior venue residuals vs their own league mean)
    k        = within-venue / between-venue variance of residuals (method of moments, one global number)
    """
    kept = m[m.kept].copy()
    h = ((kept.max_balls1 == 120) & ~kept.revised_recorded & ~kept.revised_unrecorded).to_numpy(dtype=bool)
    d = kept.match_date.values.astype("datetime64[D]").astype(np.int64)
    tot = kept.tot1.values.astype(float)
    grp, ven = kept.strength.values, kept.venue_id.values

    dd = d[:, None] - d[None, :]
    prior = (dd > 0) & (dd <= C.PAR_WINDOW_DAYS) & h[None, :]
    global_mean = tot[h].mean()

    def masked_mean(mask):
        n = mask.sum(1)
        return np.where(n > 0, (mask * tot[None, :]).sum(1) / np.maximum(n, 1), np.nan), n

    lg, n_g = masked_mean(prior & (grp[:, None] == grp[None, :]))
    la, n_a = masked_mean(prior)
    league = np.where(n_g >= 10, lg, np.where(n_a > 0, la, global_mean))
    league_src = np.where(n_g >= 10, "group", np.where(n_a > 0, "all_groups", "global_fallback"))

    resid = tot - league
    r = pd.DataFrame({"v": ven[h], "r": resid[h]})
    g = r.groupby("v").r
    n_v, var_w = g.size(), g.var(ddof=1)
    within = np.average(var_w[n_v >= 2], weights=(n_v[n_v >= 2] - 1))
    between = max(np.average((g.mean() - r.r.mean()) ** 2, weights=n_v) - within * (len(n_v) / n_v.sum()), 1e-6)
    k = within / between

    vm = prior & (ven[:, None] == ven[None, :])
    nv = vm.sum(1)
    vres = np.where(nv > 0, (vm * resid[None, :]).sum(1) / np.maximum(nv, 1), 0.0)
    par = league + nv / (nv + k) * vres

    out = pd.DataFrame({"match_id": kept.match_id.values, "par_total": par.astype("float32"),
                        "league_mean": league, "league_src": league_src, "venue_prior_n": nv,
                        "venue_resid_mean": vres})
    info = {"k_shrink": float(k), "within_var": float(within), "between_var": float(between),
            "hist_matches": int(h.sum()), "league_src_counts": pd.Series(league_src).value_counts().to_dict()}
    return out, info


def stage_curve(b, m):
    """Average share of the final 1st-innings total reached after k legal balls (shape only, no outcomes)."""
    ok = m.loc[m.kept & (m.max_balls1 == 120) & ~m.revised_recorded & ~m.revised_unrecorded, "match_id"]
    x = b[(b.innings == 1) & b.match_id.isin(ok)]
    s = x.groupby(["match_id", "legal_after"]).score_after.max().unstack().reindex(columns=range(0, 121))
    s[0] = 0
    s = s.ffill(axis=1)
    return (s.mean() / s[120].mean()).values


# ================================================================= Markov chase model
def outcome_data(rows):
    """(over-left bucket, wickets lost, outcome class) for 2nd-innings deliveries in `rows`."""
    x = rows[(rows.innings == 2) & ~rows.is_penalty]
    tr = x.total_runs.values.astype(int)
    legal_cls = np.select([tr <= 0, tr == 1, tr == 2, tr == 3, tr <= 5], [0, 1, 2, 3, 4], 5)
    code = np.where(~x.legal.values, 7, np.where(x.is_wicket.values, 6, legal_cls)).astype(int)
    ob = ((x.balls_left.values - 1) // 6).clip(0, 19).astype(int)
    wl = x.wkts_before.values.clip(0, 9).astype(int)
    return ob, wl, code, x.fold.values


def outcome_table(ob, wl, code, alpha):
    """p(outcome | over-left bucket, wickets lost), smoothed cell -> (over, wicket group) -> over."""
    c = np.zeros((20, 10, N_OUT))
    np.add.at(c, (ob, wl, code), 1)
    grand = c.sum(1) + 1
    pg = grand / grand.sum(-1, keepdims=True)
    out = np.empty_like(c)
    for gi in ([0, 1, 2], [3, 4, 5], [6, 7, 8, 9]):
        pc = c[:, gi].sum(1)
        parent = (pc + alpha * pg) / (pc.sum(-1, keepdims=True) + alpha)
        out[:, gi] = (c[:, gi] + alpha * parent[:, None]) / (c[:, gi].sum(-1, keepdims=True) + alpha)
    return out


def choose_alpha(ob, wl, code, fold):
    res = {}
    for a in ALPHAS:
        ll = 0.0
        for fo in range(N_FOLDS):
            t = outcome_table(ob[fold != fo], wl[fold != fo], code[fold != fo], a)
            te = fold == fo
            ll -= np.log(t[ob[te], wl[te], code[te]]).sum()
        res[a] = round(ll / len(code), 5)
    return min(res, key=res.get), res


def chase_dp(P):
    """V[b, w, r] = P(chase won) with b balls left, w wickets lost, r runs needed (r=0 won; r=1 level -> tie 0.5 at end)."""
    V = np.zeros((121, 10, R_MAX + 1))
    V[:, :, 0] = 1.0
    V[0, :, 1] = 0.5
    rr = np.arange(R_MAX + 1)
    allout = np.where(rr == 1, 0.5, 0.0)
    allout[0] = 1.0
    for b in range(1, 121):
        p = P[(b - 1) // 6]
        A = np.zeros((10, R_MAX + 1))
        for k, v in enumerate(OUT_RUNS):
            A += p[:, k, None] * V[b - 1][:, np.maximum(rr - v, 0)]
        A += p[:, 6, None] * np.vstack([V[b - 1, 1:], allout[None]])
        px, Vb = p[:, 7], V[b]
        for r in range(1, R_MAX + 1):                 # wide/no-ball: +1 run, ball not used
            Vb[:, r] = A[:, r] + px * Vb[:, r - 1]
    raw = V.copy()
    V = np.maximum.accumulate(V, axis=0)              # more balls never worse
    V = np.minimum.accumulate(V, axis=1)              # more wickets lost never better
    V = np.minimum.accumulate(V, axis=2)              # more runs needed never better
    return V, float(np.abs(V - raw).max())


def run_moments(P):
    """Mean and s.d. of runs still to come with b balls left and w wickets lost (w=10: 0)."""
    M, S = np.zeros((121, 11)), np.zeros((121, 11))
    v = OUT_RUNS.astype(float)
    for b in range(1, 121):
        p = P[(b - 1) // 6]
        pl, pw, px = p[:, :6], p[:, 6], p[:, 7]
        Mp, Sp, Mw, Sw = M[b - 1, :10], S[b - 1, :10], M[b - 1, 1:], S[b - 1, 1:]
        m = (pl @ v + pl.sum(1) * Mp + pw * Mw + px) / (1 - px)
        s = (pl @ v ** 2 + 2 * (pl @ v) * Mp + pl.sum(1) * Sp + pw * Sw + px * (1 + 2 * m)) / (1 - px)
        M[b, :10], S[b, :10] = m, s
    return M, np.sqrt(np.maximum(S - M ** 2, 1e-6))


def make_aux(ob, wl, code, alpha):
    P = outcome_table(ob, wl, code, alpha)
    V, adj = chase_dp(P)
    M, SD = run_moments(P)
    return {"V": V, "M": M, "SD": SD, "adj": adj}


def aux_columns(aux, balls_left, wkts_lost, runs_needed):
    b = np.clip(np.nan_to_num(balls_left), 0, 120).astype(int)
    w = np.clip(np.nan_to_num(wkts_lost), 0, 10).astype(int)
    rn = np.nan_to_num(runs_needed, nan=1.0)
    v = np.clip(aux["V"][b, np.minimum(w, 9), np.clip(rn, 0, R_MAX).astype(int)], 1e-4, 1 - 1e-4)
    zmv = (aux["M"][b, w] - rn) / np.maximum(aux["SD"][b, w], 0.5)
    return np.log(v / (1 - v)), zmv


# ================================================================= features
def build_features(df, ctx, aux=None):
    """df: innings, score, wkts, legal, max_balls, target, par_total, strength (+ fold if aux is a fold dict)."""
    out = pd.DataFrame(index=df.index)
    mb = df.max_balls.astype(float).clip(lower=1)
    out["balls_left"] = (mb - df.legal).clip(lower=0)
    out["wkts_in_hand"] = 10 - df.wkts
    k_eff = np.clip(np.rint(df.legal * 120 / mb), 0, 120).astype(int)
    out["score_vs_par"] = df.score - df.par_total * (mb / 120) * ctx["f_curve"][k_eff]
    out["runs_needed"] = df.target - df.score
    with np.errstate(divide="ignore", invalid="ignore"):
        rrr = np.where(out.balls_left > 0, out.runs_needed * 6 / out.balls_left, np.nan)
    out["rrr_minus_par"] = rrr - df.par_total / 20.0
    out["strength_code"] = pd.Categorical(df.strength, categories=STRENGTHS).codes
    out["resources"] = out.wkts_in_hand * out.balls_left / 120.0
    out["rrr_c"] = out.rrr_minus_par.clip(-12, 24).fillna(24)
    out["ln_need"] = np.log1p(out.runs_needed.clip(lower=0))
    nb = out.balls_left.clip(lower=1)
    out["chase_z"] = (nb * df.par_total / 120.0 - out.runs_needed) / (ctx["sigma_ball"] * np.sqrt(nb))
    for k, name in zip(WB_K, WB):
        out[name] = out.wkts_in_hand * np.minimum(out.balls_left, k) / k
    for g in STRENGTHS[1:]:
        out[f"g_{g}"] = (out.strength_code == STRENGTHS.index(g)).astype(float)
    out["dp_logit"], out["zmv"] = 0.0, 0.0
    if aux is not None:
        fo = df["fold"].values
        for f_ in np.unique(fo):
            sel = fo == f_
            dl, z = aux_columns(aux[f_], out.balls_left.values[sel], df.wkts.values[sel], out.runs_needed.values[sel])
            out.loc[sel, "dp_logit"], out.loc[sel, "zmv"] = dl, z
    return out


def blend_weight(balls_left):
    return np.clip((BLEND_HI - np.asarray(balls_left, float)) / (BLEND_HI - BLEND_LO), 0, 1)


def predict_chase(X, folds, models):
    """2nd-innings WP for live states. Dispatches to the endgame model chosen in wp_endgame (default: current)."""
    mode = models.get("chase_mode", "current")
    if mode == "current":
        return predict_current(X, np.asarray(folds), models)
    from . import wp_endgame
    return wp_endgame.chase_predict(mode, X, np.asarray(folds), models)


def predict_current(X, folds, models):
    p = np.empty(len(X))
    lam = blend_weight(X.balls_left.values)
    for fo in np.unique(folds):
        sel = folds == fo
        pm = models["m2"][fo].predict_proba(X.loc[sel, FEATS2].values)[:, 1]
        pe = models["m2e"][fo].predict_proba(X.loc[sel, FEATSE].values)[:, 1]
        p[sel] = lam[sel] * pe + (1 - lam[sel]) * pm
    return p


# ================================================================= WP for arbitrary states
def wp_state(df, models):
    """WP of the batting side. Terminal states exact: chase reached -> 1; all out / balls done -> 0 (level -> 0.5);
    end of 1st innings -> 1 - WP(start of chase, target = score + 1)."""
    df = df.reset_index(drop=True)
    wp = np.full(len(df), np.nan)
    inn2 = df.innings.values == 2
    done = (df.wkts.values >= 10) | (df.legal.values >= df.max_balls.values)
    won2 = inn2 & (df.score.values >= df.target.values)
    wp[won2] = 1.0
    lost2 = inn2 & ~won2 & done
    wp[lost2] = np.where(df.score.values[lost2] == df.target.values[lost2] - 1, 0.5, 0.0)

    X = build_features(df, models, models["aux"])
    live2 = inn2 & ~won2 & ~done
    if live2.any():
        wp[live2] = predict_chase(X.loc[live2], df.fold.values[live2], models)
    live1 = ~inn2 & ~done
    if live1.any():
        f1 = df.fold.values[live1]
        res = np.empty(live1.sum())
        for fo in np.unique(f1):
            sel = f1 == fo
            res[sel] = models["m1"][fo].predict_proba(X.loc[live1, FEATS1].values[sel])[:, 1]
        wp[live1] = res
    end1 = ~inn2 & done
    if end1.any():
        s = df.loc[end1].copy()
        s["innings"], s["target"], s["score"], s["wkts"], s["legal"] = 2, s.score + 1, 0, 0, 0
        s["max_balls"] = s.max_balls  # chase assumed as long as the 1st innings: later rain revisions are unknowable here
        X2 = build_features(s, models, models["aux"])
        wp[end1] = 1.0 - predict_chase(X2, s.fold.values, models)
    return wp


def states_before(b):
    return pd.DataFrame({
        "innings": b.innings.values, "score": b.score_before.values, "wkts": b.wkts_before.values,
        "legal": b.legal_before.values, "max_balls": b.max_balls.values, "max_balls2": b.max_balls2.values,
        "target": b.target_eff.values, "par_total": b.par_total.values, "strength": b.strength.astype(str).values,
        "fold": b.fold.values})


def states_after(b):
    s = states_before(b)
    s["score"], s["wkts"], s["legal"] = b.score_after.values, b.wkts_after.values, b.legal_after.values
    return s


# ================================================================= comparison model helpers
def fit_hgb(X, y, groups, mono, rng):
    """Monotonic HGB; number of trees chosen on a 20% held-out set of MATCHES, then refit on all."""
    def make(n):
        return HistGradientBoostingClassifier(max_iter=n, monotonic_cst=mono, categorical_features=[len(mono) - 1],
                                              **HGB_PARAMS)
    ug = np.unique(groups)
    hm = np.isin(groups, rng.choice(ug, size=max(1, len(ug) // 5), replace=False))
    probe = make(500).fit(X[~hm], y[~hm])
    best = int(np.argmin([log_loss(y[hm], p[:, 1], labels=[0, 1]) for p in probe.staged_predict_proba(X[hm])])) + 1
    return make(best).fit(X, y), best


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"n": int(len(y)), "log_loss": round(log_loss(y, p, labels=[0, 1]), 4),
            "brier": round(brier_score_loss(y, p), 4), "auc": round(roc_auc_score(y, p), 4)}


# ================================================================= charts
def calibration_chart(df, path):
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.plot([0, 1], [0, 1], color=TEXT2, lw=1, ls="--", label="perfect calibration")
    bins = np.linspace(0, 1, 21)
    for i, (inn, lab) in enumerate([(1, "1st innings"), (2, "2nd innings")]):
        d = df[df.innings == inn]
        idx = np.clip(np.digitize(d.wp_before, bins) - 1, 0, 19)
        g = d.groupby(idx).agg(p=("wp_before", "mean"), y=("bat_win", "mean"), n=("bat_win", "size"))
        g = g[g.n >= 100]
        ax.plot(g.p, g.y, marker="o", ms=5, color=SERIES[i], label=f"{lab} (out-of-fold)")
    ax.set_xlabel("Predicted win probability (batting side)")
    ax.set_ylabel("Observed win rate")
    ax.set_title("WP calibration — out-of-fold, 20 bins (≥100 balls each)")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    save(fig, path)


def curve_chart(bm, title, path):
    """WP of the team batting first after every delivery."""
    bm = bm.sort_values(["innings", "innings_ball_num"]).reset_index(drop=True)
    wp_first = np.where(bm.innings == 1, bm.wp_after, 1 - bm.wp_after)
    x = np.arange(1, len(bm) + 1)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.plot(np.r_[0, x], np.r_[bm.wp_before.iloc[0], wp_first], color=SERIES[0], lw=2)
    w = bm.is_wicket.values
    ax.scatter(x[w], wp_first[w], s=30, color=SERIES[1], zorder=3, edgecolor="#fcfcfb", linewidth=1.5, label="wicket")
    ax.axvline((bm.innings == 1).sum() + 0.5, color=TEXT2, lw=1, ls=":")
    ax.axhline(0.5, color=TEXT2, lw=0.8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Delivery number (both innings; dotted line = innings break)")
    ax.set_ylabel(f"WP — {bm.bat_team_first.iloc[0]}")
    ax.set_title("\n".join(__import__("textwrap").wrap(title, 95)), fontsize=9.5)
    ax.legend(loc="upper left")
    save(fig, path)


# ================================================================= main
def run():
    rng = np.random.default_rng(C.SEED)
    b = pd.read_parquet(C.PROC / "balls.parquet")
    m = pd.read_parquet(C.PROC / "matches.parquet")

    par, par_info = venue_par(b, m)
    f_curve = stage_curve(b, m)
    b = b.merge(par[["match_id", "par_total"]], on="match_id", how="left")
    mm = m.set_index("match_id")
    b["max_balls2"] = b.match_id.map(mm.max_balls2).astype("int16")
    b["target_eff"] = np.where(b.innings == 2, b.target, np.nan)

    train = b.wp_train_ok.to_numpy(dtype=bool)
    perm = rng.permutation(m.loc[m.wp_train_ok.to_numpy(dtype=bool), "match_id"].values)
    fold_of = {mid: i % N_FOLDS for i, mid in enumerate(perm)}
    b["fold"] = b.match_id.map(fold_of).fillna(-1).astype(int)

    step = b[train].groupby(["match_id", "innings", "legal_after"]).total_runs.sum()
    ctx = {"f_curve": f_curve, "sigma_ball": float(step.std())}

    # ---- Markov outcome tables per fold (training-fold matches only)
    ob, wl, code, ofold = outcome_data(b[train])
    alpha, alpha_ll = choose_alpha(ob, wl, code, ofold)
    aux = {fo: make_aux(ob[ofold != fo], wl[ofold != fo], code[ofold != fo], alpha) for fo in FOLDS_ALL}
    print(f"[phase2] Markov outcome model alpha={alpha}; DP monotone-projection max adj "
          f"{max(a['adj'] for a in aux.values()):.3f}")

    s_before = states_before(b)
    X = build_features(s_before, ctx)                       # fold-independent columns
    auxcols = {fo: aux_columns(aux[fo], X.balls_left.values, s_before.wkts.values, X.runs_needed.values)
               for fo in FOLDS_ALL}

    def Xf(rows, feats, fo):
        Z = X.loc[rows].copy()
        Z["dp_logit"], Z["zmv"] = auxcols[fo][0][rows], auxcols[fo][1][rows]
        return Z[feats].values

    y = b.bat_win.values
    models = {**ctx, "aux": aux, "alpha": alpha, "alpha_ll": alpha_ll, "m1": {}, "m2": {}, "m2e": {},
              "coefs": {}, "hgb_trees": {}}
    rows_out = []
    oof = {}

    # ---- 1st innings
    r1 = train & (b.innings.values == 1)
    y1, g1, f1 = y[r1].astype(int), b.match_id.values[r1], b.fold.values[r1]
    for fo in FOLDS_ALL:
        k = f1 != fo if fo >= 0 else np.ones(len(y1), bool)
        models["m1"][fo] = SignedLogit(SIGN1).fit(Xf(r1, FEATS1, fo)[k], y1[k])
    # ---- 2nd innings (main + endgame)
    r2 = train & (b.innings.values == 2)
    y2, g2, f2 = y[r2].astype(int), b.match_id.values[r2], b.fold.values[r2]
    bl2 = X.balls_left.values[r2]
    for fo in FOLDS_ALL:
        k = f2 != fo if fo >= 0 else np.ones(len(y2), bool)
        models["m2"][fo] = SignedLogit(SIGN2).fit(Xf(r2, FEATS2, fo)[k], y2[k])
        ke = k & (bl2 <= END_FIT_MAX)
        models["m2e"][fo] = SignedLogit(SIGNE).fit(Xf(r2, FEATSE, fo)[ke], y2[ke])
    for key, feats in [("m1", FEATS1), ("m2", FEATS2), ("m2e", FEATSE)]:
        models["coefs"][key] = dict(zip(feats, [round(float(v), 5) for v in models[key][-1].coef_raw_]))

    b["wp_before"] = wp_state(s_before, models).astype("float32")
    b["wp_after"] = wp_state(states_after(b), models).astype("float32")
    ev = b[train].copy()

    # ---- comparison models on the same folds (OOF)
    for inn, r, yi, gi, fi, hf, hm in [(1, r1, y1, g1, f1, HFEATS1, HMONO1), (2, r2, y2, g2, f2, HFEATS2, HMONO2)]:
        e = ev[ev.innings == inn]
        rows_out.append({"model": "CHOSEN: sign-constrained logistic" + (" + endgame blend" if inn == 2 else ""),
                         "innings": inn, **metrics(e.bat_win.values, e.wp_before.values)})
        feats = FEATS1 if inn == 1 else FEATS2
        comp = {k_: np.empty(len(yi)) for k_ in ["Monotonic gradient boosting (HGB)", "Logistic, unconstrained (same features)",
                                                 "Logistic naive (score or runs needed + balls left)"]}
        if inn == 2:
            comp["Sign-constrained logistic, no endgame model"] = np.empty(len(yi))
        Xh = X.loc[r, hf].values
        Xn = X.loc[r, ["score_vs_par", "balls_left"] if inn == 1 else ["runs_needed", "balls_left"]].values
        for fo in range(N_FOLDS):
            tr, te = fi != fo, fi == fo
            Xi = Xf(r, feats, fo)
            h, models["hgb_trees"][f"inn{inn}_fold{fo}"] = fit_hgb(Xh[tr], yi[tr], gi[tr], hm, rng)
            comp["Monotonic gradient boosting (HGB)"][te] = h.predict_proba(Xh[te])[:, 1]
            u = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xi[tr], yi[tr])
            comp["Logistic, unconstrained (same features)"][te] = u.predict_proba(Xi[te])[:, 1]
            n = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xn[tr], yi[tr])
            comp["Logistic naive (score or runs needed + balls left)"][te] = n.predict_proba(Xn[te])[:, 1]
            if inn == 2:
                comp["Sign-constrained logistic, no endgame model"][te] = models["m2"][fo].predict_proba(Xi[te])[:, 1]
        for name, pp in comp.items():
            rows_out.append({"model": name, "innings": inn, **metrics(yi, pp)})
            oof[(inn, name)] = pp
    met = pd.DataFrame(rows_out).sort_values(["innings", "log_loss"]).reset_index(drop=True)
    met.to_csv(C.TABLES / "wp_metrics.csv", index=False)

    by_grp = pd.DataFrame([{"strength": g_, "innings": inn, **metrics(e.bat_win.values, e.wp_before.values)}
                           for (g_, inn), e in ev.groupby([ev.strength.astype(str), "innings"])])
    by_grp.to_csv(C.TABLES / "wp_metrics_by_group.csv", index=False)

    # ---- last-over check
    e2 = ev[(ev.innings == 2) & (ev.balls_left <= 6) & (ev.runs_needed > 0)].copy()
    e2["no_endgame"] = pd.Series(oof[(2, "Sign-constrained logistic, no endgame model")], index=ev[ev.innings == 2].index)
    e2["need"] = pd.cut(e2.runs_needed, [0, 1, 2, 4, 6, 9, 12, 18, 400], labels=["1", "2", "3-4", "5-6", "7-9", "10-12", "13-18", "19+"])
    e2["balls"] = pd.cut(e2.balls_left, [0, 1, 3, 6], labels=["1", "2-3", "4-6"])
    death = (e2.groupby(["balls", "need"], observed=True)
             .agg(deliveries=("bat_win", "size"), matches=("match_id", "nunique"), actual=("bat_win", "mean"),
                  chosen=("wp_before", "mean"), no_endgame=("no_endgame", "mean")).reset_index())
    death = death[death.matches >= 5].round(3)
    death.to_csv(C.TABLES / "wp_last_over_check.csv", index=False)
    last6 = {k_: round(log_loss(e2.bat_win, np.clip(e2[k_], 1e-6, 1 - 1e-6)), 4) for k_ in ["wp_before", "no_endgame"]}

    # ---- season-forward (train 2022..t, test t+1): Markov tables and models refit on the training years only
    sf = []
    for t in range(2022, 2026):
        trn, tst = train & (b.year.values <= t), train & (b.year.values == t + 1)
        o_, w_, c_, _ = outcome_data(b[trn])
        a_ = make_aux(o_, w_, c_, alpha)
        dl, z = aux_columns(a_, X.balls_left.values, s_before.wkts.values, X.runs_needed.values)
        Xs = X.assign(dp_logit=dl, zmv=z)
        for inn in (1, 2):
            tr_, te_ = trn & (b.innings.values == inn), tst & (b.innings.values == inn)
            if inn == 1:
                p = SignedLogit(SIGN1).fit(Xs.loc[tr_, FEATS1].values, y[tr_]).predict_proba(Xs.loc[te_, FEATS1].values)[:, 1]
            else:
                mm_ = SignedLogit(SIGN2).fit(Xs.loc[tr_, FEATS2].values, y[tr_])
                ke = tr_ & (Xs.balls_left.values <= END_FIT_MAX)
                me_ = SignedLogit(SIGNE).fit(Xs.loc[ke, FEATSE].values, y[ke])
                lam = blend_weight(Xs.balls_left.values[te_])
                p = lam * me_.predict_proba(Xs.loc[te_, FEATSE].values)[:, 1] + (1 - lam) * mm_.predict_proba(Xs.loc[te_, FEATS2].values)[:, 1]
            sf.append({"train_years": f"2022-{t}", "test_year": t + 1, "innings": inn, **metrics(y[te_].astype(int), p)})
    sf = pd.DataFrame(sf)
    sf.to_csv(C.TABLES / "wp_season_forward.csv", index=False)

    # ---- monotonicity checks
    grid = pd.DataFrame({"innings": 2, "score": 145, "wkts": np.arange(9, -1, -1), "legal": 96, "max_balls": 120,
                         "max_balls2": 120, "target": 175, "par_total": 175.0, "strength": "IPL", "fold": -1})
    mono_wp = wp_state(grid, models)
    samp = s_before.sample(30000, random_state=C.SEED).reset_index(drop=True)
    live = (samp.wkts < 9) & (samp.legal >= 1) & (samp.legal < samp.max_balls - 1) & \
           ~((samp.innings == 2) & (samp.score >= samp.target - 1))
    samp = samp[live].reset_index(drop=True)
    base = wp_state(samp, models)
    viol = {}
    for name, mod in [("one more wicket in hand", lambda d: d.assign(wkts=d.wkts - 1)),
                      ("one more ball left", lambda d: d.assign(legal=d.legal - 1)),
                      ("one more run scored", lambda d: d.assign(score=d.score + 1))]:
        diff = wp_state(mod(samp.copy()), models) - base
        viol[name] = {"states": len(samp), "violations": int((diff < -1e-9).sum()),
                      "largest_drop": round(float(-diff.min()) if diff.min() < 0 else 0.0, 4)}

    # ---- spot states: final model vs Markov chain vs what happened (training matches)
    spots = []
    for need, balls, wih in [(2, 1, 7), (1, 1, 7), (5, 3, 5), (9, 6, 6), (12, 6, 6), (16, 6, 7), (30, 24, 5)]:
        st = pd.DataFrame({"innings": [2], "score": [175 - need], "wkts": [10 - wih], "legal": [120 - balls],
                           "max_balls": [120], "max_balls2": [120], "target": [175], "par_total": [175.0],
                           "strength": ["IPL"], "fold": [-1]})
        emp = b[(b.innings == 2) & b.bat_win.notna() & (b.balls_left == balls) & ((b.runs_needed - need).abs() <= 1)
                & ((10 - b.wkts_before - wih).abs() <= 1)].drop_duplicates("match_id")
        spots.append({"state": f"need {need} off {balls}, {wih} wkts in hand", "final_model": round(float(wp_state(st, models)[0]), 3),
                      "markov_chain": round(float(aux[-1]["V"][balls, 10 - wih, need]), 3),
                      "empirical (need ±1, wkts in hand ±1)": f"{emp.bat_win.mean():.2f} ({len(emp)} matches)" if len(emp) else "–"})
    spots = pd.DataFrame(spots)
    spots.to_csv(C.TABLES / "wp_spot_states.csv", index=False)

    # ---- charts
    calibration_chart(ev, C.CHARTS / "wp_calibration.png")
    cand = m[m.kept & ~m.revised_recorded & ~m.revised_unrecorded & m.has_balls].copy()
    cand["close"] = np.select([cand.is_tie, cand.result_kind.eq("runs"), cand.result_kind.eq("wkts")],
                              [0, cand.margin, 120 - cand.legal2 + 0.5], 99)
    rr_tie = cand[cand.is_tie & ((cand.home_team_id == C.RR_TEAM_ID) | (cand.away_team_id == C.RR_TEAM_ID))]
    ko = cand[(cand.knock_out == 1) & ~cand.is_tie].sort_values(["close", "match_date"])
    curves = []
    for _, row in pd.concat([ko.head(2), rr_tie.head(1)]).iterrows():
        bm = b[b.match_id == row.match_id].copy()
        first = bm.loc[bm.innings == 1, "batting_team_id"].iloc[0]
        bm["bat_team_first"] = (row.home_team if first == row.home_team_id else row.away_team) + " (batting first)"
        title = f"{row.home_team} v {row.away_team}, {row.match_date.date()} ({row.stage}) — {row.result.strip()}"
        curve_chart(bm, title, C.CHARTS / f"wp_curve_{row.match_id}.png")
        last = bm[bm.innings == 2].sort_values("innings_ball_num").tail(3)
        curves.append((row.match_id, title, last[["runs_needed", "balls_left", "wkts_before", "wp_before"]].round(3)))

    # ---- persist
    joblib.dump(models, C.PROC / "wp_models.joblib")
    b[["match_id", "innings", "innings_ball_num", "fold", "par_total", "max_balls2", "target_eff",
       "wp_before", "wp_after"]].to_parquet(C.PROC / "wp_preds.parquet", index=False)
    par.to_parquet(C.PROC / "venue_par.parquet", index=False)

    write_report(met, by_grp, sf, par_info, par, models, mono_wp, viol, curves, b, m, death, last6, spots)
    print(spots.to_string(index=False))
    print(met.to_string(index=False))
    print(f"[phase2] last-6-balls log-loss chosen {last6['wp_before']} vs no-endgame {last6['no_endgame']}")
    print(f"[phase2] monotonic perturbation: {viol}")


def _md(df):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def write_report(met, by_grp, sf, par_info, par, models, mono_wp, viol, curves, b, m, death, last6, spots):
    f_curve = models["f_curve"]
    L = ["# Phase 2 — Win probability model\n",
         "WP = probability that the **batting side of the ball** wins. Separate models for the 1st and 2nd innings.\n",
         "## Venue par (leak-free)\n",
         f"- League mean for a match = mean first-innings total of the same competition group in the previous "
         f"{C.PAR_WINDOW_DAYS} days, strictly before the match date (fallback to all groups if <10 such matches; to the overall "
         f"mean if none). Sources: {par_info['league_src_counts']}.",
         f"- Venue adjustment = mean residual (total − its own league mean) of earlier matches at the venue in the window, shrunk by "
         f"n/(n+k), k = within-venue variance / between-venue variance = {par_info['within_var']:.0f} / {par_info['between_var']:.1f} "
         f"= **{par_info['k_shrink']:.1f}** (estimated from data).",
         f"- History pool: {par_info['hist_matches']} full-length, un-revised first innings. Par totals "
         f"{par.par_total.min():.0f}–{par.par_total.max():.0f} (median {par.par_total.median():.0f}).",
         f"- Par at a stage = par total × f(k), f = average share of the final total reached after k legal balls "
         f"(f(36) = {f_curve[36]:.2f}, f(90) = {f_curve[90]:.2f}). This is a shape curve and uses no match outcomes.\n",
         "## How the model was chosen (and what failed)\n",
         "1. **Planned:** LightGBM with monotonic constraints. LightGBM cannot load on this Mac (missing libomp; no system "
         "libraries installed, as agreed), so the boosted candidate is scikit-learn HistGradientBoosting with the same monotonic "
         "constraints. It **lost to logistic regression** on grouped out-of-fold log-loss in both innings, even when regularised "
         "(≥3000 balls per leaf). Balls within a match share one outcome, so trees fit individual matches.",
         "2. **Logistic regression with sign constraints** won overall, but a first version gave a side needing 2 off the last ball "
         "with 8 wickets in hand **99.6%**. Additive terms (especially wickets in hand) do not fade as balls run out.",
         "3. **Fix (2nd innings):** features from a Markov ball-outcome model of the chase (below), wickets entered only as "
         "wickets × min(balls left, k)/k so their value can fade, plus a separate endgame model fitted on the last "
         f"{END_FIT_MAX} balls and blended in linearly between {BLEND_HI} and {BLEND_LO} balls left (weight 1 at ≤{BLEND_LO}). "
         f"Last-6-balls log-loss: **{last6['no_endgame']} → {last6['wp_before']}**.\n",
         "**Markov ball-outcome model.** p(outcome | overs left, wickets lost) for outcomes 0/1/2/3/4/6/wicket/wide-or-no-ball, "
         "from 2nd-innings deliveries of the fold's own training matches, smoothed cell → (over, wicket group) → over with "
         f"pseudo-count α chosen by grouped-CV outcome log-loss (α = {models['alpha']}; {models['alpha_ll']}). From it: "
         "(a) **dp_logit** = logit of the exact chase win probability by backward induction over (balls left, wickets lost, "
         f"runs needed ≤ {R_MAX}); extras add a run without using a ball; level at the end = tie = 0.5; a monotone projection "
         "guarantees more balls / more wickets in hand / fewer runs needed never lowers it. (b) **zmv** = (expected runs still to "
         "come − runs needed) / s.d. of runs still to come, both computed exactly from the same chain (a data-estimated, DLS-like "
         "resource table).\n",
         "### Features, required signs, fitted coefficients (all-training-data models, raw units)\n",
         "| Model | Feature | Required sign | Coefficient |", "|---|---|---|---|"]
    for key, feats, signs, lab in [("m1", FEATS1, SIGN1, "1st inns"), ("m2", FEATS2, SIGN2, "2nd inns main"),
                                   ("m2e", FEATSE, SIGNE, "2nd inns endgame")]:
        for f_, sg in zip(feats, signs):
            L.append(f"| {lab} | {f_} | {'+' if sg > 0 else '−' if sg < 0 else 'free'} | {models['coefs'][key][f_]} |")
    L += ["",
          "Definitions: score_vs_par = score − par at this stage; resources = wickets in hand × balls left / 120; "
          "rrr_c = (required rate − par total/20) clipped to [−12, 24]; ln_need = log(1 + runs needed); "
          f"chase_z = (balls left × par total/120 − runs needed) / (σ√balls left), σ = s.d. of runs per legal-ball step = "
          f"{models['sigma_ball']:.3f}; wbK = wickets in hand × min(balls left, K)/K; g_* = competition-group dummies (IPL = reference). "
          "Coefficients are fitted with L2 penalty (as sklearn C=1 on standardised inputs) and sign bounds (L-BFGS-B). "
          "A coefficient of exactly 0 means the sign bound was active: the data wanted the 'wrong' sign, so the feature is unused.\n",
          "Terminal states are exact: chase reached → 1; all out / overs done → 0 (tie → 0.5); end of 1st innings → "
          "1 − WP(start of chase with target = score + 1).\n",
          "## Training data and validation\n",
          f"- Training matches: {int(m.wp_train_ok.sum())} (excludes ties, rain-revised recorded and unrecorded, non-reconciling). "
          f"Deliveries: {int(b.wp_train_ok.sum()):,}.",
          "- 5-fold cross-validation grouped by match. Each training-match ball is scored by models and Markov tables built "
          "without that match. Other kept matches are scored by the all-training-data model (never trained on them).\n",
          "### Out-of-fold metrics (ball level)\n", _md(met),
          "\n### By competition group (chosen model, out-of-fold)\n", _md(by_grp),
          "\n### Season-forward (train 2022..t, test t+1; Markov tables refit on training years only)\n", _md(sf),
          "\n### Last-over check (chases with ≤6 balls left, out-of-fold; cells with ≥5 matches)\n",
          "`chosen` = final model; `no_endgame` = same model without the endgame blend. Cells of 20–70 matches have sampling "
          "error of roughly ±0.06–0.10.\n", _md(death),
          "\n### Spot states: final model vs Markov chain vs what happened\n",
          "Final model in an IPL chase of 175 at par 175; Markov chain = state-only ball-outcome model (ties count 0.5); "
          "empirical = chase win rate in all kept matches at that balls-left with runs needed ±1 and wickets in hand ±1 "
          "(one state per match).\n", _md(spots),
          "\n**Reading this table honestly.** The state-only Markov chain gives wickets in hand little value in the last over, "
          "because its last-over outcome rates average deep batting line-ups with tail-enders. Real chases with wickets in hand "
          "convert far more often (e.g. 10–15 needed off the last over with 6+ wickets in hand: 7 of 10 won), and the final model "
          "agrees with the wicket-matched data. That is why the chain is used as a feature, not as the model. Every endgame cell "
          "holds only 5–25 matches, so last-over WP is the least certain part of the model. Phase 3 leverage is computed from "
          "WP *changes*, which are less sensitive to a level error than WP itself.\n",
          "\n## Checks\n",
          f"- Monotonic spot check (chase 30 off 24, IPL, par 175), WP for 1..10 wickets in hand: "
          f"{', '.join(f'{v:.2f}' for v in mono_wp)}",
          f"- Perturbation check on random live states (each change should never lower WP): {viol}. The only source of possible "
          "violations is the linear blend between the main and endgame models (12–24 balls left); each model is monotone on its own.",
          "- Calibration: `outputs/charts/wp_calibration.png`",
          "- Sanity curves (chosen by rule: the two closest knockout finishes + RR's tied match). Last three chase deliveries shown:"]
    for mid, t, last in curves:
        L.append(f"  - `outputs/charts/wp_curve_{mid}.png` — {t}")
        L.append("    " + "; ".join(f"need {int(r.runs_needed)} off {int(r.balls_left)}, {int(10 - r.wkts_before)} wkts in hand → WP {r.wp_before:.2f}"
                                    for r in last.itertuples()))
    (C.OUT / "02_wp_model.md").write_text("\n".join(L))
