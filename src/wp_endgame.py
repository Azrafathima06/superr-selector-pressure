"""Phase 2b: endgame study for the 2nd-innings WP model.

Candidates (all complete 2nd-innings predictors, all scored out-of-fold with the same match folds):
  current        main sign-constrained logistic + endgame logistic, blended 24 -> 12 balls left
  main_only      main sign-constrained logistic, no endgame model (reference)
  markov         Markov chain with outcome probabilities by (over = balls left/phase, wickets lost, competition
                 strength) for <= 30 balls left; main model above 30
  hybrid         Markov for the last 12 balls, linear blend from 24 to 12 with the main model
  spline         sign-constrained logistic with a richer chase form: hinge (linear-spline) bases on log required rate,
                 their interaction with wickets lost, wickets x balls-left bases, runs-needed bases (whole innings)
  spline_markov  spline + Markov features (logit of the strength-aware chain, expected-runs z-score)

Model choice: lowest out-of-fold log-loss on 2nd-innings deliveries with <= 30 legal balls left (Brier reported too).
The chosen candidate becomes the production 2nd-innings WP (models["chase_mode"]) and all WP values are re-scored.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from . import config as C
from . import wp_model as W
from .plotstyle import plt, save, SERIES, TEXT2

ENDGAME_MAX = 30
HYB_LO, HYB_HI = 12, 24
ALPHA_S = [30, 100, 300, 1000, 3000]
CANDIDATES = ["current", "main_only", "markov", "hybrid", "spline", "spline_markov"]
LABELS = {"current": "Current (logistic + endgame logistic)", "main_only": "Main logistic only",
          "markov": "Markov chain (≤30 balls)", "hybrid": "Hybrid (Markov ≤12, blend 24→12)",
          "spline": "Spline logistic", "spline_markov": "Spline logistic + Markov features"}
BL_BINS, BL_LAB = [0, 6, 12, 18, 30], ["1–6", "7–12", "13–18", "19–30"]
RRR_BINS, RRR_LAB = [0, 6, 9, 12, 15, 18, np.inf], ["<6", "6–9", "9–12", "12–15", "15–18", "18+"]
WIH_BINS, WIH_LAB = [0, 3, 6, 10], ["1–3", "4–6", "7–10"]


# ================================================================= strength-aware Markov chain
def outcome_data_s(rows):
    ob, wl, code, fold = W.outcome_data(rows)
    x = rows[(rows.innings == 2) & ~rows.is_penalty]
    sc = pd.Categorical(x.strength.astype(str), categories=W.STRENGTHS).codes.astype(int)
    return ob, wl, sc, code, fold


def table_s(ob, wl, sc, code, alpha_base, alpha_s):
    """p(outcome | over-left bucket, wickets lost, strength), shrunk toward the pooled (over, wickets) table."""
    base = W.outcome_table(ob, wl, code, alpha_base)
    out = np.empty((3,) + base.shape)
    for s in range(3):
        k = sc == s
        c = np.zeros_like(base)
        np.add.at(c, (ob[k], wl[k], code[k]), 1)
        out[s] = (c + alpha_s * base) / (c.sum(-1, keepdims=True) + alpha_s)
    return out


def choose_alpha_s(ob, wl, sc, code, fold, alpha_base):
    res = {}
    for a in ALPHA_S:
        ll = 0.0
        for fo in range(W.N_FOLDS):
            k = fold != fo
            t = table_s(ob[k], wl[k], sc[k], code[k], alpha_base, a)
            te = fold == fo
            ll -= np.log(t[sc[te], ob[te], wl[te], code[te]]).sum()
        res[a] = round(ll / len(code), 5)
    return min(res, key=res.get), res


def markov_tables(ob, wl, sc, code, alpha_base, alpha_s):
    P = table_s(ob, wl, sc, code, alpha_base, alpha_s)
    return np.stack([W.chase_dp(P[s])[0] for s in range(3)]).astype(np.float32)


def mk_lookup(Vs, X):
    sc = np.clip(X.strength_code.values, 0, 2).astype(int)
    b = np.clip(X.balls_left.values, 0, 120).astype(int)
    w = np.clip(10 - X.wkts_in_hand.values, 0, 9).astype(int)
    r = np.clip(np.nan_to_num(X.runs_needed.values, nan=1), 0, W.R_MAX).astype(int)
    return np.clip(Vs[sc, b, w, r].astype(float), 1e-4, 1 - 1e-4)


# ================================================================= spline logistic
def make_knots(X):
    n = X.runs_needed.clip(lower=1).values
    lr = np.log(6 * n / X.balls_left.clip(lower=1).values)
    return {"lrrr": np.quantile(lr, [0.01, 0.10, 0.25, 0.50, 0.75, 0.90]).round(4).tolist(),
            "need": np.quantile(n, [0.10, 0.25, 0.50, 0.75]).round(1).tolist()}


def spline_feats(X, knots, mk=None):
    """Monotone bases; the returned signs make WP non-decreasing in balls left and wickets in hand and
    non-increasing in runs needed."""
    n = X.runs_needed.clip(lower=1).values.astype(float)
    bl = X.balls_left.clip(lower=1).values.astype(float)
    lr = np.log(6 * n / bl)
    wl = (10 - X.wkts_in_hand.values) / 10.0
    F, S = {"ln_need": np.log1p(n), "ln_balls": np.log1p(bl)}, {"ln_need": -1, "ln_balls": 1}
    for k in knots["need"]:
        F[f"need_min{k:g}"], S[f"need_min{k:g}"] = np.minimum(n, k), -1
    for j, k in enumerate(knots["lrrr"]):
        h = np.maximum(0.0, lr - k)
        F[f"lrrr_h{j}"], S[f"lrrr_h{j}"] = h, -1
        F[f"lrrr_h{j}_x_wktslost"], S[f"lrrr_h{j}_x_wktslost"] = h * wl, -1
    for name in W.WB:
        F[name], S[name] = X[name].values, 1
    F["rrr_c"], S["rrr_c"] = X.rrr_c.values, -1
    for g in W.G:
        F[g], S[g] = X[g].values, 0
    if mk is not None:
        F["zmv"], S["zmv"] = X.zmv.values, 1
        F["markov_logit"], S["markov_logit"] = np.log(mk / (1 - mk)), 1
    names = list(F)
    return np.column_stack([F[k] for k in names]), [S[k] for k in names], names


# ================================================================= prediction for any candidate
def hybrid_weight(bl):
    return np.clip((HYB_HI - np.asarray(bl, float)) / (HYB_HI - HYB_LO), 0, 1)


def chase_predict(mode, X, folds, models):
    out = np.empty(len(X))
    bl = X.balls_left.values
    for fo in np.unique(folds):
        idx = np.where(folds == fo)[0]
        Xs = X.iloc[idx]
        if mode in ("current", "main_only", "markov", "hybrid"):
            main = models["m2"][fo].predict_proba(Xs[W.FEATS2].values)[:, 1]
        if mode == "current":
            p = W.predict_current(Xs, np.full(len(Xs), fo), models)
        elif mode == "main_only":
            p = main
        elif mode == "markov":
            p = np.where(bl[idx] <= ENDGAME_MAX, mk_lookup(models["markov_s"][fo], Xs), main)
        elif mode == "hybrid":
            lam = hybrid_weight(bl[idx])
            p = lam * mk_lookup(models["markov_s"][fo], Xs) + (1 - lam) * main
        elif mode in ("spline", "spline_markov"):
            mk = mk_lookup(models["markov_s"][fo], Xs) if mode == "spline_markov" else None
            F, _, _ = spline_feats(Xs, models["knots"], mk)
            p = models[mode][fo].predict_proba(F)[:, 1]
        else:
            raise ValueError(mode)
        out[idx] = p
    return out


# ================================================================= helpers
def wilson(p, n, z=1.96):
    n = np.maximum(np.asarray(n, float), 1)
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def _md(df):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def endgame_chart(cells, path):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    shapes = ["o", "s", "^", "D"]
    show = [c for c in CANDIDATES if c != "main_only"]
    c10 = cells[cells.matches >= 10]
    for ax, cand in zip(axes.flat, show):
        ax.plot([0, 1], [0, 1], color=TEXT2, lw=1, ls="--")
        for i, lab in enumerate(BL_LAB):
            d = c10[c10.balls_left == lab]
            ax.errorbar(d[cand], d.actual, yerr=[d.actual - d.ci_lo, d.ci_hi - d.actual], fmt=shapes[i], ms=5,
                        color=SERIES[i], ecolor=SERIES[i], elinewidth=0.8, alpha=0.9, label=f"{lab} balls left")
        ax.set_title(LABELS[cand], fontsize=10)
        ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    for ax in axes[1]:
        ax.set_xlabel("Predicted win probability (mean over cell)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Actual chase win rate (95% CI)")
    last = axes.flat[len(show)]
    last.axis("off")
    h, l_ = axes.flat[0].get_legend_handles_labels()
    last.legend(h, l_, loc="center", title="Cell: balls left × required-rate band\n× wickets in hand (≥10 matches)")
    fig.suptitle("Endgame calibration — 2nd innings, ≤30 balls left, out-of-fold", fontsize=12)
    save(fig, path)


# ================================================================= main
def run():
    rng_seed = C.SEED
    b = pd.read_parquet(C.PROC / "balls.parquet")
    p = pd.read_parquet(C.PROC / "wp_preds.parquet")
    b = b.merge(p.drop(columns=["wp_before", "wp_after"]), on=["match_id", "innings", "innings_ball_num"])
    models = joblib.load(C.PROC / "wp_models.joblib")
    models["chase_mode"] = "current"

    train = b.wp_train_ok.to_numpy(dtype=bool)
    r2 = train & (b.innings.values == 2)
    s2 = W.states_before(b.loc[r2]).reset_index(drop=True)
    y = b.bat_win.values[r2].astype(int)
    fold = b.fold.values[r2]
    mid = b.match_id.values[r2]

    # features of every 2nd-innings training row under each fold's Markov tables (fold -1 = all data)
    XF = {fo: W.build_features(s2.assign(fold=fo), models, models["aux"]) for fo in W.FOLDS_ALL}

    # ---- strength-aware Markov chain per fold
    ob, wl, sc, code, ofold = outcome_data_s(b[train])
    alpha_s, alpha_s_ll = choose_alpha_s(ob, wl, sc, code, ofold, models["alpha"])
    models["markov_s"] = {fo: markov_tables(ob[ofold != fo], wl[ofold != fo], sc[ofold != fo], code[ofold != fo],
                                            models["alpha"], alpha_s) for fo in W.FOLDS_ALL}
    models["alpha_s"], models["alpha_s_ll"] = alpha_s, alpha_s_ll
    print(f"[phase2b] strength Markov alpha_s={alpha_s} {alpha_s_ll}")

    # ---- spline logistic candidates (fit on all 2nd-innings deliveries of the training folds)
    models["knots"] = make_knots(XF[-1])
    models["spline_coefs"] = {}
    for mode in ("spline", "spline_markov"):
        models[mode] = {}
        for fo in W.FOLDS_ALL:
            mk = mk_lookup(models["markov_s"][fo], XF[fo]) if mode == "spline_markov" else None
            F, S, names = spline_feats(XF[fo], models["knots"], mk)
            k = fold != fo if fo >= 0 else np.ones(len(y), bool)
            models[mode][fo] = W.SignedLogit(S).fit(F[k], y[k])
        models["spline_coefs"][mode] = dict(zip(names, [round(float(v), 4) for v in models[mode][-1].coef_raw_]))
    print("[phase2b] spline models fitted")

    # ---- out-of-fold predictions for every candidate
    oof = {c: np.empty(len(y)) for c in CANDIDATES}
    for fo in range(W.N_FOLDS):
        idx = np.where(fold == fo)[0]
        Xs = XF[fo].iloc[idx]
        for c in CANDIDATES:
            oof[c][idx] = chase_predict(c, Xs, np.full(len(idx), fo), models)

    bl = XF[-1].balls_left.values
    end = bl <= ENDGAME_MAX
    rows = []
    for c in CANDIDATES:
        pc = np.clip(oof[c], 1e-6, 1 - 1e-6)
        rows.append({"candidate": LABELS[c], "key": c,
                     "log_loss_le30": round(log_loss(y[end], pc[end], labels=[0, 1]), 4),
                     "brier_le30": round(brier_score_loss(y[end], pc[end]), 4),
                     "log_loss_le12": round(log_loss(y[bl <= 12], pc[bl <= 12], labels=[0, 1]), 4),
                     "log_loss_full_2nd_inns": round(log_loss(y, pc, labels=[0, 1]), 4),
                     "brier_full_2nd_inns": round(brier_score_loss(y, pc), 4)})
    comp = pd.DataFrame(rows).sort_values("log_loss_le30").reset_index(drop=True)

    # ---- pooled endgame calibration cells
    X30 = XF[-1][end]
    e = pd.DataFrame({"match_id": mid[end], "y": y[end],
                      "balls_left": pd.cut(X30.balls_left, BL_BINS, labels=BL_LAB),
                      "rrr_band": pd.cut(6 * X30.runs_needed / X30.balls_left.clip(lower=1), RRR_BINS, labels=RRR_LAB, right=False),
                      "wkts_in_hand": pd.cut(X30.wkts_in_hand, WIH_BINS, labels=WIH_LAB)})
    for c in CANDIDATES:
        e[c] = oof[c][end]
    g = e.groupby(["balls_left", "rrr_band", "wkts_in_hand"], observed=True)
    cells = g.agg(deliveries=("y", "size"), matches=("match_id", "nunique"), actual=("y", "mean"),
                  **{c: (c, "mean") for c in CANDIDATES}).reset_index()
    # CI uses the number of matches (deliveries within a match share one result, so they are not independent)
    cells["ci_lo"], cells["ci_hi"] = wilson(cells.actual.values, cells.matches.values)
    cells = cells.round(3)
    cells.to_csv(C.TABLES / "wp_endgame_cells.csv", index=False)
    c10 = cells[cells.matches >= 10]
    summ = []
    for c in CANDIDATES:
        err = (c10[c] - c10.actual).abs()
        outside = ((c10[c] < c10.ci_lo) | (c10[c] > c10.ci_hi)).sum()
        summ.append({"key": c, "cells_mean_abs_error (weighted by matches)": round(np.average(err, weights=c10.matches), 3),
                     f"cells outside actual 95% CI (of {len(c10)})": int(outside)})
    comp = comp.merge(pd.DataFrame(summ), on="key")
    comp.to_csv(C.TABLES / "wp_endgame_candidates.csv", index=False)
    endgame_chart(cells, C.CHARTS / "wp_endgame_calibration.png")

    # ---- monotonicity check for every candidate on endgame states
    samp = s2[(XF[-1].balls_left.values <= ENDGAME_MAX + 1) & (s2.wkts.values < 9) & (s2.legal.values >= 1) &
              (XF[-1].runs_needed.values >= 2)].sample(15000, random_state=rng_seed).reset_index(drop=True)
    samp["fold"] = -1
    mono = {}
    for c in CANDIDATES:
        def wp(d):
            X = W.build_features(d, models, models["aux"])
            return chase_predict(c, X, d.fold.values, models)
        base = wp(samp)
        res = {}
        for name, mod in [("wicket", lambda d: d.assign(wkts=d.wkts - 1)), ("ball", lambda d: d.assign(legal=d.legal - 1)),
                          ("run", lambda d: d.assign(score=d.score + 1))]:
            diff = wp(mod(samp.copy())) - base
            res[name] = f"{int((diff < -1e-6).sum())} (max drop {max(0.0, float(-diff.min())):.3f})"
        mono[c] = res
    mono_df = pd.DataFrame(mono).T.rename_axis("key").reset_index()
    mono_df.columns = ["key", "violations: +1 wicket in hand", "violations: +1 ball left", "violations: +1 run scored"]
    comp = comp.merge(mono_df, on="key")

    # ---- spot states
    spot_rows = []
    allb = b.assign(wih=10 - b.wkts_before)
    for need, balls, tol in [(2, 1, 0), (12, 6, 1), (16, 6, 1), (30, 24, 2)]:
        for band, w_eval, lo, hi in [("7+", 8, 7, 10), ("3–5", 4, 3, 5)]:
            st = pd.DataFrame({"innings": [2], "score": [175 - need], "wkts": [10 - w_eval], "legal": [120 - balls],
                               "max_balls": [120], "max_balls2": [120], "target": [175], "par_total": [175.0],
                               "strength": ["IPL"], "fold": [-1]})
            X = W.build_features(st, models, models["aux"])
            emp = allb[(allb.innings == 2) & allb.bat_win.notna() & (allb.balls_left == balls) &
                       ((allb.runs_needed - need).abs() <= tol) & allb.wih.between(lo, hi)].drop_duplicates("match_id")
            row = {"state": f"need {need} off {balls}", "wkts in hand": f"{band} (model at {w_eval})"}
            for c in CANDIDATES:
                row[c] = round(float(chase_predict(c, X, np.array([-1]), models)[0]), 3)
            if len(emp):
                lo_, hi_ = wilson(np.array([emp.bat_win.mean()]), np.array([len(emp)]))
                row["empirical"] = f"{emp.bat_win.mean():.2f} [{lo_[0]:.2f}–{hi_[0]:.2f}] n={len(emp)}"
            else:
                row["empirical"] = "no matches"
            spot_rows.append(row)
    spots = pd.DataFrame(spot_rows)
    spots.to_csv(C.TABLES / "wp_endgame_spots.csv", index=False)

    # ---- choose and re-score
    chosen = comp.iloc[0].key
    models["chase_mode"] = chosen
    models["endgame_comparison"] = comp
    b_all = pd.read_parquet(C.PROC / "balls.parquet").merge(p.drop(columns=["wp_before", "wp_after"]),
                                                            on=["match_id", "innings", "innings_ball_num"])
    b_all["wp_before"] = W.wp_state(W.states_before(b_all), models).astype("float32")
    b_all["wp_after"] = W.wp_state(W.states_after(b_all), models).astype("float32")
    b_all[p.columns].to_parquet(C.PROC / "wp_preds.parquet", index=False)
    joblib.dump(models, C.PROC / "wp_models.joblib")

    # full-innings OOF check of the final production model (1st innings unchanged)
    ev = b_all[b_all.wp_train_ok.to_numpy(dtype=bool)]
    final = pd.DataFrame([{"innings": inn, **W.metrics(d.bat_win.values, d.wp_before.values)} for inn, d in ev.groupby("innings")])
    W.calibration_chart(ev, C.CHARTS / "wp_calibration.png")

    write_report(comp, cells, spots, chosen, models, final, alpha_s, alpha_s_ll)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(comp.drop(columns=["key"]).to_string(index=False))
    print(spots.to_string(index=False))
    print(f"[phase2b] chosen endgame model: {chosen}")


def write_report(comp, cells, spots, chosen, models, final, alpha_s, alpha_s_ll):
    W_ = pd.read_csv(C.TABLES / "wp_metrics.csv")
    L = ["# Phase 2b — Endgame study (2nd innings, ≤30 legal balls left)\n",
         "All candidates are complete 2nd-innings WP models scored **out-of-fold** with the same match folds as Phase 2 "
         "(every delivery scored by models and Markov tables built without its match). Selection rule, fixed before "
         "looking at results: lowest out-of-fold log-loss on deliveries with ≤30 balls left; Brier also reported.\n",
         "## Candidates\n",
         "- **current** — Phase 2 model: main sign-constrained logistic + endgame logistic (fit on ≤18 balls), blended 24→12.",
         "- **main_only** — main logistic without the endgame model (reference).",
         f"- **markov** — Markov chain: ball outcomes 0/1/2/3/4/6/wicket/wide-or-no-ball with probabilities by over "
         f"(i.e. balls left; the phase is implied by the over), wickets lost and competition strength. Strength tables are shrunk toward "
         f"the pooled over × wickets table with pseudo-count α_s = {alpha_s} chosen by grouped-CV outcome log-loss {alpha_s_ll}. "
         f"Wides/no-balls add a run without using a ball; level scores at the end = tie = 0.5. Exact chase probability by "
         f"backward induction. Used for ≤{ENDGAME_MAX} balls left; main model above.",
         f"- **hybrid** — Markov for the last {HYB_LO} balls, linear blend from {HYB_HI} to {HYB_LO} with the main model.",
         f"- **spline** — one sign-constrained logistic for the whole chase with a richer form: log(1+runs needed), min(runs needed, k) "
         f"at knots {models['knots']['need']}, log(1+balls left), hinge bases max(0, log RRR − k) at knots "
         f"{models['knots']['lrrr']} (quantiles of the feature, no outcomes used), each hinge × wickets lost/10, wickets × "
         "min(balls left, K)/K for K in 3, 6, 12, 24, 48, 120, required rate vs venue par, competition dummies. Every basis is "
         "monotone in the state and its sign is enforced, so monotonic sense holds by construction.",
         "- **spline_markov** — spline + logit of the strength-aware Markov chain + expected-runs z-score (both signs +).\n",
         "## All candidates (out-of-fold)\n", _md(comp.drop(columns=["key"])),
         f"\n**Chosen: {LABELS[chosen]}** (lowest log-loss on ≤30 balls left). It is now the production 2nd-innings WP; "
         "all WP values were re-scored.\n",
         "Monotonicity violations are counted on 15,000 sampled endgame states with fold −1 models; a violation is a WP drop "
         "> 0.000001 when the state improves by one wicket in hand, one ball left or one run.\n",
         "## Final production model, full innings (out-of-fold)\n", _md(final),
         "\n## Phase 2 comparison table (full innings, from Phase 2)\n", _md(W_),
         "\n## Spot states (model: IPL chase of 175 at par 175; empirical: all kept matches, one state per match, "
         "same balls left, runs needed within ±0 / ±1 / ±1 / ±2, wickets in hand in the band; 95% Wilson interval)\n",
         _md(spots),
         "\n## Pooled endgame calibration\n",
         "`outputs/charts/wp_endgame_calibration.png` — one panel per candidate; each point is a cell (balls left × required-rate "
         "band × wickets in hand) with ≥10 matches; vertical bars = 95% Wilson interval of the actual win rate using the number of "
         "matches (deliveries within one match share one result, so they are not independent). Full cell table: "
         "`outputs/tables/wp_endgame_cells.csv` (all cells, including those with <10 matches).\n",
         _md(cells[cells.matches >= 10][["balls_left", "rrr_band", "wkts_in_hand", "matches", "actual", "ci_lo", "ci_hi"] + CANDIDATES])]
    (C.OUT / "02b_wp_endgame.md").write_text("\n".join(L))
