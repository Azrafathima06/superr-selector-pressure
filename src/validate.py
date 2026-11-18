"""Phase 8: validation of the LOCKED lists (results are reported, never used to change the lists).

1. Out-of-time: the Phase 7 rating stage is rebuilt on 2022–2024 data only (same methods and rules; activity window
   2023–24, the analogue of 2025–26), then used to predict each player's 2025–2026 raw run value on H.
   Benchmarks (all from 2022–24): (a) career average run value (all deliveries), (b) raw death-over strike rate (bat) /
   economy (bowl), (c) simple pressure delta: strike rate on H − on L (bat) / economy on H − on L (bowl).
   Note: per-ball values come from league-level models (WP, outcome) that were fitted out-of-fold over all seasons; the
   player-level rating stage uses no 2025–26 data.
2. Knockout check (31 matches): top-20 players' run value on H vs all other players. Small sample.
3. Face validity: where named players sit, with a one-line data reason.
4. Vignesh Puthur through the model.
5. Charts.
"""
import numpy as np
import pandas as pd

from . import config as C
from . import rating as RT
from .plotstyle import plt, save, SERIES, TEXT2

N_BOOT = 2000
TEST_MIN_H = 30


def _md(df):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join(out)


def boot_corr(x, y, rng, method="pearson"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    f = (lambda a, b: pd.Series(a).corr(pd.Series(b), method=method))
    est = f(x, y)
    bt = [f(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(N_BOOT))]
    lo, hi = np.nanpercentile(bt, [2.5, 97.5])
    return est, lo, hi


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def conventional_window(years, side):
    b = pd.read_parquet(C.PROC / "balls.parquet")
    b = b[~b.is_penalty & b.year.isin(years)]
    lv = pd.read_parquet(C.PROC / "leverage.parquet")[["match_id", "innings", "innings_ball_num", C.PRIMARY_H]]
    b = b.merge(lv, on=["match_id", "innings", "innings_ball_num"], how="left")
    death = b.phase.astype(str) == "death"
    if side == "bat":
        f = b[b.ball_faced & ~b.is_wide]
        sr = lambda x: 100 * x.groupby("striker_id").batter_runs.sum() / x.groupby("striker_id").size()   # noqa: E731
        H = f[C.PRIMARY_H].fillna(False).astype(bool)
        valid = f[C.PRIMARY_H].notna()
        return pd.DataFrame({"death": sr(f[f.phase.astype(str) == "death"]),
                             "pressure_delta": sr(f[valid & H]) - sr(f[valid & ~H])})
    lg = b[b.legal]
    econ = lambda x, y: 6 * x.groupby("bowler_id").bowler_runs.sum() / y.groupby("bowler_id").size()   # noqa: E731
    H = b[C.PRIMARY_H].fillna(False).astype(bool)
    valid = b[C.PRIMARY_H].notna()
    Hl, vl = lg[C.PRIMARY_H].fillna(False).astype(bool), lg[C.PRIMARY_H].notna()
    # lower economy is better: negate so that higher = better for every predictor
    return pd.DataFrame({"death": -econ(b[death], lg[lg.phase.astype(str) == "death"]),
                         "pressure_delta": -(econ(b[valid & H], lg[vl & Hl]) - econ(b[valid & ~H], lg[vl & ~Hl]))})


def out_of_time(comp, side, rng):
    train, test = comp[comp.year <= 2024], comp[comp.year >= 2025]
    min_h, med, nreg = RT.min_h_rule(train, side)
    rngs = [RT.seed_rng(side, s) for s in RT.SEEDS]
    t, info, P = RT.build(train, side, "H", min_h, rngs, f"{side}: 2022-24", RT.METHOD[side], "strict", active_from=2023)
    rv_tr = train[(train.side == side) & (train.metric == "rv")]
    rv_te = test[(test.side == side) & (test.metric == "rv")]
    y = rv_te[rv_te.H].groupby("player_id").val.agg(["mean", "size"])
    conv = conventional_window([2022, 2023, 2024], side)
    pred = t.set_index("player_id")[["rating", "n_H", "eligible", "eligible_volume", "eligible_active", "rank_score"]]
    pred["career_avg_rv"] = rv_tr.groupby("player_id").val.mean()
    pred = pred.join(conv)
    ev = pred[pred.eligible_volume].join(y.rename(columns={"mean": "y", "size": "n_test"}), how="inner")
    ev = ev[ev.n_test >= TEST_MIN_H].dropna(subset=["rating", "career_avg_rv", "death", "pressure_delta"])
    rows = []
    for lab, col in (("Our rating (2022–24 rebuild)", "rating"), ("(a) career average run value", "career_avg_rv"),
                     ("(b) raw death-over " + ("strike rate" if side == "bat" else "economy (negated)"), "death"),
                     ("(c) simple pressure " + ("SR delta (H − L)" if side == "bat" else "economy delta (negated)"), "pressure_delta")):
        r, lo, hi = boot_corr(ev[col], ev.y, rng)
        s, slo, shi = boot_corr(ev[col], ev.y, rng, "spearman")
        rows.append({"side": side, "predictor": lab, "players": len(ev), "pearson_r": round(r, 3), "pearson_ci95": f"[{lo:.3f}, {hi:.3f}]",
                     "spearman": round(s, 3), "spearman_ci95": f"[{slo:.3f}, {shi:.3f}]", "lo": lo, "hi": hi})
    # top-20 (full rules, 2022-24) -> above average on H in 2025-26?
    top20 = t[t.eligible].sort_values("rank_score", ascending=False).player_id.head(20).tolist()
    avg = ev.y.mean()
    in_ev = [p for p in top20 if p in ev.index]
    k = int((ev.loc[in_ev, "y"] > avg).sum())
    base = float((ev.y > avg).mean())
    lo, hi = wilson(k, len(in_ev))
    share = {"side": side, "top20_2022_24": len(top20), "top20_with_2025_26_H_data": len(in_ev),
             "top20_above_average_on_H_2025_26": k, "share": round(k / max(len(in_ev), 1), 3), "share_ci95_wilson": f"[{lo:.2f}, {hi:.2f}]",
             "share_for_random_eligible_player": round(base, 3), "average_used (runs per H ball, eligible test players)": round(avg, 4),
             "top20_above_expectation_(>0)": int((ev.loc[in_ev, "y"] > 0).sum())}
    meta = {"min_H_2022_24": min_h, "eligible_2022_24": int(t.eligible.sum()), "evaluated_players": len(ev),
            "test_min_H_2025_26": TEST_MIN_H}
    return pd.DataFrame(rows), share, meta


def knockout(comp, side, rng):
    m = pd.read_parquet(C.PROC / "matches.parquet")
    ko = set(m.loc[m.knock_out == 1, "match_id"])
    top20 = pd.read_csv(C.TABLES / ("top10_batters.csv" if side == "bat" else "top10_bowlers.csv")).player_id.tolist()
    rv = comp[(comp.side == side) & (comp.metric == "rv") & comp.H]
    out = []
    for lab, mask in (("knockout matches", rv.match_id.isin(ko)), ("all other matches", ~rv.match_id.isin(ko))):
        x = rv[mask]
        for grp, sel in (("final top 20", x.player_id.isin(top20)), ("all other players", ~x.player_id.isin(top20))):
            z = x[sel]
            mids = z.match_id.unique()
            g = z.groupby("match_id").val.agg(["sum", "size"])
            bt = []
            for _ in range(N_BOOT):
                i = rng.choice(len(g), len(g))
                bt.append(g["sum"].values[i].sum() / g["size"].values[i].sum())
            lo, hi = np.percentile(bt, [2.5, 97.5])
            out.append({"side": side, "matches": lab, "group": grp, "players": z.player_id.nunique(), "H_deliveries": len(z),
                        "n_matches": len(mids), "run_value_per_100_H_balls": round(100 * z.val.mean(), 2),
                        "ci95 (bootstrap over matches)": f"[{100 * lo:.2f}, {100 * hi:.2f}]"})
    return pd.DataFrame(out)


def face_validity(side, names):
    r = pd.read_csv(C.TABLES / f"ratings_all_{side}.csv")
    e = r[r.eligible].sort_values("rank_score", ascending=False).reset_index(drop=True)
    min_h = int(pd.read_csv(C.TABLES / "rating_variants.csv").query("variant == @side + ': primary'").min_H.iat[0])
    rows = []
    for n in names:
        x = r[r.player == n]
        if x.empty:
            rows.append({"player": n, "side": side, "status": "no deliveries in this role", "reason": ""})
            continue
        x = x.iloc[0]
        if x.eligible:
            rk = int(np.where(e.player_id == x.player_id)[0][0]) + 1
            status = f"eligible, rank {rk} of {len(e)}"
            reason = (f"rating {100 * x.rating:+.1f} (core {100 * x.prior:+.1f}, pressure dev {100 * x.pressure_dev:+.1f}); "
                      f"rank score {100 * x.rank_score:+.1f}; raw H {100 * x.H_mean_raw:+.1f} ± {100 * x.H_raw_se:.1f} over {int(x.n_H)} H balls")
        else:
            why = []
            if not x.eligible_volume:
                why.append(f"only {int(x.n_H)} H balls (< {min_h})")
            if not x.eligible_active:
                why.append(f"last season {int(x.last_season)} (needs 2025 or 2026)")
            if x.eligible_volume and x.eligible_active and x.excluded_by_retention:
                why.append(f"retention interval entirely below 0 [{x.R_lo:+.3f}, {x.R_hi:+.3f}]")
            if x.eligible_volume and x.eligible_active and not x.excluded_by_retention and not x.delivers_on_H:
                why.append(f"raw H run value {100 * x.H_mean_raw:+.1f} per 100 balls ≤ 0 (delivery requirement)")
            status = "not eligible"
            reason = "; ".join(why) + f". Rating {100 * x.rating:+.1f} (core {100 * x.prior:+.1f}, pressure dev {100 * x.pressure_dev:+.1f}); raw H {100 * x.H_mean_raw:+.1f}"
        rows.append({"player": n, "side": side, "status": status, "reason": reason})
    return pd.DataFrame(rows)


def vignesh(comp):
    out = []
    for side in ("bat", "bowl"):
        r = pd.read_csv(C.TABLES / f"ratings_all_{side}.csv")
        x = r[r.player == "Vignesh Puthur"]
        c = comp[(comp.side == side) & (comp.metric == "rv")]
        pid = x.player_id.iat[0] if len(x) else None
        if pid is None:
            out.append({"role": side, "in_data": False})
            continue
        d = c[c.player_id == pid]
        x = x.iloc[0]
        se_simple = d[d.H].val.std(ddof=1) / np.sqrt(d.H.sum()) if d.H.sum() > 1 else np.nan
        min_h = int(pd.read_csv(C.TABLES / "rating_variants.csv").query("variant == @side + ': primary'").min_H.iat[0])
        out.append({"role": "batter" if side == "bat" else "bowler", "player_id": int(pid), "matches": d.match_id.nunique(),
                    "seasons": ", ".join(map(str, sorted(d.year.unique()))), "deliveries": len(d), "H_deliveries": int(d.H.sum()),
                    "min_H_for_eligibility": min_h, "eligible": bool(x.eligible),
                    "rating_per_100 (model, shrunk)": round(100 * x.rating, 2), "core_skill_per_100": round(100 * x.prior, 2),
                    "raw_H_per_100": round(100 * x.H_mean_raw, 2) if not np.isnan(x.H_mean_raw) else None,
                    "raw_H_se_per_100 (simple)": round(100 * se_simple, 2) if not np.isnan(se_simple) else None})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------ charts
def chart_trust(path):
    t = pd.read_csv(C.TABLES / "metric_repeatability.csv")
    t = t[t.role != "target"].copy()
    ci = t.ci95_random50.str.extract(r"\[([-\d.]+),\s*([-\d.]+)\]").astype(float)
    t["lo"], t["hi"] = ci[0], ci[1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharex=True)
    for ax, side, title in zip(axes, ("bat", "bowl"), ("Batters", "Bowlers")):
        d = t[t.side == side].sort_values("split_half_random50")
        y = np.arange(len(d))
        col = [SERIES[0] if p else SERIES[1] for p in d["passes (random-split CI > 0)"]]
        ax.hlines(y, d.lo, d.hi, color=col, lw=2)
        ax.scatter(d.split_half_random50, y, color=col, s=36, zorder=3, edgecolor="#fcfcfb", linewidth=1.2)
        ax.axvline(0, color=TEXT2, lw=1)
        ax.set_yticks(y, d.measure, fontsize=8.5)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("Split-half reliability (mean of 50 random splits, Spearman–Brown) with 95% CI")
    axes[0].scatter([], [], color=SERIES[0], label="interval above 0 (passes)")
    axes[0].scatter([], [], color=SERIES[1], label="interval includes 0")
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Trust table — which pressure metrics repeat", fontsize=11.5)
    save(fig, path)


def chart_core_vs_pressure(comp, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, side, title in zip(axes, ("bat", "bowl"), ("Batters", "Bowlers")):
        r = pd.read_csv(C.TABLES / f"ratings_all_{side}.csv")
        e = r[r.eligible]
        rv = comp[(comp.side == side) & (comp.metric == "rv") & comp.player_id.isin(e.player_id)]
        g = rv.groupby(["player_id", "H"]).val.mean().unstack()
        g.columns = ["L", "H"]
        top = pd.read_csv(C.TABLES / ("top10_batters.csv" if side == "bat" else "top10_bowlers.csv")).head(10)
        ax.scatter(100 * g.L, 100 * g.H, s=14, color=SERIES[0], alpha=0.45, edgecolor="none", label="eligible players")
        tg = g.loc[top.player_id]
        ax.scatter(100 * tg.L, 100 * tg.H, s=40, color=SERIES[1], edgecolor="#fcfcfb", linewidth=1.2, zorder=3, label="final top 10")
        for pid, nm in zip(top.player_id, top.player):
            ax.annotate(nm.split(" (")[0], (100 * g.L[pid], 100 * g.H[pid]), fontsize=7.5, xytext=(3, 3), textcoords="offset points")
        lim = np.nanpercentile(np.r_[100 * g.L, 100 * g.H], [1, 99])
        ax.plot(lim, lim, color=TEXT2, lw=0.8, ls="--")
        ax.axhline(0, color=TEXT2, lw=0.6)
        ax.axvline(0, color=TEXT2, lw=0.6)
        ax.set_xlabel("Run value on low-leverage deliveries (L), per 100 balls")
        ax.set_ylabel("Run value on high-leverage deliveries (H), per 100 balls")
        ax.set_title(f"{title}: r(L, H) = {g.L.corr(g.H):.2f}, {len(g)} eligible players", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Core skill vs pressure performance (raw career means; dashed = equal)", fontsize=11.5)
    save(fig, path)


def chart_oot(res, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    for ax, side, title in zip(axes, ("bat", "bowl"), ("Batters", "Bowlers")):
        d = res[res.side == side].reset_index(drop=True)
        y = np.arange(len(d))[::-1]
        ax.hlines(y, d.lo, d.hi, color=SERIES[0], lw=2)
        ax.scatter(d.pearson_r, y, color=[SERIES[1] if i == 0 else SERIES[0] for i in range(len(d))], s=40, zorder=3,
                   edgecolor="#fcfcfb", linewidth=1.2)
        ax.axvline(0, color=TEXT2, lw=1)
        ax.set_yticks(y, d.predictor, fontsize=8.5)
        ax.set_title(f"{title} ({d.players.iat[0]} players)", fontsize=10.5)
        ax.set_xlabel("Correlation with 2025–26 run value on H (Pearson, 95% bootstrap CI)")
    fig.suptitle("Out-of-time test: built on 2022–24, scored on 2025–26", fontsize=11.5)
    save(fig, path)


def run():
    rng = np.random.default_rng(C.SEED)
    comp = RT.load()
    oot, shares, metas = [], [], {}
    for side in ("bat", "bowl"):
        r, sh, meta = out_of_time(comp, side, rng)
        oot.append(r)
        shares.append(sh)
        metas[side] = meta
    oot = pd.concat(oot, ignore_index=True)
    shares = pd.DataFrame(shares)
    oot.drop(columns=["lo", "hi"]).to_csv(C.TABLES / "out_of_time_validation.csv", index=False)
    shares.to_csv(C.TABLES / "out_of_time_top20_share.csv", index=False)
    ko = pd.concat([knockout(comp, s, rng) for s in ("bat", "bowl")], ignore_index=True)
    ko.to_csv(C.TABLES / "knockout_check.csv", index=False)
    fv = pd.concat([face_validity("bat", ["Rinku Singh", "Hardik Pandya", "Virat Kohli", "Mahendra Singh Dhoni"]),
                    face_validity("bowl", ["Arshdeep Singh", "Kuldeep Yadav", "Hardik Pandya"])], ignore_index=True)
    fv.to_csv(C.TABLES / "face_validity.csv", index=False)
    vp = vignesh(comp)
    vp.to_csv(C.TABLES / "vignesh_puthur.csv", index=False)
    chart_trust(C.CHARTS / "trust_table.png")
    chart_core_vs_pressure(comp, C.CHARTS / "core_vs_pressure.png")
    chart_oot(oot, C.CHARTS / "out_of_time_validation.png")
    L = ["# Phase 8 — Validation (lists LOCKED; results reported, not used to change them)\n", __doc__.strip(), "",
         "## 1. Out-of-time test\n", f"Setup per side: {metas}. Evaluated players: eligible by volume in 2022–24 and ≥ {TEST_MIN_H} "
         "H deliveries in 2025–26. Bootstrap over players (2,000).\n", _md(oot.drop(columns=["lo", "hi"])),
         "\nTop-20 of the 2022–24 rebuild (full rules) vs a random eligible player — share above the average of evaluated players on H in 2025–26:\n",
         _md(shares), "\nChart: `outputs/charts/out_of_time_validation.png`\n",
         "## 2. Knockout check (31 knockout matches — SMALL SAMPLE)\n",
         "Run value per 100 H balls vs expectation; 95% interval by bootstrap over matches.\n", _md(ko),
         "\n## 3. Face validity (data reasons only)\n", _md(fv),
         "\n## 4. Vignesh Puthur\n", _md(vp),
         "\n## 5. Charts\n",
         "- WP calibration: `outputs/charts/wp_calibration.png`",
         "- Leverage by innings and phase: `outputs/charts/leverage_share_by_innings_phase.png`",
         "- Trust table: `outputs/charts/trust_table.png`",
         "- Core skill vs pressure (H vs L): `outputs/charts/core_vs_pressure.png`",
         "- Out-of-time validation: `outputs/charts/out_of_time_validation.png`"]
    (C.OUT / "08_validation.md").write_text("\n".join(L))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 140)
    print(metas)
    print(oot.drop(columns=["lo", "hi"]).to_string(index=False))
    print(shares.to_string(index=False))
    print(ko.to_string(index=False))
    print(fv.to_string(index=False))
    print(vp.T.to_string())
