"""Sign and sanity checks, run at the end of each phase by run_all.py.

A failing check raises CheckFailed and stops the pipeline. Checks must never be edited just to make them pass.
Reputation is used here ONLY as a sanity check of signs, never for ranking.
"""
import numpy as np
import pandas as pd

from . import config as C

KNOWN_DEATH_BOWLERS = ["Jasprit Bumrah"]
KNOWN_FINISHERS = ["Heinrich Klaasen", "Tim David", "Andre Russell", "Rinku Singh", "David Miller"]


class CheckFailed(AssertionError):
    pass


def _check(results, name, ok, detail):
    results.append({"check": name, "passed": bool(ok), "detail": detail})


def standardised_means_ok(df, cols, tol=1e-6):
    """Population average of z-scored metrics ≈ 0 (for Phases 6–7 tables)."""
    return {c: float(df[c].mean()) for c in cols if abs(df[c].mean()) > tol}


def run(phase):
    R = []
    if phase >= 1:
        b = pd.read_parquet(C.PROC / "balls.parquet")
        m = pd.read_parquet(C.PROC / "matches.parquet")
        _check(R, "no duplicate deliveries", not b.duplicated(["match_id", "innings", "innings_ball_num"]).any(), "")
        rec = m.recon_ok.dropna()
        _check(R, "result reconciliation ≥ 99%", rec.mean() >= 0.99, f"{rec.mean():.4f}")
        _check(R, "each player_id has one name (striker/bowler)",
               (b.groupby("striker_id").striker_name.nunique() <= 1).all() and (b.groupby("bowler_id").bowler_name.nunique() <= 1).all(), "")
    if phase >= 2:
        p = pd.read_parquet(C.PROC / "wp_preds.parquet")
        _check(R, "WP within [0, 1]", p.wp_before.between(0, 1).all() and p.wp_after.between(0, 1).all(), "")
    if phase >= 3:
        lv = pd.read_parquet(C.PROC / "leverage.parquet")
        _check(R, "mean LI = 1 on included deliveries", abs(lv.LI.mean() - 1) < 1e-3, f"{lv.LI.mean():.5f}")
        sh = lv.groupby("innings")[C.PRIMARY_H].mean()
        _check(R, "primary H ≈ 25% within each innings", sh.between(0.24, 0.26).all(), sh.round(4).to_dict())
    if phase >= 5:
        for f in ("metrics_batters.csv", "metrics_bowlers.csv"):
            t = pd.read_csv(C.TABLES / f)
            _check(R, f"no duplicate player-season in {f}", not t.duplicated(["player_id", "season"]).any(), "")
            _check(R, f"one name per player_id in {f}", (t.groupby("player_id").player.nunique() <= 1).all(), "")
        bw = pd.read_csv(C.TABLES / "metrics_bowlers.csv")
        bw = bw[bw.season == "career"]
        for n in KNOWN_DEATH_BOWLERS:
            v = bw.loc[bw.player == n, "economy_suppression_H"]
            _check(R, f"{n}: economy suppression on H > 0", len(v) == 1 and v.iat[0] > 0, f"{v.round(3).tolist()}")
        bt = pd.read_csv(C.TABLES / "metrics_batters.csv")
        bt = bt[bt.season == "career"]
        for n in KNOWN_FINISHERS:
            v = bt.loc[bt.player == n, "sr_delta_H"]
            _check(R, f"{n}: strike-rate delta on H > 0", len(v) == 1 and v.iat[0] > 0, f"{v.round(2).tolist()}")
        for f in ("roles_batters.csv", "roles_bowlers.csv"):
            t = pd.read_csv(C.TABLES / f)
            _check(R, f"no duplicate player in {f}", not t.player_id.duplicated().any(), "")
    if phase >= 6:
        tr = pd.read_csv(C.TABLES / "metric_repeatability.csv")
        _check(R, "trust table: one row per side × measure", not tr.duplicated(["side", "measure"]).any(), "")
        w = pd.read_csv(C.TABLES / "metric_weights.csv")
        if len(w):
            _check(R, "weights non-negative", (w.weight >= 0).all(), "")
            s = w.groupby("side").weight.sum()
            _check(R, "weights sum to 1 per side", np.allclose(s.values, 1, atol=1e-3), s.round(4).to_dict())
            passed = tr[tr["passes (random-split CI > 0)"]]
            ok = all(((passed.side == r.side) & (passed.measure == r.metric)).any() for r in w.itertuples())
            _check(R, "only metrics that passed reliability carry weight", ok, "")
        sr = pd.read_csv(C.TABLES / "signal_recovery_summary.csv")
        _check(R, "6B summary: one row per method × side × variant",
               not sr.duplicated(["method", "side", "variant"]).any(), "")
        _check(R, "6B summary covers methods A–E", set(sr.method.str[0]) >= set("ABCDE"), sorted(set(sr.method.str[0])))
        zm = pd.read_csv(C.TABLES / "phase6_feature_z_means.csv")
        _check(R, "standardised feature means ≈ 0", (zm.mean_z.abs() < 1e-6).all(), zm.mean_z.abs().max())
    if phase >= 7:
        var = pd.read_csv(C.TABLES / "rating_variants.csv")
        for side, f in (("bat", "top10_batters.csv"), ("bowl", "top10_bowlers.csv")):
            t = pd.read_csv(C.TABLES / f)
            mh = int(var[(var.side == side) & var.variant.str.endswith("primary")].min_H.iat[0])
            _check(R, f"{f}: 20 rows, unique players", len(t) == 20 and not t.player_id.duplicated().any(), len(t))
            _check(R, f"{f}: all meet H-delivery minimum ({mh})", (t.n_H >= mh).all(), int(t.n_H.min()))
            _check(R, f"{f}: all active in 2025 or 2026", t.seasons_played.str.contains("2025|2026").all(), "")
            lo = t.R_ci95.str.extract(r"\[([^,]+),\s*([^\]]+)\]").astype(float)
            _check(R, f"{f}: nobody with R interval entirely below 0", (lo[1] >= 0).all(), "")
            _check(R, f"{f}: rank score = rating − SE", np.allclose(t.rank_score_per_100,
                   t.rating_runs_per_100_H_balls - t.se_per_100, atol=0.02), "")
            _check(R, f"{f}: sorted by rank score", t.rank_score_per_100.is_monotonic_decreasing, "")
            _check(R, f"{f}: delivery condition (raw H run value > 0) for every listed player", (t.raw_H_per_100 > 0).all(),
                   float(t.raw_H_per_100.min()))
            _check(R, f"{f}: every name matches the data spelling", t.name_matches_data.all(), t.player[~t.name_matches_data].tolist())
            _check(R, f"{f}: names shared by two ids carry a team tag",
                   (~t.name_shared_by_two_ids | t.player.str.contains(r"\(")).all(), t.player[t.name_shared_by_two_ids].tolist())
            _check(R, f"{f}: tiers start at 1 and never decrease down the list", t.tier.iat[0] == 1 and t.tier.is_monotonic_increasing, "")
        tb = pd.read_csv(C.TABLES / "top10_bowlers.csv")
        tau = float(var[(var.side == "bowl") & var.variant.str.endswith("primary")].tau2_dev_used.iat[0])
        _check(R, "bowlers: H-update weight is 0 when pressure-specific variance ≈ 0",
               (tau > 0) or (tb.k_update.abs() < 1e-9).all(), tau)
    if phase >= 8:
        for side, f in (("bat", "top10_batters.csv"), ("bowl", "top10_bowlers.csv")):
            cur = pd.read_csv(C.TABLES / f).player_id.tolist()
            locked = pd.read_csv(C.TABLES / ("locked_" + f)).player_id.tolist()
            _check(R, f"{f}: identical to the LOCKED list (order and players)", cur == locked, "")
    if phase >= 9:
        cards = list((C.OUT / "cards").glob("*.md"))
        _check(R, "20 player cards", len(cards) == 20, len(cards))
        fin = (C.OUT / "FINAL.md").read_text()
        names = pd.concat([pd.read_csv(C.TABLES / "top10_batters.csv").player, pd.read_csv(C.TABLES / "top10_bowlers.csv").player])
        _check(R, "FINAL.md contains every final top-20 name", all(n in fin for n in names), [n for n in names if n not in fin])
        _check(R, "FINAL.md has no missing sections", "not found)" not in fin, fin.count("not found)"))
        _check(R, "FINAL.md opening line present",
               "Pressure skill is real. We measured it, tested that it lasts, and ranked the players who have it." in fin, "")
        for i in range(7):
            _check(R, f"FINAL.md section {i} present", f"\n## {i}. " in fin, "")
        import re as _re
        missing = [p_ for p_ in set(_re.findall(r"`(outputs/[^`]+\.(?:png|csv|md))`", fin)) if not (C.ROOT / p_).exists()]
        _check(R, "every file referenced in FINAL.md exists", not missing, missing[:5])
    res = pd.DataFrame(R)
    res.to_csv(C.TABLES / f"checks_phase{phase}.csv", index=False)
    failed = res[~res.passed]
    print(f"[checks] phase {phase}: {int(res.passed.sum())}/{len(res)} passed")
    if len(failed):
        print(failed.to_string(index=False))
        raise CheckFailed(f"{len(failed)} check(s) failed after phase {phase}")
    return res
