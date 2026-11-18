"""Phase 6: repeatability of the candidate metrics, target choice and weights.

Part B (run first): pressure-band check — split-half reliability of P and the skill metrics for different
definitions of the high-leverage set.
"""
import numpy as np
import pandas as pd

from . import config as C

SKILL = {"bat": ["pressure_v", "sr_delta", "dot_suppression", "boundary_elevation", "wicket_security"],
         "bowl": ["pressure_v", "economy_suppression", "dot_rate", "wicket_taking", "boundary_suppression"]}
MIN_N = 100          # band deliveries per player (career), as in Phase 5
N_BOOT = 200


def innings_cuts():
    lv = pd.read_parquet(C.PROC / "leverage.parquet")[["innings", "LI"]]
    return {q: lv.groupby("innings").LI.quantile(q).to_dict() for q in (0.5, 2 / 3, 0.75)}, float(lv.LI.mean())


def halves(c):
    """Odd/even matches in date order within each player."""
    order = c[["player_id", "match_date", "match_id"]].drop_duplicates().sort_values(["player_id", "match_date", "match_id"])
    order["half"] = order.groupby("player_id").cumcount() % 2
    return c.merge(order[["player_id", "match_id", "half"]], on=["player_id", "match_id"])


def split_half_stats(c, w=None, players=None, rng=None):
    c = c.assign(w=1.0 if w is None else w)
    c = c.assign(wv=c.w * c.val)
    g = c.groupby(["player_id", "half"]).agg(sw=("w", "sum"), swv=("wv", "sum"), n=("val", "size")).unstack()
    tot = g[("n", 0)].fillna(0) + g[("n", 1)].fillna(0)
    keep = (tot >= MIN_N) & (g[("n", 0)] >= MIN_N / 4) & (g[("n", 1)] >= MIN_N / 4)
    if players is not None:
        keep &= g.index.isin(players) & (g[("n", 0)] >= 5) & (g[("n", 1)] >= 5)
    g = g[keep]
    a, b = (g[("swv", 0)] / g[("sw", 0)]).values, (g[("swv", 1)] / g[("sw", 1)]).values

    def sb(x, y):
        r = np.corrcoef(x, y)[0, 1]
        return 2 * r / (1 + r) if r > -1 else np.nan

    est = sb(a, b)
    idx = rng.integers(0, len(a), (N_BOOT, len(a)))
    boots = np.array([sb(a[i], b[i]) for i in idx])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return len(a), est, lo, hi, set(g.index)


def band_check():
    rng = np.random.default_rng(C.SEED)
    comp = pd.read_parquet(C.PROC / "metric_components.parquet")
    cuts, _ = innings_cuts()
    rows, conc = [], []
    lv = pd.read_parquet(C.PROC / "leverage.parquet")[["innings", "LI", C.PRIMARY_H]]
    versions = {"H25 (primary)": (0.75, False), "H33": (2 / 3, False), "H50": (0.5, False), "LI-weighted, LI > median": (0.5, True)}
    for name, (q, wtd) in versions.items():
        thr = lv.innings.map(cuts[q])
        sel = lv.LI >= thr
        w = lv.LI[sel] if wtd else pd.Series(1.0, index=lv.index[sel])
        conc.append({"version": name, "share_of_deliveries": round(float(sel.mean()), 3),
                     "mean_LI_in_set": round(float((lv.LI[sel] * w).sum() / w.sum()), 3),
                     "share_of_top25_deliveries_in_set": round(float(lv[C.PRIMARY_H][sel].mean()), 3) if not wtd else
                     round(float((lv[C.PRIMARY_H][sel] * w).sum() / w.sum()), 3)})
    common = {}
    for side, mets in SKILL.items():
        for met in mets:
            c0 = comp[(comp.side == side) & (comp.metric == met)].copy()
            c0["thr"] = c0.innings.map(cuts[0.75])
            for name, (q, wtd) in versions.items():
                thr = c0.innings.map(cuts[q])
                c = halves(c0[c0.LI >= thr])
                w = c.LI.values if wtd else None
                n, est, lo, hi, pl = split_half_stats(c, w, rng=rng)
                if name.startswith("H25"):
                    common[(side, met)] = pl
                nc, estc, loc, hic, _ = split_half_stats(c, w, players=common[(side, met)], rng=rng)
                rows.append({"side": side, "metric": met, "version": name, "players": n,
                             "reliability": round(est, 3), "ci95": f"[{lo:.2f}, {hi:.2f}]",
                             "players_common_set": nc, "reliability_common_set": round(estc, 3),
                             "ci95_common_set": f"[{loc:.2f}, {hic:.2f}]"})
    res = pd.DataFrame(rows)
    res.to_csv(C.TABLES / "pressure_band_check.csv", index=False)
    conc = pd.DataFrame(conc)
    conc.to_csv(C.TABLES / "pressure_band_concentration.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 100)
    print(conc.to_string(index=False))
    print(res.to_string(index=False))
    return res, conc


# =====================================================================================================
# Part C/D: repeatability of every candidate, target choice, pressure-deviation variance, weights
# =====================================================================================================
from sklearn.linear_model import Ridge                                   # noqa: E402
from .plotstyle import plt, save, SERIES, TEXT2                          # noqa: E402
from . import metrics as MT                                              # noqa: E402

N_SPLITS = 50
MIN = {"skill": 100, "value": 100, "temperament": 30, "variation": 60, "partner": 100, "target": 100}
D_MIN_THIS = 60          # deliveries this season (all) to enter the weights data
D_MIN_NEXT = {"H": 30, "all": 60, "LIw": 60}
ALPHAS = [0.01, 0.1, 1, 3, 10, 30, 100, 300, 1000]
PROCESS = {"bat": ["sr_delta", "dot_suppression", "boundary_elevation"],
           "bowl": ["dot_rate", "boundary_suppression", "economy_suppression", "wicket_taking"]}


def measure_list():
    M = []

    def add(side, name, typ, parts, sset, kind, role):
        M.append(dict(side=side, name=name, type=typ, parts=parts, set=sset, kind=kind, role=role, min_n=MIN[kind]))
    for m in ["pressure_v", "sr_delta", "dot_suppression", "boundary_elevation", "wicket_security"]:
        add("bat", m, "mean", [m], "H", "value" if m == "pressure_v" else "skill", "candidate")
    add("bat", "squeeze_response (pooled H+L)", "mean", ["squeeze_response"], "pooled", "temperament", "candidate")
    for m in ["entry_resilience", "collapse_resistance", "post_wicket_response", "runout_avoidance"]:
        add("bat", m, "mean", [m], "H", "temperament", "candidate")
    add("bat", "partner_effect (on trial)", "partner", [], "H", "partner", "candidate")
    add("bat", "retention R (P − B)", "diff", ["pressure_v"], "H-L", "value", "condition")
    for m in ["pressure_v", "economy_suppression", "dot_rate", "wicket_taking", "boundary_suppression"]:
        add("bowl", m, "mean", [m], "H", "value" if m == "pressure_v" else "skill", "candidate")
    for m in ["bounce_back", "post_wicket_response"]:
        add("bowl", m, "mean", [m], "H", "temperament", "candidate")
    add("bowl", "execution_drift (H − L)", "diff", ["execution_error"], "H-L", "skill", "candidate")
    add("bowl", "variation (entropy H − L)", "entropy", [], "H-L", "variation", "candidate")
    add("bowl", "retention R (P − B)", "diff", ["pressure_v"], "H-L", "value", "condition")
    for side in ("bat", "bowl"):
        add(side, "TARGET run value on H", "mean", ["rv"], "H", "target", "target")
        add(side, "TARGET run value, all deliveries", "mean", ["rv"], "all", "target", "target")
        add(side, "TARGET run value, all, LI-weighted", "mean", ["rv"], "LIw", "target", "target")
        add(side, "TARGET process composite on H", "zmean", PROCESS[side], "H", "target", "target")
        add(side, "TARGET process composite, all deliveries", "zmean", PROCESS[side], "all", "target", "target")
    return M


# ------------------------------------------------------------------ per (player, match, part) aggregates
def build_agg(meas, comp, extra):
    if meas["type"] == "custom":                      # Phase 6B methods supply their own aggregates
        return meas["agg"]
    c = comp[(comp.side == meas["side"])]
    if meas["type"] in ("mean", "zmean", "diff"):
        x = c[c.metric.isin(meas["parts"])]
        if meas["set"] == "H":
            x = x[x.H]
        w = x.LI.astype(float) if meas["set"] == "LIw" else 1.0
        part = np.where(x.H, "H", "L") if meas["type"] == "diff" else x.metric.astype(str)
        x = x.assign(w=w, part=part, wv=lambda t: t.w * t.val)
        a = x.groupby(["player_id", "match_id", "part"], observed=True).agg(
            sw=("w", "sum"), swv=("wv", "sum"), n=("val", "size"), match_date=("match_date", "first"),
            year=("year", "first")).reset_index()
        return a
    if meas["type"] == "entropy":
        return extra["styles"]
    if meas["type"] == "partner":
        return extra["pairs"]


def values(meas, agg, g, extra):
    """Measure value per (player, group). `g` = Series aligned with agg rows (group label)."""
    a = agg.assign(g=np.asarray(g))
    t = meas["type"]
    if t == "custom":
        return meas["fn"](a)
    if t == "mean":
        s = a.groupby(["player_id", "g"]).agg(sw=("sw", "sum"), swv=("swv", "sum"), n=("n", "sum"))
        return pd.DataFrame({"value": s.swv / s.sw, "n": s.n})
    if t == "diff":
        s = a.groupby(["player_id", "g", "part"]).agg(sw=("sw", "sum"), swv=("swv", "sum"), n=("n", "sum")).unstack("part")
        ok = (s[("n", "H")] > 0) & (s[("n", "L")] > 0)
        s = s[ok]
        return pd.DataFrame({"value": s[("swv", "H")] / s[("sw", "H")] - s[("swv", "L")] / s[("sw", "L")], "n": s[("n", "H")]})
    if t == "zmean":
        s = a.groupby(["player_id", "g", "part"]).agg(sw=("sw", "sum"), swv=("swv", "sum"), n=("n", "sum"))
        s["m"] = s.swv / s.sw
        s = s.reset_index()
        # z within each group level across players with enough deliveries in that part
        ok = s.n >= 10
        s["z"] = np.nan
        s.loc[ok, "z"] = s[ok].groupby(["g", "part"]).m.transform(lambda v: (v - v.mean()) / v.std(ddof=0))
        out = s.groupby(["player_id", "g"]).agg(value=("z", "mean"), n=("n", "max"))
        return out
    if t == "entropy":
        s = a.groupby(["player_id", "g", "part", "style"]).n.sum()
        tot = s.groupby(level=[0, 1, 2]).transform("sum")
        p = s / tot
        ent = (-(p * np.log2(p))).groupby(level=[0, 1, 2]).sum()
        k = s.groupby(level=[0, 1, 2]).size()
        nn = s.groupby(level=[0, 1, 2]).sum()
        ent = (ent + (k - 1) / (2 * nn * np.log(2))).unstack("part")
        nn = nn.unstack("part")
        ok = ent.notna().all(axis=1)
        return pd.DataFrame({"value": (ent["H"] - ent["L"])[ok], "n": nn["H"][ok]})
    if t == "partner":
        s = a.groupby(["player_id", "g", "striker_id"]).agg(sv=("sv", "sum"), n=("n", "sum")).reset_index()
        s = s.merge(extra["pair_tot"], on=["player_id", "striker_id"]).merge(extra["str_tot"], on="striker_id")
        s = s[s.tn > s.pn]
        s["without"] = (s.tsv - s.psv) / (s.tn - s.pn)
        s["dw"] = s.sv - s.n * s.without
        out = s.groupby(["player_id", "g"]).agg(dw=("dw", "sum"), n=("n", "sum"))
        return pd.DataFrame({"value": out.dw / out.n, "n": out.n})


# ------------------------------------------------------------------ correlation helpers
def wcorr(x, y, w):
    w = w / w.sum()
    mx, my = (w * x).sum(), (w * y).sum()
    cov = (w * (x - mx) * (y - my)).sum()
    return cov / np.sqrt((w * (x - mx) ** 2).sum() * (w * (y - my) ** 2).sum())


def sb(r):
    return 2 * r / (1 + r) if r > -1 else np.nan


def two_group(meas, agg, g, extra, eligible, min_half):
    v = values(meas, agg, g, extra)
    v = v[v.index.get_level_values(0).isin(eligible)]
    a = v.xs(0, level="g").reindex(eligible)
    b = v.xs(1, level="g").reindex(eligible)
    ok = (a.n >= min_half) & (b.n >= min_half)
    return np.where(ok, a.value, np.nan), np.where(ok, b.value, np.nan)


def corr_cols(A, B):
    """Pearson per column ignoring NaN pairs (A, B: players x splits)."""
    m = ~(np.isnan(A) | np.isnan(B))
    n = m.sum(0)
    A0, B0 = np.where(m, A, 0.0), np.where(m, B, 0.0)
    ma, mb = A0.sum(0) / n, B0.sum(0) / n
    da, db = np.where(m, A - ma, 0.0), np.where(m, B - mb, 0.0)
    return (da * db).sum(0) / np.sqrt((da ** 2).sum(0) * (db ** 2).sum(0))


def reliability(meas, comp, extra, rng):
    agg = build_agg(meas, comp, extra)
    key = agg[["player_id", "match_id", "match_date"]].drop_duplicates(["player_id", "match_id"])
    tot = values(meas, agg, np.zeros(len(agg), int), extra).xs(0, level="g")
    nm = key.groupby("player_id").size()
    eligible = tot.index[(tot.n >= meas["min_n"]) & tot.index.isin(nm.index[nm >= 4])].values
    min_half = meas["min_n"] / 4
    res = {"side": meas["side"], "measure": meas["name"], "role": meas["role"], "set": meas["set"],
           "min_deliveries": meas["min_n"], "players": len(eligible)}
    if len(eligible) < 10:
        return res, None
    # random splits
    k = key.copy()
    A, B = [], []
    for s in range(N_SPLITS):
        k["r"] = rng.random(len(k))
        k["rank"] = k.groupby("player_id").r.rank(method="first")
        k["half"] = (k["rank"] > k.player_id.map(nm) / 2).astype(int)
        g = agg[["player_id", "match_id"]].merge(k[["player_id", "match_id", "half"]], how="left").half.values
        a, b = two_group(meas, agg, g, extra, eligible, min_half)
        A.append(a), B.append(b)
    A, B = np.column_stack(A), np.column_stack(B)
    est = np.nanmean([sb(r) for r in corr_cols(A, B)])
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(eligible), len(eligible))
        boots.append(np.nanmean([sb(r) for r in corr_cols(A[i], B[i])]))
    res.update({"split_half_random50": round(est, 3),
                "ci95_random50": f"[{np.nanpercentile(boots, 2.5):.2f}, {np.nanpercentile(boots, 97.5):.2f}]",
                "ci_lo": np.nanpercentile(boots, 2.5), "ci_hi": np.nanpercentile(boots, 97.5),
                "players_in_splits": int(np.nanmean((~np.isnan(A) & ~np.isnan(B)).sum(0)))})
    # odd/even (date order) and chronological halves
    k = key.sort_values(["player_id", "match_date", "match_id"]).copy()
    k["pos"] = k.groupby("player_id").cumcount()
    k["oe"] = k.pos % 2
    k["chrono"] = (k.pos >= k.player_id.map(nm) / 2).astype(int)
    for lab, col in (("odd_even", "oe"), ("chronological", "chrono")):
        g = agg[["player_id", "match_id"]].merge(k[["player_id", "match_id", col]], how="left")[col].values
        a, b = two_group(meas, agg, g, extra, eligible, min_half)
        ok = ~(np.isnan(a) | np.isnan(b))
        r = np.corrcoef(a[ok], b[ok])[0, 1]
        bt = []
        for _ in range(N_BOOT):
            i = rng.integers(0, ok.sum(), ok.sum())
            bt.append(sb(np.corrcoef(a[ok][i], b[ok][i])[0, 1]))
        res[f"split_half_{lab}"] = round(sb(r), 3)
        res[f"ci95_{lab}"] = f"[{np.nanpercentile(bt, 2.5):.2f}, {np.nanpercentile(bt, 97.5):.2f}]"
        if lab == "chronological":
            scatter = (a[ok], b[ok])
    # season t -> t+1 (adjacent calendar years; weights = harmonic mean of deliveries)
    v = values(meas, agg, agg.year.values, extra).reset_index()
    v = v[v.n >= meas["min_n"] / 2]
    nxt = v.assign(g=v.g - 1)
    pr = v.merge(nxt, on=["player_id", "g"], suffixes=("_t", "_t1"))
    if len(pr) >= 10:
        w = 2 / (1 / pr.n_t + 1 / pr.n_t1)
        r = wcorr(pr.value_t.values, pr.value_t1.values, w.values)
        pids = pr.player_id.unique()
        bt = []
        for _ in range(N_BOOT):
            s_ = rng.choice(pids, len(pids))
            q = pd.DataFrame({"player_id": s_}).merge(pr, on="player_id")
            bt.append(wcorr(q.value_t.values, q.value_t1.values, (2 / (1 / q.n_t + 1 / q.n_t1)).values))
        res.update({"season_pairs": len(pr), "season_players": len(pids), "season_r_weighted": round(r, 3),
                    "ci95_season": f"[{np.nanpercentile(bt, 2.5):.2f}, {np.nanpercentile(bt, 97.5):.2f}]"})
    else:
        res.update({"season_pairs": len(pr), "season_players": pr.player_id.nunique()})
    return res, scatter


def extras(d):
    sty = d[d.delivery_style.notna()]
    styles = sty.assign(part=np.where(sty.H, "H", "L"), style=sty.delivery_style.astype(str)).groupby(
        ["bowler_id", "match_id", "part", "style"]).agg(n=("LI", "size"), match_date=("match_date", "first"),
                                                       year=("year", "first")).reset_index().rename(columns={"bowler_id": "player_id"})
    x = d[d.H & d.ball_faced & ~d.is_wide]
    pairs = x.groupby(["non_striker_id", "match_id", "striker_id"]).agg(
        sv=("v_bat", "sum"), n=("v_bat", "size"), match_date=("match_date", "first"), year=("year", "first")).reset_index()
    pairs = pairs.rename(columns={"non_striker_id": "player_id"})
    pair_tot = pairs.groupby(["player_id", "striker_id"]).agg(psv=("sv", "sum"), pn=("n", "sum")).reset_index()
    str_tot = x.groupby("striker_id").v_bat.agg(tsv="sum", tn="size").reset_index()
    return {"styles": styles, "pairs": pairs, "pair_tot": pair_tot, "str_tot": str_tot}


def scatter_grid(items, title, path):
    n = len(items)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.3 * rows))
    for ax, (name, (a, b), rel) in zip(axes.flat, items):
        ax.scatter(a, b, s=10, color=SERIES[0], alpha=0.55, edgecolor="none")
        lim = np.nanpercentile(np.r_[a, b], [1, 99])
        ax.plot(lim, lim, color=TEXT2, lw=0.8, ls="--")
        ax.set_title(f"{name}\nchronological r(SB) = {rel}", fontsize=8.5)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(title + " — earlier half vs later half of each player's matches", fontsize=11)
    fig.supxlabel("earlier half", fontsize=9)
    fig.supylabel("later half", fontsize=9)
    save(fig, path)


# ------------------------------------------------------------------ pressure-deviation variance (for Phase 7)
def deviation_variance(comp, side, rng):
    c = comp[(comp.side == side) & (comp.metric == "rv")]
    g = c.groupby(["player_id", "H"]).val.agg(["mean", "var", "size"]).unstack("H")
    g = g[(g[("size", True)] >= 100) & (g[("size", False)] >= 100)]
    h, l = g[("mean", True)].values, g[("mean", False)].values
    seh, sel = (g[("var", True)] / g[("size", True)]).values, (g[("var", False)] / g[("size", False)]).values

    def est(i):
        tH = np.var(h[i], ddof=1) - seh[i].mean()
        tL = np.var(l[i], ddof=1) - sel[i].mean()
        cov = np.cov(h[i], l[i])[0, 1]
        dev = np.var(h[i] - l[i], ddof=1) - (seh[i] + sel[i]).mean()
        resid = tH - cov ** 2 / tL if tL > 0 else np.nan
        return tH, tL, cov, dev, resid
    full = est(np.arange(len(h)))
    bt = np.array([est(rng.integers(0, len(h), len(h))) for _ in range(N_BOOT)])
    lo, hi = np.nanpercentile(bt, 2.5, axis=0), np.nanpercentile(bt, 97.5, axis=0)
    names = ["true var of H run value (tau2_H)", "true var of L run value (tau2_L)", "true cov(H, L)",
             "true var of (H − L) deviation", "pressure-specific residual var: tau2_H − cov²/tau2_L"]
    out = pd.DataFrame({"side": side, "quantity": names, "estimate": full, "ci_lo": lo, "ci_hi": hi})
    out["share_of_tau2_H"] = out.estimate / full[0]
    out["players"] = len(h)
    out["mean_sampling_var_H"] = seh.mean()
    return out


# ------------------------------------------------------------------ weights (Phase 6 D)
def season_values(meas, comp, extra):
    agg = build_agg(meas, comp, extra)
    return values(meas, agg, agg.year.values, extra).reset_index().rename(columns={"g": "year"})


def shrink_c(meas, comp, sv):
    """c = within-player variance per delivery / between-player variance (method of moments) for mean-type metrics;
    for other types c from the measured split-half reliability (r = n/(n+c) at the mean half size)."""
    if meas["type"] == "mean":
        x = comp[(comp.side == meas["side"]) & comp.metric.isin(meas["parts"])]
        x = x[x.H] if meas["set"] == "H" else x
        s2 = float(x.val.var())
        tau2 = max(np.average((sv.value - np.average(sv.value, weights=sv.n)) ** 2, weights=sv.n) - s2 * len(sv) / sv.n.sum(), 1e-12)
        return s2 / tau2
    r = max(meas.get("half_r", 0.05), 0.02)
    return meas.get("half_n", 50) * (1 - r) / r


def weights(side, passed, target_meas, comp, extra, base_meas, rng):
    # features (season t) — shrunk, centred, standardised
    feats, cs = {}, {}
    for m in passed:
        sv = season_values(m, comp, extra)
        sv = sv[sv.n > 0]
        c = shrink_c(m, comp, sv)
        mu = np.average(sv.value, weights=sv.n)
        sv["theta"] = sv.n / (sv.n + c) * (sv.value - mu)
        feats[m["name"]] = sv.set_index(["player_id", "year"]).theta
        cs[m["name"]] = round(float(c), 1)
    F = pd.DataFrame(feats).fillna(0.0)
    bsv = season_values(base_meas, comp, extra)
    cb = shrink_c(base_meas, comp, bsv)
    bsv["theta"] = bsv.n / (bsv.n + cb) * (bsv.value - np.average(bsv.value, weights=bsv.n))
    base = bsv.set_index(["player_id", "year"])
    tv = season_values(target_meas, comp, extra).set_index(["player_id", "year"])
    # this-season volume (all deliveries) and next-season target
    allv = comp[(comp.side == side) & (comp.metric == "pressure_v")].groupby(["player_id", "year"]).size().rename("n_this")
    df = pd.DataFrame({"n_this": allv}).join(F, how="left").join(base.theta.rename("baseline"), how="left")
    nxt = tv.rename(columns={"value": "y", "n": "n_next"}).reset_index()
    nxt["year"] = nxt.year - 1
    df = df.reset_index().merge(nxt, on=["player_id", "year"])
    df = df[(df.n_this >= D_MIN_THIS) & (df.n_next >= D_MIN_NEXT[target_meas["set"]])].fillna(0.0)
    df["w"] = 2 / (1 / df.n_this + 1 / df.n_next)
    names = list(feats)
    Z = df[names].values
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1)
    zb = df.baseline.values
    zb = (zb - zb.mean()) / zb.std()
    y, w, pid = df.y.values, df.w.values, df.player_id.values
    # grouped folds by player
    up = np.unique(pid)
    fold_of = dict(zip(rng.permutation(up), np.arange(len(up)) % 5))
    fo = np.array([fold_of[p] for p in pid])

    def fit_ridge(X, yy, ww, alpha):
        return Ridge(alpha=alpha, positive=True).fit(X, yy, sample_weight=ww)

    def choose_alpha(X, yy, ww, gg):
        inner = np.array([hash((int(p), 7)) % 4 for p in gg])
        best, bl = None, np.inf
        for a in ALPHAS:
            err = 0.0
            for f in range(4):
                tr, te = inner != f, inner == f
                if tr.sum() < 10 or te.sum() < 5:
                    continue
                m_ = fit_ridge(X[tr], yy[tr], ww[tr], a)
                err += (ww[te] * (yy[te] - m_.predict(X[te])) ** 2).sum()
            if err < bl:
                best, bl = a, err
        return best

    def lin(score_tr, y_tr, w_tr, score_te):
        A = np.c_[np.ones(len(score_tr)), score_tr]
        coef = np.linalg.lstsq(A * np.sqrt(w_tr)[:, None], y_tr * np.sqrt(w_tr), rcond=None)[0]
        return coef[0] + coef[1] * score_te

    pred = {k: np.empty(len(y)) for k in ["ridge (non-negative)", "equal weights (passed metrics)", "baseline skill only"]}
    alphas = []
    for f in range(5):
        tr, te = fo != f, fo == f
        a = choose_alpha(Z[tr], y[tr], w[tr], pid[tr])
        alphas.append(a)
        pred["ridge (non-negative)"][te] = fit_ridge(Z[tr], y[tr], w[tr], a).predict(Z[te])
        eq = Z.mean(1)
        pred["equal weights (passed metrics)"][te] = lin(eq[tr], y[tr], w[tr], eq[te])
        pred["baseline skill only"][te] = lin(zb[tr], y[tr], w[tr], zb[te])
    comp_rows = []
    for k, p in pred.items():
        r = wcorr(p, y, w)
        sse = (w * (y - p) ** 2).sum()
        sst = (w * (y - np.average(y, weights=w)) ** 2).sum()
        bt = []
        for _ in range(N_BOOT):
            s_ = rng.choice(up, len(up))
            idx = np.concatenate([np.where(pid == q)[0] for q in s_])
            bt.append(wcorr(p[idx], y[idx], w[idx]))
        comp_rows.append({"side": side, "model": k, "player_seasons": len(y), "players": len(up),
                          "oof_weighted_r": round(r, 3), "ci95": f"[{np.nanpercentile(bt, 2.5):.2f}, {np.nanpercentile(bt, 97.5):.2f}]",
                          "oof_R2": round(1 - sse / sst, 4)})
    # difference ridge − equal, bootstrap over players
    dif = []
    for _ in range(N_BOOT):
        s_ = rng.choice(up, len(up))
        idx = np.concatenate([np.where(pid == q)[0] for q in s_])
        dif.append(wcorr(pred["ridge (non-negative)"][idx], y[idx], w[idx]) - wcorr(pred["equal weights (passed metrics)"][idx], y[idx], w[idx]))
    alpha_all = choose_alpha(Z, y, w, pid)
    final = fit_ridge(Z, y, w, alpha_all)
    beta = final.coef_
    wts = beta / beta.sum() if beta.sum() > 0 else np.full(len(beta), 1 / len(beta))
    W_ = pd.DataFrame({"side": side, "metric": names, "ridge_coef": beta.round(6), "weight": wts.round(4),
                       "shrinkage_c": [cs[n] for n in names]})
    info = {"alpha_per_fold": alphas, "alpha_final": alpha_all,
            "ridge_minus_equal_r": f"{np.nanmean(dif):.3f} [{np.nanpercentile(dif, 2.5):.3f}, {np.nanpercentile(dif, 97.5):.3f}]",
            "z_feature_means": np.round(Z.mean(0), 8).tolist()}
    return W_, pd.DataFrame(comp_rows), info


def _md(df):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join(out)


def run():
    band_check()                                              # Part B (pressure-band check), regenerated every run
    rng = np.random.default_rng(C.SEED)
    comp = pd.read_parquet(C.PROC / "metric_components.parquet")
    d, _ = MT.load()
    extra = extras(d)
    M = measure_list()
    rows, scat = [], {"bat": [], "bowl": [], "targets": []}
    for meas in M:
        res, sc = reliability(meas, comp, extra, rng)
        rows.append(res)
        if sc is not None:
            key = "targets" if meas["role"] == "target" else meas["side"]
            scat[key].append(((meas["side"] + ": " if key == "targets" else "") + meas["name"], sc, res.get("split_half_chronological")))
        print(f"[phase6] {meas['side']:4s} {meas['name']:42s} players {res['players']:4d} "
              f"random50 {res.get('split_half_random50')} {res.get('ci95_random50', '')}")
    trust = pd.DataFrame(rows)
    trust["passes (random-split CI > 0)"] = trust.ci_lo > 0
    trust_out = trust.drop(columns=["ci_lo", "ci_hi"])
    trust_out.to_csv(C.TABLES / "metric_repeatability.csv", index=False)
    scatter_grid(scat["bat"], "Batters: candidate metrics", C.CHARTS / "repeatability_batters.png")
    scatter_grid(scat["bowl"], "Bowlers: candidate metrics", C.CHARTS / "repeatability_bowlers.png")
    scatter_grid(scat["targets"], "Target candidates", C.CHARTS / "repeatability_targets.png")

    # ---- choose targets
    chosen, dev, Wt, cmp_, infos = {}, [], [], [], {}
    for side in ("bat", "bowl"):
        t = trust[(trust.side == side) & (trust.role == "target")].copy()
        press = t[t.set.isin(["H", "LIw"]) & (t.ci_lo > 0)]
        pick = press.sort_values("split_half_random50", ascending=False).iloc[0] if len(press) else \
            t[t.set == "all"].sort_values("split_half_random50", ascending=False).iloc[0]
        chosen[side] = pick
        dev.append(deviation_variance(comp, side, rng))
        # weights: candidates passing reliability (random-split CI > 0), excluding retention (a condition)
        cand = [m for m in M if m["side"] == side and m["role"] == "candidate"]
        passed = []
        for m in cand:
            r_ = trust[(trust.side == side) & (trust.measure == m["name"])].iloc[0]
            if r_.ci_lo > 0:
                rh = r_.split_half_random50 / (2 - r_.split_half_random50)    # back to half-length r
                passed.append({**m, "half_r": rh, "half_n": r_.min_deliveries / 2})
        tm = [m for m in M if m["side"] == side and m["name"] == pick.measure][0]
        bm = [m for m in M if m["side"] == side and m["name"] == "TARGET run value, all deliveries"][0]
        if passed:
            W_, c_, info = weights(side, passed, tm, comp, extra, bm, rng)
        else:
            W_, c_, info = pd.DataFrame(), pd.DataFrame(), {"note": "no candidate metric passed reliability"}
        Wt.append(W_), cmp_.append(c_)
        infos[side] = info
    zrows = [{"side": sd, "feature_index": i, "mean_z": v} for sd, inf in infos.items() for i, v in enumerate(inf.get("z_feature_means", []))]
    pd.DataFrame(zrows, columns=["side", "feature_index", "mean_z"]).to_csv(C.TABLES / "phase6_feature_z_means.csv", index=False)
    dev = pd.concat(dev, ignore_index=True)
    dev.to_csv(C.TABLES / "pressure_deviation_variance.csv", index=False)
    Wt = pd.concat(Wt, ignore_index=True)
    Wt.to_csv(C.TABLES / "metric_weights.csv", index=False)
    cmp_ = pd.concat(cmp_, ignore_index=True)
    cmp_.to_csv(C.TABLES / "weights_comparison.csv", index=False)
    write_report(trust_out, chosen, dev, Wt, cmp_, infos)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 200)
    print(trust_out[["side", "measure", "set", "players", "split_half_random50", "ci95_random50", "split_half_odd_even",
                     "split_half_chronological", "season_players", "season_r_weighted", "ci95_season",
                     "passes (random-split CI > 0)"]].to_string(index=False))
    for s, p in chosen.items():
        print("TARGET", s, p.measure, p.split_half_random50, p.ci95_random50)
    print(dev.round(5).to_string(index=False))
    print(Wt.to_string(index=False))
    print(cmp_.to_string(index=False))
    print(infos)


HEADLINE_6 = """## Headline finding
**What repeats under pressure is process, not outcome.** Split-half reliability (mean of 50 random match splits, Spearman–Brown, 95% bootstrap interval), H = top 25% LI within innings:
- Repeats (batters): strike-rate delta 0.43 [0.26, 0.55], dot suppression 0.42 [0.26, 0.54], boundary elevation 0.45 [0.29, 0.58]; process composite on H 0.41 [0.23, 0.52]; run-out avoidance 0.23 [0.05, 0.37].
- Does not repeat (batters): WP value P 0.02 [−0.49, 0.31], wicket security 0.19 [−0.11, 0.36], retention R 0.06 [−0.35, 0.31], entry resilience, collapse resistance, post-wicket response, squeeze (pooled), partner effect.
- Bowlers: no H metric repeats individually — P −0.14, economy −0.12, dots 0.06, wickets 0.14, boundaries −0.11, bounce-back, post-wicket, execution drift, retention R (all intervals span 0). Only variation (entropy H − L) passes, 0.22 [0.06, 0.38], and it predicts next-season performance *negatively* (r −0.13 [−0.22, −0.03]).
- True variance of pressure-specific run value (H beyond what L predicts): batters 0.0029 runs²/ball [−0.012, 0.014] (26% of true H variance, interval spans 0); bowlers ≈ 0 (true H variance itself estimated negative). Phase 7's empirical Bayes H update is therefore small for batters and ≈ 0 for bowlers.
- Out of sample (next season, grouped by player), baseline skill alone predicts the pressure target better than weighted pressure metrics: batters r 0.16 [0.01, 0.30] vs ridge 0.07 / equal weights 0.10; bowlers 0.07 [−0.02, 0.18] vs ridge −0.12 / equal −0.13."""


def band_section():
    r = pd.read_csv(C.TABLES / "pressure_band_check.csv")
    c = pd.read_csv(C.TABLES / "pressure_band_concentration.csv")
    return ["## B. Pressure-band check (run first; decision P6-B: H25 within innings kept)\n",
            "Split-half reliability (odd vs even matches in date order within player, Spearman–Brown corrected, 200 bootstrap "
            "resamples over players for 95% intervals). Eligible: ≥100 band deliveries and ≥25 in each half. `common set` = the "
            "players eligible under H25, so differences there come from the band, not from who is included.\n",
            "### How much each version concentrates leverage\n", _md(c), "\n### Reliability by version\n", _md(r), ""]


def write_report(trust, chosen, dev, Wt, cmp_, infos):
    L = ["# Phase 6 — Repeatability, target and weights\n"] + band_section() + [HEADLINE_6, "",
         "## C. Repeatability of every candidate (trust table)\n",
         f"Primary figure: mean Spearman–Brown split-half over {N_SPLITS} random splits of each player's matches into two "
         f"halves; 95% interval from {N_BOOT} bootstrap resamples of players. Also: odd/even matches in date order, and "
         "chronological (earlier half vs later half of each player's matches). Season t → t+1: all adjacent calendar-year "
         "pairs pooled (match dates, so split seasons such as 2022/23 fall in the year of each match), weighted by the "
         "harmonic mean of deliveries; **season results reflect regular players only** (≥ half the minimum deliveries in "
         "both seasons). Minimum career deliveries in the measured set: skill/value/target 100, temperament 30, variation 60 "
         "(H), partner 100; ≥4 matches; each half needs ≥ ¼ of the minimum. Eligible players differ by metric "
         "(column `players`).\n",
         "Pass rule: the random-split 95% interval lies above 0.\n",
         _md(trust),
         "\nCharts: `outputs/charts/repeatability_batters.png`, `repeatability_bowlers.png`, `repeatability_targets.png` "
         "(earlier vs later half per metric).\n",
         "## D. Target\n"]
    for side, p in chosen.items():
        L.append(f"- **{'Batters' if side == 'bat' else 'Bowlers'}: {p.measure}** — random-split reliability "
                 f"{p.split_half_random50} {p.ci95_random50}; season-to-season r {p.get('season_r_weighted')} {p.get('ci95_season', '')}.")
    L += ["\n### Pressure-specific deviation (for the Phase 7 empirical Bayes update)\n",
          "Method of moments on career run value per delivery, players with ≥100 deliveries on both H and L (L used as "
          "the clean baseline: its sampling noise is independent of H). tau2 = observed between-player variance − mean "
          "sampling variance. The pressure-specific residual = true H variance not explained by true L performance.\n",
          _md(dev.round(6)),
          "\n### Weights\n",
          "Non-negative ridge predicting next-season target from this season's shrunk, centred, standardised metrics "
          "(only metrics whose random-split interval is above 0). Folds grouped by player; ridge penalty chosen by inner "
          "grouped CV inside each training fold, and by grouped CV on all data for the final weights. Shrinkage "
          "θ = n/(n + c)·(M − mean); c from within/between variance (mean-type metrics) or from measured reliability.\n",
          _md(Wt) if len(Wt) else "(none)",
          f"\nRidge details: {infos}\n",
          "### D3 — out-of-sample comparison (grouped by player)\n", _md(cmp_) if len(cmp_) else "(none)"]
    (C.OUT / "06_repeatability.md").write_text("\n".join(L))
