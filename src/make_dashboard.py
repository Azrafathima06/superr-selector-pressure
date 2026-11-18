"""Build the interactive dashboard: docs/index.html (single self-contained page) + docs/img/.

Reads only pipeline outputs (outputs/tables, outputs/charts, outputs/*.md, data_processed/metric_components.parquet).
Every displayed number is formatted here, half-up on its decimal value, so it matches the report.
"""
import json
import re
import shutil
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

from . import config as C
from .rating import PROC_H, NICE

DOCS = C.ROOT / "docs"
IMG = DOCS / "img"
T = C.TABLES
MINUS = "−"


def fmt(x, dp=1, sign=False):
    """Half-up decimal rounding on the value as written, unicode minus."""
    q = Decimal(1).scaleb(-dp)
    d = Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP)
    if d == 0:
        d = abs(d)
    s = f"{d:,.{dp}f}" if abs(d) >= 1000 else f"{d:.{dp}f}"
    if sign and d > 0:
        s = "+" + s
    return s.replace("-", MINUS)


def md_text(name):
    return (C.OUT / name).read_text(encoding="utf-8")


def short_name(name):
    """'Rashid Khan (Afghanistan, Gujarat Titans)' -> 'Rashid Khan (Afghanistan)'."""
    return re.sub(r"\(([^,()]+),[^()]*\)", r"(\1)", name)


def md_table(text, heading):
    """Rows of the first markdown table after a heading line, as lists of cells."""
    i = text.index(heading)
    rows = []
    for line in text[i:].splitlines()[1:]:
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not set("".join(cells)) <= set("-: "):
                rows.append(cells)
        elif rows:
            break
    return rows[1:]


def sigma_matrix(side, players):
    comp = pd.read_parquet(C.PROC / "metric_components.parquet", columns=["side", "metric", "H", "player_id", "val"])
    c = comp[(comp.side == side) & comp.metric.isin(PROC_H[side]) & comp.H]
    m = c.groupby(["player_id", "metric"]).val.mean().unstack()
    m = m.loc[m.index.isin(players)]
    return (m - m.mean()) / m.std()


def parse_strongest(s):
    out = []
    for part in str(s).split(";"):
        mm = re.match(r"\s*(.+?)\s+([+\-]\d+\.\d+)σ", part)
        if mm:
            out.append({"metric": mm.group(1), "sigma": fmt(float(mm.group(2)), 2, sign=True),
                        "pos": float(mm.group(2)) > 0})
    return out


def build_side(side):
    long = {"bat": "batters", "bowl": "bowlers"}[side]
    r = pd.read_csv(T / f"ratings_all_{side}.csv")
    e = r[r.eligible].sort_values("rank_score", ascending=False).reset_index(drop=True)
    e["rank"] = e.index + 1
    roles = pd.read_csv(T / f"roles_{long}.csv").set_index("player_id").role
    locked = pd.read_csv(T / f"locked_top10_{long}.csv")
    z = sigma_matrix(side, e.player_id.values)

    top = []
    for _, t in locked.head(10).iterrows():
        top.append({"rank": int(t["rank"]), "name": short_name(t.player), "team": t.latest_team, "role": t.role,
                    "rating": float(t.rating_runs_per_100_H_balls), "se": float(t.se_per_100),
                    "rating_s": fmt(t.rating_runs_per_100_H_balls), "se_s": fmt(t.se_per_100),
                    "nH": int(t.n_H), "seasons": t.seasons_played, "strongest": parse_strongest(t.strongest_process_on_H)})
    assert [p["name"] for p in top] == [short_name(n) for n in e.player.head(10)], "locked list drifted"

    players = []
    for _, p in e.iterrows():
        proc = []
        for k in PROC_H[side]:
            v = z.loc[p.player_id, k] if (p.player_id in z.index and k in z.columns) else float("nan")
            if pd.notna(v):
                proc.append({"metric": NICE[k], "z": round(float(v), 4), "z_s": fmt(v, 2, sign=True)})
        n_seasons = int(p.seasons)
        players.append({
            "side": side, "rank": int(p["rank"]), "name": short_name(p.player),
            "rating_s": fmt(100 * p.rating), "se_s": fmt(100 * p.se), "nH": int(p.n_H),
            "seasons": f"{int(p.first_season)}–{int(p.last_season)} ({n_seasons} season{'s' if n_seasons != 1 else ''})",
            "role": roles.get(p.player_id, ""), "proc": proc})
    return top, players


def build():
    IMG.mkdir(parents=True, exist_ok=True)
    shutil.copy2(C.CHARTS / "wp_curve_45513.png", IMG / "wp_curve_45513.png")

    top_bat, all_bat = build_side("bat")
    top_bowl, all_bowl = build_side("bowl")

    audit = md_text("01_data_audit.md")
    mm = re.search(r"Kept matches: ([\d,]+); kept balls \(innings 1–2\): ([\d,]+)", audit)
    matches, deliveries = mm.group(1), mm.group(2)
    lev = md_text("03_leverage.md")
    li_cut = fmt(float(re.search(r"H25 global cut-off: LI ≥ (\d+\.\d+)", lev).group(1)), 2)
    sr = md_text("06b_signal_recovery.md")
    pm = re.search(r"p = ([\d.]+)e-(\d+) batters, ([\d.]+)e-(\d+) bowlers", sr)
    pvals = {"bat": [pm.group(1), str(int(pm.group(2)))], "bowl": [pm.group(3), str(int(pm.group(4)))]}
    n_boot = int(re.search(r"^N_BOOT_SE\s*=\s*(\d+)", (C.ROOT / "src" / "rating.py").read_text(), re.M).group(1))

    variants = pd.read_csv(T / "rating_variants.csv").set_index("variant")
    thresholds = {s: int(variants.loc[f"{s}: primary", "min_H"]) for s in ("bat", "bowl")}

    rec = pd.read_csv(T / "reconciliation.csv")
    recon = [int((rec.recon_ok == 1).sum()), int(rec.recon_ok.notna().sum())]
    rain_unrec = int(rec.revised_unrecorded.sum())

    rep = pd.read_csv(T / "metric_repeatability.csv")
    rel_rows = [("boundary_elevation", "Boundary elevation"), ("sr_delta", "Strike-rate delta"),
                ("dot_suppression", "Dot suppression"), ("TARGET process composite on H", "Combined process score")]
    reliability = []
    for key, label in rel_rows:
        x = rep[(rep.side == "bat") & (rep.measure == key)].iloc[0]
        lo, hi = [float(v) for v in x.ci95_random50.strip("[]").split(",")]
        reliability.append({"label": label, "r": fmt(x.split_half_random50, 2), "ci": f"{fmt(lo, 2)} to {fmt(hi, 2)}"})

    r7 = md_text("07_rating.md")
    sens = {}
    for s in ("bat", "bowl"):
        part = r7[:r7.index("## Bowlers")] if s == "bat" else r7[r7.index("## Bowlers"):]
        rows = md_table(part, "### Sensitivity — top-10 overlap with the primary list")
        sens[s] = {row[0]: row[1] for row in rows}
    robustness = [
        {"label": "pressure set to the top 20% or top 33%",
         "bat": f"{sens['bat']['H20']} and {sens['bat']['H33']}", "bowl": f"{sens['bowl']['H20']} and {sens['bowl']['H33']}"},
        {"label": "eligibility threshold up or down by 25%",
         "bat": f"{sens['bat']['threshold +25%']} and {sens['bat']['threshold −25%']}",
         "bowl": f"{sens['bowl']['threshold +25%']} and {sens['bowl']['threshold −25%']}"},
        {"label": "a softer delivery condition", "bat": sens["bat"]["noise-aware delivery condition"],
         "bowl": sens["bowl"]["noise-aware delivery condition"]},
    ]

    # out-of-time validation
    v = pd.read_csv(T / "out_of_time_validation.csv")
    val = {}
    for s in ("bat", "bowl"):
        rows = []
        for _, x in v[v.side == s].iterrows():
            lo, hi = [float(t) for t in x.pearson_ci95.strip("[]").split(",")]
            ours = x.predictor.startswith("Our rating")
            label = "This rating" if ours else re.sub(r"^\([a-z]\)\s*", "", x.predictor)
            label = label[0].upper() + label[1:]
            rows.append({"label": label, "ours": ours, "r": float(x.pearson_r), "lo": lo, "hi": hi,
                         "r_s": fmt(x.pearson_r, 2), "ci_s": f"{fmt(lo, 2)} to {fmt(hi, 2)}",
                         "rho_s": fmt(x.spearman, 2)})
        val[s] = {"n": int(v[v.side == s].players.iloc[0]), "rows": rows}
    std_delta = v[(v.side == "bowl") & v.predictor.str.contains("pressure economy delta")].iloc[0]
    ours_bowl = v[(v.side == "bowl") & v.predictor.str.startswith("Our rating")].iloc[0]
    death_econ = v[(v.side == "bowl") & v.predictor.str.contains("death-over economy")].iloc[0]
    olo, ohi = [float(t) for t in ours_bowl.pearson_ci95.strip("[]").split(",")]
    finding = {"delta": fmt(std_delta.pearson_r, 2), "ours": fmt(ours_bowl.pearson_r, 2),
               "ours_ci": f"{fmt(olo, 2)} to {fmt(ohi, 2)}", "death": fmt(death_econ.pearson_r, 2)}
    t20 = pd.read_csv(T / "out_of_time_top20_share.csv").set_index("side")
    top20 = {s: {"share": fmt(100 * t20.loc[s, "share"], 1 if s == "bat" else 0) + "%",
                 "rand": fmt(100 * t20.loc[s, "share_for_random_eligible_player"], 0) + "%"} for s in ("bat", "bowl")}

    # value gap
    gap = {}
    gap_col = "gap (conventional − pressure rank)"
    gb = pd.read_csv(T / "pressure_value_gap_bat.csv")
    gw = pd.read_csv(T / "pressure_value_gap_bowl.csv")
    for s, g in (("bat", gb), ("bowl", gw)):
        gap[s] = [{"name": short_name(x.player), "pr": int(x.pressure_rank), "cr": int(x.conventional_rank),
                   "gap": int(x[gap_col]), "side": s} for _, x in g.iterrows()]
    gap_notes = {"bat_avg_top10": int((gb.head(10).avg_rank <= 10).sum()),
                 "bowl_wkt_top10": int((gw.head(10).wkt_rank <= 10).sum())}
    gw_i = gw.set_index("player")
    rec2 = sorted(int(gw_i.loc[n, gap_col]) for n in ("Eshan Malinga", "Matheesha Pathirana", "Nathan Ellis"))

    # RR
    nice_scen = {"death overs, 2nd innings": "Death overs of a chase",
                 "entry under pressure": "New batter arriving under pressure",
                 "post-wicket (12 deliveries after a wicket)": "The 12 balls after a wicket",
                 "powerplay vs set openers": "Powerplay against set openers",
                 "dot squeeze (after 3+ dots)": "Dot squeeze, after 3+ dots"}
    sc = pd.read_csv(T / "rr_fit_scenarios.csv")
    wins_col = "RR_total_WP_vs_league (wins)"
    rr = [{"label": nice_scen.get(x.scenario, x.scenario), "role": x.role.replace("RR ", ""),
           "balls": f"{int(x.RR_H_deliveries):,}", "matches": int(x.RR_matches),
           "wins": float(x[wins_col]), "wins_s": fmt(x[wins_col], 2, sign=True)}
          for _, x in sc.sort_values(wins_col).iterrows()]
    worst = sc.sort_values(wins_col).iloc[0]
    rr_head = {"label": nice_scen.get(worst.scenario, worst.scenario).lower(), "wins": fmt(-worst[wins_col], 2),
               "balls": int(worst.RR_H_deliveries)}
    bp = pd.read_csv(T / "rr_fit_best_players.csv")
    bp = bp[bp.final_rank <= 10]
    best = []
    for scen, g in bp.groupby("RR's costly scenario", sort=False):
        base = re.sub(r"\s*\(RR (bowling|batting)\)$", "", scen)
        best.append({"label": nice_scen.get(base, base),
                     "players": [{"name": short_name(x.player), "v": fmt(x.run_value_per_100_balls_in_scenario_on_H, 0, sign=True),
                                  "rank": int(x.final_rank)} for _, x in g.iterrows()]})

    data = {
        "hero": {"matches": matches, "deliveries": deliveries, "bat": str(len(all_bat)), "bowl": str(len(all_bowl))},
        "top": {"bat": top_bat, "bowl": top_bowl},
        "players": all_bat + all_bowl,
        "counts": {"bat": len(all_bat), "bowl": len(all_bowl)},
        "val": val, "finding": finding, "top20": top20,
        "gap": gap, "gapNotes": gap_notes,
        "rr": rr, "rrHead": rr_head, "best": best,
        "rec2": [rec2[0], rec2[-1]],
        "method": {"liCut": li_cut, "p": pvals, "nBoot": f"{n_boot:,}", "thr": thresholds,
                   "recon": recon, "rainUnrec": rain_unrec, "rel": reliability, "rob": robustness},
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    print(f"dashboard: wrote {DOCS / 'index.html'} ({len(html) / 1024:.0f} KB)")


def run():
    build()


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Proven Under Pressure</title>
<meta name="description" content="Pressure reliability ratings for IPL and T20I batters and bowlers, built for the Rajasthan Royals SupeRR Selector.">
<link rel="icon" href="rr_logo.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@1,400;1,600&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700;9..40,800&display=swap" rel="stylesheet">
<style>
:root{
  --navy:#141B3A; --navy-2:#1d2650; --pink:#E4177E; --pink-soft:#fde7f2; --blue:#254AA5; --blue-soft:#e6ecf8;
  --line:#E3E5EC; --bg:#F6F7FA; --card:#FFFFFF; --ink:#141B3A; --ink-2:#4a5068; --ink-3:#7b8199;
  --radius:18px; --shadow:0 1px 2px rgba(20,27,58,.05),0 8px 24px rgba(20,27,58,.06);
  --shadow-h:0 2px 4px rgba(20,27,58,.06),0 14px 34px rgba(20,27,58,.10);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"DM Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:0 24px}
section{padding:72px 0}
h2{font-size:clamp(28px,4vw,40px);line-height:1.15;margin:0 0 12px;letter-spacing:-.02em;font-weight:800}
h3{font-size:18px;margin:0 0 4px;font-weight:700;letter-spacing:-.01em}
p{margin:0 0 14px}
.eyebrow{display:inline-block;font-size:12px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--pink);margin-bottom:10px}
.lede{font-size:18px;color:var(--ink-2);max-width:760px;margin-bottom:32px}
.card{background:var(--card);border-radius:var(--radius);box-shadow:var(--shadow);padding:28px;border:1px solid var(--line);
  transition:box-shadow .25s ease,transform .25s ease}
.card:hover{box-shadow:var(--shadow-h)}
.chart-title{font-weight:700;font-size:17px;margin:0 0 4px}
.chart-sub{color:var(--ink-3);font-size:14px;margin:0 0 20px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:24px}
.quote{font-family:"Crimson Text",Georgia,serif;font-style:italic;font-size:clamp(22px,2.6vw,28px);line-height:1.35}
nav.top{position:sticky;top:0;z-index:20;background:rgba(20,27,58,.92);backdrop-filter:saturate(1.4) blur(10px);-webkit-backdrop-filter:blur(10px)}
nav.top .wrap{display:flex;align-items:center;gap:20px;height:56px;overflow-x:auto;scrollbar-width:none}
nav.top .wrap::-webkit-scrollbar{display:none}
nav.top img{height:30px;width:auto;flex:none}
nav.top a{color:#c9cde0;text-decoration:none;font-size:14px;font-weight:500;white-space:nowrap;transition:color .2s}
nav.top a:hover,nav.top a.on{color:#fff}
nav.top a.on{box-shadow:inset 0 -2px 0 var(--pink)}
/* hero */
.hero{background:radial-gradient(1200px 500px at 85% -10%,rgba(228,23,126,.28),transparent 60%),
  radial-gradient(900px 500px at -10% 110%,rgba(37,74,165,.45),transparent 60%),var(--navy);color:#fff;padding:72px 0 64px}
.hero-top{display:flex;align-items:center;gap:28px;flex-wrap:wrap}
.hero-logo{width:112px;height:auto;filter:drop-shadow(0 6px 20px rgba(0,0,0,.35))}
.hero h1{font-size:clamp(44px,7.5vw,84px);line-height:1;margin:0;letter-spacing:-.035em;font-weight:800}
.hero .quote{color:#e9ebf5;max-width:720px;margin:28px 0 18px}
.halla{font-weight:800;letter-spacing:.2em;color:var(--pink);font-size:18px;margin:0 0 6px}
.byline{color:#aab0cc;font-size:15px}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:44px}
.tile{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:16px;padding:20px 22px;transition:background .25s,transform .25s}
.tile:hover{background:rgba(255,255,255,.1);transform:translateY(-2px)}
.tile b{display:block;font-size:clamp(26px,3.4vw,38px);line-height:1.1;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.tile span{color:#aab0cc;font-size:14px}
/* tabs */
.tabs{display:inline-flex;background:#eceef4;border-radius:999px;padding:4px;margin-bottom:22px;gap:2px;flex-wrap:wrap}
.tabs button{border:0;background:transparent;font:inherit;font-size:14px;font-weight:600;color:var(--ink-2);padding:8px 18px;border-radius:999px;cursor:pointer;transition:background .2s,color .2s}
.tabs button:hover{color:var(--ink)}
.tabs button.on{background:var(--pink);color:#fff;box-shadow:0 2px 8px rgba(228,23,126,.3)}
.tabs button:focus-visible,.search input:focus-visible,summary:focus-visible{outline:2px solid var(--blue);outline-offset:2px}
/* bar charts */
.hbar{position:relative}
.hrow{display:grid;grid-template-columns:var(--lab,220px) 1fr var(--val,96px);align-items:center;gap:14px;padding:7px 8px;border-radius:10px;transition:background .15s;cursor:default}
.hrow:hover,.hrow.hl{background:#f3f4f8}
.hlab{display:flex;align-items:baseline;gap:10px;min-width:0}
.hlab .rk{color:var(--pink);font-weight:800;font-variant-numeric:tabular-nums;width:22px;text-align:right;flex:none}
.hlab .nm{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hlab .sm{color:var(--ink-3);font-size:13px;white-space:nowrap}
.track{position:relative;height:22px}
.bar{position:absolute;top:4px;height:14px;border-radius:0 4px 4px 0;background:var(--blue);transition:width .6s cubic-bezier(.2,.8,.2,1),left .6s cubic-bezier(.2,.8,.2,1)}
.bar.neg{border-radius:4px 0 0 4px}
.bar.pink{background:var(--pink)}
.whisk{position:absolute;top:10px;height:2px;background:var(--navy);opacity:.75}
.whisk::before,.whisk::after{content:"";position:absolute;top:-4px;width:2px;height:10px;background:var(--navy)}
.whisk::before{left:0}.whisk::after{right:0}
.hval{text-align:right;font-variant-numeric:tabular-nums;font-weight:600;font-size:14px;white-space:nowrap}
.hval small{color:var(--ink-3);font-weight:500}
.axis{display:grid;grid-template-columns:var(--lab,220px) 1fr var(--val,96px);gap:14px;padding:0 8px;margin-top:6px}
.axis .ticks{position:relative;height:20px;border-top:1px solid var(--line)}
.axis .ticks span{position:absolute;top:4px;transform:translateX(-50%);font-size:12px;color:var(--ink-3);font-variant-numeric:tabular-nums}
.zero{position:absolute;top:0;bottom:0;width:1px;background:#c9ccd8}
.gridl{position:absolute;top:0;bottom:0;width:1px;background:var(--line)}
.note{color:var(--ink-3);font-size:13px;margin-top:14px}
/* tooltip */
#tip{position:fixed;z-index:50;pointer-events:none;background:var(--navy);color:#fff;border-radius:12px;padding:12px 14px;font-size:13px;line-height:1.5;
  box-shadow:0 10px 30px rgba(20,27,58,.3);max-width:290px;opacity:0;transform:translateY(4px);transition:opacity .12s,transform .12s}
#tip.on{opacity:1;transform:none}
#tip b{font-size:14px}
#tip .k{color:#aab0cc}
#tip .pk{color:#ff7cbc;font-weight:700}
/* explorer */
.search{position:relative;max-width:520px}
.search input{width:100%;font:inherit;font-size:16px;padding:14px 18px 14px 46px;border:1px solid var(--line);border-radius:14px;background:#fff;color:var(--ink);box-shadow:var(--shadow);transition:border-color .2s}
.search input:focus{border-color:var(--pink);outline:none}
.search svg{position:absolute;left:16px;top:50%;transform:translateY(-50%);color:var(--ink-3)}
.results{position:absolute;left:0;right:0;top:calc(100% + 6px);background:#fff;border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow-h);max-height:320px;overflow:auto;z-index:10;display:none;padding:6px}
.results.on{display:block}
.results button{display:flex;width:100%;justify-content:space-between;align-items:center;gap:12px;border:0;background:none;font:inherit;text-align:left;padding:10px 12px;border-radius:10px;cursor:pointer;color:var(--ink)}
.results button:hover,.results button.act{background:var(--pink-soft)}
.results .tag{font-size:12px;color:var(--ink-3);white-space:nowrap}
.pcard{margin-top:24px;display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:28px}
.pname{font-size:clamp(24px,3vw,32px);font-weight:800;letter-spacing:-.02em;line-height:1.15;margin:4px 0 6px}
.prank{display:inline-flex;align-items:center;gap:8px;background:var(--pink-soft);color:var(--pink);font-weight:700;border-radius:999px;padding:4px 12px;font-size:14px}
.facts{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-top:20px}
.fact{background:var(--bg);border-radius:12px;padding:12px 14px}
.fact span{display:block;font-size:12px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.08em;font-weight:600}
.fact b{font-size:18px;font-variant-numeric:tabular-nums}
.proc .hrow{--lab:150px;--val:62px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px}
.chip{border:1px solid var(--line);background:#fff;border-radius:999px;padding:6px 12px;font:inherit;font-size:13px;cursor:pointer;color:var(--ink-2);transition:all .2s}
.chip:hover{border-color:var(--pink);color:var(--pink)}
/* finding */
.big{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-bottom:24px}
.bignum{font-size:clamp(56px,9vw,96px);font-weight:800;letter-spacing:-.04em;line-height:1;font-variant-numeric:tabular-nums}
.bignum.grey{color:#9aa0b6}
.bignum.pink{color:var(--pink)}
.big .card p{color:var(--ink-2);margin:12px 0 0}
.dotrow{display:grid;grid-template-columns:var(--lab,250px) 1fr var(--val,70px);gap:14px;align-items:center;padding:8px;border-radius:10px;cursor:default;transition:background .15s}
.dotrow:hover{background:#f3f4f8}
.dotrow .lb{font-size:14px;font-weight:500}
.dotrow.ours .lb{color:var(--pink);font-weight:700}
.ci{position:absolute;top:10px;height:2px;background:#9aa0b6;border-radius:2px}
.dot{position:absolute;top:5px;width:12px;height:12px;border-radius:50%;background:var(--blue);transform:translateX(-50%);box-shadow:0 0 0 2px #fff}
.ours .ci{background:var(--pink);height:3px;top:9.5px}
.ours .dot{background:var(--pink);width:14px;height:14px;top:4px}
.facet{margin-bottom:26px}
.facet h4{margin:0 0 6px;font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3)}
/* pressure */
.formula{background:var(--navy);color:#fff;border-radius:14px;padding:18px 22px;font-size:clamp(15px,1.8vw,18px);overflow-x:auto;margin:16px 0;
  font-family:"DM Sans",sans-serif;font-variant-numeric:tabular-nums;line-height:1.9}
.formula i{font-style:italic}
.formula .op{color:#ff7cbc;padding:0 4px}
.formula sub,.formula sup{font-size:.7em}
.figure img{width:100%;height:auto;border-radius:12px;display:block;border:1px solid var(--line)}
.figure figcaption{color:var(--ink-3);font-size:13px;margin-top:10px}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:20px}
.kpi{background:var(--bg);border-radius:12px;padding:14px}
.kpi b{display:block;font-size:24px;color:var(--blue);letter-spacing:-.02em}
.kpi span{font-size:13px;color:var(--ink-2)}
/* RR */
.rrsec{background:var(--navy);color:#fff}
.rrsec .lede{color:#c9cde0}
.rrsec .card{color:var(--ink)}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;color:var(--ink-2);margin:-8px 0 16px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.best{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:24px}
.best .card{padding:22px}
.best ol{list-style:none;padding:0;margin:12px 0 0}
.best li{display:flex;justify-content:space-between;gap:10px;padding:7px 0;border-top:1px solid var(--line);font-size:14px}
.best li b{color:var(--pink);font-variant-numeric:tabular-nums}
.recs{counter-reset:r;display:grid;gap:16px;margin-top:28px}
.rec{display:grid;grid-template-columns:56px 1fr;gap:16px;align-items:start;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:16px;padding:22px;transition:background .25s}
.rec:hover{background:rgba(255,255,255,.1)}
.rec .n{width:44px;height:44px;border-radius:50%;background:var(--pink);display:grid;place-items:center;font-weight:800;font-size:18px}
.rec h3{color:#fff}
.rec p{color:#c9cde0;margin:4px 0 0}
#rrChart .hrow,#rrChart .axis{--lab:290px}
#topChart .hrow,#topChart .axis{--lab:230px}
#gapChart .hrow,#gapChart .axis{--lab:290px}
.rr-note{margin-top:28px;color:#e9ebf5}
/* method */
details{background:#fff;border:1px solid var(--line);border-radius:16px;margin-bottom:12px;box-shadow:var(--shadow);transition:box-shadow .25s}
details:hover{box-shadow:var(--shadow-h)}
summary{list-style:none;cursor:pointer;padding:20px 24px;font-weight:700;font-size:18px;display:flex;justify-content:space-between;align-items:center;gap:12px}
summary::-webkit-details-marker{display:none}
summary .pl{width:28px;height:28px;border-radius:50%;background:var(--bg);display:grid;place-items:center;flex:none;transition:transform .25s,background .25s,color .25s;color:var(--ink-2)}
details[open] summary .pl{transform:rotate(45deg);background:var(--pink);color:#fff}
.dbody{padding:0 24px 22px;color:var(--ink-2)}
.dbody .formula{color:#fff}
table{width:100%;border-collapse:collapse;font-size:14px;margin:12px 0}
th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--ink-3);font-weight:700}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.tscroll{overflow-x:auto}
footer{background:var(--navy);color:#c9cde0;padding:56px 0}
footer .wrap{display:flex;justify-content:space-between;align-items:center;gap:24px;flex-wrap:wrap}
footer a{color:#fff;font-weight:600;text-decoration:none;border-bottom:2px solid var(--pink);padding-bottom:2px}
footer a:hover{color:#ff7cbc}
footer .halla{font-size:22px;margin:0}
.reveal{opacity:0;transform:translateY(16px);transition:opacity .7s ease,transform .7s ease}
.reveal.in{opacity:1;transform:none}
@media (prefers-reduced-motion:reduce){*{transition:none!important;scroll-behavior:auto!important}.reveal{opacity:1;transform:none}}
@media (max-width:900px){
  .grid2,.pcard,.best{grid-template-columns:1fr}
  .tiles{grid-template-columns:repeat(2,1fr)}
}
@media (max-width:640px){
  .wrap{padding:0 16px}
  section{padding:52px 0}
  .card{padding:20px 16px}
  .hrow,.axis{--lab:130px;--val:74px;gap:8px;padding-left:4px;padding-right:4px}
  .hlab .sm{display:none}
  .hlab{gap:6px}.hlab .rk{width:18px}
  .hlab .nm{font-size:14px}
  .dotrow{--lab:118px;--val:44px;gap:8px}
  .dotrow .lb{font-size:12.5px;line-height:1.3}
  .big{grid-template-columns:1fr}
  .kpis{grid-template-columns:1fr}
  .facts{grid-template-columns:1fr 1fr}
  .proc .hrow{--lab:118px;--val:54px}
  .stack .hrow{grid-template-columns:1fr var(--val);row-gap:2px}
  .stack .hrow .hlab{grid-column:1 / -1}
  .stack .axis{grid-template-columns:1fr var(--val)}
  .stack .axis > div:first-child{display:none}
  .stack .hlab .sm{display:inline}
  .hero-logo{width:84px}
  .rec{grid-template-columns:1fr}
  summary{padding:18px 16px;font-size:16px}.dbody{padding:0 16px 18px}
}
</style>
</head>
<body>
<nav class="top" aria-label="Sections"><div class="wrap">
  <img src="rr_logo.png" alt="Rajasthan Royals">
  <a href="#top10">Top 10</a><a href="#explorer">Explorer</a><a href="#finding">The Finding</a><a href="#pressure">Pressure</a>
  <a href="#hidden">Hidden value</a><a href="#royals">For the Royals</a><a href="#method">Method</a>
</div></nav>

<header class="hero"><div class="wrap">
  <div class="hero-top">
    <img class="hero-logo" src="rr_logo.png" alt="Rajasthan Royals logo">
    <h1>Proven Under<br>Pressure</h1>
  </div>
  <p class="quote">One ball can turn a season. This is about the players who turn it your way.</p>
  <p class="halla">HALLA BOL.</p>
  <p class="byline">By Azra Fathima Q, B.Tech Computer Science, VIT Chennai</p>
  <div class="tiles" id="tiles"></div>
</div></header>

<main>
<section id="top10"><div class="wrap">
  <span class="eyebrow">01 &middot; The lists</span>
  <h2>The Top 10</h2>
  <p class="lede">Every player here pairs elite core skill with a record of beating expectation on high-pressure balls. Ratings are runs per 100 high-pressure balls above expectation.</p>
  <div class="card">
    <div class="tabs" role="tablist" id="topTabs">
      <button role="tab" data-k="bat" class="on" aria-selected="true">Batters</button>
      <button role="tab" data-k="bowl" aria-selected="false">Bowlers</button>
    </div>
    <p class="chart-title" id="topTitle"></p>
    <p class="chart-sub">Rating with &plusmn; one standard error. Ordered by rank score (rating minus one standard error). Hover a row for role and strengths.</p>
    <div class="hbar stack" id="topChart"></div>
  </div>
</div></section>

<section id="explorer" style="padding-top:0"><div class="wrap">
  <span class="eyebrow">02 &middot; Explorer</span>
  <h2>Find any ranked player</h2>
  <p class="lede" id="explorerLede"></p>
  <div class="search">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
    <input id="q" type="search" placeholder="Search a batter or bowler" autocomplete="off" aria-label="Search players" role="combobox" aria-expanded="false" aria-controls="res">
    <div class="results" id="res" role="listbox"></div>
  </div>
  <div class="chips" id="chips"></div>
  <div class="card pcard" id="pcard"></div>
</div></section>

<section id="finding" style="background:#fff;border-top:1px solid var(--line);border-bottom:1px solid var(--line)"><div class="wrap">
  <span class="eyebrow">03 &middot; The Finding</span>
  <h2>The usual way of measuring pressure doesn't predict who'll deliver next season. This rating does.</h2>
  <p class="lede">Every rating was rebuilt from 2022&ndash;24 data alone, then tested on how well it predicted each player's 2025&ndash;26 performance on high-pressure balls.</p>
  <div class="big">
    <div class="card"><div class="bignum grey" id="bnDelta"></div><p><b>"Pressure economy minus normal economy"</b>, the standard pressure metric</p></div>
    <div class="card"><div class="bignum pink" id="bnOurs"></div><p><b>This rating</b>, same test <span id="bnCi"></span></p></div>
  </div>
  <div class="card">
    <p class="chart-title">Among bowlers, this rating predicts next season's pressure performance; the standard delta does not</p>
    <p class="chart-sub">Correlation (Pearson r) between each 2022&ndash;24 measure and 2025&ndash;26 run value on high-pressure balls, with 95% intervals.</p>
    <div id="valChart"></div>
  </div>
  <p class="lede" style="margin:28px 0 0" id="findingText"></p>
</div></section>

<section id="pressure"><div class="wrap">
  <span class="eyebrow">04 &middot; Defining pressure</span>
  <h2>How pressure is measured</h2>
  <p class="lede">A ball is under pressure when it can swing the result. So instead of guessing which situations feel tense, the analysis measures that swing for every single delivery.</p>
  <div class="grid2">
    <div class="card">
      <h3>Win probability</h3>
      <p>A model of the batting side's chance of winning from any point in a match. In a chase it reads runs needed, balls left, wickets in hand, the required rate against the venue's usual scoring, and a ball-by-ball simulation of how the rest of the chase could unfold. In the first innings it compares the score with par for that stage, using only matches played before.</p>
      <h3 style="margin-top:22px">Leverage Index</h3>
      <p>For each ball, a second model gives the chance of every result: dot, one, two, three, four, six, wicket, wide or no-ball. Leverage is how far the win probability is expected to move on that one ball.</p>
      <div class="formula">LI<sub>raw</sub> = &Sigma;<sub><i>o</i></sub> <i>p</i>(<i>o</i> | state) <span class="op">&times;</span> | WP(after <i>o</i>) &minus; expected WP after the ball |<br>LI = LI<sub>raw</sub> <span class="op">&divide;</span> average LI<sub>raw</sub></div>
      <p>An average ball scores 1. The top 25% of balls by leverage within each innings form the pressure set, <b>H</b>; the rest form the comparison set, <b>L</b>.</p>
      <div class="kpis" id="pkpis"></div>
    </div>
    <figure class="card figure" style="margin:0">
      <p class="chart-title">The steep swings are the high-pressure balls</p>
      <p class="chart-sub">Pakistan's win probability through the 2025 Asia Cup final against India. Each dot is a wicket.</p>
      <img src="img/wp_curve_45513.png" alt="Win probability curve of the 2025 Asia Cup final, Pakistan against India" decoding="async">
    </figure>
  </div>
</div></section>

<section id="hidden" style="padding-top:0"><div class="wrap">
  <span class="eyebrow">05 &middot; Hidden value</span>
  <h2>The players the usual numbers miss</h2>
  <p class="lede">Each player's pressure rank set against his rank on conventional stats (average and strike rate, or economy and wickets). The Pressure Value Gap is how many places higher he ranks on pressure.</p>
  <div class="card">
    <div class="tabs" role="tablist" id="gapTabs">
      <button role="tab" data-k="top" class="on" aria-selected="true">Top 10 picks</button>
      <button role="tab" data-k="bat" aria-selected="false">All ranked batters</button>
      <button role="tab" data-k="bowl" aria-selected="false">All ranked bowlers</button>
    </div>
    <p class="chart-title" id="gapTitle"></p>
    <p class="chart-sub">Places gained: conventional rank minus pressure rank. Only players whose pressure rank beats their conventional rank are shown.</p>
    <div class="hbar stack" id="gapChart"></div>
    <p class="note" id="gapNote"></p>
  </div>
</div></section>

<section id="royals" class="rrsec"><div class="wrap">
  <span class="eyebrow">06 &middot; For the Royals</span>
  <h2 id="rrH"></h2>
  <p class="lede">Wins gained or lost against a league-average side in each pressure situation since 2022.</p>
  <div class="card">
    <p class="chart-title" id="rrTitle"></p>
    <p class="chart-sub">Total win probability against league average, summed over RR's high-pressure balls in each situation.</p>
    <div class="legend"><span><i style="background:var(--pink)"></i>below league average</span><span><i style="background:var(--blue)"></i>above league average</span></div>
    <div class="hbar stack" id="rrChart"></div>
  </div>
  <h3 style="margin-top:44px;color:#fff;font-size:22px">The ranked bowlers with the strongest records in exactly those moments</h3>
  <p style="color:#c9cde0;margin:0">Run value per 100 high-pressure balls in that situation, top-10 ranked bowlers.</p>
  <div class="best" id="best"></div>
  <p class="rr-note">And RR already holds the top-ranked batter. Vaibhav Sooryavanshi is in the 2026 squad, and with just two seasons behind him, his pressure record is only getting started.</p>
  <h3 style="margin-top:44px;color:#fff;font-size:22px">For the 2027 auction</h3>
  <div class="recs" id="recs"></div>
  <p class="quote" style="margin-top:44px;color:#fff">The Royals have always backed the ones others overlooked.<br>The data says: back the ones who don't blink.</p>
</div></section>

<section id="method"><div class="wrap">
  <span class="eyebrow">07 &middot; Method</span>
  <h2>How it's built</h2>
  <p class="lede">The same definitions and formulas as the report. Open any panel.</p>
  <div id="methodPanels"></div>
</div></section>
</main>

<footer><div class="wrap">
  <div>
    <p style="margin:0 0 6px;color:#fff;font-weight:700;font-size:18px">Proven Under Pressure</p>
    <p style="margin:0">Code, outputs and the full write-up: <a href="https://github.com/Azrafathima06/superr-selector-pressure" target="_blank" rel="noopener">github.com/Azrafathima06/superr-selector-pressure</a></p>
  </div>
  <p class="halla">Halla Bol.</p>
</div></footer>

<div id="tip" role="tooltip"></div>

<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
"use strict";
var D = JSON.parse(document.getElementById("data").textContent);
var $ = function(id){return document.getElementById(id);};
var esc = function(s){return String(s).replace(/[&<>"]/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c];});};
var MINUS = "−";

/* tooltip */
var tip = $("tip");
function showTip(html, e){ tip.innerHTML = html; tip.classList.add("on"); moveTip(e); }
function moveTip(e){
  var x = (e.touches ? e.touches[0].clientX : e.clientX), y = (e.touches ? e.touches[0].clientY : e.clientY);
  var w = tip.offsetWidth, h = tip.offsetHeight, vw = window.innerWidth;
  var left = Math.min(Math.max(8, x + 14), vw - w - 8), top = y + 16;
  if (top + h > window.innerHeight - 8) top = y - h - 14;
  tip.style.left = left + "px"; tip.style.top = top + "px";
}
function hideTip(){ tip.classList.remove("on"); }
function bindTip(el, fn){
  el.addEventListener("mouseenter", function(e){ showTip(fn(), e); });
  el.addEventListener("mousemove", moveTip);
  el.addEventListener("mouseleave", hideTip);
  el.addEventListener("touchstart", function(e){ showTip(fn(), e); }, {passive:true});
}
document.addEventListener("touchstart", function(e){ if(!e.target.closest(".hrow,.dotrow")) hideTip(); }, {passive:true});
window.addEventListener("scroll", hideTip, {passive:true});

function niceMax(v){ var steps=[5,10,20,25,50,100]; for(var i=0;i<steps.length;i++){ var s=steps[i]; if(v/s<=6) return Math.ceil(v/s)*s; } return Math.ceil(v/100)*100; }
function tickStep(max){ return max<=30?5:max<=60?10:max<=120?20:50; }
function axisHtml(lo, hi, step, fmt){
  var t = "";
  for (var v = Math.ceil(lo/step)*step; v <= hi + 1e-9; v += step){
    var p = (v - lo)/(hi - lo)*100;
    t += '<span style="left:'+p+'%">'+(fmt?fmt(v):String(v).replace("-",MINUS))+'</span>';
  }
  return '<div class="axis"><div></div><div class="ticks">'+t+'</div><div></div></div>';
}
function gridHtml(lo, hi, step){
  var g = "";
  for (var v = Math.ceil(lo/step)*step; v <= hi + 1e-9; v += step){
    g += '<div class="'+(Math.abs(v)<1e-9?"zero":"gridl")+'" style="left:'+((v-lo)/(hi-lo)*100)+'%"></div>';
  }
  return g;
}

/* hero */
$("tiles").innerHTML = [
  [D.hero.matches,"matches"],[D.hero.deliveries,"deliveries"],[D.hero.bat,"batters ranked"],[D.hero.bowl,"bowlers ranked"]
].map(function(t){return '<div class="tile"><b>'+t[0]+'</b><span>'+t[1]+'</span></div>';}).join("");

/* tabs helper */
function tabs(id, cb){
  var box = $(id);
  box.addEventListener("click", function(e){
    var b = e.target.closest("button"); if(!b) return;
    box.querySelectorAll("button").forEach(function(x){ x.classList.toggle("on", x===b); x.setAttribute("aria-selected", x===b); });
    cb(b.getAttribute("data-k"));
  });
}

/* 1. top 10 */
var TOP_TITLES = {
  bat: "Batter #1 stands clear statistically; the ratings below him sit close together",
  bowl: "Bowlers #1 and #2 stand clear statistically; the ratings below them sit close together"
};
function renderTop(k){
  var rows = D.top[k];
  var max = niceMax(Math.max.apply(null, rows.map(function(r){return r.rating + r.se;})));
  var step = tickStep(max);
  $("topTitle").textContent = TOP_TITLES[k];
  var h = "";
  rows.forEach(function(r, i){
    var w = r.rating/max*100, lo = (r.rating-r.se)/max*100, hi = (r.rating+r.se)/max*100;
    h += '<div class="hrow" data-i="'+i+'"><div class="hlab"><span class="rk">'+r.rank+'</span><span class="nm">'+esc(r.name)+'</span></div>'+
      '<div class="track">'+gridHtml(0,max,step)+'<div class="bar" style="left:0;width:0" data-w="'+w+'"></div><div class="whisk" style="left:'+lo+'%;width:'+(hi-lo)+'%"></div></div>'+
      '<div class="hval">'+r.rating_s+' <small>&plusmn; '+r.se_s+'</small></div></div>';
  });
  h += axisHtml(0, max, step);
  var el = $("topChart"); el.innerHTML = h;
  requestAnimationFrame(function(){ el.querySelectorAll(".bar").forEach(function(b){ b.style.width = b.getAttribute("data-w")+"%"; }); });
  el.querySelectorAll(".hrow").forEach(function(row){
    var r = rows[+row.getAttribute("data-i")];
    bindTip(row, function(){
      var s = r.strongest.filter(function(x){return x.pos;}).map(function(x){return esc(x.metric)+' <span class="pk">'+x.sigma+'&sigma;</span>';}).join("<br>");
      return '<b>'+r.rank+'. '+esc(r.name)+'</b><br><span class="k">'+esc(r.role)+' &middot; '+esc(r.team)+'</span><br>'+
        'Rating <b>'+r.rating_s+'</b> &plusmn; '+r.se_s+'<br><span class="k">Pressure balls</span> '+r.nH.toLocaleString("en-US")+' &middot; <span class="k">'+esc(r.seasons)+'</span>'+
        (s ? '<br><br><span class="k">Strongest under pressure</span><br>'+s : '');
    });
  });
}
tabs("topTabs", renderTop); renderTop("bat");

/* 2. explorer */
$("explorerLede").textContent = "Search all " + D.counts.bat + " ranked batters and " + D.counts.bowl + " ranked bowlers. Process metrics are on high-pressure balls, in standard deviations against the other ranked players.";
var P = D.players, q = $("q"), res = $("res"), act = -1, hits = [];
function norm(s){ return s.normalize("NFD").replace(/[̀-ͯ]/g,"").toLowerCase(); }
function sideName(s){ return s==="bat"?"batters":"bowlers"; }
function renderResults(){
  var t = norm(q.value.trim());
  if (!t){ res.classList.remove("on"); q.setAttribute("aria-expanded","false"); return; }
  hits = P.filter(function(p){ return norm(p.name).indexOf(t) >= 0; }).slice(0, 12);
  act = hits.length ? 0 : -1;
  res.innerHTML = hits.length ? hits.map(function(p, i){
    return '<button role="option" data-i="'+i+'" class="'+(i===act?"act":"")+'"><span>'+esc(p.name)+'</span><span class="tag">#'+p.rank+' '+(p.side==="bat"?"batter":"bowler")+'</span></button>';
  }).join("") : '<div style="padding:12px;color:var(--ink-3);font-size:14px">No ranked player by that name</div>';
  res.classList.add("on"); q.setAttribute("aria-expanded","true");
}
q.addEventListener("input", renderResults);
q.addEventListener("focus", renderResults);
q.addEventListener("keydown", function(e){
  if (!hits.length) return;
  if (e.key==="ArrowDown"||e.key==="ArrowUp"){ e.preventDefault(); act=(act+(e.key==="ArrowDown"?1:-1)+hits.length)%hits.length;
    res.querySelectorAll("button").forEach(function(b,i){ b.classList.toggle("act", i===act); if(i===act) b.scrollIntoView({block:"nearest"}); }); }
  if (e.key==="Enter" && act>=0){ e.preventDefault(); pick(hits[act]); }
  if (e.key==="Escape"){ res.classList.remove("on"); }
});
res.addEventListener("mousedown", function(e){ var b=e.target.closest("button"); if(b){ e.preventDefault(); pick(hits[+b.getAttribute("data-i")]); } });
document.addEventListener("click", function(e){ if(!e.target.closest(".search")) res.classList.remove("on"); });
function pick(p){ q.value = p.name; res.classList.remove("on"); q.setAttribute("aria-expanded","false"); renderCard(p); }

function renderCard(p){
  var n = D.counts[p.side];
  var proc = "";
  var lim = 3;
  p.proc.forEach(function(m){
    var z = Math.max(-lim, Math.min(lim, m.z));
    var left = z >= 0 ? 50 : 50 + z/lim*50, w = Math.abs(z)/lim*50;
    proc += '<div class="hrow"><div class="hlab"><span class="nm" style="font-weight:500;font-size:14px">'+esc(m.metric)+'</span></div>'+
      '<div class="track">'+gridHtml(-lim,lim,1)+'<div class="bar '+(m.z<0?"neg":"")+' '+(m.z>=0?"pink":"")+'" style="left:'+left+'%;width:'+w+'%;'+(m.z<0?"background:#9aa0b6":"")+'"></div></div>'+
      '<div class="hval">'+m.z_s+'&sigma;</div></div>';
  });
  proc += axisHtml(-lim, lim, 1, function(v){ return v%2 ? "" : (v>0?"+":"")+String(v).replace("-",MINUS)+"σ"; });
  $("pcard").innerHTML =
    '<div><span class="prank">#'+p.rank+' of '+n+' '+sideName(p.side)+'</span>'+
    '<div class="pname">'+esc(p.name)+'</div><div style="color:var(--ink-3)">'+esc(p.role)+'</div>'+
    '<div class="facts">'+
      '<div class="fact"><span>Rating</span><b>'+p.rating_s+' <small style="color:var(--ink-3);font-weight:500">&plusmn; '+p.se_s+'</small></b></div>'+
      '<div class="fact"><span>Pressure balls</span><b>'+p.nH.toLocaleString("en-US")+'</b></div>'+
      '<div class="fact"><span>Seasons</span><b style="font-size:15px">'+esc(p.seasons)+'</b></div>'+
      '<div class="fact"><span>Role</span><b style="font-size:15px">'+esc(p.role)+'</b></div>'+
    '</div><p class="note">Rating: runs per 100 high-pressure balls above expectation.</p></div>'+
    '<div class="proc"><p class="chart-title">Process on pressure balls</p><p class="chart-sub">Standard deviations from the average ranked '+(p.side==="bat"?"batter":"bowler")+'.</p>'+proc+'</div>';
}
var chips = D.top.bat.slice(0,3).concat(D.top.bowl.slice(0,3));
$("chips").innerHTML = '<span style="font-size:13px;color:var(--ink-3);align-self:center">Try</span>' + chips.map(function(c,i){ return '<button class="chip" data-i="'+i+'">'+esc(c.name)+'</button>'; }).join("");
$("chips").addEventListener("click", function(e){
  var b = e.target.closest("button"); if(!b) return;
  var nm = chips[+b.getAttribute("data-i")].name;
  var p = P.filter(function(x){ return x.name===nm; })[0]; if (p) pick(p);
});
renderCard(P.filter(function(x){return x.side==="bat";})[0]);

/* 3. finding */
$("bnDelta").textContent = D.finding.delta;
$("bnOurs").textContent = D.finding.ours;
$("bnCi").textContent = "(95% interval " + D.finding.ours_ci + ")";
$("findingText").innerHTML = "Take a bowler's normal economy away from his pressure economy and you're left with a number that predicts nothing. This rating predicts bowlers' future pressure performance at <b>r = "+D.finding.ours+"</b>, ahead of raw death-over economy at "+D.finding.death+". The picks held up, too: of the 2022&ndash;24 top-20 bowlers, <b>"+D.top20.bowl.share+"</b> were above average under pressure in 2025&ndash;26, against "+D.top20.bowl.rand+" for a typical eligible bowler. For batters it was "+D.top20.bat.share+" against "+D.top20.bat.rand+".";
(function(){
  var lo = -0.5, hi = 0.8, h = "";
  [["bowl","Bowlers"],["bat","Batters"]].forEach(function(s){
    var f = D.val[s[0]];
    h += '<div class="facet"><h4>'+s[1]+' &middot; '+f.n+' players</h4>';
    f.rows.forEach(function(r, i){
      var x = (r.r-lo)/(hi-lo)*100, a = (r.lo-lo)/(hi-lo)*100, b = (r.hi-lo)/(hi-lo)*100;
      h += '<div class="dotrow '+(r.ours?"ours":"")+'" data-s="'+s[0]+'" data-i="'+i+'"><div class="lb">'+esc(r.label)+'</div>'+
        '<div class="track">'+gridHtml(lo,hi,0.25)+'<div class="ci" style="left:'+a+'%;width:'+(b-a)+'%"></div><div class="dot" style="left:'+x+'%"></div></div>'+
        '<div class="hval" style="'+(r.ours?"color:var(--pink)":"")+'">'+r.r_s+'</div></div>';
    });
    h += '</div>';
  });
  var small = window.innerWidth < 640;
  h += axisHtml(lo, hi, 0.25, function(v){ return (small && Math.abs(v*2 - Math.round(v*2)) > 1e-9) ? "" : v.toFixed(2).replace("-",MINUS); }).replace(/class="axis"/,'class="axis" style="grid-template-columns:var(--lab,250px) 1fr var(--val,70px)"');
  var el = $("valChart"); el.innerHTML = h;
  el.querySelectorAll(".dotrow").forEach(function(row){
    var r = D.val[row.getAttribute("data-s")].rows[+row.getAttribute("data-i")];
    bindTip(row, function(){ return '<b>'+esc(r.label)+'</b><br>Pearson r <b>'+r.r_s+'</b><br><span class="k">95% interval</span> '+r.ci_s+'<br><span class="k">Spearman</span> '+r.rho_s; });
  });
})();
(function(){ var s = document.createElement("style"); s.textContent = "@media (max-width:640px){#valChart .axis{grid-template-columns:118px 1fr 44px!important;gap:8px}}"; document.head.appendChild(s); })();

/* 4. pressure kpis */
$("pkpis").innerHTML = [
  ["1", "the leverage of an average ball"],
  [D.method.liCut, "where the pressure set starts"],
  ["25%", "of balls in each innings form H"]
].map(function(k){ return '<div class="kpi"><b>'+k[0]+'</b><span>'+k[1]+'</span></div>'; }).join("");

/* 5. value gap */
function renderGap(k){
  var rows;
  if (k === "top") rows = D.gap.bat.slice(0,10).concat(D.gap.bowl.slice(0,10));
  else rows = D.gap[k].slice();
  rows = rows.filter(function(r){ return r.gap > 0; }).sort(function(a,b){ return b.gap - a.gap || a.pr - b.pr; });
  if (k !== "top") rows = rows.slice(0, 15);
  var max = niceMax(rows[0].gap), step = max<=30?5:10, h = "";
  var t = rows[0];
  $("gapTitle").textContent = t.name + " ranks " + t.gap + " places higher on pressure than on " + (t.side==="bat"?"average and strike rate":"economy and wickets");
  rows.forEach(function(r, i){
    var hl = k==="top" ? i < 6 : r.pr <= 10;
    h += '<div class="hrow" data-i="'+i+'"><div class="hlab"><span class="rk">'+r.pr+'</span><span class="nm">'+esc(r.name)+'</span><span class="sm">'+(r.side==="bat"?"batter":"bowler")+'</span></div>'+
      '<div class="track">'+gridHtml(0,max,step)+'<div class="bar '+(hl?"pink":"")+'" style="left:0;width:0" data-w="'+(r.gap/max*100)+'"></div></div>'+
      '<div class="hval">+'+r.gap+'</div></div>';
  });
  h += axisHtml(0, max, step, function(v){ return v ? "+"+v : "0"; });
  var el = $("gapChart"); el.innerHTML = h;
  requestAnimationFrame(function(){ el.querySelectorAll(".bar").forEach(function(b){ b.style.width = b.getAttribute("data-w")+"%"; }); });
  el.querySelectorAll(".hrow").forEach(function(row){
    var r = rows[+row.getAttribute("data-i")];
    bindTip(row, function(){ return '<b>'+esc(r.name)+'</b><br><span class="k">Pressure rank</span> <span class="pk">#'+r.pr+'</span><br><span class="k">Conventional rank</span> #'+r.cr+'<br>'+r.gap+' places higher on pressure'; });
  });
  $("gapNote").textContent = k==="top"
    ? "Rank numbers are pressure ranks. Just " + D.gapNotes.bat_avg_top10 + " of the top-10 batters are in the top 10 by batting average, and only " + D.gapNotes.bowl_wkt_top10 + " of the top-10 bowlers are in the top 10 by wickets."
    : "Rank numbers are pressure ranks. Top-10 pressure picks in pink.";
}
tabs("gapTabs", renderGap); renderGap("top");

/* 6. RR */
$("rrH").textContent = "Since 2022, RR's costliest pressure situation has been bowling at the death of a chase.";
$("rrTitle").textContent = "About " + D.rrHead.wins + " wins below a league-average side across " + D.rrHead.balls.toLocaleString("en-US") + " high-pressure balls at the death of a chase";
(function(){
  var rows = D.rr, m = Math.max.apply(null, rows.map(function(r){return Math.abs(r.wins);}));
  var lim = Math.ceil(m*2)/2, h = "";
  rows.forEach(function(r, i){
    var w = Math.abs(r.wins)/lim*50, left = r.wins >= 0 ? 50 : 50 - w;
    h += '<div class="hrow" data-i="'+i+'"><div class="hlab"><span class="nm" style="font-weight:600">'+esc(r.label)+'</span><span class="sm">'+esc(r.role)+'</span></div>'+
      '<div class="track">'+gridHtml(-lim,lim,0.5)+'<div class="bar '+(r.wins<0?"neg pink":"")+'" style="left:'+left+'%;width:'+w+'%"></div></div>'+
      '<div class="hval">'+r.wins_s+'</div></div>';
  });
  var small = window.innerWidth < 640;
  h += axisHtml(-lim, lim, 0.5, function(v){ return (small && v % 1) ? "" : (v>0?"+":"")+v.toFixed(1).replace("-",MINUS); });
  var el = $("rrChart"); el.innerHTML = h;
  el.querySelectorAll(".hrow").forEach(function(row){
    var r = rows[+row.getAttribute("data-i")];
    bindTip(row, function(){ return '<b>'+esc(r.label)+'</b><br><span class="k">RR '+esc(r.role)+'</span><br>'+r.wins_s+' wins vs league average<br><span class="k">Pressure balls</span> '+r.balls+' &middot; <span class="k">matches</span> '+r.matches; });
  });
})();
$("best").innerHTML = D.best.map(function(b){
  return '<div class="card"><h3>'+esc(b.label)+'</h3><span style="font-size:13px;color:var(--ink-3)">RR bowling</span><ol>'+
    b.players.map(function(p){ return '<li><span>'+esc(p.name)+' <span style="color:var(--ink-3)">#'+p.rank+'</span></span><b>'+p.v+'</b></li>'; }).join("")+'</ol></div>';
}).join("");
$("recs").innerHTML = [
  ["Fix the death and the breakthrough overs first.", "Narine, Hazlewood, Rashid, Pathirana and Mujeeb pair a top-10 rating with strong records in RR's two costliest situations."],
  ["Start with the value-gap names.", "Eshan Malinga, Matheesha Pathirana and Nathan Ellis rank "+D.rec2[0]+" to "+D.rec2[1]+" places higher on pressure than on economy and wickets."],
  ["Pay for proven skill, not reputation.", "Core skill in players who've already delivered under pressure is what carries into the next season."]
].map(function(r,i){ return '<div class="rec"><div class="n">'+(i+1)+'</div><div><h3>'+r[0]+'</h3><p>'+r[1]+'</p></div></div>'; }).join("");

/* 7. method */
var M = D.method;
var plus = '<span class="pl" aria-hidden="true"><svg width="12" height="12" viewBox="0 0 12 12"><path d="M6 1v10M1 6h10" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg></span>';
var panels = [
  ["Pressure definition",
   '<p>Pressure is measured, not guessed: the Leverage Index is how far the batting side\'s win probability is expected to move on a single ball.</p>'+
   '<div class="formula">LI<sub>raw</sub> = &Sigma;<sub><i>o</i></sub> <i>p</i>(<i>o</i> | state) <span class="op">&times;</span> | WP(after <i>o</i>) &minus; expected WP after the ball |<br>LI = LI<sub>raw</sub> <span class="op">&divide;</span> average LI<sub>raw</sub></div>'+
   '<p>An average ball scores 1. Required rate, wickets, balls left and how close the contest is all flow through this one number. No weights are picked by hand, and it works in the first innings too, before a target even exists.</p>'+
   '<p>The top 25% of balls by leverage within each innings form the pressure set, <b>H</b>; the rest form the comparison set, <b>L</b>. Cutting within each innings keeps first-innings specialists in the picture. The pressure set starts at LI '+M.liCut+'.</p>'],
  ["Scoring every ball",
   '<p>Every ball is judged against what an average player would have done in exactly that situation, facing exactly that quality of opponent.</p>'+
   '<div class="formula">RV = (runs &minus; expected runs) <span class="op">&minus;</span> c<sub>w</sub> &times; (wicket &minus; expected wicket chance)</div>'+
   '<p>c<sub>w</sub> is what a wicket costs in runs at that moment. For bowlers, the sign flips. Expectations account for innings, phase, balls and wickets left, venue scoring, batting hand against bowling type, pace or spin, the level of competition and the opponent.</p>'+
   '<div class="tscroll"><table><thead><tr><th>Metric</th><th>Formula</th><th>What it shows</th></tr></thead><tbody>'+
   '<tr><td>Strike-rate delta</td><td>100 &times; &Sigma; (runs &minus; E[runs]) &divide; n</td><td>scoring above what the situation allows</td></tr>'+
   '<tr><td>Dot suppression</td><td>&Sigma; (E[dot] &minus; dot) &divide; n</td><td>keeping the scoreboard moving</td></tr>'+
   '<tr><td>Boundary elevation</td><td>&Sigma; (boundary &minus; E[boundary]) &divide; n</td><td>finding the rope when it\'s needed</td></tr>'+
   '<tr><td>Wicket security</td><td>&Sigma; (E[out] &minus; out) &divide; n</td><td>staying in when it matters</td></tr>'+
   '<tr><td>Economy suppression</td><td>6 &times; &Sigma; (E[conceded] &minus; conceded) &divide; balls</td><td>runs saved per over</td></tr>'+
   '<tr><td>Bowler dots, boundaries, wickets</td><td>each one, actual against expected</td><td>how those runs are saved</td></tr>'+
   '</tbody></table></div>'+
   '<div class="formula">P = average value on H &nbsp;&nbsp; B = average value on L<br>R = (P &minus; B) <span class="op">&minus;</span> league average of (P &minus; B)</div>'+
   '<p>R, the retention score, asks whether a player keeps his usual level under pressure compared with his peers. It works as a gate that removes clear fades, not as the ranking itself.</p>'],
  ["The rating",
   '<p>The rating is the best estimate of how a player performs on high-pressure balls: his core skill, plus his own pressure record, weighted by how much of that record the data shows is real.</p>'+
   '<h3 style="color:var(--ink)">Batters</h3><div class="formula">Rating = Prior<sub>L</sub> + <i>k</i> &times; (RV<sub>H</sub> &minus; Prior<sub>L</sub>)<br><i>k</i> = &tau;&sup2;<sub>dev</sub> &divide; (&tau;&sup2;<sub>dev</sub> + &sigma;&sup2;<sub>H</sub> &divide; n<sub>H</sub>)</div>'+
   '<p>The prior comes from a batter\'s normal balls. &tau;&sup2;<sub>dev</sub> is the true spread of pressure-specific skill between batters, estimated from the data itself. A batter\'s own pressure record moves his rating more as his sample grows.</p>'+
   '<h3 style="color:var(--ink)">Bowlers</h3><div class="formula">Rating = core skill from every ball + &delta;<br>&delta; = shrunk (RV<sub>H</sub> &minus; RV<sub>L</sub> &minus; league shift)</div>'+
   '<p>Every ball a bowler sends down is evidence of his skill. &delta; then adds his pressure-specific edge, scaled to how big those edges really are among bowlers.</p>'+
   '<div class="formula">Rank score = Rating <span class="op">&minus;</span> one standard error</div>'+
   '<p>The standard error comes from '+M.nBoot+' resamples of each player\'s matches. Ranking on the cautious end rewards players who are strong and consistent. Players really do differ in how they handle pressure: a model that lets each player respond in his own way fits far better (p = '+M.p.bat[0]+' &times; 10<sup>&minus;'+M.p.bat[1]+'</sup> for batters and '+M.p.bowl[0]+' &times; 10<sup>&minus;'+M.p.bowl[1]+'</sup> for bowlers).</p>'],
  ["Fair play",
   '<div class="tscroll"><table><thead><tr><th>To be ranked, a player needs</th><th>Batters</th><th>Bowlers</th><th>Why</th></tr></thead><tbody>'+
   '<tr><td>high-pressure balls</td><td>'+M.thr.bat+' or more</td><td>'+M.thr.bowl+' or more</td><td>about two regular seasons of pressure moments</td></tr>'+
   '<tr><td>recent cricket</td><td colspan="2">played in 2025 or 2026</td><td>the ranking is for the next auction</td></tr>'+
   '<tr><td>proven delivery</td><td colspan="2">beat expectation on high-pressure balls</td><td>"deliver when it matters"</td></tr>'+
   '<tr><td>no clear fade</td><td colspan="2">retention not clearly below peers</td><td>consistency</td></tr>'+
   '<tr><td><b>Players who qualify</b></td><td><b>'+D.counts.bat+'</b></td><td><b>'+D.counts.bowl+'</b></td><td></td></tr>'+
   '</tbody></table></div>'+
   '<p>Every value is pulled toward the league average in proportion to its noise, so a handful of lucky innings can\'t carry anyone to the top. IPL and T20I are pooled, and every model knows the level of competition: IPL, full-member internationals, or associate and A sides.</p>'+
   '<p>Before any modelling, run totals were reconciled with '+M.recon[0]+' of '+M.recon[1]+' recorded results, and rain-shortened games were kept out of the pressure sets, including '+M.rainUnrec+' that the data never marked as rain-affected.</p>'],
  ["Holding up",
   '<p>For each metric, every player\'s matches were split into two random halves and compared, repeated across 50 different splits. The process side of batting under pressure repeats clearly:</p>'+
   '<div class="tscroll"><table><thead><tr><th>Batting under pressure</th><th class="num">Reliability</th><th class="num">95% interval</th></tr></thead><tbody>'+
   M.rel.map(function(r){ return '<tr><td>'+r.label+'</td><td class="num">'+r.r+'</td><td class="num">'+r.ci+'</td></tr>'; }).join("")+
   '</tbody></table></div>'+
   '<p>The lists were also tested against changes to the analysis\'s own choices. Top-10 players kept:</p>'+
   '<div class="tscroll"><table><thead><tr><th>If the analysis changes</th><th class="num">Batters kept</th><th class="num">Bowlers kept</th></tr></thead><tbody>'+
   M.rob.map(function(r){ return '<tr><td>'+r.label+'</td><td class="num">'+r.bat+'</td><td class="num">'+r.bowl+'</td></tr>'; }).join("")+
   '</tbody></table></div>']
];
$("methodPanels").innerHTML = panels.map(function(p, i){
  return '<details'+(i===0?' open':'')+'><summary>'+p[0]+plus+'</summary><div class="dbody">'+p[1]+'</div></details>';
}).join("");

/* nav highlight + reveal */
var links = document.querySelectorAll("nav.top a");
if ("IntersectionObserver" in window){
  var secs = Array.prototype.map.call(links, function(a){ return document.querySelector(a.getAttribute("href")); });
  var io = new IntersectionObserver(function(es){
    es.forEach(function(e){ if (e.isIntersecting){ links.forEach(function(a){ a.classList.toggle("on", a.getAttribute("href")==="#"+e.target.id); }); } });
  }, {rootMargin:"-45% 0px -50% 0px"});
  secs.forEach(function(s){ if(s) io.observe(s); });
  var ro = new IntersectionObserver(function(es){ es.forEach(function(e){ if(e.isIntersecting){ e.target.classList.add("in"); ro.unobserve(e.target); } }); }, {threshold:.08});
  document.querySelectorAll("main .card, main details, .rec").forEach(function(el){ el.classList.add("reveal"); ro.observe(el); });
}
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    run()
