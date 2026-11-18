"""Role tags from the data (all kept matches, incl. rain-shortened: these are conventional descriptors).

Batters: median batting position, mean entry over (over number of the first delivery at the crease, as striker or
non-striker), innings batted. Bowlers: share of legal balls bowled in powerplay / middle / death, pace or spin.
Labels are descriptive only (thresholds below are naming conventions, not model parameters).
"""
import numpy as np
import pandas as pd

from . import config as C


def batter_label(pos):
    return np.select([pos <= 2, pos <= 3, pos <= 5, pos <= 7], ["opener", "No.3", "middle order (4-5)", "finisher (6-7)"],
                     "lower order (8+)")


def run():
    b = pd.read_parquet(C.PROC / "balls.parquet")
    b = b[~b.is_penalty]
    # --- batters
    at = pd.concat([b[["match_id", "innings", "innings_ball_num", "overid", "striker_id"]].rename(columns={"striker_id": "pid"}),
                    b[["match_id", "innings", "innings_ball_num", "overid", "non_striker_id"]].rename(columns={"non_striker_id": "pid"})])
    entry = at.sort_values("innings_ball_num").groupby(["match_id", "innings", "pid"]).overid.first().rename("entry_over")
    pos = b.groupby(["match_id", "innings", "striker_id"]).batting_order.first().rename("position")
    pos.index = pos.index.set_names(["match_id", "innings", "pid"])
    inn = pd.concat([entry, pos], axis=1).reset_index()
    names = b.groupby("striker_id").striker_name.agg(lambda s: str(s.iloc[0]))
    names = names.combine_first(b.groupby("non_striker_id").non_striker_name.agg(lambda s: str(s.iloc[0])))
    bt = inn.groupby("pid").agg(innings=("match_id", "size"), median_position=("position", "median"),
                                mean_entry_over=("entry_over", "mean"),
                                share_entering_overs_16_20=("entry_over", lambda s: (s >= 16).mean())).reset_index()
    bt.insert(1, "player", bt.pid.map(names))
    bt["role"] = batter_label(bt.median_position.fillna(11).values)
    bt = bt.rename(columns={"pid": "player_id"}).round(3)
    bt.to_csv(C.TABLES / "roles_batters.csv", index=False)
    # --- bowlers
    lg = b[b.legal]
    sh = lg.groupby(["bowler_id", lg.phase.astype(str)]).size().unstack(fill_value=0)
    tot = sh.sum(1)
    bw = pd.DataFrame({"player": lg.groupby("bowler_id").bowler_name.agg(lambda s: str(s.iloc[0])),
                       "legal_balls": tot, "matches": lg.groupby("bowler_id").match_id.nunique(),
                       "share_powerplay": sh.get("powerplay", 0) / tot, "share_middle": sh.get("middle", 0) / tot,
                       "share_death": sh.get("death", 0) / tot,
                       "pace_or_spin": lg.groupby("bowler_id").pace_or_spin.agg(lambda s: s.astype(str).mode().iat[0])})
    bw["overs_per_match"] = bw.legal_balls / 6 / bw.matches
    bw["main_phase"] = bw[["share_powerplay", "share_middle", "share_death"]].idxmax(axis=1).str.replace("share_", "")
    bw["role"] = bw.pace_or_spin.str.lower() + ", mainly " + bw.main_phase
    bw = bw.reset_index().rename(columns={"bowler_id": "player_id"}).round(3)
    bw.to_csv(C.TABLES / "roles_bowlers.csv", index=False)
    print(f"[roles] batters {len(bt)} {bt.role.value_counts().to_dict()}; bowlers {len(bw)} {bw.role.value_counts().to_dict()}")
