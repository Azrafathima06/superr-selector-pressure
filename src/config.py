"""Paths, seeds and starting-value thresholds (every threshold is tested for sensitivity later)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT
PROC = ROOT / "data_processed"
OUT = ROOT / "outputs"
TABLES = OUT / "tables"
CHARTS = OUT / "charts"
for _d in (PROC, OUT, TABLES, CHARTS):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
N_JOBS = 2

RR_TEAM_ID = 3

# ICC full members (12). Any other national side, or an "A" side, is "weaker T20I".
FULL_MEMBERS = {
    "Afghanistan", "Australia", "Bangladesh", "England", "India", "Ireland",
    "New Zealand", "Pakistan", "South Africa", "Sri Lanka", "West Indies", "Zimbabwe",
}

# Extras: verified by reconciliation in data.py (see outputs/01_data_audit.md)
WIDE = "wide"
NOBALL_TYPES = {"Overstep", "other"}          # 'other' = height/beamer/full-toss no-balls
NOT_A_DELIVERY = {"Penalty"}                 # 5-run penalty row, not a ball
BYE_TYPES = {"byes", "legbyes"}
BOWLER_WICKETS = {"Caught", "Bowled", "LBW", "Stump", "Hit Wicket"}

# Phases by over number (1-indexed overid)
PHASES = [(1, 6, "powerplay"), (7, 15, "middle"), (16, 20, "death")]

# Venue par window (starting value)
PAR_WINDOW_DAYS = 730

# Phase 3: rain-shortened matches (recorded or detected revisions, or < 20-over innings) are excluded from the
# pressure sets and player analysis: the WP model was never trained on shortened innings. Set False to include them.
EXCLUDE_SHORTENED_FROM_PRESSURE = True

# Primary high-leverage set (decided after Phase 3 review): top 25% of LI within each innings.
# Robustness sets: H20_within, H33_within (and the global versions, reported only).
PRIMARY_H = "H25_within"
