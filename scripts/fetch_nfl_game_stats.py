"""
Fetch raw NFL game stats from nflverse via nfl_data_py.
Outputs only raw data - no derived fields.

Raw columns kept:
  Join keys:    season, week, game_id, home_team, away_team
  Situational:  roof, surface, temp, wind
  Home stats:   home_score, home_total_yards, home_turnovers,
                home_epa_per_play, home_scoring_efficiency,
                home_completion_pct, home_touchdowns
  Away stats:   away_score, away_total_yards, away_turnovers,
                away_epa_per_play, away_scoring_efficiency,
                away_completion_pct, away_touchdowns

Usage:
    pip install nfl_data_py pandas --no-deps
    python fetch_nfl_game_stats_raw.py
"""

import pandas as pd
import nfl_data_py as nfl

SEASONS = list(range(2007, 2022))  # 2007 through 2021

print("Fetching schedules...")
games = nfl.import_schedules(SEASONS)

SCHEDULE_COLS = [
    "season", "week", "game_id",
    "home_team", "away_team",
    "home_score", "away_score",
    "roof", "surface", "temp", "wind",
]
games = games[[c for c in SCHEDULE_COLS if c in games.columns]].copy()
print(f"  Schedule: {len(games)} games")

print("Fetching play-by-play data (this takes a few minutes)...")
pbp = nfl.import_pbp_data(
    SEASONS,
    columns=[
        "season", "week", "game_id", "posteam",
        "yards_gained", "pass_attempt", "rush_attempt",
        "complete_pass", "interception", "fumble_lost",
        "touchdown", "epa",
    ],
    downcast=True,
)

print("  Aggregating offensive stats per team per game...")
off = pbp[pbp["posteam"].notna()].groupby(
    ["season", "week", "game_id", "posteam"]
).agg(
    total_yards        = ("yards_gained",  "sum"),
    pass_attempts      = ("pass_attempt",  "sum"),
    rush_attempts      = ("rush_attempt",  "sum"),
    completions        = ("complete_pass", "sum"),
    interceptions      = ("interception",  "sum"),
    fumbles_lost       = ("fumble_lost",   "sum"),
    touchdowns         = ("touchdown",     "sum"),
    epa_per_play       = ("epa",           "mean"),
).reset_index()

off["turnovers"]          = off["interceptions"] + off["fumbles_lost"]
off["total_plays"]        = off["pass_attempts"] + off["rush_attempts"]
off["scoring_efficiency"] = (off["touchdowns"] / off["total_plays"].replace(0, float("nan")) * 100).round(4)
off["completion_pct"]     = (off["completions"] / off["pass_attempts"].replace(0, float("nan"))).round(4)
off["epa_per_play"]       = off["epa_per_play"].round(6)

# Keep only what's needed
off = off[["season", "week", "game_id", "posteam",
           "total_yards", "turnovers", "touchdowns",
           "epa_per_play", "scoring_efficiency", "completion_pct"]]

# Split into home and away
home_off = off.merge(
    games[["season","week","game_id","home_team"]],
    left_on=["season","week","game_id","posteam"],
    right_on=["season","week","game_id","home_team"],
    how="inner"
).rename(columns={
    "total_yards":        "home_total_yards",
    "turnovers":          "home_turnovers",
    "touchdowns":         "home_touchdowns",
    "epa_per_play":       "home_epa_per_play",
    "scoring_efficiency": "home_scoring_efficiency",
    "completion_pct":     "home_completion_pct",
}).drop(columns=["posteam","home_team"])

away_off = off.merge(
    games[["season","week","game_id","away_team"]],
    left_on=["season","week","game_id","posteam"],
    right_on=["season","week","game_id","away_team"],
    how="inner"
).rename(columns={
    "total_yards":        "away_total_yards",
    "turnovers":          "away_turnovers",
    "touchdowns":         "away_touchdowns",
    "epa_per_play":       "away_epa_per_play",
    "scoring_efficiency": "away_scoring_efficiency",
    "completion_pct":     "away_completion_pct",
}).drop(columns=["posteam","away_team"])

# Merge everything
df = games.copy()
df = df.merge(home_off, on=["season","week","game_id"], how="left")
df = df.merge(away_off, on=["season","week","game_id"], how="left")

# Final column order
FINAL_COLS = [
    "season", "week", "game_id",
    "home_team", "away_team",
    "home_score", "away_score",
    "roof", "surface", "temp", "wind",
    "home_total_yards", "home_turnovers", "home_touchdowns",
    "home_epa_per_play", "home_scoring_efficiency", "home_completion_pct",
    "away_total_yards", "away_turnovers", "away_touchdowns",
    "away_epa_per_play", "away_scoring_efficiency", "away_completion_pct",
]
df = df[[c for c in FINAL_COLS if c in df.columns]]
df = df.sort_values(["season","week"]).reset_index(drop=True)

output = "nfl_game_stats_raw.csv"
df.to_csv(output, index=False)
print(f"\nSaved: {len(df)} rows x {len(df.columns)} columns -> {output}")
print(f"Seasons: {sorted(df['season'].unique())}")
print(f"\nNulls:\n{df.isnull().sum()}")
