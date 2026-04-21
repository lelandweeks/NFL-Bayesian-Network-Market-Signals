"""
CS 610 Advanced AI 
Project Title: Bayesian Network Modeling of NFL Game Stats and Market Signals
               for Value Identification
Authors: Leland Weeks & Johnny Belichev
Date: April 2026
"""

import pandas as pd


TEAM_NAME_MAP = {
    "Arizona":      "ARI",
    "Atlanta":      "ATL",
    "Baltimore":    "BAL",
    "Buffalo":      "BUF",
    "Carolina":     "CAR",
    "Chicago":      "CHI",
    "Cincinnati":   "CIN",
    "Cleveland":    "CLE",
    "Dallas":       "DAL",
    "Denver":       "DEN",
    "Detroit":      "DET",
    "GreenBay":     "GB",
    "Houston":      "HOU",
    "HoustonTexans":"HOU",
    "Indianapolis": "IND",
    "Jacksonville": "JAX",
    "Kansas":       "KC",
    "KansasCity":   "KC",
    "KCChiefs":     "KC",
    "LAChargers":   "LAC",
    "LARams":       "LA",
    "LasVegas":     "LV",
    "LosAngeles":   "LA",
    "Miami":        "MIA",
    "Minnesota":    "MIN",
    "NewEngland":   "NE",
    "NewOrleans":   "NO",
    "NewYork":      "NYG",
    "NYGiants":     "NYG",
    "NYJets":       "NYJ",
    "Oakland":      "OAK",
    "Philadelphia": "PHI",
    "Pittsburgh":   "PIT",
    "SanDiego":     "SD",
    "SanFrancisco": "SF",
    "Seattle":      "SEA",
    "St.Louis":     "STL",
    "Tampa":        "TB",
    "TampaBay":     "TB",
    "Tennessee":    "TEN",
    "Washington":   "WAS",
}

def data_loader():


    # ==========
    # 1. Load Data
    # ==========

    game_stats = pd.read_csv("data/raw/nfl_game_stats_raw.csv")
    game_odds = pd.read_csv("data/raw/sbr_nfl_odds_raw.csv")



    # ==========
    # 2. Preprocessing odds dataset
    # ==========

    # line quality value indicates
    # 0 = good data
    # 1 = closing columns need to be swapped
    # 2 = opening columns need to be swapped
    # 3 = "pk" (pick'em) placeholder exists — drop the row
    # 4 = extreme outlier value exists — drop the row

    rows_to_drop = []
    for i, row in game_odds.iterrows():
        if row["line_quality"] in (3, 4):
            rows_to_drop.append(i)
        elif row["line_quality"] == 2:
            game_odds.at[i, "open_total"], game_odds.at[i, "open_spread"] = row["open_spread"], row["open_total"]
        elif row["line_quality"] == 1:
            game_odds.at[i, "close_total"], game_odds.at[i, "close_spread"] = row["close_spread"], row["close_total"]

    game_odds = game_odds.drop(index=rows_to_drop).copy()

    # ==========
    # 3. Create composite key to join tables
    # ==========

    # nfl stats key format (single column): game_id=2007_01_NO_IND

    # odds key format (multiple columns): 
    # season=2007-08, date=0906, visitor=NewOrleans, home_team=Indianapolis


    # get the first 4 characters, which is the year 
    game_odds["season"] = game_odds["season"].str[:4].astype(int)

    # need to map team names to their abbreviations
    game_odds["visitor"]   = game_odds["visitor"].map(TEAM_NAME_MAP)
    game_odds["home_team"] = game_odds["home_team"].map(TEAM_NAME_MAP)



    # ==========
    # 4. Drop rows with missing or bad data
    # ==========

    game_odds = game_odds.drop_duplicates(subset=["season", "home_team", "visitor"], keep="first")



    # ==========
    # 5. join the tables based on their keys
    # ==========

    merged = pd.merge(
        left      = game_stats,
        right     = game_odds,
        left_on   = ["season", "home_team", "away_team"],
        right_on  = ["season", "home_team", "visitor"],
        how       = "left",
    )



    # ==========
    # 6. Drop rows that don't have a matching odds row, and drop redundant columns
    # ==========

    merged = merged.dropna(subset=["close_spread"])
    merged = merged.drop(columns=["visitor", "score_home", "score_visitor"])


    # ==========
    # 7. set the index and return the data
    # ==========


    #print("Merge complete.")
    #print(f"  game_stats rows : {len(game_stats)}")
    #print(f"  odds rows       : {len(game_odds)}")
    #print(f"  merged rows     : {len(merged)}")
    #print(f"  merged columns  : {len(merged.columns)}")
    #print(f"\nColumns in final DataFrame:\n{list(merged.columns)}")

    return merged.reset_index(drop=True)







