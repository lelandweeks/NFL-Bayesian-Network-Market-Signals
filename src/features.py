"""
CS 610 Advanced AI
Project Title: Bayesian Network Modeling of NFL Game Stats and Market Signals
               for Value Identification
Authors: Leland Weeks, Johnny Belichev, & Ishant Somal
Date: June 2026
"""

import pandas as pd

TARGET = "ats_result"

NON_FEATURES = ["game_id", "date", "line_quality",
                "home_scoring_efficiency", "away_scoring_efficiency"]

CONTINUOUS_FEATURES = ["temp", "wind",
                       "home_total_yards", "away_total_yards", 
                       "home_turnovers", "away_turnovers",  
                       "home_touchdowns", "away_touchdowns", 
                       "home_epa_per_play", "away_epa_per_play",
                       "home_completion_pct", "away_completion_pct",
                       "open_spread", "close_spread", 
                       "open_total", "close_total",
                       "ml_home", "ml_visitor"]

CATEGORICAL_FEATURES = ["season", "week", "roof", "surface",
                        "home_team", "away_team"]
                        

# get continuous features
def get_cont(df):

    print(df.head())
    print(df.columns)

    # copy the data instead of modifying in place
    df = df.copy()

    # need to sort by season+week to ensure rolling features only use prior games
    df = df.sort_values(["season", "week"]).reset_index(drop=True)

    # target ATS: cover, push, loss
    margin = df["home_score"] - df["away_score"]
    df[TARGET] = margin #TODO
    df.drop(columns=["home_score", "away_score"], inplace=True)

    # use these game stat features directly
    df["temp"] = df["temp"]
    df["wind"] = df["wind"]

    # derive game stat features
    df["epa_diff"] = df["home_epa_per_play"] - df["away_epa_per_play"]
    df["yards_diff"] = df["home_total_yards"] - df["away_total_yards"]
    df["turnover_diff"] = df["home_turnovers"] - df["away_turnovers"]
    df["td_diff"] = df["home_touchdowns"] - df["away_touchdowns"]
    df["completion_diff"] = df["home_completion_pct"] - df["away_completion_pct"]
    df.drop(columns=["home_epa_per_play", "away_epa_per_play", 
                     "home_total_yards", "away_total_yards",
                     "home_turnovers", "away_turnovers", 
                     "home_touchdowns", "away_touchdowns",
                     "home_completion_pct", "away_completion_pct"], inplace=True)

    # use these market features directly
    df["close_spread"] = df["close_spread"]
    df["close_total"] = df["close_total"]

    # derive market features
    df["spread_move"] = df["close_spread"] - df["open_spread"] 
    df["total_move"] = df["close_total"]  - df["open_total"]

    # TODO
    df["p_ml_home"] = None # df["ml_home"] / (df["ml_home"] + df["ml_visitor"])
    df["p_ml_visitor"] = None # df["ml_visitor"] / (df["ml_home"] + df["ml_visitor"])

    df.drop(columns=["open_spread", "open_total",
                     "ml_home", "ml_visitor"], inplace=True)

    # make sure to not introduce data leakage by using future games
    # TODO
    df["home_rolling_ats"] = None

    # drop non-feature columns that are not needed for the model
    df = df.drop(columns=NON_FEATURES)
    #df = df.drop(columns=CATEGORICAL_FEATURES)

    return df


# convert continuous features to categorical
def get_cat(df):

    # copy the data instead of modifying in place
    df = df.copy()

    # use these categorical features directly
    df["roof"] = df["roof"]
    df["surface"] = df["surface"]

    # continuous features to categorical bins
    df["temp"] = df["temp"].apply(lambda x: "cold" if x < 40 else "mild" if x < 80 else "warm")
    df["wind"] = df["wind"].apply(lambda x: "calm" if x < 5 else "breezy" if x < 15 else "windy")
    df["epa_diff"] = df["epa_diff"].apply(lambda x: "neg" if x < -0.05 else "pos" if x > 0.05 else "neutral")
    df["yards_diff"] = df["yards_diff"].apply(lambda x: "neg" if x < -5 else "pos" if x > 5 else "neutral")
    df["turnover_diff"] = df["turnover_diff"].apply(lambda x: "neg" if x < -1 else "pos" if x > 1 else "neutral")
    df["td_diff"] = df["td_diff"].apply(lambda x: "neg" if x < -1 else "pos" if x > 1 else "neutral")
    df["completion_diff"] = df["completion_diff"].apply(lambda x: "neg" if x < -0.05 else "pos" if x > 0.05 else "neutral")

    df["close_spread"] = df["close_spread"].apply(lambda x: "home_fav" if x < 0 else "visitor_fav")
    df["close_total"] = df["close_total"].apply(lambda x: "over" if x > 45 else "under")
    df["spread_move"] = df["spread_move"].apply(lambda x: "towards_home" if x < 0 else "towards_visitor" if x > 0 else "no_move")
    df["total_move"] = df["total_move"].apply(lambda x: "towards_over" if x > 0 else "towards_under" if x < 0 else "no_move")
    df["p_ml_home"] = None #TODO
    df["p_ml_visitor"] = None #TODO

    df["home_rolling_ats"] = None # TODO
 
    df.drop(columns=["temp", "wind", "home_team", "away_team"], inplace=True)

    return df 



