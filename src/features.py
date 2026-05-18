"""
CS 610 Advanced AI
Project Title: Bayesian Network Modeling of NFL Game Stats and Market Signals
               for Value Identification
Authors: Leland Weeks, Johnny Belichev, & Ishant Somal
Date: June 2026
"""

import pandas as pd

NON_FEATURES = ["game_id", "date", "line_quality",
                "home_scoring_efficiency", "away_scoring_efficiency"]

# get continuous features
def get_cont(df):

    print(df.head())
    print(df.columns)

    # copy the data instead of modifying in place
    df = df.copy()

    # need to sort by season+week to ensure rolling features only use prior games
    df = df.sort_values(["season", "week"]).reset_index(drop=True)

    # drop non-feature columns that are not needed for the model
    df = df.drop(columns=NON_FEATURES)

    # target ATS: cover, push, loss
    margin = df["home_score"] - df["away_score"]
    df["ats_result"] = None # TODO
    df.drop(columns=["home_score", "away_score"], inplace=True)

    # use these game stat features directly
    df["temp"] = df["temp"]
    df["wind"] = df["wind"]
    df["roof"] = df["roof"]
    df["surface"] = df["surface"]

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
    df["p_ml_home"] = df["ml_home"] / (df["ml_home"] + df["ml_visitor"])
    df["p_ml_visitor"] = df["ml_visitor"] / (df["ml_home"] + df["ml_visitor"])
    df.drop(columns=["open_spread", "open_total",
                     "ml_home", "ml_visitor"], inplace=True)

    # make sure to not introduce data leakage by using future games
    # TODO
    df["home_rolling_ats"] = None

    return df


# get categorical features
def get_cat(df):

    # copy the data instead of modifying in place
    df = df.copy()

    return df
