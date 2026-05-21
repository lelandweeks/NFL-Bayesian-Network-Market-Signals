"""
CS 610 Advanced AI
Project Title: Bayesian Network Modeling of NFL Game Stats and Market Signals
               for Value Identification
Authors: Leland Weeks, Johnny Belichev, & Ishant Somal
Date: June 2026
"""


import argparse

from data_loader import get_data
from features import get_cont, get_cat
from models.naive_bayes import run_nb
from models.hill_climbing import run_hc
from models.pc_algorithm import run_pc
from evaluate import eval

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["nb", "hc", "pc", "all"], default="all")
args = parser.parse_args()



# get the data
print("Loading data...")
df = get_data()

# get the continuous and then
# convert to categorical features
print("Processing features...")
df_cont = get_cont(df)
print("Converting to categorical features...")
df_cat = get_cat(df_cont)


# NAIVE BAYES BASELINE
if args.model in ("nb", "all"):
    # 81.6% accuracy with the following class distribution
    # loss     0.580172
    # cover    0.389368
    # push     0.030460
    # signals that 81.6% is a valid baseline because it is significantly
    # better than the naive majority class baseline of 58%
    # i.e. if we just predicted "loss" for every game, we would be correct 58% of the time

    print("Running Naive Bayes baseline...")
    X = df_cont.drop(columns=["ats_result", "roof", "surface",
                              "p_ml_home", "p_ml_visitor", "home_rolling_ats"])
    #print(X.isnull().sum())
    #print(X.shape)
    y = df_cont["ats_result"]
    print(y.value_counts(normalize=True))
    model, preds, probs, y_test = run_nb(X, y)
    metrics = eval(y_test, preds, probs)
    print("Naive Bayes results:", metrics)
    #print(pd.Series(preds).value_counts(normalize=True))


# BAYESIAN NETWORK WITH HILL CLIMBING STRUCTURE LEARNING
if args.model in ("hc", "all"):

    print("Running Hill Climbing for Bayes Network...")
    X = df_cat.drop(columns=["ats_result", "roof", "surface",
                            "p_ml_home", "p_ml_visitor", "home_rolling_ats"])
    y = df_cat["ats_result"]
    #print(df_cat.dtypes)
    #print(df_cat.isnull().sum())

    df_hc = df_cat.drop(columns=["p_ml_home", "p_ml_visitor", "home_rolling_ats"])
    model = run_hc(df_hc)
    #print("Hill Climbing edges:", model.edges())



if args.model in ("pc", "all"):
    print("Running PC Algorithm...")
    df_pc = df_cat.drop(columns=["p_ml_home", "p_ml_visitor", "home_rolling_ats"])
    model_pc = run_pc(df_pc)
    print("PC Algorithm edges:", model_pc.edges())

