"""
CS 610 Advanced AI
Project Title: Bayesian Network Modeling of NFL Game Stats and Market Signals
               for Value Identification
Authors: Leland Weeks, Johnny Belichev, & Ishant Somal
Date: June 2026
"""


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["nb", "hc", "pc", "all"], default="all")
args = parser.parse_args()

from data_loader import get_data
from features import get_cont, get_cat
from models.naive_bayes import run_nb
from models.hill_climbing import run_hc
from models.pc_algorithm import run_pc
from evaluate import eval


# save the dag diagram to disk
OUTPUT_DIR = 'output/'
def save_dag(model, model_name):
    path = OUTPUT_DIR + model_name + '_dag.txt'
    edges = sorted(list(model.edges()))
    with open(path, 'w') as f:
        f.write(str(len(edges)) + ' edges total\n')
        for src, dst in edges:
            f.write(f"  {src} -> {dst}\n")
    print('DAG saved: ' + path)


# get the data
print("Loading data...")
df = get_data()

# get the continuous features and then
# convert to categorical features
print("Processing features...")
df_cont = get_cont(df)
print("Converting to categorical features...")
df_cat = get_cat(df_cont)

# realized that data column "line_quality" was not being respected 
#print(df_cat["close_spread"].value_counts())


# NAIVE BAYES BASELINE
if args.model in ("nb", "all"):
    # first results before all features implemented:
    # 81.6% accuracy with the following class distribution
    # loss     0.580172
    # cover    0.389368
    # push     0.030460
    # signals that 81.6% is a valid baseline because it is significantly
    # better than the naive majority class baseline of 58%
    # i.e. if we just predicted "loss" for every game, we would be correct 58% of the time

    print("Running Naive Bayes baseline...")
    X = df_cont.drop(columns=["ats_result", "roof", "surface"])
    X = X.dropna(subset=["home_rolling_ats"])

    #print(X.isnull().sum())
    #print(X.shape)
    #y = df_cont["ats_result"]
    y = df_cont.loc[X.index, "ats_result"]
    print(y.value_counts(normalize=True))
    model_nb, preds, probs, y_test = run_nb(X, y)
    metrics = eval(y_test, preds, probs)
    print("Naive Bayes Results:", metrics)
    #print(pd.Series(preds).value_counts(normalize=True))


# BAYESIAN NETWORK WITH HILL CLIMBING STRUCTURE LEARNING
if args.model in ("hc", "all"):
    print("Running Hill Climbing for Bayes Network...")
    df_hc = df_cat.drop(columns=["roof", "surface", "home_rolling_ats"])
    model_hc, preds, probs, y_test = run_hc(df_hc)
    metrics_hc = eval(y_test, preds, probs)
    print("Hill Climbing Results:", metrics_hc)
    save_dag(model_hc, "hc")


if args.model in ("pc", "all"):
    print("Running PC Algorithm for Bayes Network...")
    df_pc = df_cat.drop(columns=["roof", "surface", "home_rolling_ats"])
    model_pc, preds, probs, y_test = run_pc(df_pc)
    metrics_pc = eval(y_test, preds, probs)
    print("PC Algorithm Results:", metrics_pc)
    save_dag(model_pc, "pc")

