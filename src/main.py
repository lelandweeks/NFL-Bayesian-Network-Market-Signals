"""
CS 610 Advanced AI
Project Title: Bayesian Network Modeling of NFL Game Stats and Market Signals
               for Value Identification
Authors: Leland Weeks, Johnny Belichev, & Ishant Somal
Date: June 2026
"""


from data_loader import get_data
from features import get_cont, get_cat
from models.naive_bayes import run as run_nb
from evaluate import eval

# get the data
print("Loading data...")
df = get_data()

# get the continuous and then
# convert to categorical features
print("Processing features...")
df_cont = get_cont(df)
print("Converting to categorical features...")
df_cat = get_cat(df_cont)

# get the base line
print("Running Naive Bayes baseline...")
X = df_cont.drop(columns=["ats_result", "roof", "surface",
                          "p_ml_home", "p_ml_visitor", "home_rolling_ats"])
y = df_cont["ats_result"]

#print(X.isnull().sum())
#print(X.shape)
model, preds, probs, y_test = run_nb(X, y)

metrics = eval(y_test, preds, probs)
print("Naive Bayes results:", metrics)