"""
CS 610 Advanced AI 
Project Title: Bayesian Network Modeling of NFL Game Stats and Market Signals
               for Value Identification
Authors: Leland Weeks, Johnny Belichev, & Ishant Somals
Date: June 2026
"""

import logging
logging.getLogger("pgmpy").setLevel(logging.WARNING)

from sklearn.model_selection import train_test_split

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BayesianEstimator, BIC

TARGET = "ats_result"


def run_hc(df):
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    hc = HillClimbSearch(train)

    # BIC is the standard scoring method for hill climbing
    best_structure = hc.estimate(scoring_method=BIC(train), max_iter=10000)

    # post-process: remove reversed edge and re-add in correct causal direction
    # ats_result cannot cause game stats — outcomes don't cause their own predictors
    edges = list(best_structure.edges())
    edges = [e for e in edges if e[0] != TARGET]
    if (TARGET, "epa_diff") in list(best_structure.edges()):
        edges.append(("epa_diff", TARGET))

    # discrete model for categorical features
    model = DiscreteBayesianNetwork(edges)

    # smooth with BDeu so category combos not seen don't get zero probability
    model.fit(train, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=5)

    y_test = test[TARGET]
    X_test = test.drop(columns=[TARGET])

    # remove columns not in the model
    X_test = X_test[[col for col in X_test.columns if col in list(model.nodes())]]

    # drop test rows from categories not seen during training
    # this happens for rare categories (e.g. visitor_fav with 1 row) that can land entirely in test and cause inference to crash
    for col in X_test.columns:
        train_cols = set(train[col].unique())
        X_test = X_test[X_test[col].isin(train_cols)]
    y_test = y_test[X_test.index]

    preds = model.predict(X_test)
    probs = model.predict_probability(X_test)
    probs = probs[['ats_result_cover', 'ats_result_loss', 'ats_result_push']]

    return model, preds[TARGET].values, probs.values, y_test.values
