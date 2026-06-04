"""
CS 610 Advanced AI 
Project Title: Bayesian Network Modeling of NFL Game Stats and Market Signals
               for Value Identification
Authors: Leland Weeks, Johnny Belichev, & Ishant Somal
Date: June 2026
"""

import logging
logging.getLogger("pgmpy").setLevel(logging.WARNING)

from sklearn.model_selection import train_test_split

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import PC, BayesianEstimator

TARGET = "ats_result"

def run_pc(df):
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    pc = PC(train)

    skeleton = pc.estimate(
        variant="stable",
        ci_test="chi_square",
        significance_level=0.05, # changed from default to produce a sparser graph
        return_type="dag"
    )

    # discrete model for categorical features
    model = DiscreteBayesianNetwork(skeleton.edges())

    # smooth with BDeu so category combos not seen don't get zero probability
    model.fit(train, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=5)

    y_test = test[TARGET]
    X_test = test.drop(columns=[TARGET])

    model_nodes = list(model.nodes())
    X_test = X_test[[col for col in X_test.columns if col in model_nodes]]

    # drop test rows from categories not seen during training
    # this happens for rare categories (e.g. visitor_fav with 1 row) that can land entirely in test
    for col in X_test.columns:
        train_cols = set(train[col].unique())
        X_test = X_test[X_test[col].isin(train_cols)]
    y_test = y_test[X_test.index]

    preds = model.predict(X_test)
    probs = model.predict_probability(X_test)
    probs = probs[['ats_result_cover', 'ats_result_loss', 'ats_result_push']]

    return model, preds[TARGET].values, probs.values, y_test.values
