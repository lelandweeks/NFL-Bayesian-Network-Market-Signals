"""
CS 610 Advanced AI 
Project Title: Bayesian Network Modeling of NFL Game Stats and Market Signals
               for Value Identification
Authors: Leland Weeks, Johnny Belichev, & Ishant Somal
Date: June 2026
"""

from pgmpy.estimators import PC
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

def run_pc(df, significance_level=0.05):
    pc = PC(df)

    skeleton = pc.estimate(
        variant="stable",
        ci_test="chi_square",
        significance_level=significance_level,
        return_type="dag"
    )

    model = DiscreteBayesianNetwork(skeleton.edges())
    model.fit(df, estimator=MaximumLikelihoodEstimator)

    return model
