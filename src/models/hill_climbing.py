"""
CS 610 Advanced AI 
Project Title: Bayesian Network Modeling of NFL Game Stats and Market Signals
               for Value Identification
Authors: Leland Weeks, Johnny Belichev, & Ishant Somals
Date: June 2026
"""

from pgmpy.estimators import HillClimbSearch, BIC
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

def run_hc(df):
    hc = HillClimbSearch(df)
    best_structure = hc.estimate(scoring_method=BIC(df), max_iter=10000)
    model = DiscreteBayesianNetwork(best_structure.edges())
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    
    return model