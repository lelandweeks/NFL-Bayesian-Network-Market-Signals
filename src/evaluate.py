"""
CS 610 Advanced AI 
Project Title: Bayesian Network Modeling of NFL Game Stats and Market Signals
               for Value Identification
Authors: Leland Weeks, Johnny Belichev, & Ishant Somals
Date: June 2026
"""

from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

def eval(y_true, y_pred, y_proba, class_index=1):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba[:, class_index])
    }