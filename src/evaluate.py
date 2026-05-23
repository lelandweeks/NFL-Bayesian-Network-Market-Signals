"""
CS 610 Advanced AI
Project Title: Bayesian Network Modeling of NFL Game Stats and Market Signals
               for Value Identification
Authors: Leland Weeks, Johnny Belichev, & Ishant Somal
Date: June 2026
"""

from sklearn.metrics import (
    accuracy_score,
    log_loss,
    classification_report,
    confusion_matrix
)

CLASSES = ["cover", "loss", "push"]

def get_eval(y_true, y_pred, y_proba):
    return f"\naccuracy: {accuracy_score(y_true, y_pred):.4f}" + f"\nlog_loss: {log_loss(y_true, y_proba):.4f}"

    #    "accuracy": accuracy_score(y_true, y_pred),
    #    "log_loss": log_loss(y_true, y_proba),
    #    "brier_score": brier_score_loss(y_true, y_proba[:, class_index])
    #}
def print_report(y_true, y_pred, model_name):
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_true, y_pred, labels=CLASSES, zero_division=0))
    print(f"{model_name} Confusion Matrix:")
    print(f"  {'':12} {'cover':>8} {'loss':>8} {'push':>8}")
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    for i, label in enumerate(CLASSES):
        print(f"  {label:12} {cm[i][0]:>8} {cm[i][1]:>8} {cm[i][2]:>8}")
    print()
