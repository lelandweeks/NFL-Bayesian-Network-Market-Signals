"""
CS 610 Advanced AI
Project Title: Bayesian Network Modeling of NFL Game Stats and Market Signals
               for Value Identification
Authors: Leland Weeks, Johnny Belichev, & Ishant Somal
Date: June 2026
"""

from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import classification_report, confusion_matrix


CLASSES = ["cover", "loss", "push"]

def get_eval(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss":  log_loss(y_true, y_proba, labels=CLASSES),
    }

def print_report(y_true, y_pred, model_name):
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_true, y_pred, labels=CLASSES, zero_division=0))
    print(f"{model_name} Confusion Matrix:")
    print(f"{'cover'} {'loss'} {'push'}")
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    for i, label in enumerate(CLASSES):
        print(f"{label} {cm[i][0]} {cm[i][1]} {cm[i][2]}")
    print()
