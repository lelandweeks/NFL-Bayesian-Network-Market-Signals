"""
CS 610 Advanced AI 
Project Title: Bayesian Network Modeling of NFL Game Stats and Market Signals
               for Value Identification
Authors: Leland Weeks, Johnny Belichev, & Ishant Somals
Date: June 2026
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

def run_nb(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)
    
    return model, preds, probs, y_test
