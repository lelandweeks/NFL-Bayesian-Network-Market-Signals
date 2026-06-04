"""
CS 610 Advanced AI
Project Title: Bayesian Network Modeling of NFL Game Stats and Market Signals
               for Value Identification
Authors: Leland Weeks, Johnny Belichev, & Ishant Somal
Date: June 2026
"""

from models.naive_bayes import run_nb
from models.hill_climbing import run_hc
from models.pc_algorithm import run_pc
from evaluate import get_eval, print_report

TARGET = "ats_result"

STATS_FEATURES = [
    "epa_diff", "yards_diff", "turnover_diff", 
    "td_diff", "completion_diff",
    "temp", "wind", "roof", "surface", "home_rolling_ats"
]

MARKET_FEATURES = [
    "close_spread", "close_total", "spread_move",
    "total_move", "home_win_prob"
]

COMBINED_FEATURES = STATS_FEATURES + MARKET_FEATURES

# columns NB/BN can't use 
DROP = ["roof", "surface"]

# runs the model with config
def run_ablation(df_cont, df_cat, models, configs, output_dir):
    results = []

    for config in configs:
        features = COMBINED_FEATURES
        if config == "stats":
            features = STATS_FEATURES
        elif config == "market":
            features = MARKET_FEATURES

        # get the relevant columns for this config
        cols = [c for c in features + [TARGET]]
        cont_cols = [c for c in cols if c in df_cont.columns]
        cat_cols  = [c for c in cols if c in df_cat.columns]
        subset_cont = df_cont[cont_cols]
        subset_cat  = df_cat[cat_cols]

        print(f"\nStarting Config: {config}")

        if "nb" in models:
            metrics = _run_nb(subset_cont, config, output_dir)
            results.append({"config": config, "model": "nb", 
                            "accuracy": metrics["accuracy"],
                            "log_loss": metrics["log_loss"]})

        if "hc" in models:
            metrics = _run_hc(subset_cat, config, output_dir)
            results.append({"config": config, "model": "hc",
                            "accuracy": metrics["accuracy"],
                            "log_loss": metrics["log_loss"]})
            
        if "pc" in models:
            metrics = _run_pc(subset_cat, config, output_dir)
            results.append({"config": config, "model": "pc", 
                            "accuracy": metrics["accuracy"],
                            "log_loss": metrics["log_loss"]})

    #print("DEBUG results:", results)
    save_summary(results, output_dir)
    return results


# naive bayes baseline
def _run_nb(subset_cont, config_name, output_dir):
    print(f"Running Naive Bayes ({config_name})...")
    X = subset_cont.drop(columns=[TARGET])
    X = X.drop(columns=[c for c in DROP if c in X.columns])
    if "home_rolling_ats" in X.columns:
        X = X.dropna(subset=["home_rolling_ats"])
    y = subset_cont.loc[X.index, TARGET]
    _, preds, probs, y_test = run_nb(X, y)
    metrics = get_eval(y_test, preds, probs)
    print_report(y_test, preds, f"NB ({config_name})")
    return metrics


# bayesian network with hill climbing
def _run_hc(subset_cat, config_name, output_dir):
    print(f"Running Hill Climbing ({config_name})...")
    df = subset_cat.drop(columns=[c for c in DROP if c in subset_cat.columns])
    if "home_rolling_ats" in df.columns:
        df = df.dropna(subset=["home_rolling_ats"])
    model, preds, probs, y_test = run_hc(df)
    metrics = get_eval(y_test, preds, probs)
    print_report(y_test, preds, f"HC ({config_name})")
    save_dag(model, f"hc_{config_name}", output_dir)
    return metrics


# pc algorithm for bayesian network
def _run_pc(subset_cat, config_name, output_dir):
    print(f"Running PC Algorithm ({config_name})...")
    df = subset_cat.drop(columns=[c for c in DROP if c in subset_cat.columns])
    if "home_rolling_ats" in df.columns:
        df = df.dropna(subset=["home_rolling_ats"])
    model, preds, probs, y_test = run_pc(df)
    metrics = get_eval(y_test, preds, probs)
    print_report(y_test, preds, f"PC ({config_name})")
    save_dag(model, f"pc_{config_name}", output_dir)
    return metrics

# save the dag diagram to disk
def save_dag(model, name, output_dir):
    path = output_dir + name + "_dag.txt"
    edges = sorted(list(model.edges()))
    with open(path, "w") as f:
        f.write(f"{len(edges)} edges total\n")
        for src, dst in edges:
            f.write(f"{src} -> {dst}\n")
    print(f"DAG saved: {path}")

# save the metrics to disk
def save_summary(results, output_dir):
    import pandas as pd
    path = output_dir + "ablation_summary.csv"
    pd.DataFrame(results).to_csv(path, index=False)
    print(f"\nAblation summary saved: {path}")
