"""
CS 610 Advanced AI
Project Title: Bayesian Network Modeling of NFL Game Stats and Market Signals
               for Value Identification
Authors: Leland Weeks, Johnny Belichev, & Ishant Somal
Date: June 2026
"""

import argparse
import logging

from data_loader import get_data
from features import get_cont, get_cat
from ablation import run_ablation

OUTPUT_DIR = 'output/'

logging.getLogger("pgmpy").setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("--model",  
                    choices=["nb", "hc", "pc", "all"], 
                    default="all")
parser.add_argument("--config", 
                    choices=["stats", "market", "combined", "all"], 
                    default="combined")
args = parser.parse_args()

# get the data
print("Loading data...")
df = get_data()

# get the continuous features and then
# convert to categorical features
print("Processing features...")
df_cont = get_cont(df)
print("Converting to categorical features...")
df_cat = get_cat(df_cont)

# run
models = ["nb", "hc", "pc"] 
if args.model and args.model != "all":
    models = [args.model]
configs = ["stats", "market", "combined"]
if args.config and args.config != "all":
    configs = [args.config]

results = run_ablation(df_cont, df_cat, models, configs, OUTPUT_DIR)

# print summary
print("\nResults Summary")
print(f"{'config'} {'model'} {'accuracy'} {'log_loss'}")
for r in results:
    print(f"{r['config']} {r['model']} {r['accuracy']} {r['log_loss']}")
