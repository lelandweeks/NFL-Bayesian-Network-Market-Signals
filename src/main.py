"""
CS 610 Advanced AI
Project Title: Bayesian Network Modeling of NFL Game Stats and Market Signals
               for Value Identification
Authors: Leland Weeks, Johnny Belichev, & Ishant Somal
Date: June 2026
"""


from data_loader import get_data
from features import get_cont, get_cat

# get the data
df = get_data()

# get the continuous and then
# convert to categorical features
df_cont = get_cont(df)
df_cat = get_cat(df_cont)

