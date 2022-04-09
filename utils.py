"""
    This module contains helper functions for exploring and preprocessing data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def check_class_balance(df, var):
    """Calculates the fraction of data points that have a label of 1 for variable var.
    Args:
        df (DataFrame): dataframe of all data points, has a column var
        var (string): column name of label class
    Returns:
        float: percentage of data points labeled as one
    """
    
    counts = df[var].value_counts()
        
    return counts[1] / len(df)
    
def check_var_types(df, var_type):
    """Creates a list of categorical/ numerical variables
    Args:
        df (DataFrame): dataframe to check variables
        var_type (string): variable type, must be "categorical" or "numerical"
    Returns:
        list: a list containing the df columns for the given var_type
    """
    var_list = []
    err_msg = "Invalid variable type. var_type parameter must be set to 'categorical' or 'numerical'."
    if var_type == "categorical":
        var_list = [var for var in df.columns if df[var].dtypes == 'O']
    elif var_type == "numerical":
        var_list = [var for var in df.columns if df[var].dtypes != 'O']
    else:
        raise ValueError(err_msg)
        
    print(f'Number of {var_type} variables: {len(var_list)}')     
    
    return var_list

def check_unique_values(df, verbose="True"):
    """Checks and prints number of unique values 
    Args:
        df (DataFrame): dataframe to check unique variables
        verbose: if True, also prints unique values
    Returns:
        None
    """
    for col in df:

        print(f"Variable: {col} - Number of unique values: {df[col].nunique()}")
        if verbose == "True":
            print(f"Unique values:\n{df[col].unique()}\n")
        
def analyse_rare_labels(df, var, rare_perc=0.01, verbose=True):
    """Analyses labels that are present only in a small number of instances
    Args:
        df (DataFrame): dataframe to analyse
        var (string): column name to check
        rare_perc: percentage threshold for determining rare labels. Default is 0.01
        verbose: if True, also prints info about rare values
    Returns:
        Series: rare labels with percentages
    """
    df = df.copy()

    tmp = df.groupby(var)['INCOME'].count() / len(df)
    
    rares = tmp[tmp < rare_perc]
    
    if verbose:
        if len(rares) > 0:
            print(f"{rares.to_string()}\nRares / All Labels Count: {rares.count()}/{df[var].nunique()}\n")
        else:
            print(f"{rares.index.name}\nNo Rare Labels\n" )

    return rares

