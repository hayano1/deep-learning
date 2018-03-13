#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 15:10:55 2018

@author: ngilmore
"""

# Import needed libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')

# Describe the dataset
print(dataset.describe().T)

# Round values to 2 decimals
print(np.round(dataset.describe(), 2).T)

# View the first 5 records
dataset.head(5)

# View the last 5 records
dataset.tail(5)

# Identify the columns
list(dataset)

# Identify column datatypes
dataset.dtypes

# Identify missing data (basic)
pd.isnull(dataset)

# Identify correlation between numerical variables
dataset.corr() # Pearson correlation
dataset.corr('kendall') # Kendall Tau correlation
dataset.corr('spearman') # Spearman Rank correlation

# Identify missing data
# pip install missingno
import missingno as msno # Provides a library of data missingness functions 
#%matplotlib inline
msno.matrix(dataset)
msno.bar(dataset)
msno.heatmap(dataset)
msno.dendrogram(dataset)

