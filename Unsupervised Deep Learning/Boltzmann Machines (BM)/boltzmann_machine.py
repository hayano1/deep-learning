#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 17:50:13 2018

@author: ngilmore
"""

# Deep Learning: Boltzmann Machine in Python

# Import needed libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Import the dataset
movies = pd.read_csv('ml-1m/movies.dat',
                     sep = '::', 
                     header = None,
                     engine = 'python',
                     encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat',
                     sep = '::', 
                     header = None,
                     engine = 'python',
                     encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat',
                     sep = '::', 
                     header = None,
                     engine = 'python',
                     encoding = 'latin-1')


# Describe the dataset
print(movies.describe().T)
print(users.describe().T)
print(ratings.describe().T)


# Round values to 2 decimals
print(np.round(movies.describe(), 2).T)
print(np.round(users.describe(), 2).T)
print(np.round(ratings.describe(), 2).T)

# View the first 5 records
movies.head(5)
users.head(5)
ratings.head(5)

# View the last 5 records
movies.tail(5)
users.tail(5)
ratings.tail(5)

# Identify the columns
list(movies)
list(ratings)
list(users)

# Identify column datatypes
movies.dtypes
users.dtypes
ratings.dtypes

# Identify missing data (basic)
pd.isnull(movies)
pd.isnull(users)
pd.isnull(ratings)

'''
# Identify correlation between numerical variables
movies.corr() # Pearson correlation
movies.corr('kendall') # Kendall Tau correlation
movies.corr('spearman') # Spearman Rank correlation

# Identify missing data
# pip install missingno
import missingno as msno # Provides a library of data missingness functions 
#%matplotlib inline
msno.matrix(movies)
msno.bar(movies)
msno.heatmap(movies)
msno.dendrogram(movies)
'''

# Prepare the Training and Test sets
