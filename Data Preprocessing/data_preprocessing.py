#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:45:07 2018

@author: ngilmore
"""

# Import needed libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Data.csv')

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

# Separate dependent and independent variables (ensure dependent variable is the final column in the dataset)
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Handle missing data (If necessary)
from sklearn.preprocessing import Imputer # Imputes numerical variables
imputer = Imputer(missing_values = 'NaN',
                  strategy = 'mean',
                  axis = 0)
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encode categorical variables (If necessary)
from sklearn.preprocessing import LabelEncoder # Encodes categorical variables
from sklearn.preprocessing import OneHotEncoder # Converts categorical variables to dummy variables

labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Split the dataset into Training and Test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Scale Features (If necessary)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

