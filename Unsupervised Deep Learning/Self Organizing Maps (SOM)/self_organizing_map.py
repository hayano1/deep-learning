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

# Separate dependent and independent variables (ensure dependent variable is the final column in the dataset)
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Scale Features
from sklearn.preprocessing import MinMaxScaler # Normalization scaling
sc = MinMaxScaler(feature_range = (0,1))
x = sc.fit_transform(x)

# Train the Self Organizing Map (SOM) Model

# Steps to Train a Self Organizing Map (SOM)
# Step 1: Start with a dataset composed of n_features independent variables
# Step 2: Create a grid composed of nodes, each having a weight vector of n_features elements
# Step 3: Randomly initialize the values of the weight vectors to small numbers close to 0 (but not 0)
# Step 4: Select one random observation point from the dataset
# Step 5: Compute the Euclidean distances from this point to the different neurons in the network
# Step 6: Select the neuron that has the minimum distance to the point. This neuron is called the winning node
# Step 7: Update the weights of the winning node to move it closer to the point
# Step 8: Using a Gaussian neighbourhood function of mean the winning node, also update the weights of the winning node neighbours to move them closer to the point. The neighbourhood radius is the sigma in the Gaussian function
# Step 9: Repeat Steps 1-5 and update the weights after each observation (Reinforcement Learning) or after a batch of observations (Batch Learning), until the network converges to a point where the neighbourhood stops decreasing

# pip install minisom
from minisom import MiniSom
som = MiniSom(x = 10, 
              y = 10,
              input_len = 15,
              sigma = 1.0,
              learning_rate = 0.5,
              decay_function = None,
              neighborhood_function = 'gaussian',
              random_seed = None)
som.random_weights_init(x)
som.train_random(data = x,
                 num_iteration = 100)

# Visualize the results to identify the outlying neurons in the Self Organizing Map (SOM)
from pylab import bone, pcolor, colorbar, plot, show

bone()
pcolor(som.distance_map().T) # Take the transpose of the Mean Interneuron Distances (MID)
colorbar() # Add a Legend
markers = ['o', 's'] # Add markers to identify those that were approved / not approved
colors = ['r', 'b'] # Add colors to identify those that were approved / not approved
for i, j  in enumerate(x):
    w = som.winner(j) # Identify the winning node
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2) # Add the marker to the center of the square of the winning node
show()

# Identify the potentially fraudulent applications
mappings = som.win_map(x) # Create a dictionary of mappings
potential_fraud = mappings[(4, 4)] # Select the matrix locations for fraud from visualization above (all the white squares)
potential_fraud = sc.inverse_transform(potential_fraud)
