#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 14:02:38 2018

@author: ngilmore
"""

# Deep Learning: Stacked LSTM Recurrent Neural Network (RNN) Model in Python

# Predict Google stock price using a Long-Short Term Memory (LSTM) Recurrent Neural Network (RNN) 

# Part 1: Data Preprocessing

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values # crease a numpy array of 1 column that we care about - Google Stock Price

# Feature Scaling
# With RNNs it is recommended to apply normalization for feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1),
                  copy = True)
training_set_scaled = sc.fit_transform(training_set)

# Create a data structure with 60 timesteps and 1 output (use the previous 60 days' stock prices to predict the next output = 3 months of prices)
x_train = []
y_train = []

for i in range(60, 1258): # gives last 60 days, upper bound is the last record in the training_set
    x_train.append(training_set_scaled[i-60:i, 0]) # append the previous 60 days' stock prices
    y_train.append(training_set_scaled[i, 0]) # predict the stock price on the next day

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data to add additional indicators (e.g. volume, closing price, etc.)
x_train = np.reshape(x_train, (batch_size = x_train.shape[0], # number of rows in x_train
                               timesteps = x_train.shape[1], # number of columns in x_train
                               input_dim = 1))

# Part 2: Build the Recurrent Neural Network (RNN) Model



# Part 3: Make Prediction and Visualize the Results

