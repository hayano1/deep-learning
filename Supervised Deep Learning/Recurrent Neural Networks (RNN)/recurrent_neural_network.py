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


# Part 2: Build the Recurrent Neural Network (RNN) Model



# Part 3: Make Prediction and Visualize the Results

