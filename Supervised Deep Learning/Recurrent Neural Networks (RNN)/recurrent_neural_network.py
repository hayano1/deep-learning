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
training_set = dataset_train.iloc[:, 1:2].values # create a numpy array of 1 column that we care about - Google Stock Price

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
x_train = np.reshape(x_train, (x_train.shape[0], # number of rows in x_train
                               x_train.shape[1], # number of columns in x_train
                               1)) # number of input layers (currently only opening price)

# Part 2: Build the Recurrent Neural Network (RNN) Model

# Import the required Keras libraries and packages
from keras.models import Sequential # Initializes the Neural Network
from keras.layers import Dense # Builds the layers of the Neural Network
from keras.layers import LSTM # Long Short Term Memory
from keras.layers import Dropout # Dropout regularization to reduce overfitting

# Add a timer
from timeit import default_timer as timer
start = timer()

# Initialize the RNN
regressor = Sequential()

# Add the 1st LSTM layer with Dropout regularization
regressor.add(LSTM(units = 50, # number of memory cells (neurons) in this layer
                   return_sequences = True,
                   input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(rate = 0.2))

# Add a 2nd LSTM layer with Dropout regularization
regressor.add(LSTM(units = 50, # number of memory cells (neurons) in this layer
                   return_sequences = True))
regressor.add(Dropout(rate = 0.2))

# Add a 3rd LSTM layer with Dropout regularization
regressor.add(LSTM(units = 50, # number of memory cells (neurons) in this layer
                   return_sequences = True))
regressor.add(Dropout(rate = 0.2))

# Add a 4th (and last) LSTM layer with Dropout regularization
regressor.add(LSTM(units = 50, # number of memory cells (neurons) in this layer
                   return_sequences = False))
regressor.add(Dropout(rate = 0.2))

# Add the output layer
regressor.add(Dense(units = 1,
                    kernel_initializer = 'uniform', 
                    activation = 'sigmoid')) # activation is the sigmoid activation function for the output layer

# Compile the Recurrent Neural Network (RNN)
regressor.compile(optimizer = 'adam',
                  loss = 'mean_squared_error')

# Fit the Recurrent Neural Network (RNN) to the Training Set
regressor.fit(x = x_train,
              y = y_train,
              batch_size = 32,
              epochs = 100) # Batch_size and epochs selection is part artistry

# Elapsed time in minutes
end = timer()
print('Elapsed time in minutes: ')
print(0.1 * round((end - start) / 6))

# Add an end of work message
import os
os.system('say "your model has finished processing"')

# Part 3: Make Prediction and Visualize the Results

# Get the real Google stock prices for Jan 2017
# Import the test set
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values 

# Get the predicted Google stock prices for Jan 2017


# Visualize the results
