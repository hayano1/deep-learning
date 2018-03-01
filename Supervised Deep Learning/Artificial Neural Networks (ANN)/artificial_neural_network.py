#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 16:13:24 2018

@author: ngilmore
"""

# Deep Learning: Artificial Neural Network (ANN) Model in Python

#%reset -f

# Install Theano library in the terminal
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Install Tensorflow library in the terminal
# Install Tensorflow from https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html

# Install Keras library in the terminal
# pip install --upgrade keras

# Part 1: Data Pre-processing

# Import needed libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# Separate dependent (target) and independent variables
x = dataset.iloc[:, 3:13].values # independent variables
y = dataset.iloc[:, 13].values # dependent variable

# Encode categorical variables
# Encode the independent categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
labelencoder_x_2 = LabelEncoder()
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])

# Create dummy variables for categorical variables with more than 2 options
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()

# remove one of the dummy variables to avoid the dummy variable trap
x = x[:, 1:]

# Encode the dependent variable - not needed here as y is either 0 or 1
# labelencoder_y = LabelEncoder()
# y = labelencoder_y.fit_transform(y)

# Split the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Feature scaling (required for neural networks)
# Standardization - x_stand = (x - mean(x)) / standard deviation (x)
# Normalization - x_norm = (x - min(x)) / (max(x) - min(x))
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Create the Artificial Neural Network (ANN)
# Import the Keras libraries and packages
import keras
from keras.models import Sequential # Initializes the ANN
from keras.layers import Dense # Builds the layers of the ANN

# Add a timer
from timeit import default_timer as timer
start = timer()

# Initialize the Artificial Neural Network (ANN)
classifier = Sequential()

# Add the Input Layer and the first Hidden Layer
# Tip: select the average number of input and output nodes as the number of nodes in the hidden layer as a quick way to determein number of nodes
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11)) # activation is the rectifier activation function for the hidden layers

# Add the second Hidden Layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu')) # activation is the rectifier activation function for the hidden lyers

# Add the Output Layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) # activation is the sigmoid activation function for the output layer

# Compile the Artificial Neural Network (ANN)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit the Artificial Neural Network (ANN) to the Training Set
classifier.fit(x = x_train, y = y_train, batch_size = 10, epochs = 100) # Batch_size and epochs selection is part artistry

# Make predictions and and evaluate the Artificial Neural Network (ANN) Model

# Predict Test set results with the Artificial Neural Network (ANN) Model
y_pred = classifier.predict(x_test)

# Convert probabilities into True or False
y_pred = (y_pred > 0.5)

# Make a Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate the accuracy of the Artificial Neural Network (ANN) Model (measures accuracy)
# accuracy = (TP + TN) / (TP + TN + FP + FN)
accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0]) 
accuracy

# Calculate the precision of the model (measures exactness)
# precision = TP / (TP + FP)
precision = cm[0][0] / (cm[0][0] + cm[0][1])
precision

# Calculate the recall of the model (measures completeness)
# recall = TP / (TP + FN)
recall = cm[0][0] / (cm[0][0] + cm[1][0])
recall

# F1 Score (compromise between Precision and Recall)
# F1 Score = 2 * Precision * Recall / (Precision + Recall)
f1_score = 2 * precision * recall / (precision + recall)
f1_score

# Elapsed time in minutes
end = timer()
print('Elapsed time in minutes: ')
print(0.1 * round((end - start) / 6))

# Add an end of work message
import os
os.system('say "your program has finished"')
