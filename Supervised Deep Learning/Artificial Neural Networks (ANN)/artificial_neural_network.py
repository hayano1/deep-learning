#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 16:13:24 2018

@author: ngilmore
"""

# Deep Learning: Artificial Neural Network (ANN) Model in Python

#%reset -f

# Install Theano library in the terminal
# pip install theano

# Install Tensorflow library in the terminal
# pip install tensorflow

# Install Keras library in the terminal
# pip install --upgrade keras

# Update Conda in the terminal
# conda update --all

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
# Encode all independent categorical variables
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

# Part 2: Create the Artificial Neural Network (ANN)
# Import the Keras libraries and packages
import keras
from keras.models import Sequential # Initializes the ANN
from keras.layers import Dense # Builds the layers of the ANN
from keras.layers import Dropout # Dropout regularization to reduce overfitting

# Add a timer
from timeit import default_timer as timer
start = timer()

# Initialize the Artificial Neural Network (ANN)
classifier = Sequential()

# Add the Input Layer and the first Hidden Layer with dropout
# Tip: select the average number of input and output nodes as the number of nodes in the hidden layer as a quick way to determein number of nodes
classifier.add(Dense(units = 6, 
                     kernel_initializer = 'uniform', 
                     activation = 'relu', 
                     input_dim = 11)) # activation is the rectifier activation function for the hidden layers
classifier.add(Dropout(p = 0.1))

# Add the second Hidden Layer
classifier.add(Dense(units = 6, 
                     kernel_initializer = 'uniform', 
                     activation = 'relu')) # activation is the rectifier activation function for the hidden lyers
classifier.add(Dropout(p = 0.1))

# Add the third Hidden Layer
classifier.add(Dense(units = 6, 
                     kernel_initializer = 'uniform', 
                     activation = 'relu')) # activation is the rectifier activation function for the hidden lyers
classifier.add(Dropout(p = 0.1))

# Add the Output Layer
classifier.add(Dense(units = 1, 
                     kernel_initializer = 'uniform', 
                     activation = 'sigmoid')) # activation is the sigmoid activation function for the output layer

# Compile the Artificial Neural Network (ANN)
classifier.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])

# Fit the Artificial Neural Network (ANN) to the Training Set
classifier.fit(x = x_train, 
               y = y_train, 
               batch_size = 10, 
               epochs = 100) # Batch_size and epochs selection is part artistry

# Make predictions and and evaluate the Artificial Neural Network (ANN) Model
# Predict Test set results with the Artificial Neural Network (ANN) Model
y_pred = classifier.predict(x_test)

# Convert probabilities into True or False
y_pred = (y_pred > 0.5)

# Make a Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cm = confusion_matrix(y_test, y_pred)
cm

# Print model precision, recall, f1-score, and support metrics
print(classification_report(y_test, y_pred))

# Elapsed time in minutes
end = timer()
print('Elapsed time in minutes: ')
print(0.1 * round((end - start) / 6))

# Add an end of work message
import os
os.system('say "your model has finished"')

# Predict a single new observation
""" Predict if the customer with the following characteristics will leave the bank
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000
"""

new_prediction = classifier.predict(sc_x.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))) # Remember to convert categorical variables 
new_prediction = (new_prediction > 0.5)
new_prediction

# Make a Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

# Part 4: Evaluating, Improving, and Tuning the ANN
# Evaluate the ANN
from keras.wrappers.scikit_learn import KerasClassifier # Keras classifier wrapper for scikit learn
from sklearn.model_selection import cross_val_score # k-fold cross validation function in sklearn
from keras.models import Sequential # Initializes the ANN
from keras.layers import Dense # Builds the layers of the ANN
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6,
                         kernel_initializer = 'uniform', 
                         activation = 'relu', 
                         input_dim = 11)) # activation is the rectifier activation function for the hidden layers
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = 6, 
                         kernel_initializer = 'uniform', 
                         activation = 'relu')) # activation is the rectifier activation function for the hidden lyers
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = 6, 
                         kernel_initializer = 'uniform', 
                         activation = 'relu')) # activation is the rectifier activation function for the hidden lyers
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = 1, 
                         kernel_initializer = 'uniform', 
                         activation = 'sigmoid')) # activation is the sigmoid activation function for the output layer
    classifier.compile(optimizer = 'adam', 
                       loss = 'binary_crossentropy', 
                       metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier,
                             batch_size = 10,
                             epochs = 100)

accuracies = cross_val_score(estimator = classifier,
                             X = x_train,
                             y = y_train,
                             cv = 10, # cv = number of folds
                             n_jobs = -1) # -1 uses all available cpus in parallel

mean = accuracies.mean()
mean

variance = accuracies.std()
variance

# Improve the ANN
# Dropout regularization to reduce overfitting if needed (added above)

# Tune the ANN hyperparameters
# Grid Search recommends optimum hyperparameter configuration
from keras.wrappers.scikit_learn import KerasClassifier # Keras classifier wrapper for scikit learn
from sklearn.model_selection import GridSearchCV # grid search hyperparameter tuning function in sklearn
from keras.models import Sequential # Initializes the ANN
from keras.layers import Dense # Builds the layers of the ANN
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6,
                         kernel_initializer = 'uniform', 
                         activation = 'relu', 
                         input_dim = 11)) # activation is the rectifier activation function for the hidden layers
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = 6, 
                         kernel_initializer = 'uniform', 
                         activation = 'relu')) # activation is the rectifier activation function for the hidden lyers
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = 1, 
                         kernel_initializer = 'uniform', 
                         activation = 'sigmoid')) # activation is the sigmoid activation function for the output layer
    classifier.compile(optimizer = optimizer, 
                       loss = 'binary_crossentropy', # handles stochastic gradient descent 
                       metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop', 'sgd']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(x_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

# best_parameters
# Out[3]: {'batch_size': 25, 'epochs': 500, 'optimizer': 'sgd'}

# best_accuracy
# Out[4]: 0.856125

# Tune the ANN hyperparameters v2 - three hidden layers and no Dropout 
# Grid Search recommends optimum hyperparameter configuration
# Add a timer
from timeit import default_timer as timer
start = timer()

from keras.wrappers.scikit_learn import KerasClassifier # Keras classifier wrapper for scikit learn
from sklearn.model_selection import GridSearchCV # grid search hyperparameter tuning function in sklearn
from keras.models import Sequential # Initializes the ANN
from keras.layers import Dense # Builds the layers of the ANN
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6,
                         kernel_initializer = 'uniform', 
                         activation = 'relu', 
                         input_dim = 11)) # activation is the rectifier activation function for the hidden layers
    classifier.add(Dense(units = 6, 
                         kernel_initializer = 'uniform', 
                         activation = 'relu')) # activation is the rectifier activation function for the hidden lyers
    classifier.add(Dense(units = 6, 
                         kernel_initializer = 'uniform', 
                         activation = 'relu')) # activation is the rectifier activation function for the hidden lyers
    classifier.add(Dense(units = 1, 
                         kernel_initializer = 'uniform', 
                         activation = 'sigmoid')) # activation is the sigmoid activation function for the output layer
    classifier.compile(optimizer = optimizer, 
                       loss = 'binary_crossentropy', # handles stochastic gradient descent 
                       metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop', 'sgd']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(x_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

# Elapsed time in minutes
end = timer()
print('Elapsed time in minutes: ')
print(0.1 * round((end - start) / 6))




