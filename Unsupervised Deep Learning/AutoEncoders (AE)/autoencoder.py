#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 11:35:24 2018

@author: ngilmore
"""

# Deep Learning: AutoEncoder in Python

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

# Prepare the Training and Test sets
training_set = pd.read_csv('ml-100k/u1.base',
                     delimiter = '\t')
training_set = np.array(training_set, 
                        dtype = int)

test_set = pd.read_csv('ml-100k/u1.test',
                     delimiter = '\t')
test_set = np.array(test_set, 
                    dtype = int)

# Get the number of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0]))) # The maximum user ID in either the training or test set
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1]))) # The maximum movie ID in either the training or test set

# Convert the data into an array with users in lines and movies in columns
def convert(data):
    new_data = [] # create a list of lists
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings # for each user, add the ratings for all movies
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Convert the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Create the architecture of the Neural Network (Stacked AutoEncoder (SAE))
class SAE(nn.Module): # Create child class of torch.nn.Module class
    def __init__(self, ):
        super(SAE, self).__init__() # Inherit from torch.nn.Module all functions
        self.fc1 = nn.Linear(nb_movies, 20) # First full connection hidden layer, # of nodes in first hidden layer (20) was found through trial and error
        self.fc2 = nn.Linear(20, 10) # Second full connection hidden layer based on first hidden layer
        self.fc3 = nn.Linear(10, 20) # Third full connection hidden layer based on second hidden layer
        self.fc4 = nn.Linear(20, nb_movies) # Output layer has same size as first connection hidden layer
        self.activation = nn.Sigmoid() # Activate the network - choice between rectifier and sigmoid done by trial and error
    def forward(self, x): # 
        x = self.activation(self.fc1(x)) # First encoding vector
        x = self.activation(self.fc2(x)) # Second encoding vector
        x = self.activation(self.fc3(x)) # First decoding vector
        x = self.fc4(x) # Output vector
        return x # Vector of predicted ratings

# Instantiate the Stacked AutoEncoder (SAE) Model
sae = SAE()
criterion = nn.MSELoss() # Mean Squared Error
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) # Use Stochastic Gradient Descent to update the different weights to reduce the error at each epoch (Adam, RMSprop)

# Add a timer
from timeit import default_timer as timer
start = timer()

# Train the Stacked AutoEncoder (SAE) Model
nb_epoch = 100
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0. # Counter (float) of number of users who provided a rating
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0) # Creates Input Batch required by PyTorch
        target = input.clone()
        if torch.sum(target.data > 0) > 0: # Only look at users with at least one rating
            output = sae(input) # Instantiates forward() function and outputs a vector of predicted ratings
            target.require_grad = False # Apply stochastic gradient descent only to the inputs not the target to optimize the code
            output[target == 0] = 0 # Take the same indexes as the input vector so that non-ratings will not count to optimize the code
            loss = criterion(output, target)
            mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10) # Average of the error for movies that were rated (1-5) and add 1e-10 to ensure non-NULL denominator
            loss.backward() # Determines which direction to adjust the weights up or down
            train_loss += np.sqrt(loss.data[0] * mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s))

# Elapsed time in minutes
end = timer()
print('Elapsed time in minutes: ')
print(0.1 * round((end - start) / 6))

# Add an end of work message
import os
os.system('say "your model has finished training"')

# Test the AutoEncoder Model
