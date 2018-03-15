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

# Convert the Ratings into binary ratings 1 (Liked) or 0 (Not Liked) - binary ratings required by Restricted Boltzmann Machine (RBM) as input type and output type must match 
training_set[training_set == 0] = -1 # Set all unrated movies to -1
training_set[training_set == 1] = 0 # Set all movies rated 1 or 2 to 0
training_set[training_set == 2] = 0 # Set all movies rated 1 or 2 to 0
training_set[training_set >= 3] = 1 # Set all movies rated 3, 4, or 5 to 1

test_set[test_set == 0] = -1 # Set all unrated movies to -1
test_set[test_set == 1] = 0 # Set all movies rated 1 or 2 to 0
test_set[test_set == 2] = 0 # Set all movies rated 1 or 2 to 0
test_set[test_set >= 3] = 1 # Set all movies rated 3, 4, or 5 to 1

# Create the architecture of the Neural Network (Restricted Boltzmann Machine (RBM))
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv) # Weights set on a normal distribution of mean 0 and variance of 1
        self.a = torch.randn(1, nh) # Bias (a) for the hidden nodes of size of the number of hidden nodes and size of the batch (1)
        self.b = torch.randn(1, nv) # Bias (b) for the visible nodes of size of the number of visible nodes and size of the batch (1)
    def sample_h(self, x):
        # Probability of h given v (sigmoid activation function)
        wx = torch.mm(x, self.W.t()) # Product of the visible node and the matrix of weights
        activation = wx + self.a.expand_as(wx) # Ensure the bias is applied to each line of the mini batch
        p_h_given_v = torch.sigmoid(activation) # Vector of probabilities that each hidden node is activated given the value of the visible node
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        # Probability of v given h (sigmoid activation function)
        wy = torch.mm(y, self.W) # Product of the hidden node and the matrix of weights
        activation = wy + self.b.expand_as(wy) # Ensure the bias is applied to each line of the mini batch
        p_v_given_h = torch.sigmoid(activation) # Vector of probabilities that each visible node is activated given the value of the hidden node
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk): # Implementation of k-step contrastive divergence
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0) # Keep format of b as a tensor with 2 dimensions
        self.a += torch.sum((ph0 - phk), 0) # Keep format of a as a tensor with 2 dimensions
   
# Set RBM Model input parameters     
nv = len(training_set[0])
nh = 100 # Adjust with tunning / experience
batch_size = 100 # Batch size (1 is Online or Reinforcement Learning)

# Instantiate the RBM model
rbm = RBM(nv, nh) 

# Add a timer
from timeit import default_timer as timer
start = timer()

# Train the RBM Model
nb_epoch = 10 
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0. # Counter (float)
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user + batch_size] # Input batch vk 
        v0 = training_set[id_user:id_user + batch_size]
        ph0,_ = rbm.sample_h(v0) # Returns the first element of the sample_h function to initialize probabilities is based on the ratings already given by the users
        for k in range(10):
            _,hk = rbm.sample_h(vk) # Returns the second element
            _,vk = rbm.sample_v(hk) # Update vk
            vk[v0 < 0] = v0[v0 < 0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk) # Update the weights
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0])) # Update the train loss based on existant ratings
        s += 1. # Increment the counter
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s))

# Elapsed time in minutes
end = timer()
print('Elapsed time in minutes: ')
print(0.1 * round((end - start) / 6))

# Add an end of work message
import os
os.system('say "your model has finished training"')