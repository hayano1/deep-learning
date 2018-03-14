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
    
        



