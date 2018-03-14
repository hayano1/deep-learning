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
    # create a list of lists
