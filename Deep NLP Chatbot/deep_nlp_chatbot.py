#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 07:24:15 2018

@author: ngilmore
"""

# Deep NLP Chatbot using Tensorflow

# Part 0: Environment Setup
# Create virtual environment in Terminal:
# sudo conda create -n chatbot python=3.5 anaconda
# source activate chatbot
# sudo pip install tensorflow==1.0.0

# Get dataset
# Google "Cornell Movie Dialogs Corpus"
# Download the zip file and extract to required folder
# Select the movie_conversations.txt and movie_lines.txt files

# Open Anaconda
# Applications on --> chatbot
# Launch Spyder from here

########## PART 1: DATA PREPROCESSING ##########

#%reset -f

# Import required libraries
import numpy as np
import tensorflow as tf
import re # Text cleaning library
import time

# Import the dataset
lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

# Create a dictionary mapping lines to ids
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

# Create a list of all conversations
conversations_ids = []
for conversation in conversations[: -1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "") # split the dataset and remove the square brackets, the quotes, and spaces
    conversations_ids.append(_conversation.split(','))

# Separate questions and answers
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i + 1]])

# First stage cleaning of the text: case, punctuation...
def clean_text(text):
    

########## PART 2: BUILD THE SEQ2SEQ MODEL ##########
########## PART 3: TRAIN THE SEQ2SEQ MODEL ##########
########## PART 4: TEST THE SEQ2SEQ MODEL ##########