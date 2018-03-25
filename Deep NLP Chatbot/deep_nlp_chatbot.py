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

# First stage cleaning of the text: case, punctuation, and special characters
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", " will not", text)
    text = re.sub(r"can't", " cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text

# Clean the questions
clean_questions =[]
for question in questions:
    clean_questions.append(clean_text(question))

# Clean the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))  

# Create a dictionary that maps each word to its number of occurrences
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

# Create two dictionaries that map the question words and the answer words to a unique integer (tokenization and filtering)
threshold = 20 # ~5% of the max count
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        questionswords2int[word] = word_number
        word_number += 1

answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        answerswords2int[word] = word_number
        word_number += 1

# Add the last tokens to these two dictionaries
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1

for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1

# Create the inverse dictionary of the answerswords2int dictionary
answersints2word = {w_i: w for w, w_i in answerswords2int.items()}

# Add the EOS token to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

# Translate all questions and answers into integers and replace all the words that were wiltered out with <OUT>
questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)

answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)
    
# Sort questions and answers by the length of the questions to speed up the training
sorted_clean_questions = []
sorted_clean_anwers = []
for 

########## PART 2: BUILD THE SEQ2SEQ MODEL ##########
########## PART 3: TRAIN THE SEQ2SEQ MODEL ##########
########## PART 4: TEST THE SEQ2SEQ MODEL ##########