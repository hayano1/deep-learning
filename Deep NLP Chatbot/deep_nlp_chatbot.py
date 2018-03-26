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
sorted_clean_answers = []
for length in range(1, 25 + 1):
    for i in enumerate(questions_into_int): # enumerate the questions_into_int list 
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])

########## PART 2: BUILD THE SEQ2SEQ MODEL ##########
            
# Create placeholders for the inputs and targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input') # Inputs placeholder
    targets = tf.placeholder(tf.int32, [None, None], name = 'target') # Targets placeholder
    lr = tf.placeholder(tf.int32, [None, None], name = 'learning_rate') # Learning Rate placeholder
    keep_prob = tf.placeholder(tf.int32, [None, None], name = 'keep_prob') # Dropout Rate placeholder
    return inputs, targets, lr, keep_prob

# Preprocess the targets
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
    preprocessed_targets = tf.concat([left_side, right_side], axis = 1)
    return preprocessed_targets

# Create the Encoder RNN Layer
def encoder_rnn_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, 
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_inputs,
                                                       dtype = tf.float32) # Dynamic version of bidirectional rnn. Ensure input size of forward and backward cells are the same size (only encoder_state required)
    return encoder_state

# Decode the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0], 
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = 'attn_dec_train')
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)

# Decode the test/validation sets
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0], 
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = 'attn_dec_inf')
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions

# Create the Decoder RNN Layer
def decoder_rnn_layer(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope('decoding') as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1) # Initialize weights associated with the neurons of the fully connected layer
        biases = tf.zeros_initializer() # Biases of the fully connected layers
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None, 
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state, 
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<SOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions

# Build the SEQ2SEQ Model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn_layer(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn_layer(decoder_embedded_input,
                                                               decoder_embeddings_matrix,
                                                               encoder_state,
                                                               questions_num_words,
                                                               sequence_length,
                                                               rnn_size,
                                                               num_layers,
                                                               questionswords2int,
                                                               keep_prob,
                                                               batch_size)
    return training_predictions, test_predictions


########## PART 3: TRAIN THE SEQ2SEQ MODEL ##########
    
# Set the hyperparameters
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512 # Number of columns in encoder
decoding_embedding_size = 512 # Number of columns in decoder
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.50 # Dropout rate recommended by Geoffrey Hinton p.1933 of JMLRdropout.pdf

# Define a session
tf.reset_default_graph()
session = tf.InteractiveSession()

# Load the model inputs
inputs, targets, lr, keep_prob = model_inputs()

# Set the sequence length
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')

########## PART 4: TEST THE SEQ2SEQ MODEL ##########