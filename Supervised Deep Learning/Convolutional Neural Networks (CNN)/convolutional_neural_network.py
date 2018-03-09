#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 07:03:37 2018

@author: ngilmore
"""

# Deep Learning: Convolutional Neural Network (CNN) in Python

#%reset -f

# Install Theano library in the terminal
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Install Tensorflow library in the terminal
# Install Tensorflow from https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html

# Install Keras library in the terminal
# pip install --upgrade keras

# Part 1: Build the Convolutional Neural Network (CNN)

# Steps to Build a Convolutional Neural Network (CNN)
# Step 1: Convolution
# Step 2: Max Pooling
# Step 3: Flattening
# Step 4: Full Connection (to ANN)

# Import needed libraries
from keras.models import Sequential # Initializes the Neural Network
from keras.layers import Convolution2D # Creates the convolutional layers
from keras.layers import MaxPooling2D # Adds the pooling layers
from keras.layers import Flatten # Adds the flattening layer
from keras.layers import Dense # Builds the layers of the Neural Network

# Add a timer
from timeit import default_timer as timer
start = timer()

# Initialize the Convolutional Neural Network (CNN)
classifier = Sequential()

# Step 1: Add the Convolutional Layer
# Apply Feature Detectors to Input Image = Feature Map
classifier.add(Convolution2D(32, (3, 3), 
                             padding = 'same', 
                             input_shape = (64, 64, 3), 
                             activation = 'relu'))

# Step 2: Apply Max Pooling to the CNN
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Add a 2nd Convolutional Layer to improve accuracy and performance results
classifier.add(Convolution2D(32, (3, 3), 
                             padding = 'same',
                             activation = 'relu'))

# Apply Max Pooling to the 2nd Convolutional Layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Add a 3rd Convolutional Layer to improve accuracy and performance results
classifier.add(Convolution2D(64, (3, 3), 
                             padding = 'same',
                             activation = 'relu'))

# Apply Max Pooling to the 3rd Convolutional Layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3: Flatten the CNN
classifier.add(Flatten())

# Step 4: Create a Full Connection Artificial Neural Network
# Add the Input Layer and the first Hidden Layer
# Tip: select the average number of input and output nodes as the number of nodes in the hidden layer as a quick way to determine the number of nodes
classifier.add(Dense(units = 128, 
                     kernel_initializer = 'uniform', 
                     activation = 'relu')) # activation is the rectifier activation function for the hidden layers

# Add a second Full Connection Hidden Layer to increase accuracy and performance results
#classifier.add(Dense(units = 64, 
#                     kernel_initializer = 'uniform', 
#                     activation = 'relu')) # activation is the rectifier activation function for the hidden layers

# Add the Output Layer
classifier.add(Dense(units = 1, 
                     kernel_initializer = 'uniform', 
                     activation = 'sigmoid')) # activation is the sigmoid activation function for the output layer

# Compile the Convolutional Neural Network (CNN)
classifier.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])

# Part 2: Fit the Convolutional Neural Network (CNN) to the images
from keras.preprocessing.image import ImageDataGenerator

# Apply image augmentation on the provided images (reference: keras documentation) to enrich the training_set images to reduce overfitting
train_datagen = ImageDataGenerator(
        rescale = 1./255, # rescale pixels between 0 and 1
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('cnn_dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('cnn_dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)

# Elapsed time in minutes
end = timer()
print('Elapsed time in minutes: ')
print(0.1 * round((end - start) / 6))

# Add an end of work message
import os
os.system('say "your program has finished"')

# Results 
# =============================================================================
# =============================================================================
# Using TensorFlow backend.
# Found 8000 images belonging to 2 classes.
# Found 2000 images belonging to 2 classes.
# Epoch 1/25
# 8000/8000 [==============================] - 2449s 306ms/step - loss: 0.3935 - acc: 0.8109 - val_loss: 0.6193 - val_acc: 0.7751
# Epoch 2/25
# 8000/8000 [==============================] - 2461s 308ms/step - loss: 0.1214 - acc: 0.9538 - val_loss: 0.9379 - val_acc: 0.7992
# Epoch 3/25
# 8000/8000 [==============================] - 2464s 308ms/step - loss: 0.0587 - acc: 0.9791 - val_loss: 1.2442 - val_acc: 0.7825
# Epoch 4/25
# 8000/8000 [==============================] - 2462s 308ms/step - loss: 0.0398 - acc: 0.9859 - val_loss: 1.3576 - val_acc: 0.7843
# Epoch 5/25
# 8000/8000 [==============================] - 2435s 304ms/step - loss: 0.0328 - acc: 0.9888 - val_loss: 1.3713 - val_acc: 0.7795
# Epoch 6/25
# 8000/8000 [==============================] - 2456s 307ms/step - loss: 0.0270 - acc: 0.9910 - val_loss: 1.4536 - val_acc: 0.7753
# Epoch 7/25
# 8000/8000 [==============================] - 2441s 305ms/step - loss: 0.0234 - acc: 0.9923 - val_loss: 1.5741 - val_acc: 0.7851
# Epoch 8/25
# 8000/8000 [==============================] - 2439s 305ms/step - loss: 0.0209 - acc: 0.9935 - val_loss: 1.5551 - val_acc: 0.7865
# Epoch 9/25
# 8000/8000 [==============================] - 2427s 303ms/step - loss: 0.0189 - acc: 0.9939 - val_loss: 1.6450 - val_acc: 0.7739
# Epoch 10/25
# 8000/8000 [==============================] - 2397s 300ms/step - loss: 0.0171 - acc: 0.9945 - val_loss: 1.6073 - val_acc: 0.7804
# Epoch 11/25
# 8000/8000 [==============================] - 2433s 304ms/step - loss: 0.0145 - acc: 0.9953 - val_loss: 1.6377 - val_acc: 0.7886
# Epoch 12/25
# 8000/8000 [==============================] - 2386s 298ms/step - loss: 0.0137 - acc: 0.9956 - val_loss: 1.6775 - val_acc: 0.7853
# Epoch 13/25
# 8000/8000 [==============================] - 2460s 308ms/step - loss: 0.0131 - acc: 0.9957 - val_loss: 1.6811 - val_acc: 0.7858
# Epoch 14/25
# 8000/8000 [==============================] - 2433s 304ms/step - loss: 0.0119 - acc: 0.9963 - val_loss: 1.6190 - val_acc: 0.7821
# Epoch 15/25
# 8000/8000 [==============================] - 2432s 304ms/step - loss: 0.0124 - acc: 0.9963 - val_loss: 1.6974 - val_acc: 0.7805
# Epoch 16/25
# 8000/8000 [==============================] - 2421s 303ms/step - loss: 0.0118 - acc: 0.9965 - val_loss: 1.6775 - val_acc: 0.7866
# Epoch 17/25
# 8000/8000 [==============================] - 2426s 303ms/step - loss: 0.0105 - acc: 0.9969 - val_loss: 1.7370 - val_acc: 0.7903
# Epoch 18/25
# 8000/8000 [==============================] - 2421s 303ms/step - loss: 0.0111 - acc: 0.9966 - val_loss: 1.7607 - val_acc: 0.7818
# Epoch 19/25
# 8000/8000 [==============================] - 2436s 304ms/step - loss: 0.0099 - acc: 0.9970 - val_loss: 1.8596 - val_acc: 0.7821
# Epoch 20/25
# 8000/8000 [==============================] - 2418s 302ms/step - loss: 0.0098 - acc: 0.9971 - val_loss: 1.7806 - val_acc: 0.7919
# Epoch 21/25
# 8000/8000 [==============================] - 2421s 303ms/step - loss: 0.0094 - acc: 0.9972 - val_loss: 1.8604 - val_acc: 0.7772
# Epoch 22/25
# 8000/8000 [==============================] - 2421s 303ms/step - loss: 0.0082 - acc: 0.9974 - val_loss: 1.7844 - val_acc: 0.7941
# Epoch 23/25
# 8000/8000 [==============================] - 2437s 305ms/step - loss: 0.0084 - acc: 0.9975 - val_loss: 1.9289 - val_acc: 0.7868
# Epoch 24/25
# 8000/8000 [==============================] - 2465s 308ms/step - loss: 0.0088 - acc: 0.9974 - val_loss: 1.9441 - val_acc: 0.7934
# Epoch 25/25
# 8000/8000 [==============================] - 2465s 308ms/step - loss: 0.0077 - acc: 0.9978 - val_loss: 1.9185 - val_acc: 0.7882
# Out[1]: <keras.callbacks.History at 0x1825331b70>
# =============================================================================
# =============================================================================
