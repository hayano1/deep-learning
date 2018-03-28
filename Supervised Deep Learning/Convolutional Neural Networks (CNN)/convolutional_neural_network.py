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
from keras.layers import Dropout # Adds dropout

# Add a timer
from timeit import default_timer as timer
start = timer()

# Initialize the Convolutional Neural Network (CNN)
classifier = Sequential()

# Step 1: Add the Convolutional Layer
# Apply Feature Detectors to Input Image = Feature Map
classifier.add(Convolution2D(32, (3, 3), 
                             padding = 'same', 
                             input_shape = (128, 128, 3), 
                             activation = 'relu'))
classifier.add(Dropout(p = 0.1))

# Step 2: Apply Max Pooling to the CNN
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Add a 2nd Convolutional Layer to improve accuracy and performance results
classifier.add(Convolution2D(32, (3, 3), 
                             padding = 'same',
                             activation = 'relu'))
classifier.add(Dropout(p = 0.1))

# Apply Max Pooling to the 2nd Convolutional Layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Add a 3rd Convolutional Layer to improve accuracy and performance results
classifier.add(Convolution2D(64, (3, 3), 
                             padding = 'same',
                             activation = 'relu'))
classifier.add(Dropout(p = 0.1))

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
classifier.add(Dropout(p = 0.1))

# Add a second Full Connection Hidden Layer to increase accuracy and performance results
classifier.add(Dense(units = 128, 
                     kernel_initializer = 'uniform', 
                     activation = 'relu')) # activation is the rectifier activation function for the hidden layers
classifier.add(Dropout(p = 0.1))

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
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('cnn_dataset/test_set',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'binary')

batch_size = 32
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000 // batch_size,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000 // batch_size)

# Elapsed time in minutes
end = timer()
print('Elapsed time in minutes: ')
print(0.1 * round((end - start) / 6))

# Add an end of work message
import os
os.system('say "your program has finished"')

# Results 
# =============================================================================
'''
Epoch 1/25
8000/8000 [==============================] - 8209s 1s/step - loss: 0.3101 - acc: 0.8570 - val_loss: 0.5589 - val_acc: 0.8440
Epoch 2/25
8000/8000 [==============================] - 8254s 1s/step - loss: 0.0793 - acc: 0.9709 - val_loss: 0.7584 - val_acc: 0.8393
Epoch 3/25
8000/8000 [==============================] - 8047s 1s/step - loss: 0.0410 - acc: 0.9859 - val_loss: 0.9707 - val_acc: 0.8429
Epoch 4/25
8000/8000 [==============================] - 8020s 1s/step - loss: 0.0290 - acc: 0.9902 - val_loss: 0.9424 - val_acc: 0.8470
Epoch 5/25
8000/8000 [==============================] - 8020s 1s/step - loss: 0.0236 - acc: 0.9924 - val_loss: 1.0720 - val_acc: 0.8360
Epoch 6/25
8000/8000 [==============================] - 8021s 1s/step - loss: 0.0200 - acc: 0.9935 - val_loss: 1.0179 - val_acc: 0.8359
Epoch 7/25
8000/8000 [==============================] - 8042s 1s/step - loss: 0.0167 - acc: 0.9946 - val_loss: 1.0515 - val_acc: 0.8400
Epoch 8/25
8000/8000 [==============================] - 8107s 1s/step - loss: 0.0147 - acc: 0.9955 - val_loss: 1.1569 - val_acc: 0.8280
Epoch 9/25
8000/8000 [==============================] - 8064s 1s/step - loss: 0.0144 - acc: 0.9956 - val_loss: 1.1177 - val_acc: 0.8445
Epoch 10/25
8000/8000 [==============================] - 8023s 1s/step - loss: 0.0130 - acc: 0.9958 - val_loss: 1.1270 - val_acc: 0.8423
Epoch 11/25
8000/8000 [==============================] - 8019s 1s/step - loss: 0.0125 - acc: 0.9962 - val_loss: 1.1919 - val_acc: 0.8479
Epoch 12/25
8000/8000 [==============================] - 8057s 1s/step - loss: 0.0115 - acc: 0.9965 - val_loss: 1.2285 - val_acc: 0.8492
Epoch 13/25
8000/8000 [==============================] - 8029s 1s/step - loss: 0.0119 - acc: 0.9965 - val_loss: 1.1494 - val_acc: 0.8510
Epoch 14/25
8000/8000 [==============================] - 7981s 998ms/step - loss: 0.0111 - acc: 0.9968 - val_loss: 1.2161 - val_acc: 0.8483
Epoch 15/25
8000/8000 [==============================] - 7967s 996ms/step - loss: 0.0111 - acc: 0.9970 - val_loss: 1.3651 - val_acc: 0.8569
Epoch 16/25
8000/8000 [==============================] - 7970s 996ms/step - loss: 0.0117 - acc: 0.9968 - val_loss: 1.3811 - val_acc: 0.8479
Epoch 17/25
8000/8000 [==============================] - 7973s 997ms/step - loss: 0.0113 - acc: 0.9969 - val_loss: 1.4357 - val_acc: 0.8440
Epoch 18/25
8000/8000 [==============================] - 7965s 996ms/step - loss: 0.0112 - acc: 0.9972 - val_loss: 1.3357 - val_acc: 0.8469
Epoch 19/25
8000/8000 [==============================] - 7963s 995ms/step - loss: 0.0114 - acc: 0.9971 - val_loss: 1.4432 - val_acc: 0.8403
Epoch 20/25
8000/8000 [==============================] - 7962s 995ms/step - loss: 0.0111 - acc: 0.9973 - val_loss: 1.5387 - val_acc: 0.8430
Epoch 21/25
8000/8000 [==============================] - 7971s 996ms/step - loss: 0.0102 - acc: 0.9973 - val_loss: 1.4371 - val_acc: 0.8544
Epoch 22/25
8000/8000 [==============================] - 8015s 1s/step - loss: 0.0118 - acc: 0.9973 - val_loss: 1.6304 - val_acc: 0.8418
Epoch 23/25
8000/8000 [==============================] - 8022s 1s/step - loss: 0.0110 - acc: 0.9975 - val_loss: 1.4144 - val_acc: 0.8549
Epoch 24/25
8000/8000 [==============================] - 7995s 999ms/step - loss: 0.0121 - acc: 0.9974 - val_loss: 1.6801 - val_acc: 0.8354
Epoch 25/25
8000/8000 [==============================] - 7984s 998ms/step - loss: 0.0124 - acc: 0.9973 - val_loss: 1.6786 - val_acc: 0.8259
Elapsed time in minutes: 
3344.9
---
'''

# Part 3: Make predictions based on the model
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('cnn_dataset/single_prediction/cat_or_dog_2.jpeg', 
                              target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, 0) # add 4th dimension (batch size) so that ANN can process

result = classifier.predict(test_image)

training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

prediction