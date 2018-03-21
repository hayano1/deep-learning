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
criterion = nn.MSELoss() # Mean Squared Error (MSE)
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) # Use Stochastic Gradient Descent to update the different weights to reduce the error at each epoch (Adam, RMSprop)

# Add a timer
from timeit import default_timer as timer
start = timer()

# Train the Stacked AutoEncoder (SAE) Model
nb_epoch = 200
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
    print('epoch: ' + str(epoch) + ' training loss: ' + str(train_loss / s))

# Elapsed time in minutes
end = timer()
print('Elapsed time in minutes: ')
print(0.1 * round((end - start) / 6))

# Add an end of work message
import os
os.system('say "your model has finished training"')

# Test the Stacked AutoEncoder (SAE) Model
test_loss = 0
s = 0. # Counter (float) of number of users who provided a rating
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0) # The training_set is the input to the test_set
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) > 0:
        output = sae(input) 
        target.require_grad = False 
        output[target == 0] = 0 
        loss = criterion(output, target)
        mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10) 
        test_loss += np.sqrt(loss.data[0] * mean_corrector)
        s += 1.
print('test loss: ' + str(test_loss / s))

'''
epoch: 1 training loss: 1.7716442298095365
epoch: 2 training loss: 1.0967092028404766
epoch: 3 training loss: 1.0536122634939697
epoch: 4 training loss: 1.0383468372664748
epoch: 5 training loss: 1.0307633086254164
epoch: 6 training loss: 1.0263843626972722
epoch: 7 training loss: 1.0239616451386186
epoch: 8 training loss: 1.0219362814860762
epoch: 9 training loss: 1.020765236136667
epoch: 10 training loss: 1.0195002206633408
epoch: 11 training loss: 1.0190008317214945
epoch: 12 training loss: 1.0182511951567985
epoch: 13 training loss: 1.0177923951232855
epoch: 14 training loss: 1.0176488140564424
epoch: 15 training loss: 1.0171683736476707
epoch: 16 training loss: 1.016917641557965
epoch: 17 training loss: 1.0169067588475973
epoch: 18 training loss: 1.0165988190619615
epoch: 19 training loss: 1.0161595482451664
epoch: 20 training loss: 1.0161831098402005
epoch: 21 training loss: 1.015938151953662
epoch: 22 training loss: 1.0159984114400087
epoch: 23 training loss: 1.0158912945432468
epoch: 24 training loss: 1.0159862232854409
epoch: 25 training loss: 1.0156777045411707
epoch: 26 training loss: 1.0156502880249931
epoch: 27 training loss: 1.0151186400211927
epoch: 28 training loss: 1.014986841475755
epoch: 29 training loss: 1.01290534082313
epoch: 30 training loss: 1.0110502105697154
epoch: 31 training loss: 1.009130479832253
epoch: 32 training loss: 1.0067189586348195
epoch: 33 training loss: 1.0071299031544492
epoch: 34 training loss: 1.003058957667865
epoch: 35 training loss: 1.0018825895540324
epoch: 36 training loss: 0.9998880361796728
epoch: 37 training loss: 0.999218867865085
epoch: 38 training loss: 0.9947154741706842
epoch: 39 training loss: 0.9945438598766192
epoch: 40 training loss: 0.9911347913241144
epoch: 41 training loss: 0.9917201986695585
epoch: 42 training loss: 0.9880103961731581
epoch: 43 training loss: 0.9869939016556867
epoch: 44 training loss: 0.9848759954147834
epoch: 45 training loss: 0.9826445365145768
epoch: 46 training loss: 0.9785247426488182
epoch: 47 training loss: 0.9796689933576174
epoch: 48 training loss: 0.9739135938801404
epoch: 49 training loss: 0.9733971692533466
epoch: 50 training loss: 0.9695294296055694
epoch: 51 training loss: 0.9671899302713485
epoch: 52 training loss: 0.9663724136115869
epoch: 53 training loss: 0.9643428420315745
epoch: 54 training loss: 0.9638057024770981
epoch: 55 training loss: 0.9587156379554691
epoch: 56 training loss: 0.957792837086354
epoch: 57 training loss: 0.9589190291470372
epoch: 58 training loss: 0.9555535550068867
epoch: 59 training loss: 0.9563380129433167
epoch: 60 training loss: 0.952969429976086
epoch: 61 training loss: 0.9526122401092924
epoch: 62 training loss: 0.9503661647316789
epoch: 63 training loss: 0.9497967877528576
epoch: 64 training loss: 0.9464912427629297
epoch: 65 training loss: 0.9468574139166557
epoch: 66 training loss: 0.9461856620490885
epoch: 67 training loss: 0.9468756950334646
epoch: 68 training loss: 0.9464173377123787
epoch: 69 training loss: 0.9476481175422086
epoch: 70 training loss: 0.9481345306848687
epoch: 71 training loss: 0.9485907774419845
epoch: 72 training loss: 0.944810786320882
epoch: 73 training loss: 0.9462957116392428
epoch: 74 training loss: 0.9428954631650336
epoch: 75 training loss: 0.9434404698917563
epoch: 76 training loss: 0.9425027626843427
epoch: 77 training loss: 0.9468171240251935
epoch: 78 training loss: 0.9460276204784362
epoch: 79 training loss: 0.9438001871444018
epoch: 80 training loss: 0.9412012895821771
epoch: 81 training loss: 0.9409175078083314
epoch: 82 training loss: 0.9399169677631278
epoch: 83 training loss: 0.940028663360646
epoch: 84 training loss: 0.9388469542747689
epoch: 85 training loss: 0.9383596301641032
epoch: 86 training loss: 0.9380769825297399
epoch: 87 training loss: 0.9377934369816278
epoch: 88 training loss: 0.9361967505727112
epoch: 89 training loss: 0.9361812660012876
epoch: 90 training loss: 0.9364500523834247
epoch: 91 training loss: 0.935952385571004
epoch: 92 training loss: 0.9352484012134292
epoch: 93 training loss: 0.9353370361781569
epoch: 94 training loss: 0.9340377333230055
epoch: 95 training loss: 0.9340464317474739
epoch: 96 training loss: 0.9330448724823434
epoch: 97 training loss: 0.9330631026360404
epoch: 98 training loss: 0.9329678706613892
epoch: 99 training loss: 0.9327095033207395
epoch: 100 training loss: 0.9317003433246174
epoch: 101 training loss: 0.9322239680966106
epoch: 102 training loss: 0.9312642452201281
epoch: 103 training loss: 0.9314323992823486
epoch: 104 training loss: 0.9303349806162945
epoch: 105 training loss: 0.9306609959848147
epoch: 106 training loss: 0.930101202617221
epoch: 107 training loss: 0.9302175185203823
epoch: 108 training loss: 0.9291620795551867
epoch: 109 training loss: 0.9297338599273766
epoch: 110 training loss: 0.928426153221447
epoch: 111 training loss: 0.9283811311266212
epoch: 112 training loss: 0.928037245725236
epoch: 113 training loss: 0.9284948004553284
epoch: 114 training loss: 0.9276338822491488
epoch: 115 training loss: 0.9275588381269633
epoch: 116 training loss: 0.9268138390238465
epoch: 117 training loss: 0.9272460189373707
epoch: 118 training loss: 0.9264197102340116
epoch: 119 training loss: 0.9263792050707553
epoch: 120 training loss: 0.9256019886667766
epoch: 121 training loss: 0.9257736365405856
epoch: 122 training loss: 0.9250977498614954
epoch: 123 training loss: 0.9253803718829846
epoch: 124 training loss: 0.9247051088823675
epoch: 125 training loss: 0.9244797616367397
epoch: 126 training loss: 0.923928172054627
epoch: 127 training loss: 0.9239734001825102
epoch: 128 training loss: 0.9232552173974182
epoch: 129 training loss: 0.9232903278081867
epoch: 130 training loss: 0.9226474883252654
epoch: 131 training loss: 0.9228550970032
epoch: 132 training loss: 0.922262507274932
epoch: 133 training loss: 0.9225579974519074
epoch: 134 training loss: 0.921834938520119
epoch: 135 training loss: 0.9222363679866538
epoch: 136 training loss: 0.9213307711118274
epoch: 137 training loss: 0.921801269606442
epoch: 138 training loss: 0.9206753541046993
epoch: 139 training loss: 0.9210576377298592
epoch: 140 training loss: 0.9204500425811952
epoch: 141 training loss: 0.9210521396802144
epoch: 142 training loss: 0.920043082054395
epoch: 143 training loss: 0.9204231678884549
epoch: 144 training loss: 0.9192632672470638
epoch: 145 training loss: 0.9195538055344273
epoch: 146 training loss: 0.918814688268977
epoch: 147 training loss: 0.919470153691655
epoch: 148 training loss: 0.9187561335107272
epoch: 149 training loss: 0.919318362626889
epoch: 150 training loss: 0.9185854468514912
epoch: 151 training loss: 0.9193426563186495
epoch: 152 training loss: 0.9184444378691048
epoch: 153 training loss: 0.9188102218525395
epoch: 154 training loss: 0.9177508206169368
epoch: 155 training loss: 0.9183092100918385
epoch: 156 training loss: 0.9176000144562961
epoch: 157 training loss: 0.9178764064099192
epoch: 158 training loss: 0.9170891370040137
epoch: 159 training loss: 0.9174279890571942
epoch: 160 training loss: 0.9168138590872683
epoch: 161 training loss: 0.9175216563711729
epoch: 162 training loss: 0.9165651240309536
epoch: 163 training loss: 0.9170937675368025
epoch: 164 training loss: 0.9161348679318732
epoch: 165 training loss: 0.9167274009638609
epoch: 166 training loss: 0.9156982650882706
epoch: 167 training loss: 0.9164552032328761
epoch: 168 training loss: 0.9154544314279809
epoch: 169 training loss: 0.9159920045502691
epoch: 170 training loss: 0.9154468717419293
epoch: 171 training loss: 0.9158179808708482
epoch: 172 training loss: 0.9149690364748196
epoch: 173 training loss: 0.9155897820390946
epoch: 174 training loss: 0.9151813868395863
epoch: 175 training loss: 0.9156585301532993
epoch: 176 training loss: 0.9147104996478519
epoch: 177 training loss: 0.9151700929547756
epoch: 178 training loss: 0.9143664236805181
epoch: 179 training loss: 0.9148426676212599
epoch: 180 training loss: 0.9141707955026743
epoch: 181 training loss: 0.9145552597123991
epoch: 182 training loss: 0.9136425373548678
epoch: 183 training loss: 0.9141738224374645
epoch: 184 training loss: 0.9133968214614422
epoch: 185 training loss: 0.9138041345357708
epoch: 186 training loss: 0.9130700368300568
epoch: 187 training loss: 0.9138224672240671
epoch: 188 training loss: 0.9126548070868725
epoch: 189 training loss: 0.9135240409126764
epoch: 190 training loss: 0.9126008294758227
epoch: 191 training loss: 0.9132850378257716
epoch: 192 training loss: 0.9123174110425011
epoch: 193 training loss: 0.9128880438586928
epoch: 194 training loss: 0.9120279045244784
epoch: 195 training loss: 0.9126871424686274
epoch: 196 training loss: 0.9117115807985308
epoch: 197 training loss: 0.9122879286268558
epoch: 198 training loss: 0.9116302036736874
epoch: 199 training loss: 0.9122160300877563
epoch: 200 training loss: 0.9116175453515095
Elapsed time in minutes: 
5.300000000000001
test loss: 0.949453297628295
'''