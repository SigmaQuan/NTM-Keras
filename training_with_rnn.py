# -*- coding: utf-8 -*-
'''An implementation of learning copying algorithm with RNN (basic RNN, LSTM,
GRU).
Input sequence length: "1 ~ 20: (1*2+1)=3 ~ (20*2+1)=41"
Input dimension: "8"
Output sequence length: equal to input sequence length.
Output dimension: equal to input dimension.
'''

from __future__ import print_function
from keras.models import Sequential
# from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
import numpy as np
# from six.moves import range

import dataset                               # Add by Steven Robot
import visualization                         # Add by Steven
from keras.utils.visualize_util import plot  # Add by Steven

# Parameters for the model to train copying algorithm
###
# TRAINING_SIZE = 128000
TRAINING_SIZE = 1280
INPUT_DIMENSION_SIZE = 8 + 1
MAX_COPY_LENGTH = 20
MAX_INPUT_LENGTH = MAX_COPY_LENGTH + 1 + MAX_COPY_LENGTH

# Try replacing SimpleRNN, GRU, or LSTM
# RNN = recurrent.SimpleRNN
# RNN = recurrent.GRU
RNN = recurrent.LSTM
HIDDEN_SIZE = 128
LAYERS = 1
BATCH_SIZE = 128

# print('Generating data...')
# input_sequence, output_sequence = \
#     dataset.generate_copy_data(INPUT_DIMENSION_SIZE, MAX_INPUT_LENGTH)
# print(input_sequence.shape)
# print(input_sequence)
# print(output_sequence)
# visualization.show_copy_data(input_sequence.transpose(),
#                              output_sequence.transpose(),
#                              "experiment/copy_data_sample.png")
print('Generating data sets...')
# train, valid, test = dataset.generate_copy_data_sets(
#     INPUT_DIMENSION_SIZE, MAX_COPY_LENGTH, TRAINING_SIZE)
train_X, train_Y = dataset.generate_copy_data_set(
    INPUT_DIMENSION_SIZE, MAX_COPY_LENGTH, TRAINING_SIZE)
valid_X, valid_Y = dataset.generate_copy_data_set(
    INPUT_DIMENSION_SIZE, MAX_COPY_LENGTH, TRAINING_SIZE/10)

# print("Train")
# # for i in range(BATCH_SIZE):
# #     print(train_X[i])
# print(train_X)
# print()
# print()
# print()
# print("Valid")
# print(valid_X)
# # for i in range(BATCH_SIZE):
# #     print(valid_X[i])


print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
hidden_layer = RNN(
    HIDDEN_SIZE,
    input_shape=(MAX_INPUT_LENGTH, INPUT_DIMENSION_SIZE),
    init='glorot_uniform',
    inner_init='orthogonal',
    activation='tanh',
    W_regularizer=None,
    U_regularizer=None,
    b_regularizer=None,
    dropout_W=0.0,
    dropout_U=0.0)
model.add(hidden_layer)
# For the decoder's input, we repeat the encoded input for each time step
model.add(RepeatVector(MAX_INPUT_LENGTH))
# The decoder RNN could be multiple layers stacked or a single layer
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# For each of step of the output sequence, decide which character should be chosen
model.add(TimeDistributed(Dense(INPUT_DIMENSION_SIZE)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("Model architecture")
plot(model, show_shapes=True, to_file="experiment/model_simple_rnn_for_copying.png")


print("Training...")
# Train the model each generation and show predictions against the
# validation dataset
for iteration in range(1, 200):
    print()
    print('-' * 78)
    print('Iteration', iteration)
    model.fit(train_X,
              train_Y,
              batch_size=BATCH_SIZE,
              nb_epoch=1,
              validation_data=(valid_X, valid_Y))
    ###
    # Select 3 samples from the validation set at random so we can
    # visualize errors
    for i in range(1):
        ind = np.random.randint(0, len(valid_X))
        # inputs = valid_X[ind]
        # outputs = valid_Y[ind]
        inputs, outputs = valid_X[np.array([ind])], valid_Y[np.array([ind])]
        predicts = model.predict(inputs, verbose=0)
        # print(inputs)
        # print(outputs)
        # print(predicts)
        visualization.show_copy_data(outputs[0].transpose(),
                                     predicts[0].transpose(),
                                     "Target",
                                     "Prediction",
                                     "experiment/copy_data_predict_%3d.png"%iteration)
