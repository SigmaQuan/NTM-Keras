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
import time                                  # Add by Steven Robot
from util import LossHistory                 # Add by Steven Robot


# Parameters for the model to train copying algorithm
TRAINING_SIZE = 1024000         # for 8-bits length
# TRAINING_SIZE = 128000         # for 4-bits length
# TRAINING_SIZE = 1280
# INPUT_DIMENSION_SIZE = 4 + 1   # for 4-bits length
# MAX_COPY_LENGTH = 10           # for 4-bits length
INPUT_DIMENSION_SIZE = 8 + 1    # for 8-bits length
MAX_COPY_LENGTH = 20            # for 8-bits length
MAX_INPUT_LENGTH = MAX_COPY_LENGTH + 1 + MAX_COPY_LENGTH

# Try replacing SimpleRNN, GRU, or LSTM
# RNN = recurrent.SimpleRNN
# RNN = recurrent.GRU
RNN = recurrent.LSTM
# HIDDEN_SIZE = 128
HIDDEN_SIZE = 128*4
LAYERS = 1
# BATCH_SIZE = 2048
BATCH_SIZE = 1024
# BATCH_SIZE = 512
# BATCH_SIZE = 256
# BATCH_SIZE = 128

print()
print(time.strftime('%Y-%m-%d %H:%M:%S'))
print('Generating data sets...')
# train, valid, test = dataset.generate_copy_data_sets(
#     INPUT_DIMENSION_SIZE, MAX_COPY_LENGTH, TRAINING_SIZE)
train_X, train_Y = dataset.generate_copy_data_set(
    INPUT_DIMENSION_SIZE, MAX_COPY_LENGTH, TRAINING_SIZE)
valid_X, valid_Y = dataset.generate_copy_data_set(
    INPUT_DIMENSION_SIZE, MAX_COPY_LENGTH, TRAINING_SIZE/10)

print()
print(time.strftime('%Y-%m-%d %H:%M:%S'))
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
    # activation='hard_sigmoid',
    # activation='sigmoid',
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
# model.add(Activation('softmax'))
# model.add(Activation('hard_sigmoid'))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              # loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

print()
print(time.strftime('%Y-%m-%d %H:%M:%S'))
print("Model architecture")
plot(model, show_shapes=True, to_file="experiment/model_simple_rnn_for_copying.png")


print()
print(time.strftime('%Y-%m-%d %H:%M:%S'))
print("Training...")
# Train the model each generation and show predictions against the
# validation dataset
matrix_list = []
matrix_list.append(train_X[0].transpose())
matrix_list.append(train_Y[0].transpose())
matrix_list.append(train_Y[0].transpose())
name_list = []
name_list.append("Input")
name_list.append("Target")
name_list.append("Predict")
show_matrix = visualization.PlotDynamicalMatrix(matrix_list, name_list)
for iteration in range(1, 200):
    print()
    print('-' * 78)
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    print('Iteration', iteration)
    model.fit(train_X,
              train_Y,
              batch_size=BATCH_SIZE,
              nb_epoch=10,
              validation_data=(valid_X, valid_Y))
    ###
    # Select 3 samples from the validation set at random so we can
    # visualize errors
    for i in range(10):
        ind = np.random.randint(0, len(valid_X))
        # inputs = valid_X[ind]
        # outputs = valid_Y[ind]
        inputs, outputs = valid_X[np.array([ind])], valid_Y[np.array([ind])]
        predicts = model.predict(inputs, verbose=0)
        # print(inputs)
        # print(outputs)
        # print(predicts)
        matrix_list_update = []
        matrix_list_update.append(inputs[0].transpose())
        matrix_list_update.append(outputs[0].transpose())
        matrix_list_update.append(predicts[0].transpose())
        show_matrix.update(matrix_list_update, name_list)
        show_matrix.save("experiment/copy_data_predict_%3d.png"%iteration)

show_matrix.close()