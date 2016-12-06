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
import visualization                         # Add by Steven Robot
from keras.utils.visualize_util import plot  # Add by Steven Robot
import time                                  # Add by Steven Robot
from keras.callbacks import ModelCheckpoint  # Add by Steven Robot
from keras.callbacks import Callback         # Add by Steven Robot
from util import LossHistory                 # Add by Steven Robot
import os                                    # Add by Steven Robot
import ntm                                    # Add by Steven Robot


# Parameters for the model to train copying algorithm
TRAINING_SIZE = 1024000
# TRAINING_SIZE = 128000
# TRAINING_SIZE = 1280
INPUT_DIMENSION_SIZE = 6
ITEM_SIZE = 3
MAX_EPISODE_SIZE = 6
MAX_INPUT_LENGTH = (ITEM_SIZE+1) * (MAX_EPISODE_SIZE+2)

# Try replacing SimpleRNN, GRU, or LSTM
# RNN = recurrent.SimpleRNN
# RNN = recurrent.GRU
RNN = recurrent.LSTM
HIDDEN_SIZE = 256
LAYERS = 2
# LAYERS = MAX_REPEAT_TIMES
# BATCH_SIZE = 1024
BATCH_SIZE = 128
FOLDER = "experiment_results/associative_recall_ntm/"
if not os.path.isdir(FOLDER):
    os.makedirs(FOLDER)
    print("create folder: %s" % FOLDER)

print()
print(time.strftime('%Y-%m-%d %H:%M:%S'))
print('Generating data sets...')
train_X, train_Y = dataset.generate_associative_recall_data_set(
        INPUT_DIMENSION_SIZE, ITEM_SIZE, MAX_EPISODE_SIZE, TRAINING_SIZE)
valid_X, valid_Y = dataset.generate_associative_recall_data_set(
        INPUT_DIMENSION_SIZE, ITEM_SIZE, MAX_EPISODE_SIZE, TRAINING_SIZE/5)

matrix_list = []
matrix_list.append(train_X[0].transpose())
matrix_list.append(train_Y[0].transpose())
matrix_list.append(train_Y[0].transpose())
name_list = []
name_list.append("Input")
name_list.append("Target")
name_list.append("Predict")
show_matrix = visualization.PlotDynamicalMatrix(matrix_list, name_list)
random_index = np.random.randint(1, 128, 20)
for i in range(20):
    matrix_list_update = []
    matrix_list_update.append(train_X[random_index[i]].transpose())
    matrix_list_update.append(train_Y[random_index[i]].transpose())
    matrix_list_update.append(train_Y[random_index[i]].transpose())
    show_matrix.update(matrix_list_update, name_list)
    show_matrix.save(FOLDER+"associative_recall_data_training_%2d.png" % i)

print()
print(time.strftime('%Y-%m-%d %H:%M:%S'))
print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
model.add(RNN(
    HIDDEN_SIZE,
    input_shape=(MAX_INPUT_LENGTH, INPUT_DIMENSION_SIZE+2),
    init='glorot_uniform',
    inner_init='orthogonal',
    activation='tanh',
    return_sequences=True,
    # activation='hard_sigmoid',
    # activation='sigmoid',
    W_regularizer=None,
    U_regularizer=None,
    b_regularizer=None,
    dropout_W=0.0,
    dropout_U=0.0))


# For the decoder's input, we repeat the encoded input for each time step
# model.add(RepeatVector(MAX_INPUT_LENGTH))
# The decoder RNN could be multiple layers stacked or a single layer
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# For each of step of the output sequence, decide which character should be chosen
model.add(TimeDistributed(Dense(INPUT_DIMENSION_SIZE+2)))
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
plot(model, show_shapes=True, to_file=FOLDER+"lstm_associative_recall.png")
print("Model summary")
print(model.summary())
print("Model parameter count")
print(model.count_params())

print()
print(time.strftime('%Y-%m-%d %H:%M:%S'))
print("Training...")
# Train the model each generation and show predictions against the
# validation dataset
for iteration in range(1, 200):
    print()
    print('-' * 78)
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    print('Iteration', iteration)
    history = LossHistory()
    check_pointer = ModelCheckpoint(
        filepath=FOLDER+"associative_recall_model_weights.hdf5",
        verbose=1, save_best_only=True)
    model.fit(train_X,
              train_Y,
              batch_size=BATCH_SIZE,
              nb_epoch=30,
              # nb_epoch=1,
              callbacks=[check_pointer, history],
              validation_data=(valid_X, valid_Y))
    print(len(history.losses))
    print(history.losses)
    print(len(history.acces))
    print(history.acces)

    ###
    # Select 20 samples from the validation set at random so we can
    # visualize errors
    for i in range(20):
        ind = np.random.randint(0, len(valid_X))
        inputs, outputs = valid_X[np.array([ind])], \
                                  valid_Y[np.array([ind])]
        predicts = model.predict(inputs, verbose=0)
        matrix_list_update = []
        matrix_list_update.append(inputs[0].transpose())
        matrix_list_update.append(outputs[0].transpose())
        matrix_list_update.append(predicts[0].transpose())
        show_matrix.update(matrix_list_update,
                           name_list)
        show_matrix.save(FOLDER+"associative_data_predict_%3d.png"%iteration)

show_matrix.close()
