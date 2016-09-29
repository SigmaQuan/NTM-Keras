# -*- coding: utf-8 -*-
"""The goal of dynamic N-Grams task was to test whether NTM could rapidly
adapt to new predictive distributions. In particular we were interested to
see if it were able to use its memory as a re-writable that it could use to
keep count of transition statistics, thereby emulating a conventional
N-Gram model.

We considered the set of all possible 6-Gram distributions over binary
sequences. Each 6-Gram distribution can be expressed as a table of
$2^{5}=32$ numbers, specifying the probability that the next bit will be
one, given all possible length five binary histories.

For each training example, we first generated random 6-Gram probabilities by
independently drawing all 32 probabilities from the $Beta(0.5, 0.5)$
distribution. We then generated a particular training sequence by drawing
200 successive bits using the current lookup table. The network observes the
sequence one bit at a time and is then asked to predict the next bit. The
optimal estimator for the problem can be determined by Bayesian analysis
$$P(B=1|N_{1}, N_{2}, c) = \frac{N_{1} + 0.5}{N_{1} + N_{0} + 1.0}$$
where c is the five bit previous context, B is the value of the next bit and
$N_{0}$ and $N_{1}$ are respectively the number of zeros and ones observed
after c so far in the sequence.

To assess performance we used a validation set of 1000 length 200 sequences
sampled from the same distribution as the training data.

Input sequence length: "200"
Input dimension: "1"
Output sequence length: equal to input sequence length.
Output dimension: equal to input dimension.
"""

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
from keras.layers import Merge               # Add by Steven Robot
from keras.callbacks import ModelCheckpoint  # Add by Steven Robot
from keras.callbacks import Callback         # Add by Steven Robot
from util import LossHistory                 # Add by Steven Robot
import cPickle as pickle
import random
import os


# Parameters for the model to train dynamic N Gram
# EXAMPLE_SIZE = 1024000   # need 70000 seconds to genarate these sequences
EXAMPLE_SIZE = 128000  # need 700 seconds to genarate these sequences
# EXAMPLE_SIZE = 12800  # need 70 seconds to genarate these sequences
# EXAMPLE_SIZE = 1280  # need 7 seconds to genarate these sequences
A = 0.5
B = 0.5
N_GRAM_SIZE = 6
INPUT_LENGTH = 40
# INPUT_LENGTH = 100
INPUT_DIMENSION_SIZE = 2


# Try replacing SimpleRNN, GRU, or LSTM
# RNN = recurrent.SimpleRNN
# RNN = recurrent.GRU
RNN = recurrent.LSTM
HIDDEN_SIZE = 128*1    # *4B/1024/1024=12MB
# HIDDEN_SIZE = 128*4    # 3152385*4B/1024/1024=12MB
# HIDDEN_SIZE = 128*8    # 12596225*4B/1024/1024=48MB
# HIDDEN_SIZE = 128*16   # 50358273*4B/1024/1024=192MB
# HIDDEN_SIZE = 128*32   # 2,0137,9841*4B/1024/1024=768MB
# HIDDEN_SIZE = 128*40   # 3,1463,9361*4/1024/1024=1200MB
# HIDDEN_SIZE = 128*128  # MemoryError: ('Error allocating 1073741824 bytes
# of device memory (CNMEM_STATUS_OUT_OF_MEMORY).',
# "you might consider using 'theano.shared(..., borrow=True)'")
LAYERS = 1
# LAYERS = MAX_REPEAT_TIMES
BATCH_SIZE = 1024
# BATCH_SIZE = 512
# BATCH_SIZE = 360
# BATCH_SIZE = 256
# BATCH_SIZE = 128  # if the batch size is larger than example size the
                  # CUDA will report error
# BATCH_SIZE = 64
# BATCH_SIZE = 8  # if the batch size is large and the hidden size is 128*16 the
                  # CUDA will report error

print()
print(time.strftime('%Y-%m-%d %H:%M:%S'))
print('Generating data sets...')
print('  generating look up table...')
look_up_table = dataset.generate_probability_of_n_gram_by_beta(
    A, B, N_GRAM_SIZE)
print(look_up_table)
print("  dumping look up table...")
pickle.dump(look_up_table,
            # open("experiment/inputs/n_gram_look_up_table.txt", "w"),
            open("experiment/inputs/n_gram_look_up_table.txt", "wb"),
            True)
print("  loading look up table...")
look_up_table = pickle.load(
    open("experiment/inputs/n_gram_look_up_table.txt", "rb"))  # "rb"
print("  Look_up_table = ")
print(look_up_table)

print(time.strftime('%Y-%m-%d %H:%M:%S'))
print('  generating training x, y...')
train_X, train_Y = dataset.generate_dynamical_n_gram_data_set(
    look_up_table, N_GRAM_SIZE, INPUT_LENGTH, EXAMPLE_SIZE)
print("  dumping training x, y...")
pickle.dump(train_X,
            open("experiment/inputs/n_gram_train_X.txt", "wb"),
            True)
pickle.dump(train_Y,
            open("experiment/inputs/n_gram_train_Y.txt", "wb"),
            True)
print("  loading training x, y...")
train_X = pickle.load(
    open("experiment/inputs/n_gram_train_X.txt", "rb"))
train_Y = pickle.load(
    open("experiment/inputs/n_gram_train_Y.txt", "rb"))

print(time.strftime('%Y-%m-%d %H:%M:%S'))
print("  train_X.shape = ")
print(train_X.shape)
print("  train_Y.shape = ")
print(train_Y.shape)

print(time.strftime('%Y-%m-%d %H:%M:%S'))
print('  generating validation x, y...')
valid_X, valid_Y = dataset.generate_dynamical_n_gram_data_set(
    look_up_table, N_GRAM_SIZE, INPUT_LENGTH, EXAMPLE_SIZE/10)
print("  dumping validation x, y...")
pickle.dump(valid_X,
            open("experiment/inputs/n_gram_valid_X.txt", "wb"),
            True)
pickle.dump(valid_Y,
            open("experiment/inputs/n_gram_valid_Y.txt", "wb"),
            True)
print("  validation training x, y...")
valid_X = pickle.load(
            open("experiment/inputs/n_gram_valid_X.txt", "rb"))
valid_Y = pickle.load(
            open("experiment/inputs/n_gram_valid_Y.txt", "rb"))

print(time.strftime('%Y-%m-%d %H:%M:%S'))
print("  valid_X.shape = ")
print(valid_X.shape)
print("  valid_Y.shape = ")
print(valid_Y.shape)

print()
print(time.strftime('%Y-%m-%d %H:%M:%S'))
print('Showing data sets...')
# show training sample
show_matrix = visualization.PlotDynamicalMatrix4NGram(
    train_X[0].transpose(), train_Y[0].transpose(), train_Y[0].transpose())
# show_size = EXAMPLE_SIZE/10
show_size = 20
random_index = np.random.randint(1, EXAMPLE_SIZE, show_size)
for i in range(show_size):
    show_matrix.update(
        train_X[random_index[i]].transpose(),
        train_Y[random_index[i]].transpose(),
        train_Y[random_index[i]].transpose())
    show_matrix.save("experiment/inputs/n_gram_data_training_%2d.png"%i)

print()
print(time.strftime('%Y-%m-%d %H:%M:%S'))
print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
hidden_layer = RNN(
    HIDDEN_SIZE,
    input_shape=(INPUT_LENGTH*2-N_GRAM_SIZE+2, INPUT_DIMENSION_SIZE+1),
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
model.add(RepeatVector(INPUT_LENGTH*2-N_GRAM_SIZE+2))
# The decoder RNN could be multiple layers stacked or a single layer
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# For each of step of the output sequence, decide which character should be chosen
model.add(TimeDistributed(Dense(INPUT_DIMENSION_SIZE+1)))
model.add(Activation('softmax'))
# model.add(Activation('hard_sigmoid'))
# model.add(Activation('sigmoid'))

model.compile(#loss='binary_crossentropy',
              # loss='mse',
              loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print()
print(time.strftime('%Y-%m-%d %H:%M:%S'))
print("Model architecture")
plot(model, show_shapes=True, to_file="experiment/lstm_n_gram.png")
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
        filepath="experiment/n_gram_model_weights.hdf5",
        verbose=1, save_best_only=True)
    model.fit(train_X,
              train_Y,
              batch_size=BATCH_SIZE,
              nb_epoch=1,
              # nb_epoch=1,
              # callbacks=[check_pointer, history],
              validation_data=(valid_X, valid_Y))
    # print(len(history.losses))
    # print(history.losses)
    # print(len(history.acces))
    # print(history.acces)

    ###
    # Select 20 samples from the validation set at random so we can
    # visualize errors
    for i in range(20):
        ind = np.random.randint(0, len(valid_X))
        inputs, outputs = valid_X[np.array([ind])], \
                                  valid_Y[np.array([ind])]
        predicts = model.predict(inputs, verbose=0)

        show_matrix.update(
            inputs[0].transpose(),
            outputs[0].transpose(),
            predicts[0].transpose())
        show_matrix.save("experiment/inputs/n_gram_data_predict_%2d.png"%i)

show_matrix.close()

