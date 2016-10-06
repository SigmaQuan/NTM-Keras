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
from keras.layers import Merge               # Add by Steven Robot
from keras.callbacks import ModelCheckpoint  # Add by Steven Robot
from keras.callbacks import Callback         # Add by Steven Robot
from util import LossHistory                 # Add by Steven Robot


# Parameters for the model to train copying algorithm
# EXAMPLE_SIZE = 2560000
# EXAMPLE_SIZE = 1024000
EXAMPLE_SIZE = 128000
# EXAMPLE_SIZE = 1280
# INPUT_DIMENSION_SIZE = 8
INPUT_DIMENSION_SIZE = 4
INPUT_SEQUENCE_LENGTH = 20
PRIORITY_OUTPUT_SEQUENCE_LENGTH = 16
SEQUENCE_LENGTH = INPUT_SEQUENCE_LENGTH + PRIORITY_OUTPUT_SEQUENCE_LENGTH + 1
PRIORITY_LOWER_BOUND = 0
PRIORITY_UPPER_BOUND = 1

# Try replacing SimpleRNN, GRU, or LSTM
# RNN = recurrent.SimpleRNN
# RNN = recurrent.GRU
RNN = recurrent.LSTM
# HIDDEN_SIZE = 128  # acc. 99.9%
# HIDDEN_SIZE = 128*30  # 191919370 parameters
# HIDDEN_SIZE = 128*16  #  54646794 parameters
# HIDDEN_SIZE = 128*8   #  13691914 parameters
HIDDEN_SIZE = 128*2   #   3438090 parameters
# HIDDEN_SIZE = 128*1   #    220554 parameters
# HIDDEN_SIZE = 64       #     57034 parameters
LAYERS = 1
# LAYERS = MAX_REPEAT_TIMES
BATCH_SIZE = 1024
# BATCH_SIZE = 16

print()
print(time.strftime('%Y-%m-%d %H:%M:%S'))
print('Generating data sets...')
train_x_seq, train_y_seq = \
    dataset.generate_associative_priority_sort_data_set(
        INPUT_DIMENSION_SIZE,
        INPUT_SEQUENCE_LENGTH,
        PRIORITY_OUTPUT_SEQUENCE_LENGTH,
        PRIORITY_LOWER_BOUND,
        PRIORITY_UPPER_BOUND,
        EXAMPLE_SIZE)
print(train_x_seq.shape)
print(train_y_seq.shape)
validation_x_seq, validation_y_seq = \
    dataset.generate_associative_priority_sort_data_set(
        INPUT_DIMENSION_SIZE,
        INPUT_SEQUENCE_LENGTH,
        PRIORITY_OUTPUT_SEQUENCE_LENGTH,
        PRIORITY_LOWER_BOUND,
        PRIORITY_UPPER_BOUND,
        EXAMPLE_SIZE/10)
print(validation_x_seq.shape)
print(validation_y_seq.shape)

input_matrix = np.zeros(
    (SEQUENCE_LENGTH, INPUT_DIMENSION_SIZE+1),
    dtype=np.float32)
output_matrix = np.zeros(
    (SEQUENCE_LENGTH, INPUT_DIMENSION_SIZE+1),
    dtype=np.float32)
predict_matrix = np.zeros(
    (SEQUENCE_LENGTH, INPUT_DIMENSION_SIZE+1),
    dtype=np.float32)
input_matrix = train_x_seq[0]
output_matrix = train_y_seq[0]
predict_matrix = output_matrix
show_matrix = visualization.PlotDynamicalMatrix4PrioritySort(
    input_matrix.transpose(),
    output_matrix.transpose(),
    predict_matrix.transpose())
random_index = np.random.randint(1, 128, 20)
for i in range(20):
    input_matrix = train_x_seq[random_index[i]]
    output_matrix = train_y_seq[random_index[i]]
    predict_matrix = output_matrix
    show_matrix.update(input_matrix.transpose(),
                       output_matrix.transpose(),
                       predict_matrix.transpose())
    show_matrix.save("experiment/priority_data_training_%2d.png"%i)
# show_matrix.close()

print()
print(time.strftime('%Y-%m-%d %H:%M:%S'))
print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
hidden_layer = RNN(
    HIDDEN_SIZE,
    input_shape=(SEQUENCE_LENGTH, INPUT_DIMENSION_SIZE+2),
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
model.add(
    Dense(HIDDEN_SIZE, input_shape=(SEQUENCE_LENGTH, INPUT_DIMENSION_SIZE+2)))

# For the decoder's input, we repeat the encoded input for each time step
model.add(RepeatVector(SEQUENCE_LENGTH))
# The decoder RNN could be multiple layers stacked or a single layer
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# For each of step of the output sequence, decide which character should be chosen
model.add(TimeDistributed(Dense(INPUT_DIMENSION_SIZE+2)))
# model.add(Activation('softmax'))
# model.add(Activation('hard_sigmoid'))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              #loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

print()
print(time.strftime('%Y-%m-%d %H:%M:%S'))
print("Model architecture")
plot(model, show_shapes=True, to_file="experiment/lstm_priority_sort.png")
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
        filepath="experiment/priority_sort_model_weights.hdf5",
        verbose=1, save_best_only=True)
    model.fit([train_x_seq],
              train_y_seq,
              batch_size=BATCH_SIZE,
              nb_epoch=10,
              # callbacks=[check_pointer, history],
              validation_data=([validation_x_seq], validation_y_seq))
    # print(len(history.losses))
    # print(history.losses)
    # print(len(history.acces))
    # print(history.acces)

    ###
    # Select 20 samples from the validation set at random so we can
    # visualize errors
    for i in range(20):
        ind = np.random.randint(0, len(validation_x_seq))
        inputs, outputs = validation_x_seq[np.array([ind])],\
                                    validation_y_seq[np.array([ind])]
        predicts = model.predict([inputs], verbose=0)

        input_matrix = validation_x_seq[np.array([ind])]
        output_matrix = validation_y_seq[np.array([ind])]
        predict_matrix = predicts

        show_matrix.update(input_matrix[0].transpose(),
                           output_matrix[0].transpose(),
                           predict_matrix[0].transpose())
        show_matrix.save("experiment/priority_data_training_%2d_%2d.png"%(iteration, i))

show_matrix.close()

