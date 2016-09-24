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
# EXAMPLE_SIZE = 1024000
# EXAMPLE_SIZE = 128000
EXAMPLE_SIZE = 1280
INPUT_DIMENSION_SIZE = 8
INPUT_SEQUENCE_LENGTH = 20
PRIORITY_OUTPUT_SEQUENCE_LENGTH = 16
SEQUENCE_LENGTH = INPUT_SEQUENCE_LENGTH + PRIORITY_OUTPUT_SEQUENCE_LENGTH
PRIORITY_LOWER_BOUND = -1
PRIORITY_UPPER_BOUND = 1

# Try replacing SimpleRNN, GRU, or LSTM
# RNN = recurrent.SimpleRNN
# RNN = recurrent.GRU
RNN = recurrent.LSTM
# HIDDEN_SIZE = 128  # acc. 99.9%
HIDDEN_SIZE = 128*2
LAYERS = 1
# LAYERS = MAX_REPEAT_TIMES
BATCH_SIZE = 1024

print()
print(time.strftime('%Y-%m-%d %H:%M:%S'))
print('Generating data sets...')
# Fix 2 times copying
# train_X, train_Y = dataset.generate_repeat_copy_data_set(
#     INPUT_DIMENSION_SIZE, MAX_COPY_LENGTH, TRAINING_SIZE, REPEAT_TIMES)
# valid_X, valid_Y = dataset.generate_repeat_copy_data_set(
#     INPUT_DIMENSION_SIZE, MAX_COPY_LENGTH, TRAINING_SIZE/10, REPEAT_TIMES)
# train_X, train_Y, train_repeats_times = dataset.generate_repeat_copy_data_set(
#     INPUT_DIMENSION_SIZE, MAX_COPY_LENGTH, TRAINING_SIZE, MAX_REPEAT_TIMES)
# valid_X, valid_Y, valid_repeats_times = dataset.generate_repeat_copy_data_set(
#     INPUT_DIMENSION_SIZE, MAX_COPY_LENGTH, TRAINING_SIZE/10, MAX_REPEAT_TIMES)
# print(train_repeats_times)
# print(valid_repeats_times)
# train_repeats_times = (train_repeats_times - 1.0) / (MAX_REPEAT_TIMES - 1.0)
# valid_repeats_times = (valid_repeats_times - 1.0) / (MAX_REPEAT_TIMES - 1.0)
# print(train_repeats_times)
# print(valid_repeats_times)
train_x_seq, train_x_priority, train_y_seq, train_y_priority = \
    dataset.generate_associative_priority_sort_data_set(
        INPUT_DIMENSION_SIZE,
        INPUT_SEQUENCE_LENGTH,
        PRIORITY_OUTPUT_SEQUENCE_LENGTH,
        PRIORITY_LOWER_BOUND,
        PRIORITY_UPPER_BOUND,
        EXAMPLE_SIZE)
print(train_x_seq.shape)
print(train_x_priority.shape)
print(train_y_seq.shape)
print(train_y_priority.shape)
validation_x_seq, validation_x_priority, \
validation_y_seq, validation_y_priority = \
    dataset.generate_associative_priority_sort_data_set(
        INPUT_DIMENSION_SIZE,
        INPUT_SEQUENCE_LENGTH,
        PRIORITY_OUTPUT_SEQUENCE_LENGTH,
        PRIORITY_LOWER_BOUND,
        PRIORITY_UPPER_BOUND,
        EXAMPLE_SIZE/10)
print(validation_x_seq.shape)
print(validation_x_priority.shape)
print(validation_y_seq.shape)
print(validation_y_priority.shape)

input_matrix = np.zeros(
    (SEQUENCE_LENGTH, INPUT_DIMENSION_SIZE+1),
    dtype=np.float32)
output_matrix = np.zeros(
    (SEQUENCE_LENGTH, INPUT_DIMENSION_SIZE+1),
    dtype=np.float32)
predict_matrix = np.zeros(
    (SEQUENCE_LENGTH, INPUT_DIMENSION_SIZE+1),
    dtype=np.float32)
input_matrix[:, :-1] = train_x_seq[0]
input_matrix[:, -1] = train_x_priority[0].reshape(SEQUENCE_LENGTH)
output_matrix[:, :-1] = train_y_seq[0]
output_matrix[:, -1] = train_y_priority[0].reshape(SEQUENCE_LENGTH)
predict_matrix = output_matrix
show_matrix = visualization.PlotDynamicalMatrix4PrioritySort(
    input_matrix.transpose(),
    output_matrix.transpose(),
    output_matrix.transpose())
random_index = np.random.randint(1, 128, 20)
# for i in range(20):
#     input_matrix[:, :-1] = train_x_seq[random_index[i]]
#     input_matrix[:, -1] = train_x_priority[random_index[i]].reshape(SEQUENCE_LENGTH)
#     output_matrix[:, :-1] = train_y_seq[random_index[i]]
#     output_matrix[:, -1] = train_y_priority[random_index[i]].reshape(SEQUENCE_LENGTH)
#     show_matrix.update(input_matrix.transpose(),
#                        output_matrix.transpose(),
#                        output_matrix.transpose())
#     show_matrix.save("experiment/priority_data_training_%2d.png"%i)
# show_matrix.close()

print()
print(time.strftime('%Y-%m-%d %H:%M:%S'))
print('Build model...')
input_sequence = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
hidden_layer = RNN(
    HIDDEN_SIZE,
    input_shape=(SEQUENCE_LENGTH, INPUT_DIMENSION_SIZE),
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
# input_sequence.add(hidden_layer)
input_sequence.add(
    Dense(HIDDEN_SIZE, input_shape=(SEQUENCE_LENGTH, INPUT_DIMENSION_SIZE)))

sequence_priority = Sequential()
hidden_layer_priority = RNN(
    HIDDEN_SIZE,
    input_shape=(SEQUENCE_LENGTH, 1),
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
# sequence_priority.add(hidden_layer_priority)
sequence_priority.add(
    Dense(HIDDEN_SIZE, input_shape=(SEQUENCE_LENGTH, 1)))

merged = Merge([input_sequence, sequence_priority], mode='concat')

model = Sequential()
model.add(merged)

# For the decoder's input, we repeat the encoded input for each time step
model.add(RepeatVector(SEQUENCE_LENGTH))
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
    model.fit([train_x_seq, train_x_priority],
              train_y_seq,
              batch_size=BATCH_SIZE,
              nb_epoch=1,
              # callbacks=[check_pointer, history],
              validation_data=([validation_x_seq, validation_x_priority], validation_y_seq))
    # print(len(history.losses))
    # print(history.losses)
    # print(len(history.acces))
    # print(history.acces)

    ###
    # Select 20 samples from the validation set at random so we can
    # visualize errors
    for i in range(20):
        ind = np.random.randint(0, len(validation_x_seq))
        inputs, priority, outputs = validation_x_seq[np.array([ind])], \
                                    validation_x_priority[np.array([ind])], \
                                    validation_y_seq[np.array([ind])]
        predicts = model.predict([inputs, priority], verbose=0)

        input_matrix[:, :-1] = validation_x_seq[np.array([ind])]
        input_matrix[:, -1] = validation_x_priority[np.array([ind])].reshape(SEQUENCE_LENGTH)
        output_matrix[:, :-1] = validation_y_seq[np.array([ind])]
        output_matrix[:, -1] = validation_y_priority[np.array([ind])].reshape(SEQUENCE_LENGTH)
        predict_matrix[:, :-1] = predicts[0]
        # predict_matrix[:, -1] = validation_y_priority[np.array([ind])].reshape(SEQUENCE_LENGTH)

        show_matrix.update(input_matrix.transpose(),
                           output_matrix.transpose(),
                           predict_matrix.transpose())
        show_matrix.save("experiment/priority_data_training_%2d_%2d.png"%(i, iteration))

show_matrix.close()

