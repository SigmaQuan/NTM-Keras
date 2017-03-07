# -*- coding: utf-8 -*-
"""An implementation of learning copying algorithm_learning with RNN (basic RNN, LSTM,
GRU).
Input sequence length: "1 ~ 20: (1*2+1)=3 ~ (20*2+1)=41"
Input dimension: "4"
Repeat times: "5"
Output sequence length: equal to input sequence length * repeat times.
Output dimension: equal to input dimension.
"""

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
from keras.layers import Merge               # Add by Steven Robot
from keras.callbacks import ModelCheckpoint  # Add by Steven Robot
from keras.callbacks import Callback         # Add by Steven Robot
from util import LossHistory                 # Add by Steven Robot
import time                                  # Add by Steven Robot
import os                                    # Add by Steven Robot
import sys                                   # Add by Steven Robot



# Parameters for the model to train copying algorithm_learning
TRAINING_SIZE = 1024000
# TRAINING_SIZE = 10240
# TRAINING_SIZE = 128000
# TRAINING_SIZE = 1280
# INPUT_DIMENSION_SIZE = 4 + 1
# INPUT_DIMENSION_SIZE = 7 + 1
INPUT_DIMENSION_SIZE = 8 + 1
MAX_COPY_LENGTH = 10
# REPEAT_TIMES = 2
# MAX_INPUT_LENGTH = MAX_COPY_LENGTH + 1 + REPEAT_TIMES * MAX_COPY_LENGTH + 1
# MAX_REPEAT_TIMES = 5
MAX_REPEAT_TIMES = 10
MAX_INPUT_LENGTH = MAX_COPY_LENGTH + 1 + MAX_REPEAT_TIMES * MAX_COPY_LENGTH  # + 1

# Try replacing SimpleRNN, GRU, or LSTM
# RNN = recurrent.SimpleRNN
# RNN = recurrent.GRU
RNN = recurrent.LSTM
# HIDDEN_SIZE = 128  # acc. 99.9%
HIDDEN_SIZE = 128*4
LAYERS = 2
# LAYERS = MAX_REPEAT_TIMES
# BATCH_SIZE = 1024
BATCH_SIZE = 128
# BATCH_SIZE = 128


folder_name = time.strftime('experiment_results/re_copy_lstm/%Y-%m-%d-%H-%M-%S/')
# os.makedirs(folder_name)
FOLDER = folder_name
if not os.path.isdir(FOLDER):
    os.makedirs(FOLDER)
    print("create folder: %s" % FOLDER)

start_time = time.time()
sys_stdout = sys.stdout
log_file = '%s/recall.log' % (folder_name)
sys.stdout = open(log_file, 'a')


print()
print(time.strftime('%Y-%m-%d %H:%M:%S'))
print('Generating data sets...')
# Fix 2 times copying
# train_X, train_Y = dataset.generate_repeat_copy_data_set(
#     INPUT_DIMENSION_SIZE, MAX_COPY_LENGTH, TRAINING_SIZE, REPEAT_TIMES)
# valid_X, valid_Y = dataset.generate_repeat_copy_data_set(
#     INPUT_DIMENSION_SIZE, MAX_COPY_LENGTH, TRAINING_SIZE/10, REPEAT_TIMES)
train_X, train_Y, train_repeats_times = dataset.generate_repeat_copy_data_set(
    INPUT_DIMENSION_SIZE, MAX_COPY_LENGTH, TRAINING_SIZE, MAX_REPEAT_TIMES)
valid_X, valid_Y, valid_repeats_times = dataset.generate_repeat_copy_data_set(
    INPUT_DIMENSION_SIZE, MAX_COPY_LENGTH, TRAINING_SIZE/10, MAX_REPEAT_TIMES)
print(train_repeats_times)
print(valid_repeats_times)
train_repeats_times = (train_repeats_times - 1.0) / (MAX_REPEAT_TIMES - 1.0)
valid_repeats_times = (valid_repeats_times - 1.0) / (MAX_REPEAT_TIMES - 1.0)
print(train_repeats_times)
print(valid_repeats_times)

matrix_list = []
matrix_list.append(train_X[0].transpose())
matrix_list.append(train_Y[0].transpose())
matrix_list.append(train_Y[0].transpose())
name_list = []
name_list.append("Input")
name_list.append("Target")
name_list.append("Predict")
show_matrix = visualization.PlotDynamicalMatrix4Repeat(
    matrix_list, name_list, train_repeats_times[0])
random_index = np.random.randint(1, 128, 20)
for i in range(20):
    matrix_list_update = []
    matrix_list_update.append(train_X[random_index[i]].transpose())
    matrix_list_update.append(train_Y[random_index[i]].transpose())
    matrix_list_update.append(train_Y[random_index[i]].transpose())
    show_matrix.update(matrix_list_update, name_list, train_repeats_times[random_index[i]])
    show_matrix.save(FOLDER+"repeat_copy_data_training_%2d.png"%i)

print()
print(time.strftime('%Y-%m-%d %H:%M:%S'))
print('Build model...')
input_sequence = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
hidden_layer = RNN(
    HIDDEN_SIZE,
    input_shape=(MAX_INPUT_LENGTH, INPUT_DIMENSION_SIZE),
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
    dropout_U=0.0)
input_sequence.add(hidden_layer)

repeat_times = Sequential()
repeat_times.add(Dense(16, input_dim=1))
repeat_times.add(Activation('sigmoid'))
repeat_times.add(RepeatVector(MAX_INPUT_LENGTH))   # add

merged = Merge([input_sequence, repeat_times], mode='concat')

model = Sequential()
model.add(merged)

# For the decoder's input, we repeat the encoded input for each time step
# model.add(RepeatVector(MAX_INPUT_LENGTH))
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
plot(model, show_shapes=True, to_file=FOLDER+"lstm_repeat_copying.png")
print("Model summary")
print(model.summary())
print("Model parameter count")
print(model.count_params())

print()
print(time.strftime('%Y-%m-%d %H:%M:%S'))
print("Training...")
# Train the model each generation and show predictions against the
# validation dataset
losses = []
acces = []
for iteration in range(1, 3):
    print()
    print('-' * 78)
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    print('Iteration', iteration)
    history = LossHistory()
    check_pointer = ModelCheckpoint(
        filepath=FOLDER+"repeat_copying_model_weights.hdf5",
        verbose=1, save_best_only=True)
    model.fit([train_X, train_repeats_times],
              train_Y,
              batch_size=BATCH_SIZE,
              # nb_epoch=30,
              nb_epoch=1,
              callbacks=[check_pointer, history],
              validation_data=([valid_X, valid_repeats_times], valid_Y))
    # print(len(history.losses))
    # print(history.losses)
    # print(len(history.acces))
    # print(history.acces)
    losses.append(history.losses)
    acces.append(history.acces)

    ###
    # Select 20 samples from the validation set at random so we can
    # visualize errors
    for i in range(20):
        ind = np.random.randint(0, len(valid_X))
        inputs, repeats, outputs = valid_X[np.array([ind])], \
                                  valid_repeats_times[np.array([ind])], \
                                  valid_Y[np.array([ind])]
        predicts = model.predict([inputs, repeats], verbose=0)
        matrix_list_update = []
        matrix_list_update.append(inputs[0].transpose())
        matrix_list_update.append(outputs[0].transpose())
        matrix_list_update.append(predicts[0].transpose())
        show_matrix.update(matrix_list_update,
                           name_list,
                           valid_repeats_times[ind] * (MAX_REPEAT_TIMES - 1.0) + 1)
        show_matrix.save(FOLDER+"repeat_copy_data_predict_%2d_%2d.png" % (iteration, i))

show_matrix.close()

# end of training

# print loss and accuracy
print("\nlosses")
print(len(losses))
print(len(losses[0]))
# print(losses.shape)
sample_num = 1
for los in losses:
    for lo in los:
        if sample_num % 100 == 1:
            print("(%d, %f)" % (sample_num, lo))
        sample_num = sample_num + 1
# print(losses)

print("********************************************")
print("\naccess")
print(len(acces))
print(len(acces[0]))
# print(acces.shape)
sample_num = 1
for acc in acces:
    for ac in acc:
        if sample_num % 100 == 1:
            print("(%d, %f)" % (sample_num, ac))
        sample_num = sample_num + 1
# print(acces)

# print loss and accuracy
print("\nlosses")
print(len(losses))
print(len(losses[0]))
# print(losses.shape)
sample_num = 1
for los in losses:
    for lo in los:
        print("(%d, %f)" % (sample_num, lo))
        sample_num = sample_num + 1
# print(losses)

print("********************************************")
print("\naccess")
print(len(acces))
print(len(acces[0]))
# print(acces.shape)
sample_num = 1
for acc in acces:
    for ac in acc:
        print("(%d, %f)" % (sample_num, ac))
        sample_num = sample_num + 1
# print(acces)

print ("task took %.3fs" % (float(time.time()) - start_time))
sys.stdout.close()
sys.stdout = sys_stdout

