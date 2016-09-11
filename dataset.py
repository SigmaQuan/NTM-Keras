import numpy as np
import random


np.random.seed(5456)
random.seed(5456)


def generate_random_binomial(row, col):
    return np.random.binomial(
        1, 0.5, (row, col)).astype(np.uint8)


def generate_weightings(row, col):
    write_weightings = np.zeros((row, col), dtype=np.float32)
    read_weightings = np.zeros((row, col), dtype=np.float32)
    r = (row * 3) / 4
    for i in np.arange(0, col/2):
        write_weightings[r][i] = 1
        read_weightings[r][i + col/2] = 1
        r = r - 1

    return write_weightings, read_weightings


def generate_copy_data(input_size, sequence_length):
    sequence = np.random.binomial(
        1, 0.5, (sequence_length, input_size - 1)).astype(np.uint8)
    input_sequence = np.zeros(
        (sequence_length * 2 + 1, input_size), dtype=np.bool)
    output_sequence = np.zeros(
        (sequence_length * 2 + 1, input_size), dtype=np.bool)

    input_sequence[:sequence_length, :-1] = sequence
    input_sequence[sequence_length, -1] = 1
    output_sequence[sequence_length + 1:, :-1] = sequence

    return input_sequence, output_sequence


def generate_copy_data_set(input_size, max_size, training_size):
    sequence_lengths = np.random.randint(1, max_size + 1, training_size)
    input_sequences = np.zeros((training_size, max_size*2+1, input_size), dtype=np.bool)
    output_sequences = np.zeros((training_size, max_size*2+1, input_size), dtype=np.bool)
    for i in range(training_size):
        input_sequence, output_sequence = generate_copy_data(
            input_size, sequence_lengths[i])
        for j in range(sequence_lengths[i]*2+1):
            index_1 = max_size*2+1-j-1
            index_2 = sequence_lengths[i]*2+1-j-1
            input_sequences[i][index_1] = input_sequence[index_2]
            output_sequences[i][index_1] = output_sequence[index_2]

    return input_sequences, output_sequences


def generate_copy_data_sets(input_size, max_size, training_size):
    input_train, output_train = generate_copy_data_set(
        input_size, max_size, training_size)
    train = (input_train, output_train)

    input_valid, output_valid = generate_copy_data_set(
        input_size, max_size, training_size/10)
    valid = (input_valid, output_valid)

    input_test, output_test = generate_copy_data_set(
        input_size, max_size, training_size/10)
    test = (input_test, output_test)

    return train, valid, test


def generate_repeat_copy_data(input_size, max_size, num_repeats):
    sequence_length = max_size
    sequence = np.random.binomial(
        1, 0.5, (sequence_length, input_size - 1)).astype(np.uint8)
    input_sequence = np.zeros(
        (sequence_length + 1 + sequence_length * num_repeats + 1, input_size), dtype=np.bool)
    output_sequence = np.zeros(
        (sequence_length + 1 + sequence_length * num_repeats + 1, input_size), dtype=np.bool)

    input_sequence[:sequence_length, :-1] = sequence
    input_sequence[sequence_length, -1] = num_repeats
    output_sequence[sequence_length + 1:-1, :-
                    1] = np.tile(sequence, (num_repeats, 1))
    output_sequence[-1, -1] = 1

    return input_sequence, output_sequence
