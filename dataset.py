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


def generate_copy_data(input_size, max_size):
    sequence_length = max_size
    sequence = np.random.binomial(
        1, 0.5, (sequence_length, input_size - 1)).astype(np.uint8)
    input_sequence = np.zeros(
        (sequence_length * 2 + 1, input_size), dtype=np.float32)
    output_sequence = np.zeros(
        (sequence_length * 2 + 1, input_size), dtype=np.float32)

    input_sequence[:sequence_length, :-1] = sequence
    input_sequence[sequence_length, -1] = 1
    output_sequence[sequence_length + 1:, :-1] = sequence
    return input_sequence, output_sequence


def generate_repeat_copy_data(input_size, max_size, num_repeats):
    sequence_length = max_size
    sequence = np.random.binomial(
        1, 0.5, (sequence_length, input_size - 1)).astype(np.uint8)
    input_sequence = np.zeros(
        (sequence_length + 1 + sequence_length * num_repeats + 1, input_size), dtype=np.float32)
    output_sequence = np.zeros(
        (sequence_length + 1 + sequence_length * num_repeats + 1, input_size), dtype=np.float32)

    input_sequence[:sequence_length, :-1] = sequence
    input_sequence[sequence_length, -1] = num_repeats
    output_sequence[sequence_length + 1:-1, :-
                    1] = np.tile(sequence, (num_repeats, 1))
    output_sequence[-1, -1] = 1
    return input_sequence, output_sequence
