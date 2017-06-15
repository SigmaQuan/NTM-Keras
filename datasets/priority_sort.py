# -*- coding: utf-8 -*-
import numpy as np
from utils import initialize_random_seed


# Initialize the random seed
initialize_random_seed()


def generate_one_sample(
        dimension,
        input_sequence_length,
        output_sequence_length,
        priority_lower_bound,
        priority_upper_bound):
    """Generate one sample of priority sort algorithm.

    Arguments
        dimension: the dimension of input output sequences.
        input_sequence_length: the length of input sequence.
        output_sequence_length: the length of output sequence.
        priority_lower_bound: the lower bound of priority.
        priority_upper_bound: the upper bound of priority.

    Returns
        input_sequence: the input sequence of a sample.
        output_sequence: the output sequence of a sample.
    """
    sequence = input_sequence_length + output_sequence_length + 1
    input_sequence = np.random.binomial(
        1, 0.5, (input_sequence_length, dimension + 1)).astype(np.uint8)
    output_sequence = np.zeros(
        (output_sequence_length, dimension + 1), dtype=np.uint8)
    input_priority = np.random.uniform(priority_lower_bound,
                                       priority_upper_bound,
                                       (input_sequence_length, 1))
    output_priority = sorted(
        input_priority, reverse=True)[:output_sequence_length]
    pair = [(input_sequence[i], input_priority[i])
            for i in range(input_sequence_length)]
    sorted_input_sequence = sorted(
        pair, key=lambda prior: prior[1], reverse=True)
    for i in range(output_sequence_length):
        output_sequence[i] = sorted_input_sequence[i][0]

    input_sequence_ = np.zeros((sequence, dimension + 2), dtype=np.float32)
    input_priority_ = np.zeros((sequence, 1), dtype=np.float32)
    output_sequence_ = np.zeros((sequence, dimension + 2), dtype=np.float32)
    output_priority_ = np.zeros((sequence, 1), dtype=np.float32)

    input_sequence_[:input_sequence_length, :-1] = input_sequence
    input_sequence_[input_sequence_length][-1] = 1
    input_priority_[:input_sequence_length] = input_priority
    output_sequence_[input_sequence_length+1:sequence, :-1] = output_sequence
    output_priority_[input_sequence_length+1:sequence] = output_priority

    # return input sequence, priority of each input, output sequence, priority
    # of each output
    return input_sequence_, input_priority_, output_sequence_, output_priority_


def generate_data_set(
        dimension,
        input_sequence_length,
        output_sequence_length,
        priority_lower_bound,
        priority_upper_bound,
        data_set_size):
    """Generate samples for learning priority sort algorithm.

    Arguments
        dimension: the dimension of input output sequences.
        input_sequence_length: the length of input sequence.
        output_sequence_length: the length of output sequence.
        priority_lower_bound: the lower bound of priority.
        priority_upper_bound: the upper bound of priority.
        data_set_size: the size of one episode.

    Returns
        input_sequence: the input sequence of a sample.
        output_sequence: the output sequence of a sample.
    """
    sequence_length = input_sequence_length + output_sequence_length
    input_sequences = np.zeros(
        (data_set_size, sequence_length + 1, dimension + 2), dtype=np.float32)
    output_sequences = np.zeros(
        (data_set_size, sequence_length + 1, dimension + 2), dtype=np.float32)
    for i in range(data_set_size):
        input_sequence, input_priority, output_sequence, output_priority = \
            generate_one_sample(
                dimension,
                input_sequence_length,
                output_sequence_length,
                priority_lower_bound,
                priority_upper_bound)
        input_sequences[i] = input_sequence
        output_sequences[i] = output_sequence
        input_sequences[i][:, -2] = input_priority.transpose()
        output_sequences[i][:, -2] = output_priority.transpose()

    # return the total samples
    return input_sequences, output_sequences
